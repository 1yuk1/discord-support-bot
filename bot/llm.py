"""Работа с LLM через OpenRouter: клиент, генерация ответов и сводок."""

import base64

import httpx

from bot import incidents, settings
from bot.escalation import TRANSFER_ANSWER
from bot.filters import IMAGE_ONLY_PLACEHOLDER, is_short_clarification
from bot.logging_setup import log_exception, logger
from bot.prompt import prompts
from bot.rag import KnowledgeIndex

# Тексты ошибок для игрока. Маркер ⚠️ используется вызывающим кодом, чтобы
# отличать сбой от нормального ответа и не засчитывать его как успешный.
ERROR_PREFIX = "⚠️"
ERROR_GENERIC = f"{ERROR_PREFIX} Произошла ошибка. Попробуйте ещё раз."
ERROR_RATE_LIMIT = f"{ERROR_PREFIX} Временная перегрузка сервиса. Попробуйте через минуту."
ERROR_CONNECTION = f"{ERROR_PREFIX} Нет подключения к сервису. Попробуйте позже."
ERROR_TIMEOUT = f"{ERROR_PREFIX} AI слишком долго отвечает. Попробуйте ещё раз чуть позже."
IMAGE_LOAD_FAILED = (
    "Не удалось загрузить скриншот. Пожалуйста, отправьте его ещё раз "
    "или опишите проблему текстом."
)


def build_proxy_url() -> str:
    """HTTP-прокси с учётом необязательной авторизации."""
    if settings.PROXY_USERNAME and settings.PROXY_PASSWORD:
        return (
            f"http://{settings.PROXY_USERNAME}:{settings.PROXY_PASSWORD}"
            f"@{settings.PROXY_HOST}:{settings.PROXY_PORT}"
        )
    return f"http://{settings.PROXY_HOST}:{settings.PROXY_PORT}"


class ModelRegistry:
    """Текущая модель. Меняется командой !model без рестарта."""

    def __init__(self, model: str) -> None:
        self._model = model

    def get(self) -> str:
        return self._model

    def set(self, model: str) -> str:
        previous = self._model
        self._model = model
        return previous


models = ModelRegistry(settings.OPENROUTER_MODEL)


def create_client():
    """Создаёт клиент OpenRouter. Падает с понятной ошибкой при пустом конфиге."""
    from openai import OpenAI

    if not settings.OPENROUTER_API_KEY:
        raise SystemExit("settings.toml: [ai.openrouter].api_key не указан")
    if not settings.OPENROUTER_MODEL:
        raise SystemExit("settings.toml: [ai.openrouter].model не указана")

    headers = {}
    if settings.OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
    if settings.OPENROUTER_APP_NAME:
        headers["X-Title"] = settings.OPENROUTER_APP_NAME

    logger.info("AI провайдер: OpenRouter | модель=%s", settings.OPENROUTER_MODEL)

    if settings.USE_PROXY:
        logger.info("OpenRouter через прокси %s:%s", settings.PROXY_HOST, settings.PROXY_PORT)
        timeout = httpx.Timeout(
            settings.AI_REQUEST_TIMEOUT_SECONDS,
            connect=min(settings.AI_REQUEST_TIMEOUT_SECONDS, 15),
        )
        http_client = httpx.Client(
            transport=httpx.HTTPTransport(proxy=build_proxy_url()),
            timeout=timeout,
        )
        return OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_API_URL,
            default_headers=headers,
            http_client=http_client,
            max_retries=1,
        )

    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_API_URL,
        default_headers=headers,
        timeout=settings.AI_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )


def fetch_images_as_base64(image_urls: list[str]) -> list[dict]:
    """Скачивает картинки и возвращает content-блоки в формате data URL.

    Слишком большие файлы пропускаются: они бесполезно расходуют токены и
    могут упереться в лимит запроса OpenRouter.
    """
    proxies = build_proxy_url() if settings.USE_PROXY else None
    content_parts: list[dict] = []

    for url in image_urls:
        try:
            with httpx.Client(
                timeout=settings.IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
                follow_redirects=True,
                proxy=proxies,
            ) as client:
                response = client.get(url)
                response.raise_for_status()

            if len(response.content) > settings.IMAGE_MAX_BYTES:
                logger.warning(
                    "Скриншот пропущен: %s байт больше лимита %s | url=%s",
                    len(response.content),
                    settings.IMAGE_MAX_BYTES,
                    url[:120],
                )
                continue

            mime = (response.headers.get("content-type") or "image/png").split(";")[0].strip()
            encoded = base64.b64encode(response.content).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        except Exception as exc:
            log_exception("Не удалось скачать изображение", exc, url=url[:200])

    return content_parts


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "rate_limit" in text or "rate limit" in text:
        return ERROR_RATE_LIMIT
    if "connect" in text or "timeout" in text or "network" in text:
        return ERROR_CONNECTION
    return ERROR_GENERIC


class SupportAgent:
    """Генерация ответов игроку и сводок для администраторов."""

    def __init__(self, client, index: KnowledgeIndex) -> None:
        self._client = client
        self._index = index

    def _build_context(self, user_input: str) -> str:
        # Короткие реплики («60», «да», «шлемофон») в RAG не идут: эмбеддер
        # вытащит случайный документ с тем же числом или словом, и LLM
        # уверенно ответит невпопад. Пусть отвечает по истории диалога.
        if is_short_clarification(user_input):
            return ""
        try:
            return self._index.search(user_input)
        except Exception as exc:
            log_exception("Поиск в базе знаний не выполнен", exc, query_preview=user_input[:200])
            return ""

    def generate_answer(
        self,
        user_input: str,
        conversation_history: list[dict] | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        """Ответ игроку. История подаётся отдельными сообщениями user/assistant."""
        context = self._build_context(user_input)

        if context:
            context_block = f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context}"
        else:
            context_block = (
                "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n"
                "(Информация не найдена — задай один самый важный уточняющий вопрос)"
            )

        messages: list[dict] = [{"role": "system", "content": prompts.system}]

        # Инциденты идут отдельным system-сообщением после основного промпта:
        # так они не тонут в истории диалога и действуют на любой вопрос.
        # Когда инцидентов нет, блок не добавляется — токены не тратятся.
        incidents_block = incidents.prompt_block()
        if incidents_block:
            messages.append({"role": "system", "content": incidents_block})

        for entry in conversation_history or []:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            content = entry.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        image_blocks = fetch_images_as_base64(image_urls) if image_urls else []
        only_image = user_input == IMAGE_ONLY_PLACEHOLDER
        if image_urls and not image_blocks and only_image:
            return IMAGE_LOAD_FAILED

        text_content = f"{context_block}\n\nТЕКУЩИЙ ВОПРОС ИГРОКА:\n{user_input}"
        if image_blocks:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": text_content}] + image_blocks,
            })
        else:
            messages.append({"role": "user", "content": text_content})

        model = models.get()
        logger.info(
            "Запрос к AI | model=%s | messages=%s | has_context=%s | инцидентов=%s | "
            "images=%s | preview=%s",
            model,
            len(messages),
            bool(context),
            len(incidents.active()),
            len(image_blocks),
            user_input[:200].replace("\n", " "),
        )

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.AI_TEMPERATURE,
                max_tokens=settings.AI_MAX_TOKENS,
            )
            answer = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "length":
                logger.warning(
                    "Ответ AI обрезан (finish_reason=length), возможно не хватает max_tokens | "
                    "model=%s | max_tokens=%s | preview=%s",
                    model,
                    settings.AI_MAX_TOKENS,
                    answer[:200].replace("\n", " "),
                )
            
            logger.info(
                "Ответ AI получен | model=%s | finish_reason=%s | preview=%s",
                model,
                finish_reason,
                answer[:200].replace("\n", " "),
            )
            return answer or ERROR_GENERIC
        except Exception as exc:
            log_exception(
                "Ошибка генерации ответа AI",
                exc,
                model=model,
                user_input_preview=user_input[:200],
            )
            return _classify_error(exc)

    def compose_reminder(self, transcript: str) -> str:
        """Текст напоминания игроку, который давно ждёт ответа.

        Вызывающая сторона обязана иметь запасную статичную фразу: любая
        ошибка здесь не должна отменять само напоминание.
        """
        prompt = prompts.reminder.replace("{TRANSCRIPT}", transcript)
        response = self._client.chat.completions.create(
            model=models.get(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты пишешь короткие вежливые напоминания игрокам поддержки. "
                        "Никогда не называешь сроки и не выдумываешь факты."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        
        # Если модель не закончила генерацию, это обрезанный ответ.
        if finish_reason == "length":
            logger.warning(
                "Ответ напоминания обрезан (finish_reason=length), "
                "используем статичную фразу | model=%s",
                models.get(),
            )
            return ""
        
        return answer

    def summarize_ticket(self, transcript: str) -> str:
        """Сводка тикета для администратора."""
        prompt = prompts.summary.replace("{TRANSCRIPT}", transcript)
        response = self._client.chat.completions.create(
            model=models.get(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты помогаешь администраторам быстро понять суть тикета. "
                        "Не выдумывай факты."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason == "length":
            logger.warning(
                "Сводка тикета обрезана (finish_reason=length) | model=%s",
                models.get(),
            )
        
        return answer or "Не удалось получить сводку."


__all__ = [
    "ERROR_PREFIX",
    "ERROR_TIMEOUT",
    "SupportAgent",
    "TRANSFER_ANSWER",
    "build_proxy_url",
    "create_client",
    "models",
]
