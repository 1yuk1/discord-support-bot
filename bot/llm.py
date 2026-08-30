"""Работа с LLM через OpenRouter и резервный OpenAI-совместимый провайдер."""

import base64
import random
import time
import base64
from dataclasses import dataclass, field
import mimetypes
import random
import time

import httpx

from bot import incidents, settings
from bot.escalation import TRANSFER_ANSWER
from bot.filters import IMAGE_ONLY_PLACEHOLDER, is_short_clarification
from bot.logging_setup import log_exception, logger
from bot.prompt import prompts
from bot.proxy import proxy_pool
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


def build_proxy_url(scheme: str | None = None) -> str:
    """Формирует URL прокси (HTTP или SOCKS5) с учётом пула, авторизации и очистки хоста."""
    try:
        active = proxy_pool.get_next_proxy()
        if active:
            return active
    except Exception:
        pass

    raw_host = (settings.PROXY_HOST or "127.0.0.1").strip()
    detected_scheme = None
    if "://" in raw_host:
        detected_scheme, raw_host = raw_host.split("://", 1)

    clean_host = raw_host.lstrip("/").split("@")[-1].split(":")[0]
    selected_scheme = (
        scheme or detected_scheme or getattr(settings, "PROXY_TYPE", "http") or "http"
    ).lower().strip()
    if selected_scheme not in ("http", "https", "socks5", "socks5h", "socks4"):
        selected_scheme = "http"

    if settings.PROXY_USERNAME and settings.PROXY_PASSWORD:
        return (
            f"{selected_scheme}://{settings.PROXY_USERNAME}:{settings.PROXY_PASSWORD}"
            f"@{clean_host}:{settings.PROXY_PORT}"
        )
    return f"{selected_scheme}://{clean_host}:{settings.PROXY_PORT}"


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
fallback_models = ModelRegistry(settings.FALLBACK_AI_MODEL)


@dataclass
class LlmProvider:
    """OpenAI-совместимый провайдер и его независимая модель."""

    name: str
    client: object
    model_registry: ModelRegistry
    circuit_open_until: float = 0.0
    failure_count: int = 0
    # Настройки для динамического пересоздания клиента при смене прокси
    api_key: str = ""
    api_url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    use_proxy: bool = False
    timeout_seconds: float = 90.0
    current_proxy: str | None = None

    def get_client(self, force_refresh_proxy: bool = False):
        """Возвращает клиент, обновляя прокси при необходимости."""
        if not self.use_proxy or not self.api_key:
            return self.client

        active_proxy = proxy_pool.get_next_proxy()
        if force_refresh_proxy or self.current_proxy != active_proxy:
            self.current_proxy = active_proxy
            self.client = _create_openai_client(
                self.api_key,
                self.api_url,
                self.headers,
                use_proxy=bool(active_proxy),
                timeout_seconds=self.timeout_seconds,
                proxy_url=active_proxy,
            )
        return self.client


def _is_retryable_error(exc: Exception) -> bool:
    """Определяет, является ли ошибка временной (timeout, 429, 5xx, network).
    
    Не повторяет постоянные ошибки (400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found).
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        if status_code in (400, 401, 403, 404):
            return False
        if status_code == 429 or status_code >= 500:
            return True

    text = str(exc).lower()
    for fatal in ("400", "401", "403", "404", "unauthorized", "invalid_api_key", "invalid api key", "authentication"):
        if fatal in text:
            return False

    # Временные сетевые ошибки и таймауты
    retryable_patterns = (
        "timeout", "timed out", "429", "rate limit", "rate_limit",
        "500", "502", "503", "504", "server error", "internal error",
        "connection", "network", "pool", "econnreset", "socket", "closed", "proxy",
    )
    return any(p in text for p in retryable_patterns) or isinstance(
        exc, (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
    )


def _is_proxy_or_network_error(exc: Exception) -> bool:
    """Определяет, связан ли сбой с сетевым транспортом или прокси."""
    if isinstance(exc, (httpx.ProxyError, httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError)):
        return True
    text = str(exc).lower()
    patterns = ("proxy", "connect", "connection", "socket", "econnreset", "timed out", "timeout", "refused")
    return any(p in text for p in patterns)


def _create_openai_client(
    api_key: str,
    api_url: str,
    headers: dict[str, str],
    use_proxy: bool,
    timeout_seconds: float,
    proxy_url: str | None = None,
):
    """Создаёт OpenAI-совместимый клиент с тонкой настройкой таймаутов (connect/read/write/pool)."""
    from openai import OpenAI

    connect_timeout = min(10.0, max(2.0, timeout_seconds * 0.2))
    pool_timeout = min(5.0, max(2.0, timeout_seconds * 0.1))
    read_timeout = max(timeout_seconds, 5.0)
    write_timeout = min(15.0, max(5.0, timeout_seconds * 0.3))

    httpx_timeout = httpx.Timeout(
        timeout=timeout_seconds,
        connect=connect_timeout,
        read=read_timeout,
        write=write_timeout,
        pool=pool_timeout,
    )

    resolved_proxy = proxy_url if proxy_url is not None else (build_proxy_url() if use_proxy else None)
    if use_proxy and resolved_proxy:
        http_client = httpx.Client(
            transport=httpx.HTTPTransport(proxy=resolved_proxy),
            timeout=httpx_timeout,
        )
    else:
        http_client = httpx.Client(
            timeout=httpx_timeout,
        )

    return OpenAI(
        api_key=api_key,
        base_url=api_url,
        default_headers=headers,
        http_client=http_client,
        max_retries=0,
    )


def create_providers() -> list[LlmProvider]:
    """Создаёт основной OpenRouter и необязательный резервный провайдер."""
    if not settings.OPENROUTER_API_KEY:
        raise SystemExit("settings.toml: [ai.openrouter].api_key не указан")
    if not settings.OPENROUTER_MODEL:
        raise SystemExit("settings.toml: [ai.openrouter].model не указана")
    if settings.FALLBACK_AI_ENABLED and not settings.FALLBACK_AI_API_KEY:
        raise SystemExit("settings.toml: [ai.fallback].api_key не указан")
    if settings.FALLBACK_AI_ENABLED and not settings.FALLBACK_AI_MODEL:
        raise SystemExit("settings.toml: [ai.fallback].model не указана")
    if settings.FALLBACK_AI_ENABLED and not settings.FALLBACK_AI_API_URL:
        raise SystemExit("settings.toml: [ai.fallback].api_url не указан")

    provider_count = 2 if settings.FALLBACK_AI_ENABLED else 1
    timeout_seconds = settings.AI_REQUEST_TIMEOUT_SECONDS / provider_count

    headers = {}
    if settings.OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
    if settings.OPENROUTER_APP_NAME:
        headers["X-Title"] = settings.OPENROUTER_APP_NAME

    openrouter_proxy = proxy_pool.get_next_proxy() if settings.USE_PROXY else None
    providers = [
        LlmProvider(
            name="openrouter",
            client=_create_openai_client(
                settings.OPENROUTER_API_KEY,
                settings.OPENROUTER_API_URL,
                headers,
                settings.USE_PROXY,
                timeout_seconds,
                proxy_url=openrouter_proxy,
            ),
            model_registry=models,
            api_key=settings.OPENROUTER_API_KEY,
            api_url=settings.OPENROUTER_API_URL,
            headers=headers,
            use_proxy=settings.USE_PROXY,
            timeout_seconds=timeout_seconds,
            current_proxy=openrouter_proxy,
        )
    ]
    logger.info("AI провайдер: OpenRouter | модель=%s", settings.OPENROUTER_MODEL)
    if settings.USE_PROXY:
        logger.info("OpenRouter прокси пул: %s", proxy_pool.get_status())

    if settings.FALLBACK_AI_ENABLED:
        fallback_proxy = proxy_pool.get_next_proxy() if settings.FALLBACK_AI_USE_PROXY else None
        providers.append(
            LlmProvider(
                name="fallback",
                client=_create_openai_client(
                    settings.FALLBACK_AI_API_KEY,
                    settings.FALLBACK_AI_API_URL,
                    {},
                    settings.FALLBACK_AI_USE_PROXY,
                    timeout_seconds,
                    proxy_url=fallback_proxy,
                ),
                model_registry=fallback_models,
                api_key=settings.FALLBACK_AI_API_KEY,
                api_url=settings.FALLBACK_AI_API_URL,
                headers={},
                use_proxy=settings.FALLBACK_AI_USE_PROXY,
                timeout_seconds=timeout_seconds,
                current_proxy=fallback_proxy,
            )
        )
        logger.info("AI резервный провайдер | модель=%s", settings.FALLBACK_AI_MODEL)
        if settings.FALLBACK_AI_USE_PROXY:
            logger.info("Резервный AI-провайдер прокси пул: %s", proxy_pool.get_status())

    return providers


def create_client():
    """Совместимость для внешнего кода, которому нужен только основной клиент."""
    return create_providers()[0].client


def fetch_images_as_base64(image_urls: list[str]) -> list[dict]:
    """Скачивает картинки и возвращает content-блоки в формате data URL.

    Слишком большие файлы пропускаются: они бесполезно расходуют токены и
    могут упереться в лимит запроса OpenRouter.
    """
    content_parts: list[dict] = []

    for url in image_urls:
        use_proxy = settings.USE_PROXY and settings.DISCORD_USE_PROXY
        max_attempts = 2 if use_proxy and not proxy_pool.is_empty else 1

        for attempt in range(max_attempts):
            proxy_url = proxy_pool.get_next_proxy() if use_proxy else None
            try:
                with httpx.Client(
                    timeout=settings.IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
                    follow_redirects=True,
                    proxy=proxy_url,
                ) as client:
                    response = client.get(url)
                    response.raise_for_status()

                if proxy_url:
                    proxy_pool.report_success(proxy_url)

                if len(response.content) > settings.IMAGE_MAX_BYTES:
                    logger.warning(
                        "Скриншот пропущен: %s байт больше лимита %s | url=%s",
                        len(response.content),
                        settings.IMAGE_MAX_BYTES,
                        url[:120],
                    )
                    break

                mime = (response.headers.get("content-type") or "image/png").split(";")[0].strip()
                encoded = base64.b64encode(response.content).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                })
                break
            except Exception as exc:
                if proxy_url:
                    proxy_pool.report_failure(proxy_url, exc)
                if attempt == max_attempts - 1:
                    log_exception("Не удалось скачать изображение", exc, url=url[:200])

    return content_parts


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "rate_limit" in text or "rate limit" in text:
        return ERROR_RATE_LIMIT
    if "timeout" in text:
        return ERROR_TIMEOUT
    if "connect" in text or "network" in text:
        return ERROR_CONNECTION
    return ERROR_GENERIC


from bot.metrics import metrics


class SupportAgent:
    """Генерация ответов игроку и сводок для администраторов."""

    def __init__(self, client_or_providers, index: KnowledgeIndex) -> None:
        if isinstance(client_or_providers, (list, tuple)):
            self._providers = list(client_or_providers)
        elif isinstance(client_or_providers, LlmProvider):
            self._providers = [client_or_providers]
        else:
            # Сохраняет совместимость с изолированными тестами и внешними вызовами.
            self._providers = [LlmProvider("openrouter", client_or_providers, models)]
        if not self._providers:
            raise ValueError("Не настроен ни один AI-провайдер")
        self._client = self._providers[0].client
        self._index = index

    def _complete(self, messages: list[dict], temperature: float, max_tokens: int, task: str = "general"):
        """Запрашивает провайдеров по очереди (OpenRouter -> Fallback) с retry, backoff и circuit breaker."""
        last_error: Exception | None = None
        now = time.time()

        for position, provider in enumerate(self._providers):
            # Проверка Circuit Breaker: если провайдер временно отключен
            if provider.circuit_open_until > now:
                logger.warning(
                    "Провайдер %s временно отключен Circuit Breaker до %s, пропуск",
                    provider.name,
                    provider.circuit_open_until,
                )
                continue

            model = provider.model_registry.get()
            max_attempts = 2  # до 2 попыток на провайдер для временных сбоев
            base_backoff = 0.5
            client = provider.get_client() if hasattr(provider, "get_client") else provider.client

            for attempt in range(1, max_attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    answer = response.choices[0].message.content or ""
                    if not answer.strip():
                        raise ValueError("Провайдер вернул пустой ответ")

                    # Фиксируем успех прокси
                    if getattr(provider, "current_proxy", None):
                        proxy_pool.report_success(provider.current_proxy)

                    # Учёт расхода токенов
                    usage = getattr(response, "usage", None)
                    p_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
                    c_tok = getattr(usage, "completion_tokens", 0) if usage else 0
                    t_tok = getattr(usage, "total_tokens", 0) if usage else 0
                    if p_tok or c_tok or t_tok:
                        metrics.record_usage(provider.name, model, task, p_tok, c_tok, t_tok)

                    # Успех — сбрасываем счетчик ошибок
                    provider.failure_count = 0
                    provider.circuit_open_until = 0.0
                    return response, provider
                except Exception as exc:
                    last_error = exc

                    # Если сбой связан с прокси или сетью, отправляем текущий прокси в кулдаун и переключаем
                    if _is_proxy_or_network_error(exc) and getattr(provider, "current_proxy", None):
                        proxy_pool.report_failure(provider.current_proxy, exc)
                        if hasattr(provider, "get_client"):
                            client = provider.get_client(force_refresh_proxy=True)

                    # Проверяем, стоит ли делать retry
                    is_retryable = _is_retryable_error(exc)

                    if not is_retryable:
                        logger.warning(
                            "Неповторяемая ошибка AI-провайдера (auth/config/bad_request) | provider=%s | model=%s | error=%s",
                            provider.name,
                            model,
                            exc,
                        )
                        # Мгновенно отключаем провайдера при фатальных ошибках (401, 403, лимиты, бан IP)
                        cooldown = getattr(settings, "CIRCUIT_BREAKER_COOLDOWN_SECONDS", 600)
                        provider.circuit_open_until = time.time() + cooldown
                        provider.failure_count = getattr(settings, "CIRCUIT_BREAKER_THRESHOLD", 3)
                        logger.error(
                            "Circuit Breaker мгновенно отключил %s на %sс из-за фатальной ошибки",
                            provider.name,
                            cooldown,
                        )
                        break

                    if attempt < max_attempts:
                        # Exponential backoff с jitter
                        jitter = random.uniform(0.1, 0.4)
                        delay = (base_backoff * (2 ** (attempt - 1))) + jitter
                        logger.warning(
                            "Временный сбой AI (%s), повторная попытка через %.2fс | provider=%s | model=%s | error=%s",
                            attempt,
                            delay,
                            provider.name,
                            model,
                            exc.__class__.__name__,
                        )
                        time.sleep(delay)

            # Если все попытки у провайдера провалились
            if is_retryable:
                provider.failure_count += 1
                threshold = getattr(settings, "CIRCUIT_BREAKER_THRESHOLD", 3)
                cooldown = getattr(settings, "CIRCUIT_BREAKER_COOLDOWN_SECONDS", 600)
                if provider.failure_count >= threshold:
                    provider.circuit_open_until = time.time() + cooldown
                    logger.error(
                        "Circuit Breaker сработал для %s (%s сбоев подряд), отключен на %sс",
                        provider.name,
                        provider.failure_count,
                        cooldown,
                    )

            if position + 1 < len(self._providers):
                logger.warning(
                    "Сбой AI-провайдера, включён резерв | provider=%s | model=%s | error=%s",
                    provider.name,
                    model,
                    last_error.__class__.__name__ if last_error else "unknown",
                )

        assert last_error is not None
        raise last_error

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

        logger.info(
            "Запрос к AI | primary_model=%s | messages=%s | has_context=%s | инцидентов=%s | "
            "images=%s | preview=%s",
            self._providers[0].model_registry.get(),
            len(messages),
            bool(context),
            len(incidents.active()),
            len(image_blocks),
            user_input[:200].replace("\n", " "),
        )

        try:
            response, provider = self._complete(
                messages, settings.AI_TEMPERATURE, settings.AI_MAX_TOKENS, task="reply"
            )
            answer = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "length":
                logger.warning(
                    "Ответ AI обрезан (finish_reason=length), возможно не хватает max_tokens | "
                    "model=%s | max_tokens=%s | preview=%s",
                    provider.model_registry.get(),
                    settings.AI_MAX_TOKENS,
                    answer[:200].replace("\n", " "),
                )
            
            usage = getattr(response, "usage", None)
            tokens_str = f" | tokens={getattr(usage, 'total_tokens', '?')}" if usage else ""
            logger.info(
                "Ответ AI получен | provider=%s | model=%s%s | finish_reason=%s | preview=%s",
                provider.name,
                provider.model_registry.get(),
                tokens_str,
                finish_reason,
                answer[:200].replace("\n", " "),
            )
            return answer or ERROR_GENERIC
        except Exception as exc:
            log_exception(
                "Ошибка генерации ответа AI",
                exc,
                model=self._providers[0].model_registry.get(),
                user_input_preview=user_input[:200],
            )
            return _classify_error(exc)

    def compose_reminder(self, transcript: str) -> str:
        """Текст напоминания игроку, который давно ждёт ответа.

        Вызывающая сторона обязана иметь запасную статичную фразу: любая
        ошибка здесь не должна отменять само напоминание.
        """
        prompt = prompts.reminder.replace("{TRANSCRIPT}", transcript)
        response, provider = self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты пишешь короткие вежливые напоминания игрокам поддержки. "
                        "Никогда не называешь сроки и не выдумываешь факты."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            0.5,
            512,  # Для напоминаний max_tokens=512 (с запасом для reasoning-моделей)
            task="reminder",
        )
        answer = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        
        # Если модель не закончила генерацию, это обрезанный ответ.
        if finish_reason == "length":
            logger.warning(
                "Ответ напоминания обрезан (finish_reason=length), "
                "используем статичную фразу | model=%s",
                provider.model_registry.get(),
            )
            return ""
        
        return answer

    def summarize_ticket(self, transcript: str) -> str:
        """Сводка тикета для администратора."""
        prompt = prompts.summary.replace("{TRANSCRIPT}", transcript)
        response, provider = self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты помогаешь администраторам быстро понять суть тикета. "
                        "Не выдумывай факты."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            0.1,
            1024,
            task="summary",
        )
        answer = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason == "length":
            logger.warning(
                "Сводка тикета обрезана (finish_reason=length) | model=%s",
                provider.model_registry.get(),
            )
        
        return answer or "Не удалось получить сводку."


__all__ = [
    "ERROR_PREFIX",
    "ERROR_TIMEOUT",
    "SupportAgent",
    "TRANSFER_ANSWER",
    "build_proxy_url",
    "create_client",
    "create_providers",
    "models",
    "fallback_models",
]
