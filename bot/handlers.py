"""Обработка входящих сообщений: фильтры, склейка, вызов AI."""

import asyncio
import time

from bot import settings
from bot.discord_client import moderate_ping_spam, safe_send
from bot.limits import (
    channel_cooldown_remaining,
    global_rate_limiter,
    is_duplicate_message,
    user_flood_detected,
)
from bot.escalation import (
    TRANSFER_ANSWER,
    is_llm_human_transfer,
    is_user_human_transfer,
    should_force_human_transfer,
)
from bot.filters import (
    IMAGE_ONLY_PLACEHOLDER,
    extract_image_urls,
    extract_message_text,
    is_ticket_close_notification,
    is_ticket_opening_message,
    should_use_message_as_question,
)
from bot.llm import ERROR_PREFIX, ERROR_TIMEOUT
from bot.logging_setup import logger
from bot.state import store
from bot.text_utils import normalize_for_dedup
from bot import reminders, ticket_logs

COOLDOWN_NOTICE = "⏳ Подождите {seconds} секунд перед следующим вопросом."
GLOBAL_LIMIT_NOTICE = "⏳ Слишком много сообщений. Подождите минуту."
FLOOD_NOTICE = "⏳ Не флудите! Подождите перед следующим вопросом."


class MessageRouter:
    """Решает, отвечать ли на сообщение, и как именно."""

    def __init__(self, bot, agent) -> None:
        self._bot = bot
        self._agent = agent
        self._semaphore = asyncio.Semaphore(max(settings.AI_MAX_CONCURRENT_REQUESTS, 1))
        self._ticket_categories = set(settings.TICKET_CATEGORY_IDS)
        self._bot_role_ids = set(settings.BOT_ROLE_IDS)
        self._ignored_role_ids = set(settings.IGNORED_ROLE_IDS)
        self.reminder_service = reminders.ReminderService(bot, agent)

    def refresh_settings(self) -> None:
        """Перечитывает id категорий и ролей после /config reload.

        Роутер копирует их в множества при создании: без этого вызова правка
        ticket_category_ids в файле применилась бы только после рестарта.
        Семафор не пересоздаём — активные запросы к AI держат его слоты.
        """
        self._ticket_categories = set(settings.TICKET_CATEGORY_IDS)
        self._bot_role_ids = set(settings.BOT_ROLE_IDS)
        self._ignored_role_ids = set(settings.IGNORED_ROLE_IDS)

    # ── Проверки доступа ─────────────────────────────────────────────────────
    def is_ticket_channel(self, channel) -> bool:
        if not self._ticket_categories:
            return True
        return getattr(channel, "category_id", None) in self._ticket_categories

    def has_ignored_role(self, member) -> bool:
        if not self._ignored_role_ids or member is None:
            return False
        return any(
            role.id in self._ignored_role_ids for role in getattr(member, "roles", []) or []
        )

    def bot_has_required_role(self, guild) -> bool:
        if not self._bot_role_ids:
            return True
        if guild is None or self._bot.user is None:
            return False
        member = guild.get_member(self._bot.user.id)
        if member is None:
            return False
        return any(role.id in self._bot_role_ids for role in member.roles)

    # ── Debounce ─────────────────────────────────────────────────────────────
    @staticmethod
    def cancel_debounce(state: dict) -> None:
        task = state.get("debounce_task")
        if task is not None and not task.done():
            task.cancel()
        state["debounce_task"] = None
        state["pending_parts"] = []

    @staticmethod
    def _merge_parts(parts: list[dict]) -> tuple[str, list[str], bool]:
        """Склеивает буфер в один текст, отбрасывая подряд идущие повторы."""
        texts: list[str] = []
        images: list[str] = []
        author_is_bot = False

        for part in parts:
            text = (part.get("text") or "").strip()
            if text and (
                not texts or normalize_for_dedup(texts[-1]) != normalize_for_dedup(text)
            ):
                texts.append(text)
            for url in part.get("image_urls") or []:
                if url not in images:
                    images.append(url)
            author_is_bot = author_is_bot or bool(part.get("author_is_bot"))

        return "\n".join(texts).strip(), images, author_is_bot

    def queue_message(self, channel, state: dict, text: str, image_urls, author_is_bot) -> bool:
        """Кладёт сообщение в буфер склейки. False — склейка отключена."""
        if settings.MESSAGE_DEBOUNCE_SECONDS <= 0:
            return False

        state.setdefault("pending_parts", []).append({
            "text": text,
            "image_urls": list(image_urls or []),
            "author_is_bot": bool(author_is_bot),
        })

        task = state.get("debounce_task")
        if task is not None and not task.done():
            task.cancel()
        state["debounce_task"] = asyncio.create_task(self._debounce_flush(channel, state))

        logger.info(
            "Сообщение в буфере склейки | channel_id=%s | частей=%s | пауза=%.1fс",
            getattr(channel, "id", "unknown"),
            len(state["pending_parts"]),
            settings.MESSAGE_DEBOUNCE_SECONDS,
        )
        return True

    async def _debounce_flush(self, channel, state: dict) -> None:
        try:
            await asyncio.sleep(max(settings.MESSAGE_DEBOUNCE_SECONDS, 0.1))
        except asyncio.CancelledError:
            return

        parts = state.get("pending_parts") or []
        state["pending_parts"] = []
        state["debounce_task"] = None

        if not parts or state.get("human_mode") or state.get("bot_disabled"):
            return

        text, image_urls, author_is_bot = self._merge_parts(parts)
        if not text and not image_urls:
            return
        if not text:
            text = IMAGE_ONLY_PLACEHOLDER

        logger.info(
            "Склейка завершена | channel_id=%s | частей=%s | preview=%s",
            getattr(channel, "id", "unknown"),
            len(parts),
            text[:200].replace("\n", " "),
        )
        await self.run_ai_reply(channel, state, text, image_urls, author_is_bot)

    # ── Ответ AI ─────────────────────────────────────────────────────────────
    async def run_ai_reply(self, channel, state: dict, text: str, image_urls, author_is_bot) -> None:
        channel_id = getattr(channel, "id", "unknown")

        if state.get("human_mode") or state.get("bot_disabled"):
            return
        if state.get("ai_busy"):
            logger.info("AI уже отвечает в этом канале, пропуск | channel_id=%s", channel_id)
            return

        if is_duplicate_message(state, text):
            logger.info(
                "Пропуск дубликата | channel_id=%s | preview=%s",
                channel_id,
                text[:200].replace("\n", " "),
            )
            return

        cooldown = channel_cooldown_remaining(state)
        if cooldown > 0:
            await safe_send(channel, COOLDOWN_NOTICE.format(seconds=cooldown))
            return

        if not global_rate_limiter.allow():
            await safe_send(channel, GLOBAL_LIMIT_NOTICE)
            return

        logger.info(
            "Генерация ответа | channel_id=%s | preview=%s",
            channel_id,
            text[:200].replace("\n", " "),
        )

        state["ai_busy"] = True
        try:
            async with channel.typing():
                async with self._semaphore:
                    try:
                        answer = await asyncio.wait_for(
                            asyncio.to_thread(
                                self._agent.generate_answer,
                                text,
                                state["history"],
                                image_urls,
                            ),
                            timeout=settings.AI_REQUEST_TIMEOUT_SECONDS + 10,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "AI превысил таймаут | channel_id=%s | timeout=%sс",
                            channel_id,
                            settings.AI_REQUEST_TIMEOUT_SECONDS,
                        )
                        answer = ERROR_TIMEOUT
        finally:
            state["ai_busy"] = False

        is_error = bool(answer) and answer.startswith(ERROR_PREFIX)
        if is_error:
            logger.warning(
                "Игроку отправлен текст ошибки | channel_id=%s | answer=%s", channel_id, answer
            )

        await safe_send(channel, answer)

        state["last_message"] = text
        state["last_message_time"] = time.time()
        if not is_error:
            state["last_answer_time"] = time.time()

        ticket_logs.append_entry(channel, 0, "user", text, bot_response=answer)
        store.append_turn(state, text, answer, author_is_bot)
        store.touch(state)
        store.mark_dirty()

        if answer and is_llm_human_transfer(answer):
            self._activate_human_mode(channel, state, "llm_in_answer")

    def _activate_human_mode(self, channel, state: dict, reason: str) -> None:
        state["human_mode"] = True
        store.mark_dirty()
        store.save(force=True)

        bot_user = self._bot.user
        ticket_logs.append_entry(
            channel,
            bot_user.id if bot_user is not None else 0,
            str(bot_user) if bot_user is not None else "bot",
            "Режим передачи человеку активирован",
            is_human_transfer=True,
            transfer_reason=reason,
        )
        logger.info(
            "Тикет передан человеку | channel_id=%s | reason=%s",
            getattr(channel, "id", "unknown"),
            reason,
        )

    async def _handle_transfer_request(self, message, state: dict, text: str, reason: str) -> None:
        """Просьба о человеке идёт в обход рейтлимитов."""
        self.cancel_debounce(state)
        await safe_send(message.channel, TRANSFER_ANSWER)

        now = time.time()
        state["last_message"] = text
        state["last_message_time"] = now
        state["last_answer_time"] = now
        state["human_mode"] = True
        store.mark_dirty()
        store.save(force=True)

        ticket_logs.append_entry(
            message.channel,
            message.author.id,
            str(message.author),
            text,
            bot_response=TRANSFER_ANSWER,
            is_human_transfer=True,
            transfer_reason=reason,
        )
        store.append_turn(state, text, TRANSFER_ANSWER, message.author.bot)
        logger.info(
            "Тикет передан человеку | channel_id=%s | reason=%s",
            getattr(message.channel, "id", "unknown"),
            reason,
        )

    # ── Напоминания ──────────────────────────────────────────────────────────
    def _record_reminder_activity(self, message, channel_id) -> None:
        """Отмечает в состоянии, кто писал последним: персонал или игрок."""
        if not settings.REMINDERS_ENABLED:
            return

        category_id = getattr(message.channel, "category_id", None)
        config = settings.reminder_config_for(category_id)
        if not config.get("enabled"):
            return

        state = store.get_or_create(channel_id)
        reminders.record_activity(
            state,
            is_staff_author=reminders.is_staff(message.author, config.get("staff_role_ids")),
            is_bot_author=bool(message.author.bot),
        )
        store.mark_dirty()

    # ── Точка входа ──────────────────────────────────────────────────────────
    async def handle_message(self, message) -> None:
        channel = message.channel
        channel_id = getattr(channel, "id", "unknown")
        bot_user = self._bot.user

        # Свои же сообщения игнорируем всегда и до любых проверок.
        if bot_user is not None and message.author.id == bot_user.id:
            return

        if not self.is_ticket_channel(channel):
            await self._bot.process_commands(message)
            return

        # Команды обрабатываем до всех фильтров, иначе !start не сработает
        # в канале, где бот выключен.
        if not message.author.bot and (message.content or "").lstrip().startswith(
            settings.COMMAND_PREFIX
        ):
            await self._bot.process_commands(message)
            return

        if message.author.bot and not should_use_message_as_question(message):
            return

        # Активность отмечаем ДО игнора по роли: ответ хелпера обязан сбросить
        # отсчёт напоминаний, хотя сам бот на такие сообщения не реагирует.
        self._record_reminder_activity(message, channel_id)

        if not message.author.bot and self.has_ignored_role(message.author):
            logger.info(
                "Игнор по роли | channel_id=%s | author=%s", channel_id, message.author
            )
            return

        state = store.get_or_create(channel_id)
        store.touch(state)

        if message.id in state["processed_message_ids"]:
            return

        text = extract_message_text(message)
        image_urls = extract_image_urls(message)
        if not text and not image_urls:
            return
        if not text:
            text = IMAGE_ONLY_PLACEHOLDER

        # Системные сообщения тикет-бота.
        if message.author.bot:
            if is_ticket_close_notification(text):
                store.remember_processed(state, message.id)
                logger.info("Уведомление о закрытии тикета пропущено | channel_id=%s", channel_id)
                return
            if is_ticket_opening_message(text):
                if state["ticket_opening_handled"]:
                    store.remember_processed(state, message.id)
                    return
                state["ticket_opening_handled"] = True
                store.mark_dirty()

        logger.info(
            "Сообщение принято | channel_id=%s | author=%s | preview=%s",
            channel_id,
            message.author,
            text[:300].replace("\n", " "),
        )
        store.remember_processed(state, message.id)

        # Бот выключен админом: ведём лог, но не отвечаем и не зовём человека.
        if state.get("bot_disabled"):
            ticket_logs.append_entry(
                message.channel,
                message.author.id,
                str(message.author),
                text,
                image_urls=image_urls or None,
            )
            logger.info("Бот выключен в канале, ответ не отправлен | channel_id=%s", channel_id)
            return

        transfer_reason = None
        if is_user_human_transfer(text):
            transfer_reason = "phrase"
        elif should_force_human_transfer(text):
            transfer_reason = "forced_keyword"

        if transfer_reason is None and not message.author.bot and user_flood_detected(state):
            await safe_send(message.channel, FLOOD_NOTICE)
            return

        ticket_logs.append_entry(
            message.channel,
            message.author.id,
            str(message.author),
            text,
            image_urls=image_urls or None,
        )

        if state["human_mode"]:
            logger.info("Канал в режиме передачи человеку | channel_id=%s", channel_id)
            self.cancel_debounce(state)
            await moderate_ping_spam(message, state)
            store.mark_dirty()
            return

        if not self.bot_has_required_role(message.guild):
            logger.info(
                "У бота нет требуемой роли | channel_id=%s | required=%s",
                channel_id,
                sorted(self._bot_role_ids),
            )
            return

        if transfer_reason is not None:
            await self._handle_transfer_request(message, state, text, transfer_reason)
            return

        if self.queue_message(message.channel, state, text, image_urls, message.author.bot):
            return

        await self.run_ai_reply(
            message.channel, state, text, image_urls, message.author.bot
        )
