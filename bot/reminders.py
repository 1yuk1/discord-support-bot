"""Напоминания персоналу о тикетах, которые остались без ответа.

Задача: игрок написал, персонал молчит — через idle_hours в канал уходит
вежливое сообщение игроку с пингом роли поддержки. Одно сообщение решает две
задачи: игрок видит, что его не забыли, персонал получает уведомление.

Почему точка отсчёта — последнее сообщение ПЕРСОНАЛА, а не игрока:
тикет может решаться днями, и всё это время игрок ждёт законно. Как только
хелпер написал хоть что-то, отсчёт стартует заново, и спама напоминаниями нет.

Решение о необходимости напоминания (should_remind) отделено от отправки:
это чистая функция на словаре состояния, её можно тестировать без Discord.
"""

import random
import time
from datetime import datetime, time as dtime, timezone
import zoneinfo

from bot import settings
from bot.logging_setup import log_exception, logger

# Сколько последних сообщений канала перечитываем перед отправкой.
_VERIFY_HISTORY_LIMIT = 5

MODE_LLM = "llm"
MODE_STATIC = "static"


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reset_daily_counter_if_needed(state: dict) -> None:
    """Обнуляет суточный счётчик при смене календарного дня (UTC)."""
    today = _today_key()
    if state.get("reminder_day") != today:
        state["reminder_day"] = today
        state["reminder_count_today"] = 0


def is_staff(member, staff_role_ids) -> bool:
    """Есть ли у автора роль персонала."""
    if member is None or not staff_role_ids:
        return False
    allowed = set(staff_role_ids)
    return any(
        getattr(role, "id", None) in allowed for role in getattr(member, "roles", []) or []
    )


def record_activity(state: dict, is_staff_author: bool, is_bot_author: bool = False) -> None:
    """Отмечает, кто писал последним.

    Вызывается из роутера на КАЖДОМ сообщении, включая те, что бот игнорирует
    по роли: ответ хелпера обязан сбрасывать отсчёт, иначе напоминание уйдёт
    в канал, где уже разбираются.

    Сообщения ботов (тикет-система, сам бот) не считаются ни за кого: они не
    ответ персонала и не обращение игрока.
    """
    if is_bot_author:
        return

    now = time.time()
    if is_staff_author:
        state["last_staff_message_time"] = now
        # Персонал ответил — цикл напоминаний начинается заново.
        state["last_reminder_time"] = 0.0
    else:
        state["last_player_message_time"] = now


def waiting_since(state: dict) -> float:
    """С какого момента игрок ждёт ответа. 0 — не ждёт.

    Если персонал отвечал позже игрока, ожидания нет: мяч на стороне игрока.
    """
    player_time = float(state.get("last_player_message_time") or 0.0)
    staff_time = float(state.get("last_staff_message_time") or 0.0)

    if not player_time:
        return 0.0
    if staff_time >= player_time:
        return 0.0
    return player_time


def is_in_quiet_hours(config: dict, now_dt: datetime | None = None) -> bool:
    """Проверяет, попадает ли текущее время в ночное окно тишины.
    
    quiet_hours_timezone: строка временной зоны (напр. 'Europe/Moscow')
    quiet_hours_start: 'HH:MM' (напр. '23:00')
    quiet_hours_end: 'HH:MM' (напр. '09:00')
    """
    start_str = str(config.get("quiet_hours_start") or "").strip()
    end_str = str(config.get("quiet_hours_end") or "").strip()
    
    if not start_str or not end_str:
        return False

    tz_name = str(config.get("quiet_hours_timezone") or "Europe/Moscow").strip()
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    if now_dt is None:
        local_now = datetime.now(tz)
    else:
        if now_dt.tzinfo is None:
            local_now = now_dt.replace(tzinfo=timezone.utc).astimezone(tz)
        else:
            local_now = now_dt.astimezone(tz)

    current_time = local_now.time()

    try:
        sh, sm = map(int, start_str.split(":"))
        eh, em = map(int, end_str.split(":"))
        start_time = dtime(sh, sm)
        end_time = dtime(eh, em)
    except Exception:
        return False

    if start_time == end_time:
        return False

    if start_time < end_time:
        # В пределах одного дня, например 01:00 - 06:00
        return start_time <= current_time < end_time
    else:
        # Переход через полночь, например 23:00 - 09:00
        return current_time >= start_time or current_time < end_time


def should_remind(state: dict, config: dict, now: float | None = None) -> bool:
    """Пора ли напоминать в этом канале."""
    if not config.get("enabled"):
        return False
    if state.get("reminders_disabled"):
        return False
    if not config.get("ping_role_ids"):
        # Пинговать некого — на тестовом сервере роли поддержки может не быть.
        return False

    now_ts = time.time() if now is None else now

    # Проверка окна тишины (ночное время). В окне тишины напоминание не отправляется.
    dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    if is_in_quiet_hours(config, dt):
        return False

    started = waiting_since(state)
    if not started:
        return False

    idle_seconds = max(float(config.get("idle_hours", 1)), 0) * 3600
    if now_ts - started < idle_seconds:
        return False

    _reset_daily_counter_if_needed(state)
    max_per_day = int(config.get("max_per_day", 0))
    if max_per_day > 0 and state.get("reminder_count_today", 0) >= max_per_day:
        return False

    last_reminder = float(state.get("last_reminder_time") or 0.0)
    if last_reminder:
        repeat_seconds = max(float(config.get("repeat_hours", 6)), 0) * 3600
        if now_ts - last_reminder < repeat_seconds:
            return False

    return True


def register_sent(state: dict, now: float | None = None) -> None:
    """Фиксирует отправленное напоминание."""
    _reset_daily_counter_if_needed(state)
    state["last_reminder_time"] = time.time() if now is None else now
    state["reminder_count_today"] = int(state.get("reminder_count_today", 0)) + 1


def static_phrase(config: dict) -> str:
    phrases = config.get("phrases") or settings.REMINDER_PHRASES
    return random.choice(list(phrases))


def format_reminder(text: str, ping_role_ids) -> str:
    """Собирает итоговое сообщение: текст игроку плюс пинг персонала."""
    mentions = " ".join(f"<@&{role_id}>" for role_id in ping_role_ids)
    body = (text or "").strip()
    return f"{body}\n{mentions}".strip() if mentions else body


def _is_valid_reminder_text(text: str) -> bool:
    """Проверяет, что текст выглядит как нормальное напоминание.
    
    Отсекает мусор: обрезанные ответы, утёкший промпт, фрагменты инструкций.
    """
    if not text or len(text.strip()) < 20:
        return False
    
    # Фрагменты инструкций из промпта или leaked reasoning.
    junk_patterns = [
        "transcript", "{transcript}", "rules:", "example:",
        "role:", "system:", "user:", "assistant:",
        "never mention", "no facts", "don't invent",
        "reasons/statuses", ", no requests for data",
    ]
    lower = text.lower()
    if any(pattern in lower for pattern in junk_patterns):
        return False
    
    # Обрезанные ответы: заканчиваются на незакрытое предложение.
    # Проверяем, что есть хотя бы одна точка/восклицательный/вопрос в середине.
    stripped = text.strip()
    if not any(stripped.count(p) > 0 for p in [".", "!", "?"]):
        return False
    
    return True


class ReminderService:
    """Периодически проверяет тикеты и отправляет напоминания."""

    def __init__(self, bot, agent) -> None:
        self._bot = bot
        self._agent = agent

    # ── Текст ────────────────────────────────────────────────────────────────
    async def _build_text(self, channel, config: dict) -> str:
        """Текст напоминания. При любой проблеме — статичная фраза.

        LLM может уйти в таймаут или упереться в лимит; напоминание из-за
        этого падать не должно.
        """
        if config.get("message_mode") != MODE_LLM:
            return static_phrase(config)

        compose = getattr(self._agent, "compose_reminder", None)
        if compose is None:
            return static_phrase(config)

        try:
            import asyncio

            from bot.commands import _collect_transcript

            transcript = await _collect_transcript(
                channel, settings.REMINDER_HISTORY_LIMIT
            )
            if not transcript:
                return static_phrase(config)

            text = await asyncio.wait_for(
                asyncio.to_thread(compose, transcript),
                timeout=settings.AI_REQUEST_TIMEOUT_SECONDS + 10,
            )
            candidate = (text or "").strip()
            if not _is_valid_reminder_text(candidate):
                logger.warning(
                    "Текст напоминания от LLM не прошёл проверку, беру заготовку | "
                    "channel_id=%s | preview=%s",
                    getattr(channel, "id", "unknown"),
                    candidate[:120].replace("\n", " "),
                )
                return static_phrase(config)
            return candidate
        except Exception as exc:
            log_exception(
                "Не удалось сгенерировать текст напоминания, беру заготовку",
                exc,
                channel_id=getattr(channel, "id", "unknown"),
            )
            return static_phrase(config)

    # ── Проверка перед отправкой ─────────────────────────────────────────────
    async def _staff_replied_recently(self, channel, config: dict, started: float) -> bool:
        """Дочитывает хвост канала: не ответил ли персонал в обход состояния.

        Страховка от пропущенных событий и рестартов. Без неё возможен пинг в
        канал, где хелпер уже всё написал.
        """
        try:
            async for message in channel.history(limit=_VERIFY_HISTORY_LIMIT):
                author = getattr(message, "author", None)
                if author is None or getattr(author, "bot", False):
                    continue

                created = getattr(message, "created_at", None)
                timestamp = created.timestamp() if created is not None else 0.0
                if timestamp <= started:
                    continue

                if is_staff(author, config.get("staff_role_ids")):
                    return True
        except Exception as exc:
            log_exception(
                "Не удалось перечитать историю перед напоминанием",
                exc,
                channel_id=getattr(channel, "id", "unknown"),
            )
        return False

    # ── Основной проход ──────────────────────────────────────────────────────
    async def run_once(self) -> int:
        """Один проход по каналам. Возвращает число отправленных напоминаний."""
        if not settings.REMINDERS_ENABLED:
            return 0

        from bot.state import store

        excluded = set(settings.REMINDER_EXCLUDED_CATEGORY_IDS)
        ticket_categories = set(settings.TICKET_CATEGORY_IDS)
        sent = 0

        for channel_id, state in list(store.items()):
            channel = self._bot.get_channel(channel_id)
            if channel is None:
                continue

            category_id = getattr(channel, "category_id", None)
            if ticket_categories and category_id not in ticket_categories:
                continue
            if category_id in excluded:
                continue

            config = settings.reminder_config_for(category_id)
            if not should_remind(state, config):
                continue

            started = waiting_since(state)
            if await self._staff_replied_recently(channel, config, started):
                # Обновляем состояние, чтобы следующий проход не повторял проверку.
                state["last_staff_message_time"] = time.time()
                state["last_reminder_time"] = 0.0
                store.mark_dirty()
                continue

            if await self._send(channel, state, config, started):
                sent += 1

        if sent:
            store.save(force=True)
        return sent

    async def _send(self, channel, state: dict, config: dict, started: float) -> bool:
        from bot.discord_client import send_reminder

        text = await self._build_text(channel, config)
        ping_role_ids = config.get("ping_role_ids") or []
        
        # Фильтруем несуществующие роли: "@Неизвестная роль" в Discord = мусор.
        valid_role_ids = []
        guild = getattr(channel, "guild", None)
        if guild is not None and ping_role_ids:
            for role_id in ping_role_ids:
                if guild.get_role(role_id) is not None:
                    valid_role_ids.append(role_id)
                else:
                    logger.warning(
                        "Роль %s не найдена на сервере %s | channel_id=%s",
                        role_id,
                        guild.id,
                        getattr(channel, "id", "unknown"),
                    )
        else:
            valid_role_ids = list(ping_role_ids)
        
        if not valid_role_ids and ping_role_ids:
            logger.warning(
                "Все роли для пинга недействительны | channel_id=%s | "
                "было=%s | напоминание не отправлено",
                getattr(channel, "id", "unknown"),
                ping_role_ids,
            )
            return False
        
        content = format_reminder(text, valid_role_ids)
        message = await send_reminder(channel, content, valid_role_ids)
        if message is None:
            return False

        register_sent(state)
        from bot.state import store

        store.mark_dirty()

        waited_hours = (time.time() - started) / 3600 if started else 0
        logger.info(
            "Напоминание отправлено | channel_id=%s | ожидание=%.1fч | "
            "за сутки=%s/%s | роли=%s",
            getattr(channel, "id", "unknown"),
            waited_hours,
            state.get("reminder_count_today"),
            config.get("max_per_day"),
            valid_role_ids,
        )
        return True
