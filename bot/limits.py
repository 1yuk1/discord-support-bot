"""Ограничители частоты и дедупликация.

Модуль намеренно не зависит от discord.py и aiohttp: это чистая логика,
которую нужно уметь тестировать без установки Discord-зависимостей.
Всё, что реально общается с Discord, лежит в bot/discord_client.py.
"""

import time
from collections import deque

from bot import settings
from bot.text_utils import normalize_for_dedup


class GlobalRateLimiter:
    """Общий лимит ответов на всех пользователей и каналы."""

    def __init__(self) -> None:
        self._times: deque[float] = deque()

    def allow(self) -> bool:
        if not settings.RATE_LIMIT_ENABLED or settings.RATE_LIMIT <= 0 or settings.RATE_WINDOW <= 0:
            return True

        now = time.time()
        while self._times and now - self._times[0] > settings.RATE_WINDOW:
            self._times.popleft()

        if len(self._times) >= settings.RATE_LIMIT:
            return False

        self._times.append(now)
        return True


global_rate_limiter = GlobalRateLimiter()


def channel_cooldown_remaining(state: dict) -> int:
    """Сколько секунд осталось до следующего разрешённого ответа в канале."""
    if not settings.RATE_LIMIT_ENABLED or settings.CHANNEL_COOLDOWN <= 0:
        return 0

    elapsed = time.time() - state.get("last_answer_time", 0)
    if elapsed < settings.CHANNEL_COOLDOWN:
        return max(1, int(settings.CHANNEL_COOLDOWN - elapsed))
    return 0


def is_duplicate_message(state: dict, message_content: str) -> bool:
    """Повтор сообщения в канале.

    Проверяет окно последних сообщений (ловит повтор, между которым влезло
    чужое сообщение) и строгий «дубль подряд» по таймауту.
    """
    if not settings.RATE_LIMIT_ENABLED or settings.DUPLICATE_CHECK_TIME <= 0:
        return False

    normalized = normalize_for_dedup(message_content)
    if not normalized:
        return False

    recent = state.get("recent_normalized")
    if recent is not None and normalized in recent:
        return True

    last_normalized = normalize_for_dedup(state.get("last_message", ""))
    elapsed = time.time() - state.get("last_message_time", 0)
    if last_normalized == normalized and elapsed < settings.DUPLICATE_CHECK_TIME:
        return True

    if recent is not None:
        recent.append(normalized)
    return False


def user_flood_detected(state: dict) -> bool:
    """Пользователь превысил лимит сообщений в окне."""
    if not settings.RATE_LIMIT_ENABLED or settings.USER_MESSAGE_LIMIT <= 0:
        return False

    now = time.time()
    times = state.get("user_messages")
    if times is None:
        times = deque()
        state["user_messages"] = times

    while times and now - times[0] > settings.USER_MESSAGE_WINDOW:
        times.popleft()

    times.append(now)
    return len(times) > settings.USER_MESSAGE_LIMIT


def register_ping_burst(state: dict, mention_count: int) -> bool:
    """Учитывает пинги и сообщает, пора ли считать это спамом."""
    if mention_count <= 0:
        return False

    now = time.time()
    ping_times = state.get("human_mode_ping_times")
    if ping_times is None:
        ping_times = deque()
        state["human_mode_ping_times"] = ping_times

    while ping_times and now - ping_times[0] > settings.PING_SPAM_WINDOW:
        ping_times.popleft()
    ping_times.extend([now] * mention_count)

    return len(ping_times) >= settings.PING_SPAM_LIMIT


def add_reply_footer(text: str, footer_text: str) -> str:
    """Добавляет мелкую подпись под ответ бота."""
    footer = (footer_text or "").strip()
    if not footer:
        return text
    return f"{text.rstrip()}\n-# {footer}"
