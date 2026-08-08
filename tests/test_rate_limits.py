"""Тесты ограничителей: дедуп, кулдаун, флуд, общий лимит."""

import time

import pytest

from bot import settings
from bot.limits import (
    GlobalRateLimiter,
    add_reply_footer,
    channel_cooldown_remaining,
    is_duplicate_message,
    register_ping_burst,
    user_flood_detected,
)
from bot.state import create_channel_state


@pytest.fixture
def state():
    return create_channel_state()


@pytest.fixture(autouse=True)
def enable_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "DUPLICATE_CHECK_TIME", 5)
    monkeypatch.setattr(settings, "CHANNEL_COOLDOWN", 5)
    monkeypatch.setattr(settings, "USER_MESSAGE_LIMIT", 3)
    monkeypatch.setattr(settings, "USER_MESSAGE_WINDOW", 10)


def test_first_message_is_not_duplicate(state):
    assert not is_duplicate_message(state, "не могу зайти")


def test_repeat_detected(state):
    is_duplicate_message(state, "не могу зайти")
    assert is_duplicate_message(state, "не могу зайти")


def test_repeat_with_different_mention_detected(state):
    """Тикет-бот шлёт то же сообщение с другим пингом."""
    is_duplicate_message(state, "<@111> тикет будет закрыт")
    assert is_duplicate_message(state, "<@999> тикет  будет   закрыт")


def test_different_messages_pass(state):
    assert not is_duplicate_message(state, "первый вопрос")
    assert not is_duplicate_message(state, "второй вопрос")


def test_repeat_after_other_message_still_caught(state):
    """Окно последних сообщений ловит повтор, между которым влезло чужое."""
    is_duplicate_message(state, "донат не пришёл")
    is_duplicate_message(state, "жду ответа")
    assert is_duplicate_message(state, "донат не пришёл")


def test_empty_message_not_duplicate(state):
    assert not is_duplicate_message(state, "")
    assert not is_duplicate_message(state, "   ")


def test_dedup_disabled(state, monkeypatch):
    monkeypatch.setattr(settings, "DUPLICATE_CHECK_TIME", 0)
    is_duplicate_message(state, "текст")
    assert not is_duplicate_message(state, "текст")


def test_cooldown_counts_down(state):
    state["last_answer_time"] = time.time()
    assert channel_cooldown_remaining(state) > 0


def test_cooldown_expired(state):
    state["last_answer_time"] = time.time() - 100
    assert channel_cooldown_remaining(state) == 0


def test_cooldown_disabled(state, monkeypatch):
    monkeypatch.setattr(settings, "CHANNEL_COOLDOWN", 0)
    state["last_answer_time"] = time.time()
    assert channel_cooldown_remaining(state) == 0


def test_flood_after_limit(state):
    for _ in range(settings.USER_MESSAGE_LIMIT):
        assert not user_flood_detected(state)
    assert user_flood_detected(state)


def test_flood_window_slides(state, monkeypatch):
    monkeypatch.setattr(settings, "USER_MESSAGE_WINDOW", 1)
    for _ in range(settings.USER_MESSAGE_LIMIT):
        user_flood_detected(state)

    state["user_messages"].clear()
    assert not user_flood_detected(state)


def test_global_limiter_blocks_after_limit(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT", 3)
    monkeypatch.setattr(settings, "RATE_WINDOW", 60)
    limiter = GlobalRateLimiter()

    assert all(limiter.allow() for _ in range(3))
    assert not limiter.allow()


def test_global_limiter_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    limiter = GlobalRateLimiter()
    assert all(limiter.allow() for _ in range(100))


def test_ping_burst_triggers_after_limit(state, monkeypatch):
    monkeypatch.setattr(settings, "PING_SPAM_LIMIT", 3)
    monkeypatch.setattr(settings, "PING_SPAM_WINDOW", 300)

    assert not register_ping_burst(state, 1)
    assert not register_ping_burst(state, 1)
    assert register_ping_burst(state, 1)


def test_ping_burst_counts_multiple_mentions_at_once(state, monkeypatch):
    monkeypatch.setattr(settings, "PING_SPAM_LIMIT", 3)
    assert register_ping_burst(state, 3)


def test_ping_burst_ignores_messages_without_mentions(state):
    assert not register_ping_burst(state, 0)
    assert not register_ping_burst(state, -1)


def test_footer_appended():
    assert add_reply_footer("ответ", "подсказка") == "ответ\n-# подсказка"


def test_footer_skipped_when_empty():
    assert add_reply_footer("ответ", "") == "ответ"
    assert add_reply_footer("ответ", "   ") == "ответ"
