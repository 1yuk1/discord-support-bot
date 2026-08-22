"""Тесты надежности LLM и тихих ночных часов напоминаний."""

import asyncio
from datetime import datetime, timezone
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from bot import llm, reminders, settings
from bot.llm import LlmProvider, SupportAgent, _is_retryable_error
from bot.reminders import is_in_quiet_hours, should_remind


def test_is_retryable_error():
    assert _is_retryable_error(Exception("ReadTimeout: The read operation timed out")) is True
    assert _is_retryable_error(Exception("Rate limit exceeded 429")) is True
    assert _is_retryable_error(Exception("502 Bad Gateway")) is True
    assert _is_retryable_error(Exception("500 Internal Server Error")) is True
    assert _is_retryable_error(SimpleNamespace(status_code=429)) is True
    assert _is_retryable_error(SimpleNamespace(status_code=503)) is True

    # Fatal / Non-retryable
    assert _is_retryable_error(SimpleNamespace(status_code=401)) is False
    assert _is_retryable_error(SimpleNamespace(status_code=400)) is False
    assert _is_retryable_error(SimpleNamespace(status_code=403)) is False
    assert _is_retryable_error(SimpleNamespace(status_code=404)) is False
    assert _is_retryable_error(Exception("401 Unauthorized: Invalid API key")) is False
    assert _is_retryable_error(Exception("400 Bad Request: Model not found")) is False


def test_llm_retries_and_fallback():
    # Мок первичного провайдера с ошибкой 429
    primary_client = MagicMock()
    primary_client.chat.completions.create.side_effect = Exception("429 Too Many Requests")

    # Мок резервного провайдера (успех)
    fallback_client = MagicMock()
    success_resp = MagicMock()
    success_resp.choices = [
        MagicMock(finish_reason="stop", message=MagicMock(content="Ответ от fallback"))
    ]
    fallback_client.chat.completions.create.return_value = success_resp

    provider1 = LlmProvider("openrouter", primary_client, llm.models)
    provider2 = LlmProvider("fallback", fallback_client, llm.fallback_models)

    index_mock = MagicMock()
    index_mock.search.return_value = "Контекст"
    agent = SupportAgent([provider1, provider2], index_mock)

    ans = agent.generate_answer("Привет")
    assert ans == "Ответ от fallback"
    assert primary_client.chat.completions.create.call_count == 2
    assert fallback_client.chat.completions.create.call_count == 1


def test_llm_no_retry_on_401():
    primary_client = MagicMock()
    primary_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

    fallback_client = MagicMock()
    success_resp = MagicMock()
    success_resp.choices = [
        MagicMock(finish_reason="stop", message=MagicMock(content="Ответ от fallback"))
    ]
    fallback_client.chat.completions.create.return_value = success_resp

    provider1 = LlmProvider("openrouter", primary_client, llm.models)
    provider2 = LlmProvider("fallback", fallback_client, llm.fallback_models)

    index_mock = MagicMock()
    index_mock.search.return_value = ""
    agent = SupportAgent([provider1, provider2], index_mock)

    ans = agent.generate_answer("Привет")
    assert ans == "Ответ от fallback"
    # На 401 не должно быть повторов (1 вызов вместо 2)
    assert primary_client.chat.completions.create.call_count == 1
    assert fallback_client.chat.completions.create.call_count == 1


def test_compose_reminder_max_tokens_and_fallback():
    client_mock = MagicMock()
    success_resp = MagicMock()
    success_resp.choices = [
        MagicMock(finish_reason="stop", message=MagicMock(content="Короткое напоминание"))
    ]
    client_mock.chat.completions.create.return_value = success_resp

    provider = LlmProvider("openrouter", client_mock, llm.models)
    agent = SupportAgent(provider, MagicMock())

    text = agent.compose_reminder("транскрипт")
    assert text == "Короткое напоминание"
    # Проверяем max_tokens=160
    call_kwargs = client_mock.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] == 160


def test_quiet_hours_check():
    cfg = {
        "quiet_hours_timezone": "Europe/Moscow",
        "quiet_hours_start": "23:00",
        "quiet_hours_end": "09:00",
    }

    # 23:30 MSK (20:30 UTC) -> в окне тишины
    dt_night = datetime(2026, 8, 22, 20, 30, tzinfo=timezone.utc)
    assert is_in_quiet_hours(cfg, dt_night) is True

    # 04:00 MSK (01:00 UTC) -> в окне тишины
    dt_early = datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc)
    assert is_in_quiet_hours(cfg, dt_early) is True

    # 08:59 MSK (05:59 UTC) -> в окне тишины
    dt_before_end = datetime(2026, 8, 22, 5, 59, tzinfo=timezone.utc)
    assert is_in_quiet_hours(cfg, dt_before_end) is True

    # 09:01 MSK (06:01 UTC) -> вне окна тишины
    dt_day = datetime(2026, 8, 22, 6, 1, tzinfo=timezone.utc)
    assert is_in_quiet_hours(cfg, dt_day) is False

    # 15:00 MSK (12:00 UTC) -> вне окна тишины
    dt_afternoon = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
    assert is_in_quiet_hours(cfg, dt_afternoon) is False


def test_should_remind_skips_during_quiet_hours_without_consuming_limits():
    cfg = {
        "enabled": True,
        "ping_role_ids": [900],
        "staff_role_ids": [500],
        "idle_hours": 1.0,
        "repeat_hours": 6.0,
        "max_per_day": 3,
        "quiet_hours_timezone": "Europe/Moscow",
        "quiet_hours_start": "23:00",
        "quiet_hours_end": "09:00",
    }

    # Игрок ждёт 2 часа
    state = {
        "last_player_message_time": 1000.0,
        "last_staff_message_time": 0.0,
        "last_reminder_time": 0.0,
        "reminder_count_today": 0,
        "reminder_day": "2026-08-22",
    }

    # Ночь в Москве (02:00 MSK, 23:00 UTC предыдущего дня)
    dt_night = datetime(2026, 8, 21, 23, 0, tzinfo=timezone.utc)
    now_night_ts = dt_night.timestamp()
    state["last_player_message_time"] = now_night_ts - 7200

    # В тихие часы should_remind = False
    assert should_remind(state, cfg, now=now_night_ts) is False
    # Лимиты и счетчики не изменились
    assert state["reminder_count_today"] == 0
    assert state["last_reminder_time"] == 0.0

    # Утро после 09:00 (10:00 MSK, 07:00 UTC)
    dt_morning = datetime(2026, 8, 22, 7, 0, tzinfo=timezone.utc)
    now_morning_ts = dt_morning.timestamp()

    # Утром должно сработать
    assert should_remind(state, cfg, now=now_morning_ts) is True
