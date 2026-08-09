"""Тесты напоминаний персоналу.

Проверяют главное: напоминание уходит, только когда игрок реально ждёт, и не
превращается в спам. Discord и LLM подменяются заглушками.

Async-тесты запускаются через asyncio.run — так же, как в test_handlers.py:
pytest-asyncio в зависимостях проекта нет.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from bot import reminders, settings
from bot.state import create_channel_state

HOUR = 3600


def run(coro):
    return asyncio.run(coro)


def make_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "staff_role_ids": [500],
        "ping_role_ids": [900],
        "idle_hours": 1.0,
        "repeat_hours": 6.0,
        "max_per_day": 3,
        "message_mode": "static",
        "phrases": ["Ваш вопрос всё ещё в работе."],
    }
    config.update(overrides)
    return config


def waiting_state(hours_ago: float = 2.0) -> dict:
    """Состояние канала, где игрок написал hours_ago часов назад."""
    state = create_channel_state()
    state["last_player_message_time"] = time.time() - hours_ago * HOUR
    state["reminder_day"] = reminders._today_key()
    return state


# ── Кто писал последним ──────────────────────────────────────────────────────
def test_player_message_starts_waiting():
    state = create_channel_state()
    reminders.record_activity(state, is_staff_author=False)
    assert reminders.waiting_since(state) > 0


def test_staff_reply_stops_waiting():
    state = waiting_state()
    reminders.record_activity(state, is_staff_author=True)
    assert reminders.waiting_since(state) == 0


def test_staff_reply_resets_reminder_cycle():
    """Иначе после ответа хелпера остался бы висеть старый repeat-таймер."""
    state = waiting_state()
    state["last_reminder_time"] = time.time()
    reminders.record_activity(state, is_staff_author=True)
    assert state["last_reminder_time"] == 0.0


def test_bot_messages_count_for_nobody():
    """Сообщение тикет-бота — не ответ персонала и не обращение игрока."""
    state = create_channel_state()
    reminders.record_activity(state, is_staff_author=False, is_bot_author=True)
    assert state["last_player_message_time"] == 0.0
    assert state["last_staff_message_time"] == 0.0


def test_is_staff_checks_roles():
    member = SimpleNamespace(roles=[SimpleNamespace(id=500)])
    assert reminders.is_staff(member, [500]) is True
    assert reminders.is_staff(member, [777]) is False
    assert reminders.is_staff(None, [500]) is False
    assert reminders.is_staff(member, []) is False


# ── Условия отправки ─────────────────────────────────────────────────────────
def test_reminds_after_idle_threshold():
    assert reminders.should_remind(waiting_state(2.0), make_config()) is True


def test_silent_before_idle_threshold():
    assert reminders.should_remind(waiting_state(0.5), make_config()) is False


def test_silent_when_staff_answered_last():
    """Тикет в работе: хелпер ответил позже игрока, ожидания нет."""
    state = waiting_state(5.0)
    state["last_staff_message_time"] = time.time() - HOUR
    assert reminders.should_remind(state, make_config()) is False


def test_silent_when_nobody_wrote():
    assert reminders.should_remind(create_channel_state(), make_config()) is False


def test_silent_without_ping_roles():
    """Тестовый сервер без роли поддержки: пинговать некого."""
    assert reminders.should_remind(waiting_state(), make_config(ping_role_ids=[])) is False


def test_silent_when_disabled_in_config():
    assert reminders.should_remind(waiting_state(), make_config(enabled=False)) is False


def test_silent_when_disabled_in_channel():
    state = waiting_state()
    state["reminders_disabled"] = True
    assert reminders.should_remind(state, make_config()) is False


def test_repeat_interval_respected():
    state = waiting_state(10.0)
    state["last_reminder_time"] = time.time() - HOUR
    assert reminders.should_remind(state, make_config(repeat_hours=6)) is False


def test_reminds_again_after_repeat_interval():
    state = waiting_state(20.0)
    state["last_reminder_time"] = time.time() - 7 * HOUR
    assert reminders.should_remind(state, make_config(repeat_hours=6)) is True


def test_daily_limit_respected():
    state = waiting_state(10.0)
    state["reminder_count_today"] = 3
    assert reminders.should_remind(state, make_config(max_per_day=3)) is False


def test_daily_counter_resets_on_new_day():
    state = waiting_state(10.0)
    state["reminder_count_today"] = 3
    state["reminder_day"] = "2000-01-01"
    assert reminders.should_remind(state, make_config(max_per_day=3)) is True
    assert state["reminder_count_today"] == 0


def test_register_sent_increments_counter():
    state = waiting_state()
    reminders.register_sent(state)
    reminders.register_sent(state)
    assert state["reminder_count_today"] == 2
    assert state["last_reminder_time"] > 0


# ── Текст сообщения ──────────────────────────────────────────────────────────
def test_format_reminder_appends_role_mentions():
    text = reminders.format_reminder("Ждём немного.", [900, 901])
    assert "<@&900>" in text
    assert "<@&901>" in text
    assert text.startswith("Ждём немного.")


def test_format_reminder_without_roles():
    assert reminders.format_reminder("Текст", []) == "Текст"


def test_static_phrase_from_config():
    assert reminders.static_phrase(make_config()) == "Ваш вопрос всё ещё в работе."


# ── Настройки по категориям ──────────────────────────────────────────────────
def test_category_override_replaces_ping_roles(monkeypatch):
    monkeypatch.setattr(settings, "REMINDER_PING_ROLE_IDS", [111])
    monkeypatch.setattr(
        settings, "REMINDER_CATEGORY_OVERRIDES", {42: {"ping_role_ids": [222], "idle_hours": 3}}
    )

    config = settings.reminder_config_for(42)
    assert config["ping_role_ids"] == [222]
    assert config["idle_hours"] == 3.0

    assert settings.reminder_config_for(999)["ping_role_ids"] == [111]


def test_category_override_can_disable(monkeypatch):
    """Тестовый сервер: категория есть, роли поддержки нет."""
    monkeypatch.setattr(settings, "REMINDER_CATEGORY_OVERRIDES", {42: {"enabled": False}})
    assert settings.reminder_config_for(42)["enabled"] is False


def test_staff_roles_default_to_ignored_roles(monkeypatch):
    """ignored_role_ids — это и есть «не игроки», дублировать их не нужно."""
    monkeypatch.setattr(settings, "REMINDER_STAFF_ROLE_IDS", [])
    monkeypatch.setattr(settings, "IGNORED_ROLE_IDS", [999])
    monkeypatch.setattr(settings, "REMINDER_CATEGORY_OVERRIDES", {})

    assert settings.reminder_config_for(1)["staff_role_ids"] == [999]


# ── Заглушки Discord ─────────────────────────────────────────────────────────
class FakeMessage:
    def __init__(self, author, created_at):
        self.author = author
        self.created_at = SimpleNamespace(timestamp=lambda: created_at)


class FakeChannel:
    def __init__(self, channel_id=1, category_id=111, messages=()):
        self.id = channel_id
        self.category_id = category_id
        self.sent = []
        self._messages = list(messages)

    async def send(self, content, **kwargs):
        self.sent.append(content)
        return SimpleNamespace(id=len(self.sent))

    def history(self, limit=None, oldest_first=False):
        messages = self._messages

        class Iterator:
            def __aiter__(self):
                async def generator():
                    for message in messages:
                        yield message

                return generator()

        return Iterator()


class FakeBot:
    def __init__(self, channels):
        self._channels = {channel.id: channel for channel in channels}

    def get_channel(self, channel_id):
        return self._channels.get(channel_id)


class FakeAgent:
    def __init__(self, text="Сгенерированный текст."):
        self.text = text
        self.calls = 0

    def compose_reminder(self, transcript):
        self.calls += 1
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


@pytest.fixture(autouse=True)
def reminder_environment(tmp_path, monkeypatch):
    from bot.state import store

    monkeypatch.setattr(settings, "REMINDERS_ENABLED", True)
    monkeypatch.setattr(settings, "TICKET_CATEGORY_IDS", [111])
    monkeypatch.setattr(settings, "REMINDER_PING_ROLE_IDS", [900])
    monkeypatch.setattr(settings, "REMINDER_STAFF_ROLE_IDS", [500])
    monkeypatch.setattr(settings, "REMINDER_EXCLUDED_CATEGORY_IDS", [])
    monkeypatch.setattr(settings, "REMINDER_CATEGORY_OVERRIDES", {})
    monkeypatch.setattr(settings, "REMINDER_MESSAGE_MODE", "static")
    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(tmp_path / "state.json"))

    store._channels.clear()
    yield store
    store._channels.clear()


# ── Сервис ───────────────────────────────────────────────────────────────────
def test_service_sends_reminder(reminder_environment):
    channel = FakeChannel()
    reminder_environment._channels[channel.id] = waiting_state(2.0)
    service = reminders.ReminderService(FakeBot([channel]), FakeAgent())

    assert run(service.run_once()) == 1
    assert len(channel.sent) == 1
    assert "<@&900>" in channel.sent[0]


def test_service_skips_channel_outside_ticket_categories(reminder_environment):
    channel = FakeChannel(category_id=777)
    reminder_environment._channels[channel.id] = waiting_state(2.0)
    service = reminders.ReminderService(FakeBot([channel]), FakeAgent())

    assert run(service.run_once()) == 0
    assert channel.sent == []


def test_service_skips_excluded_category(reminder_environment, monkeypatch):
    monkeypatch.setattr(settings, "REMINDER_EXCLUDED_CATEGORY_IDS", [111])
    channel = FakeChannel()
    reminder_environment._channels[channel.id] = waiting_state(2.0)
    service = reminders.ReminderService(FakeBot([channel]), FakeAgent())

    assert run(service.run_once()) == 0


def test_service_rechecks_history_before_sending(reminder_environment):
    """Страховка от пропущенных событий: хелпер ответил, состояние отстало."""
    state = waiting_state(2.0)
    staff = SimpleNamespace(bot=False, roles=[SimpleNamespace(id=500)])
    channel = FakeChannel(messages=[FakeMessage(staff, time.time())])
    reminder_environment._channels[channel.id] = state

    service = reminders.ReminderService(FakeBot([channel]), FakeAgent())

    assert run(service.run_once()) == 0
    assert channel.sent == []
    assert reminders.waiting_since(state) == 0


def test_service_ignores_player_messages_in_history_check(reminder_environment):
    """Игрок писал ещё раз — это не ответ персонала, напоминание нужно."""
    player = SimpleNamespace(bot=False, roles=[])
    channel = FakeChannel(messages=[FakeMessage(player, time.time())])
    reminder_environment._channels[channel.id] = waiting_state(2.0)

    service = reminders.ReminderService(FakeBot([channel]), FakeAgent())
    assert run(service.run_once()) == 1


def test_llm_failure_falls_back_to_static_phrase(reminder_environment, monkeypatch):
    """Ошибка модели не должна отменять напоминание."""
    monkeypatch.setattr(settings, "REMINDER_MESSAGE_MODE", "llm")
    channel = FakeChannel()
    reminder_environment._channels[channel.id] = waiting_state(2.0)

    service = reminders.ReminderService(
        FakeBot([channel]), FakeAgent(RuntimeError("модель недоступна"))
    )

    assert run(service.run_once()) == 1
    assert len(channel.sent) == 1
    assert "<@&900>" in channel.sent[0]


def test_service_respects_daily_limit(reminder_environment):
    channel = FakeChannel()
    state = waiting_state(2.0)
    state["reminder_count_today"] = 3
    reminder_environment._channels[channel.id] = state

    service = reminders.ReminderService(FakeBot([channel]), FakeAgent())
    assert run(service.run_once()) == 0


def test_service_does_not_repeat_immediately(reminder_environment):
    channel = FakeChannel()
    reminder_environment._channels[channel.id] = waiting_state(2.0)
    service = reminders.ReminderService(FakeBot([channel]), FakeAgent())

    assert run(service.run_once()) == 1
    assert run(service.run_once()) == 0
    assert len(channel.sent) == 1
