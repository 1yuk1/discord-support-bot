"""Тесты состояния каналов и его сохранения.

Отдельное внимание разделению human_mode и bot_disabled: это два независимых
способа заставить бота молчать, и они не должны влиять друг на друга.
"""

import time

import pytest

from bot import settings
from bot.state import ConversationStore, create_channel_state

CHANNEL_ID = 424242


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(tmp_path / "state.json"))
    return ConversationStore()


def test_new_state_defaults():
    state = create_channel_state()
    assert state["human_mode"] is False
    assert state["bot_disabled"] is False
    assert state["history"] == []


def test_get_or_create_is_stable(store):
    first = store.get_or_create(CHANNEL_ID)
    first["human_mode"] = True
    assert store.get_or_create(CHANNEL_ID) is first


def test_history_trimmed_to_max(store, monkeypatch):
    monkeypatch.setattr(settings, "MAX_HISTORY", 2)
    state = store.get_or_create(CHANNEL_ID)
    for index in range(10):
        store.append_turn(state, f"вопрос {index}", f"ответ {index}", author_is_bot=False)

    assert len(state["history"]) == 4
    assert state["history"][-1]["content"] == "ответ 9"


def test_append_turn_labels_author(store):
    state = store.get_or_create(CHANNEL_ID)
    store.append_turn(state, "текст", "ответ", author_is_bot=True)
    assert state["history"][0]["content"] == "[Система] текст"

    store.append_turn(state, "текст", "ответ", author_is_bot=False)
    assert state["history"][2]["content"] == "[Пользователь] текст"


def test_processed_ids_are_capped(store):
    state = store.get_or_create(CHANNEL_ID)
    for message_id in range(500):
        store.remember_processed(state, message_id)

    assert len(state["processed_message_ids"]) <= 200
    assert state["last_processed_message_id"] == 499


def test_reset_keeps_manual_disable(store):
    """!clear_history не должен включать бота там, где админ его выключил."""
    state = store.get_or_create(CHANNEL_ID)
    state["bot_disabled"] = True
    state["disabled_by"] = "admin#1"
    state["human_mode"] = True
    store.append_turn(state, "вопрос", "ответ", author_is_bot=False)

    fresh = store.reset(CHANNEL_ID)
    assert fresh["history"] == []
    assert fresh["human_mode"] is False
    assert fresh["bot_disabled"] is True
    assert fresh["disabled_by"] == "admin#1"


def test_disabled_flag_survives_restart(store, tmp_path, monkeypatch):
    state = store.get_or_create(CHANNEL_ID)
    state["bot_disabled"] = True
    state["disabled_by"] = "admin#1"
    store.mark_dirty()
    store.save()

    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(tmp_path / "state.json"))
    restored = ConversationStore()
    restored.load()

    reloaded = restored.get(CHANNEL_ID)
    assert reloaded is not None
    assert reloaded["bot_disabled"] is True
    assert reloaded["disabled_by"] == "admin#1"


def test_human_mode_survives_restart(store, tmp_path, monkeypatch):
    state = store.get_or_create(CHANNEL_ID)
    state["human_mode"] = True
    store.mark_dirty()
    store.save()

    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(tmp_path / "state.json"))
    restored = ConversationStore()
    restored.load()
    assert restored.get(CHANNEL_ID)["human_mode"] is True


def test_clean_channels_are_not_persisted(store):
    """Каналы без залипающих флагов в snapshot не попадают."""
    store.get_or_create(CHANNEL_ID)
    store.mark_dirty()
    store.save()

    restored = ConversationStore()
    restored.load()
    assert restored.get(CHANNEL_ID) is None


def test_expired_states_removed(store, monkeypatch):
    monkeypatch.setattr(settings, "STATE_TTL_SECONDS", 60)
    state = store.get_or_create(CHANNEL_ID)
    state["last_activity"] = time.time() - 600

    fresh_id = 999
    store.get_or_create(fresh_id)

    assert store.cleanup_expired() == 1
    assert store.get(CHANNEL_ID) is None
    assert store.get(fresh_id) is not None


def test_expired_states_not_restored(store, tmp_path, monkeypatch):
    state = store.get_or_create(CHANNEL_ID)
    state["human_mode"] = True
    state["last_activity"] = time.time() - 10 * 24 * 3600
    store.mark_dirty()
    store.save()

    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(settings, "STATE_TTL_SECONDS", 7 * 24 * 3600)
    restored = ConversationStore()
    restored.load()
    assert restored.get(CHANNEL_ID) is None


def test_remove_channel(store):
    store.get_or_create(CHANNEL_ID)
    assert store.remove(CHANNEL_ID) is True
    assert store.remove(CHANNEL_ID) is False
    assert store.remove(None) is False


def test_corrupted_snapshot_does_not_crash(store, tmp_path, monkeypatch):
    snapshot = tmp_path / "state.json"
    snapshot.write_text("{ это не json", encoding="utf-8")
    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(snapshot))

    restored = ConversationStore()
    restored.load()
    assert len(restored) == 0
