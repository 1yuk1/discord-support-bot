"""Тесты JSON-логов тикетов и архивации."""

import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot import settings, ticket_logs


@pytest.fixture
def log_dirs(tmp_path, monkeypatch):
    active = tmp_path / "active"
    archive = tmp_path / "archives"
    active.mkdir()
    archive.mkdir()
    monkeypatch.setattr(settings, "LOG_ACTIVE_DIR", str(active))
    monkeypatch.setattr(settings, "LOG_ARCHIVE_DIR", str(archive))
    monkeypatch.setattr(settings, "LOG_ARCHIVE_ENABLED", True)
    monkeypatch.setattr(
        settings,
        "LOG_TICKET_CATEGORIES",
        {"donate": ["донат", "donate"], "bug": ["баг", "bug"]},
    )
    return SimpleNamespace(active=active, archive=archive)


def make_channel(channel_id=555, name="donate-123"):
    return SimpleNamespace(id=channel_id, name=name)


def test_append_and_load_entry(log_dirs):
    channel = make_channel()
    ticket_logs.append_entry(channel, 42, "player", "донат не пришёл")

    entries = ticket_logs.load_log(channel)
    assert len(entries) == 1
    assert entries[0]["message"] == "донат не пришёл"
    assert entries[0]["username"] == "player"
    assert entries[0]["is_human_transfer"] is False


def test_entries_accumulate(log_dirs):
    channel = make_channel()
    for index in range(3):
        ticket_logs.append_entry(channel, 42, "player", f"сообщение {index}")
    assert len(ticket_logs.load_log(channel)) == 3


def test_optional_fields_recorded(log_dirs):
    channel = make_channel()
    ticket_logs.append_entry(
        channel,
        42,
        "player",
        "меня взломали",
        bot_response="Передам специалисту",
        is_human_transfer=True,
        transfer_reason="forced_keyword",
        image_urls=["https://cdn/1.png"],
    )
    entry = ticket_logs.load_log(channel)[0]
    assert entry["transfer_reason"] == "forced_keyword"
    assert entry["image_urls"] == ["https://cdn/1.png"]


@pytest.mark.parametrize(
    "channel_name,expected",
    [
        ("donate-123", "donate"),
        ("донат-456", "donate"),
        ("bug-report-1", "bug"),
        ("баг-2", "bug"),
        ("random-channel", "other"),
        ("", "other"),
    ],
)
def test_category_resolution(log_dirs, channel_name, expected):
    assert ticket_logs.resolve_category(channel_name) == expected


def test_corrupted_log_returns_empty(log_dirs):
    channel = make_channel()
    ticket_logs.get_log_path(channel).write_text("{ битый json", encoding="utf-8")
    assert ticket_logs.load_log(channel) == []


def test_archive_closed_ticket_moves_file(log_dirs):
    channel = make_channel(name="donate-777")
    ticket_logs.append_entry(channel, 42, "player", "оплатил через бусти")
    log_path = ticket_logs.get_log_path(channel)
    assert log_path.exists()

    ticket_logs.archive_closed_ticket(channel)

    assert not log_path.exists()
    archives = list((log_dirs.archive / "donate").glob("*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        assert log_path.name in archive.namelist()


def test_archive_skips_missing_log(log_dirs):
    ticket_logs.archive_closed_ticket(make_channel(channel_id=999))
    assert list(log_dirs.archive.rglob("*.zip")) == []


def test_archive_disabled_keeps_file(log_dirs, monkeypatch):
    monkeypatch.setattr(settings, "LOG_ARCHIVE_ENABLED", False)
    channel = make_channel()
    ticket_logs.append_entry(channel, 42, "player", "текст")

    ticket_logs.archive_closed_ticket(channel)
    assert ticket_logs.get_log_path(channel).exists()


def test_two_tickets_same_day_both_stored(log_dirs):
    """Второй тикет не должен вытеснять первый из общего архива."""
    for channel_id in (111, 222):
        channel = make_channel(channel_id=channel_id, name="donate-x")
        ticket_logs.append_entry(channel, 42, "player", "текст")
        ticket_logs.archive_closed_ticket(channel)

    archives = list((log_dirs.archive / "donate").glob("*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        assert len(archive.namelist()) == 2


def test_orphaned_log_archived_with_correct_category(log_dirs, monkeypatch):
    """Категория берётся из содержимого лога, а не из имени файла.

    Имя файла содержит только channel_id, поэтому раньше все брошенные логи
    складывались в "other" независимо от темы тикета.
    """
    monkeypatch.setattr(settings, "LOG_ARCHIVE_SAFETY_NET_DAYS", 7)
    channel = make_channel(channel_id=333, name="donate-old")
    ticket_logs.append_entry(channel, 42, "player", "старый тикет")

    log_path = ticket_logs.get_log_path(channel)
    old_time = time.time() - 10 * 24 * 3600
    import os

    os.utime(log_path, (old_time, old_time))

    ticket_logs.archive_orphaned_logs()

    assert not log_path.exists()
    assert list((log_dirs.archive / "donate").glob("*.zip"))


def test_recent_log_not_archived(log_dirs, monkeypatch):
    monkeypatch.setattr(settings, "LOG_ARCHIVE_SAFETY_NET_DAYS", 7)
    channel = make_channel(channel_id=444)
    ticket_logs.append_entry(channel, 42, "player", "свежий тикет")

    ticket_logs.archive_orphaned_logs()
    assert ticket_logs.get_log_path(channel).exists()


def test_safety_net_disabled(log_dirs, monkeypatch):
    monkeypatch.setattr(settings, "LOG_ARCHIVE_SAFETY_NET_DAYS", 0)
    channel = make_channel(channel_id=555)
    ticket_logs.append_entry(channel, 42, "player", "текст")

    log_path = ticket_logs.get_log_path(channel)
    old_time = time.time() - 100 * 24 * 3600
    import os

    os.utime(log_path, (old_time, old_time))

    ticket_logs.archive_orphaned_logs()
    assert log_path.exists()
