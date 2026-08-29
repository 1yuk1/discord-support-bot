"""Тесты модерации запрещённых упоминаний."""

import asyncio
from datetime import timedelta
import time
from types import SimpleNamespace
import pytest

from bot import mention_moderation, settings
from bot.state import store


def run(coro):
    return asyncio.run(coro)


class FakeGuild:
    def __init__(self, guild_id=123):
        self.id = guild_id


class FakeUser:
    def __init__(self, user_id, name="TestUser", is_bot=False, is_system=False):
        self.id = user_id
        self.name = name
        self.display_name = name
        self.bot = is_bot
        self.system = is_system
        self.timed_out_duration = None
        self.timed_out_reason = None
        self.timeout_raises = None

    async def timeout(self, duration, reason=None):
        if self.timeout_raises:
            raise self.timeout_raises
        self.timed_out_duration = duration
        self.timed_out_reason = reason


class FakeRole:
    def __init__(self, role_id, name="TestRole"):
        self.id = role_id
        self.name = name


class FakeChannel:
    def __init__(self, channel_id=456):
        self.id = channel_id
        self.sent = []

    async def send(self, content, **kwargs):
        self.sent.append(content)
        return SimpleNamespace(id=999)


class FakeMessage:
    def __init__(self, author, mentions=None, role_mentions=None, channel=None, guild=None):
        self.id = 888
        self.author = author
        self.mentions = mentions or []
        self.role_mentions = role_mentions or []
        self.channel = channel or FakeChannel()
        self.guild = guild or FakeGuild()
        self.content = "ping"


@pytest.fixture(autouse=True)
def setup_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MENTION_TIMEOUT_ENABLED", True)
    monkeypatch.setattr(settings, "MENTION_PROTECTED_USER_IDS", [100, 200])
    monkeypatch.setattr(settings, "MENTION_PROTECTED_ROLE_IDS", [300, 400])
    monkeypatch.setattr(
        settings,
        "MENTION_TIMEOUT_DURATIONS",
        [timedelta(minutes=1), timedelta(minutes=10), timedelta(hours=1)],
    )
    monkeypatch.setattr(settings, "MENTION_ESCALATION_RESET_DAYS", 30)
    monkeypatch.setattr(settings, "MENTION_TIMEOUT_REASON", "Запрещённый пинг")
    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(tmp_path / "state.json"))

    store._channels.clear()
    store._mention_escalations.clear()
    yield
    store._channels.clear()
    store._mention_escalations.clear()


def test_format_duration():
    assert mention_moderation.format_duration(timedelta(seconds=30)) == "30 сек."
    assert mention_moderation.format_duration(timedelta(minutes=1)) == "1 мин."
    assert mention_moderation.format_duration(timedelta(minutes=10)) == "10 мин."
    assert mention_moderation.format_duration(timedelta(hours=1)) == "1 ч."
    assert mention_moderation.format_duration(timedelta(days=2)) == "2 дн."


def test_check_forbidden_mentions():
    user1 = FakeUser(100, "Admin")
    user_unprotected = FakeUser(999, "Regular")
    role1 = FakeRole(300, "Support")
    role_unprotected = FakeRole(888, "Member")

    msg = FakeMessage(
        author=FakeUser(555),
        mentions=[user1, user_unprotected],
        role_mentions=[role1, role_unprotected],
    )

    targets = mention_moderation.check_forbidden_mentions(msg)
    assert "@Admin" in targets
    assert "@Support" in targets
    assert len(targets) == 2


def test_no_forbidden_mentions():
    msg = FakeMessage(
        author=FakeUser(555),
        mentions=[FakeUser(999, "Regular")],
        role_mentions=[FakeRole(888, "Member")],
    )
    assert mention_moderation.check_forbidden_mentions(msg) == []


def test_moderation_escalation_ladder():
    author = FakeUser(555, "Troublemaker")
    protected_user = FakeUser(100, "Admin")
    channel = FakeChannel()
    guild = FakeGuild(123)

    msg = FakeMessage(author=author, mentions=[protected_user], channel=channel, guild=guild)

    # 1-е нарушение -> 1 мин.
    assert run(mention_moderation.handle_mention_moderation(msg)) is True
    assert author.timed_out_duration == timedelta(minutes=1)
    assert len(channel.sent) == 1
    assert "1 мин." in channel.sent[0]

    # 2-е нарушение -> 10 мин.
    assert run(mention_moderation.handle_mention_moderation(msg)) is True
    assert author.timed_out_duration == timedelta(minutes=10)
    assert len(channel.sent) == 2
    assert "10 мин." in channel.sent[1]

    # 3-е нарушение -> 1 ч.
    assert run(mention_moderation.handle_mention_moderation(msg)) is True
    assert author.timed_out_duration == timedelta(hours=1)
    assert len(channel.sent) == 3
    assert "1 ч." in channel.sent[2]

    # 4-е нарушение -> остается 1 ч. (последняя ступень)
    assert run(mention_moderation.handle_mention_moderation(msg)) is True
    assert author.timed_out_duration == timedelta(hours=1)
    assert len(channel.sent) == 4


def test_multiple_pings_in_one_message_gives_single_timeout():
    author = FakeUser(555, "Spammer")
    p_user1 = FakeUser(100, "Admin1")
    p_user2 = FakeUser(200, "Admin2")
    p_role = FakeRole(300, "Support")
    channel = FakeChannel()

    msg = FakeMessage(
        author=author,
        mentions=[p_user1, p_user2],
        role_mentions=[p_role],
        channel=channel,
    )

    assert run(mention_moderation.handle_mention_moderation(msg)) is True
    assert author.timed_out_duration == timedelta(minutes=1)
    # Только одно сообщение с упоминанием всех нарушенных целей
    assert len(channel.sent) == 1
    assert "@Admin1" in channel.sent[0]
    assert "@Admin2" in channel.sent[0]
    assert "@Support" in channel.sent[0]


def test_ignores_bots_and_system():
    bot_author = FakeUser(555, "Bot", is_bot=True)
    msg_bot = FakeMessage(author=bot_author, mentions=[FakeUser(100)])
    assert run(mention_moderation.handle_mention_moderation(msg_bot)) is False

    system_author = FakeUser(666, "System", is_system=True)
    msg_sys = FakeMessage(author=system_author, mentions=[FakeUser(100)])
    assert run(mention_moderation.handle_mention_moderation(msg_sys)) is False


def test_admin_and_staff_bypass_mention_moderation(monkeypatch):
    # Настраиваем ID администраторов и персонала
    monkeypatch.setattr(settings, "ADMIN_USER_IDS", [1001])
    monkeypatch.setattr(settings, "ADMIN_ROLE_IDS", [2001])
    monkeypatch.setattr(settings, "REMINDER_STAFF_ROLE_IDS", [3001])

    # 1. Админ по user_id пингует защищённого админа 100
    admin_user = FakeUser(1001, "SuperAdmin")
    ch1 = FakeChannel()
    msg1 = FakeMessage(author=admin_user, mentions=[FakeUser(100)], channel=ch1)
    assert run(mention_moderation.handle_mention_moderation(msg1)) is False
    assert admin_user.timed_out_duration is None
    assert len(ch1.sent) == 0

    # 2. Админ по правам Discord Administrator пингует защищённую роль 300
    discord_admin = FakeUser(5555, "DiscordAdmin")
    discord_admin.guild_permissions = SimpleNamespace(administrator=True)
    ch2 = FakeChannel()
    msg2 = FakeMessage(author=discord_admin, role_mentions=[FakeRole(300)], channel=ch2)
    assert run(mention_moderation.handle_mention_moderation(msg2)) is False
    assert discord_admin.timed_out_duration is None
    assert len(ch2.sent) == 0

    # 3. Саппорт (роль персонала) пингует защищённого админа
    staff_user = FakeUser(7777, "Helper")
    staff_user.roles = [FakeRole(3001)]
    ch3 = FakeChannel()
    msg3 = FakeMessage(author=staff_user, mentions=[FakeUser(100)], channel=ch3)
    assert run(mention_moderation.handle_mention_moderation(msg3)) is False
    assert staff_user.timed_out_duration is None
    assert len(ch3.sent) == 0


def test_escalation_persists_across_restart():
    guild_id = 123
    user_id = 555
    store.record_mention_violation(guild_id, user_id)
    store.record_mention_violation(guild_id, user_id)
    store.save(force=True)

    # Имитация рестарта
    store._mention_escalations.clear()
    store.load()

    esc = store.get_mention_escalation(guild_id, user_id)
    assert esc["violations"] == 2


def test_escalation_resets_after_inactivity_period():
    guild_id = 123
    user_id = 555
    # Запись 35 дней назад (лимит 30 дней)
    old_time = time.time() - (35 * 86400)
    store.record_mention_violation(guild_id, user_id, now=old_time)

    esc = store.get_mention_escalation(guild_id, user_id)
    assert esc["violations"] == 0


def test_forbidden_error_handled_gracefully():
    import discord

    author = FakeUser(555, "AdminAboveBot")
    author.timeout_raises = discord.Forbidden(
        SimpleNamespace(status=403, reason="Forbidden"), "Missing Permissions"
    )
    channel = FakeChannel()
    msg = FakeMessage(author=author, mentions=[FakeUser(100)], channel=channel)

    # Не должно упасть с исключением
    assert run(mention_moderation.handle_mention_moderation(msg)) is True
    assert len(channel.sent) == 1
    assert "запрещено" in channel.sent[0]
