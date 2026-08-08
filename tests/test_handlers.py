"""Тесты роутера сообщений.

Проверяют главное: в каких случаях бот отвечает, а в каких молчит.
Discord-объекты подменяются заглушками, реальных запросов нет.
"""

import asyncio
from types import SimpleNamespace

import pytest

from bot import settings, ticket_logs
from bot.handlers import MessageRouter
from bot.state import store

TICKET_CATEGORY = 111
CHANNEL_ID = 7000


class FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeChannel:
    def __init__(self, channel_id=CHANNEL_ID, category_id=TICKET_CATEGORY, name="donate-1"):
        self.id = channel_id
        self.category_id = category_id
        self.name = name
        self.sent: list[str] = []

    async def send(self, content):
        self.sent.append(content)
        return SimpleNamespace(id=len(self.sent))

    def typing(self):
        return FakeTyping()


class FakeAuthor:
    def __init__(self, author_id=42, is_bot=False, roles=()):
        self.id = author_id
        self.bot = is_bot
        self.roles = [SimpleNamespace(id=role_id) for role_id in roles]

    def __str__(self):
        return f"user-{self.id}"


class FakeMessage:
    _next_id = 1

    def __init__(self, content="", channel=None, author=None, attachments=(), embeds=()):
        FakeMessage._next_id += 1
        self.id = FakeMessage._next_id
        self.content = content
        self.channel = channel or FakeChannel()
        self.author = author or FakeAuthor()
        self.attachments = list(attachments)
        self.embeds = list(embeds)
        self.mentions = []
        self.role_mentions = []
        self.guild = SimpleNamespace(get_member=lambda _: None)


class FakeBot:
    def __init__(self):
        self.user = SimpleNamespace(id=999, __str__=lambda self: "bot")
        self.processed_commands = 0

    async def process_commands(self, message):
        self.processed_commands += 1


class FakeAgent:
    def __init__(self, answer="Вот решение вашей проблемы."):
        self.answer = answer
        self.calls: list[str] = []

    def generate_answer(self, text, history=None, image_urls=None):
        self.calls.append(text)
        return self.answer

    def summarize_ticket(self, transcript):
        return "сводка"


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Изолирует состояние, логи и отключает склейку для предсказуемости."""
    monkeypatch.setattr(settings, "TICKET_CATEGORY_IDS", [TICKET_CATEGORY])
    monkeypatch.setattr(settings, "BOT_ROLE_IDS", [])
    monkeypatch.setattr(settings, "IGNORED_ROLE_IDS", [])
    monkeypatch.setattr(settings, "MESSAGE_DEBOUNCE_SECONDS", 0)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "LOG_ACTIVE_DIR", str(tmp_path / "active"))
    monkeypatch.setattr(settings, "LOG_ARCHIVE_DIR", str(tmp_path / "archives"))
    monkeypatch.setattr(settings, "STATE_SNAPSHOT_FILE", str(tmp_path / "state.json"))
    ticket_logs.ensure_directories()

    store._channels.clear()
    yield
    store._channels.clear()


@pytest.fixture
def router():
    return MessageRouter(FakeBot(), FakeAgent())


def run(coro):
    return asyncio.run(coro)


def test_answers_regular_question(router):
    channel = FakeChannel()
    run(router.handle_message(FakeMessage("не могу зайти на сервер", channel=channel)))
    assert len(channel.sent) == 1
    assert "решение" in channel.sent[0]


def test_ignores_own_messages(router):
    channel = FakeChannel()
    message = FakeMessage("текст", channel=channel, author=FakeAuthor(author_id=999, is_bot=True))
    run(router.handle_message(message))
    assert channel.sent == []


def test_ignores_channels_outside_ticket_categories(router):
    channel = FakeChannel(category_id=555)
    run(router.handle_message(FakeMessage("вопрос про донат", channel=channel)))
    assert channel.sent == []


def test_ignores_users_with_ignored_role(router, monkeypatch):
    monkeypatch.setattr(settings, "IGNORED_ROLE_IDS", [777])
    router._ignored_role_ids = {777}

    channel = FakeChannel()
    message = FakeMessage("вопрос", channel=channel, author=FakeAuthor(roles=[777]))
    run(router.handle_message(message))
    assert channel.sent == []


def test_ignores_trivial_messages(router):
    channel = FakeChannel()
    for text in ("??", "+", "ау", "60"):
        run(router.handle_message(FakeMessage(text, channel=channel)))
    assert channel.sent == []


def test_stop_flag_silences_bot(router):
    """Главное поведение !stop: сообщения логируются, но ответа нет."""
    channel = FakeChannel()
    state = store.get_or_create(channel.id)
    state["bot_disabled"] = True

    run(router.handle_message(FakeMessage("почему сервер не работает", channel=channel)))

    assert channel.sent == []
    entries = ticket_logs.load_log(channel)
    assert len(entries) == 1
    assert entries[0]["message"] == "почему сервер не работает"


def test_stop_flag_blocks_escalation_phrases(router):
    """Даже просьба позвать человека не должна пробивать !stop."""
    channel = FakeChannel()
    store.get_or_create(channel.id)["bot_disabled"] = True

    run(router.handle_message(FakeMessage("позови человека", channel=channel)))

    assert channel.sent == []
    assert store.get(channel.id)["human_mode"] is False


def test_stop_flag_blocks_forced_keywords(router):
    channel = FakeChannel()
    store.get_or_create(channel.id)["bot_disabled"] = True

    run(router.handle_message(FakeMessage("меня взломали", channel=channel)))

    assert channel.sent == []
    assert store.get(channel.id)["human_mode"] is False


def test_commands_still_reach_processing_when_disabled(router):
    """!start должен срабатывать в канале, где бот выключен."""
    channel = FakeChannel()
    store.get_or_create(channel.id)["bot_disabled"] = True

    run(router.handle_message(FakeMessage("!start", channel=channel)))
    assert router._bot.processed_commands == 1


def test_human_mode_silences_bot(router):
    channel = FakeChannel()
    store.get_or_create(channel.id)["human_mode"] = True

    run(router.handle_message(FakeMessage("ещё вопрос", channel=channel)))
    assert channel.sent == []


def test_explicit_transfer_request_activates_human_mode(router):
    channel = FakeChannel()
    run(router.handle_message(FakeMessage("позови человека", channel=channel)))

    assert len(channel.sent) == 1
    assert "старшему специалисту" in channel.sent[0]
    assert store.get(channel.id)["human_mode"] is True


def test_forced_keyword_activates_human_mode(router):
    channel = FakeChannel()
    run(router.handle_message(FakeMessage("меня взломали, украли вещи", channel=channel)))

    assert store.get(channel.id)["human_mode"] is True


def test_llm_transfer_answer_activates_human_mode():
    channel = FakeChannel()
    agent = FakeAgent("Я передам ваш тикет старшему специалисту. Ожидайте.")
    router = MessageRouter(FakeBot(), agent)

    run(router.handle_message(FakeMessage("тут всё сложно", channel=channel)))
    assert store.get(channel.id)["human_mode"] is True


def test_duplicate_message_answered_once(router, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "DUPLICATE_CHECK_TIME", 5)
    monkeypatch.setattr(settings, "CHANNEL_COOLDOWN", 0)
    monkeypatch.setattr(settings, "USER_MESSAGE_LIMIT", 0)

    channel = FakeChannel()
    run(router.handle_message(FakeMessage("донат не пришёл", channel=channel)))
    run(router.handle_message(FakeMessage("донат не пришёл", channel=channel)))

    assert len(channel.sent) == 1


def test_ticket_close_notification_never_answered(router):
    channel = FakeChannel()
    message = FakeMessage(
        "Тикет будет закрыт из-за бездействия",
        channel=channel,
        author=FakeAuthor(author_id=555, is_bot=True),
    )
    run(router.handle_message(message))
    assert channel.sent == []


def test_ticket_opening_answered_once(router):
    channel = FakeChannel()
    system_bot = FakeAuthor(author_id=555, is_bot=True)

    for _ in range(3):
        run(router.handle_message(
            FakeMessage("Игрок создал новый тикет: не работает вход", channel=channel, author=system_bot)
        ))

    assert len(channel.sent) == 1


def test_history_grows_after_answer(router):
    channel = FakeChannel()
    run(router.handle_message(FakeMessage("как купить донат", channel=channel)))

    history = store.get(channel.id)["history"]
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_screenshot_without_text_uses_placeholder():
    channel = FakeChannel()
    agent = FakeAgent()
    router = MessageRouter(FakeBot(), agent)

    attachment = SimpleNamespace(
        url="https://cdn/screen.png", content_type="image/png", filename="screen.png"
    )
    run(router.handle_message(FakeMessage("", channel=channel, attachments=[attachment])))

    assert agent.calls == ["[Игрок прислал скриншот]"]


def test_same_message_processed_once(router):
    channel = FakeChannel()
    message = FakeMessage("вопрос про режимы", channel=channel)

    run(router.handle_message(message))
    run(router.handle_message(message))

    assert len(channel.sent) == 1


def test_missing_bot_role_blocks_answer(router, monkeypatch):
    monkeypatch.setattr(settings, "BOT_ROLE_IDS", [321])
    router._bot_role_ids = {321}

    channel = FakeChannel()
    run(router.handle_message(FakeMessage("вопрос", channel=channel)))
    assert channel.sent == []
