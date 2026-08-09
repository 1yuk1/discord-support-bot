"""Тесты подмешивания инцидентов в запрос к LLM.

Инцидент должен влиять на ЛЮБОЙ вопрос игрока, а не только на тот, что угадал
векторный поиск. Поэтому он идёт отдельным system-сообщением, минуя RAG.
"""

import pytest

from bot import incidents, settings
from bot.llm import SupportAgent


class FakeIndex:
    def search(self, query, top_k=None):
        return "## Проблема\nСтарая инструкция из базы"


class FakeCompletions:
    def __init__(self):
        self.last_messages = None

    def create(self, model, messages, **kwargs):
        self.last_messages = messages

        class Choice:
            message = type("Message", (), {"content": "Ответ бота"})()

        return type("Response", (), {"choices": [Choice()]})()


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


@pytest.fixture(autouse=True)
def isolated_incidents(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "INCIDENTS_FILE", str(tmp_path / "incidents.md"))
    monkeypatch.setattr(settings, "INCIDENTS_ENABLED", True)
    incidents.store.invalidate()
    incidents.store._cache = []
    yield
    incidents.store.invalidate()
    incidents.store._cache = []


@pytest.fixture
def agent():
    return SupportAgent(FakeClient(), FakeIndex())


def system_messages(agent) -> str:
    messages = agent._client.chat.completions.last_messages
    return "\n".join(
        str(message["content"])
        for message in messages
        if message["role"] == "system"
    )


def test_incident_reaches_prompt(agent):
    incidents.add("Не работает вход", "Авторизация недоступна, чиним.")
    agent.generate_answer("почему не могу зайти")

    assert "Авторизация недоступна" in system_messages(agent)


def test_incident_applies_to_unrelated_question(agent):
    """Главное отличие от базы знаний: работает без совпадения по смыслу."""
    incidents.add("Сервер лежит", "Идут аварийные работы.")
    agent.generate_answer("как купить донат")

    assert "Идут аварийные работы" in system_messages(agent)


def test_no_incident_block_when_empty(agent):
    agent.generate_answer("как купить донат")

    messages = agent._client.chat.completions.last_messages
    system_count = sum(1 for message in messages if message["role"] == "system")
    assert system_count == 1


def test_incident_block_absent_when_disabled(agent, monkeypatch):
    incidents.add("Сервер лежит", "Идут работы.")
    monkeypatch.setattr(settings, "INCIDENTS_ENABLED", False)
    agent.generate_answer("вопрос")

    assert "Идут работы" not in system_messages(agent)
