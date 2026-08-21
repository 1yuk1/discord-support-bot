"""Проверка переключения OpenRouter -> резервный провайдер без реальных API-запросов."""

from bot.llm import LlmProvider, ModelRegistry, SupportAgent


class FakeIndex:
    def search(self, query, top_k=None):
        return "Контекст"


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        choice = type(
            "Choice",
            (), {"message": type("Message", (), {"content": outcome})(), "finish_reason": "stop"},
        )()
        return type("Response", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, outcomes):
        self.chat = type("Chat", (), {"completions": FakeCompletions(outcomes)})()


def make_agent(primary_outcomes, fallback_outcomes):
    primary = FakeClient(primary_outcomes)
    fallback = FakeClient(fallback_outcomes)
    providers = [
        LlmProvider("openrouter", primary, ModelRegistry("primary-model")),
        LlmProvider("fallback", fallback, ModelRegistry("fallback-model")),
    ]
    return SupportAgent(providers, FakeIndex()), primary, fallback


def test_primary_success_does_not_call_fallback():
    agent, primary, fallback = make_agent(["Основной ответ"], ["Резервный ответ"])

    assert agent.generate_answer("вопрос") == "Основной ответ"
    assert len(primary.chat.completions.calls) == 1
    assert fallback.chat.completions.calls == []


def test_fallback_reuses_request_with_its_own_model():
    agent, primary, fallback = make_agent([TimeoutError("timeout")], ["Резервный ответ"])

    assert agent.generate_answer("вопрос") == "Резервный ответ"
    assert primary.chat.completions.calls[0]["model"] == "primary-model"
    assert fallback.chat.completions.calls[0]["model"] == "fallback-model"
    assert fallback.chat.completions.calls[0]["messages"] == primary.chat.completions.calls[0]["messages"]


def test_fallback_is_used_for_any_primary_error():
    agent, _, fallback = make_agent([ValueError("bad request")], ["Резервный ответ"])

    assert agent.generate_answer("вопрос") == "Резервный ответ"
    assert len(fallback.chat.completions.calls) == 1


def test_reminder_and_summary_use_fallback():
    reminder_agent, _, reminder_fallback = make_agent([RuntimeError("down")], ["Напоминание"])
    summary_agent, _, summary_fallback = make_agent([RuntimeError("down")], ["Сводка"])

    assert reminder_agent.compose_reminder("история") == "Напоминание"
    assert summary_agent.summarize_ticket("история") == "Сводка"
    assert len(reminder_fallback.chat.completions.calls) == 1
    assert len(summary_fallback.chat.completions.calls) == 1
