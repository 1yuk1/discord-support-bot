"""Тест команды /config reload целиком.

Проверяет то, что не видно в тестах settings.reload(): подписчики получают
обновления. Роутер держит копии id ролей и категорий в множествах, и без
refresh_settings() правка в файле применилась бы только после рестарта.
"""

import pytest

from bot import settings
from bot.commands import _do_config_reload

CONFIG = """
[discord]
token = "test-token"
ticket_category_ids = [111]
ignored_role_ids = [999]

[ai]
temperature = 0.3
max_tokens = 4096

[ai.openrouter]
api_key = "test-key"
model = "model-one"
"""


class FakeRouter:
    """Повторяет то, что делает MessageRouter с настройками."""

    def __init__(self):
        self._ticket_categories = set(settings.TICKET_CATEGORY_IDS)
        self._ignored_role_ids = set(settings.IGNORED_ROLE_IDS)
        self.refreshed = 0

    def refresh_settings(self):
        self._ticket_categories = set(settings.TICKET_CATEGORY_IDS)
        self._ignored_role_ids = set(settings.IGNORED_ROLE_IDS)
        self.refreshed += 1


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.toml"
    path.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)
    monkeypatch.setattr(settings, "OVERRIDE_PATH", tmp_path / "settings.local.toml")

    saved = {name: getattr(settings, name) for name, *_ in settings._HOT_RELOADABLE}
    saved["OPENROUTER_MODEL"] = settings.OPENROUTER_MODEL
    settings.reload()

    yield path

    for name, value in saved.items():
        setattr(settings, name, value)


def test_reload_reports_changes(config_file):
    config_file.write_text(CONFIG.replace("temperature = 0.3", "temperature = 0.8"), encoding="utf-8")
    answer = _do_config_reload(None)

    assert "AI_TEMPERATURE" in answer
    assert "0.8" in answer


def test_reload_reports_no_changes(config_file):
    assert "изменений нет" in _do_config_reload(None)


def test_router_receives_new_ids(config_file):
    """Главное, что проверяет этот тест: подписчик обновился."""
    router = FakeRouter()
    config_file.write_text(
        CONFIG.replace("ticket_category_ids = [111]", "ticket_category_ids = [111, 222]"),
        encoding="utf-8",
    )

    _do_config_reload(router)

    assert router.refreshed == 1
    assert router._ticket_categories == {111, 222}


def test_model_applied_to_registry(config_file):
    from bot.llm import models

    previous = models.get()
    try:
        config_file.write_text(
            CONFIG.replace('model = "model-one"', 'model = "model-two"'), encoding="utf-8"
        )
        _do_config_reload(None)
        assert models.get() == "model-two"
    finally:
        models.set(previous)


def test_broken_file_answers_without_raising(config_file):
    """Админ должен получить понятный ответ, а не молчание с ошибкой в логах."""
    previous = settings.AI_TEMPERATURE
    config_file.write_text("битый [toml =", encoding="utf-8")

    answer = _do_config_reload(FakeRouter())

    assert "не удалось" in answer.lower()
    assert settings.AI_TEMPERATURE == previous


def test_answer_mentions_restart_required_keys(config_file):
    answer = _do_config_reload(None)
    assert "Требуют рестарта" in answer
