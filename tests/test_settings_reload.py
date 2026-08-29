"""Тесты горячей перезагрузки настроек.

Проверяют, что /config reload применяет безопасные правки, не ломает бота на
битом файле и не пытается менять то, что требует рестарта.
"""

import pytest

from bot import settings

BASE_CONFIG = """
[discord]
token = "test-token"
command_prefix = "!"
ticket_category_ids = [111]
ignored_role_ids = [999]

[ai]
temperature = 0.3
max_tokens = 4096

[ai.openrouter]
api_key = "test-key"
model = "model-one"

[rate_limit]
channel_cooldown = 5
message_debounce_seconds = 2.5

[reminders]
enabled = true
idle_hours = 1
ping_role_ids = [900]
"""


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """Подменяет путь к настройкам и восстанавливает значения после теста."""
    path = tmp_path / "settings.toml"
    path.write_text(BASE_CONFIG, encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)
    monkeypatch.setattr(settings, "OVERRIDE_PATH", tmp_path / "settings.local.toml")

    saved = {
        name: getattr(settings, name)
        for name, *_ in settings._HOT_RELOADABLE
    }
    saved["REMINDER_CATEGORY_OVERRIDES"] = settings.REMINDER_CATEGORY_OVERRIDES
    saved["OPENROUTER_MODEL"] = settings.OPENROUTER_MODEL

    # Выравниваем значения по BASE_CONFIG: модуль загружен с настройками из
    # conftest, и без этого первый reload показал бы разницу с ними, а не с
    # правкой, которую проверяет тест.
    settings.reload()

    yield path

    for name, value in saved.items():
        setattr(settings, name, value)


def write(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_reload_applies_numeric_change(settings_file):
    write(settings_file, BASE_CONFIG.replace("temperature = 0.3", "temperature = 0.9"))
    changes = settings.reload()

    assert "AI_TEMPERATURE" in changes
    assert settings.AI_TEMPERATURE == 0.9


def test_reload_applies_id_list_change(settings_file):
    write(
        settings_file,
        BASE_CONFIG.replace("ticket_category_ids = [111]", "ticket_category_ids = [111, 222]"),
    )
    settings.reload()
    assert settings.TICKET_CATEGORY_IDS == [111, 222]


def test_reload_applies_reminder_settings(settings_file):
    write(settings_file, BASE_CONFIG.replace("idle_hours = 1", "idle_hours = 4"))
    settings.reload()
    assert settings.REMINDER_IDLE_HOURS == 4.0


def test_reload_picks_up_category_overrides(settings_file):
    write(
        settings_file,
        BASE_CONFIG + "\n[reminders.categories.555]\nping_role_ids = [777]\n",
    )
    settings.reload()

    assert 555 in settings.REMINDER_CATEGORY_OVERRIDES
    assert settings.reminder_config_for(555)["ping_role_ids"] == [777]


def test_reload_reports_only_changed_keys(settings_file):
    """Показывает только то, что реально изменилось."""
    write(settings_file, BASE_CONFIG.replace("max_tokens = 4096", "max_tokens = 2048"))
    changes = settings.reload()

    assert set(changes) == {"AI_MAX_TOKENS"}
    assert changes["AI_MAX_TOKENS"] == (4096, 2048)


def test_reload_without_changes_returns_empty(settings_file):
    settings.reload()
    assert settings.reload() == {}


def test_reload_detects_model_change(settings_file):
    write(settings_file, BASE_CONFIG.replace('model = "model-one"', 'model = "model-two"'))
    changes = settings.reload()

    assert changes["OPENROUTER_MODEL"] == ("model-one", "model-two")
    assert settings.OPENROUTER_MODEL == "model-two"


def test_broken_file_raises_and_keeps_values(settings_file):
    """Битый конфиг не должен обрушить работающего бота."""
    previous = settings.AI_TEMPERATURE
    write(settings_file, "это [не] = валидный toml =")

    with pytest.raises(ValueError):
        settings.reload()

    assert settings.AI_TEMPERATURE == previous


def test_garbage_value_skipped_others_applied(settings_file):
    """Мусор в одном ключе не отменяет перезагрузку остальных."""
    write(
        settings_file,
        BASE_CONFIG.replace("max_tokens = 1024", 'max_tokens = "много"').replace(
            "temperature = 0.3", "temperature = 0.7"
        ),
    )
    settings.reload()

    assert settings.AI_TEMPERATURE == 0.7


def test_override_file_still_wins(settings_file, tmp_path):
    """settings.local.toml переопределяет основной файл и после перезагрузки."""
    override = tmp_path / "settings.local.toml"
    override.write_text("[ai]\nmax_tokens = 4096\n", encoding="utf-8")

    settings.reload()
    assert settings.AI_MAX_TOKENS == 4096


def test_restart_required_keys_documented():
    """Список нужен, чтобы команда честно сказала, что не применится."""
    keys = settings.RESTART_REQUIRED_KEYS
    assert any("token" in key for key in keys)
    assert any("embedding_model" in key for key in keys)
    assert any("proxy" in key for key in keys)


def test_hot_reload_never_touches_token():
    """Токен в белом списке означал бы попытку сменить его на живом соединении."""
    names = {name for name, *_ in settings._HOT_RELOADABLE}
    for forbidden in ("DISCORD_TOKEN", "EMBEDDING_MODEL", "DB_PATH", "USE_PROXY"):
        assert forbidden not in names
