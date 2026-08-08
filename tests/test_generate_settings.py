"""Тесты генератора settings.toml.

Регрессии, которые ловят эти тесты (все воспроизводились на старом bash-heredoc):
  - USE_PROXY=True давал `enabled = True` вместо `true` → невалидный TOML;
  - PROXY_PORT с нечисловым значением → невалидный TOML;
  - кавычка или бэкслеш в токене рвали строку;
  - `$` в токене подставлялся оболочкой.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "generate_settings.py"

BASE_ENV = {
    "DISCORD_TOKEN": "test-token",
    "OPENROUTER_API_KEY": "test-key",
    "SYSTEMROOT": "C:\\Windows",
    "PATH": str(Path(sys.executable).parent),
}


def generate(tmp_path, **env_overrides) -> dict:
    """Запускает генератор в отдельном процессе и разбирает результат."""
    target = tmp_path / "settings.toml"
    env = {**BASE_ENV, "SETTINGS_PATH": str(target)}
    env.update({key: str(value) for key, value in env_overrides.items()})

    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise AssertionError(f"Генератор упал: {result.stderr}")

    with open(target, "rb") as f:
        return tomllib.load(f)


def test_minimal_config_is_valid(tmp_path):
    config = generate(tmp_path)
    assert config["discord"]["token"] == "test-token"
    assert config["ai"]["openrouter"]["api_key"] == "test-key"
    assert config["proxy"]["enabled"] is False


def run_expecting_failure(tmp_path, **env_overrides):
    target = tmp_path / "settings.toml"
    env = {**BASE_ENV, "SETTINGS_PATH": str(target)}
    env.update({key: str(value) for key, value in env_overrides.items()})

    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 1
    assert not target.exists()
    return result.stderr


def test_missing_token_fails(tmp_path):
    stderr = run_expecting_failure(tmp_path, DISCORD_TOKEN="")
    assert "DISCORD_TOKEN" in stderr


def test_all_missing_variables_reported_at_once(tmp_path):
    """Сообщать по одной неудобно: запуск падал бы на каждой по очереди."""
    stderr = run_expecting_failure(tmp_path, DISCORD_TOKEN="", OPENROUTER_API_KEY="")
    assert "DISCORD_TOKEN" in stderr
    assert "OPENROUTER_API_KEY" in stderr


def test_placeholder_value_rejected(tmp_path):
    stderr = run_expecting_failure(tmp_path, DISCORD_TOKEN="YOUR_DISCORD_TOKEN")
    assert "DISCORD_TOKEN" in stderr


def test_empty_environment_hints_at_egg_import(tmp_path):
    """Частый случай: после импорта egg.json значения переменных сбросились."""
    stderr = run_expecting_failure(tmp_path, DISCORD_TOKEN="", OPENROUTER_API_KEY="")
    assert "egg.json" in stderr


def test_provided_variables_listed_for_diagnostics(tmp_path):
    stderr = run_expecting_failure(
        tmp_path, OPENROUTER_API_KEY="", USE_PROXY="false", OPENROUTER_MODEL="test"
    )
    assert "Панель передала" in stderr
    assert "USE_PROXY" in stderr


def test_secret_values_never_printed(tmp_path):
    """В логи панели не должны попадать значения токенов."""
    secret = "super-secret-token-value"
    stderr = run_expecting_failure(tmp_path, DISCORD_TOKEN=secret, OPENROUTER_API_KEY="")
    assert secret not in stderr


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("1", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("no", False),
        ("0", False),
        ("мусор", False),
    ],
)
def test_boolean_variants(tmp_path, raw, expected):
    config = generate(tmp_path, USE_PROXY=raw)
    assert config["proxy"]["enabled"] is expected


@pytest.mark.parametrize("raw", ["not-a-number", "10808abc", "", "1.5"])
def test_invalid_port_falls_back(tmp_path, raw):
    config = generate(tmp_path, PROXY_PORT=raw)
    assert config["proxy"]["port"] == 10808


def test_valid_port_used(tmp_path):
    config = generate(tmp_path, PROXY_PORT="3128")
    assert config["proxy"]["port"] == 3128


@pytest.mark.parametrize(
    "token",
    [
        'token"with-quote',
        "token\\with-backslash",
        "token$with-dollar",
        "token`with-backtick",
        'mix"$`\\all',
    ],
)
def test_special_characters_in_token_preserved(tmp_path, token):
    config = generate(tmp_path, DISCORD_TOKEN=token)
    assert config["discord"]["token"] == token


def test_id_lists_parsed(tmp_path):
    config = generate(tmp_path, TICKET_CATEGORY_IDS="111, 222, 0, мусор", BOT_ROLE_ID="333")
    assert config["discord"]["ticket_category_ids"] == [111, 222]
    assert config["discord"]["bot_role_ids"] == [333]


def test_id_list_falls_back_to_singular(tmp_path):
    config = generate(tmp_path, TICKET_CATEGORY_ID="555")
    assert config["discord"]["ticket_category_ids"] == [555]


def test_empty_id_list(tmp_path):
    config = generate(tmp_path)
    assert config["discord"]["ticket_category_ids"] == []


def test_dev_log_filename_matches_panel_default(tmp_path):
    """egg.json читает logs/latest.log — имена должны совпадать."""
    config = generate(tmp_path)
    assert config["developer_logs"]["filename"] == "latest.log"


def test_generated_file_stays_valid_with_override_present(tmp_path):
    """Наличие override не должно влиять на валидность сгенерированного файла.

    Слияние делает bot.settings на уровне словарей: склейка текстом невозможна,
    TOML запрещает повторное объявление секции.
    """
    override = tmp_path / "settings.local.toml"
    override.write_text("[ai]\nsearch_top_k = 7\n", encoding="utf-8")
    config = generate(tmp_path, SETTINGS_OVERRIDE_PATH=str(override))
    assert config["ai"]["search_top_k"] == 2


def test_broken_override_does_not_break_generation(tmp_path):
    override = tmp_path / "settings.local.toml"
    override.write_text("это [не] = валидный toml =", encoding="utf-8")
    config = generate(tmp_path, SETTINGS_OVERRIDE_PATH=str(override))
    assert config["discord"]["token"] == "test-token"


def test_regenerates_over_existing_file(tmp_path):
    """Главная эксплуатационная правка: файл перезаписывается каждый запуск."""
    generate(tmp_path, OPENROUTER_MODEL="model-one")
    config = generate(tmp_path, OPENROUTER_MODEL="model-two")
    assert config["ai"]["openrouter"]["model"] == "model-two"
