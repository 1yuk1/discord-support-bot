"""Тесты загрузки настроек: парсинг ID, пути, слияние override."""

import tomllib
from pathlib import Path

import pytest

from bot.settings import _as_path, _deep_merge, _parse_id_list


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        (0, []),
        (123, [123]),
        ("123", [123]),
        ("123,456", [123, 456]),
        ("123, 456 , 0", [123, 456]),
        ([123, 456], [123, 456]),
        (["123", "456"], [123, 456]),
        ([123, 0, 456], [123, 456]),
        ("", []),
        ("0", []),
        ("мусор", []),
        ("123,мусор,456", [123, 456]),
        (True, []),
        ([True, 123], [123]),
        ([123, 123, 456], [123, 456]),
    ],
)
def test_parse_id_list(value, expected):
    assert _parse_id_list(value) == expected


def test_deep_merge_overrides_leaf_values():
    base = {"ai": {"search_top_k": 2, "temperature": 0.3}, "proxy": {"enabled": False}}
    override = {"ai": {"search_top_k": 7}}
    result = _deep_merge(base, override)

    assert result["ai"]["search_top_k"] == 7
    assert result["ai"]["temperature"] == 0.3
    assert result["proxy"]["enabled"] is False


def test_deep_merge_does_not_mutate_inputs():
    base = {"ai": {"search_top_k": 2}}
    override = {"ai": {"search_top_k": 7}}
    _deep_merge(base, override)
    assert base["ai"]["search_top_k"] == 2


def test_deep_merge_adds_new_sections():
    result = _deep_merge({"ai": {}}, {"server": {"site_url": "https://example.com"}})
    assert result["server"]["site_url"] == "https://example.com"


def test_relative_paths_resolve_from_base_dir():
    resolved = Path(_as_path("chroma_db", "chroma_db"))
    assert resolved.is_absolute()
    assert resolved.name == "chroma_db"


def test_absolute_paths_kept_as_is(tmp_path):
    absolute = tmp_path / "custom_db"
    assert _as_path(str(absolute), "chroma_db") == str(absolute)


def test_generated_settings_cover_all_read_sections(tmp_path):
    """Все секции, которые читает bot.settings, должны быть в генераторе.

    Иначе значение живёт на дефолте, и правка переменной в панели ничего не даёт.
    """
    import subprocess
    import sys

    script = Path(__file__).resolve().parent.parent / "scripts" / "generate_settings.py"
    target = tmp_path / "settings.toml"
    subprocess.run(
        [sys.executable, "-X", "utf8", str(script)],
        env={
            "DISCORD_TOKEN": "t",
            "OPENROUTER_API_KEY": "k",
            "SETTINGS_PATH": str(target),
            "SYSTEMROOT": "C:\\Windows",
            "PATH": str(Path(sys.executable).parent),
        },
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    with open(target, "rb") as f:
        generated = tomllib.load(f)

    expected_sections = {
        "discord",
        "ai",
        "proxy",
        "paths",
        "knowledge",
        "developer_logs",
        "logs",
        "rate_limit",
        "server",
        "state",
    }
    assert expected_sections.issubset(generated.keys())
