"""Тесты загрузки промптов."""

import pytest

from bot import settings
from bot.prompt import PromptError, PromptLibrary, load_prompt


def test_system_prompt_loads():
    text = load_prompt("system.md")
    assert "агент службы поддержки" in text
    assert len(text) > 500



def test_summary_prompt_has_transcript_placeholder():
    text = load_prompt("summary.md")
    assert "{TRANSCRIPT}" in text


def test_server_placeholders_substituted():
    """Ссылки и версии не должны оставаться шаблонами в готовом промпте."""
    text = load_prompt("system.md")
    assert "{SERVER_RECOMMENDED_VERSION}" not in text
    assert settings.SERVER_RECOMMENDED_VERSION in text



def test_transfer_wording_matches_escalation_marker():
    """Промпт и детектор эскалации должны говорить одинаково.

    Если формулировка в промпте разойдётся с LLM_TRANSFER_MARKERS, бот будет
    обещать позвать человека, но human_mode не включится.
    """
    from bot.escalation import LLM_TRANSFER_MARKERS

    text = load_prompt("system.md").lower()
    assert any(marker in text for marker in LLM_TRANSFER_MARKERS)


def test_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PROMPTS_DIR", str(tmp_path))
    with pytest.raises(PromptError, match="не найден"):
        load_prompt("system.md")


def test_empty_file_raises(monkeypatch, tmp_path):
    (tmp_path / "system.md").write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(settings, "PROMPTS_DIR", str(tmp_path))
    with pytest.raises(PromptError, match="пуст"):
        load_prompt("system.md")


def test_unknown_placeholder_raises(monkeypatch, tmp_path):
    (tmp_path / "system.md").write_text("Версия {SERVER_TYPO}", encoding="utf-8")
    monkeypatch.setattr(settings, "PROMPTS_DIR", str(tmp_path))
    with pytest.raises(PromptError, match="неизвестные подстановки"):
        load_prompt("system.md")


def test_transcript_placeholder_allowed(monkeypatch, tmp_path):
    (tmp_path / "summary.md").write_text("История:\n{TRANSCRIPT}", encoding="utf-8")
    monkeypatch.setattr(settings, "PROMPTS_DIR", str(tmp_path))
    assert "{TRANSCRIPT}" in load_prompt("summary.md")


def test_library_caches_and_reloads(monkeypatch, tmp_path):
    prompt_file = tmp_path / "system.md"
    prompt_file.write_text("первая версия", encoding="utf-8")
    monkeypatch.setattr(settings, "PROMPTS_DIR", str(tmp_path))

    library = PromptLibrary()
    assert library.get("system.md") == "первая версия"

    prompt_file.write_text("вторая версия", encoding="utf-8")
    assert library.get("system.md") == "первая версия"

    library.reload()
    assert library.get("system.md") == "вторая версия"


def test_reload_keeps_cache_on_error(monkeypatch, tmp_path):
    prompt_file = tmp_path / "system.md"
    prompt_file.write_text("рабочая версия", encoding="utf-8")
    monkeypatch.setattr(settings, "PROMPTS_DIR", str(tmp_path))

    library = PromptLibrary()
    library.get("system.md")

    prompt_file.write_text("сломано {SERVER_TYPO}", encoding="utf-8")
    with pytest.raises(PromptError):
        library.reload()
    assert library.get("system.md") == "рабочая версия"
