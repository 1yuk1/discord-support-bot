"""Тесты текстовых утилит: раскладка, нормализация, разбивка сообщений."""

import pytest

from bot.text_utils import (
    apply_templates,
    find_unknown_placeholders,
    looks_like_wrong_layout,
    normalize_for_dedup,
    query_variants,
    sanitize_filename_part,
    split_discord_text,
    strip_noise,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("crjkmr", "скольк"),
        ("ghbdtn", "привет"),
        ("rfr pfqnb", "как зайти"),
    ],
)
def test_detects_wrong_layout(text, expected):
    assert looks_like_wrong_layout(text)
    assert expected in query_variants(text)


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "how are you",
        "boosty",
        "minecraft",
        "привет",
        "",
        "ab",
        "1.21.10",
    ],
)
def test_ignores_normal_text(text):
    """Английские слова, бренды и русский текст конвертировать не нужно."""
    assert not looks_like_wrong_layout(text)


def test_query_variants_keeps_original_first():
    """Оригинал всегда первый: в LLM уходит именно он, конвертация — только для поиска."""
    variants = query_variants("ghbdtn")
    assert variants[0] == "ghbdtn"
    assert variants[1] == "привет"


def test_query_variants_keeps_single_for_normal_text():
    assert query_variants("как зайти на сервер") == ["как зайти на сервер"]


def test_strip_noise_removes_discord_markup():
    assert strip_noise("<@123456> привет <:smile:789>") == "привет"
    assert strip_noise("<@!111> <#222> текст <@&333>") == "текст"


def test_normalize_for_dedup_ignores_mentions_and_case():
    first = normalize_for_dedup("<@111> Тикет   будет ЗАКРЫТ")
    second = normalize_for_dedup("<@999> тикет будет закрыт")
    assert first == second


def test_split_short_text_stays_whole():
    assert split_discord_text("короткий текст") == ["короткий текст"]


def test_split_respects_limit_and_keeps_words():
    text = " ".join(["слово"] * 800)
    chunks = split_discord_text(text, limit=2000)
    assert len(chunks) > 1
    assert all(len(chunk) <= 2000 for chunk in chunks)
    # Слова не должны рваться посередине.
    assert all(part == "слово" for chunk in chunks for part in chunk.split())


def test_split_prefers_newline_boundary():
    text = "первая строка\n" + "x" * 1990 + "\nпоследняя"
    chunks = split_discord_text(text, limit=2000)
    assert chunks[0] == "первая строка"


def test_split_handles_text_without_separators():
    chunks = split_discord_text("a" * 4500, limit=2000)
    assert [len(chunk) for chunk in chunks] == [2000, 2000, 500]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("донат/оплата", "донат-оплата"),
        ("bug: crash", "bug- crash"),
        ("...", "ticket"),
        ("", "ticket"),
    ],
)
def test_sanitize_filename_part(value, expected):
    assert sanitize_filename_part(value) == expected


def test_apply_templates_walks_nested_structures():
    data = {
        "text": "версия {SERVER_RECOMMENDED_VERSION}",
        "items": ["сайт {SERVER_SITE_URL}"],
        "nested": {"key": "{SERVER_MIN_VERSION}"},
        "number": 42,
    }
    result = apply_templates(
        data,
        {
            "SERVER_RECOMMENDED_VERSION": "1.21.10",
            "SERVER_SITE_URL": "https://example.com",
            "SERVER_MIN_VERSION": "1.19.4",
        },
    )
    assert result["text"] == "версия 1.21.10"
    assert result["items"] == ["сайт https://example.com"]
    assert result["nested"]["key"] == "1.19.4"
    assert result["number"] == 42


def test_find_unknown_placeholders():
    known = {"SERVER_SITE_URL": "https://example.com"}
    assert find_unknown_placeholders("зайдите на {SERVER_SITE_URL}", known) == []
    assert find_unknown_placeholders("версия {SERVER_TYPO}", known) == ["SERVER_TYPO"]
