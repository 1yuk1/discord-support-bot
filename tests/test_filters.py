"""Тесты фильтрации сообщений."""

from types import SimpleNamespace

import pytest

from bot.filters import (
    extract_image_urls,
    extract_message_text,
    is_short_clarification,
    is_ticket_close_notification,
    is_ticket_opening_message,
    is_trivial_text,
    should_use_message_as_question,
)


def make_message(content="", attachments=(), embeds=()):
    return SimpleNamespace(content=content, attachments=list(attachments), embeds=list(embeds))


def make_attachment(url, content_type=None, filename=""):
    return SimpleNamespace(url=url, content_type=content_type, filename=filename)


@pytest.mark.parametrize("text", ["??", "!!!", "+", "60", "200+", ".........", "ау", "хм", "ок"])
def test_trivial_text_detected(text):
    assert is_trivial_text(text)


@pytest.mark.parametrize(
    "text",
    ["не могу зайти на сервер", "донат не пришёл", "как заприватить базу"],
)
def test_meaningful_text_not_trivial(text):
    assert not is_trivial_text(text)


@pytest.mark.parametrize("text", ["да", "нет", "хз", "60", "200+", "что", "ага", ""])
def test_short_clarification_detected(text):
    """На такие реплики поиск в базе даёт мусорный контекст."""
    assert is_short_clarification(text)


@pytest.mark.parametrize(
    "text",
    ["не заходит на сервер", "донат не пришел через бусти", "какая версия нужна"],
)
def test_real_questions_go_to_search(text):
    assert not is_short_clarification(text)


def test_ticket_close_notification():
    assert is_ticket_close_notification("Тикет скоро будет закрыт из-за бездействия")
    assert is_ticket_close_notification("Канал будет удалён через 5 минут")
    assert not is_ticket_close_notification("Я не могу зайти на сервер")


def test_ticket_opening_message():
    assert is_ticket_opening_message("Игрок создал новый тикет")
    assert is_ticket_opening_message("Пользователь создал(а) новый тикет")
    assert not is_ticket_opening_message("создал базу на спавне")


def test_extract_image_urls_by_mime_and_extension():
    message = make_message(
        attachments=[
            make_attachment("https://cdn/1.png", content_type="image/png"),
            make_attachment("https://cdn/2.webp", filename="screen.webp"),
            make_attachment("https://cdn/3.txt", content_type="text/plain", filename="log.txt"),
        ]
    )
    urls = extract_image_urls(message)
    assert urls == ["https://cdn/1.png", "https://cdn/2.webp"]


def test_extract_message_text_strips_noise():
    message = make_message(content="<@123> помогите с донатом <:sad:456>")
    assert extract_message_text(message) == "помогите с донатом"


def test_extract_message_text_reads_embeds():
    embed = SimpleNamespace(
        title="Новый тикет",
        description="Игрок сообщает о проблеме",
        fields=[SimpleNamespace(name="Режим", value="Лайт")],
    )
    message = make_message(embeds=[embed])
    text = extract_message_text(message)
    assert "Новый тикет" in text
    assert "Игрок сообщает о проблеме" in text
    assert "Режим Лайт" in text


def test_should_use_message_as_question():
    assert should_use_message_as_question(make_message(content="не работает вход"))
    assert not should_use_message_as_question(make_message(content="??"))
    assert should_use_message_as_question(
        make_message(attachments=[make_attachment("https://cdn/a.png", content_type="image/png")])
    )
