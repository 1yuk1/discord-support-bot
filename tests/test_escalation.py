"""Тесты логики эскалации на человека.

Эти тесты защищают от расширения списков фраз. Раньше в settings.toml лежал
параллельный список [transfer].phrases с корнями "админ", "передам", "модер" —
если бы бот на него переключился, фразы вида "передам скрин оплаты" или
"администрация уже ответила" глушили бы бота на весь тикет.
"""

import pytest

from bot.escalation import (
    is_llm_human_transfer,
    is_user_human_transfer,
    should_force_human_transfer,
)


@pytest.mark.parametrize(
    "message",
    [
        "задонатил через бусти, донат не пришел",
        "когда вернут донат через boosty?",
        "не пришла покупка через сайт",
        "передам ник и скрин оплаты",
        "пропал ли коннект из-за версии?",
        "администрация уже ответила",
        "я уже передал скрин",
        "модерну написал вчера",
    ],
)
def test_regular_messages_do_not_escalate(message):
    """Обычные обращения не должны уводить тикет к человеку."""
    assert not should_force_human_transfer(message)
    assert not is_user_human_transfer(message)


@pytest.mark.parametrize(
    "message",
    [
        "меня взломали",
        "украли вещи",
        "обжалую бан",
        "хочу разбан",
        "сетнули уровень",
        "забыл пароль от аккаунта",
        "reset password please",
    ],
)
def test_serious_complaints_force_transfer(message):
    """Доступ к аккаунту и наказания бот не разруливает сам."""
    assert should_force_human_transfer(message)


@pytest.mark.parametrize(
    "message",
    [
        "позови человека пожалуйста",
        "переведи на админа",
        "хочу человека",
        "call admin",
        "talk to human",
    ],
)
def test_explicit_request_escalates(message):
    assert is_user_human_transfer(message)


def test_llm_marker_is_narrow():
    """Маркер в ответе LLM должен срабатывать на фразы передачи и тег."""
    assert is_llm_human_transfer(
        "Я передам ваш тикет старшему специалисту. Пожалуйста, ожидайте."
    )
    assert is_llm_human_transfer(
        "I will transfer your ticket to a senior specialist. [TRANSFER_TO_HUMAN]"
    )
    assert is_llm_human_transfer("[TRANSFER_TO_HUMAN]")
    assert is_llm_human_transfer("Ожидайте ответа [TRANSFER_TO_HUM")
    assert not is_llm_human_transfer("Если проблема останется, можно обратиться к специалисту.")
    assert not is_llm_human_transfer("Передам информацию в ответе ниже.")


def test_strip_transfer_tag():
    from bot.escalation import strip_transfer_tag

    raw = "Ваш вопрос принят. [TRANSFER_TO_HUMAN]"
    cleaned = strip_transfer_tag(raw)
    assert cleaned == "Ваш вопрос принят."
    assert "[TRANSFER_TO_HUMAN]" not in cleaned

    english_raw = "Please wait a moment. [TRANSFER_TO_HUMAN]"
    assert strip_transfer_tag(english_raw) == "Please wait a moment."

    # Обрезанные теги из-за лимита токенов
    truncated_raw = "Передал ваш тикет старшему специалисту. [TRANSFER_TO_HUM"
    assert strip_transfer_tag(truncated_raw) == "Передал ваш тикет старшему специалисту."

    unclosed_raw = "Передал ваш тикет. [TRANSFER_TO_HUMAN"
    assert strip_transfer_tag(unclosed_raw) == "Передал ваш тикет."

    short_truncated_raw = "Ожидайте ответа. [TRANSFER"
    assert strip_transfer_tag(short_truncated_raw) == "Ожидайте ответа."

def test_empty_input_is_safe():
    for check in (is_user_human_transfer, is_llm_human_transfer, should_force_human_transfer):
        assert not check("")
        assert not check(None)

