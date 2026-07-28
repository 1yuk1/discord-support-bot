from escalation import (
    is_llm_human_transfer,
    is_user_human_transfer,
    should_force_human_transfer,
)


def test_regular_donate_cases_do_not_force_transfer():
    messages = [
        "задонатил через бусти, донат не пришел",
        "когда вернут донат через boosty?",
        "не пришла покупка через сайт",
        "передам ник и скрин оплаты",
        "пропал ли коннект из-за версии?",
    ]

    for message in messages:
        assert not should_force_human_transfer(message)
        assert not is_user_human_transfer(message)


def test_serious_complaints_force_transfer():
    messages = [
        "меня взломали",
        "украли вещи",
        "обжалую бан",
        "хочу разбан",
        "сетнули уровень",
    ]

    for message in messages:
        assert should_force_human_transfer(message)


def test_explicit_user_transfer_request():
    assert is_user_human_transfer("позови человека пожалуйста")
    assert is_user_human_transfer("переведи на админа")
    assert not is_user_human_transfer("администрация уже ответила")
    assert not is_user_human_transfer("передам скрин оплаты")


def test_llm_transfer_marker_is_narrow():
    assert is_llm_human_transfer("Я передам ваш тикет старшему специалисту. Пожалуйста, ожидайте.")
    assert not is_llm_human_transfer("Если проблема останется, можно обратиться к специалисту.")
    assert not is_llm_human_transfer("Передам информацию в ответе ниже.")
