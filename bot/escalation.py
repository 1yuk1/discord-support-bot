"""Определение необходимости передать тикет человеку.

Единственный источник правды по фразам эскалации. Раньше рядом существовал
неиспользуемый список [transfer].phrases в settings.toml — он был удалён,
потому что содержал слишком широкие корни ("админ", "передам", "модер"),
которые ловят обычные фразы вида "передам скрин оплаты" или
"администрация уже ответила" и глушат бота на весь тикет.
"""

# Игрок явно просит человека.
USER_HUMAN_TRANSFER_PHRASES: tuple[str, ...] = (
    "тех поддержка",
    "техподдержка",
    "переведи на человека",
    "позови человека",
    "живой человек",
    "оператор",
    "переведи на админа",
    "позови админа",
    "переведи к админу",
    "соедини с человеком",
    "хочу человека",
    "говорить с человеком",
    "поговорить с человеком",
    "пригласи человека",
    "старший специалист",
    "позови специалиста",
    "переведи на специалиста",
    "передать человеку",
    "передаю тикет",
    # английские фразы
    "call admin",
    "call moderator",
    "call mod",
    "real person",
    "human support",
    "talk to admin",
    "talk to moderator",
    "talk to a person",
    "talk to human",
    "speak to admin",
    "speak to human",
    "speak to a person",
    "contact admin",
    "need admin",
    "want admin",
    "want human",
    "want a real person",
)

TRANSFER_TAG = "[TRANSFER_TO_HUMAN]"

# Маркеры в ответе LLM.
LLM_TRANSFER_MARKERS: tuple[str, ...] = (
    TRANSFER_TAG.lower(),
    "я передам ваш тикет старшему специалисту",
    "передам ваш тикет старшему специалисту",
    "передаю ваш тикет старшему специалисту",
    "передаю тикет старшему специалисту",
    "передам тикет старшему специалисту",
    "i will transfer your ticket",
    "transferring your ticket",
    "transfer your ticket to a senior",
)


def strip_transfer_tag(text: str) -> str:
    """Удаляет служебный маркер [TRANSFER_TO_HUMAN] из текста ответа перед отправкой игроку."""
    if not text:
        return ""
    import re
    cleaned = re.sub(r"\[TRANSFER_TO_HUMAN\]", "", text, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


# Темы, которые бот не должен разруливать сам: доступ к аккаунту, наказания,
# возвраты. Здесь корни слов, а не полные фразы.
FORCE_HUMAN_KEYWORDS: tuple[str, ...] = (
    "взлом",
    "взломал",
    "взломали",
    "украл",
    "украли",
    "обжалов",
    "обжалую",
    "обжалуй",
    "жалоб",
    "жалуюсь",
    "забанил",
    "забанили",
    "разбан",
    "разбана",
    "купил разбан",
    "сетнул",
    "сетнули",
    "размут",
    "размута",
    "замутил",
    "замутили",
    "снять мут",
    "снимите мут",
    "убери мут",
    "убрать мут",
    "сброс пароля",
    "сбросить пароль",
    "забыл пароль",
    "восстановить пароль",
    "reset password",
    "forgot password",
    "recover account",
    "account recovery",
)

TRANSFER_ANSWER = (
    "Я передам ваш тикет старшему специалисту. "
    "Пожалуйста, ожидайте, в ближайшее свободное время вам ответят."
)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrases)


def is_user_human_transfer(text: str) -> bool:
    """Игрок явно просит перевести на человека."""
    return _contains_phrase(text, USER_HUMAN_TRANSFER_PHRASES)


def is_llm_human_transfer(text: str) -> bool:
    """Ответ LLM содержит обещание передать тикет человеку."""
    return _contains_phrase(text, LLM_TRANSFER_MARKERS)


def should_force_human_transfer(text: str) -> bool:
    """Тема требует человека независимо от формулировки игрока."""
    return _contains_phrase(text, FORCE_HUMAN_KEYWORDS)
