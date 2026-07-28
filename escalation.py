USER_HUMAN_TRANSFER_PHRASES = (
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

LLM_TRANSFER_MARKERS = (
    "я передам ваш тикет старшему специалисту",
    "передам ваш тикет старшему специалисту",
)

FORCE_HUMAN_KEYWORDS = (
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


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrases)


def is_user_human_transfer(text: str) -> bool:
    return _contains_phrase(text, USER_HUMAN_TRANSFER_PHRASES)


def is_llm_human_transfer(text: str) -> bool:
    return _contains_phrase(text, LLM_TRANSFER_MARKERS)


def should_force_human_transfer(text: str) -> bool:
    return _contains_phrase(text, FORCE_HUMAN_KEYWORDS)
