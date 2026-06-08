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
