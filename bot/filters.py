"""Фильтры входящих сообщений: шум, дубли, системные сообщения тикет-бота.

Все пороги подобраны по реальным инцидентам, комментарии объясняют почему.
"""

import re

from bot.text_utils import strip_noise

# ── Системные сообщения тикет-бота ───────────────────────────────────────────
# Закрытие/неактивность тикета: бот не управляет закрытием и не должен обещать
# «не закроем», поэтому такие сообщения не идут в LLM вообще.
TICKET_CLOSE_MARKERS: tuple[str, ...] = (
    "будет закрыт",
    "тикет закрыт",
    "канал будет удален",
    "канал будет удалён",
    "закрыт из-за бездействия",
    "тикет скоро будет закрыт",
)

# Шапка открытия тикета. Приходит 1-2 раза подряд, отвечаем максимум один раз.
TICKET_OPENING_MARKERS: tuple[str, ...] = (
    "создал новый тикет",
    "создал(а) новый тикет",
)

# ── Тривиальные сообщения ────────────────────────────────────────────────────
# 1-3 «не-буквенных» символа: ??, !!!, +, -
_TRIVIAL_NONWORD_RE = re.compile(r"^[\W_]{1,3}$", re.UNICODE)
# Голые числа до 3 знаков: почти всегда пинг/уровень/таймер, на которые RAG
# цепляет случайный контекст («60 венков», «200+ видов динамита»).
_TRIVIAL_DIGITS_RE = re.compile(r"^\d{1,3}\+?$")
# Мусор любой длины из одних знаков препинания: «????», «.........»
_ONLY_PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# Короткие междометия, на которые бот не должен реагировать.
INTERJECTIONS = frozenset({
    "ау", "ауу", "ауууу", "ауууууу",
    "алле", "але", "алло", "ало",
    "эй", "хм", "ну", "ок", "окей",
    "пупупу", "ааа", "ааааа", "эаа", "эаэа",
})

# Короткие ответы-уточнения без самостоятельного смысла. На них RAG почти всегда
# возвращает мусор: «60» → бот рассказал про «60 венков», «шлемофон» → выдал
# гайд по Plasmo Voice. Для таких реплик поиск отключается, отвечаем по истории.
SHORT_CLARIFICATION_WORDS = frozenset({
    "да", "нет", "не", "ага", "угу", "неа",
    "не знаю", "незнаю", "хз", "хрен знает",
    "ок", "окей", "хорошо",
    "что", "чё", "че", "почему", "зачем",
})

_IMAGE_MIME_PREFIX = "image/"
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

IMAGE_ONLY_PLACEHOLDER = "[Игрок прислал скриншот]"


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def is_ticket_close_notification(text: str) -> bool:
    return _contains_any(text, TICKET_CLOSE_MARKERS)


def is_ticket_opening_message(text: str) -> bool:
    return _contains_any(text, TICKET_OPENING_MARKERS)


def is_trivial_text(text: str) -> bool:
    """True, если в тексте нет смысловой нагрузки для LLM."""
    if not text:
        return True
    if text.lower() in INTERJECTIONS:
        return True
    if _TRIVIAL_NONWORD_RE.match(text):
        return True
    if _TRIVIAL_DIGITS_RE.match(text):
        return True
    if _ONLY_PUNCTUATION_RE.match(text):
        return True
    return False


def is_short_clarification(text: str) -> bool:
    """Короткая реплика, для которой поиск в базе знаний только вредит."""
    if not text:
        return True

    stripped = text.strip()
    if stripped.lower() in SHORT_CLARIFICATION_WORDS:
        return True
    # Голые числа и числа с «+»: ответ на ранее заданный ботом вопрос.
    if re.fullmatch(r"\d{1,4}\+?", stripped):
        return True
    if len(stripped) < 5:
        return True
    meaningful_words = [word for word in re.findall(r"\w+", stripped) if len(word) >= 3]
    return len(meaningful_words) < 2


def extract_image_urls(message) -> list[str]:
    """Ссылки на картинки из вложений сообщения."""
    urls: list[str] = []
    for attachment in getattr(message, "attachments", []) or []:
        url = getattr(attachment, "url", None)
        if not url:
            continue
        content_type = getattr(attachment, "content_type", None) or ""
        filename = (getattr(attachment, "filename", None) or "").lower()
        if content_type.startswith(_IMAGE_MIME_PREFIX) or filename.endswith(_IMAGE_EXTENSIONS):
            urls.append(url)
    return urls


def extract_message_text(message) -> str:
    """Собирает осмысленный текст из сообщения, включая embed-поля."""
    parts: list[str] = []

    content = strip_noise(getattr(message, "content", "") or "")
    if content and not is_trivial_text(content):
        parts.append(content)

    for embed in getattr(message, "embeds", []) or []:
        title = getattr(embed, "title", None)
        if title:
            parts.append(str(title).strip())
        description = getattr(embed, "description", None)
        if description:
            parts.append(str(description).strip())
        for field in getattr(embed, "fields", []) or []:
            field_text = " ".join(
                str(piece) for piece in (getattr(field, "name", None), getattr(field, "value", None)) if piece
            ).strip()
            if field_text:
                parts.append(field_text)

    return "\n".join(part for part in parts if part).strip()


def should_use_message_as_question(message) -> bool:
    """Есть ли в сообщении бота что-то, на что вообще стоит отвечать."""
    content = strip_noise(getattr(message, "content", "") or "")
    if content and not is_trivial_text(content):
        return True
    if getattr(message, "embeds", None):
        return True
    return bool(extract_image_urls(message))


# ── Роли, персонал, администраторы и категории ──────────────────────────────
def is_admin_member(member, admin_role_ids: list[int] = (), admin_user_ids: list[int] = ()) -> bool:
    """Проверяет, является ли пользователь администратором (полный байпасс всех ограничений)."""
    if member is None:
        return False

    # ID пользователя в списке админов
    member_id = getattr(member, "id", None)
    if member_id and member_id in set(admin_user_ids or []):
        return True

    # Права Administrator в Discord
    guild_permissions = getattr(member, "guild_permissions", None)
    if guild_permissions and getattr(guild_permissions, "administrator", False):
        return True

    # Роли администратора
    if admin_role_ids:
        admin_roles_set = set(admin_role_ids)
        if any(getattr(role, "id", None) in admin_roles_set for role in getattr(member, "roles", []) or []):
            return True

    return False


def is_staff_member(
    member,
    staff_role_ids: list[int] = (),
    admin_role_ids: list[int] = (),
    admin_user_ids: list[int] = (),
) -> bool:
    """Проверяет, является ли пользователь персоналом поддержки (или администратором)."""
    if member is None:
        return False
    if is_admin_member(member, admin_role_ids, admin_user_ids):
        return True
    if staff_role_ids:
        staff_roles_set = set(staff_role_ids)
        if any(getattr(role, "id", None) in staff_roles_set for role in getattr(member, "roles", []) or []):
            return True
    return False


def is_ticket_channel_allowed(
    channel,
    allowed_category_ids: list[int] = (),
    excluded_category_ids: list[int] = (),
) -> bool:
    """Проверяет, разрешена ли работа бота в канале/категории (Whitelist и Blacklist)."""
    category_id = getattr(channel, "category_id", None)

    # Blacklist ("везде кроме")
    if excluded_category_ids and category_id is not None and category_id in set(excluded_category_ids):
        return False

    # Whitelist ("нигде кроме")
    if allowed_category_ids:
        return category_id in set(allowed_category_ids)

    return True
