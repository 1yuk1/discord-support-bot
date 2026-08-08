"""Текстовые утилиты: раскладка клавиатуры, нормализация, разбивка сообщений.

Здесь только чистые функции без зависимостей от Discord и AI — их удобно
тестировать и переиспользовать.
"""

import re

# Конвертация раскладки QWERTY ↔ ЙЦУКЕН (для опечаток вида "ghbdtn" → "привет").
LAYOUT_EN_TO_RU = str.maketrans(
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./`QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?~",
    "йцукенгшщзхъфывапролджэячсмитьбю.ёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,Ё",
)
LAYOUT_RU_TO_EN = str.maketrans(
    "йцукенгшщзхъфывапролджэячсмитьбю.ёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,Ё",
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./`QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?~",
)

# Латинские термины проекта, которые не нужно «переводить» из раскладки.
_SERVER_TERMS = frozenset({
    "lite1", "lite2", "lite3", "prac", "warp", "sinussmp",
    "boosty", "easydonate", "minecraft", "discord", "vpn",
    "vip", "mvp", "elite", "unity", "wizard", "pro", "mystic",
    "play.sinussmp.ru", "play.sinussmp.com",
})

# Символы, которые в русской раскладке стоят на месте ;'[] — сильный сигнал.
_LAYOUT_SPECIAL_CHARS = frozenset(";'[]")

# Типичные английские паттерны: если они есть, текст скорее английский.
_ENGLISH_MARKERS = (
    "th", "ng", "ing", "sh", "wh", "ee", "oo", "ay",
    "ou", "er", "ed ", "ly", " the", " is", " are",
    " you", " how", " what", "ello", "orld",
)

_RU_VOWELS = frozenset("аеёиоуыэюя")

# Discord-разметка: <@123>, <@!123>, <@&123>, <#123>, <a:name:123>, <:name:123>
NOISE_RE = re.compile(r"<(?:@[!&]?|#)\d+>|<a?:\w+:\d+>")
_INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def looks_like_wrong_layout(text: str) -> bool:
    """Похож ли текст на русскую фразу, набранную в EN-раскладке.

    Срабатывает, если выполнены ВСЕ условия:
      1) в строке нет ни одной кириллической буквы;
      2) есть минимум 3 латинских буквы;
      3) есть «подозрительный» спецсимвол (;'[]) ИЛИ после конвертации
         получается строка с заметной долей русских гласных и без типичных
         английских диграфов.

    Это отсекает обычные английские слова ("hello", "how are you") и бренды
    ("boosty"), но ловит реальные опечатки раскладки ("ghbdtn" = "привет").
    """
    if not text or len(text.strip()) < 3:
        return False

    sample = text.lower()
    en_letters = sum(1 for ch in sample if "a" <= ch <= "z")
    ru_letters = sum(1 for ch in sample if "а" <= ch <= "я" or ch == "ё")

    if ru_letters > 0 or en_letters < 3:
        return False

    tokens = {token for token in sample.replace(",", " ").replace(".", " ").split() if token}
    if tokens and tokens.issubset(_SERVER_TERMS):
        return False

    if any(ch in _LAYOUT_SPECIAL_CHARS for ch in sample):
        return True

    if any(marker in sample for marker in _ENGLISH_MARKERS):
        return False

    translated = sample.translate(LAYOUT_EN_TO_RU)
    ru_vowels = sum(1 for ch in translated if ch in _RU_VOWELS)
    ru_consonants = sum(
        1 for ch in translated if "а" <= ch <= "я" and ch not in _RU_VOWELS
    )

    if ru_vowels == 0:
        return False

    # В осмысленном русском на 2-3 согласные приходится примерно одна гласная.
    if ru_consonants > 0 and ru_vowels / max(ru_consonants, 1) < 0.15:
        return False

    return True


def query_variants(text: str) -> list[str]:
    """Варианты текста для поиска в базе знаний.

    Всегда содержит оригинал, плюс вариант со сменой раскладки, если текст
    похож на набранный не в той раскладке. Нужно ТОЛЬКО для retrieval —
    в LLM всегда уходит оригинал.
    """
    variants = [text]
    if looks_like_wrong_layout(text):
        converted = text.translate(LAYOUT_EN_TO_RU)
        if converted != text:
            variants.append(converted)
    return variants


def strip_noise(text: str) -> str:
    """Убирает Discord-разметку (упоминания, эмодзи) и схлопывает пробелы."""
    if not text:
        return ""
    return " ".join(NOISE_RE.sub(" ", text).split()).strip()


def normalize_for_dedup(text: str) -> str:
    """Нормализация для дедупа: без разметки, в нижнем регистре, без лишних пробелов.

    Системный тикет-бот часто шлёт одно и то же сообщение с разницей в пинге
    или пробелах; обычное сравнение строк это не ловит.
    """
    if not text:
        return ""
    return " ".join(NOISE_RE.sub(" ", text).lower().split())


def sanitize_filename_part(value, fallback: str = "ticket") -> str:
    """Делает из произвольной строки безопасное имя файла."""
    cleaned = _INVALID_FILENAME_CHARS_RE.sub("-", str(value or ""))
    cleaned = " ".join(cleaned.split()).strip(" .-_")
    return cleaned[:120] or fallback


def split_discord_text(text: str, limit: int = 2000) -> list[str]:
    """Разбивает текст на части в пределах лимита Discord.

    Режет по переводу строки, затем по пробелу, и только в крайнем случае —
    по символу, чтобы не рвать слова.
    """
    if limit <= 0:
        return [text]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit

        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def apply_templates(value, template_vars: dict[str, str]):
    """Рекурсивно подставляет {ПЕРЕМЕННЫЕ} в строках, списках и словарях."""
    if isinstance(value, str):
        for name, replacement in template_vars.items():
            value = value.replace("{" + name + "}", str(replacement))
        return value
    if isinstance(value, list):
        return [apply_templates(item, template_vars) for item in value]
    if isinstance(value, dict):
        return {key: apply_templates(item, template_vars) for key, item in value.items()}
    return value


def find_unknown_placeholders(text: str, known: dict[str, str]) -> list[str]:
    """Возвращает {ПЛЕЙСХОЛДЕРЫ}, для которых нет подстановки.

    Без этой проверки опечатка вида {SERVER_TYPO} молча уезжает игроку.
    """
    return [
        name
        for name in re.findall(r"\{([A-Z][A-Z0-9_]*)\}", text)
        if name not in known
    ]
