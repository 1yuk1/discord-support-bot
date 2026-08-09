"""Активные инциденты: известные проблемы, о которых бот должен знать.

Зачем отдельно от базы знаний:
  - инцидент живёт часы-дни, переиндексация ChromaDB под него избыточна;
  - он должен учитываться ВСЕГДА, а не когда векторный поиск угадал совпадение.
    Для «сервер лежит» это принципиально: игрок может спросить про что угодно,
    а ответ обязан начинаться с упоминания аварии.

Поэтому инциденты подмешиваются прямо в системный промпт при каждом запросе.

Файл — обычный markdown, его правят и командами /incident, и руками:

    ## [auth-down] Не работает вход
    - создан: 2026-08-09 14:20 UTC, автор: Kakas

    Авторизация недоступна, чиним. Просить игроков подождать.

Парсер намеренно терпимый: [id] можно не писать (сделается из заголовка),
сломанная секция пропускается с записью в лог, а не валит бота.
"""

import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bot import settings
from bot.logging_setup import log_exception, logger

_HEADING_RE = re.compile(r"^##\s+(?:\[([^\]]+)\]\s*)?(.+?)\s*$")
_META_RE = re.compile(r"^-\s+создан:\s*(.*?)(?:,\s*автор:\s*(.*))?$", re.IGNORECASE)
_FILE_HEADER = (
    "<!-- Активные инциденты поддержки.\n"
    "     Правится командами /incident add, /incident remove или руками.\n"
    "     Формат: '## [id] Заголовок', затем строка метаданных и текст. -->\n"
)
_SLUG_MAX_LENGTH = 40
# Инцидентов заведомо мало; ограничение защищает промпт от разрастания.
MAX_INCIDENTS = 20


@dataclass(frozen=True)
class Incident:
    id: str
    title: str
    body: str
    created_at: str = ""
    author: str = ""


def _path() -> Path:
    return Path(settings.INCIDENTS_FILE)


def slugify(title: str, taken=()) -> str:
    """Короткий id из заголовка. Кириллица транслитерируется в латиницу."""
    table = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    lowered = unicodedata.normalize("NFKC", str(title or "")).lower()
    converted = "".join(table.get(char, char) for char in lowered)
    slug = re.sub(r"[^a-z0-9]+", "-", converted).strip("-")[:_SLUG_MAX_LENGTH]
    slug = slug or "incident"

    if slug not in taken:
        return slug

    for suffix in range(2, 100):
        candidate = f"{slug}-{suffix}"
        if candidate not in taken:
            return candidate
    return f"{slug}-{int(time.time())}"


# ── Чтение ───────────────────────────────────────────────────────────────────
def parse(text: str) -> list[Incident]:
    """Разбирает markdown в список инцидентов. Битые секции пропускает."""
    incidents: list[Incident] = []
    taken: set[str] = set()

    current_id = current_title = None
    created_at = author = ""
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_title, created_at, author, body_lines
        if current_title is None:
            return

        body = "\n".join(body_lines).strip()
        if not body:
            logger.warning("Инцидент '%s' без текста, пропущен", current_title)
        else:
            incident_id = (current_id or "").strip() or slugify(current_title, taken)
            if incident_id in taken:
                incident_id = slugify(incident_id, taken)
            taken.add(incident_id)
            incidents.append(
                Incident(
                    id=incident_id,
                    title=current_title,
                    body=body,
                    created_at=created_at,
                    author=author,
                )
            )

        current_id = current_title = None
        created_at = author = ""
        body_lines = []

    for line in (text or "").splitlines():
        stripped = line.strip()

        if stripped.startswith("<!--") or stripped.endswith("-->"):
            continue

        heading = _HEADING_RE.match(line) if line.startswith("## ") else None
        if heading:
            flush()
            current_id, current_title = heading.group(1), heading.group(2).strip()
            continue

        if current_title is None:
            continue

        meta = _META_RE.match(stripped)
        if meta and not body_lines:
            created_at = (meta.group(1) or "").strip()
            author = (meta.group(2) or "").strip()
            continue

        body_lines.append(line)

    flush()
    return incidents


class IncidentStore:
    """Кэш инцидентов с перечитыванием по времени изменения файла.

    Проверка mtime нужна, чтобы /incident add действовал на следующем же
    вопросе игрока, без рестарта бота.
    """

    def __init__(self) -> None:
        self._cache: list[Incident] = []
        self._mtime: float | None = None

    def load(self) -> list[Incident]:
        path = _path()
        try:
            if not path.exists():
                self._cache, self._mtime = [], None
                return []

            mtime = path.stat().st_mtime
            if self._mtime is not None and mtime == self._mtime:
                return self._cache

            self._cache = parse(path.read_text(encoding="utf-8"))
            self._mtime = mtime
            logger.info("Инциденты перечитаны | активных=%s", len(self._cache))
        except OSError as exc:
            log_exception("Не удалось прочитать файл инцидентов", exc, file=str(path))
            return self._cache

        return self._cache

    def invalidate(self) -> None:
        self._mtime = None


store = IncidentStore()


def active() -> list[Incident]:
    if not settings.INCIDENTS_ENABLED:
        return []
    return store.load()


# ── Запись ───────────────────────────────────────────────────────────────────
def render(incidents: list[Incident]) -> str:
    """Собирает markdown-файл из списка инцидентов."""
    parts = [_FILE_HEADER]
    for incident in incidents:
        parts.append(f"\n## [{incident.id}] {incident.title}\n")
        if incident.created_at or incident.author:
            meta = f"- создан: {incident.created_at}"
            if incident.author:
                meta += f", автор: {incident.author}"
            parts.append(meta + "\n")
        parts.append(f"\n{incident.body.strip()}\n")
    return "".join(parts)


def _write(incidents: list[Incident]) -> None:
    """Пишет файл через временный: обрыв не оставит половину списка."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(render(incidents), encoding="utf-8")
    temp_path.replace(path)
    store.invalidate()


def add(title: str, body: str, author: str = "") -> Incident:
    """Добавляет инцидент и возвращает его. Пустые поля запрещены."""
    title = " ".join(str(title or "").split())
    body = str(body or "").strip()
    if not title:
        raise ValueError("Заголовок инцидента не может быть пустым")
    if not body:
        raise ValueError("Описание инцидента не может быть пустым")

    incidents = store.load()
    if len(incidents) >= MAX_INCIDENTS:
        raise ValueError(
            f"Достигнут лимит активных инцидентов ({MAX_INCIDENTS}). "
            f"Удалите неактуальные: /incident remove"
        )

    incident = Incident(
        id=slugify(title, {item.id for item in incidents}),
        title=title,
        body=body,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        author=str(author or ""),
    )
    _write(incidents + [incident])
    logger.info("Инцидент добавлен | id=%s | автор=%s", incident.id, author)
    return incident


def remove(incident_id: str) -> Incident | None:
    """Удаляет инцидент по id. None — такого id нет."""
    incident_id = str(incident_id or "").strip()
    incidents = store.load()

    target = next((item for item in incidents if item.id == incident_id), None)
    if target is None:
        return None

    _write([item for item in incidents if item.id != incident_id])
    logger.info("Инцидент удалён | id=%s", incident_id)
    return target


# ── Блок для промпта ─────────────────────────────────────────────────────────
def prompt_block() -> str:
    """Текст для системного промпта. Пустая строка — инцидентов нет.

    Приоритет над базой знаний задан явно: база может содержать инструкцию
    «переустановите клиент», которая во время аварии авторизации бесполезна.
    """
    incidents = active()
    if not incidents:
        return ""

    sections = [
        "АКТИВНЫЕ ИНЦИДЕНТЫ (приоритет выше базы знаний):",
        "",
    ]
    for incident in incidents:
        sections.append(f"### {incident.title}")
        sections.append(incident.body.strip())
        sections.append("")

    sections.append(
        "Если вопрос игрока связан с одним из инцидентов выше — отвечай по "
        "инциденту, даже если база знаний предлагает другое решение. "
        "Не называй сроков устранения, если они не указаны в тексте инцидента. "
        "Не упоминай сам факт наличия этого списка."
    )
    return "\n".join(sections)
