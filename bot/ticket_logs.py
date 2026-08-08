"""JSON-логи тикетов: запись, архивация в zip, safety-net для брошенных файлов.

Каждый тикет — отдельный файл logs/active/ticket-<channel_id>.json. При удалении
канала лог переезжает в zip-архив внутри logs/archives/<категория>/.
"""

import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

from bot import settings
from bot.logging_setup import log_exception, logger
from bot.text_utils import sanitize_filename_part

_ACTIVE_LOG_GLOB = "ticket-*.json"


def ensure_directories() -> None:
    for path in (settings.LOGS_PATH, settings.LOG_ACTIVE_DIR, settings.LOG_ARCHIVE_DIR):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log_exception("Не удалось создать каталог логов", exc, path=str(path))


def get_log_path(channel) -> Path:
    channel_id = getattr(channel, "id", "unknown")
    return Path(settings.LOG_ACTIVE_DIR) / f"ticket-{channel_id}.json"


def resolve_category(channel_name: str) -> str:
    """Категория тикета по имени канала, из [logs.categories] в конфиге."""
    name_lower = (channel_name or "").lower()
    for category, patterns in settings.LOG_TICKET_CATEGORIES.items():
        if patterns and any(pattern in name_lower for pattern in patterns):
            return category
    return "other"


def load_log(channel) -> list[dict]:
    path = get_log_path(channel)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        log_exception(
            "Не удалось прочитать лог тикета",
            exc,
            channel_id=getattr(channel, "id", "unknown"),
            file=str(path),
        )
        return []


def save_log(channel, log_data: list[dict]) -> None:
    path = get_log_path(channel)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        log_exception(
            "Не удалось сохранить лог тикета",
            exc,
            channel_id=getattr(channel, "id", "unknown"),
            file=str(path),
        )


def append_entry(
    channel,
    user_id,
    username: str,
    message: str,
    bot_response: str | None = None,
    is_human_transfer: bool = False,
    transfer_reason: str | None = None,
    image_urls: list[str] | None = None,
) -> None:
    """Добавляет запись в лог тикета."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "channel_id": str(getattr(channel, "id", "unknown")),
        "channel_name": getattr(channel, "name", "unknown"),
        "user_id": str(user_id),
        "username": username,
        "message": message,
        "bot_response": bot_response,
        "is_human_transfer": is_human_transfer,
    }
    if transfer_reason:
        entry["transfer_reason"] = transfer_reason
    if image_urls:
        entry["image_urls"] = image_urls

    log_data = load_log(channel)
    log_data.append(entry)
    save_log(channel, log_data)


def _channel_name_from_log(log_path: Path) -> str:
    """Достаёт имя канала из содержимого лога.

    Нужно для safety-net: имя файла содержит только channel_id, а категория
    определяется по имени канала. Раньше все брошенные логи складывались
    в "other" независимо от темы тикета.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""

    if not isinstance(data, list):
        return ""
    for entry in reversed(data):
        if isinstance(entry, dict):
            name = entry.get("channel_name")
            if name and name != "unknown":
                return str(name)
    return ""


def _archive_path_for(category: str, date_key: str) -> Path:
    category_dir = Path(settings.LOG_ARCHIVE_DIR) / sanitize_filename_part(category, "other")
    category_dir.mkdir(parents=True, exist_ok=True)

    name = settings.LOG_ARCHIVE_FILENAME_TEMPLATE.format(date=date_key, count=1)
    name = sanitize_filename_part(name, f"tickets-{date_key}.zip")
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    return category_dir / name


def _move_log_into_archive(log_path: Path, category: str, date_key: str) -> Path | None:
    """Кладёт лог в zip и удаляет исходник. Возвращает путь к архиву."""
    archive_path = _archive_path_for(category, date_key)
    try:
        arcname = log_path.name
        with zipfile.ZipFile(archive_path, "a", zipfile.ZIP_DEFLATED) as archive:
            # Повторное имя создало бы дубликат записи внутри zip.
            if arcname in archive.namelist():
                arcname = f"{log_path.stem}-{int(time.time())}.json"
            archive.write(log_path, arcname=arcname)
        log_path.unlink(missing_ok=True)
        return archive_path
    except (OSError, zipfile.BadZipFile) as exc:
        log_exception(
            "Не удалось заархивировать лог тикета",
            exc,
            file=str(log_path),
            archive=str(archive_path),
        )
        return None


def archive_closed_ticket(channel) -> None:
    """Архивирует лог закрытого тикета. Вызывается при удалении канала."""
    if not settings.LOG_ARCHIVE_ENABLED:
        return

    log_path = get_log_path(channel)
    if not log_path.exists():
        return

    channel_name = getattr(channel, "name", "") or ""
    category = resolve_category(channel_name)
    date_key = datetime.now().strftime(settings.LOG_ARCHIVE_DATE_FORMAT)

    archive_path = _move_log_into_archive(log_path, category, date_key)
    if archive_path:
        logger.info(
            "Тикет заархивирован | channel_id=%s | category=%s | archive=%s",
            getattr(channel, "id", "unknown"),
            category,
            archive_path,
        )


def archive_orphaned_logs() -> None:
    """Архивирует логи без активности дольше archive_safety_net_days."""
    if not settings.LOG_ARCHIVE_ENABLED or settings.LOG_ARCHIVE_SAFETY_NET_DAYS <= 0:
        return

    active_dir = Path(settings.LOG_ACTIVE_DIR)
    if not active_dir.is_dir():
        return

    cutoff = time.time() - settings.LOG_ARCHIVE_SAFETY_NET_DAYS * 24 * 3600
    archived = 0

    for log_path in sorted(active_dir.glob(_ACTIVE_LOG_GLOB)):
        try:
            modified_at = log_path.stat().st_mtime
        except OSError:
            continue
        if modified_at > cutoff:
            continue

        category = resolve_category(_channel_name_from_log(log_path))
        date_key = datetime.fromtimestamp(modified_at).strftime(
            settings.LOG_ARCHIVE_DATE_FORMAT
        )
        if _move_log_into_archive(log_path, category, date_key):
            archived += 1
            logger.info(
                "Safety-net архивирование | file=%s | category=%s",
                log_path.name,
                category,
            )

    if archived:
        logger.info("Заархивировано брошенных логов: %s", archived)
