"""Настройка логирования для разработчиков."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bot import settings

LOGGER_NAME = "discord_support_bot"


def setup_logger() -> logging.Logger:
    """Создаёт логгер с выводом в консоль и в ротируемый файл.

    Имя файла берётся из конфига и должно совпадать с путём в egg.json,
    иначе вкладка логов в панели Pterodactyl останется пустой.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, settings.DEV_LOG_LEVEL, logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logs_dir = Path(settings.LOGS_PATH)
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            logs_dir / settings.DEV_LOG_FILENAME,
            maxBytes=settings.DEV_LOG_MAX_BYTES,
            backupCount=settings.DEV_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # Без файлового лога работать можно, без консольного — нет.
        logger.warning("Не удалось открыть файл логов: %s", exc)

    return logger


logger = setup_logger()


def channel_label(channel) -> str:
    """Форматирует канал для логов: '#ticket-1234 [1526152871538593882]' или 'channel_id'."""
    if channel is None:
        return "unknown"
    if isinstance(channel, (int, str)):
        return str(channel)
    name = getattr(channel, "name", None)
    channel_id = getattr(channel, "id", None)
    if name and channel_id:
        return f"#{name} [{channel_id}]"
    if name:
        return f"#{name}"
    if channel_id:
        return str(channel_id)
    return str(channel)


def log_exception(message: str, exc: BaseException, **context) -> None:
    """Логирует исключение с контекстом и полным traceback."""
    context_text = ""
    if context:
        items = []
        for key, value in context.items():
            if key in ("channel", "channel_id") and not isinstance(value, (int, str)):
                items.append(f"channel={channel_label(value)}")
            else:
                items.append(f"{key}={value}")
        context_text = " | " + " | ".join(items)
    logger.error(
        "%s%s | exception=%s: %s",
        message,
        context_text,
        exc.__class__.__name__,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
