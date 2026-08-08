"""Загрузка и валидация settings.toml.

Единственный источник правды по настройкам. BASE_DIR считается от расположения
пакета, а не от cwd: иначе при запуске из другого каталога бот и indexer.py
разъезжаются по путям к базе знаний и ChromaDB.
"""

import os
import sys
import tomllib
from pathlib import Path

BASE_DIR = Path(os.environ.get("APP_BASE_DIR") or Path(__file__).resolve().parent.parent)
SETTINGS_PATH = BASE_DIR / "settings.toml"
# Необязательный файл для ручных правок на сервере. settings.toml
# перегенерируется при каждом старте, а этот — нет.
OVERRIDE_PATH = BASE_DIR / "settings.local.toml"


def _fail(message: str) -> "None":
    """Печатает ошибку в stderr и завершает процесс."""
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _read_toml(path: Path, required: bool) -> dict:
    if not path.exists():
        if required:
            _fail(
                f"Файл настроек не найден: {path}\n"
                "   На сервере он создаётся автоматически из переменных окружения.\n"
                "   Локально: заполните settings.toml по образцу из README."
            )
        return {}

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        if required:
            _fail(f"{path.name} содержит синтаксическую ошибку: {exc}")
        print(f"{path.name} содержит ошибку и пропущен: {exc}", file=sys.stderr)
    except OSError as exc:
        if required:
            _fail(f"Не удалось прочитать {path.name}: {exc}")
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивно накладывает override на base, не мутируя аргументы."""
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _load_config() -> dict:
    config = _read_toml(SETTINGS_PATH, required=True)
    override = _read_toml(OVERRIDE_PATH, required=False)
    if override:
        print(f"Применены ручные настройки из {OVERRIDE_PATH.name}")
        return _deep_merge(config, override)
    return config


_cfg = _load_config()


def _section(name: str) -> dict:
    value = _cfg.get(name)
    return value if isinstance(value, dict) else {}


def _require(section: str, key: str):
    value = _section(section).get(key)
    if value in (None, ""):
        _fail(f"settings.toml: обязательный параметр [{section}].{key} не задан")
    return value


def _parse_id_list(value) -> list[int]:
    """Принимает int, список или строку с запятыми. Нули и мусор отбрасывает."""
    if value is None:
        return []

    raw_items: list = []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = value.split(",")

    result: list[int] = []
    for item in raw_items:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            parsed = item
        else:
            stripped = str(item).strip()
            if not stripped:
                continue
            try:
                parsed = int(stripped)
            except ValueError:
                continue
        if parsed != 0 and parsed not in result:
            result.append(parsed)
    return result


def _as_path(raw: str, default: str) -> str:
    """Относительные пути разрешаются от BASE_DIR, абсолютные остаются как есть."""
    candidate = Path(str(raw or default))
    return str(candidate if candidate.is_absolute() else BASE_DIR / candidate)


# ── Discord ──────────────────────────────────────────────────────────────────
_discord_cfg = _section("discord")
DISCORD_TOKEN: str = _require("discord", "token")
COMMAND_PREFIX: str = _discord_cfg.get("command_prefix", "!")
TICKET_CATEGORY_IDS: list[int] = _parse_id_list(
    _discord_cfg.get("ticket_category_ids", _discord_cfg.get("ticket_category_id"))
)
BOT_ROLE_IDS: list[int] = _parse_id_list(
    _discord_cfg.get("bot_role_ids", _discord_cfg.get("bot_role_id"))
)
IGNORED_ROLE_IDS: list[int] = _parse_id_list(_discord_cfg.get("ignored_role_ids"))
BOT_REPLY_FOOTER: str = _discord_cfg.get(
    "reply_footer",
    "Этот бот только обучается, поэтому может неправильно отвечать на вопросы. "
    "Чтобы позвать человека, напишите: позови человека",
)

# ── AI ───────────────────────────────────────────────────────────────────────
_ai_cfg = _section("ai")
EMBEDDING_MODEL: str = _ai_cfg.get("embedding_model", "intfloat/multilingual-e5-large-instruct")
EMBEDDING_MODEL_TYPE: str = _ai_cfg.get("embedding_model_type", "e5-instruct")
SEARCH_TOP_K: int = int(_ai_cfg.get("search_top_k", 2))
AI_REQUEST_TIMEOUT_SECONDS: int = int(_ai_cfg.get("request_timeout_seconds", 90))
AI_MAX_CONCURRENT_REQUESTS: int = int(_ai_cfg.get("max_concurrent_requests", 2))
AI_TEMPERATURE: float = float(_ai_cfg.get("temperature", 0.3))
AI_MAX_TOKENS: int = int(_ai_cfg.get("max_tokens", 1024))
IMAGE_DOWNLOAD_TIMEOUT_SECONDS: int = int(_ai_cfg.get("image_download_timeout_seconds", 30))
IMAGE_MAX_BYTES: int = int(_ai_cfg.get("image_max_bytes", 8 * 1024 * 1024))

_openrouter_cfg = _ai_cfg.get("openrouter")
_openrouter_cfg = _openrouter_cfg if isinstance(_openrouter_cfg, dict) else {}
OPENROUTER_API_KEY: str = _openrouter_cfg.get("api_key", "")
OPENROUTER_MODEL: str = _openrouter_cfg.get("model", "")
OPENROUTER_API_URL: str = _openrouter_cfg.get("api_url", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL: str = _openrouter_cfg.get("site_url", "")
OPENROUTER_APP_NAME: str = _openrouter_cfg.get("app_name", "SinusSMP Support Bot")

# ── Proxy ────────────────────────────────────────────────────────────────────
_proxy_cfg = _section("proxy")
USE_PROXY: bool = bool(_proxy_cfg.get("enabled", False))
PROXY_HOST: str = _proxy_cfg.get("host", "127.0.0.1")
PROXY_PORT: int = int(_proxy_cfg.get("port", 10808))
PROXY_USERNAME: str = _proxy_cfg.get("username", "")
PROXY_PASSWORD: str = _proxy_cfg.get("password", "")

# ── Paths ────────────────────────────────────────────────────────────────────
_paths_cfg = _section("paths")
MODEL_CACHE_PATH: str = _as_path(_paths_cfg.get("model_cache"), "model_cache")
DB_PATH: str = _as_path(_paths_cfg.get("database"), "chroma_db")
LOGS_PATH: str = _as_path(_paths_cfg.get("logs"), "logs")
KNOWLEDGE_DIR: str = _as_path(
    os.environ.get("KNOWLEDGE_DIR") or _paths_cfg.get("knowledge"), "knowledge"
)
PROMPTS_DIR: str = _as_path(_paths_cfg.get("prompts"), "prompts")
AUTO_UPDATE_CHROMA_DB: bool = bool(_paths_cfg.get("auto_update_chroma_db", True))

# ── Knowledge base ───────────────────────────────────────────────────────────
_kb_cfg = _section("knowledge")
CHROMA_COLLECTION_NAME: str = _kb_cfg.get("collection_name", "sinussmp_support")
CHROMA_DISTANCE_METRIC: str = _kb_cfg.get("distance_metric", "cosine")
EMBEDDING_BATCH_SIZE: int = int(_kb_cfg.get("batch_size", 32))
# Совместимость векторов: бот отказывается работать с базой, собранной другой
# моделью. Без этой проверки поиск тихо возвращает мусор.
STRICT_EMBEDDING_CHECK: bool = bool(_kb_cfg.get("strict_embedding_check", True))

# ── Developer logs ───────────────────────────────────────────────────────────
_dev_logs_cfg = _section("developer_logs")
DEV_LOG_FILENAME: str = _dev_logs_cfg.get("filename", "latest.log")
DEV_LOG_LEVEL: str = str(_dev_logs_cfg.get("level", "INFO")).upper()
DEV_LOG_MAX_BYTES: int = int(_dev_logs_cfg.get("max_bytes", 5 * 1024 * 1024))
DEV_LOG_BACKUP_COUNT: int = int(_dev_logs_cfg.get("backup_count", 5))

# ── Ticket logs ──────────────────────────────────────────────────────────────
_logs_cfg = _section("logs")
LOG_ACTIVE_DIR: str = _as_path(_logs_cfg.get("active_dir"), "logs/active")
LOG_ARCHIVE_ENABLED: bool = bool(_logs_cfg.get("archive_enabled", True))
LOG_ARCHIVE_DIR: str = _as_path(_logs_cfg.get("archive_dir"), "logs/archives")
LOG_ARCHIVE_INTERVAL_HOURS: int = int(_logs_cfg.get("archive_interval_hours", 24))
LOG_ARCHIVE_SAFETY_NET_DAYS: int = int(_logs_cfg.get("archive_safety_net_days", 7))
LOG_ARCHIVE_DATE_FORMAT: str = _logs_cfg.get("archive_date_format", "%d.%m")
LOG_ARCHIVE_FILENAME_TEMPLATE: str = _logs_cfg.get(
    "archive_filename_template", "tickets-{date}-{count}tickets.zip"
)

_categories_raw = _logs_cfg.get("categories")
_categories_raw = _categories_raw if isinstance(_categories_raw, dict) else {}
LOG_TICKET_CATEGORIES: dict[str, list[str]] = {
    category: [str(pattern).lower() for pattern in patterns]
    for category, patterns in _categories_raw.items()
    if isinstance(patterns, list)
}

# ── Rate limit ───────────────────────────────────────────────────────────────
_rate_limit_cfg = _section("rate_limit")
RATE_LIMIT_ENABLED: bool = bool(_rate_limit_cfg.get("enabled", True))
RATE_LIMIT: int = int(_rate_limit_cfg.get("global_limit", 30))
RATE_WINDOW: int = int(_rate_limit_cfg.get("global_window", 60))
CHANNEL_COOLDOWN: int = int(_rate_limit_cfg.get("channel_cooldown", 5))
DUPLICATE_CHECK_TIME: int = int(_rate_limit_cfg.get("duplicate_check_time", 5))
USER_MESSAGE_LIMIT: int = int(_rate_limit_cfg.get("user_message_limit", 3))
USER_MESSAGE_WINDOW: int = int(_rate_limit_cfg.get("user_message_window", 10))
MAX_HISTORY: int = int(_rate_limit_cfg.get("max_history", 6))
# Склейка быстрых сообщений в один запрос к AI (0 = выкл).
MESSAGE_DEBOUNCE_SECONDS: float = float(_rate_limit_cfg.get("message_debounce_seconds", 2.5))
PING_SPAM_LIMIT: int = int(_rate_limit_cfg.get("ping_spam_limit", 3))
PING_SPAM_WINDOW: int = int(_rate_limit_cfg.get("ping_spam_window", 300))

# ── Server facts ─────────────────────────────────────────────────────────────
_server_cfg = _section("server")
SERVER_MIN_VERSION: str = _server_cfg.get("min_version", "1.19.4")
SERVER_MAX_VERSION: str = _server_cfg.get("max_version", "1.21.10")
SERVER_RECOMMENDED_VERSION: str = _server_cfg.get("recommended_version", "1.21.10")
SERVER_SUPPORTED_VERSIONS: str = _server_cfg.get(
    "supported_versions", f"{SERVER_MIN_VERSION}, {SERVER_MAX_VERSION}"
)
SERVER_SITE_URL: str = _server_cfg.get("site_url", "https://sinussmp.com")
SERVER_BOOSTY_URL: str = _server_cfg.get("boosty_url", "https://boosty.to/ingrog")

# Подстановки для базы знаний и системного промпта.
TEMPLATE_VARS: dict[str, str] = {
    "SERVER_MIN_VERSION": SERVER_MIN_VERSION,
    "SERVER_MAX_VERSION": SERVER_MAX_VERSION,
    "SERVER_RECOMMENDED_VERSION": SERVER_RECOMMENDED_VERSION,
    "SERVER_SUPPORTED_VERSIONS": SERVER_SUPPORTED_VERSIONS,
    "SERVER_SITE_URL": SERVER_SITE_URL,
    "SERVER_BOOSTY_URL": SERVER_BOOSTY_URL,
}

# ── Runtime state ────────────────────────────────────────────────────────────
_state_cfg = _section("state")
STATE_SNAPSHOT_FILE: str = _as_path(
    _state_cfg.get("snapshot_file"), "logs/conversation_state.json"
)
STATE_SAVE_INTERVAL_SECONDS: int = int(_state_cfg.get("save_interval_seconds", 30))
STATE_TTL_SECONDS: int = int(_state_cfg.get("ttl_seconds", 7 * 24 * 60 * 60))
