import tomllib
import os
import sys
from pathlib import Path

BASE_DIR = Path(os.environ.get("APP_BASE_DIR", os.getcwd()))
SETTINGS_PATH = BASE_DIR / "settings.toml"


def _load_config() -> dict:
    if not SETTINGS_PATH.exists():
        print(f"❌ Файл настроек не найден: {SETTINGS_PATH}")
        print("   Скопируйте settings.toml.example -> settings.toml и заполните значения.")
        sys.exit(1)

    with open(SETTINGS_PATH, "rb") as f:
        return tomllib.load(f)


_cfg = _load_config()


def _parse_id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value] if value != 0 else []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, int) and item != 0:
                result.append(item)
            elif isinstance(item, str):
                stripped = item.strip()
                if stripped and stripped != "0":
                    result.append(int(stripped))
        return result
    if isinstance(value, str):
        result = []
        for item in value.split(","):
            stripped = item.strip()
            if stripped and stripped != "0":
                result.append(int(stripped))
        return result
    return []

# ── Discord ──────────────────────────────────────────────────────────────────
DISCORD_TOKEN: str = _cfg["discord"]["token"]
TICKET_CATEGORY_ID: int | None = _cfg["discord"].get("ticket_category_id")
TICKET_CATEGORY_IDS: list[int] = _parse_id_list(
    _cfg["discord"].get("ticket_category_ids", TICKET_CATEGORY_ID)
)
BOT_ROLE_ID: int | None = _cfg["discord"].get("bot_role_id")
BOT_ROLE_IDS: list[int] = _parse_id_list(
    _cfg["discord"].get("bot_role_ids", BOT_ROLE_ID)
)
IGNORED_ROLE_IDS: list[int] = _parse_id_list(_cfg["discord"].get("ignored_role_ids"))
DEFAULT_REPLY_FOOTER = (
    "Этот бот только обучается, поэтому может неправильно отвечать на вопросы. "
    "Чтобы позвать человека, напишите: позови человека"
)
BOT_REPLY_FOOTER: str = _cfg["discord"].get("reply_footer", DEFAULT_REPLY_FOOTER)

# ── AI ───────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = _cfg["ai"].get("embedding_model", "BAAI/bge-m3")
EMBEDDING_MODEL_TYPE: str = _cfg["ai"].get("embedding_model_type", "bge")
SEARCH_TOP_K: int = _cfg["ai"].get("search_top_k", 2)
AI_REQUEST_TIMEOUT_SECONDS: int = _cfg["ai"].get("request_timeout_seconds", 90)
AI_MAX_CONCURRENT_REQUESTS: int = _cfg["ai"].get("max_concurrent_requests", 2)

OPENROUTER_API_KEY: str = _cfg["ai"].get("openrouter", {}).get("api_key", "")
OPENROUTER_MODEL: str = _cfg["ai"].get("openrouter", {}).get("model", "")
OPENROUTER_API_URL: str = _cfg["ai"].get("openrouter", {}).get(
    "api_url", "https://openrouter.ai/api/v1"
)
OPENROUTER_SITE_URL: str = _cfg["ai"].get("openrouter", {}).get("site_url", "")
OPENROUTER_APP_NAME: str = _cfg["ai"].get("openrouter", {}).get("app_name", "SinusSMP Support Bot")

# ── Proxy ────────────────────────────────────────────────────────────────────
USE_PROXY: bool = _cfg["proxy"]["enabled"]
PROXY_HOST: str = _cfg["proxy"]["host"]
PROXY_PORT: int = _cfg["proxy"]["port"]
PROXY_USERNAME: str = _cfg["proxy"].get("username", "")
PROXY_PASSWORD: str = _cfg["proxy"].get("password", "")

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_CACHE_PATH: str = str(BASE_DIR / _cfg["paths"]["model_cache"])
DB_PATH: str = str(BASE_DIR / _cfg["paths"]["database"])
LOGS_PATH: str = str(BASE_DIR / _cfg["paths"]["logs"])
AUTO_UPDATE_CHROMA_DB: bool = _cfg["paths"].get("auto_update_chroma_db", True)

# ── Ticket logs ──────────────────────────────────────────────────────────────
_logs_cfg = _cfg.get("logs", {})
LOG_TICKET_FILENAME_TEMPLATE: str = _logs_cfg.get(
    "ticket_filename_template",
    "{channel_name}-{created_at}",
)
LOG_TICKET_DATETIME_FORMAT: str = _logs_cfg.get("ticket_datetime_format", "%d.%m-%H-%M")
LOG_TICKET_FILE_EXTENSION: str = _logs_cfg.get("ticket_file_extension", "json").lstrip(".")
LOG_ARCHIVE_ENABLED: bool = _logs_cfg.get("archive_enabled", True)
LOG_ARCHIVE_DIR: str = str(BASE_DIR / _logs_cfg.get("archive_dir", "logs/archives"))
LOG_ARCHIVE_INTERVAL_HOURS: int = _logs_cfg.get("archive_interval_hours", 24)
LOG_ARCHIVE_AFTER_HOURS: int = _logs_cfg.get("archive_after_hours", 24)
LOG_ARCHIVE_DATE_FORMAT: str = _logs_cfg.get("archive_date_format", "%d.%m")
LOG_ARCHIVE_FILENAME_TEMPLATE: str = _logs_cfg.get(
    "archive_filename_template",
    "tickets-{date}-{count}tickets.zip",
)

# ── Rate Limit ───────────────────────────────────────────────────────────────
_rate_limit_cfg = _cfg.get("rate_limit", {})
RATE_LIMIT_ENABLED: bool = _rate_limit_cfg.get("enabled", True)
RATE_LIMIT: int = _rate_limit_cfg.get("global_limit", 30)
RATE_WINDOW: int = _rate_limit_cfg.get("global_window", 60)
CHANNEL_COOLDOWN: int = _rate_limit_cfg.get("channel_cooldown", 5)
DUPLICATE_CHECK_TIME: int = _rate_limit_cfg.get("duplicate_check_time", 5)
USER_MESSAGE_LIMIT: int = _rate_limit_cfg.get("user_message_limit", 3)
USER_MESSAGE_WINDOW: int = _rate_limit_cfg.get("user_message_window", 10)
MAX_HISTORY: int = _rate_limit_cfg.get("max_history", 6)

# ── Server facts ─────────────────────────────────────────────────────────────
_server_cfg = _cfg.get("server", {})
SERVER_MIN_VERSION: str = _server_cfg.get("min_version", "1.19.4")
SERVER_MAX_VERSION: str = _server_cfg.get("max_version", "1.21.10")
SERVER_RECOMMENDED_VERSION: str = _server_cfg.get("recommended_version", "1.21.10")
SERVER_SUPPORTED_VERSIONS: str = _server_cfg.get(
    "supported_versions",
    f"{SERVER_MIN_VERSION}, {SERVER_MAX_VERSION}",
)

# ── Runtime state ────────────────────────────────────────────────────────────
_state_cfg = _cfg.get("state", {})
STATE_SNAPSHOT_FILE: str = str(
    BASE_DIR / _state_cfg.get("snapshot_file", "logs/conversation_state.json")
)
STATE_SAVE_INTERVAL_SECONDS: int = _state_cfg.get("save_interval_seconds", 30)
STATE_TTL_SECONDS: int = _state_cfg.get("ttl_seconds", 7 * 24 * 60 * 60)

# ── Transfer ─────────────────────────────────────────────────────────────────
HUMAN_TRANSFER_PHRASES: list[str] = _cfg["transfer"]["phrases"]
