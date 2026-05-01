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
IGNORED_ROLE_IDS: list[int] = _parse_id_list(_cfg["discord"].get("ignored_role_ids"))

# ── AI ───────────────────────────────────────────────────────────────────────
AI_PROVIDER: str = _cfg["ai"]["provider"]
EMBEDDING_MODEL: str = _cfg["ai"].get("embedding_model", "BAAI/bge-m3")
EMBEDDING_MODEL_TYPE: str = _cfg["ai"].get("embedding_model_type", "bge")

GROQ_API_KEY: str = _cfg["ai"]["groq"]["api_key"]
GROQ_MODEL: str = _cfg["ai"]["groq"]["model"]

OPENROUTER_API_KEY: str = _cfg["ai"].get("openrouter", {}).get("api_key", "")
OPENROUTER_MODEL: str = _cfg["ai"].get("openrouter", {}).get("model", "")
OPENROUTER_API_URL: str = _cfg["ai"].get("openrouter", {}).get(
    "api_url", "https://openrouter.ai/api/v1"
)
OPENROUTER_SITE_URL: str = _cfg["ai"].get("openrouter", {}).get("site_url", "")
OPENROUTER_APP_NAME: str = _cfg["ai"].get("openrouter", {}).get("app_name", "SinusSMP Support Bot")

LOCAL_API_URL: str = _cfg["ai"]["local"]["api_url"]
LOCAL_API_KEY: str = _cfg["ai"]["local"]["api_key"]
LOCAL_MODEL: str = _cfg["ai"]["local"]["model"]

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

# ── Transfer ─────────────────────────────────────────────────────────────────
HUMAN_TRANSFER_PHRASES: list[str] = _cfg["transfer"]["phrases"]
