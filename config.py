import tomllib
import os
import sys
from pathlib import Path

BASE_DIR = Path(os.environ.get("APP_BASE_DIR", Path(__file__).resolve().parent))
SETTINGS_PATH = BASE_DIR / "settings.toml"


def _load_config() -> dict:
    if not SETTINGS_PATH.exists():
        print(f"❌ Файл настроек не найден: {SETTINGS_PATH}")
        print("   Скопируйте settings.toml.example -> settings.toml и заполните значения.")
        sys.exit(1)

    with open(SETTINGS_PATH, "rb") as f:
        return tomllib.load(f)


_cfg = _load_config()

# ── Discord ──────────────────────────────────────────────────────────────────
DISCORD_TOKEN: str = _cfg["discord"]["token"]
TICKET_CATEGORY_ID: int | None = _cfg["discord"].get("ticket_category_id")
BOT_ROLE_ID: int | None = _cfg["discord"].get("bot_role_id")

# ── AI ───────────────────────────────────────────────────────────────────────
AI_PROVIDER: str = _cfg["ai"]["provider"]

GROQ_API_KEY: str = _cfg["ai"]["groq"]["api_key"]
GROQ_MODEL: str = _cfg["ai"]["groq"]["model"]

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

# ── Rate Limit ───────────────────────────────────────────────────────────────
RATE_LIMIT: int = _cfg["rate_limit"]["global_limit"]
RATE_WINDOW: int = _cfg["rate_limit"]["global_window"]
CHANNEL_COOLDOWN: int = _cfg["rate_limit"]["channel_cooldown"]
DUPLICATE_CHECK_TIME: int = _cfg["rate_limit"]["duplicate_check_time"]
MAX_HISTORY: int = _cfg["rate_limit"]["max_history"]

# ── Transfer ─────────────────────────────────────────────────────────────────
HUMAN_TRANSFER_PHRASES: list[str] = _cfg["transfer"]["phrases"]
