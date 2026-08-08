"""Генерация settings.toml из переменных окружения.

Запускается entrypoint.sh при каждом старте контейнера, поэтому изменения
переменных в панели Pterodactyl применяются после обычного рестарта.
Раньше файл создавался только при отсутствии, и правки не применялись никогда.

Экранирование значений делается по правилам TOML: токен с кавычкой, бэкслешем
или значение USE_PROXY=True больше не ломают файл.
"""

import os
import sys
import tomllib
from pathlib import Path

TARGET = Path(os.environ.get("SETTINGS_PATH", "settings.toml"))
OVERRIDE = Path(os.environ.get("SETTINGS_OVERRIDE_PATH", "settings.local.toml"))

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "да"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "нет", ""}


def env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value.strip()


def quote(value: str) -> str:
    """Basic string по правилам TOML."""
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def boolean(name: str, default: bool = False) -> str:
    raw = env(name).lower()
    if raw in _TRUE_VALUES:
        return "true"
    if raw in _FALSE_VALUES:
        return "false"
    print(
        f"[settings] {name}='{raw}' не похоже на true/false, беру {str(default).lower()}",
        file=sys.stderr,
    )
    return str(default).lower()


def integer(name: str, default: int) -> str:
    raw = env(name)
    if not raw:
        return str(default)
    try:
        return str(int(raw))
    except ValueError:
        print(f"[settings] {name}='{raw}' не число, беру {default}", file=sys.stderr)
        return str(default)


def number(name: str, default: float) -> str:
    raw = env(name)
    if not raw:
        return str(default)
    try:
        return str(float(raw))
    except ValueError:
        print(f"[settings] {name}='{raw}' не число, беру {default}", file=sys.stderr)
        return str(default)


def id_list(*names: str) -> str:
    """Собирает список ID из первой непустой переменной."""
    for name in names:
        raw = env(name)
        if not raw:
            continue
        ids = []
        for chunk in raw.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk or chunk == "0":
                continue
            try:
                ids.append(str(int(chunk)))
            except ValueError:
                print(f"[settings] {name}: '{chunk}' не ID, пропускаю", file=sys.stderr)
        if ids:
            return "[" + ", ".join(ids) + "]"
    return "[]"


def require(name: str, placeholder: str) -> str:
    value = env(name)
    if not value or value == placeholder:
        print(
            f"Ошибка: переменная {name} не задана.\n"
            f"   Добавьте её в Pterodactyl: Startup -> Environment Variables",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def build() -> str:
    token = require("DISCORD_TOKEN", "YOUR_DISCORD_TOKEN")
    api_key = require("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

    return f"""# Сгенерирован автоматически из переменных окружения при старте контейнера.
# Правки здесь будут потеряны при следующем рестарте.
# Для постоянных ручных настроек используйте {OVERRIDE.name}.

[discord]
token = {quote(token)}
command_prefix = {quote(env("COMMAND_PREFIX", "!"))}
ticket_category_ids = {id_list("TICKET_CATEGORY_IDS", "TICKET_CATEGORY_ID")}
bot_role_ids = {id_list("BOT_ROLE_IDS", "BOT_ROLE_ID")}
ignored_role_ids = {id_list("IGNORED_ROLE_IDS")}

[ai]
embedding_model = {quote(env("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"))}
embedding_model_type = {quote(env("EMBEDDING_MODEL_TYPE", "e5-instruct"))}
search_top_k = {integer("SEARCH_TOP_K", 2)}
request_timeout_seconds = {integer("AI_REQUEST_TIMEOUT_SECONDS", 90)}
max_concurrent_requests = {integer("AI_MAX_CONCURRENT_REQUESTS", 2)}
temperature = {number("AI_TEMPERATURE", 0.3)}
max_tokens = {integer("AI_MAX_TOKENS", 1024)}

[ai.openrouter]
api_key = {quote(api_key)}
model = {quote(env("OPENROUTER_MODEL", "mimo-2.5-pro"))}
api_url = {quote(env("OPENROUTER_API_URL", "https://openrouter.ai/api/v1"))}
site_url = {quote(env("OPENROUTER_SITE_URL"))}
app_name = {quote(env("OPENROUTER_APP_NAME", "SinusSMP Support Bot"))}

[proxy]
enabled = {boolean("USE_PROXY", False)}
host = {quote(env("PROXY_HOST", "127.0.0.1"))}
port = {integer("PROXY_PORT", 10808)}
username = {quote(env("PROXY_USERNAME"))}
password = {quote(env("PROXY_PASSWORD"))}

[paths]
model_cache = "model_cache"
database = "chroma_db"
logs = "logs"
knowledge = "knowledge"
prompts = "prompts"

[knowledge]
collection_name = {quote(env("CHROMA_COLLECTION_NAME", "sinussmp_support"))}
strict_embedding_check = {boolean("STRICT_EMBEDDING_CHECK", True)}

[developer_logs]
# Должно совпадать с путём логов в egg.json, иначе вкладка логов в панели пуста.
filename = {quote(env("DEV_LOG_FILENAME", "latest.log"))}
level = {quote(env("DEV_LOG_LEVEL", "INFO"))}

[logs]
active_dir = "logs/active"
archive_enabled = {boolean("LOG_ARCHIVE_ENABLED", True)}
archive_dir = "logs/archives"
archive_interval_hours = {integer("LOG_ARCHIVE_INTERVAL_HOURS", 24)}
archive_safety_net_days = {integer("LOG_ARCHIVE_SAFETY_NET_DAYS", 7)}

[logs.categories]
donate = ["донат", "donate", "оплат", "payment"]
bug = ["баг", "bug", "ошибк", "краш", "crash"]
connection = ["подключ", "connect", "вход", "лаг", "пинг"]
account = ["аккаунт", "пароль", "account", "password", "взлом"]

[rate_limit]
enabled = {boolean("RATE_LIMIT_ENABLED", True)}
global_limit = {integer("RATE_LIMIT_GLOBAL_LIMIT", 30)}
global_window = {integer("RATE_LIMIT_GLOBAL_WINDOW", 60)}
channel_cooldown = {integer("RATE_LIMIT_CHANNEL_COOLDOWN", 5)}
duplicate_check_time = {integer("RATE_LIMIT_DUPLICATE_CHECK_TIME", 5)}
user_message_limit = {integer("RATE_LIMIT_USER_MESSAGE_LIMIT", 3)}
user_message_window = {integer("RATE_LIMIT_USER_MESSAGE_WINDOW", 10)}
max_history = {integer("RATE_LIMIT_MAX_HISTORY", 6)}
message_debounce_seconds = {number("MESSAGE_DEBOUNCE_SECONDS", 2.5)}

[server]
min_version = {quote(env("SERVER_MIN_VERSION", "1.19.4"))}
max_version = {quote(env("SERVER_MAX_VERSION", "1.21.10"))}
recommended_version = {quote(env("SERVER_RECOMMENDED_VERSION", "1.21.10"))}
site_url = {quote(env("SERVER_SITE_URL", "https://sinussmp.com"))}
boosty_url = {quote(env("SERVER_BOOSTY_URL", "https://boosty.to/ingrog"))}

[state]
snapshot_file = "logs/conversation_state.json"
save_interval_seconds = {integer("STATE_SAVE_INTERVAL_SECONDS", 30)}
ttl_seconds = {integer("STATE_TTL_SECONDS", 604800)}
"""


def check_override() -> None:
    """Проверяет, что ручной override — валидный TOML.

    Сам файл не сливается сюда: TOML не допускает повторного объявления секции,
    поэтому склейка текстом даёт битый результат. Слияние делает bot.settings
    при загрузке — там оно происходит на уровне словарей.
    """
    if not OVERRIDE.exists():
        return
    try:
        tomllib.loads(OVERRIDE.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        print(f"[settings] {OVERRIDE.name} содержит ошибку и будет пропущен: {exc}", file=sys.stderr)
        return
    except OSError as exc:
        print(f"[settings] Не удалось прочитать {OVERRIDE.name}: {exc}", file=sys.stderr)
        return
    print(f"[settings] Найден {OVERRIDE.name}, его ключи переопределят сгенерированные")


def main() -> None:
    content = build()

    # Проверяем результат до записи: битый файл на диске означает,
    # что бот не запустится вообще.
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        print(f"Сгенерирован некорректный TOML: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    check_override()

    TARGET.write_text(content, encoding="utf-8")
    try:
        TARGET.chmod(0o600)
    except OSError:
        pass
    print(f"settings.toml сгенерирован из переменных окружения ({TARGET})")


if __name__ == "__main__":
    main()
