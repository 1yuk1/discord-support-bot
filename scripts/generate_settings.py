"""Генерация settings.toml из переменных окружения.

Файл создаётся ОДИН раз — при первом старте, когда его ещё нет. Дальше он
принадлежит пользователю: правки прямо в settings.toml сохраняются между
рестартами, и панель их не перетирает.

Заставить перегенерировать можно двумя способами:
  - SETTINGS_FORCE_REGENERATE=true в переменных окружения;
  - удалить settings.toml (создастся заново из переменных).

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
_FALSE_VALUES = {"0", "false", "no", "n", "off", "нет"}


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
    """Значение флага из переменной окружения.

    Незаданная переменная означает «бери default», а не false: иначе все
    флаги с default=True (напоминания, инциденты, рейт-лимит, архивация
    логов) молча выключались бы, если панель их не передала.
    """
    raw = env(name).lower()
    if not raw:
        return str(default).lower()
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


def str_list(name: str, default: list[str]) -> str:
    """Парсит список строк из переменной окружения."""
    raw = env(name)
    if not raw:
        items = default
    else:
        items = [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]
    quoted = [quote(item) for item in items]
    return "[" + ", ".join(quoted) + "]"


# Обязательные переменные и их placeholder-значения из egg.json.
REQUIRED_VARS = (
    ("DISCORD_TOKEN", "YOUR_DISCORD_TOKEN"),
    ("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY"),
)

# Переменные, которые бот умеет читать. Нужны только для диагностики:
# показываем, что панель вообще передала в контейнер.
KNOWN_OPTIONAL_VARS = (
    "SETTINGS_FORCE_REGENERATE",
    "OPENROUTER_MODEL",
    "FALLBACK_AI_ENABLED",
    "FALLBACK_AI_API_KEY",
    "FALLBACK_AI_MODEL",
    "FALLBACK_AI_API_URL",
    "FALLBACK_AI_USE_PROXY",
    "TICKET_CATEGORY_IDS",
    "TICKET_CATEGORY_ID",
    "BOT_ROLE_IDS",
    "BOT_ROLE_ID",
    "IGNORED_ROLE_IDS",
    "EMBEDDING_MODEL",
    "USE_PROXY",
    "BOT_AUTO_UPDATE",
    "REMINDER_QUIET_HOURS_TIMEZONE",
    "REMINDER_QUIET_HOURS_START",
    "REMINDER_QUIET_HOURS_END",
    "MENTION_TIMEOUT_ENABLED",
    "MENTION_PROTECTED_USER_IDS",
    "MENTION_PROTECTED_ROLE_IDS",
    "MENTION_TIMEOUT_DURATIONS",
    "MENTION_ESCALATION_RESET_DAYS",
    "MENTION_TIMEOUT_REASON",
)


def check_required() -> None:
    """Проверяет все обязательные переменные сразу.

    Сообщать про них по одной неудобно: после исправления первой запуск
    падает на второй. Значения переменных не печатаем — это секреты.
    """
    missing = [
        name
        for name, placeholder in REQUIRED_VARS
        if not env(name) or env(name) == placeholder
    ]
    if not missing:
        return

    print("Ошибка: не заданы обязательные переменные окружения:", file=sys.stderr)
    for name in missing:
        print(f"   - {name}", file=sys.stderr)
    print(
        "\n   Задайте их в Pterodactyl: Startup -> Environment Variables,\n"
        "   затем перезапустите сервер.",
        file=sys.stderr,
    )

    provided = [name for name in KNOWN_OPTIONAL_VARS if env(name)]
    if provided:
        print(f"\n   Панель передала: {', '.join(provided)}", file=sys.stderr)
    else:
        print(
            "\n   Панель не передала ни одной известной переменной.\n"
            "   Похоже, после импорта egg.json значения сбросились — "
            "заполните их заново.",
            file=sys.stderr,
        )

    raise SystemExit(1)


def build() -> str:
    token = env("DISCORD_TOKEN")
    api_key = env("OPENROUTER_API_KEY")

    return f"""# Создан автоматически из переменных окружения при первом старте.
# Файл больше не перегенерируется: правьте его смело, рестарт правки сохранит.
#
# Пересоздать из переменных окружения: удалите этот файл либо задайте
# SETTINGS_FORCE_REGENERATE=true в панели.
# Ещё можно держать ручные правки отдельно в {OVERRIDE.name} — его ключи
# переопределяют значения отсюда.

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

[ai.fallback]
# Резервный OpenAI-совместимый провайдер. При ошибке OpenRouter бот повторит
# запрос через него. При enabled=true обязательны api_key, model и api_url.
enabled = {boolean("FALLBACK_AI_ENABLED", False)}
api_key = {quote(env("FALLBACK_AI_API_KEY"))}
model = {quote(env("FALLBACK_AI_MODEL"))}
api_url = {quote(env("FALLBACK_AI_API_URL"))}
use_proxy = {boolean("FALLBACK_AI_USE_PROXY", False)}

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
data = "data"

[incidents]
# Известные проблемы, которые бот учитывает в каждом ответе.
# Управление: /incident add, /incident list, /incident remove.
enabled = {boolean("INCIDENTS_ENABLED", True)}
file = "data/incidents.md"

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

[reminders]
# Напоминание персоналу, если игрок давно ждёт ответа.
# ping_role_ids обязателен: пока он пуст, напоминания не отправляются.
enabled = {boolean("REMINDERS_ENABLED", True)}
staff_role_ids = {id_list("REMINDER_STAFF_ROLE_IDS")}
ping_role_ids = {id_list("REMINDER_PING_ROLE_IDS")}
idle_hours = {number("REMINDER_IDLE_HOURS", 1.0)}
repeat_hours = {number("REMINDER_REPEAT_HOURS", 6.0)}
max_per_day = {integer("REMINDER_MAX_PER_DAY", 3)}
check_interval_minutes = {integer("REMINDER_CHECK_INTERVAL_MINUTES", 10)}
excluded_category_ids = {id_list("REMINDER_EXCLUDED_CATEGORY_IDS")}
# llm — текст пишет модель под конкретный тикет; static — фраза из phrases.
message_mode = {quote(env("REMINDER_MESSAGE_MODE", "llm"))}
quiet_hours_timezone = {quote(env("REMINDER_QUIET_HOURS_TIMEZONE", "Europe/Moscow"))}
quiet_hours_start = {quote(env("REMINDER_QUIET_HOURS_START", "23:00"))}
quiet_hours_end = {quote(env("REMINDER_QUIET_HOURS_END", "09:00"))}

# Роли и тайминги можно переопределить для отдельной категории. Пример:
# на тестовом сервере роли поддержки нет, поэтому там напоминания выключены.
#
# [reminders.categories.123456789012345678]
# ping_role_ids = [1119360384395132973]
# staff_role_ids = [1119360384395132973]
# idle_hours = 2
#
# [reminders.categories.987654321098765432]
# enabled = false

[mention_timeout]
enabled = {boolean("MENTION_TIMEOUT_ENABLED", True)}
protected_user_ids = {id_list("MENTION_PROTECTED_USER_IDS")}
protected_role_ids = {id_list("MENTION_PROTECTED_ROLE_IDS")}
durations = {str_list("MENTION_TIMEOUT_DURATIONS", ["1m", "10m", "1h"])}
escalation_reset_days = {integer("MENTION_ESCALATION_RESET_DAYS", 30)}
reason = {quote(env("MENTION_TIMEOUT_REASON", "Запрещённый пинг участника или роли поддержки"))}

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
        tomllib.loads(OVERRIDE.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as exc:
        print(f"[settings] {OVERRIDE.name} содержит ошибку и будет пропущен: {exc}", file=sys.stderr)
        return
    except OSError as exc:
        print(f"[settings] Не удалось прочитать {OVERRIDE.name}: {exc}", file=sys.stderr)
        return
    print(f"[settings] Найден {OVERRIDE.name}, его ключи переопределят сгенерированные")


def force_regenerate() -> bool:
    return env("SETTINGS_FORCE_REGENERATE").lower() in _TRUE_VALUES


def validate_existing() -> None:
    """Проверяет, что готовый settings.toml читается.

    Обязательные переменные окружения здесь НЕ проверяются: значения уже
    лежат в файле, а пустой DISCORD_TOKEN в панели не повод отказываться от
    запуска. Отсутствие токена внутри файла поймает bot.settings.
    """
    try:
        # utf-8-sig: редакторы на Windows дописывают BOM, и tomllib на нём
        # падает с невнятным «Invalid statement at line 1».
        tomllib.loads(TARGET.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as exc:
        print(f"{TARGET.name} содержит синтаксическую ошибку: {exc}", file=sys.stderr)
        print(
            "   Исправьте файл, либо удалите его — тогда он создастся заново\n"
            "   из переменных окружения панели.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except OSError as exc:
        print(f"Не удалось прочитать {TARGET.name}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def main() -> None:
    if TARGET.exists() and not force_regenerate():
        validate_existing()
        check_override()
        print(
            f"{TARGET.name} уже существует — генерация пропущена, ручные правки сохранены.\n"
            f"   Пересоздать из переменных: удалите файл или задайте "
            f"SETTINGS_FORCE_REGENERATE=true"
        )
        return

    if TARGET.exists():
        print(f"SETTINGS_FORCE_REGENERATE=true — {TARGET.name} будет перезаписан")

    check_required()
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
    print(f"settings.toml создан из переменных окружения ({TARGET})")


if __name__ == "__main__":
    main()
