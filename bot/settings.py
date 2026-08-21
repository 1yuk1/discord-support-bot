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
        # utf-8-sig, а не rb: файл правят руками, и редакторы на Windows
        # дописывают BOM. tomllib на нём падает с невнятным
        # «Invalid statement at line 1», хотя содержимое корректно.
        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
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

_fallback_cfg = _ai_cfg.get("fallback")
_fallback_cfg = _fallback_cfg if isinstance(_fallback_cfg, dict) else {}
FALLBACK_AI_ENABLED: bool = bool(_fallback_cfg.get("enabled", False))
FALLBACK_AI_API_KEY: str = _fallback_cfg.get("api_key", "")
FALLBACK_AI_MODEL: str = _fallback_cfg.get("model", "")
FALLBACK_AI_API_URL: str = _fallback_cfg.get("api_url", "")
FALLBACK_AI_USE_PROXY: bool = bool(_fallback_cfg.get("use_proxy", False))

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
# data/ — то, что бот пишет сам. Автообновление кода этот каталог не трогает,
# в отличие от knowledge/ и prompts/, которые перезаливаются из репозитория.
DATA_DIR: str = _as_path(_paths_cfg.get("data"), "data")

# ── Incidents ────────────────────────────────────────────────────────────────
# Известные проблемы («сервер лежит», «не работает вход»). Живут часы-дни,
# поэтому не индексируются, а подмешиваются в системный промпт напрямую.
_incidents_cfg = _section("incidents")
INCIDENTS_ENABLED: bool = bool(_incidents_cfg.get("enabled", True))
INCIDENTS_FILE: str = _as_path(_incidents_cfg.get("file"), "data/incidents.md")

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

# ── Reminders ────────────────────────────────────────────────────────────────
# Напоминание персоналу, если игрок ждёт ответа слишком долго. Роли задаются
# и глобально, и по категориям: на тестовом сервере роли ticket support может
# не быть вовсе, и пинговать там нечего.
_reminders_cfg = _section("reminders")
REMINDERS_ENABLED: bool = bool(_reminders_cfg.get("enabled", True))
REMINDER_STAFF_ROLE_IDS: list[int] = _parse_id_list(_reminders_cfg.get("staff_role_ids"))
REMINDER_PING_ROLE_IDS: list[int] = _parse_id_list(_reminders_cfg.get("ping_role_ids"))
REMINDER_IDLE_HOURS: float = float(_reminders_cfg.get("idle_hours", 1))
REMINDER_REPEAT_HOURS: float = float(_reminders_cfg.get("repeat_hours", 6))
REMINDER_MAX_PER_DAY: int = int(_reminders_cfg.get("max_per_day", 3))
REMINDER_CHECK_INTERVAL_MINUTES: int = int(_reminders_cfg.get("check_interval_minutes", 10))
REMINDER_EXCLUDED_CATEGORY_IDS: list[int] = _parse_id_list(
    _reminders_cfg.get("excluded_category_ids")
)
# llm — текст пишет модель под конкретный тикет; static — фраза из списка ниже.
REMINDER_MESSAGE_MODE: str = str(_reminders_cfg.get("message_mode", "llm")).lower()
REMINDER_HISTORY_LIMIT: int = int(_reminders_cfg.get("history_limit", 25))

_DEFAULT_REMINDER_PHRASES = [
    "Благодарим за обращение. Ваш вопрос всё ещё в работе — решение занимает "
    "чуть больше времени, чем мы рассчитывали. Спасибо за терпение.",
    "Ваш тикет не забыт: мы продолжаем разбираться с обращением. "
    "Благодарим за ожидание.",
    "Обращение всё ещё в обработке. Спасибо, что ждёте — вам ответят, "
    "как только появится свободное время.",
]
_phrases_raw = _reminders_cfg.get("phrases")
REMINDER_PHRASES: list[str] = [
    str(phrase).strip()
    for phrase in (_phrases_raw if isinstance(_phrases_raw, list) else [])
    if str(phrase).strip()
] or _DEFAULT_REMINDER_PHRASES

# Переопределения по категориям: {category_id: {ключ: значение}}.
_overrides_raw = _reminders_cfg.get("categories")
_overrides_raw = _overrides_raw if isinstance(_overrides_raw, dict) else {}
REMINDER_CATEGORY_OVERRIDES: dict[int, dict] = {}
for _raw_key, _override in _overrides_raw.items():
    if not isinstance(_override, dict):
        continue
    try:
        _category_id = int(str(_raw_key).strip())
    except (TypeError, ValueError):
        continue
    REMINDER_CATEGORY_OVERRIDES[_category_id] = _override


def reminder_config_for(category_id) -> dict:
    """Итоговые настройки напоминаний для категории канала.

    Значения из [reminders.categories.<id>] перекрывают глобальные. Роли
    персонала по умолчанию берутся из ignored_role_ids: это ровно те роли,
    чьи сообщения бот и так считает «не игроком».
    """
    override = REMINDER_CATEGORY_OVERRIDES.get(category_id, {})

    def pick(key: str, fallback):
        return override.get(key, fallback) if isinstance(override, dict) else fallback

    staff_roles = _parse_id_list(pick("staff_role_ids", REMINDER_STAFF_ROLE_IDS))
    if not staff_roles:
        staff_roles = list(IGNORED_ROLE_IDS)

    return {
        "enabled": bool(pick("enabled", REMINDERS_ENABLED)),
        "staff_role_ids": staff_roles,
        "ping_role_ids": _parse_id_list(pick("ping_role_ids", REMINDER_PING_ROLE_IDS)),
        "idle_hours": float(pick("idle_hours", REMINDER_IDLE_HOURS)),
        "repeat_hours": float(pick("repeat_hours", REMINDER_REPEAT_HOURS)),
        "max_per_day": int(pick("max_per_day", REMINDER_MAX_PER_DAY)),
        "message_mode": str(pick("message_mode", REMINDER_MESSAGE_MODE)).lower(),
        "phrases": pick("phrases", REMINDER_PHRASES) or _DEFAULT_REMINDER_PHRASES,
    }


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


# ── Горячая перезагрузка ─────────────────────────────────────────────────────
# Перечитывать можно не всё. Токен и прокси участвуют в уже установленном
# соединении, пути прочитаны при старте, а смена модели эмбеддингов или имени
# коллекции на живом боте означает несовместимые векторы и тихий мусор в
# поиске — такие правки требуют рестарта.
#
# Формат: (имя переменной, секция, ключ, преобразователь, значение по умолчанию)
_HOT_RELOADABLE: tuple[tuple, ...] = (
    ("AI_TEMPERATURE", "ai", "temperature", float, 0.3),
    ("AI_MAX_TOKENS", "ai", "max_tokens", int, 1024),
    ("AI_REQUEST_TIMEOUT_SECONDS", "ai", "request_timeout_seconds", int, 90),
    ("SEARCH_TOP_K", "ai", "search_top_k", int, 2),
    ("IMAGE_MAX_BYTES", "ai", "image_max_bytes", int, 8 * 1024 * 1024),
    ("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "ai", "image_download_timeout_seconds", int, 30),

    ("TICKET_CATEGORY_IDS", "discord", "ticket_category_ids", _parse_id_list, []),
    ("BOT_ROLE_IDS", "discord", "bot_role_ids", _parse_id_list, []),
    ("IGNORED_ROLE_IDS", "discord", "ignored_role_ids", _parse_id_list, []),
    ("BOT_REPLY_FOOTER", "discord", "reply_footer", str, BOT_REPLY_FOOTER),

    ("RATE_LIMIT_ENABLED", "rate_limit", "enabled", bool, True),
    ("RATE_LIMIT", "rate_limit", "global_limit", int, 30),
    ("RATE_WINDOW", "rate_limit", "global_window", int, 60),
    ("CHANNEL_COOLDOWN", "rate_limit", "channel_cooldown", int, 5),
    ("DUPLICATE_CHECK_TIME", "rate_limit", "duplicate_check_time", int, 5),
    ("USER_MESSAGE_LIMIT", "rate_limit", "user_message_limit", int, 3),
    ("USER_MESSAGE_WINDOW", "rate_limit", "user_message_window", int, 10),
    ("MAX_HISTORY", "rate_limit", "max_history", int, 6),
    ("MESSAGE_DEBOUNCE_SECONDS", "rate_limit", "message_debounce_seconds", float, 2.5),
    ("PING_SPAM_LIMIT", "rate_limit", "ping_spam_limit", int, 3),
    ("PING_SPAM_WINDOW", "rate_limit", "ping_spam_window", int, 300),

    ("REMINDERS_ENABLED", "reminders", "enabled", bool, True),
    ("REMINDER_STAFF_ROLE_IDS", "reminders", "staff_role_ids", _parse_id_list, []),
    ("REMINDER_PING_ROLE_IDS", "reminders", "ping_role_ids", _parse_id_list, []),
    ("REMINDER_IDLE_HOURS", "reminders", "idle_hours", float, 1.0),
    ("REMINDER_REPEAT_HOURS", "reminders", "repeat_hours", float, 6.0),
    ("REMINDER_MAX_PER_DAY", "reminders", "max_per_day", int, 3),
    ("REMINDER_HISTORY_LIMIT", "reminders", "history_limit", int, 25),
    (
        "REMINDER_EXCLUDED_CATEGORY_IDS",
        "reminders",
        "excluded_category_ids",
        _parse_id_list,
        [],
    ),

    ("INCIDENTS_ENABLED", "incidents", "enabled", bool, True),

    ("STATE_TTL_SECONDS", "state", "ttl_seconds", int, 7 * 24 * 60 * 60),
)

# Требуют рестарта. Список нужен, чтобы /config reload честно сказал, что
# именно не применится, вместо тихого игнорирования правки.
RESTART_REQUIRED_KEYS: tuple[str, ...] = (
    "[discord].token",
    "[proxy].*",
    "[paths].*",
    "[ai].embedding_model",
    "[ai].embedding_model_type",
    "[ai].max_concurrent_requests",
    "[ai.openrouter].*",
    "[ai.fallback].*",
    "[knowledge].collection_name",
    "[reminders].check_interval_minutes",
    "[developer_logs].*",
)


def reload() -> dict[str, tuple]:
    """Перечитывает settings.toml и обновляет безопасные значения.

    Возвращает {имя: (было, стало)} только по изменившимся ключам. Ошибка
    чтения файла поднимается наружу, текущие значения при этом сохраняются:
    битый конфиг не должен обрушить работающего бота.
    """
    global _cfg

    fresh = _read_toml(SETTINGS_PATH, required=False)

    if not fresh:
        raise ValueError(
            f"{SETTINGS_PATH.name} не прочитан или пуст — прежние настройки сохранены"
        )

    override = _read_toml(OVERRIDE_PATH, required=False)
    merged = _deep_merge(fresh, override) if override else fresh

    module = globals()
    changes: dict[str, tuple] = {}

    def section_of(name: str) -> dict:
        value = merged.get(name)
        return value if isinstance(value, dict) else {}

    for variable, section_name, key, converter, default in _HOT_RELOADABLE:
        raw = section_of(section_name).get(key, default)
        try:
            value = converter(raw) if raw is not None else default
        except (TypeError, ValueError):
            # Мусор в одном ключе не должен отменять перезагрузку остальных.
            continue

        previous = module.get(variable)
        if previous != value:
            changes[variable] = (previous, value)
            module[variable] = value

    # Переопределения по категориям: структура вложенная, парсится отдельно.
    raw_overrides = section_of("reminders").get("categories")
    raw_overrides = raw_overrides if isinstance(raw_overrides, dict) else {}
    fresh_overrides: dict[int, dict] = {}
    for raw_key, override_value in raw_overrides.items():
        if not isinstance(override_value, dict):
            continue
        try:
            fresh_overrides[int(str(raw_key).strip())] = override_value
        except (TypeError, ValueError):
            continue

    if fresh_overrides != REMINDER_CATEGORY_OVERRIDES:
        changes["REMINDER_CATEGORY_OVERRIDES"] = (
            sorted(REMINDER_CATEGORY_OVERRIDES),
            sorted(fresh_overrides),
        )
        module["REMINDER_CATEGORY_OVERRIDES"] = fresh_overrides

    # Модель меняется через ModelRegistry: она хранит своё значение, поэтому
    # здесь только обновляем эталон из файла.
    fresh_model = section_of("ai").get("openrouter")
    fresh_model = fresh_model if isinstance(fresh_model, dict) else {}
    new_model = fresh_model.get("model", OPENROUTER_MODEL)
    if new_model and new_model != OPENROUTER_MODEL:
        changes["OPENROUTER_MODEL"] = (OPENROUTER_MODEL, new_model)
        module["OPENROUTER_MODEL"] = new_model

    _cfg = merged
    return changes
