"""Общая настройка тестов.

bot.settings читает settings.toml при импорте, поэтому подкладываем временный
файл до того, как тестовые модули импортируют пакет. Настоящий settings.toml
с токенами в тестах не участвует.
"""

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SETTINGS_TEMPLATE = """
[discord]
token = "test-token"
command_prefix = "!"
ticket_category_ids = [111, 222]
bot_role_ids = []
ignored_role_ids = [999]

[ai]
embedding_model = "intfloat/multilingual-e5-large-instruct"
embedding_model_type = "e5-instruct"
search_top_k = 2

[ai.openrouter]
api_key = "test-key"
model = "test-model"

[ai.fallback]
enabled = false
api_key = ""
model = ""
api_url = ""
use_proxy = false

[proxy]
enabled = false
host = "127.0.0.1"
port = 10808

[paths]
model_cache = "model_cache"
database = "chroma_db"
logs = "logs"
knowledge = {knowledge_dir}
prompts = {prompts_dir}

[knowledge]
collection_name = "sinussmp_support"

[rate_limit]
enabled = true
max_history = 6

[server]
min_version = "1.19.4"
max_version = "1.21.10"
recommended_version = "1.21.10"
site_url = "https://sinussmp.com"
boosty_url = "https://boosty.to/ingrog"

[state]
ttl_seconds = 604800
"""


def _toml_path(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\") + '"'


def _prepare_settings() -> None:
    base_dir = Path(tempfile.mkdtemp(prefix="support-bot-tests-"))
    settings_content = _SETTINGS_TEMPLATE.format(
        knowledge_dir=_toml_path(REPO_ROOT / "knowledge"),
        prompts_dir=_toml_path(REPO_ROOT / "prompts"),
    )
    (base_dir / "settings.toml").write_text(settings_content, encoding="utf-8")

    os.environ["APP_BASE_DIR"] = str(base_dir)
    os.environ.pop("KNOWLEDGE_DIR", None)


_prepare_settings()
