"""Тесты скриптов запуска.

Главная защита здесь — от рассинхрона между bootstrap в образе и структурой
репозитория. Если появится новый каталог с кодом, а entrypoint.sh не научится
его копировать, бот упадёт на ModuleNotFoundError уже в продакшене.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "entrypoint.sh"
START_SCRIPT = REPO_ROOT / "scripts" / "start.sh"
EGG = REPO_ROOT / "egg.json"
DOCKERFILE = REPO_ROOT / "Dockerfile"


def test_scripts_use_lf_line_endings():
    """CRLF в .sh даёт 'bad interpreter' в Linux."""
    for script in (ENTRYPOINT, START_SCRIPT):
        assert b"\r\n" not in script.read_bytes(), f"{script.name} содержит CRLF"


def test_gitattributes_enforces_lf():
    content = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in content


def test_bootstrap_copies_every_code_directory():
    """Все каталоги пакета должны попадать в контейнер."""
    content = ENTRYPOINT.read_text(encoding="utf-8")
    match = re.search(r"for dir in ([^;]+); do", content)
    assert match, "не найден список копируемых каталогов"

    copied = set(match.group(1).split())
    required = {"bot", "scripts", "prompts", "knowledge"}
    assert required.issubset(copied), f"не копируются: {required - copied}"


def test_bootstrap_copies_root_entry_points():
    content = ENTRYPOINT.read_text(encoding="utf-8")
    for name in ("discord_bot.py", "indexer.py", "requirements.txt"):
        assert name in content, f"{name} не копируется bootstrap-скриптом"


def test_bootstrap_hands_over_to_start_script():
    content = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'exec bash "$START_SCRIPT"' in content
    assert 'START_SCRIPT="scripts/start.sh"' in content


def test_bootstrap_fails_loudly_without_code():
    """Без загруженного кода нужна понятная ошибка, а не пустой запуск."""
    content = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'if [ ! -f "$START_SCRIPT" ]; then' in content
    assert "exit 1" in content


def test_data_directories_are_not_wiped():
    """Обновление кода не должно удалять данные пользователя."""
    content = ENTRYPOINT.read_text(encoding="utf-8")
    match = re.search(r"for dir in ([^;]+); do", content)
    copied = set(match.group(1).split())

    for protected in ("chroma_db", "logs", "model_cache"):
        assert protected not in copied, f"{protected} затирается при обновлении"


def test_start_script_regenerates_settings():
    """Настройки обязаны пересоздаваться каждый старт."""
    content = START_SCRIPT.read_text(encoding="utf-8")
    assert "python scripts/generate_settings.py" in content
    # Условия «только если файла нет» быть не должно.
    assert 'if [ ! -f "settings.toml" ]' not in content


def test_start_script_writes_signature_only_after_success():
    content = START_SCRIPT.read_text(encoding="utf-8")
    indexer_position = content.index("if python indexer.py; then")
    signature_position = content.index('echo "$current_signature" > "$signature_file"')
    assert indexer_position < signature_position


def test_start_script_creates_log_directories():
    content = START_SCRIPT.read_text(encoding="utf-8")
    assert "logs/active" in content
    assert "logs/archives" in content


def test_egg_startup_points_to_image_bootstrap():
    egg = json.loads(EGG.read_text(encoding="utf-8"))
    assert egg["startup"] == "bash /entrypoint.sh"


def test_egg_log_path_matches_bot_output():
    """Иначе вкладка логов в панели Pterodactyl остаётся пустой."""
    from bot import settings

    egg = json.loads(EGG.read_text(encoding="utf-8"))
    location = json.loads(egg["config"]["logs"])["location"]
    assert location == f"logs/{settings.DEV_LOG_FILENAME}"


def test_egg_declares_required_variables():
    egg = json.loads(EGG.read_text(encoding="utf-8"))
    declared = {variable["env_variable"] for variable in egg["variables"]}
    assert {"DISCORD_TOKEN", "OPENROUTER_API_KEY"}.issubset(declared)


def test_egg_variables_used_by_generator():
    """Переменные из панели должны реально читаться генератором настроек."""
    egg = json.loads(EGG.read_text(encoding="utf-8"))
    generator = (REPO_ROOT / "scripts" / "generate_settings.py").read_text(encoding="utf-8")
    bootstrap = ENTRYPOINT.read_text(encoding="utf-8")

    declared = {variable["env_variable"] for variable in egg["variables"]}
    unused = {
        name
        for name in declared
        if name not in generator and name not in bootstrap
    }
    assert not unused, f"переменные объявлены в панели, но не используются: {unused}"


def test_dockerfile_ships_only_bootstrap():
    """Код бота приходит из репозитория, в образе его быть не должно."""
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY entrypoint.sh" in content
    assert "COPY bot" not in content
    assert "COPY discord_bot.py" not in content


def test_dockerfile_normalizes_line_endings():
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "sed -i 's/\\r$//' /entrypoint.sh" in content
