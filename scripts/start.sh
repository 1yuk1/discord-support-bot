#!/bin/bash
# Запуск бота: подготовка каталогов, настройки, база знаний.
#
# Этот файл лежит в репозитории и обновляется автообновлением, поэтому его
# правки применяются после обычного рестарта, без пересборки образа.
set -euo pipefail

cd /home/container

export HF_HOME=/home/container/model_cache
# TRANSFORMERS_CACHE устарел в пользу HF_HOME и вызывает предупреждение.
unset TRANSFORMERS_CACHE 2>/dev/null || true

mkdir -p chroma_db logs logs/active logs/archives model_cache

# ── Настройки ────────────────────────────────────────────────────────────────
# Генерируем каждый старт, чтобы правки переменных в панели Pterodactyl
# применялись после обычного рестарта. Раньше файл создавался только при
# отсутствии, и изменения не применялись никогда.
echo "Генерация settings.toml из переменных окружения..."
python scripts/generate_settings.py

# ── База знаний ──────────────────────────────────────────────────────────────
echo "Проверка базы знаний ChromaDB..."
signature_file="chroma_db/.index_signature"

current_signature=$(python - <<'PY'
"""Подпись входных данных индекса.

Учитываем только то, что влияет на содержимое векторов: сами знания, код их
обработки и настройки модели. Правки в handlers.py или commands.py
переиндексацию не требуют — иначе каждый деплой заново грузил бы модель.
"""
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()


def feed(path: Path) -> None:
    digest.update(str(path).encode())
    digest.update(path.read_bytes())
    digest.update(b"\n---file---\n")


for name in ["indexer.py", "bot/knowledge.py", "bot/embeddings.py", "bot/text_utils.py"]:
    path = Path(name)
    if path.exists():
        feed(path)

knowledge_dir = Path("knowledge")
if knowledge_dir.is_dir():
    for path in sorted(knowledge_dir.glob("*.json")):
        feed(path)

try:
    import tomllib

    with open("settings.toml", "rb") as f:
        config = tomllib.load(f)
    relevant = {
        "embedding_model": config.get("ai", {}).get("embedding_model"),
        "embedding_model_type": config.get("ai", {}).get("embedding_model_type"),
        "knowledge": config.get("knowledge", {}),
        "server": config.get("server", {}),
    }
    digest.update(repr(sorted(relevant.items())).encode())
except Exception as exc:  # noqa: BLE001
    print(f"signature-warning: {exc}", file=sys.stderr)

print(digest.hexdigest())
PY
)

stored_signature=""
if [ -f "$signature_file" ]; then
    stored_signature=$(cat "$signature_file")
fi

if [ ! -f "chroma_db/chroma.sqlite3" ] || [ "$current_signature" != "$stored_signature" ]; then
    echo "База знаний отсутствует или устарела, запускаю индексацию..."
    # Подпись пишем только после успешной индексации: иначе следующий старт
    # решит, что база готова, и бот упадёт на отсутствующей коллекции.
    if python indexer.py; then
        echo "$current_signature" > "$signature_file"
        echo "База знаний обновлена"
    else
        exit_code=$?
        if [ "$exit_code" = "2" ]; then
            echo "Индексация отключена в настройках, использую текущую базу"
        else
            echo "Индексация завершилась с ошибкой (код $exit_code)" >&2
            exit "$exit_code"
        fi
    fi
else
    echo "База знаний актуальна"
fi

echo "Запуск бота..."
exec python discord_bot.py
