#!/bin/bash
set -e

cd /home/container

# Huggingface cache
export HF_HOME=/home/container/model_cache
export TRANSFORMERS_CACHE=/home/container/model_cache

# Создаём директории для данных
mkdir -p chroma_db logs model_cache

# Всегда обновляем код из GitHub (сохраняя пользовательские данные)
echo "📦 Обновление кода из GitHub..."
git clone --depth 1 https://github.com/1yuk1/discord-support-bot.git /tmp/bot-update 2>/dev/null || {
    echo "⚠️ Не удалось скачать обновления, используем текущий код"
}

if [ -d "/tmp/bot-update" ]; then
    # Копируем только код, НЕ перезаписываем данные пользователя
    cp -f /tmp/bot-update/*.py /home/container/ 2>/dev/null || true
    cp -f /tmp/bot-update/*.sh /home/container/ 2>/dev/null || true
    cp -f /tmp/bot-update/requirements.txt /home/container/ 2>/dev/null || true
    cp -f /tmp/bot-update/quests_summary.md /home/container/ 2>/dev/null || true
    if [ -d "/tmp/bot-update/knowledge" ]; then
        mkdir -p /home/container/knowledge
        cp -f /tmp/bot-update/knowledge/*.json /home/container/knowledge/ 2>/dev/null || true
    fi
    rm -rf /tmp/bot-update
    echo "✅ Код обновлён"
fi

if [ ! -f "settings.toml" ]; then
    echo "⚙️ Создание settings.toml..."

    ticket_category_ids="${TICKET_CATEGORY_IDS:-${TICKET_CATEGORY_ID:-0}}"
    ignored_role_ids="${IGNORED_ROLE_IDS:-0}"
    
    # Проверка обязательных переменных
    if [ -z "$DISCORD_TOKEN" ] || [ "$DISCORD_TOKEN" = "YOUR_DISCORD_TOKEN" ]; then
        echo "❌ Ошибка: DISCORD_TOKEN не установлен!"
        echo "   Добавь переменные в Pterodactyl Startup → Environment Variables"
        exit 1
    fi
    if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "YOUR_GROQ_API_KEY" ]; then
        echo "❌ Ошибка: GROQ_API_KEY не установлен!"
        exit 1
    fi
    
    cat > settings.toml << EOF
# Discord Bot Settings
[discord]
token = "$DISCORD_TOKEN"
ticket_category_id = ${TICKET_CATEGORY_ID:-0}
ticket_category_ids = "$ticket_category_ids"
bot_role_id = ${BOT_ROLE_ID:-0}
ignored_role_ids = "$ignored_role_ids"

[ai]
provider = "groq"
embedding_model = "${EMBEDDING_MODEL:-intfloat/multilingual-e5-large-instruct}"
embedding_model_type = "${EMBEDDING_MODEL_TYPE:-e5-instruct}"

[ai.groq]
api_key = "$GROQ_API_KEY"
model = "groq/compound"

[ai.local]
api_url = "http://localhost:1234/v1"
api_key = "not-needed"
model = "local-model"

[proxy]
enabled = ${USE_PROXY:-false}
host = "${PROXY_HOST:-127.0.0.1}"
port = ${PROXY_PORT:-10808}
username = "${PROXY_USERNAME:-}"
password = "${PROXY_PASSWORD:-}"

[paths]
model_cache = "model_cache"
database = "chroma_db"
logs = "logs"

[rate_limit]
enabled = ${RATE_LIMIT_ENABLED:-true}
global_limit = ${RATE_LIMIT_GLOBAL_LIMIT:-30}
global_window = ${RATE_LIMIT_GLOBAL_WINDOW:-60}
channel_cooldown = ${RATE_LIMIT_CHANNEL_COOLDOWN:-5}
duplicate_check_time = ${RATE_LIMIT_DUPLICATE_CHECK_TIME:-5}
user_message_limit = ${RATE_LIMIT_USER_MESSAGE_LIMIT:-3}
user_message_window = ${RATE_LIMIT_USER_MESSAGE_WINDOW:-10}
max_history = 6

[transfer]
phrases = [
    "тех поддержка", "техподдержка", "переведи на человека", "позови человека",
    "живой человек", "оператор", "администратор", "админ", "модератор", "модер",
    "переведи на админа", "позови админа", "соедини с человеком", "хочу человека",
    "говорить с человеком", "поговорить с человеком", "пригласи человека",
    "старший специалист", "позови специалиста", "переведи на специалиста",
    "передам", "передаю", "передал", "передать человеку", "передаю тикет"
]
EOF
    echo "✅ settings.toml создан"
fi

echo "🧠 Проверка базы знаний ChromaDB..."
index_signature_file="chroma_db/.index_signature"
current_signature=$(python - <<'PY'
import hashlib
from pathlib import Path

parts = []
for filename in ["indexer.py", "config.py", "settings.toml", "requirements.txt"]:
    path = Path(filename)
    if path.exists():
        parts.append(path.read_bytes())

knowledge_dir = Path("knowledge")
if knowledge_dir.is_dir():
    for path in sorted(knowledge_dir.glob("*.json")):
        parts.append(path.read_bytes())

quests_path = Path("quests_summary.md")
if quests_path.exists():
    parts.append(quests_path.read_bytes())

digest = hashlib.sha256(b"\n---file---\n".join(parts)).hexdigest()
print(digest)
PY
)

stored_signature=""
if [ -f "$index_signature_file" ]; then
    stored_signature=$(cat "$index_signature_file")
fi

if [ ! -f "chroma_db/chroma.sqlite3" ] || [ "$current_signature" != "$stored_signature" ]; then
    echo "🔄 База знаний отсутствует или устарела, запускаю индексацию..."
    RUN_INDEXER_TESTS=0 python indexer.py
    echo "$current_signature" > "$index_signature_file"
    echo "✅ База знаний обновлена"
else
    echo "✅ База знаний актуальна"
fi

echo "🚀 Запуск бота..."
exec python discord_bot.py
