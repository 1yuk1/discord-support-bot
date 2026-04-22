#!/bin/bash
set -e

cd /home/container

# Huggingface cache
export HF_HOME=/home/container/model_cache
export TRANSFORMERS_CACHE=/home/container/model_cache

if [ ! -f "discord_bot.py" ]; then
    echo "📦 Скачивание кода бота из GitHub..."
    git clone --depth 1 https://github.com/1yuk1/discord-support-bot.git .
    rm -rf .git
fi

if [ ! -f "settings.toml" ]; then
    echo "⚙️ Создание settings.toml..."
    
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
bot_role_id = ${BOT_ROLE_ID:-0}

[ai]
provider = "groq"

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
username = ""
password = ""

[paths]
model_cache = "model_cache"
database = "chroma_db"
logs = "logs"

[rate_limit]
global_limit = 30
global_window = 60
channel_cooldown = 5
duplicate_check_time = 5
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

mkdir -p chroma_db logs model_cache

echo "🚀 Запуск бота..."
exec python discord_bot.py