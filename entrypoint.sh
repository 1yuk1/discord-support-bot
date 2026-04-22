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
    rm -rf /tmp/bot-update
    echo "✅ Код обновлён"
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
username = "${PROXY_USERNAME:-}"
password = "${PROXY_PASSWORD:-}"

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

echo "🚀 Запуск бота..."
exec python discord_bot.py