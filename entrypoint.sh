#!/bin/sh
set -e

echo "============================================"
echo "  SinusSMP Discord Support Bot"
echo "============================================"
echo ""

# Проверка наличия settings.toml
if [ ! -f "settings.toml" ]; then
    echo "❌ settings.toml не найден!"
    echo "   Скопируйте settings.toml.example -> settings.toml и заполните значения."
    exit 1
fi

echo "🤖 Запуск бота..."
exec python discord_bot.py
