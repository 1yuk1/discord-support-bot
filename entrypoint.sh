#!/bin/bash
# Bootstrap в Docker-образе. Задача одна: получить актуальный код из
# репозитория и передать ему управление.
#
# Вся рабочая логика живёт в scripts/start.sh внутри репозитория, поэтому
# её правки применяются без пересборки образа. Этот файл менять почти никогда
# не нужно — любое изменение здесь требует нового образа.
set -euo pipefail

cd /home/container

REPO_URL="${BOT_REPO_URL:-https://github.com/1yuk1/discord-support-bot.git}"
REPO_BRANCH="${BOT_REPO_BRANCH:-main}"
UPDATE_DIR="/tmp/bot-update"
START_SCRIPT="scripts/start.sh"

if [ "${BOT_AUTO_UPDATE:-true}" = "true" ]; then
    echo "Загрузка кода из GitHub (ветка ${REPO_BRANCH})..."
    rm -rf "$UPDATE_DIR"

    if git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$UPDATE_DIR" 2>&1; then
        # Файлы в корне репозитория.
        for name in discord_bot.py indexer.py requirements.txt pyproject.toml entrypoint.sh; do
            [ -f "$UPDATE_DIR/$name" ] && cp -f "$UPDATE_DIR/$name" /home/container/
        done

        # Каталоги кода синхронизируем с удалением: cp оставил бы файлы,
        # убранные из репозитория, а лишний блок знаний ломает индексацию
        # дублем id. Данные (chroma_db, logs, model_cache) не трогаем.
        for dir in bot scripts prompts knowledge; do
            if [ -d "$UPDATE_DIR/$dir" ]; then
                rm -rf "/home/container/$dir"
                cp -r "$UPDATE_DIR/$dir" /home/container/
            fi
        done

        # Байткод от прежней версии может конфликтовать с новой структурой.
        find /home/container -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

        rm -rf "$UPDATE_DIR"
        echo "Код загружен"
    else
        echo "Не удалось скачать обновления, использую код на диске"
    fi
else
    echo "Автообновление отключено (BOT_AUTO_UPDATE=false)"
fi

if [ ! -f "$START_SCRIPT" ]; then
    echo "Ошибка: не найден $START_SCRIPT" >&2
    echo "   Код бота не загружен. Проверьте доступность GitHub и" >&2
    echo "   значение BOT_AUTO_UPDATE (должно быть true при первом запуске)." >&2
    exit 1
fi

exec bash "$START_SCRIPT"
