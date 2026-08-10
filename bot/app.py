"""Сборка и запуск бота: клиент Discord, события, фоновые задачи."""

import asyncio
import signal
from pathlib import Path

import discord
from discord.ext import commands, tasks

from bot import incidents, settings, ticket_logs
from bot.commands import register_commands
from bot.handlers import MessageRouter
from bot.llm import SupportAgent, build_proxy_url, create_client
from bot.logging_setup import log_exception, logger
from bot.prompt import prompts
from bot.rag import KnowledgeIndexError, open_knowledge_index
from bot.state import store


def _create_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True

    # help_command=None: своя команда /help и !help заняли бы имя встроенной,
    # discord.py падает на конфликте при регистрации.
    kwargs = {
        "command_prefix": settings.COMMAND_PREFIX,
        "intents": intents,
        "help_command": None,
    }

    if not settings.USE_PROXY:
        return commands.Bot(**kwargs)

    import aiohttp

    if settings.PROXY_USERNAME and settings.PROXY_PASSWORD:
        kwargs["proxy_auth"] = aiohttp.BasicAuth(
            settings.PROXY_USERNAME, settings.PROXY_PASSWORD
        )
        # discord.py принимает credentials отдельно от URL.
        kwargs["proxy"] = f"http://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
    else:
        kwargs["proxy"] = build_proxy_url()

    logger.info("Discord через прокси %s:%s", settings.PROXY_HOST, settings.PROXY_PORT)
    return commands.Bot(**kwargs)


def build_application() -> commands.Bot:
    """Инициализирует всё, что нужно боту, и регистрирует обработчики."""
    ticket_logs.ensure_directories()

    # Каталог для инцидентов: команда /incident add не должна падать из-за
    # отсутствия data/ при локальном запуске без start.sh.
    try:
        Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log_exception("Не удалось создать каталог данных", exc, path=settings.DATA_DIR)

    active_incidents = incidents.active()
    if active_incidents:
        logger.info(
            "Активные инциденты: %s",
            ", ".join(item.id for item in active_incidents),
        )

    # Промпты читаем сразу: ошибка в файле должна остановить запуск, а не
    # всплыть на первом же вопросе игрока.
    prompts.system
    prompts.summary
    logger.info("Промпты загружены из %s", settings.PROMPTS_DIR)

    try:
        index = open_knowledge_index()
    except KnowledgeIndexError as exc:
        raise SystemExit(f"База знаний недоступна: {exc}") from exc

    agent = SupportAgent(create_client(), index)
    bot = _create_bot()
    router = MessageRouter(bot, agent)

    _register_events(bot, router)
    register_commands(bot, agent, router)
    return bot


async def _sync_command_tree(bot: commands.Bot) -> None:
    """Публикует slash-команды на всех серверах бота.

    Синхронизируем per-guild, а не глобально: глобальные команды Discord
    раскатывает до часа, гильдейные появляются сразу. Ошибка синхронизации
    не должна валить бота — префиксные команды остаются рабочими.
    """
    total = 0
    for guild in bot.guilds:
        try:
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
            total += len(synced)
            logger.info(
                "Slash-команды синхронизированы | guild=%s | команд=%s",
                guild.id,
                len(synced),
            )
        except discord.Forbidden as exc:
            logger.warning(
                "Нет прав на регистрацию slash-команд | guild=%s | error=%s\n"
                "   Боту не выдан scope applications.commands — переприглашите его "
                "по ссылке с этим scope. Префиксные команды продолжают работать.",
                guild.id,
                exc,
            )
        except Exception as exc:
            log_exception("Не удалось синхронизировать slash-команды", exc, guild_id=guild.id)

    if not bot.guilds:
        logger.warning("Бот не состоит ни в одной гильдии, slash-команды не зарегистрированы")
    else:
        logger.info("Всего slash-команд зарегистрировано: %s", total)


def _register_events(bot: commands.Bot, router: MessageRouter) -> None:

    @tasks.loop(seconds=max(settings.STATE_SAVE_INTERVAL_SECONDS, 5))
    async def persist_state_loop():
        store.save()

    @tasks.loop(hours=1)
    async def cleanup_state_loop():
        store.cleanup_expired()
        store.save()

    @tasks.loop(hours=max(settings.LOG_ARCHIVE_INTERVAL_HOURS, 1))
    async def archive_logs_loop():
        archive = ticket_logs.archive_orphaned_logs
        await asyncio.to_thread(archive)

    @tasks.loop(minutes=max(settings.REMINDER_CHECK_INTERVAL_MINUTES, 1))
    async def reminders_loop():
        try:
            await router.reminder_service.run_once()
        except Exception as exc:
            # Падение цикла tasks.loop останавливает его навсегда.
            log_exception("Ошибка прохода напоминаний", exc)

    @bot.event
    async def on_ready():
        logger.info("Бот запущен: %s", bot.user)
        store.load()
        
        from bot.commands import set_translator
        await set_translator(bot)
        await _sync_command_tree(bot)

        for loop in (persist_state_loop, cleanup_state_loop):
            if not loop.is_running():
                loop.start()
        if settings.LOG_ARCHIVE_ENABLED and not archive_logs_loop.is_running():
            archive_logs_loop.start()

        if settings.REMINDERS_ENABLED and not reminders_loop.is_running():
            if settings.REMINDER_PING_ROLE_IDS or settings.REMINDER_CATEGORY_OVERRIDES:
                reminders_loop.start()
                logger.info(
                    "Напоминания включены | проверка каждые %sмин | простой %sч | "
                    "повтор %sч | лимит %s/сутки | роли для пинга=%s",
                    settings.REMINDER_CHECK_INTERVAL_MINUTES,
                    settings.REMINDER_IDLE_HOURS,
                    settings.REMINDER_REPEAT_HOURS,
                    settings.REMINDER_MAX_PER_DAY,
                    settings.REMINDER_PING_ROLE_IDS or "заданы по категориям",
                )
            else:
                logger.warning(
                    "Напоминания включены, но [reminders].ping_role_ids пуст — "
                    "пинговать некого, цикл не запущен"
                )

        if settings.TICKET_CATEGORY_IDS:
            logger.info("Категории тикетов: %s", sorted(settings.TICKET_CATEGORY_IDS))
        else:
            logger.info("Категории тикетов: все каналы")
        if settings.BOT_ROLE_IDS:
            logger.info("Требуемые роли бота: %s", sorted(settings.BOT_ROLE_IDS))
        if settings.IGNORED_ROLE_IDS:
            logger.info("Игнорируемые роли: %s", sorted(settings.IGNORED_ROLE_IDS))
        logger.info("─────────────────────────")

    @bot.event
    async def on_message(message):
        await router.handle_message(message)

    @bot.event
    async def on_error(event_method, *args, **kwargs):
        logger.exception("Необработанная ошибка в событии Discord: %s", event_method)

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.MissingPermissions):
            return
        # Обычный текст, начинающийся с префикса, не должен спамить в логи.
        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.MissingRequiredArgument):
            from bot.discord_client import safe_send

            await safe_send(ctx.channel, f"Не хватает аргумента: {error.param.name}")
            return
        log_exception(
            "Ошибка выполнения команды",
            error,
            command=getattr(ctx.command, "qualified_name", "unknown"),
            channel_id=getattr(ctx.channel, "id", "unknown"),
            author=str(getattr(ctx, "author", "unknown")),
        )

    @bot.tree.error
    async def on_app_command_error(interaction: discord.Interaction, error):
        """Ошибки slash-команд.

        Отвечаем эфемерно: сообщение об ошибке — тоже служебное, игрокам
        в канале его видеть не нужно.
        """
        from bot.discord_client import reply_private

        if isinstance(error, discord.app_commands.MissingPermissions):
            await reply_private(interaction, "Команда доступна только администраторам.")
            return
        if isinstance(error, discord.app_commands.CommandOnCooldown):
            await reply_private(
                interaction, f"Команда на кулдауне, повторите через {error.retry_after:.0f}с."
            )
            return

        log_exception(
            "Ошибка выполнения slash-команды",
            error,
            command=getattr(getattr(interaction, "command", None), "qualified_name", "unknown"),
            channel_id=getattr(getattr(interaction, "channel", None), "id", "unknown"),
            author=str(getattr(interaction, "user", "unknown")),
        )
        await reply_private(interaction, "⚠️ Команда завершилась ошибкой. Подробности в логах.")

    @bot.event
    async def on_guild_channel_delete(channel):
        await asyncio.to_thread(ticket_logs.archive_closed_ticket, channel)
        if store.remove(getattr(channel, "id", None)):
            store.save(force=True)
            logger.info(
                "Состояние удалённого канала очищено | channel_id=%s",
                getattr(channel, "id", "unknown"),
            )


def _install_shutdown_handlers(bot: commands.Bot) -> None:
    """Сохраняет состояние при SIGTERM.

    Pterodactyl останавливает контейнер сигналом, и без этого обработчика
    терялись изменения, сделанные после последнего периодического сохранения:
    ручные !stop и свежие эскалации.
    """
    loop = asyncio.get_running_loop()

    async def shutdown(sig_name: str) -> None:
        logger.info("Получен сигнал %s, сохраняю состояние и останавливаюсь", sig_name)
        store.save(force=True)
        await bot.close()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, lambda s=sig_name: asyncio.create_task(shutdown(s)))
        except (NotImplementedError, RuntimeError):
            # Windows не поддерживает add_signal_handler для этих сигналов.
            pass


def run() -> None:
    bot = build_application()

    async def main() -> None:
        _install_shutdown_handlers(bot)
        async with bot:
            await bot.start(settings.DISCORD_TOKEN)

    if settings.USE_PROXY:
        logger.info("Прокси включён: %s:%s", settings.PROXY_HOST, settings.PROXY_PORT)
    else:
        logger.info("Прокси выключен")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    except Exception as exc:
        log_exception("Критическая ошибка работы бота", exc)
        raise
    finally:
        store.save(force=True)
        logger.info("Состояние сохранено, бот остановлен")
