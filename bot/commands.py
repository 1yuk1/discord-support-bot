"""Административные команды бота."""

import time
from datetime import datetime

from discord.ext import commands

from bot import settings
from bot.discord_client import safe_send
from bot.llm import models
from bot.logging_setup import log_exception, logger
from bot.filters import extract_message_text
from bot.state import store

_SUMMARY_TRANSCRIPT_LIMIT = 12000
_SUMMARY_MIN_MESSAGES = 10
_SUMMARY_MAX_MESSAGES = 150


def save_model_to_settings(model_name: str) -> None:
    """Записывает модель в секцию [ai.openrouter] файла settings.toml.

    Правка построчная, чтобы сохранить комментарии и порядок ключей.
    """
    section_header = "[ai.openrouter]"
    path = settings.SETTINGS_PATH

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_line = f'model = "{model_name}"\n'
    in_section = False
    section_found = False
    updated = False

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            # Дошли до следующей секции, а ключ так и не встретили — вставляем.
            if in_section and not updated:
                lines.insert(index, new_line)
                updated = True
                break
            in_section = stripped == section_header
            section_found = section_found or in_section
            continue

        if in_section and stripped.startswith("model ="):
            lines[index] = new_line
            updated = True
            break

    if not section_found:
        raise ValueError(f"Секция {section_header} не найдена в {path}")

    if not updated:
        lines.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def register_commands(bot, agent) -> None:
    """Регистрирует все команды на переданном экземпляре бота."""

    @bot.command(name="stop", aliases=["mute", "silence"])
    @commands.has_permissions(administrator=True)
    async def stop_in_channel(ctx):
        """Выключает ответы бота в текущем канале."""
        state = store.get_or_create(ctx.channel.id)

        if state.get("bot_disabled"):
            await safe_send(
                ctx.channel,
                f"Бот уже выключен в этом канале. Включить обратно: "
                f"{settings.COMMAND_PREFIX}start",
            )
            return

        state["bot_disabled"] = True
        state["disabled_by"] = str(ctx.author)
        state["disabled_at"] = time.time()
        # Незавершённая склейка иначе выстрелит ответом уже после выключения.
        state["pending_parts"] = []
        task = state.get("debounce_task")
        if task is not None and not task.done():
            task.cancel()
        state["debounce_task"] = None

        store.touch(state)
        store.mark_dirty()
        store.save(force=True)

        logger.info(
            "Бот выключен в канале | channel_id=%s | author=%s", ctx.channel.id, ctx.author
        )
        await safe_send(
            ctx.channel,
            f"🔇 Бот выключен в этом канале и больше не будет отвечать.\n"
            f"Включить обратно: {settings.COMMAND_PREFIX}start",
        )

    @bot.command(name="start", aliases=["unmute"])
    @commands.has_permissions(administrator=True)
    async def start_in_channel(ctx):
        """Включает ответы бота в текущем канале."""
        state = store.get_or_create(ctx.channel.id)

        if not state.get("bot_disabled"):
            await safe_send(ctx.channel, "Бот и так включён в этом канале.")
            return

        state["bot_disabled"] = False
        state["disabled_by"] = None
        state["disabled_at"] = None
        store.touch(state)
        store.mark_dirty()
        store.save(force=True)

        logger.info(
            "Бот включён в канале | channel_id=%s | author=%s", ctx.channel.id, ctx.author
        )

        # human_mode — отдельный флаг: !start не должен отменять эскалацию.
        if state.get("human_mode"):
            await safe_send(
                ctx.channel,
                f"🔈 Бот включён, но тикет передан человеку, поэтому ответов пока не будет.\n"
                f"Вернуть бота в диалог: {settings.COMMAND_PREFIX}resume_bot",
            )
        else:
            await safe_send(ctx.channel, "🔈 Бот включён и снова отвечает в этом канале.")

    @bot.command(name="clear_history")
    @commands.has_permissions(administrator=True)
    async def clear_history(ctx):
        """Очищает историю диалога, сохраняя ручное отключение бота."""
        if ctx.channel.id not in store:
            await safe_send(ctx.channel, "История пуста")
            return

        store.reset(ctx.channel.id)
        store.save(force=True)
        await safe_send(ctx.channel, "✅ История диалога очищена")

    @bot.command(name="resume_bot")
    @commands.has_permissions(administrator=True)
    async def resume_bot(ctx):
        """Снимает режим передачи человеку."""
        state = store.get(ctx.channel.id)
        if state is None:
            await safe_send(ctx.channel, "Нет данных о канале")
            return

        state["human_mode"] = False
        store.touch(state)
        store.mark_dirty()
        store.save(force=True)

        if state.get("bot_disabled"):
            await safe_send(
                ctx.channel,
                f"✅ Режим передачи человеку снят, но бот выключен командой "
                f"{settings.COMMAND_PREFIX}stop. Включить: {settings.COMMAND_PREFIX}start",
            )
        else:
            await safe_send(ctx.channel, "✅ Бот возобновил работу")

    @bot.command(name="bot_status")
    @commands.has_permissions(administrator=True)
    async def bot_status(ctx):
        """Показывает состояние бота в текущем канале."""
        state = store.get(ctx.channel.id)
        if state is None:
            await safe_send(ctx.channel, "Нет данных о канале")
            return

        last_activity = state.get("last_activity")
        last_activity_text = (
            datetime.fromtimestamp(last_activity).strftime("%Y-%m-%d %H:%M:%S")
            if last_activity
            else "неизвестно"
        )

        if state.get("bot_disabled"):
            answering = f"нет, выключен командой {settings.COMMAND_PREFIX}stop"
        elif state.get("human_mode"):
            answering = "нет, тикет передан человеку"
        else:
            answering = "да"

        lines = [
            f"отвечает: {answering}",
            f"human_mode: {state.get('human_mode')}",
            f"выключен вручную: {state.get('bot_disabled')}",
        ]
        if state.get("bot_disabled") and state.get("disabled_by"):
            lines.append(f"выключил: {state['disabled_by']}")
        lines += [
            f"модель: {models.get()}",
            f"сообщений в истории: {len(state.get('history', []))}",
            f"последняя активность: {last_activity_text}",
        ]
        await safe_send(ctx.channel, "\n".join(lines))

    @bot.command(name="summarize")
    @commands.has_permissions(administrator=True)
    async def summarize_ticket(ctx, limit: int = 80):
        """Делает сводку тикета для администратора."""
        limit = max(_SUMMARY_MIN_MESSAGES, min(limit, _SUMMARY_MAX_MESSAGES))
        lines: list[str] = []

        try:
            async for msg in ctx.channel.history(limit=limit, oldest_first=True):
                if msg.id == ctx.message.id:
                    continue
                text = extract_message_text(msg)
                if not text:
                    continue
                author = getattr(msg.author, "display_name", str(msg.author))
                lines.append(f"{author}: {text}")
        except Exception as exc:
            log_exception("Не удалось прочитать историю канала", exc, channel_id=ctx.channel.id)
            await safe_send(ctx.channel, "⚠️ Не удалось прочитать историю канала.")
            return

        if not lines:
            await safe_send(ctx.channel, "В канале пока нет сообщений для сводки.")
            return

        transcript = "\n".join(lines)[-_SUMMARY_TRANSCRIPT_LIMIT:]

        try:
            async with ctx.channel.typing():
                summary = agent.summarize_ticket(transcript)
        except Exception as exc:
            log_exception("Не удалось сгенерировать сводку", exc, channel_id=ctx.channel.id)
            await safe_send(ctx.channel, "⚠️ Не удалось сделать сводку. Подробности в логах.")
            return

        await safe_send(ctx.channel, f"Сводка тикета:\n{summary}")

    @bot.group(name="model", invoke_without_command=True)
    @commands.has_permissions(administrator=True)
    async def model_group(ctx):
        """Показывает текущую модель."""
        await safe_send(
            ctx.channel,
            f"Провайдер: openrouter\nТекущая модель: {models.get()}",
        )

    @model_group.command(name="set")
    @commands.has_permissions(administrator=True)
    async def model_set(ctx, *, model_name: str):
        """Меняет модель до перезапуска."""
        model_name = model_name.strip()
        if not model_name:
            await safe_send(
                ctx.channel, f"Укажите модель: {settings.COMMAND_PREFIX}model set <model_name>"
            )
            return

        previous = models.set(model_name)
        logger.info(
            "Модель изменена в runtime | previous=%s | current=%s | author=%s",
            previous,
            model_name,
            ctx.author,
        )
        await safe_send(
            ctx.channel,
            f"✅ Модель применена без рестарта.\n"
            f"Была: {previous}\nСтала: {model_name}",
        )

    @model_group.command(name="save")
    @commands.has_permissions(administrator=True)
    async def model_save(ctx, *, model_name: str):
        """Меняет модель и сохраняет её в settings.toml."""
        model_name = model_name.strip()
        if not model_name:
            await safe_send(
                ctx.channel, f"Укажите модель: {settings.COMMAND_PREFIX}model save <model_name>"
            )
            return

        previous = models.get()
        try:
            models.set(model_name)
            save_model_to_settings(model_name)
        except Exception as exc:
            models.set(previous)
            log_exception(
                "Не удалось сохранить модель в settings.toml",
                exc,
                requested_model=model_name,
                author=str(ctx.author),
            )
            await safe_send(
                ctx.channel, "⚠️ Не удалось сохранить модель. Подробности в логах."
            )
            return

        logger.info(
            "Модель сохранена | previous=%s | current=%s | author=%s",
            previous,
            model_name,
            ctx.author,
        )
        await safe_send(
            ctx.channel,
            f"✅ Модель применена и сохранена.\nБыла: {previous}\nСтала: {model_name}",
        )

    @bot.command(name="ping")
    async def ping(ctx):
        """Проверка задержки до Discord."""
        await safe_send(ctx.channel, f"Pong! Задержка: {round(bot.latency * 1000)}ms")
