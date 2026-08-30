"""Административные команды бота.

Все команды — slash (`/stop`, `/start`, ...), потому что только interaction
умеет эфемерные ответы: сообщение видит лишь тот, кто вызвал команду.
Префиксные `!`-команды сохранены как аварийный путь на случай, если синхронизация
дерева команд не прошла (боту не выдали scope applications.commands) — иначе
управлять ботом было бы нечем. Они отвечают в канал, всем видно.

Логика каждой команды живёт в функции `_do_*`, возвращающей текст ответа, а
slash- и префиксная обёртки только доставляют его разными способами.
"""

import asyncio
import time
from datetime import datetime

import discord
from discord import app_commands
from discord.ext import commands

from bot import incidents, reminders, settings
from bot.discord_client import reply_private, safe_send
from bot.llm import fallback_models, models
from bot.logging_setup import log_exception, logger
from bot.filters import extract_message_text, is_admin_member
from bot.metrics import metrics
from bot.prompt import prompts
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


# ── Логика команд ────────────────────────────────────────────────────────────
# Каждая функция возвращает текст ответа и не знает, как он будет доставлен.


def _do_stop(channel, author) -> str:
    state = store.get_or_create(channel.id)

    if state.get("bot_disabled"):
        return (
            f"Бот уже выключен в этом канале. "
            f"Включить обратно: /start"
        )

    state["bot_disabled"] = True
    state["disabled_by"] = str(author)
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

    logger.info("Бот выключен в канале | channel_id=%s | author=%s", channel.id, author)
    return (
        "🔇 Бот выключен в этом канале и больше не будет отвечать.\n"
        "Включить обратно: /start"
    )


def _do_start(channel, author) -> str:
    state = store.get_or_create(channel.id)

    if not state.get("bot_disabled"):
        return "Бот и так включён в этом канале."

    state["bot_disabled"] = False
    state["disabled_by"] = None
    state["disabled_at"] = None
    store.touch(state)
    store.mark_dirty()
    store.save(force=True)

    logger.info("Бот включён в канале | channel_id=%s | author=%s", channel.id, author)

    # human_mode — отдельный флаг: /start не должен отменять эскалацию.
    if state.get("human_mode"):
        return (
            "🔈 Бот включён, но тикет передан человеку, поэтому ответов пока не будет.\n"
            "Вернуть бота в диалог: /resume_bot"
        )
    return "🔈 Бот включён и снова отвечает в этом канале."


def _do_clear_history(channel) -> str:
    if channel.id not in store:
        return "История пуста"

    store.reset(channel.id)
    store.save(force=True)
    return "✅ История диалога очищена"


def _do_resume_bot(channel) -> str:
    state = store.get(channel.id)
    if state is None:
        return "Нет данных о канале"

    state["human_mode"] = False
    store.touch(state)
    store.mark_dirty()
    store.save(force=True)

    if state.get("bot_disabled"):
        return (
            "✅ Режим передачи человеку снят, но бот выключен командой "
            "/stop. Включить: /start"
        )
    return "✅ Бот возобновил работу"


def _do_status(channel) -> str:
    state = store.get(channel.id)
    if state is None:
        return "Нет данных о канале"

    last_activity = state.get("last_activity")
    last_activity_text = (
        datetime.fromtimestamp(last_activity).strftime("%Y-%m-%d %H:%M:%S")
        if last_activity
        else "неизвестно"
    )

    if state.get("bot_disabled"):
        answering = "нет, выключен командой /stop"
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
    return "\n".join(lines)


async def _collect_transcript(channel, limit: int, exclude_message_id=None) -> str | None:
    """Собирает историю канала для сводки. None — не удалось прочитать."""
    lines: list[str] = []
    try:
        async for msg in channel.history(limit=limit, oldest_first=True):
            if exclude_message_id is not None and msg.id == exclude_message_id:
                continue
            text = extract_message_text(msg)
            if not text:
                continue
            author = getattr(msg.author, "display_name", str(msg.author))
            lines.append(f"{author}: {text}")
    except Exception as exc:
        log_exception("Не удалось прочитать историю канала", exc, channel_id=channel.id)
        return None

    if not lines:
        return ""
    return "\n".join(lines)[-_SUMMARY_TRANSCRIPT_LIMIT:]


def _do_model_show() -> str:
    lines = [f"Основной провайдер: openrouter\nМодель: {models.get()}"]
    if settings.FALLBACK_AI_ENABLED:
        lines.append(f"Резервный AI-провайдер\nМодель: {fallback_models.get()}")
    else:
        lines.append("Резервный провайдер: выключен")
    return "\n\n".join(lines)


def _do_model_set(model_name: str, author) -> str:
    model_name = (model_name or "").strip()
    if not model_name:
        return "Укажите модель: /model set <model_name>"

    previous = models.set(model_name)
    logger.info(
        "Модель изменена в runtime | previous=%s | current=%s | author=%s",
        previous,
        model_name,
        author,
    )
    return (
        f"✅ Модель применена без рестарта.\n"
        f"Была: {previous}\nСтала: {model_name}"
    )


def _do_model_save(model_name: str, author) -> str:
    model_name = (model_name or "").strip()
    if not model_name:
        return "Укажите модель: /model save <model_name>"

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
            author=str(author),
        )
        return "⚠️ Не удалось сохранить модель. Подробности в логах."

    logger.info(
        "Модель сохранена | previous=%s | current=%s | author=%s",
        previous,
        model_name,
        author,
    )
    return f"✅ Модель применена и сохранена.\nБыла: {previous}\nСтала: {model_name}"


def _do_reminders_toggle(channel, enabled: bool) -> str:
    state = store.get_or_create(channel.id)
    state["reminders_disabled"] = not enabled
    store.touch(state)
    store.mark_dirty()
    store.save(force=True)

    if enabled:
        return "🔔 Напоминания в этом канале включены."
    return (
        "🔕 Напоминания в этом канале выключены.\n"
        "Полезно, когда решение объективно затягивается. Вернуть: /reminders on"
    )


def _do_reminders_status(channel) -> str:
    category_id = getattr(channel, "category_id", None)
    config = settings.reminder_config_for(category_id)
    state = store.get(channel.id)

    lines = []
    if not settings.REMINDERS_ENABLED:
        lines.append("глобально: выключены")
    elif not config.get("enabled"):
        lines.append("в этой категории: выключены")
    elif not config.get("ping_role_ids"):
        lines.append("не работают: не задан ping_role_ids для этой категории")
    elif state is not None and state.get("reminders_disabled"):
        lines.append("в этом канале: выключены командой /reminders off")
    else:
        lines.append("в этом канале: включены")

    lines += [
        f"порог простоя: {config.get('idle_hours')}ч",
        f"повтор не чаще: {config.get('repeat_hours')}ч",
        f"лимит в сутки: {config.get('max_per_day')}",
        f"режим текста: {config.get('message_mode')}",
        f"роли для пинга: {config.get('ping_role_ids') or 'не заданы'}",
        f"роли персонала: {config.get('staff_role_ids') or 'не заданы'}",
    ]

    guild = getattr(channel, "guild", None)
    ping_roles = config.get("ping_role_ids") or []
    if guild is not None and ping_roles:
        valid_roles = [r for r in ping_roles if guild.get_role(r) is not None]
        if not valid_roles:
            lines.append("⚠️ внимание: ни одна из ролей для пинга не найдена на этом сервере Discord")

    if state is not None:
        started = reminders.waiting_since(state)
        if started:
            hours = (time.time() - started) / 3600
            lines.append(f"игрок ждёт: {hours:.1f}ч")
        else:
            lines.append("игрок ждёт: нет (последним отвечал персонал)")
        lines.append(f"отправлено за сутки: {state.get('reminder_count_today', 0)}")

    return "\n".join(lines)


def _do_incident_add(title: str, text: str, author) -> str:
    try:
        incident = incidents.add(title, text, author=str(author))
    except ValueError as exc:
        return f"⚠️ {exc}"
    except OSError as exc:
        log_exception("Не удалось записать инцидент", exc, title=title)
        return "⚠️ Не удалось сохранить инцидент. Подробности в логах."

    return (
        f"✅ Инцидент добавлен: **{incident.title}**\n"
        f"id: `{incident.id}`\n\n"
        f"Бот учтёт его в ответах сразу, без рестарта.\n"
        f"Удалить, когда проблема решится: /incident remove"
    )


def _do_incident_remove(incident_id: str) -> str:
    try:
        removed = incidents.remove(incident_id)
    except OSError as exc:
        log_exception("Не удалось удалить инцидент", exc, incident_id=incident_id)
        return "⚠️ Не удалось удалить инцидент. Подробности в логах."

    if removed is None:
        return f"Инцидент `{incident_id}` не найден. Список: /incident list"
    return f"✅ Инцидент удалён: **{removed.title}**"


def _do_incident_list() -> str:
    active = incidents.active()
    if not active:
        return (
            "Активных инцидентов нет.\n"
            "Добавить: /incident add заголовок текст"
        )

    lines = [f"**Активных инцидентов: {len(active)}**", ""]
    for incident in active:
        lines.append(f"**{incident.title}**")
        lines.append(f"id: `{incident.id}`")
        if incident.created_at:
            created = f"создан: {incident.created_at}"
            if incident.author:
                created += f", автор: {incident.author}"
            lines.append(created)
        # Длинный текст в списке не нужен: он и так уходит в промпт целиком.
        preview = " ".join(incident.body.split())
        lines.append(preview[:200] + ("…" if len(preview) > 200 else ""))
        lines.append("")
    return "\n".join(lines)


def _shorten(value, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _do_config_reload(router) -> str:
    """Перечитывает settings.toml и раздаёт обновления подписчикам."""
    try:
        changes = settings.reload()
    except Exception as exc:
        log_exception("Не удалось перечитать settings.toml", exc)
        return (
            "⚠️ Не удалось перечитать settings.toml — прежние настройки сохранены.\n"
            f"Причина: {exc}"
        )

    # Роутер держит копии id в множествах, промпты и инциденты кэшируются.
    if router is not None:
        router.refresh_settings()

    notes: list[str] = []

    if "OPENROUTER_MODEL" in changes:
        models.set(settings.OPENROUTER_MODEL)

    try:
        prompts.reload()
        incidents.store.invalidate()
    except Exception as exc:
        log_exception("Не удалось перечитать промпты", exc)
        notes.append("⚠️ промпты перечитать не удалось, остались прежние")

    if not changes:
        lines = ["Настройки перечитаны, изменений нет."]
    else:
        lines = [f"✅ Настройки перечитаны, изменено: {len(changes)}", ""]
        for name in sorted(changes):
            previous, current = changes[name]
            lines.append(f"`{name}`\n  было: {_shorten(previous)}\n  стало: {_shorten(current)}")

    lines += notes
    lines += [
        "",
        "Требуют рестарта: " + ", ".join(settings.RESTART_REQUIRED_KEYS),
    ]
    return "\n".join(lines)


def _do_tokens() -> str:
    """Возвращает форматированный отчёт по расходу токенов AI."""
    return metrics.format_report()


def is_bot_admin_check(ctx) -> bool:
    """Проверка прав администратора для команд (Discord Admin, admin_role_ids, admin_user_ids)."""
    member = getattr(ctx, "author", ctx)
    guild = getattr(ctx, "guild", getattr(member, "guild", None))
    guild_cfg = settings.get_guild_config(getattr(guild, "id", None) if guild else None)
    return is_admin_member(
        member,
        admin_role_ids=guild_cfg.admin_role_ids,
        admin_user_ids=guild_cfg.admin_user_ids,
    )


# ── Русские имена slash-команд ───────────────────────────────────────────────
# Discord разрешает Unicode в именах команд, но у одной команды может быть
# только одно имя на локаль. Поэтому русские варианты отдаются через
# локализацию: клиент с русским языком видит /стоп, остальные — /stop.
# Ключ — оригинальное английское имя команды, группы или параметра.
_RU_COMMAND_NAMES: dict[str, str] = {
    # Команды и группы
    "stop": "стоп",
    "start": "старт",
    "status": "статус",
    "clear_history": "очистить-историю",
    "resume_bot": "вернуть-бота",
    "summarize": "сводка",
    "tokens": "токены",
    "metrics": "метрики",
    "model": "модель",
    "show": "показать",
    "set": "сменить",
    "save": "сохранить",
    "reminders": "напоминания",
    "on": "включить",
    "off": "выключить",
    "incident": "инцидент",
    "add": "добавить",
    "list": "список",
    "remove": "удалить",
    "config": "настройки",
    "reload": "перезагрузить",
    "ping": "пинг",
    "help": "помощь",
    # Параметры
    "limit": "лимит",
    "title": "заголовок",
    "text": "текст",
}

_TRANSLATED_LOCATIONS = frozenset({
    app_commands.TranslationContextLocation.command_name,
    app_commands.TranslationContextLocation.group_name,
    app_commands.TranslationContextLocation.parameter_name,
})


class RussianCommandTranslator(app_commands.Translator):
    """Русские имена команд для клиентов с русской локалью.

    Описания не переводим: они и так написаны по-русски в объявлениях команд.
    Для остальных локалей возвращаем None — там остаются английские имена.
    """

    async def translate(
        self,
        string: app_commands.locale_str,
        locale: discord.Locale,
        context: app_commands.TranslationContext,
    ) -> str | None:
        if locale is not discord.Locale.russian:
            return None
        if context.location not in _TRANSLATED_LOCATIONS:
            return None
        return _RU_COMMAND_NAMES.get(str(string).lower())


_HELP_TEXT = """**Команды администратора**

Управление в канале:
`/stop` — выключить бота в этом канале
`/start` — включить обратно
`/status` — состояние бота в канале
`/resume_bot` — снять режим передачи человеку
`/clear_history` — очистить историю диалога
`/summarize [limit]` — сводка тикета

Напоминания персоналу:
`/reminders status` — настройки и сколько игрок уже ждёт
`/reminders off` — выключить в этом канале
`/reminders on` — включить обратно

Инциденты (известные проблемы):
`/incident add <заголовок> <текст>` — бот учтёт сразу, без рестарта
`/incident list` — активные инциденты
`/incident remove <id>` — убрать, когда проблема решена

Модель AI и метрики:
`/tokens` — статистика расхода токенов AI
`/model show` — текущая модель
`/model set <model>` — сменить до рестарта
`/model save <model>` — сменить и записать в settings.toml

Прочее:
`/config reload` — применить правки settings.toml без рестарта
`/ping` — задержка до Discord
`/help` — эта справка

Все ответы на эти команды видны только вам."""


def register_commands(bot, agent, router=None) -> None:
    """Регистрирует slash-команды и префиксные дубли.

    router нужен только для /config reload: он держит копии id категорий и
    ролей, которые надо обновить после перечитывания файла.
    """
    _register_slash_commands(bot, agent, router)
    _register_prefix_commands(bot, agent, router)


# ── Slash-команды ────────────────────────────────────────────────────────────
def _register_slash_commands(bot, agent, router=None) -> None:
    tree = bot.tree
    admin_only = app_commands.default_permissions(administrator=True)

    @tree.command(name="stop", description="Выключить бота в этом канале")
    @admin_only
    async def slash_stop(interaction: discord.Interaction):
        await reply_private(interaction, _do_stop(interaction.channel, interaction.user))

    @tree.command(name="start", description="Включить бота в этом канале")
    @admin_only
    async def slash_start(interaction: discord.Interaction):
        await reply_private(interaction, _do_start(interaction.channel, interaction.user))

    @tree.command(name="status", description="Состояние бота в этом канале")
    @admin_only
    async def slash_status(interaction: discord.Interaction):
        await reply_private(interaction, _do_status(interaction.channel))

    @tree.command(name="clear_history", description="Очистить историю диалога в канале")
    @admin_only
    async def slash_clear_history(interaction: discord.Interaction):
        await reply_private(interaction, _do_clear_history(interaction.channel))

    @tree.command(name="resume_bot", description="Снять режим передачи человеку")
    @admin_only
    async def slash_resume_bot(interaction: discord.Interaction):
        await reply_private(interaction, _do_resume_bot(interaction.channel))

    @tree.command(name="summarize", description="Сводка тикета для администратора")
    @app_commands.describe(limit="Сколько сообщений прочитать (10-150)")
    @admin_only
    async def slash_summarize(interaction: discord.Interaction, limit: int = 80):
        limit = max(_SUMMARY_MIN_MESSAGES, min(limit, _SUMMARY_MAX_MESSAGES))
        # Сводка идёт в LLM и легко превышает 3 секунды, отведённые Discord
        # на ответ, поэтому сначала откладываем interaction.
        await interaction.response.defer(ephemeral=True)

        transcript = await _collect_transcript(interaction.channel, limit)
        if transcript is None:
            await reply_private(interaction, "⚠️ Не удалось прочитать историю канала.")
            return
        if not transcript:
            await reply_private(interaction, "В канале пока нет сообщений для сводки.")
            return

        try:
            summary = await asyncio.to_thread(agent.summarize_ticket, transcript)
        except Exception as exc:
            log_exception(
                "Не удалось сгенерировать сводку", exc, channel_id=interaction.channel.id
            )
            await reply_private(interaction, "⚠️ Не удалось сделать сводку. Подробности в логах.")
            return

        await reply_private(interaction, f"Сводка тикета:\n{summary}")

    model_group = app_commands.Group(
        name="model",
        description="Управление моделью AI",
        default_permissions=discord.Permissions(administrator=True),
    )

    @model_group.command(name="show", description="Показать модели провайдеров")
    async def slash_model_show(interaction: discord.Interaction):
        await reply_private(interaction, _do_model_show())

    @model_group.command(name="set", description="Сменить основную модель до рестарта")
    @app_commands.describe(model="Идентификатор модели OpenRouter")
    async def slash_model_set(interaction: discord.Interaction, model: str):
        await reply_private(interaction, _do_model_set(model, interaction.user))

    @model_group.command(name="save", description="Сменить основную модель и записать в settings.toml")
    @app_commands.describe(model="Идентификатор модели OpenRouter")
    async def slash_model_save(interaction: discord.Interaction, model: str):
        await reply_private(interaction, _do_model_save(model, interaction.user))

    tree.add_command(model_group)

    reminders_group = app_commands.Group(
        name="reminders",
        description="Напоминания персоналу о забытых тикетах",
        default_permissions=discord.Permissions(administrator=True),
    )

    @reminders_group.command(name="status", description="Показать настройки напоминаний")
    async def slash_reminders_status(interaction: discord.Interaction):
        await reply_private(interaction, _do_reminders_status(interaction.channel))

    @reminders_group.command(name="off", description="Выключить напоминания в этом канале")
    async def slash_reminders_off(interaction: discord.Interaction):
        await reply_private(interaction, _do_reminders_toggle(interaction.channel, False))

    @reminders_group.command(name="on", description="Включить напоминания в этом канале")
    async def slash_reminders_on(interaction: discord.Interaction):
        await reply_private(interaction, _do_reminders_toggle(interaction.channel, True))

    tree.add_command(reminders_group)

    incident_group = app_commands.Group(
        name="incident",
        description="Известные проблемы, которые бот учитывает в ответах",
        default_permissions=discord.Permissions(administrator=True),
    )

    async def incident_id_autocomplete(interaction: discord.Interaction, current: str):
        """Подсказки для /incident remove: копировать id руками не нужно."""
        current = (current or "").lower()
        choices = []
        for incident in incidents.active():
            if current and current not in incident.id.lower() \
                    and current not in incident.title.lower():
                continue
            # Лимит Discord на подпись варианта — 100 символов.
            label = f"{incident.title} [{incident.id}]"[:100]
            choices.append(app_commands.Choice(name=label, value=incident.id))
        return choices[:25]

    @incident_group.command(name="add", description="Добавить активный инцидент")
    @app_commands.describe(
        title="Короткий заголовок, например: Не работает вход",
        text="Что отвечать игрокам по этой проблеме",
    )
    async def slash_incident_add(interaction: discord.Interaction, title: str, text: str):
        await reply_private(interaction, _do_incident_add(title, text, interaction.user))

    @incident_group.command(name="list", description="Показать активные инциденты")
    async def slash_incident_list(interaction: discord.Interaction):
        await reply_private(interaction, _do_incident_list())

    @incident_group.command(name="remove", description="Удалить инцидент")
    @app_commands.describe(incident="Выберите инцидент из списка")
    @app_commands.autocomplete(incident=incident_id_autocomplete)
    async def slash_incident_remove(interaction: discord.Interaction, incident: str):
        await reply_private(interaction, _do_incident_remove(incident))

    tree.add_command(incident_group)

    config_group = app_commands.Group(
        name="config",
        description="Настройки бота",
        default_permissions=discord.Permissions(administrator=True),
    )

    @config_group.command(
        name="reload", description="Применить правки settings.toml без рестарта"
    )
    async def slash_config_reload(interaction: discord.Interaction):
        await reply_private(interaction, _do_config_reload(router))

    tree.add_command(config_group)

    @tree.command(name="tokens", description="Статистика расхода токенов AI")
    @admin_only
    async def slash_tokens(interaction: discord.Interaction):
        await reply_private(interaction, _do_tokens())

    @tree.command(name="metrics", description="Статистика расхода токенов AI")
    @admin_only
    async def slash_metrics(interaction: discord.Interaction):
        await reply_private(interaction, _do_tokens())

    @tree.command(name="ping", description="Проверить задержку до Discord")
    @admin_only
    async def slash_ping(interaction: discord.Interaction):
        await reply_private(interaction, f"Pong! Задержка: {round(bot.latency * 1000)}ms")

    @tree.command(name="help", description="Список команд администратора")
    @admin_only
    async def slash_help(interaction: discord.Interaction):
        await reply_private(interaction, _HELP_TEXT)


# ── Префиксные команды (аварийный путь) ──────────────────────────────────────
def _register_prefix_commands(bot, agent, router=None) -> None:
    """Дубли на случай, если slash-команды не синхронизировались.

    Ответы здесь видны всем в канале — иначе никак, эфемерность доступна
    только для interaction.
    """

    @bot.command(name="tokens", aliases=["токены", "метрики", "metrics"])
    @commands.check(is_bot_admin_check)
    async def token_metrics(ctx):
        await safe_send(ctx.channel, _do_tokens())

    @bot.command(name="stop", aliases=["mute", "silence", "стоп", "выкл"])
    @commands.check(is_bot_admin_check)
    async def stop_in_channel(ctx):
        await safe_send(ctx.channel, _do_stop(ctx.channel, ctx.author))

    @bot.command(name="start", aliases=["unmute", "старт", "вкл"])
    @commands.check(is_bot_admin_check)
    async def start_in_channel(ctx):
        await safe_send(ctx.channel, _do_start(ctx.channel, ctx.author))

    @bot.command(name="clear_history", aliases=["очистить_историю", "сброс"])
    @commands.check(is_bot_admin_check)
    async def clear_history(ctx):
        await safe_send(ctx.channel, _do_clear_history(ctx.channel))

    @bot.command(name="resume_bot", aliases=["вернуть_бота", "вернуть"])
    @commands.check(is_bot_admin_check)
    async def resume_bot(ctx):
        await safe_send(ctx.channel, _do_resume_bot(ctx.channel))

    @bot.command(name="bot_status", aliases=["статус"])
    @commands.check(is_bot_admin_check)
    async def bot_status(ctx):
        await safe_send(ctx.channel, _do_status(ctx.channel))

    @bot.command(name="summarize", aliases=["сводка"])
    @commands.check(is_bot_admin_check)
    async def summarize_ticket(ctx, limit: int = 80):
        limit = max(_SUMMARY_MIN_MESSAGES, min(limit, _SUMMARY_MAX_MESSAGES))
        transcript = await _collect_transcript(ctx.channel, limit, ctx.message.id)

        if transcript is None:
            await safe_send(ctx.channel, "⚠️ Не удалось прочитать историю канала.")
            return
        if not transcript:
            await safe_send(ctx.channel, "В канале пока нет сообщений для сводки.")
            return

        try:
            async with ctx.channel.typing():
                summary = await asyncio.to_thread(agent.summarize_ticket, transcript)
        except Exception as exc:
            log_exception("Не удалось сгенерировать сводку", exc, channel_id=ctx.channel.id)
            await safe_send(ctx.channel, "⚠️ Не удалось сделать сводку. Подробности в логах.")
            return

        await safe_send(ctx.channel, f"Сводка тикета:\n{summary}")

    @bot.group(name="model", aliases=["модель"], invoke_without_command=True)
    @commands.check(is_bot_admin_check)
    async def model_group(ctx):
        await safe_send(ctx.channel, _do_model_show())

    @model_group.command(name="set", aliases=["сменить"])
    @commands.check(is_bot_admin_check)
    async def model_set(ctx, *, model_name: str):
        await safe_send(ctx.channel, _do_model_set(model_name, ctx.author))

    @model_group.command(name="save", aliases=["сохранить"])
    @commands.check(is_bot_admin_check)
    async def model_save(ctx, *, model_name: str):
        await safe_send(ctx.channel, _do_model_save(model_name, ctx.author))

    @bot.group(name="reminders", aliases=["напоминания"], invoke_without_command=True)
    @commands.check(is_bot_admin_check)
    async def reminders_group(ctx):
        await safe_send(ctx.channel, _do_reminders_status(ctx.channel))

    @reminders_group.command(name="status", aliases=["статус"])
    @commands.check(is_bot_admin_check)
    async def reminders_status(ctx):
        await safe_send(ctx.channel, _do_reminders_status(ctx.channel))

    @reminders_group.command(name="off", aliases=["выкл"])
    @commands.check(is_bot_admin_check)
    async def reminders_off(ctx):
        await safe_send(ctx.channel, _do_reminders_toggle(ctx.channel, False))

    @reminders_group.command(name="on", aliases=["вкл"])
    @commands.check(is_bot_admin_check)
    async def reminders_on(ctx):
        await safe_send(ctx.channel, _do_reminders_toggle(ctx.channel, True))

    @bot.group(name="incident", aliases=["инцидент"], invoke_without_command=True)
    @commands.check(is_bot_admin_check)
    async def incident_group(ctx):
        await safe_send(ctx.channel, _do_incident_list())

    @incident_group.command(name="list", aliases=["список"])
    @commands.check(is_bot_admin_check)
    async def incident_list(ctx):
        await safe_send(ctx.channel, _do_incident_list())

    @incident_group.command(name="add", aliases=["добавить"])
    @commands.check(is_bot_admin_check)
    async def incident_add(ctx, title: str, *, text: str):
        await safe_send(ctx.channel, _do_incident_add(title, text, ctx.author))

    @incident_group.command(name="remove", aliases=["удалить"])
    @commands.check(is_bot_admin_check)
    async def incident_remove(ctx, incident_id: str):
        await safe_send(ctx.channel, _do_incident_remove(incident_id))

    @bot.group(name="config", aliases=["настройки"], invoke_without_command=True)
    @commands.check(is_bot_admin_check)
    async def config_group(ctx):
        await safe_send(ctx.channel, f"Доступно: {settings.COMMAND_PREFIX}config reload")

    @config_group.command(name="reload", aliases=["перезагрузить"])
    @commands.check(is_bot_admin_check)
    async def config_reload(ctx):
        await safe_send(ctx.channel, _do_config_reload(router))

    @bot.command(name="ping", aliases=["пинг"])
    async def ping(ctx):
        await safe_send(ctx.channel, f"Pong! Задержка: {round(bot.latency * 1000)}ms")

    @bot.command(name="help", aliases=["помощь", "справка"])
    @commands.check(is_bot_admin_check)
    async def help_command(ctx):
        await safe_send(ctx.channel, _HELP_TEXT)


async def set_translator(bot) -> None:
    """Ставит переводчик команд для русской локали.
    
    Вызывается из on_ready в async-контексте."""
    translator = RussianCommandTranslator()
    await bot.tree.set_translator(translator)
    logger.info("Переводчик команд зарегистрирован: русские имена для RU-локали")
