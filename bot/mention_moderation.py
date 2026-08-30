"""Модерация запрещённых упоминаний защищённых пользователей и ролей."""

import asyncio
from datetime import timedelta
import discord

from bot import filters, settings
from bot.discord_client import safe_send
from bot.logging_setup import channel_label, log_exception, logger
from bot.state import store


def format_duration(td: timedelta) -> str:
    """Форматирует timedelta в понятный вид (напр. '1 мин.', '10 мин.', '1 ч.', '1 дн.')."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds} сек."
    if total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes} мин."
    if total_seconds < 86400:
        hours = total_seconds // 3600
        return f"{hours} ч."
    days = total_seconds // 86400
    return f"{days} дн."


def get_timeout_duration(violation_count: int, durations: list[timedelta] | None = None) -> timedelta:
    """Возвращает длительность таймаута по ступени лестницы наказаний."""
    dur_list = durations if durations is not None else settings.MENTION_TIMEOUT_DURATIONS
    if not dur_list:
        return timedelta(minutes=1)

    # violation_count: 1 -> index 0, 2 -> index 1, 3+ -> index 2 (или последний)
    idx = min(max(violation_count - 1, 0), len(dur_list) - 1)
    return dur_list[idx]


def check_forbidden_mentions(message, guild_config: settings.GuildConfig | None = None) -> list[str]:
    """Проверяет сообщение на наличие запрещённых упоминаний пользователей и ролей.

    Возвращает список отображаемых имён/тегов найденных защищённых целей.
    """
    if guild_config is None:
        guild = getattr(message, "guild", None)
        guild_config = settings.get_guild_config(getattr(guild, "id", None) if guild else None)

    if not guild_config.mention_timeout_enabled:
        return []

    protected_user_ids = set(guild_config.protected_user_ids)
    protected_role_ids = set(guild_config.protected_role_ids)

    if not protected_user_ids and not protected_role_ids:
        return []

    found_targets: list[str] = []

    for user in getattr(message, "mentions", []) or []:
        if user.id in protected_user_ids:
            target_name = f"@{user.display_name}" if hasattr(user, "display_name") else f"<@{user.id}>"
            if target_name not in found_targets:
                found_targets.append(target_name)

    for role in getattr(message, "role_mentions", []) or []:
        if role.id in protected_role_ids:
            target_name = f"@{role.name}" if hasattr(role, "name") else f"<@&{role.id}>"
            if target_name not in found_targets:
                found_targets.append(target_name)

    return found_targets


async def handle_mention_moderation(message) -> bool:
    """Проверяет сообщение на запрещённые пинги и при нарушении выдаёт таймаут.

    Возвращает True, если нарушение обнаружено (или если проверка прошла), чтобы
    дальнейшая обработка тикета не прерывалась.
    """
    author = getattr(message, "author", None)
    if author is None or getattr(author, "bot", False) or getattr(author, "system", False):
        return False

    guild = getattr(message, "guild", None)
    guild_id = getattr(guild, "id", None)
    guild_cfg = settings.get_guild_config(guild_id)
    if not guild_cfg.mention_timeout_enabled:
        return False

    # Персонал и администраторы защищены от мутов за пинг (байпасс)
    if filters.is_staff_member(
        author,
        staff_role_ids=guild_cfg.staff_role_ids,
        admin_role_ids=guild_cfg.admin_role_ids,
        admin_user_ids=guild_cfg.admin_user_ids,
    ):
        return False

    found_targets = check_forbidden_mentions(message, guild_config=guild_cfg)
    if not found_targets:
        return False

    # Нарушение зафиксировано
    user_id = getattr(author, "id", 0)

    violation_count = store.record_mention_violation(guild_id or 0, user_id)
    duration = get_timeout_duration(violation_count, guild_cfg.mention_timeout_durations)
    duration_str = format_duration(duration)
    targets_str = ", ".join(found_targets)
    reason = guild_cfg.mention_timeout_reason or settings.MENTION_TIMEOUT_REASON

    logger.warning(
        "Запрещённое упоминание | guild_id=%s | user_id=%s (%s) | цели=%s | нарушение #%s | таймаут=%s",
        guild_id,
        user_id,
        author,
        targets_str,
        violation_count,
        duration_str,
    )

    # Применяем таймаут в Discord
    timeout_applied = False
    try:
        # discord.Member.timeout принимает timedelta или datetime
        if hasattr(author, "timeout"):
            await author.timeout(duration, reason=reason)
            timeout_applied = True
    except discord.Forbidden as exc:
        logger.warning(
            "Нет прав для выдачи таймаута (бот ниже роли нарушителя или нет Moderate Members) | user=%s | error=%s",
            author,
            exc,
        )
    except Exception as exc:
        log_exception("Не удалось выдать таймаут пользователю", exc, user_id=user_id)

    # Отправляем сообщение в тикет с уведомлением
    notice_text = (
        f"⚠️ <@{user_id}>, упоминание {targets_str} запрещено.\n"
        f"Вам выдан таймаут на **{duration_str}**.\n"
        f"Причина: {reason}."
    )
    if not timeout_applied and hasattr(author, "timeout"):
        notice_text = (
            f"⚠️ <@{user_id}>, упоминание {targets_str} запрещено.\n"
            f"Причина: {reason}."
        )

    await safe_send(message.channel, notice_text)
    store.save(force=True)
    return True
