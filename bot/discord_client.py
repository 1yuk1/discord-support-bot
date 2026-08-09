"""Отправка сообщений в Discord.

Здесь только то, что действительно требует discord.py и aiohttp. Логика
лимитов и дедупа живёт в bot/limits.py и тестируется без этих зависимостей.
"""

import asyncio

import aiohttp
import discord

from bot import settings
from bot.limits import add_reply_footer, register_ping_burst
from bot.logging_setup import log_exception, logger
from bot.text_utils import split_discord_text

# Задержки между попытками отправки, в секундах.
_SEND_RETRY_DELAYS = (0, 1, 3, 9)
DISCORD_MESSAGE_LIMIT = 2000


async def safe_send(channel, content) -> "discord.Message | None":
    """Отправляет сообщение, разбивая по лимиту и повторяя при сетевых сбоях."""
    text = "" if content is None else str(content)
    if not text.strip():
        return None

    full_text = add_reply_footer(text, settings.BOT_REPLY_FOOTER)
    sent_message = None
    channel_id = getattr(channel, "id", "unknown")

    for chunk in split_discord_text(full_text, DISCORD_MESSAGE_LIMIT):
        for attempt, delay in enumerate(_SEND_RETRY_DELAYS, start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                sent_message = await channel.send(chunk)
                break
            except (discord.HTTPException, aiohttp.ClientError, OSError) as exc:
                if attempt >= len(_SEND_RETRY_DELAYS):
                    log_exception(
                        "Не удалось отправить сообщение после всех попыток",
                        exc,
                        channel_id=channel_id,
                        content_preview=chunk[:200],
                    )
                    return sent_message
                logger.warning(
                    "Временная ошибка отправки, повтор | attempt=%s | channel_id=%s | error=%s",
                    attempt,
                    channel_id,
                    exc,
                )
            except Exception as exc:
                log_exception(
                    "Ошибка отправки сообщения в Discord",
                    exc,
                    channel_id=channel_id,
                    content_preview=chunk[:200],
                )
                return sent_message

    return sent_message


async def send_reminder(channel, content, ping_role_ids) -> "discord.Message | None":
    """Отправляет напоминание персоналу.

    От safe_send отличается двумя вещами:
      - без BOT_REPLY_FOOTER: это не ответ на вопрос, подпись про обучение
        здесь не к месту;
      - явно разрешает упоминание ролей из ping_role_ids. По умолчанию
        discord.py наследует allowed_mentions клиента, и пинг мог бы
        отрендериться текстом без уведомления.
    """
    text = "" if content is None else str(content)
    if not text.strip():
        return None

    allowed = discord.AllowedMentions(
        everyone=False,
        users=False,
        roles=[discord.Object(id=role_id) for role_id in ping_role_ids] or False,
    )
    channel_id = getattr(channel, "id", "unknown")

    for attempt, delay in enumerate(_SEND_RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await channel.send(text, allowed_mentions=allowed)
        except discord.Forbidden as exc:
            # Нет прав писать в канал — повторы не помогут.
            log_exception("Нет прав отправить напоминание", exc, channel_id=channel_id)
            return None
        except (discord.HTTPException, aiohttp.ClientError, OSError) as exc:
            if attempt >= len(_SEND_RETRY_DELAYS):
                log_exception("Не удалось отправить напоминание", exc, channel_id=channel_id)
                return None
            logger.warning(
                "Временная ошибка отправки напоминания, повтор | attempt=%s | channel_id=%s",
                attempt,
                channel_id,
            )
        except Exception as exc:
            log_exception("Ошибка отправки напоминания", exc, channel_id=channel_id)
            return None

    return None


async def reply_private(interaction, content) -> None:
    """Эфемерный ответ на slash-команду: виден только вызвавшему.

    Отличия от safe_send:
      - не клеит BOT_REPLY_FOOTER — подпись «бот только обучается» нужна игроку,
        а не админу, который смотрит статус;
      - длинный текст (например сводка тикета) уходит первым ответом,
        остаток — followup-сообщениями, тоже эфемерными.

    Эфемерность возможна только здесь: у обычного сообщения в канале такого
    флага нет, поэтому префиксные команды всегда отвечают всем.
    """
    text = "" if content is None else str(content)
    if not text.strip():
        text = "Готово."

    chunks = split_discord_text(text, DISCORD_MESSAGE_LIMIT)
    try:
        if interaction.response.is_done():
            await interaction.followup.send(chunks[0], ephemeral=True)
        else:
            await interaction.response.send_message(chunks[0], ephemeral=True)

        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True)
    except Exception as exc:
        log_exception(
            "Не удалось отправить эфемерный ответ",
            exc,
            channel_id=getattr(getattr(interaction, "channel", None), "id", "unknown"),
            content_preview=chunks[0][:200],
        )


async def moderate_ping_spam(message, state: dict) -> None:
    """Удаляет флуд пингами, когда тикет уже передан человеку."""
    if message.author.bot:
        return

    mention_count = len(getattr(message, "mentions", []) or []) + len(
        getattr(message, "role_mentions", []) or []
    )
    if not register_ping_burst(state, mention_count):
        return

    channel_id = getattr(message.channel, "id", "unknown")
    try:
        await message.delete()
        logger.info(
            "Удалён флуд пингами | channel_id=%s | author=%s | mentions=%s",
            channel_id,
            message.author,
            mention_count,
        )
    except discord.Forbidden:
        # Нет права на удаление — хотя бы обозначим реакцией.
        try:
            await message.add_reaction("🔇")
        except Exception as exc:
            log_exception("Не удалось поставить реакцию на флуд", exc, channel_id=channel_id)
    except Exception as exc:
        log_exception("Не удалось удалить флуд пингами", exc, channel_id=channel_id)
