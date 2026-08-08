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
