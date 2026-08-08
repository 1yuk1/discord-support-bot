"""Состояние диалогов по каналам и его сохранение между рестартами.

Два независимых флага «бот молчит»:

  human_mode   — эскалация. Ставится автоматически, когда тикет ушёл человеку.
                 Дополнительно включает модерацию спама пингами.
  bot_disabled — админ вручную выключил бота в канале командой !stop.
                 Никакой побочной логики не тянет, снимается только !start.

Разделение нужно, чтобы !start не сбрасывал эскалацию, а !resume не снимал
ручную блокировку.
"""

import json
import time
from collections import deque
from pathlib import Path

from bot import settings
from bot.logging_setup import log_exception, logger

# Сколько id обработанных сообщений держим на канал.
_PROCESSED_IDS_LIMIT = 200
_PROCESSED_IDS_KEEP = 100
_RECENT_NORMALIZED_MAXLEN = 10


def create_channel_state() -> dict:
    return {
        "history": [],
        "human_mode": False,
        # Ручное отключение бота в канале (!stop / !start).
        "bot_disabled": False,
        "disabled_by": None,
        "disabled_at": None,
        "last_activity": time.time(),
        "last_message": "",
        "last_message_time": 0.0,
        "last_answer_time": 0.0,
        "user_messages": deque(),
        "processed_message_ids": set(),
        "last_processed_message_id": None,
        "human_mode_ping_times": deque(),
        # Шапка открытия тикета приходит 1-2 раза подряд — отвечаем один раз.
        "ticket_opening_handled": False,
        # Окно последних нормализованных сообщений для усиленного дедупа.
        "recent_normalized": deque(maxlen=_RECENT_NORMALIZED_MAXLEN),
        # Склейка спама: части сообщений и активная задача debounce.
        "pending_parts": [],
        "debounce_task": None,
        "ai_busy": False,
    }


class ConversationStore:
    """Состояния каналов с периодическим сохранением на диск."""

    def __init__(self) -> None:
        self._channels: dict[int, dict] = {}
        self._dirty = False

    def __contains__(self, channel_id: int) -> bool:
        return channel_id in self._channels

    def __len__(self) -> int:
        return len(self._channels)

    def items(self):
        return self._channels.items()

    def get(self, channel_id: int) -> dict | None:
        return self._channels.get(channel_id)

    def get_or_create(self, channel_id: int) -> dict:
        state = self._channels.get(channel_id)
        if state is None:
            state = create_channel_state()
            self._channels[channel_id] = state
        return state

    def reset(self, channel_id: int) -> dict:
        """Сбрасывает историю, сохраняя ручное отключение бота.

        !clear_history не должен нечаянно «включать» бота там, где админ его
        выключил командой !stop.
        """
        previous = self._channels.get(channel_id) or {}
        state = create_channel_state()
        state["bot_disabled"] = bool(previous.get("bot_disabled"))
        state["disabled_by"] = previous.get("disabled_by")
        state["disabled_at"] = previous.get("disabled_at")
        self._channels[channel_id] = state
        self.mark_dirty()
        return state

    def remove(self, channel_id) -> bool:
        if channel_id is None:
            return False
        removed = self._channels.pop(channel_id, None) is not None
        if removed:
            self.mark_dirty()
        return removed

    def mark_dirty(self) -> None:
        self._dirty = True

    def touch(self, state: dict) -> None:
        state["last_activity"] = time.time()

    # ── История диалога ──────────────────────────────────────────────────────
    def append_turn(self, state: dict, user_text: str, answer: str, author_is_bot: bool) -> None:
        """Добавляет пару реплик и обрезает историю до max_history."""
        author_label = "Система" if author_is_bot else "Пользователь"
        state["history"].append({"role": "user", "content": f"[{author_label}] {user_text}"})
        state["history"].append({"role": "assistant", "content": answer or ""})

        limit = max(settings.MAX_HISTORY, 1) * 2
        if len(state["history"]) > limit:
            state["history"] = state["history"][-limit:]

    def remember_processed(self, state: dict, message_id: int) -> None:
        processed = state["processed_message_ids"]
        processed.add(message_id)
        state["last_processed_message_id"] = message_id
        if len(processed) > _PROCESSED_IDS_LIMIT:
            state["processed_message_ids"] = set(list(processed)[-_PROCESSED_IDS_KEEP:])

    # ── Сохранение и восстановление ──────────────────────────────────────────
    def _snapshot(self) -> dict:
        snapshot = {}
        for channel_id, data in self._channels.items():
            # Сохраняем только каналы с «залипающим» состоянием: историю диалога
            # после рестарта восстанавливать не нужно, а флаги — обязательно.
            if not any((
                data.get("human_mode"),
                data.get("bot_disabled"),
                data.get("ticket_opening_handled"),
            )):
                continue
            snapshot[str(channel_id)] = {
                "human_mode": bool(data.get("human_mode")),
                "bot_disabled": bool(data.get("bot_disabled")),
                "disabled_by": data.get("disabled_by"),
                "disabled_at": data.get("disabled_at"),
                "ticket_opening_handled": bool(data.get("ticket_opening_handled")),
                "last_activity": data.get("last_activity", time.time()),
                "last_processed_message_id": data.get("last_processed_message_id"),
            }
        return snapshot

    def save(self, force: bool = False) -> None:
        if not force and not self._dirty:
            return

        path = Path(settings.STATE_SNAPSHOT_FILE)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Пишем через временный файл: обрыв на середине не оставит битый JSON.
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._snapshot(), f, ensure_ascii=False, indent=2)
            temp_path.replace(path)
            self._dirty = False
        except OSError as exc:
            log_exception(
                "Не удалось сохранить состояние тикетов", exc, file=str(path)
            )

    def load(self) -> None:
        path = Path(settings.STATE_SNAPSHOT_FILE)
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log_exception("Не удалось прочитать состояние тикетов", exc, file=str(path))
            return

        if not isinstance(snapshot, dict):
            return

        now = time.time()
        restored = human_mode = disabled = 0

        for raw_channel_id, data in snapshot.items():
            if not isinstance(data, dict):
                continue
            try:
                channel_id = int(raw_channel_id)
            except (TypeError, ValueError):
                continue

            try:
                last_activity = float(data.get("last_activity", now))
            except (TypeError, ValueError):
                last_activity = now
            if settings.STATE_TTL_SECONDS > 0 and now - last_activity > settings.STATE_TTL_SECONDS:
                continue

            state = create_channel_state()
            state["human_mode"] = bool(data.get("human_mode"))
            state["bot_disabled"] = bool(data.get("bot_disabled"))
            state["disabled_by"] = data.get("disabled_by")
            state["disabled_at"] = data.get("disabled_at")
            state["ticket_opening_handled"] = bool(data.get("ticket_opening_handled"))
            state["last_activity"] = last_activity

            last_message_id = data.get("last_processed_message_id")
            if last_message_id is not None:
                try:
                    state["processed_message_ids"].add(int(last_message_id))
                    state["last_processed_message_id"] = int(last_message_id)
                except (TypeError, ValueError):
                    pass

            self._channels[channel_id] = state
            restored += 1
            human_mode += bool(state["human_mode"])
            disabled += bool(state["bot_disabled"])

        logger.info(
            "Восстановлено состояний тикетов: %s | human_mode=%s | выключен вручную=%s",
            restored,
            human_mode,
            disabled,
        )

    def cleanup_expired(self) -> int:
        if settings.STATE_TTL_SECONDS <= 0:
            return 0

        now = time.time()
        expired = [
            channel_id
            for channel_id, data in self._channels.items()
            if now - data.get("last_activity", now) > settings.STATE_TTL_SECONDS
        ]
        for channel_id in expired:
            self._channels.pop(channel_id, None)

        if expired:
            self.mark_dirty()
            logger.info("Очищены устаревшие состояния тикетов: %s", len(expired))
        return len(expired)


store = ConversationStore()
