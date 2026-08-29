"""Модуль сбора метрик и учёта расхода токенов AI."""

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from bot import settings
from bot.logging_setup import log_exception, logger


class TokenMetricsTracker:
    """Учёт и агрегация расхода токенов AI."""

    def __init__(self, filepath: str | Path | None = None) -> None:
        if filepath is None:
            self._path = Path(settings.DATA_DIR) / "metrics.json"
        else:
            self._path = Path(filepath)

        self._lock = threading.Lock()
        self._last_save_time: float = 0.0
        self._dirty: bool = False
        self._data: dict = {
            "total_requests": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "by_provider": {},
            "by_model": {},
            "by_task": {},
            "daily": {},
        }
        self.load()

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def load(self) -> None:
        """Загружает сохраненные метрики с диска."""
        with self._lock:
            if not self._path.exists():
                return
            try:
                content = self._path.read_text(encoding="utf-8")
                loaded = json.loads(content)
                if isinstance(loaded, dict):
                    self._data = loaded
            except Exception as exc:
                log_exception("Не удалось загрузить метрики токенов", exc)

    def save(self, force: bool = False) -> None:
        """Сохраняет метрики на диск с дебаунсом."""
        with self._lock:
            if not self._dirty and not force:
                return

            now = time.time()
            if not force and (now - self._last_save_time < 5.0):
                return

            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp_file = self._path.with_suffix(".tmp")
                tmp_file.write_text(
                    json.dumps(self._data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp_file.replace(self._path)
                self._dirty = False
                self._last_save_time = now
            except Exception as exc:
                log_exception("Не удалось сохранить метрики токенов", exc)

    def record_usage(
        self,
        provider: str,
        model: str,
        task: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Регистрирует один вызов AI и количество потраченных токенов."""
        if not prompt_tokens and not completion_tokens and not total_tokens:
            return

        prompt_tokens = max(int(prompt_tokens or 0), 0)
        completion_tokens = max(int(completion_tokens or 0), 0)
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = max(int(total_tokens), prompt_tokens + completion_tokens)

        today = self._today_key()
        provider_name = str(provider or "unknown").lower()
        model_name = str(model or "unknown")
        task_name = str(task or "general").lower()

        with self._lock:
            # Общие тоталы
            self._data["total_requests"] = int(self._data.get("total_requests", 0)) + 1
            self._data["total_prompt_tokens"] = int(self._data.get("total_prompt_tokens", 0)) + prompt_tokens
            self._data["total_completion_tokens"] = int(self._data.get("total_completion_tokens", 0)) + completion_tokens
            self._data["total_tokens"] = int(self._data.get("total_tokens", 0)) + total_tokens

            # По провайдерам
            by_provider = self._data.setdefault("by_provider", {})
            p_entry = by_provider.setdefault(provider_name, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            p_entry["requests"] += 1
            p_entry["prompt_tokens"] += prompt_tokens
            p_entry["completion_tokens"] += completion_tokens
            p_entry["total_tokens"] += total_tokens

            # По моделям
            by_model = self._data.setdefault("by_model", {})
            m_entry = by_model.setdefault(model_name, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            m_entry["requests"] += 1
            m_entry["prompt_tokens"] += prompt_tokens
            m_entry["completion_tokens"] += completion_tokens
            m_entry["total_tokens"] += total_tokens

            # По типу задачи
            by_task = self._data.setdefault("by_task", {})
            t_entry = by_task.setdefault(task_name, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            t_entry["requests"] += 1
            t_entry["prompt_tokens"] += prompt_tokens
            t_entry["completion_tokens"] += completion_tokens
            t_entry["total_tokens"] += total_tokens

            # По дням
            daily = self._data.setdefault("daily", {})
            d_entry = daily.setdefault(today, {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "by_model": {},
            })
            d_entry["requests"] += 1
            d_entry["prompt_tokens"] += prompt_tokens
            d_entry["completion_tokens"] += completion_tokens
            d_entry["total_tokens"] += total_tokens

            d_model = d_entry.setdefault("by_model", {})
            dm_entry = d_model.setdefault(model_name, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            dm_entry["requests"] += 1
            dm_entry["prompt_tokens"] += prompt_tokens
            dm_entry["completion_tokens"] += completion_tokens
            dm_entry["total_tokens"] += total_tokens

            self._dirty = True

        self.save(force=False)

    def get_summary(self) -> dict:
        """Возвращает копию всех агрегированных метрик."""
        with self._lock:
            return json.loads(json.dumps(self._data))

    def format_report(self) -> str:
        """Форматирует отчет по использованию токенов для Discord."""
        summary = self.get_summary()
        today = self._today_key()
        today_data = summary.get("daily", {}).get(today, {})

        lines = [
            "📊 **Статистика расхода токенов AI**",
            "",
            f"📅 **Сегодня ({today}):**",
            f"  • Запросов: `{today_data.get('requests', 0)}`",
            f"  • Входящие токены (prompt): `{today_data.get('prompt_tokens', 0):,}`",
            f"  • Исходящие токены (completion): `{today_data.get('completion_tokens', 0):,}`",
            f"  • Всего токенов: **`{today_data.get('total_tokens', 0):,}`**",
            "",
            "📈 **За всё время:**",
            f"  • Всего запросов: `{summary.get('total_requests', 0)}`",
            f"  • Входящие токены: `{summary.get('total_prompt_tokens', 0):,}`",
            f"  • Исходящие токены: `{summary.get('total_completion_tokens', 0):,}`",
            f"  • Всего токенов: **`{summary.get('total_tokens', 0):,}`**",
        ]

        by_model = summary.get("by_model", {})
        if by_model:
            lines.append("")
            lines.append("🤖 **Расход по моделям:**")
            for model_name, m_data in sorted(by_model.items(), key=lambda x: x[1].get("total_tokens", 0), reverse=True):
                lines.append(
                    f"  • `{model_name}`: `{m_data.get('total_tokens', 0):,}` токенов ({m_data.get('requests', 0)} запр.)"
                )

        by_task = summary.get("by_task", {})
        if by_task:
            task_names_ru = {
                "reply": "Ответы в тикетах",
                "reminder": "Напоминания",
                "summary": "Сводки тикетов",
                "general": "Прочее",
            }
            lines.append("")
            lines.append("🎯 **Расход по задачам:**")
            for task_key, t_data in sorted(by_task.items(), key=lambda x: x[1].get("total_tokens", 0), reverse=True):
                title = task_names_ru.get(task_key, task_key)
                lines.append(
                    f"  • {title}: `{t_data.get('total_tokens', 0):,}` токенов ({t_data.get('requests', 0)} запр.)"
                )

        return "\n".join(lines)


metrics = TokenMetricsTracker()
