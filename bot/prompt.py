"""Загрузка промптов из каталога prompts/.

Промпты держим в отдельных файлах, чтобы правка политик ответа не требовала
трогать Python-код и пересобирать образ. Поддерживаются подстановки
{SERVER_*} из конфига, поэтому ссылки и версии не дублируются в тексте.
"""

from pathlib import Path

from bot import settings
from bot.text_utils import apply_templates, find_unknown_placeholders

SYSTEM_PROMPT_FILE = "system.md"
SUMMARY_PROMPT_FILE = "summary.md"

# Заполняется вызывающей стороной, не относится к настройкам сервера.
_RUNTIME_PLACEHOLDERS = {"TRANSCRIPT"}


class PromptError(Exception):
    """Промпт отсутствует или содержит неизвестные подстановки."""


def _prompts_dir() -> Path:
    return Path(settings.PROMPTS_DIR)


def load_prompt(filename: str) -> str:
    """Читает промпт и подставляет переменные сервера."""
    path = _prompts_dir() / filename
    if not path.exists():
        raise PromptError(f"Файл промпта не найден: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"Не удалось прочитать {path}: {exc}") from exc

    if not raw.strip():
        raise PromptError(f"Файл промпта пуст: {path}")

    unknown = [
        name
        for name in find_unknown_placeholders(raw, settings.TEMPLATE_VARS)
        if name not in _RUNTIME_PLACEHOLDERS
    ]
    if unknown:
        raise PromptError(
            f"{filename}: неизвестные подстановки {sorted(set(unknown))}. "
            f"Доступны: {sorted(settings.TEMPLATE_VARS)}"
        )

    return apply_templates(raw, settings.TEMPLATE_VARS).strip()


class PromptLibrary:
    """Кэширует промпты и умеет перечитывать их без рестарта бота."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def get(self, filename: str) -> str:
        if filename not in self._cache:
            self._cache[filename] = load_prompt(filename)
        return self._cache[filename]

    @property
    def system(self) -> str:
        return self.get(SYSTEM_PROMPT_FILE)

    @property
    def summary(self) -> str:
        return self.get(SUMMARY_PROMPT_FILE)

    def reload(self) -> None:
        """Перечитывает файлы. При ошибке кэш остаётся прежним."""
        refreshed = {name: load_prompt(name) for name in self._cache}
        self._cache.update(refreshed)


prompts = PromptLibrary()
