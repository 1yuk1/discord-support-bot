"""Загрузка, валидация и подготовка базы знаний из knowledge/*.json.

Используется индексатором. Схема одного блока:

    {
      "id": str,                     уникален по всей базе
      "question": str,               формулировки вопроса
      "synonyms": [str],             поисковые варианты
      "category": str,
      "priority": "low"|"medium"|"high",
      "for_llm": {
        "problem": str,
        "diagnostics": [str],
        "quick_answer": str,
        "full_solution": str,
        "steps": [str],
        "transfer_to_human": bool,
        "immediate_action": str
      }
    }
"""

import json
from collections import defaultdict
from pathlib import Path

from bot import settings
from bot.text_utils import apply_templates, find_unknown_placeholders

REQUIRED_FIELDS = ("id", "question", "synonyms", "category", "for_llm")
ALLOWED_PRIORITIES = ("low", "medium", "high")
_LIST_OF_STR_FIELDS = ("diagnostics", "steps")


class KnowledgeError(Exception):
    """Ошибка в содержимом базы знаний."""


def validate_item(item: dict, source: str) -> None:
    """Проверяет структуру блока. Падает с понятным сообщением при ошибке."""
    item_id = item.get("id", "<без id>")
    prefix = f"{source}: блок '{item_id}'"

    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        raise KnowledgeError(f"{prefix} — отсутствуют поля {missing}")

    if not isinstance(item["id"], str) or not item["id"].strip():
        raise KnowledgeError(f"{prefix} — id должен быть непустой строкой")
    if not isinstance(item["question"], str) or not item["question"].strip():
        raise KnowledgeError(f"{prefix} — question должен быть непустой строкой")
    if not isinstance(item["category"], str) or not item["category"].strip():
        raise KnowledgeError(f"{prefix} — category должен быть непустой строкой")

    if not isinstance(item["synonyms"], list) or not all(
        isinstance(value, str) for value in item["synonyms"]
    ):
        raise KnowledgeError(f"{prefix} — synonyms должен быть списком строк")

    priority = item.get("priority", "medium")
    if priority not in ALLOWED_PRIORITIES:
        raise KnowledgeError(
            f"{prefix} — priority='{priority}', допустимо только {ALLOWED_PRIORITIES}"
        )

    for_llm = item["for_llm"]
    if not isinstance(for_llm, dict):
        raise KnowledgeError(f"{prefix} — for_llm должен быть объектом")

    if not str(for_llm.get("quick_answer", "")).strip() and not str(
        for_llm.get("full_solution", "")
    ).strip():
        raise KnowledgeError(
            f"{prefix} — нужен хотя бы один из for_llm.quick_answer / for_llm.full_solution"
        )

    for field in _LIST_OF_STR_FIELDS:
        value = for_llm.get(field, [])
        if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
            raise KnowledgeError(f"{prefix} — for_llm.{field} должен быть списком строк")

    transfer = for_llm.get("transfer_to_human", False)
    if not isinstance(transfer, bool):
        raise KnowledgeError(f"{prefix} — for_llm.transfer_to_human должен быть true/false")


def _check_placeholders(item: dict, source: str) -> None:
    """Ловит опечатки в {SERVER_*}: без проверки они уезжают игроку как есть."""
    unknown: set[str] = set()

    def walk(value) -> None:
        if isinstance(value, str):
            unknown.update(find_unknown_placeholders(value, settings.TEMPLATE_VARS))
        elif isinstance(value, list):
            for entry in value:
                walk(entry)
        elif isinstance(value, dict):
            for entry in value.values():
                walk(entry)

    walk(item)
    if unknown:
        raise KnowledgeError(
            f"{source}: блок '{item.get('id')}' — неизвестные подстановки "
            f"{sorted(unknown)}. Доступны: {sorted(settings.TEMPLATE_VARS)}"
        )


def load_knowledge(directory: str | Path | None = None) -> list[dict]:
    """Читает все JSON-файлы базы знаний, валидирует и подставляет переменные."""
    knowledge_dir = Path(directory or settings.KNOWLEDGE_DIR)
    if not knowledge_dir.is_dir():
        raise KnowledgeError(f"Директория базы знаний не найдена: {knowledge_dir}")

    json_files = sorted(knowledge_dir.glob("*.json"))
    if not json_files:
        raise KnowledgeError(f"В директории базы знаний нет JSON-файлов: {knowledge_dir}")

    items: list[dict] = []
    seen_ids: dict[str, str] = {}

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeError(f"Не удалось прочитать {path.name}: {exc}") from exc

        if not isinstance(data, list):
            raise KnowledgeError(f"{path.name}: ожидается JSON-массив объектов")

        for raw_item in data:
            if not isinstance(raw_item, dict):
                raise KnowledgeError(f"{path.name}: каждый элемент должен быть объектом")

            _check_placeholders(raw_item, path.name)
            item = apply_templates(raw_item, settings.TEMPLATE_VARS)
            validate_item(item, path.name)

            item_id = item["id"]
            if item_id in seen_ids:
                raise KnowledgeError(
                    f"Дубликат id '{item_id}': {path.name} и {seen_ids[item_id]}"
                )
            seen_ids[item_id] = path.name
            items.append(item)

    return items


def build_search_text(item: dict) -> str:
    """Текст, который векторизуется: вопрос плюс синонимы.

    В документ идёт развёрнутый for_llm, а в вектор — короткие формулировки.
    Асимметрия намеренная: поиск по тому, как спрашивает игрок, а в LLM
    уходит полный ответ.
    """
    parts = [item["question"], " ".join(item.get("synonyms", []))]
    return ". ".join(part for part in parts if part)


def build_document(item: dict) -> str:
    """Markdown-документ для контекста LLM."""
    for_llm = item.get("for_llm", {})
    sections = [f"## Проблема\n{for_llm.get('problem') or item['question']}"]

    if for_llm.get("diagnostics"):
        sections.append(
            "\n## Диагностика\n" + "\n".join(f"- {entry}" for entry in for_llm["diagnostics"])
        )
    if for_llm.get("quick_answer"):
        sections.append(f"\n## Быстрый ответ\n{for_llm['quick_answer']}")
    if for_llm.get("full_solution"):
        sections.append(f"\n## Полное решение\n{for_llm['full_solution']}")
    if for_llm.get("steps"):
        sections.append("\n## Инструкция по шагам\n" + "\n".join(for_llm["steps"]))
    if for_llm.get("immediate_action"):
        sections.append(f"\n## Следующее действие\n{for_llm['immediate_action']}")

    return "\n".join(sections)


def build_metadata(item: dict) -> dict:
    for_llm = item.get("for_llm", {})
    return {
        "id": item["id"],
        "question": item["question"],
        "category": item.get("category", "unknown"),
        "priority": item.get("priority", "medium"),
        "transfer_to_human": bool(for_llm.get("transfer_to_human", False)),
    }


def find_synonym_conflicts(knowledge_base: list[dict]) -> dict[str, list[str]]:
    """Синонимы, ведущие в несколько блоков.

    При search_top_k=2 такой запрос может занять оба слота близкими блоками и
    вытеснить действительно нужный. Это предупреждение, не ошибка.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for item in knowledge_base:
        for synonym in item.get("synonyms", []):
            key = synonym.lower().strip()
            if key and item["id"] not in owners[key]:
                owners[key].append(item["id"])
    return {synonym: ids for synonym, ids in sorted(owners.items()) if len(ids) > 1}


def count_by_category(knowledge_base: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in knowledge_base:
        category = item.get("category", "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))
