"""Загрузка, валидация и подготовка базы знаний из knowledge/*.md и knowledge/*.json.

Поддерживает два формата:
1. Markdown (.md) — простой и удобный формат:
   - Заголовок (# или ##) становится темой/вопросом.
   - Текст секции становится решением и индексируется целиком.
   - Синонимы выписывать не нужно: семантический поиск работает по всему тексту.
2. JSON (.json) — структурированный формат (для обратной совместимости):
   - Схема с question, category, for_llm и опциональными synonyms.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from bot import settings
from bot.text_utils import apply_templates, find_unknown_placeholders

ALLOWED_PRIORITIES = ("low", "medium", "high")
_LIST_OF_STR_FIELDS = ("diagnostics", "steps")


class KnowledgeError(Exception):
    """Ошибка в содержимом базы знаний."""


def _slugify(text: str) -> str:
    """Генерирует безопасный slug для идентификатора секции."""
    slug = re.sub(r"[^\w\-_]", "_", text.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:50] or "section"


def validate_item(item: dict, source: str) -> None:
    """Проверяет структуру блока. Падает с понятным сообщением при ошибке."""
    item_id = item.get("id", "<без id>")
    prefix = f"{source}: блок '{item_id}'"

    if not isinstance(item.get("id"), str) or not item["id"].strip():
        raise KnowledgeError(f"{prefix} — id должен быть непустой строкой")
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        raise KnowledgeError(f"{prefix} — question должен быть непустой строкой")
    if not isinstance(item.get("category"), str) or not item["category"].strip():
        raise KnowledgeError(f"{prefix} — category должен быть непустой строкой")

    synonyms = item.get("synonyms", [])
    if not isinstance(synonyms, list) or not all(isinstance(v, str) for v in synonyms):
        raise KnowledgeError(f"{prefix} — synonyms должен быть списком строк")

    priority = item.get("priority", "medium")
    if priority not in ALLOWED_PRIORITIES:
        raise KnowledgeError(
            f"{prefix} — priority='{priority}', допустимо только {ALLOWED_PRIORITIES}"
        )

    for_llm = item.get("for_llm")
    content = item.get("content")

    if for_llm is None and not content:
        raise KnowledgeError(f"{prefix} — нужен либо for_llm, либо content (Markdown)")

    if for_llm is not None:
        if not isinstance(for_llm, dict):
            raise KnowledgeError(f"{prefix} — for_llm должен быть объектом")

        if not str(for_llm.get("quick_answer", "")).strip() and not str(
            for_llm.get("full_solution", "")
        ).strip() and not content:
            raise KnowledgeError(
                f"{prefix} — нужен хотя бы один из for_llm.quick_answer / for_llm.full_solution"
            )

        for field in _LIST_OF_STR_FIELDS:
            value = for_llm.get(field, [])
            if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
                raise KnowledgeError(f"{prefix} — for_llm.{field} должен быть списком строк")


def _check_placeholders(item: dict | str, source: str) -> None:
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
        name = item.get("id") if isinstance(item, dict) else source
        raise KnowledgeError(
            f"{source}: '{name}' — неизвестные подстановки "
            f"{sorted(unknown)}. Доступны: {sorted(settings.TEMPLATE_VARS)}"
        )


def _parse_markdown_file(path: Path) -> list[dict]:
    """Парсит Markdown-файл в список блоков базы знаний.
    
    Разбивает файл по заголовкам ## (секциям) или оставляет единым блоком (#).
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KnowledgeError(f"Не удалось прочитать {path.name}: {exc}") from exc

    _check_placeholders(raw_text, path.name)
    text = apply_templates(raw_text, settings.TEMPLATE_VARS).strip()
    if not text:
        return []

    lines = text.splitlines()
    doc_title = path.stem.replace("_", " ").title()
    category = path.stem.lower()

    # Извлекаем # Заголовок верхнего уровня, если есть
    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            break

    # Разбиваем по ## Подзаголовкам
    sections: list[tuple[str, str]] = []
    current_title = doc_title
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_lines and "\n".join(current_lines).strip():
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif not (line.startswith("# ") and not current_lines and current_title == doc_title):
            current_lines.append(line)

    if current_lines and "\n".join(current_lines).strip():
        sections.append((current_title, "\n".join(current_lines).strip()))

    if not sections:
        sections = [(doc_title, text)]

    items: list[dict] = []
    for idx, (title, content) in enumerate(sections, 1):
        slug = _slugify(title)
        item_id = f"{category}_{slug}" if len(sections) > 1 else category
        if len(sections) > 1 and item_id in [it["id"] for it in items]:
            item_id = f"{item_id}_{idx}"

        transfer_needed = (
            "[transfer_to_human]" in content.lower()
            or "передать старшему специалисту" in content.lower()
            or "передать человеку" in content.lower()
        )

        item = {
            "id": item_id,
            "question": title,
            "synonyms": [],
            "category": category,
            "priority": "medium",
            "content": content,
            "for_llm": {
                "problem": title,
                "quick_answer": content[:300].strip(),
                "full_solution": content,
                "transfer_to_human": transfer_needed,
            },
        }
        items.append(item)

    return items


def load_knowledge(directory: str | Path | None = None) -> list[dict]:
    """Читает все Markdown (.md) и JSON (.json) файлы базы знаний, валидирует и подставляет переменные."""
    knowledge_dir = Path(directory or settings.KNOWLEDGE_DIR)
    if not knowledge_dir.is_dir():
        raise KnowledgeError(f"Директория базы знаний не найдена: {knowledge_dir}")

    json_files = sorted(knowledge_dir.glob("*.json"))
    md_files = sorted(knowledge_dir.glob("*.md"))

    if not json_files and not md_files:
        raise KnowledgeError(f"В директории базы знаний нет .md или .json файлов: {knowledge_dir}")

    items: list[dict] = []
    seen_ids: dict[str, str] = {}

    # 1. Загрузка Markdown файлов
    for path in md_files:
        md_items = _parse_markdown_file(path)
        for item in md_items:
            validate_item(item, path.name)
            item_id = item["id"]
            if item_id in seen_ids:
                raise KnowledgeError(
                    f"Дубликат id '{item_id}': {path.name} и {seen_ids[item_id]}"
                )
            seen_ids[item_id] = path.name
            items.append(item)

    # 2. Загрузка JSON файлов
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
    """Текст, который векторизуется.
    
    Включает заголовок, синонимы (если есть), суть проблемы и ключевые выжимки из решения,
    чтобы семантический поиск находил ответ даже при нестандартных формулировках игрока.
    """
    parts: list[str] = [item.get("question", "")]

    synonyms = item.get("synonyms", [])
    if synonyms:
        parts.append(" ".join(synonyms))

    if item.get("content"):
        # Для Markdown документов векторизуем весь содержательный текст
        parts.append(item["content"])
    else:
        # Для JSON блоков берем суть проблемы, быстрый ответ и диагностику
        for_llm = item.get("for_llm", {})
        if for_llm.get("problem") and for_llm["problem"] != item.get("question"):
            parts.append(for_llm["problem"])
        if for_llm.get("quick_answer"):
            parts.append(for_llm["quick_answer"])
        if for_llm.get("diagnostics"):
            parts.append(" ".join(for_llm["diagnostics"]))

    cleaned = [part.strip() for part in parts if part and part.strip()]
    return ". ".join(cleaned)


def build_document(item: dict) -> str:
    """Markdown-документ для контекста LLM."""
    if item.get("content"):
        return f"# {item['question']}\n\n{item['content']}"

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
    """Синонимы, ведущие в несколько блоков."""
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

