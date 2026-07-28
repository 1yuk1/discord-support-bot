"""Index SinusSMP support knowledge into ChromaDB.

Knowledge content lives in JSON files under knowledge/. This script only loads,
validates, templates and indexes that content. Quest data from quests_summary.md
is still supported when the file exists.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import config

reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if reconfigure_stdout:
    reconfigure_stdout(encoding="utf-8")


# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = config.DB_PATH
MODEL_CACHE_PATH = config.MODEL_CACHE_PATH
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_MODEL_TYPE = config.EMBEDDING_MODEL_TYPE
AUTO_UPDATE_CHROMA_DB = config.AUTO_UPDATE_CHROMA_DB
SEARCH_TOP_K = config.SEARCH_TOP_K
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
QUESTS_SUMMARY_PATH = BASE_DIR / "quests_summary.md"

TEMPLATE_VARS = {
    "SERVER_MIN_VERSION": config.SERVER_MIN_VERSION,
    "SERVER_MAX_VERSION": config.SERVER_MAX_VERSION,
    "SERVER_RECOMMENDED_VERSION": config.SERVER_RECOMMENDED_VERSION,
    "SERVER_SUPPORTED_VERSIONS": config.SERVER_SUPPORTED_VERSIONS,
}
_TEMPLATE_VAR_RE = re.compile(r"\{([A-Z_]+)\}")

EMBEDDING_QUERY_INSTRUCTION = (
    "Найди наиболее релевантный блок базы знаний для вопроса игрока "
    "Minecraft-сервера SinusSMP."
)


def format_embedding_text(text: str, mode: str) -> str:
    """Return model-specific text for embedding.

    This must stay in sync with discord_bot.py. For multilingual-e5-large-instruct
    queries use the recommended Instruct/Query format; passages stay plain.
    """
    if EMBEDDING_MODEL_TYPE == "e5-instruct":
        if mode == "query":
            return f"Instruct: {EMBEDDING_QUERY_INSTRUCTION}\nQuery: {text}"
        return text
    if EMBEDDING_MODEL_TYPE == "e5":
        return f"{mode}: {text}"
    return text


# ── Knowledge files ─────────────────────────────────────────────────────────
def apply_templates(value):
    """Recursively replace {SERVER_*} placeholders in JSON values."""
    if isinstance(value, str):
        return _TEMPLATE_VAR_RE.sub(
            lambda match: str(TEMPLATE_VARS.get(match.group(1), match.group(0))),
            value,
        )
    if isinstance(value, list):
        return [apply_templates(item) for item in value]
    if isinstance(value, dict):
        return {key: apply_templates(item) for key, item in value.items()}
    return value


def validate_knowledge_item(item: dict, source: Path) -> None:
    required = ("id", "question", "synonyms", "category", "for_llm")
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"{source.name}: item {item.get('id', '<no id>')} missing {missing}")
    if not isinstance(item["synonyms"], list):
        raise ValueError(f"{source.name}: item {item['id']} synonyms must be a list")
    if not isinstance(item["for_llm"], dict):
        raise ValueError(f"{source.name}: item {item['id']} for_llm must be an object")


def load_knowledge_files(directory: Path = KNOWLEDGE_DIR) -> list[dict]:
    """Load and validate all knowledge/*.json files."""
    if not directory.is_dir():
        raise SystemExit(f"❌ Директория базы знаний не найдена: {directory}")

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        raise SystemExit(f"❌ В директории базы знаний нет JSON-файлов: {directory}")

    items = []
    seen_ids = set()

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise SystemExit(f"❌ Не удалось прочитать {path.name}: {exc}") from exc

        if not isinstance(data, list):
            raise SystemExit(f"❌ {path.name}: ожидается JSON-массив объектов")

        for raw_item in data:
            if not isinstance(raw_item, dict):
                raise SystemExit(f"❌ {path.name}: каждый элемент должен быть объектом")
            item = apply_templates(raw_item)
            try:
                validate_knowledge_item(item, path)
            except ValueError as exc:
                raise SystemExit(f"❌ {exc}") from exc
            if item["id"] in seen_ids:
                raise SystemExit(f"❌ Дубликат id '{item['id']}' в {path.name}")
            seen_ids.add(item["id"])
            items.append(item)

        print(f"  ✅ {path.name}: {len(data)} блоков")

    return items


# ── Quests from markdown ────────────────────────────────────────────────────
def normalize_quest_section(raw_section):
    section = raw_section.strip()
    if "Обычные квесты" in section:
        return "Обычные квесты"
    if "Кастомные квесты" in section:
        return "Кастомные квесты"
    return section or "Квесты сервера"


def normalize_quest_description(description):
    description = description.strip()
    if description.lower().rstrip(".") == "не указано":
        return ""
    return description.lstrip("*").strip()


def make_quest_display_title(title):
    if title.lower().startswith("квест"):
        return title
    return f"Квест «{title}»"


def pluralize_ru(number, one, few, many):
    if 11 <= number % 100 <= 14:
        return many
    if number % 10 == 1:
        return one
    if 2 <= number % 10 <= 4:
        return few
    return many


def format_quest_count(number):
    return f"{number} {pluralize_ru(number, 'квест', 'квеста', 'квестов')}"


def get_quest_title_terms(title, display_title):
    terms = [title, display_title]
    quoted_parts = re.findall(r"«([^»]+)»", title)
    for quoted_part in quoted_parts:
        terms.append(quoted_part)
        terms.append(f"квест {quoted_part}")
    return terms


def build_quest_item(quest, sequence_number):
    title = quest["title"]
    display_title = make_quest_display_title(title)
    section = quest.get("section") or "Квесты сервера"
    description = quest.get("description", "")
    task = quest.get("task", "")

    search_parts = [
        *get_quest_title_terms(title, display_title),
        section,
        task,
        description,
    ]

    full_solution_parts = [f"{display_title} относится к разделу «{section}»."]
    if description:
        full_solution_parts.append(f"Описание: {description}")
    full_solution_parts.append(f"Что нужно выполнить: {task}")

    return {
        "id": f"server_quest_{sequence_number:03d}",
        "question": display_title,
        "synonyms": [part for part in search_parts if part],
        "category": "server_quests",
        "priority": "low",
        "for_llm": {
            "problem": f"Игрок спрашивает про {display_title}",
            "diagnostics": [],
            "quick_answer": f"{display_title}: {task}",
            "full_solution": " ".join(full_solution_parts),
            "steps": [
                f"1. Найдите {display_title} в меню квестов",
                f"2. Выполните условие: {task}",
            ],
            "transfer_to_human": False,
            "immediate_action": "Подсказать условие выполнения квеста",
        },
    }


def build_quest_overview_item(quest_count, section_counts):
    details = []
    if section_counts.get("Обычные квесты"):
        details.append(f"{section_counts['Обычные квесты']} обычных")
    if section_counts.get("Кастомные квесты"):
        details.append(f"{section_counts['Кастомные квесты']} кастомных")

    count_text = format_quest_count(quest_count)
    if details:
        count_text += f": {', '.join(details)}"

    return {
        "id": "server_quests_overview",
        "question": "Все квесты на сервере",
        "synonyms": [
            "все квесты на сервере",
            "список квестов",
            "какие квесты есть",
            "сколько квестов",
            "квесты сервера",
            "обычные квесты",
            "кастомные квесты",
        ],
        "category": "server_quests",
        "priority": "low",
        "for_llm": {
            "problem": "Игрок спрашивает про список квестов на сервере",
            "diagnostics": [],
            "quick_answer": f"На сервере есть {count_text}.",
            "full_solution": (
                f"На сервере есть {count_text}. Если игрок спрашивает конкретный квест, "
                "ориентируйтесь на название квеста или условие выполнения."
            ),
            "steps": [
                "1. Если игрок спрашивает общее количество, сообщите количество квестов",
                "2. Если игрок спрашивает конкретный квест, ответьте условием выполнения из найденного блока",
            ],
            "transfer_to_human": False,
            "immediate_action": "Объяснить сколько квестов есть и уточнить конкретный квест при необходимости",
        },
    }


def load_quest_items(path=QUESTS_SUMMARY_PATH):
    if not os.path.exists(path):
        print(f"  ℹ️ Файл квестов не найден, пропускаю: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    quests = []
    current_section = "Квесты сервера"
    current_quest = None

    def flush_current_quest():
        nonlocal current_quest
        if current_quest and current_quest.get("title") and current_quest.get("task"):
            quests.append(current_quest)
        current_quest = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("### "):
            flush_current_quest()
            current_quest = {
                "title": line.removeprefix("### ").strip(),
                "section": current_section,
                "description": "",
                "task": "",
            }
            continue

        if line.startswith("## "):
            flush_current_quest()
            current_section = normalize_quest_section(line.removeprefix("## "))
            continue

        if current_quest is None:
            continue

        if line.startswith("Внутренний ID") or line.startswith("ID:"):
            continue
        if line.startswith("Описание:"):
            current_quest["description"] = normalize_quest_description(line.split(":", 1)[1])
            continue
        if line.startswith("Что нужно выполнить:"):
            current_quest["task"] = line.split(":", 1)[1].strip()
            continue

    flush_current_quest()

    quest_items = [
        build_quest_item(quest, sequence_number)
        for sequence_number, quest in enumerate(quests, start=1)
    ]

    if quest_items:
        section_counts = {}
        for quest in quests:
            section = quest.get("section", "Квесты сервера")
            section_counts[section] = section_counts.get(section, 0) + 1
        quest_items.insert(0, build_quest_overview_item(len(quests), section_counts))

    return quest_items


# ── Chroma document builders ────────────────────────────────────────────────
def build_search_text(item: dict) -> str:
    parts = [item["question"], " ".join(item.get("synonyms", []))]
    return ". ".join(part for part in parts if part)


def build_document(item: dict) -> str:
    for_llm = item.get("for_llm", {})
    doc_lines = [f"## Проблема\n{for_llm.get('problem', item['question'])}"]

    if for_llm.get("diagnostics"):
        doc_lines.append(
            "\n## Диагностика\n" + "\n".join(f"- {q}" for q in for_llm["diagnostics"])
        )
    if for_llm.get("quick_answer"):
        doc_lines.append(f"\n## Быстрый ответ\n{for_llm['quick_answer']}")
    if for_llm.get("full_solution"):
        doc_lines.append(f"\n## Полное решение\n{for_llm['full_solution']}")
    if for_llm.get("steps"):
        doc_lines.append("\n## Инструкция по шагам\n" + "\n".join(for_llm["steps"]))
    if for_llm.get("immediate_action"):
        doc_lines.append(f"\n## Следующее действие\n{for_llm['immediate_action']}")

    return "\n".join(doc_lines)


def build_metadata(item: dict) -> dict:
    for_llm = item.get("for_llm", {})
    return {
        "id": item["id"],
        "question": item["question"],
        "category": item.get("category", "unknown"),
        "priority": item.get("priority", "medium"),
        "transfer_to_human": bool(for_llm.get("transfer_to_human", False)),
    }


def index_knowledge(knowledge_base: list[dict], embedder, collection) -> None:
    search_texts = [
        format_embedding_text(build_search_text(item), "passage")
        for item in knowledge_base
    ]
    embeddings = embedder.encode(
        search_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    ids = [item["id"] for item in knowledge_base]
    documents = [build_document(item) for item in knowledge_base]
    metadatas = [build_metadata(item) for item in knowledge_base]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def print_stats(knowledge_base: list[dict]) -> None:
    categories: dict[str, int] = {}
    for item in knowledge_base:
        category = item.get("category", "unknown")
        categories[category] = categories.get(category, 0) + 1

    print(f"\nЗагружено {len(knowledge_base)} блоков знаний")
    print("Статистика по категориям:")
    for category, count in sorted(categories.items()):
        print(f"   {category}: {count}")


def run_test_queries(embedder, collection) -> None:
    test_queries = [
        "не заходит на сервер connection timed out",
        "большой пинг что делать",
        "не прошел проверку на бота",
        "донат не пришел после оплаты",
        "какие квесты есть",
        "как заприватить базу",
        "забыл пароль",
    ]
    print("\n" + "=" * 60)
    print("ТЕСТОВЫЙ ПОИСК")
    print("=" * 60)
    for query in test_queries:
        query_embedding = embedder.encode(
            format_embedding_text(query, "query"),
            normalize_embeddings=True,
        ).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=SEARCH_TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        print(f"\nЗапрос: '{query}'")
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        for index, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
            print(
                f"  [{index}] distance={distance:.4f} | "
                f"{meta.get('category')} | {meta.get('question')}"
            )
            print(f"      {doc[:140].replace(chr(10), ' ')}...")


def main() -> None:
    if not AUTO_UPDATE_CHROMA_DB:
        print("Автообновление ChromaDB отключено в settings.toml (paths.auto_update_chroma_db = false).")
        print("База данных не изменена.")
        return

    print("📂 Загрузка базы знаний из JSON...")
    knowledge_base = load_knowledge_files(KNOWLEDGE_DIR)

    quest_items = load_quest_items()
    if quest_items:
        knowledge_base.extend(quest_items)
        print(f"  ✅ Квесты: {len(quest_items) - 1} шт.")

    if not knowledge_base:
        raise SystemExit("❌ База знаний пуста")

    if os.path.exists(DB_PATH):
        print(f"\n🗑️ Удаление старой базы данных: {DB_PATH}")
        shutil.rmtree(DB_PATH)

    print(f"\n🧠 Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_PATH)

    print("Создание ChromaDB коллекции...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        "sinussmp_support",
        metadata={"hnsw:space": "cosine"},
    )

    print("\n📥 Индексация базы знаний...")
    index_knowledge(knowledge_base, embedder, collection)
    print_stats(knowledge_base)

    if os.environ.get("RUN_INDEXER_TESTS") == "1":
        run_test_queries(embedder, collection)

    print("\n✅ Индексация завершена!")


if __name__ == "__main__":
    main()
