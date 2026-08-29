"""Индексация базы знаний SinusSMP в ChromaDB.

Читает knowledge/*.json, валидирует, подставляет переменные сервера и
перестраивает коллекцию с нуля. Запуск: python indexer.py

Формат текста для эмбеддингов берётся из bot.embeddings — того же модуля,
что использует бот. Это гарантирует, что индексация и поиск не разъедутся.
"""

import shutil
import sys
from pathlib import Path

from bot import settings
from bot.embeddings import MODE_PASSAGE, embedding_fingerprint, format_embedding_text, load_embedder
from bot.knowledge import (
    KnowledgeError,
    build_document,
    build_metadata,
    build_search_text,
    count_by_category,
    find_synonym_conflicts,
    load_knowledge,
)

reconfigure = getattr(sys.stdout, "reconfigure", None)
if reconfigure:
    try:
        reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass


def index_knowledge(knowledge_base: list[dict], embedder, collection) -> None:
    """Считает эмбеддинги и загружает блоки в коллекцию через upsert."""
    current_ids = [item["id"] for item in knowledge_base]
    search_texts = [
        format_embedding_text(build_search_text(item), MODE_PASSAGE)
        for item in knowledge_base
    ]
    embeddings = embedder.encode(
        search_texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Удаляем устаревшие ID, которых больше нет в файлах базы знаний
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing.get("ids") or [])
        obsolete_ids = list(existing_ids - set(current_ids))
        if obsolete_ids:
            collection.delete(ids=obsolete_ids)
            print(f"Удалено устаревших блоков: {len(obsolete_ids)}")
    except Exception:
        pass

    collection.upsert(
        ids=current_ids,
        embeddings=embeddings,
        documents=[build_document(item) for item in knowledge_base],
        metadatas=[build_metadata(item) for item in knowledge_base],
    )


def report(knowledge_base: list[dict]) -> None:
    print(f"\nЗагружено блоков: {len(knowledge_base)}")
    print("По категориям:")
    for category, count in count_by_category(knowledge_base).items():
        print(f"   {category}: {count}")

    conflicts = find_synonym_conflicts(knowledge_base)
    if not conflicts:
        return

    print(
        f"\nВнимание: {len(conflicts)} синонимов ведут в несколько блоков. "
        f"При search_top_k={settings.SEARCH_TOP_K} они могут занять все слоты "
        f"и вытеснить нужный блок:"
    )
    for synonym, ids in conflicts.items():
        print(f"   '{synonym}' -> {', '.join(ids)}")


def main() -> None:
    if not settings.AUTO_UPDATE_CHROMA_DB:
        print(
            "Автообновление ChromaDB отключено ([paths].auto_update_chroma_db = false).\n"
            "База данных не изменена."
        )
        raise SystemExit(2)

    print("Загрузка базы знаний (.md и .json)...")
    try:
        knowledge_base = load_knowledge()
    except KnowledgeError as exc:
        raise SystemExit(f"Ошибка базы знаний: {exc}") from exc

    if not knowledge_base:
        raise SystemExit("База знаний пуста")

    db_path = Path(settings.DB_PATH)

    print(f"Загрузка модели эмбеддингов: {settings.EMBEDDING_MODEL}")
    embedder = load_embedder()

    import chromadb

    print(f"Подключение к коллекции '{settings.CHROMA_COLLECTION_NAME}'...")
    client = chromadb.PersistentClient(path=str(db_path))
    
    collection = client.get_or_create_collection(
        settings.CHROMA_COLLECTION_NAME,
        metadata={
            "hnsw:space": settings.CHROMA_DISTANCE_METRIC,
            **embedding_fingerprint(embedder),
        },
    )

    print("Индексация...")
    index_knowledge(knowledge_base, embedder, collection)
    report(knowledge_base)
    print("\nИндексация завершена.")


if __name__ == "__main__":
    main()

