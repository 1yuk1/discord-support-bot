"""Поиск по базе знаний в ChromaDB."""

from bot import settings
from bot.embeddings import MODE_QUERY, format_embedding_text, load_embedder
from bot.logging_setup import log_exception, logger
from bot.text_utils import query_variants


class KnowledgeIndexError(Exception):
    """База знаний недоступна или несовместима с текущей моделью."""


class KnowledgeIndex:
    """Обёртка над коллекцией ChromaDB и моделью эмбеддингов."""

    def __init__(self, collection, embedder) -> None:
        self._collection = collection
        self._embedder = embedder

    @property
    def collection_name(self) -> str:
        return getattr(self._collection, "name", settings.CHROMA_COLLECTION_NAME)

    def _find_keyword_matches(self, query: str, limit: int = 2) -> list[str]:
        """Ищет документы по точным ключевым словам (команды /cmd, технические термины)."""
        import re
        
        # Выделяем команды вида /ps, /crafts, /mine, /garant, /b, /link, /changepass
        commands = re.findall(r"/[a-zA-Z0-9_]+", query)
        matches: list[str] = []
        seen: set[str] = set()

        for cmd in commands:
            cmd_lower = cmd.lower()
            try:
                # Поиск в Chroma по вхождению точной команды в документ
                results = self._collection.get(
                    where_document={"$contains": cmd_lower},
                    limit=limit,
                )
                for doc in (results.get("documents") or []):
                    if doc and doc not in seen:
                        seen.add(doc)
                        matches.append(doc)
                        logger.info("RAG Гибридный поиск: найдено точное совпадение по '%s'", cmd_lower)
            except Exception as exc:
                logger.debug("Ошибка поиска по ключевому слову %s: %s", cmd_lower, exc)

        return matches

    def search(self, query: str, top_k: int | None = None, threshold: float | None = None) -> str:
        """Ищет релевантные блоки и возвращает их как единый контекст.

        1. Отсекает документы с расстоянием больше settings.CHROMA_DISTANCE_THRESHOLD (защита от галлюцинаций).
        2. При settings.ENABLE_HYBRID_SEARCH находит точные совпадения по командам (/ps, /mine и др.).
        3. Дедуплицирует результаты и возвращает пустую строку, если релевантных блоков не найдено.
        """
        limit = top_k or settings.SEARCH_TOP_K
        max_dist = threshold if threshold is not None else settings.CHROMA_DISTANCE_THRESHOLD
        variants = query_variants(query)

        seen_documents: set[str] = set()
        context_parts: list[str] = []

        # 1. Гибридный поиск по точным командам
        if settings.ENABLE_HYBRID_SEARCH:
            keyword_docs = self._find_keyword_matches(query, limit=limit)
            for doc in keyword_docs:
                if doc not in seen_documents:
                    seen_documents.add(doc)
                    context_parts.append(doc)

        # 2. Семантический векторный поиск с порогом релевантности
        succeeded = 0
        for variant in variants:
            try:
                embedding = self._embedder.encode(
                    format_embedding_text(variant, MODE_QUERY),
                    normalize_embeddings=True,
                ).tolist()
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:
                log_exception(
                    "Ошибка варианта поиска в ChromaDB", exc, variant_preview=variant[:200]
                )
                continue

            succeeded += 1
            documents = (results.get("documents") or [[]])[0] or []
            metadatas = (results.get("metadatas") or [[]])[0] or []
            distances = (results.get("distances") or [[]])[0] or []

            for doc, meta, dist in zip(documents, metadatas, distances):
                doc_id = (meta or {}).get("id", "unknown") if isinstance(meta, dict) else "unknown"
                logger.info(
                    "RAG кандидат | id=%s | distance=%.4f | max_threshold=%.2f",
                    doc_id,
                    dist,
                    max_dist,
                )
                # Фильтрация по порогу косинусного расстояния
                if dist > max_dist:
                    logger.info("RAG кандидат %s отклонён: distance %.4f > %.2f", doc_id, dist, max_dist)
                    continue

                if doc and doc not in seen_documents:
                    seen_documents.add(doc)
                    context_parts.append(doc)

        if succeeded == 0 and not context_parts:
            raise KnowledgeIndexError("Ни один вариант поиска не выполнился")

        # Ограничиваем суммарное число документов лимитом
        selected = context_parts[:limit]
        return "\n\n".join(selected)


def _verify_embedding_compatibility(collection) -> None:
    """Сверяет модель, которой собрана база, с текущей моделью бота.

    Несовместимые векторы не вызывают ошибок в Chroma: поиск просто начинает
    возвращать случайные блоки, и бот уверенно отвечает не по делу.
    """
    metadata = getattr(collection, "metadata", None) or {}
    indexed_model = metadata.get("embedding_model")

    if not indexed_model:
        logger.warning(
            "В коллекции '%s' нет отпечатка модели — база собрана старой версией "
            "индексатора. Переиндексируйте её, чтобы включить проверку совместимости.",
            getattr(collection, "name", "?"),
        )
        return

    if indexed_model == settings.EMBEDDING_MODEL:
        logger.info("Модель эмбеддингов совпадает с базой: %s", indexed_model)
        return

    message = (
        f"База знаний собрана моделью '{indexed_model}', а бот запущен с "
        f"'{settings.EMBEDDING_MODEL}'. Векторы несовместимы, поиск будет "
        f"возвращать нерелевантные блоки. Требуется переиндексация: python indexer.py"
    )
    if settings.STRICT_EMBEDDING_CHECK:
        raise KnowledgeIndexError(message)
    logger.warning("%s (проверка отключена в конфиге)", message)


def open_knowledge_index() -> KnowledgeIndex:
    """Подключается к ChromaDB и загружает модель эмбеддингов."""
    import chromadb

    logger.info("Подключение к ChromaDB: %s", settings.DB_PATH)
    try:
        client = chromadb.PersistentClient(path=settings.DB_PATH)
    except Exception as exc:
        raise KnowledgeIndexError(f"Не удалось открыть ChromaDB в {settings.DB_PATH}: {exc}") from exc

    # Коллекция выбирается по имени. Раньше брался collections[0] — при
    # появлении второй коллекции бот молча подключался не к той.
    available = []
    try:
        available = [item.name for item in client.list_collections()]
    except Exception as exc:
        log_exception("Не удалось получить список коллекций ChromaDB", exc)

    if settings.CHROMA_COLLECTION_NAME not in available:
        raise KnowledgeIndexError(
            f"Коллекция '{settings.CHROMA_COLLECTION_NAME}' не найдена в {settings.DB_PATH}. "
            f"Доступны: {available or 'ни одной'}. Запустите индексацию: python indexer.py"
        )

    collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
    _verify_embedding_compatibility(collection)

    try:
        total = collection.count()
    except Exception:
        total = -1
    logger.info(
        "База знаний подключена | коллекция=%s | блоков=%s",
        settings.CHROMA_COLLECTION_NAME,
        total if total >= 0 else "?",
    )

    logger.info("Загрузка модели эмбеддингов: %s", settings.EMBEDDING_MODEL)
    try:
        embedder = load_embedder()
    except Exception as exc:
        raise KnowledgeIndexError(
            f"Не удалось загрузить модель '{settings.EMBEDDING_MODEL}': {exc}"
        ) from exc

    return KnowledgeIndex(collection, embedder)
