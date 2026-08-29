"""Интеграционные тесты индексации: запись в ChromaDB и чтение ботом.

Модель эмбеддингов подменяется детерминированной заглушкой: настоящая
multilingual-e5-large-instruct весит больше 2 ГБ, а проверить нужно логику
записи метаданных, выбора коллекции и контроля совместимости.
"""

import hashlib

import pytest

from bot import settings

chromadb = pytest.importorskip("chromadb", reason="ChromaDB не установлен")

VECTOR_SIZE = 32


class FakeEmbedder:
    """Детерминированные векторы из хеша текста."""

    def __init__(self, dimension: int = VECTOR_SIZE) -> None:
        self._dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        vectors = []
        for text in items:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [digest[index % len(digest)] / 255.0 for index in range(self._dimension)]
            if normalize_embeddings:
                norm = sum(value * value for value in raw) ** 0.5 or 1.0
                raw = [value / norm for value in raw]
            vectors.append(raw)

        return FakeArray(vectors[0] if single else vectors)


class FakeArray:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


@pytest.fixture
def chroma_db(tmp_path, monkeypatch):
    db_path = tmp_path / "chroma_db"
    monkeypatch.setattr(settings, "DB_PATH", str(db_path))
    return db_path


def build_index(db_path, knowledge_items, embedder=None, model_name="fake-model"):
    """Собирает коллекцию так же, как это делает indexer.main()."""
    from bot.embeddings import embedding_fingerprint
    from indexer import index_knowledge

    embedder = embedder or FakeEmbedder()
    client = chromadb.PersistentClient(path=str(db_path))
    fingerprint = embedding_fingerprint(embedder, model_name)
    collection = client.create_collection(
        settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", **fingerprint},
    )
    index_knowledge(knowledge_items, embedder, collection)
    return collection


def make_item(item_id, question, answer, synonyms=None):
    return {
        "id": item_id,
        "question": question,
        "synonyms": synonyms or [],
        "category": "test",
        "priority": "medium",
        "for_llm": {
            "problem": question,
            "diagnostics": [],
            "quick_answer": answer,
            "full_solution": answer,
            "steps": [],
            "transfer_to_human": False,
            "immediate_action": "ответить",
        },
    }


def test_index_and_count(chroma_db):
    items = [
        make_item("a", "Как зайти на сервер?", "Используйте актуальную версию"),
        make_item("b", "Как купить донат?", "Через сайт или Boosty"),
    ]
    collection = build_index(chroma_db, items)
    assert collection.count() == 2


def test_metadata_written(chroma_db):
    collection = build_index(chroma_db, [make_item("a", "Вопрос?", "Ответ")])
    stored = collection.get(ids=["a"], include=["metadatas", "documents"])

    assert stored["metadatas"][0]["category"] == "test"
    assert stored["metadatas"][0]["priority"] == "medium"
    assert "## Быстрый ответ" in stored["documents"][0]


def test_embedding_fingerprint_saved(chroma_db):
    collection = build_index(chroma_db, [make_item("a", "Вопрос?", "Ответ")])
    assert collection.metadata["embedding_model"] == "fake-model"
    assert collection.metadata["embedding_dimension"] == str(VECTOR_SIZE)


def test_search_returns_documents(chroma_db, monkeypatch):
    """Поиск возвращает документы из базы.

    Релевантность здесь не проверяется: у заглушки векторы из хеша, семантики
    в них нет. Тест подтверждает механику — запрос доходит до коллекции,
    документы читаются и склеиваются в контекст.
    """
    from bot.rag import KnowledgeIndex

    items = [
        make_item("connect", "Как зайти на сервер?", "Обновите клиент", ["не могу зайти"]),
        make_item("donate", "Как купить донат?", "Через сайт", ["где купить донат"]),
    ]
    build_index(chroma_db, items)

    client = chromadb.PersistentClient(path=str(chroma_db))
    collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
    index = KnowledgeIndex(collection, FakeEmbedder())

    monkeypatch.setattr(settings, "SEARCH_TOP_K", 2)
    monkeypatch.setattr(settings, "CHROMA_DISTANCE_THRESHOLD", 2.0)
    context = index.search("Как купить донат?")

    assert "## Проблема" in context
    assert "Обновите клиент" in context
    assert "Через сайт" in context


def test_search_deduplicates_across_variants(chroma_db, monkeypatch):
    """Одинаковые документы из разных вариантов запроса не должны дублироваться."""
    from bot.rag import KnowledgeIndex

    build_index(chroma_db, [make_item("only", "Сколько стоит?", "Смотрите на сайте")])

    client = chromadb.PersistentClient(path=str(chroma_db))
    collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
    index = KnowledgeIndex(collection, FakeEmbedder())

    monkeypatch.setattr(settings, "SEARCH_TOP_K", 5)
    monkeypatch.setattr(settings, "CHROMA_DISTANCE_THRESHOLD", 2.0)
    # "crjkmr" распознаётся как неверная раскладка, поэтому вариантов два.
    context = index.search("crjkmr")
    assert context.count("## Проблема") == 1



def test_search_returns_empty_for_empty_collection(chroma_db, monkeypatch):
    """Отсутствие результатов — не ошибка, а пустой контекст."""
    from bot.rag import KnowledgeIndex

    client = chromadb.PersistentClient(path=str(chroma_db))
    collection = client.create_collection(
        settings.CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    index = KnowledgeIndex(collection, FakeEmbedder())

    monkeypatch.setattr(settings, "SEARCH_TOP_K", 2)
    assert index.search("любой вопрос") == ""


def test_collection_selected_by_name(chroma_db, monkeypatch):
    """Лишняя коллекция не должна подменять рабочую."""
    from bot.rag import open_knowledge_index

    build_index(chroma_db, [make_item("a", "Вопрос?", "Правильный ответ")])

    client = chromadb.PersistentClient(path=str(chroma_db))
    other = client.create_collection("unrelated_collection")
    other.add(ids=["x"], embeddings=[[0.1] * VECTOR_SIZE], documents=["Мусор"])

    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "fake-model")
    monkeypatch.setattr("bot.rag.load_embedder", lambda *a, **k: FakeEmbedder())

    index = open_knowledge_index()
    assert index.collection_name == settings.CHROMA_COLLECTION_NAME


def test_model_mismatch_rejected(chroma_db, monkeypatch):
    """Смена модели без переиндексации должна давать явную ошибку."""
    from bot.rag import KnowledgeIndexError, open_knowledge_index

    build_index(chroma_db, [make_item("a", "Вопрос?", "Ответ")], model_name="model-a")

    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "model-b")
    monkeypatch.setattr(settings, "STRICT_EMBEDDING_CHECK", True)
    monkeypatch.setattr("bot.rag.load_embedder", lambda *a, **k: FakeEmbedder())

    with pytest.raises(KnowledgeIndexError, match="несовместимы"):
        open_knowledge_index()


def test_model_mismatch_allowed_when_check_disabled(chroma_db, monkeypatch):
    from bot.rag import open_knowledge_index

    build_index(chroma_db, [make_item("a", "Вопрос?", "Ответ")], model_name="model-a")

    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "model-b")
    monkeypatch.setattr(settings, "STRICT_EMBEDDING_CHECK", False)
    monkeypatch.setattr("bot.rag.load_embedder", lambda *a, **k: FakeEmbedder())

    assert open_knowledge_index() is not None


def test_missing_collection_reports_clearly(chroma_db, monkeypatch):
    from bot.rag import KnowledgeIndexError, open_knowledge_index

    chromadb.PersistentClient(path=str(chroma_db)).create_collection("wrong_name")
    monkeypatch.setattr("bot.rag.load_embedder", lambda *a, **k: FakeEmbedder())

    with pytest.raises(KnowledgeIndexError, match="не найдена"):
        open_knowledge_index()


def test_real_knowledge_base_indexes(chroma_db):
    """Настоящая база знаний должна индексироваться целиком."""
    from bot.knowledge import load_knowledge

    items = load_knowledge()
    collection = build_index(chroma_db, items)
    assert collection.count() == len(items)
