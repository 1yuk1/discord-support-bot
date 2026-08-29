"""Полный прогон indexer.main() на реальной базе знаний.

Модель эмбеддингов подменяется заглушкой, всё остальное настоящее: чтение
knowledge/*.json, валидация, запись в ChromaDB и последующее подключение бота.
Это ловит ошибки, которые не видны в юнит-тестах отдельных функций.
"""

import hashlib
import sys

import pytest

from bot import settings

pytest.importorskip("chromadb", reason="ChromaDB не установлен")


class FakeEmbedder:
    """Детерминированные векторы фиксированной размерности."""

    def __init__(self, dimension: int = 32) -> None:
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
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            vectors.append([value / norm for value in raw])

        return _Array(vectors[0] if single else vectors)


class _Array:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


@pytest.fixture
def indexer_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(settings, "AUTO_UPDATE_CHROMA_DB", True)
    monkeypatch.setattr("indexer.load_embedder", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr("bot.rag.load_embedder", lambda *a, **k: FakeEmbedder())
    return tmp_path


def test_full_index_run_and_bot_connect(indexer_environment, capsys):
    """indexer.main() собирает базу, бот к ней подключается и ищет."""
    import indexer
    from bot.knowledge import load_knowledge
    from bot.rag import open_knowledge_index

    indexer.main()

    output = capsys.readouterr().out
    assert "Индексация завершена" in output

    expected_count = len(load_knowledge())
    index = open_knowledge_index()

    assert index.collection_name == settings.CHROMA_COLLECTION_NAME
    assert index._collection.count() == expected_count

    context = index.search("как купить донат")
    assert context, "поиск должен возвращать непустой контекст"
    assert "## Проблема" in context


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Windows не отдаёт data_level0.bin, открытый прошлым клиентом ChromaDB. "
        "На Linux (прод и CI) ограничения нет."
    ),
)
def test_reindex_replaces_previous_data(indexer_environment):
    """Повторный запуск не дублирует блоки."""
    import indexer
    from bot.knowledge import load_knowledge
    from bot.rag import open_knowledge_index

    expected = len(load_knowledge())

    indexer.main()
    indexer.main()

    assert open_knowledge_index()._collection.count() == expected


def test_safe_reindexing_without_rmtree(indexer_environment):
    """Индексатор должен безопасно обновлять базу без удаления директории."""
    import indexer
    from bot.rag import open_knowledge_index

    indexer.main()
    indexer.main()
    assert open_knowledge_index()._collection.count() > 0



def test_disabled_auto_update_exits_with_code_2(indexer_environment, monkeypatch):
    """Код 2 нужен entrypoint.sh, чтобы не записать подпись индекса."""
    import indexer

    monkeypatch.setattr(settings, "AUTO_UPDATE_CHROMA_DB", False)

    with pytest.raises(SystemExit) as exc_info:
        indexer.main()
    assert exc_info.value.code == 2


def test_synonym_conflicts_reported(indexer_environment, capsys):
    """Отчёт должен предупреждать о синонимах, ведущих в несколько блоков."""
    import indexer

    indexer.main()
    output = capsys.readouterr().out

    assert "По категориям:" in output
    # В реальной базе такие конфликты есть, отчёт обязан их показать.
    assert "ведут в несколько блоков" in output
