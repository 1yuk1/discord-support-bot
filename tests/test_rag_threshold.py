import pytest
from unittest.mock import MagicMock
from bot import settings
from bot.rag import KnowledgeIndex


class MockEmbedder:
    def encode(self, text, normalize_embeddings=True):
        class Arr:
            def tolist(self):
                return [0.1] * 32
        return Arr()


def test_distance_threshold_rejects_irrelevant(monkeypatch):
    monkeypatch.setattr(settings, 'CHROMA_DISTANCE_THRESHOLD', 0.35)
    monkeypatch.setattr(settings, 'ENABLE_HYBRID_SEARCH', False)

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        'documents': [['# Server Rules\nAll rules...']],
        'metadatas': [[{'id': 'server_rules'}]],
        'distances': [[0.85]],
    }

    index = KnowledgeIndex(mock_collection, MockEmbedder())
    result = index.search('Weather in Moscow?')
    assert result == ''


def test_distance_threshold_accepts_relevant(monkeypatch):
    monkeypatch.setattr(settings, 'CHROMA_DISTANCE_THRESHOLD', 0.40)
    monkeypatch.setattr(settings, 'ENABLE_HYBRID_SEARCH', False)

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        'documents': [['# Claims\nHow to create claim...']],
        'metadatas': [[{'id': 'claims'}]],
        'distances': [[0.15]],
    }

    index = KnowledgeIndex(mock_collection, MockEmbedder())
    result = index.search('how to claim land')
    assert 'How to create claim' in result


def test_hybrid_search_matches_slash_commands(monkeypatch):
    monkeypatch.setattr(settings, 'ENABLE_HYBRID_SEARCH', True)
    monkeypatch.setattr(settings, 'CHROMA_DISTANCE_THRESHOLD', 0.40)

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        'documents': [['# Unrelated']],
        'metadatas': [[{'id': 'unrelated'}]],
        'distances': [[0.90]],
    }
    mock_collection.get.return_value = {
        'documents': ['# Claims /ps\nUse command /ps for claim'],
    }

    index = KnowledgeIndex(mock_collection, MockEmbedder())
    result = index.search('how does /ps work?')
    assert '/ps' in result
    assert '# Claims /ps' in result
