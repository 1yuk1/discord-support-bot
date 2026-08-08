"""Проверка, что все модули пакета импортируются.

Тяжёлые зависимости (chromadb, sentence_transformers, openai) заменяются
заглушками: на CI и локально их может не быть, но синтаксис и связи между
модулями проверить нужно.
"""

import importlib
import sys
import types

import pytest

MODULES = [
    "bot.settings",
    "bot.logging_setup",
    "bot.text_utils",
    "bot.escalation",
    "bot.filters",
    "bot.limits",
    "bot.embeddings",
    "bot.knowledge",
    "bot.prompt",
    "bot.rag",
    "bot.llm",
    "bot.ticket_logs",
    "bot.state",
    "bot.discord_client",
    "bot.handlers",
    "bot.commands",
    "bot.app",
]


@pytest.fixture(scope="module", autouse=True)
def stub_heavy_dependencies():
    """Подставляет заглушки для тяжёлых библиотек, если их нет."""
    created: list[str] = []

    if "chromadb" not in sys.modules:
        try:
            importlib.import_module("chromadb")
        except ImportError:
            module = types.ModuleType("chromadb")
            module.PersistentClient = object
            sys.modules["chromadb"] = module
            created.append("chromadb")

    if "sentence_transformers" not in sys.modules:
        try:
            importlib.import_module("sentence_transformers")
        except ImportError:
            module = types.ModuleType("sentence_transformers")
            module.SentenceTransformer = object
            sys.modules["sentence_transformers"] = module
            created.append("sentence_transformers")

    if "openai" not in sys.modules:
        try:
            importlib.import_module("openai")
        except ImportError:
            module = types.ModuleType("openai")
            module.OpenAI = object
            sys.modules["openai"] = module
            created.append("openai")

    yield

    for name in created:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_indexer_imports():
    assert importlib.import_module("indexer") is not None


def test_entry_point_exposes_run():
    module = importlib.import_module("bot.app")
    assert callable(module.run)
