"""Тесты валидации и подготовки базы знаний."""

import json

import pytest

from bot.knowledge import (
    KnowledgeError,
    build_document,
    build_metadata,
    build_search_text,
    find_synonym_conflicts,
    load_knowledge,
    validate_item,
)


def make_item(**overrides):
    item = {
        "id": "test_block",
        "question": "Как зайти на сервер?",
        "synonyms": ["как подключиться", "не могу войти"],
        "category": "connection",
        "priority": "high",
        "for_llm": {
            "problem": "Игрок не может подключиться",
            "diagnostics": ["Проверить версию"],
            "quick_answer": "Используйте актуальную версию",
            "full_solution": "Подробное решение",
            "steps": ["1. Обновите клиент"],
            "transfer_to_human": False,
            "immediate_action": "Уточнить версию",
        },
    }
    item.update(overrides)
    return item


def test_valid_item_passes():
    validate_item(make_item(), "test.json")


@pytest.mark.parametrize("field", ["id", "question", "category"])
def test_missing_required_field_rejected(field):
    item = make_item()
    del item[field]
    with pytest.raises(KnowledgeError):
        validate_item(item, "test.json")


def test_item_without_for_llm_and_content_rejected():
    item = make_item()
    del item["for_llm"]
    with pytest.raises(KnowledgeError, match="нужен либо for_llm, либо content"):
        validate_item(item, "test.json")


def test_markdown_item_valid():
    md_item = {
        "id": "md_block",
        "question": "Как зайти на сервер",
        "synonyms": [],
        "category": "connect",
        "priority": "medium",
        "content": "Используйте адрес play.sinussmp.ru для подключения.",
    }
    validate_item(md_item, "guide.md")


def test_bad_priority_rejected():
    with pytest.raises(KnowledgeError, match="priority"):
        validate_item(make_item(priority="urgent"), "test.json")


def test_synonyms_must_be_list_of_strings():
    with pytest.raises(KnowledgeError, match="synonyms"):
        validate_item(make_item(synonyms="не список"), "test.json")
    with pytest.raises(KnowledgeError, match="synonyms"):
        validate_item(make_item(synonyms=["ок", 123]), "test.json")


def test_steps_must_be_list_of_strings():
    item = make_item()
    item["for_llm"]["steps"] = "1. шаг"
    with pytest.raises(KnowledgeError, match="steps"):
        validate_item(item, "test.json")


def test_search_text_combines_question_and_content():
    text = build_search_text(make_item())
    assert "Как зайти на сервер?" in text
    assert "как подключиться" in text

    md_item = {
        "id": "md_block",
        "question": "Как зайти на сервер",
        "synonyms": [],
        "category": "connect",
        "content": "Используйте адрес play.sinussmp.ru для подключения.",
    }
    md_text = build_search_text(md_item)
    assert "Как зайти на сервер" in md_text
    assert "play.sinussmp.ru" in md_text


def test_document_contains_sections():
    document = build_document(make_item())
    assert "## Проблема" in document
    assert "## Быстрый ответ" in document
    assert "## Инструкция по шагам" in document


def test_metadata_fields():
    metadata = build_metadata(make_item())
    assert metadata["id"] == "test_block"
    assert metadata["category"] == "connection"
    assert metadata["priority"] == "high"


def test_synonym_conflicts_detected():
    conflicts = find_synonym_conflicts([
        make_item(id="first", synonyms=["читы", "моды"]),
        make_item(id="second", synonyms=["Читы", "правила"]),
    ])
    assert conflicts == {"читы": ["first", "second"]}


def test_markdown_file_parsed_and_chunked(tmp_path):
    md_file = tmp_path / "guide.md"
    md_file.write_text(
        "# Основное руководство\n\n"
        "## Раздел 1: Вход\n"
        "Текст первого раздела.\n\n"
        "## Раздел 2: Донат\n"
        "Текст второго раздела про донат.\n",
        encoding="utf-8",
    )
    items = load_knowledge(tmp_path)
    assert len(items) == 2
    assert items[0]["question"] == "Раздел 1: Вход"
    assert "Текст первого раздела" in items[0]["content"]
    assert items[1]["question"] == "Раздел 2: Донат"
    assert "Текст второго раздела" in items[1]["content"]


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps([make_item(id="same")], ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "b.json").write_text(
        json.dumps([make_item(id="same")], ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(KnowledgeError, match="Дубликат id"):
        load_knowledge(tmp_path)


def test_unknown_placeholder_rejected(tmp_path):
    """Опечатка в подстановке иначе уедет игроку как {SERVER_TYPO}."""
    item = make_item()
    item["for_llm"]["quick_answer"] = "Нужна версия {SERVER_TYPO}"
    (tmp_path / "a.json").write_text(
        json.dumps([item], ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(KnowledgeError, match="неизвестные подстановки"):
        load_knowledge(tmp_path)


def test_known_placeholder_substituted(tmp_path):
    item = make_item()
    item["for_llm"]["quick_answer"] = "Версия {SERVER_RECOMMENDED_VERSION}"
    (tmp_path / "a.json").write_text(
        json.dumps([item], ensure_ascii=False), encoding="utf-8"
    )
    loaded = load_knowledge(tmp_path)
    assert loaded[0]["for_llm"]["quick_answer"] == "Версия 1.21.10"


def test_real_knowledge_base_is_valid():
    """Настоящая база знаний должна проходить валидацию."""
    items = load_knowledge()
    assert len(items) > 0
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))

