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


@pytest.mark.parametrize("field", ["id", "question", "synonyms", "category", "for_llm"])
def test_missing_required_field_rejected(field):
    item = make_item()
    del item[field]
    with pytest.raises(KnowledgeError, match="отсутствуют поля"):
        validate_item(item, "test.json")


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


def test_transfer_to_human_must_be_bool():
    item = make_item()
    item["for_llm"]["transfer_to_human"] = "yes"
    with pytest.raises(KnowledgeError, match="transfer_to_human"):
        validate_item(item, "test.json")


def test_answer_required():
    """Блок без ответа бесполезен: LLM получит пустой контекст."""
    item = make_item()
    item["for_llm"]["quick_answer"] = ""
    item["for_llm"]["full_solution"] = "   "
    with pytest.raises(KnowledgeError, match="quick_answer"):
        validate_item(item, "test.json")


def test_search_text_combines_question_and_synonyms():
    text = build_search_text(make_item())
    assert "Как зайти на сервер?" in text
    assert "как подключиться" in text


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
    assert metadata["transfer_to_human"] is False


def test_synonym_conflicts_detected():
    conflicts = find_synonym_conflicts([
        make_item(id="first", synonyms=["читы", "моды"]),
        make_item(id="second", synonyms=["Читы", "правила"]),
    ])
    assert conflicts == {"читы": ["first", "second"]}


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
