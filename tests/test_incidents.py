"""Тесты инцидентов.

Инциденты правятся и командами, и руками в файле, поэтому парсер обязан быть
терпимым: пропущенный [id], лишние пробелы и битая секция не должны валить бота.
"""

import pytest

from bot import incidents, settings


@pytest.fixture(autouse=True)
def incidents_file(tmp_path, monkeypatch):
    """Свой файл инцидентов и чистый кэш на каждый тест."""
    path = tmp_path / "incidents.md"
    monkeypatch.setattr(settings, "INCIDENTS_FILE", str(path))
    monkeypatch.setattr(settings, "INCIDENTS_ENABLED", True)
    incidents.store.invalidate()
    incidents.store._cache = []
    yield path
    incidents.store.invalidate()
    incidents.store._cache = []


# ── Парсинг ──────────────────────────────────────────────────────────────────
def test_parses_incident_with_explicit_id():
    text = """## [auth-down] Не работает вход
- создан: 2026-08-09 14:20 UTC, автор: Kakas

Авторизация недоступна, чиним.
"""
    parsed = incidents.parse(text)
    assert len(parsed) == 1
    assert parsed[0].id == "auth-down"
    assert parsed[0].title == "Не работает вход"
    assert "Авторизация недоступна" in parsed[0].body
    assert parsed[0].author == "Kakas"


def test_id_generated_from_title_when_missing():
    """Файл правили руками и [id] не написали — не повод падать."""
    parsed = incidents.parse("## Сервер лежит\n\nЧиним, ожидайте.")
    assert len(parsed) == 1
    assert parsed[0].id == "server-lezhit"


def test_parses_multiple_incidents():
    text = """## [one] Первый

Текст один.

## [two] Второй

Текст два.
"""
    parsed = incidents.parse(text)
    assert [item.id for item in parsed] == ["one", "two"]


def test_incident_without_body_is_skipped():
    """Заголовок без текста бесполезен для промпта."""
    parsed = incidents.parse("## [empty] Пустой\n\n## [ok] Нормальный\n\nЕсть текст.")
    assert [item.id for item in parsed] == ["ok"]


def test_duplicate_ids_are_disambiguated():
    parsed = incidents.parse("## [same] Раз\n\nТекст.\n\n## [same] Два\n\nТекст.")
    assert len({item.id for item in parsed}) == 2


def test_multiline_body_preserved():
    text = """## [multi] Многострочный

Первая строка.
Вторая строка.

- пункт списка
"""
    body = incidents.parse(text)[0].body
    assert "Вторая строка." in body
    assert "- пункт списка" in body


def test_comments_ignored():
    text = "<!-- служебный комментарий -->\n## [x] Тест\n\nТекст."
    assert len(incidents.parse(text)) == 1


def test_empty_file_gives_empty_list():
    assert incidents.parse("") == []
    assert incidents.parse("просто текст без заголовков") == []


def test_slugify_handles_edge_cases():
    assert incidents.slugify("Не работает вход") == "ne-rabotaet-vhod"
    assert incidents.slugify("!!!") == "incident"
    assert incidents.slugify("Тест", taken={"test"}) == "test-2"


# ── Добавление и удаление ────────────────────────────────────────────────────
def test_add_creates_file_and_incident(incidents_file):
    incident = incidents.add("Не работает вход", "Чиним авторизацию.", author="Kakas")

    assert incidents_file.exists()
    assert incident.id == "ne-rabotaet-vhod"

    active = incidents.active()
    assert len(active) == 1
    assert active[0].title == "Не работает вход"


def test_add_appends_without_losing_previous(incidents_file):
    incidents.add("Первый", "Текст один.")
    incidents.add("Второй", "Текст два.")
    assert len(incidents.active()) == 2


def test_add_rejects_empty_fields():
    with pytest.raises(ValueError):
        incidents.add("", "текст")
    with pytest.raises(ValueError):
        incidents.add("заголовок", "   ")


def test_add_respects_limit(monkeypatch):
    monkeypatch.setattr(incidents, "MAX_INCIDENTS", 2)
    incidents.add("Раз", "Текст.")
    incidents.add("Два", "Текст.")

    with pytest.raises(ValueError, match="лимит"):
        incidents.add("Три", "Текст.")


def test_remove_deletes_incident():
    incident = incidents.add("Не работает вход", "Чиним.")
    removed = incidents.remove(incident.id)

    assert removed is not None
    assert removed.id == incident.id
    assert incidents.active() == []


def test_remove_unknown_id_returns_none():
    incidents.add("Есть", "Текст.")
    assert incidents.remove("нет-такого") is None
    assert len(incidents.active()) == 1


def test_roundtrip_survives_rewrite():
    """render → parse не должен терять данные: файл перезаписывается целиком."""
    incidents.add("Не работает вход", "Первая строка.\nВторая строка.", author="Kakas")
    incidents.store.invalidate()

    restored = incidents.active()[0]
    assert restored.title == "Не работает вход"
    assert "Вторая строка." in restored.body
    assert restored.author == "Kakas"


# ── Кэш и перечитывание ──────────────────────────────────────────────────────
def test_changes_picked_up_without_restart(incidents_file):
    """Правка файла руками должна применяться на следующем же вопросе."""
    incidents.add("Первый", "Текст.")
    assert len(incidents.active()) == 1

    incidents_file.write_text(
        "## [manual] Добавлен руками\n\nТекст.\n", encoding="utf-8"
    )
    incidents.store.invalidate()

    active = incidents.active()
    assert len(active) == 1
    assert active[0].id == "manual"


def test_missing_file_is_not_an_error():
    assert incidents.active() == []


def test_broken_file_does_not_crash(incidents_file):
    """Мусор в файле — пустой список, а не исключение в ответе игроку."""
    incidents_file.write_text("### не тот уровень\nбез заголовков", encoding="utf-8")
    incidents.store.invalidate()
    assert incidents.active() == []


# ── Блок промпта ─────────────────────────────────────────────────────────────
def test_prompt_block_empty_without_incidents():
    """Нет инцидентов — нет и расхода токенов."""
    assert incidents.prompt_block() == ""


def test_prompt_block_contains_incident_text():
    incidents.add("Не работает вход", "Авторизация недоступна, чиним.")
    block = incidents.prompt_block()

    assert "Не работает вход" in block
    assert "Авторизация недоступна" in block
    assert "приоритет выше базы знаний" in block.lower()


def test_prompt_block_forbids_promising_deadlines():
    incidents.add("Сервер лежит", "Чиним.")
    assert "сроков" in incidents.prompt_block().lower()


def test_prompt_block_empty_when_disabled(monkeypatch):
    incidents.add("Сервер лежит", "Чиним.")
    monkeypatch.setattr(settings, "INCIDENTS_ENABLED", False)
    assert incidents.prompt_block() == ""
