"""Проверка регистрации команд на настоящих объектах discord.py.

Декораторы app_commands валидируют имена, описания и типы аргументов в момент
объявления, поэтому опечатка в описании или конфликт имён ловится только здесь.
Тесты handlers/commands работают на заглушках и такого не увидят.
"""

import asyncio

import discord
import pytest
from discord.ext import commands as dcommands

from bot.commands import RussianCommandTranslator, register_commands, set_translator

ADMIN = discord.Permissions(administrator=True).value

EXPECTED_SLASH = {
    "clear_history",
    "config reload",
    "help",
    "incident add",
    "incident list",
    "incident remove",
    "model save",
    "model set",
    "model show",
    "ping",
    "reminders off",
    "reminders on",
    "reminders status",
    "resume_bot",
    "start",
    "status",
    "stop",
    "summarize",
}


class FakeAgent:
    def summarize_ticket(self, transcript):
        return "сводка"

    def compose_reminder(self, transcript):
        return "напоминание"


@pytest.fixture
def bot():
    intents = discord.Intents.default()
    intents.message_content = True
    # help_command=None: своя /help конфликтует со встроенной.
    instance = dcommands.Bot(command_prefix="!", intents=intents, help_command=None)
    register_commands(instance, FakeAgent())
    return instance


def slash_names(bot) -> set[str]:
    return {command.qualified_name for command in bot.tree.walk_commands()}


def test_all_slash_commands_registered(bot):
    assert EXPECTED_SLASH.issubset(slash_names(bot))


def test_admin_commands_hidden_from_regular_users(bot):
    """default_permissions прячет команду из списка у не-админов.

    Без этого игроки видели бы /stop в автодополнении.
    """
    for command in bot.tree.get_commands():
        permissions = command.default_permissions
        assert permissions is not None, f"{command.name} без default_permissions"
        assert permissions.value == ADMIN, f"{command.name}: не только для админов"


def test_prefix_fallback_commands_registered(bot):
    """Аварийный путь на случай, если tree.sync не прошёл."""
    names = {command.qualified_name for command in bot.walk_commands()}
    for expected in ("stop", "start", "bot_status", "incident list", "reminders status"):
        assert expected in names


def test_incident_remove_has_autocomplete(bot):
    """Иначе id пришлось бы копировать руками из /incident list."""
    command = bot.tree.get_command("incident").get_command("remove")
    parameter = command.get_parameter("incident")
    assert parameter.autocomplete is not None


def test_summarize_limit_is_optional(bot):
    command = bot.tree.get_command("summarize")
    assert command.get_parameter("limit").required is False


def test_descriptions_present(bot):
    """Discord отклоняет команду без описания."""
    for command in bot.tree.walk_commands():
        description = getattr(command, "description", "")
        assert description and description.strip(), f"{command.qualified_name}: нет описания"


# ── Русские имена команд ─────────────────────────────────────────────────────
def _ru_payload(bot):
    """Payload дерева команд с русскими локализациями."""

    async def build():
        translator = RussianCommandTranslator()
        await bot.tree.set_translator(translator)
        return [
            await command.get_translated_payload(bot.tree, translator)
            for command in bot.tree.get_commands()
        ]

    return asyncio.run(build())


def test_top_level_commands_have_russian_names(bot):
    """У каждой команды верхнего уровня есть имя для ru-локали.

    Без локализации русскоязычный админ ищет /стоп и не находит ничего.
    """
    for payload in _ru_payload(bot):
        localizations = payload.get("name_localizations") or {}
        assert localizations.get("ru"), f"{payload['name']}: нет русского имени"


def test_subcommands_and_options_have_russian_names(bot):
    """Подкоманды групп и параметры тоже переводятся."""
    for payload in _ru_payload(bot):
        for option in payload.get("options") or []:
            localizations = option.get("name_localizations") or {}
            assert localizations.get("ru"), (
                f"{payload['name']} {option['name']}: нет русского имени"
            )


def test_russian_names_are_valid_for_discord(bot):
    """Discord принимает только строчные буквы, цифры, дефис и подчёркивание."""
    import re

    pattern = re.compile(r"^[-_\w]{1,32}$")

    def check(name: str, where: str) -> None:
        assert pattern.match(name), f"{where}: недопустимое имя {name!r}"
        assert name == name.lower(), f"{where}: имя не в нижнем регистре {name!r}"

    for payload in _ru_payload(bot):
        russian = (payload.get("name_localizations") or {}).get("ru")
        check(russian, payload["name"])
        for option in payload.get("options") or []:
            option_russian = (option.get("name_localizations") or {}).get("ru")
            if option_russian:
                check(option_russian, f"{payload['name']} {option['name']}")


def test_other_locales_keep_english_names(bot):
    """Для остальных локалей переводчик молчит, имена остаются английскими."""

    async def translate(locale):
        translator = RussianCommandTranslator()
        context = discord.app_commands.TranslationContext(
            location=discord.app_commands.TranslationContextLocation.command_name,
            data=None,
        )
        return await translator.translate(
            discord.app_commands.locale_str("stop"), locale, context
        )

    assert asyncio.run(translate(discord.Locale.russian)) == "стоп"
    assert asyncio.run(translate(discord.Locale.american_english)) is None
