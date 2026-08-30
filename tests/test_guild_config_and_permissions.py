"""Тесты для мультисерверной конфигурации, ролей и прав доступа."""

from types import SimpleNamespace
import pytest

from bot import filters, settings
from bot.logging_setup import channel_label


def test_channel_label_formatting():
    assert channel_label(None) == "unknown"
    assert channel_label(12345) == "12345"
    assert channel_label("ticket-1") == "ticket-1"

    ch = SimpleNamespace(id=112233, name="ticket-001")
    assert channel_label(ch) == "#ticket-001 [112233]"

    ch_no_id = SimpleNamespace(name="general")
    assert channel_label(ch_no_id) == "#general"

    ch_no_name = SimpleNamespace(id=998877)
    assert channel_label(ch_no_name) == "998877"


def test_is_admin_member():
    # Без участника
    assert not filters.is_admin_member(None)

    # По user_id
    user_admin = SimpleNamespace(id=1001, roles=[], guild_permissions=SimpleNamespace(administrator=False))
    assert filters.is_admin_member(user_admin, admin_user_ids=[1001, 1002])
    assert not filters.is_admin_member(user_admin, admin_user_ids=[9999])

    # По discord admin permission
    perm_admin = SimpleNamespace(id=2001, roles=[], guild_permissions=SimpleNamespace(administrator=True))
    assert filters.is_admin_member(perm_admin)

    # По admin role id
    role_admin = SimpleNamespace(
        id=3001,
        roles=[SimpleNamespace(id=555)],
        guild_permissions=SimpleNamespace(administrator=False),
    )
    assert filters.is_admin_member(role_admin, admin_role_ids=[555, 666])
    assert not filters.is_admin_member(role_admin, admin_role_ids=[777])


def test_is_staff_member():
    # Админ всегда персонал
    admin = SimpleNamespace(id=1001, roles=[], guild_permissions=SimpleNamespace(administrator=True))
    assert filters.is_staff_member(admin)

    # Обычный персонал по роли
    staff = SimpleNamespace(
        id=2002,
        roles=[SimpleNamespace(id=777)],
        guild_permissions=SimpleNamespace(administrator=False),
    )
    assert filters.is_staff_member(staff, staff_role_ids=[777])
    assert not filters.is_staff_member(staff, staff_role_ids=[888])

    # Обычный игрок
    player = SimpleNamespace(
        id=3003,
        roles=[SimpleNamespace(id=999)],
        guild_permissions=SimpleNamespace(administrator=False),
    )
    assert not filters.is_staff_member(player, staff_role_ids=[777])


def test_is_ticket_channel_allowed():
    # Без фильтров разрешено всё
    assert filters.is_ticket_channel_allowed(SimpleNamespace(category_id=100))

    # Whitelist ("нигде кроме")
    ch1 = SimpleNamespace(category_id=100)
    ch2 = SimpleNamespace(category_id=200)
    assert filters.is_ticket_channel_allowed(ch1, allowed_category_ids=[100, 101])
    assert not filters.is_ticket_channel_allowed(ch2, allowed_category_ids=[100, 101])

    # Blacklist ("везде кроме")
    assert not filters.is_ticket_channel_allowed(ch1, excluded_category_ids=[100])
    assert filters.is_ticket_channel_allowed(ch2, excluded_category_ids=[100])


def test_guild_config_inheritance(monkeypatch):
    monkeypatch.setattr(
        settings,
        "SERVERS_CONFIG",
        {
            123456: {
                "name": "Custom Server",
                "admin_role_ids": [111],
                "ticket_category_ids": [222],
                "reminder_idle_hours": 3.0,
            }
        },
    )

    # Дефолтный сервер
    default_cfg = settings.get_guild_config(None)
    assert default_cfg.guild_id is None

    # Сервер с переопределениями
    custom_cfg = settings.get_guild_config(123456)
    assert custom_cfg.name == "Custom Server"
    assert custom_cfg.admin_role_ids == [111]
    assert custom_cfg.ticket_category_ids == [222]
    assert custom_cfg.reminder_idle_hours == 3.0
    # Наследование дефолтных значений
    assert custom_cfg.reminder_repeat_hours == settings.REMINDER_REPEAT_HOURS
