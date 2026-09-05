"""Тесты для модуля управления пулом прокси (Proxy Pool, Failover, Round-Robin)."""

import time
from bot import settings
from bot.proxy import (
    ProxyEndpoint,
    ProxyPool,
    build_proxy_url,
    build_single_proxy_url,
    get_active_proxy,
    normalize_proxy_url,
)


def test_build_single_proxy_url():
    assert build_single_proxy_url("1.2.3.4", 8080) == "http://1.2.3.4:8080"
    assert (
        build_single_proxy_url("1.2.3.4", 8080, "myuser", "mypass")
        == "http://myuser:mypass@1.2.3.4:8080"
    )
    assert (
        build_single_proxy_url("1.2.3.4", 8000, "user", "pass", proxy_type="socks5")
        == "socks5://user:pass@1.2.3.4:8000"
    )


def test_normalize_proxy_url():
    assert normalize_proxy_url("127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert normalize_proxy_url("127.0.0.1:8080", default_scheme="socks5") == "socks5://127.0.0.1:8080"
    assert normalize_proxy_url("socks5://user:pass@10.0.0.1:1080") == "socks5://user:pass@10.0.0.1:1080"
    # Очистка задвоенной схемы
    assert normalize_proxy_url("http://socks5://user:pass@10.0.0.1:1080") == "socks5://user:pass@10.0.0.1:1080"
    assert normalize_proxy_url("https://1.2.3.4:8443") == "https://1.2.3.4:8443"


def test_proxy_endpoint_health_and_safe_repr():
    ep = ProxyEndpoint(url="http://user:secret123@proxy.example.com:8080")
    
    # Пароль не должен утекать в safe_repr
    assert "secret123" not in ep.safe_repr
    assert "proxy.example.com:8080" in ep.safe_repr
    assert ep.is_available() is True
    assert ep.failure_count == 0

    # Ошибка
    now = 1000.0
    ep.mark_failure(cooldown_seconds=30.0)
    assert ep.failure_count == 1
    assert ep.is_available(now=now) is False

    # Сброс успеха
    ep.mark_success()
    assert ep.failure_count == 0
    assert ep.cooldown_until == 0.0
    assert ep.success_count == 1
    assert ep.is_available() is True


def test_proxy_pool_failover_strategy():
    urls = [
        "http://proxy1.com:8080",
        "http://proxy2.com:8080",
        "http://proxy3.com:8080",
    ]
    pool = ProxyPool(urls=urls, strategy="failover", cooldown_seconds=60.0)
    
    # 1. По умолчанию берётся первый прокси
    assert pool.get_next_proxy() == "http://proxy1.com:8080"

    # 2. Первый прокси упал -> переключение на второй
    pool.report_failure("http://proxy1.com:8080")
    assert pool.get_next_proxy() == "http://proxy2.com:8080"

    # 3. Второй тоже упал -> переключение на третий
    pool.report_failure("http://proxy2.com:8080")
    assert pool.get_next_proxy() == "http://proxy3.com:8080"

    # 4. Успех на третьем
    pool.report_success("http://proxy3.com:8080")
    assert pool.get_next_proxy() == "http://proxy3.com:8080"

    # 5. Если все прокси упали, выбирается лучший (с наименьшим оставшимся кулдауном)
    pool.report_failure("http://proxy3.com:8080")
    assert pool.get_next_proxy() is not None


def test_proxy_pool_round_robin_strategy():
    urls = [
        "http://proxy1.com:8080",
        "http://proxy2.com:8080",
        "http://proxy3.com:8080",
    ]
    pool = ProxyPool(urls=urls, strategy="round-robin", cooldown_seconds=60.0)

    p1 = pool.get_next_proxy()
    p2 = pool.get_next_proxy()
    p3 = pool.get_next_proxy()
    p4 = pool.get_next_proxy()

    assert p1 == "http://proxy1.com:8080"
    assert p2 == "http://proxy2.com:8080"
    assert p3 == "http://proxy3.com:8080"
    assert p4 == "http://proxy1.com:8080"


def test_proxy_pool_status():
    urls = ["http://proxy1.com:8080", "http://proxy2.com:8080"]
    pool = ProxyPool(urls=urls, strategy="failover", cooldown_seconds=60.0)
    pool.report_failure("http://proxy1.com:8080")

    status = pool.get_status()
    assert status["total"] == 2
    assert status["strategy"] == "failover"
    assert len(status["endpoints"]) == 2
    assert status["endpoints"][0]["available"] is False
    assert status["endpoints"][0]["failure_count"] == 1
    assert status["endpoints"][1]["available"] is True


def test_proxy_pool_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "USE_PROXY", True)
    monkeypatch.setattr(settings, "PROXY_URLS", ["http://p1.com:8080", "http://p2.com:8080"])
    monkeypatch.setattr(settings, "PROXY_STRATEGY", "round-robin")
    monkeypatch.setattr(settings, "PROXY_COOLDOWN_SECONDS", 45)

    pool = ProxyPool.from_settings()
    assert pool.is_empty is False
    assert pool.get_all_proxies() == ["http://p1.com:8080", "http://p2.com:8080"]
    assert pool.get_status()["strategy"] == "round-robin"


def test_proxy_pool_from_settings_single_socks5(monkeypatch):
    monkeypatch.setattr(settings, "USE_PROXY", True)
    monkeypatch.setattr(settings, "PROXY_URLS", [])
    monkeypatch.setattr(settings, "PROXY_TYPE", "socks5")
    monkeypatch.setattr(settings, "PROXY_HOST", "217.198.9.236")
    monkeypatch.setattr(settings, "PROXY_PORT", 8000)
    monkeypatch.setattr(settings, "PROXY_USERNAME", "myuser")
    monkeypatch.setattr(settings, "PROXY_PASSWORD", "mypass")

    pool = ProxyPool.from_settings()
    assert pool.is_empty is False
    assert pool.get_all_proxies() == ["socks5://myuser:mypass@217.198.9.236:8000"]
    assert pool.get_next_proxy() == "socks5://myuser:mypass@217.198.9.236:8000"


def test_proxy_endpoint_scheme_and_is_socks():
    ep1 = ProxyEndpoint("socks5://user:pass@1.2.3.4:1080")
    assert ep1.scheme == "socks5"
    assert ep1.is_socks is True

    ep2 = ProxyEndpoint("http://1.2.3.4:8080")
    assert ep2.scheme == "http"
    assert ep2.is_socks is False


def test_proxy_pool_mixed_protocols():
    urls = [
        "socks5://user:pass@socks-proxy.com:1080",
        "http://http-proxy.com:8080",
    ]
    pool = ProxyPool(urls=urls, strategy="failover")
    assert pool.get_next_proxy() == "socks5://user:pass@socks-proxy.com:1080"
    pool.report_failure(pool.get_next_proxy())
    assert pool.get_next_proxy() == "http://http-proxy.com:8080"


def test_get_active_proxy_and_build_proxy_url(monkeypatch):
    monkeypatch.setattr(settings, "USE_PROXY", True)
    monkeypatch.setattr(settings, "PROXY_URLS", ["socks5://proxy.com:1080"])
    
    # Реинициализируем глобальный пул для теста
    from bot import proxy as proxy_mod
    proxy_mod.proxy_pool = ProxyPool.from_settings()

    assert get_active_proxy() == "socks5://proxy.com:1080"
    assert build_proxy_url() == "socks5://proxy.com:1080"
    assert build_proxy_url("http") == "http://proxy.com:1080"


def test_modular_proxy_toggles(monkeypatch):
    from bot.app import _create_bot
    from bot.llm import create_providers

    monkeypatch.setattr(settings, "USE_PROXY", True)
    monkeypatch.setattr(settings, "PROXY_URLS", ["socks5://proxy.com:1080"])
    monkeypatch.setattr(settings, "DISCORD_USE_PROXY", False)
    monkeypatch.setattr(settings, "OPENROUTER_USE_PROXY", True)
    monkeypatch.setattr(settings, "FALLBACK_AI_ENABLED", True)
    monkeypatch.setattr(settings, "FALLBACK_AI_USE_PROXY", False)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", "test-model")
    monkeypatch.setattr(settings, "FALLBACK_AI_API_KEY", "test-fallback-key")
    monkeypatch.setattr(settings, "FALLBACK_AI_MODEL", "test-fallback-model")
    monkeypatch.setattr(settings, "FALLBACK_AI_API_URL", "https://fallback.api/v1")

    # Discord подключается напрямую, так как DISCORD_USE_PROXY=False
    bot = _create_bot()
    assert bot is not None

    # OpenRouter использует прокси, а Fallback AI подключен напрямую
    from bot import proxy as proxy_mod
    proxy_mod.proxy_pool = ProxyPool.from_settings()

    # Mock client creation since openai module may not be in local test env
    monkeypatch.setattr("bot.llm._create_openai_client", lambda *args, **kwargs: "fake_client")

    providers = create_providers()
    openrouter = providers[0]
    fallback = providers[1]

    assert openrouter.use_proxy is True
    assert openrouter.current_proxy == "socks5://proxy.com:1080"
    assert fallback.use_proxy is False
    assert fallback.current_proxy is None
