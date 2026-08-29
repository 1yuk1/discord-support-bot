"""Тесты для модуля управления пулом прокси (Proxy Pool, Failover, Round-Robin)."""

import time
from bot import settings
from bot.proxy import (
    ProxyEndpoint,
    ProxyPool,
    build_single_proxy_url,
)


def test_build_single_proxy_url():
    assert build_single_proxy_url("1.2.3.4", 8080) == "http://1.2.3.4:8080"
    assert (
        build_single_proxy_url("1.2.3.4", 8080, "myuser", "mypass")
        == "http://myuser:mypass@1.2.3.4:8080"
    )


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
