"""Модуль управления пулом прокси (Proxy Pool, Failover, Round-Robin)."""

from dataclasses import dataclass
import random
import threading
import time
from urllib.parse import urlparse

from bot import settings
from bot.logging_setup import logger


def build_single_proxy_url(
    host: str = "127.0.0.1",
    port: int = 10808,
    username: str = "",
    password: str = "",
) -> str:
    """Формирует URL прокси из отдельных параметров."""
    if username and password:
        return f"http://{username}:{password}@{host}:{port}"
    return f"http://{host}:{port}"


@dataclass
class ProxyEndpoint:
    """Отдельный прокси-сервер с отслеживанием доступности."""

    url: str
    failure_count: int = 0
    cooldown_until: float = 0.0
    success_count: int = 0
    last_used_time: float = 0.0

    @property
    def safe_repr(self) -> str:
        """Безопасное представление URL без вывода пароля в логи."""
        try:
            parsed = urlparse(self.url)
            netloc = parsed.hostname or "unknown"
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            if parsed.username:
                netloc = f"{parsed.username}:***@{netloc}"
            return f"{parsed.scheme}://{netloc}"
        except Exception:
            return self.url.split("@")[-1] if "@" in self.url else self.url

    def is_available(self, now: float | None = None) -> bool:
        current_time = now if now is not None else time.time()
        return current_time >= self.cooldown_until

    def mark_failure(self, cooldown_seconds: float = 60.0) -> None:
        self.failure_count += 1
        multiplier = min(self.failure_count, 5)
        self.cooldown_until = time.time() + (cooldown_seconds * multiplier)

    def mark_success(self) -> None:
        self.failure_count = 0
        self.cooldown_until = 0.0
        self.success_count += 1


class ProxyPool:
    """Потокобезопасный пул прокси с поддержкой Failover и ротации."""

    def __init__(
        self,
        urls: list[str] | None = None,
        strategy: str = "failover",
        cooldown_seconds: float = 60.0,
    ) -> None:
        clean_urls = [u.strip() for u in (urls or []) if str(u).strip()]
        self._endpoints: list[ProxyEndpoint] = [ProxyEndpoint(u) for u in clean_urls]
        self._strategy: str = strategy.lower()
        self._cooldown_seconds: float = max(cooldown_seconds, 5.0)
        self._index: int = 0
        self._lock = threading.Lock()

    @property
    def is_empty(self) -> bool:
        return len(self._endpoints) == 0

    def get_all_proxies(self) -> list[str]:
        with self._lock:
            return [e.url for e in self._endpoints]

    def get_next_proxy(self, now: float | None = None) -> str | None:
        """Выбирает лучший рабочий прокси по настроенной стратегии."""
        with self._lock:
            if not self._endpoints:
                return None

            current_time = now if now is not None else time.time()
            available = [e for e in self._endpoints if e.is_available(current_time)]

            if not available:
                # Все прокси на кулдауне: выбираем тот, у которого кулдаун истечёт раньше всех
                chosen = min(self._endpoints, key=lambda e: e.cooldown_until)
                logger.warning(
                    "Все прокси из пула на кулдауне, берём ближайший: %s",
                    chosen.safe_repr,
                )
            elif self._strategy == "round-robin":
                chosen = available[self._index % len(available)]
                self._index = (self._index + 1) % len(available)
            elif self._strategy == "random":
                chosen = random.choice(available)
            else:
                # failover: первый доступный
                chosen = available[0]

            chosen.last_used_time = current_time
            return chosen.url

    def report_failure(self, proxy_url: str | None, exc: Exception | None = None) -> None:
        """Регистрирует сбой через указанный прокси и отправляет его в кулдаун."""
        if not proxy_url:
            return

        with self._lock:
            for endpoint in self._endpoints:
                if endpoint.url == proxy_url:
                    endpoint.mark_failure(self._cooldown_seconds)
                    logger.warning(
                        "Сбой прокси %s (ошибка #%s, кулдаун %.0fс) | exc=%s",
                        endpoint.safe_repr,
                        endpoint.failure_count,
                        endpoint.cooldown_until - time.time(),
                        exc,
                    )
                    break

    def report_success(self, proxy_url: str | None) -> None:
        """Сбрасывает счётчик сбоев при успешном запросе."""
        if not proxy_url:
            return

        with self._lock:
            for endpoint in self._endpoints:
                if endpoint.url == proxy_url:
                    endpoint.mark_success()
                    break

    def get_status(self) -> dict:
        """Возвращает информацию о текущем состоянии всех прокси."""
        with self._lock:
            now = time.time()
            return {
                "total": len(self._endpoints),
                "strategy": self._strategy,
                "endpoints": [
                    {
                        "url": e.safe_repr,
                        "available": e.is_available(now),
                        "failure_count": e.failure_count,
                        "success_count": e.success_count,
                        "cooldown_remaining": max(0, int(e.cooldown_until - now)),
                    }
                    for e in self._endpoints
                ],
            }

    @classmethod
    def from_settings(cls) -> "ProxyPool":
        """Создаёт экземпляр пула прокси на основе настроек settings.toml."""
        if not settings.USE_PROXY:
            return cls(urls=[], strategy="failover")

        urls = list(settings.PROXY_URLS) if hasattr(settings, "PROXY_URLS") and settings.PROXY_URLS else []
        if not urls:
            single = build_single_proxy_url(
                settings.PROXY_HOST,
                settings.PROXY_PORT,
                settings.PROXY_USERNAME,
                settings.PROXY_PASSWORD,
            )
            urls = [single]

        strategy = getattr(settings, "PROXY_STRATEGY", "failover")
        cooldown = getattr(settings, "PROXY_COOLDOWN_SECONDS", 60)
        return cls(urls=urls, strategy=strategy, cooldown_seconds=cooldown)


proxy_pool = ProxyPool.from_settings()
