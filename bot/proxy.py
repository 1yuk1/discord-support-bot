"""Модуль управления пулом прокси (Proxy Pool, Failover, Round-Robin)."""

from dataclasses import dataclass
import random
import threading
import time
from urllib.parse import quote, unquote, urlparse

from bot import settings
from bot.logging_setup import logger

SUPPORTED_SCHEMES = ("http", "https", "socks5", "socks5h", "socks4")


def normalize_proxy_url(
    raw_url_or_host: str,
    default_scheme: str = "http",
    port: int | None = None,
    username: str = "",
    password: str = "",
) -> str:
    """Приводит адрес прокси к каноническому виду scheme://user:pass@host:port.

    Корректно обрабатывает:
    - адреса без схемы (127.0.0.1:8000 -> http://127.0.0.1:8000)
    - ошибочно задвоенные схемы (http://socks5://1.2.3.4:8000 -> socks5://1.2.3.4:8000)
    - адреса с отдельным логином/паролем или уже встроенными в URL
    """
    raw = str(raw_url_or_host or "").strip()
    if not raw:
        raw = "127.0.0.1"

    while "://" in raw:
        first_scheme, rest = raw.split("://", 1)
        if "://" in rest:
            raw = rest.lstrip("/")
        else:
            raw = f"{first_scheme}://{rest}"
            break

    parsed_scheme = None
    if "://" in raw:
        parsed_scheme, raw = raw.split("://", 1)

    effective_scheme = (
        parsed_scheme or default_scheme or getattr(settings, "PROXY_TYPE", "http") or "http"
    ).lower().strip()
    if effective_scheme not in SUPPORTED_SCHEMES:
        effective_scheme = "http"

    userinfo = ""
    host_port = raw.lstrip("/")
    if "@" in host_port:
        userinfo, host_port = host_port.rsplit("@", 1)

    effective_user = username
    effective_pass = password
    if userinfo:
        if ":" in userinfo:
            u, p = userinfo.split(":", 1)
            effective_user = unquote(u)
            effective_pass = unquote(p)
        else:
            effective_user = unquote(userinfo)

    host_part = host_port.split("/")[0].strip()
    parsed_port = port
    if ":" in host_part:
        h, pt = host_part.rsplit(":", 1)
        host_part = h
        try:
            parsed_port = int(pt)
        except ValueError:
            pass

    clean_host = host_part.strip() or "127.0.0.1"
    effective_port = parsed_port if parsed_port is not None else getattr(settings, "PROXY_PORT", 10808)

    auth_part = ""
    if effective_user and effective_pass:
        auth_part = f"{quote(effective_user)}:{quote(effective_pass)}@"
    elif effective_user:
        auth_part = f"{quote(effective_user)}@"

    return f"{effective_scheme}://{auth_part}{clean_host}:{effective_port}"


def build_single_proxy_url(
    host: str = "127.0.0.1",
    port: int = 10808,
    username: str = "",
    password: str = "",
    proxy_type: str = "http",
) -> str:
    """Формирует URL одиночного прокси из настроек с правильной схемой и очисткой хоста."""
    raw_host = str(host or "127.0.0.1").strip()
    detected_scheme = None
    if "://" in raw_host:
        detected_scheme, raw_host = raw_host.split("://", 1)

    clean_host = raw_host.lstrip("/").split("@")[-1].split(":")[0].strip() or "127.0.0.1"
    selected_scheme = (
        proxy_type or detected_scheme or getattr(settings, "PROXY_TYPE", "http") or "http"
    ).lower().strip()
    if selected_scheme not in SUPPORTED_SCHEMES:
        selected_scheme = "http"

    effective_port = port if port is not None else getattr(settings, "PROXY_PORT", 10808)
    if username and password:
        return f"{selected_scheme}://{quote(username)}:{quote(password)}@{clean_host}:{effective_port}"
    return f"{selected_scheme}://{clean_host}:{effective_port}"



@dataclass
class ProxyEndpoint:
    """Отдельный прокси-сервер с отслеживанием доступности."""

    url: str
    failure_count: int = 0
    cooldown_until: float = 0.0
    success_count: int = 0
    last_used_time: float = 0.0

    @property
    def scheme(self) -> str:
        """Схема прокси (http, socks5 и т.д.)."""
        try:
            return urlparse(self.url).scheme.lower()
        except Exception:
            return "http"

    @property
    def is_socks(self) -> bool:
        """True, если прокси работает по протоколу SOCKS."""
        return self.scheme in ("socks5", "socks5h", "socks4")

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
                        "scheme": e.scheme,
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
        if not getattr(settings, "USE_PROXY", False):
            return cls(urls=[], strategy="failover")

        urls = list(settings.PROXY_URLS) if hasattr(settings, "PROXY_URLS") and settings.PROXY_URLS else []
        default_type = getattr(settings, "PROXY_TYPE", "http") or "http"
        if not urls:
            single = build_single_proxy_url(
                settings.PROXY_HOST,
                settings.PROXY_PORT,
                settings.PROXY_USERNAME,
                settings.PROXY_PASSWORD,
                proxy_type=default_type,
            )
            urls = [single]
        else:
            urls = [normalize_proxy_url(u, default_scheme=default_type) for u in urls]

        strategy = getattr(settings, "PROXY_STRATEGY", "failover")
        cooldown = getattr(settings, "PROXY_COOLDOWN_SECONDS", 60)
        return cls(urls=urls, strategy=strategy, cooldown_seconds=cooldown)


proxy_pool = ProxyPool.from_settings()


def get_active_proxy() -> str | None:
    """Возвращает активный прокси-URL из пула, если проксирование включено."""
    if not getattr(settings, "USE_PROXY", False):
        return None
    return proxy_pool.get_next_proxy()


def build_proxy_url(scheme: str | None = None) -> str:
    """Возвращает рабочий прокси-URL с гарантией схемы (для обратной совместимости)."""
    active = get_active_proxy()
    if active:
        if scheme:
            clean_scheme = scheme.lower().strip()
            if "://" in active:
                _, rest = active.split("://", 1)
                return f"{clean_scheme}://{rest}"
            return normalize_proxy_url(active, default_scheme=clean_scheme)
        return active

    return build_single_proxy_url(
        settings.PROXY_HOST,
        settings.PROXY_PORT,
        settings.PROXY_USERNAME,
        settings.PROXY_PASSWORD,
        proxy_type=scheme or getattr(settings, "PROXY_TYPE", "http"),
    )

