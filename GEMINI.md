# Архитектурные правила Discord Support Bot

## 1. Работа с сетью и прокси (bot/proxy.py, bot/llm.py, bot/app.py)
- **Универсальная нормализация схем**: При построении адресов прокси (build_single_proxy_url, normalize_proxy_url, ProxyPool) никогда не хардкодить схему http://. Всегда учитывать тип прокси (socks5, socks5h, socks4, http, https) или явную схему из адреса.
- **Поддержка SOCKS5 в OpenAI/OpenRouter**: Создавать httpx.Client(proxy=resolved_proxy) напрямую. Не оборачивать в httpx.HTTPTransport(proxy=...), так как HTTPTransport ломает SOCKS5.
- **Модульное (раздельное) проксирование**:
  - DISCORD_USE_PROXY: проксирование трафика Discord (REST API + WebSocket Gateway).
  - OPENROUTER_USE_PROXY: проксирование основного AI (OpenRouter).
  - FALLBACK_AI_USE_PROXY: проксирование резервного AI.
  - Скачивание скриншотов из тикетов привязано к DISCORD_USE_PROXY.
  - Каждый сервис должен иметь возможность работать напрямую без прокси при отключенном модульном флаге.

## 2. Pterodactyl и настройки (egg.json, scripts/generate_settings.py, bot/settings.py)
- **Синхронизация параметров**: При добавлении или изменении параметров конфигурации всегда обновлять три места:
  1. egg.json (переменные панели управления Pterodactyl).
  2. scripts/generate_settings.py (генерация settings.toml из переменных окружения).
  3. bot/settings.py (рантайм валидация и дефолты).
- **Диагностика сбоев**: При невозможности подключиться к Discord через прокси логировать внятные подсказки по устранению (неверный протокол, недоступность хоста, рекомендация отключить DISCORD_USE_PROXY при хостинге вне РФ), предотвращая краш-луп контейнера.
