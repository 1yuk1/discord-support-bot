import sys
sys.stdout.reconfigure(encoding='utf-8')

from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import httpx
from aiohttp_socks import ProxyConnector
from openai import OpenAI
from pathlib import Path
from collections import deque
import time
import json
import os

import config

# Краткие алиасы из config
AI_PROVIDER = config.AI_PROVIDER
GROQ_API_KEY = config.GROQ_API_KEY
GROQ_MODEL = config.GROQ_MODEL
LOCAL_API_URL = config.LOCAL_API_URL
LOCAL_API_KEY = config.LOCAL_API_KEY
LOCAL_MODEL = config.LOCAL_MODEL
USE_PROXY = config.USE_PROXY
PROXY_HOST = config.PROXY_HOST
PROXY_PORT = config.PROXY_PORT
PROXY_USERNAME = config.PROXY_USERNAME
PROXY_PASSWORD = config.PROXY_PASSWORD
MODEL_CACHE_PATH = config.MODEL_CACHE_PATH
DB_PATH = config.DB_PATH
MAX_HISTORY = config.MAX_HISTORY
HUMAN_TRANSFER_PHRASES = config.HUMAN_TRANSFER_PHRASES
LOGS_PATH = config.LOGS_PATH


# ==============================================================================
# НАСТРОЙКА ПРОКСИ И AI КЛИЕНТОВ
# ==============================================================================
def get_proxy_url():
    """Возвращает URL прокси для подключения."""
    if PROXY_USERNAME and PROXY_PASSWORD:
        return f"socks5://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
    return f"socks5://{PROXY_HOST}:{PROXY_PORT}"


# Инициализация AI клиентов
if AI_PROVIDER == "groq":
    print(f"🤖 AI провайдер: Groq (модель: {GROQ_MODEL})")
    if USE_PROXY:
        proxy_url = get_proxy_url()
        print(f"🔄 Groq будет использовать прокси: {PROXY_HOST}:{PROXY_PORT}")
        http_client = httpx.Client(transport=httpx.HTTPTransport(proxy=proxy_url))
        groq_client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)
    openai_client = None

elif AI_PROVIDER == "local":
    print(f"🤖 AI провайдер: Локальная модель (URL: {LOCAL_API_URL}, модель: {LOCAL_MODEL})")
    openai_client = OpenAI(
        api_key=LOCAL_API_KEY,
        base_url=LOCAL_API_URL
    )
    groq_client = None

else:
    print(f"❌ Неизвестный AI_PROVIDER: {AI_PROVIDER}")
    exit()

# ==============================================================================
# ПОДКЛЮЧЕНИЕ К БАЗЕ ЗНАНИЙ (ChromaDB)
# ==============================================================================
print(f"🔍 Подключение к базе данных...")

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collections = client.list_collections()
    if not collections:
        print("❌ Ошибка: В базе нет коллекций!")
        exit()

    collection_name = collections[0].name
    collection = client.get_collection(collection_name)
    print(f"✅ База подключена. Коллекция: {collection_name}")

except Exception as e:
    print(f"❌ Ошибка подключения к ChromaDB: {e}")
    exit()

# ==============================================================================
# ЗАГРУЗКА МОДЕЛИ ДЛЯ ЭМБЕДДИНГОВ
# ==============================================================================
print("⬇️ Загрузка модели для поиска...")
embedder = SentenceTransformer('intfloat/multilingual-e5-base', cache_folder=MODEL_CACHE_PATH)
print("✅ Модель загружена")

# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================
Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)

def save_chat_log(chat_log, session_id):
    """Сохраняет лог чата в JSON файл."""
    filename = f"{LOGS_PATH}/console_chat_{session_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chat_log, f, ensure_ascii=False, indent=2)
    return filename

# ==============================================================================
# ФУНКЦИИ AI
# ==============================================================================
def search_knowledge(query):
    query_embedding = embedder.encode("query: " + query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    context_parts = []
    docs = results['documents'][0]
    metas = results['metadatas'][0]

    for doc, meta in zip(docs, metas):
        if isinstance(meta, dict) and meta.get('hidden') == True:
            continue
        context_parts.append(doc)

    return "\n\n".join(context_parts)

def generate_answer(user_input, conversation_history):
    context = search_knowledge(user_input)
    history_text = "\n".join(conversation_history) if conversation_history else "Нет предыдущих сообщений"

    system_instruction = """Ты — опытный агент поддержки SinusSMP.
Твоя задача: помочь игроку, используя КОНТЕКСТ.
Если в КОНТЕКСТЕ есть блок "## Диагностика", сначала задай эти вопросы игроку, прежде чем давать решение.
Если информации в контексте НЕТ, вежливо уточни детали проблемы.
Никогда не выдумывай IP-адреса или команды, которых нет в базе.

ВАЖНО — Передача на человека:
- Если игрок просит перевести на человека/техподдержку/админа/оператора/специалиста — отвечай: "Я передам ваш тикет старшему специалисту. Пожалуйста, ожидайте, в ближайшее свободное время вам ответят."
- Не задавай дополнительных вопросов если игрок явно просит перевести на человека.

- НИКОГДА не показывай игроку техническую информацию.
- Будь вежлив и краток.
- Учитывай контекст предыдущих сообщений из истории диалога.
- Не здоровайся повторно и не повторяй вопросы, на которые игрок уже ответил в истории диалога."""

    if context:
        user_message = f"""КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ИСТОРИЯ ДИАЛОГА:
{history_text}

ТЕКУЩИЙ ВОПРОС ИГРОКА:
{user_input}"""
    else:
        user_message = f"""КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
(Информация не найдена — попробуй уточнить у игрока детали проблемы и предложить передать вопрос старшему специалисту)

ИСТОРИЯ ДИАЛОГА:
{history_text}

ТЕКУЩИЙ ВОПРОС ИГРОКА:
{user_input}"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

    try:
        if AI_PROVIDER == "groq":
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content

        elif AI_PROVIDER == "local":
            response = openai_client.chat.completions.create(
                model=LOCAL_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            return "⚠️ Временная перегрузка сервиса. Попробуйте через минуту."
        elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
            return "⚠️ Нет подключения к сервису. Попробуйте позже."
        else:
            return "⚠️ Произошла ошибка. Попробуйте ещё раз."

def is_human_transfer(text):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in HUMAN_TRANSFER_PHRASES)


# ==============================================================================
# КОНСОЛЬНЫЙ ЧАТ
# ==============================================================================
def console_chat():
    session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    chat_log = {
        "session_id": session_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": None,
        "messages": []
    }

    print("\n" + "=" * 50)
    print("🎮 КОНСОЛЬНЫЙ ЧАТ С БОТОМ ПОДДЕРЖКИ SinusSMP")
    print("=" * 50)
    print("\n💡 Подсказки:")
    print("   • Введите ваш вопрос и нажмите Enter")
    print("   • Введите 'exit' или 'quit' для выхода")
    print("   • Введите 'clear' для очистки истории диалога")
    print("   • Введите 'reset' для выхода из режима человека")
    print("=" * 50 + "\n")

    conversation_history = deque(maxlen=MAX_HISTORY * 2)
    human_mode = False

    while True:
        try:
            user_input = input("👤 Вы: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('exit', 'quit', 'выход'):
                print("\n👋 До свидания!")
                break

            # Логирование сообщения пользователя
            chat_log["messages"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "role": "user",
                "content": user_input
            })

            if user_input.lower() == 'clear':
                conversation_history.clear()
                print("✅ История диалога очищена\n")
                continue

            if user_input.lower() == 'reset':
                human_mode = False
                print("✅ Бот возобновил работу\n")
                continue

            # Проверка: если игрок просит перевести на человека
            if is_human_transfer(user_input):
                transfer_answer = "Я передам ваш тикет старшему специалисту. Пожалуйста, ожидайте, в ближайшее свободное время вам ответят."
                print(f"\n🤖 Бот: {transfer_answer}\n")
                human_mode = True
                conversation_history.append(f"Пользователь: {user_input}")
                conversation_history.append(f"Бот: {transfer_answer}")
                chat_log["messages"].append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "role": "bot",
                    "content": transfer_answer,
                    "human_transfer": True
                })
                continue

            if human_mode:
                print("📝 [Режим человека активен — бот ожидает ответа оператора]\n")
                conversation_history.append(f"Пользователь: {user_input}")
                continue

            # Генерация ответа
            print("\n🤖 Бот печатает...", end="\r")
            answer = generate_answer(user_input, list(conversation_history))
            print(" " * 30, end="\r")  # Очистить строку "Бот печатает..."

            print(f"🤖 Бот: {answer}\n")

            conversation_history.append(f"Пользователь: {user_input}")
            conversation_history.append(f"Бот: {answer}")

            # Логирование ответа бота
            chat_log["messages"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "role": "bot",
                "content": answer,
                "human_transfer": is_human_transfer(answer)
            })

            # Проверка на передачу человеку в ответе бота
            if is_human_transfer(answer):
                human_mode = True
                print("⚠️ [Тикет передан старшему специалисту]\n")

        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except EOFError:
            print("\n\n👋 До свидания!")
            break

    # Сохранение лога при выходе
    chat_log["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log_filename = save_chat_log(chat_log, session_id)
    print(f"\n📄 Лог сессии сохранён: {log_filename}")


if __name__ == "__main__":
    console_chat()
