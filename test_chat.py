import sys
sys.stdout.reconfigure(encoding='utf-8')

from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import httpx
from openai import OpenAI
from pathlib import Path
from collections import deque
import time
import json

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
    if PROXY_USERNAME and PROXY_PASSWORD:
        return f"socks5://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
    return f"socks5://{PROXY_HOST}:{PROXY_PORT}"


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
# ТЕСТОВЫЕ ВОПРОСЫ ИЗ indexer.py
# ==============================================================================
test_questions = [
    "Как зайти на сервер? Какой IP?",
    "Я забыл пароль, как сбросить?",
    "Не могу подключиться, connection timed out",
    "Большой пинг на сервере, всё лагает",
    "Донат не пришёл, оплатил но не выдали",
    "Какие моды разрешены на сервере?",
    "Вылетело во время ПВП, пропали вещи",
    "Не прошёл проверку на бота",
    "Нашёл баг на сервере",
    "Как купить донат из другой страны?",
]

# ==============================================================================
# АВТОМАТИЧЕСКИЙ ТЕСТ
# ==============================================================================
def run_auto_test():
    session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    chat_log = {
        "session_id": session_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": None,
        "messages": []
    }

    print("\n" + "=" * 60)
    print("🧪 АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ БОТА ПОДДЕРЖКИ")
    print("=" * 60)
    print(f"Вопросов для теста: {len(test_questions)}")
    print("=" * 60 + "\n")

    conversation_history = deque(maxlen=MAX_HISTORY * 2)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] 👤 Вы: {question}")
        
        # Логирование вопроса
        chat_log["messages"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "role": "user",
            "content": question
        })

        # Поиск контекста для отладки
        context = search_knowledge(question)
        print(f"   🔍 Найдено контекста: {len(context)} символов")

        # Генерация ответа
        answer = generate_answer(question, list(conversation_history))
        print(f"   🤖 Бот: {answer[:200]}..." if len(answer) > 200 else f"   🤖 Бот: {answer}")

        # Логирование ответа
        chat_log["messages"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "role": "bot",
            "content": answer,
            "human_transfer": is_human_transfer(answer)
        })

        conversation_history.append(f"Пользователь: {question}")
        conversation_history.append(f"Бот: {answer}")

        time.sleep(1)  # Пауза между вопросами

    # Сохранение лога
    chat_log["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log_filename = save_chat_log(chat_log, session_id)
    
    print("\n" + "=" * 60)
    print(f"✅ Тестирование завершено")
    print(f"📄 Лог сохранён: {log_filename}")
    print("=" * 60)


if __name__ == "__main__":
    run_auto_test()
