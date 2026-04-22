import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import discord
from discord.ext import commands
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from pathlib import Path
from collections import deque
import time
import httpx
from aiohttp_socks import ProxyConnector
from openai import OpenAI

import config

# Краткие алиасы из config
DISCORD_TOKEN = config.DISCORD_TOKEN
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
TICKET_CATEGORY_ID = config.TICKET_CATEGORY_ID
BOT_ROLE_ID = config.BOT_ROLE_ID
LOGS_PATH = config.LOGS_PATH
HUMAN_TRANSFER_PHRASES = config.HUMAN_TRANSFER_PHRASES
CHANNEL_COOLDOWN = config.CHANNEL_COOLDOWN
DUPLICATE_CHECK_TIME = config.DUPLICATE_CHECK_TIME
RATE_LIMIT = config.RATE_LIMIT
RATE_WINDOW = config.RATE_WINDOW


# ==============================================================================
# НАСТРОЙКА ПРОКСИ И AI КЛИЕНТОВ
# ==============================================================================
def get_proxy_url():
    """Возвращает URL прокси для подключения."""
    proxy_type = "socks5h"  
    if PROXY_USERNAME and PROXY_PASSWORD:
        return f"{proxy_type}://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
    return f"{proxy_type}://{PROXY_HOST}:{PROXY_PORT}"


def get_http_proxy_url():
    """Возвращает HTTP прокси для Groq API."""
    if PROXY_USERNAME and PROXY_PASSWORD:
        return f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
    return f"http://{PROXY_HOST}:{PROXY_PORT}"


# Инициализация AI клиентов
if AI_PROVIDER == "groq":
    print(f"🤖 AI провайдер: Groq (модель: {GROQ_MODEL})")
    if USE_PROXY:
        proxy_url = get_http_proxy_url()
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
print(f"🔍 Подключение к базе данных в папке: {os.path.abspath(DB_PATH)}...")

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

# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================
Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)

def get_log_filename(channel_id):
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"{LOGS_PATH}/ticket_{channel_id}_{date_str}.json"

def load_ticket_log(channel_id):
    filename = get_log_filename(channel_id)
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_ticket_log(channel_id, log_data):
    filename = get_log_filename(channel_id)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def log_message(channel_id, user_id, username, message, bot_response=None, is_human_transfer=False):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": str(user_id),
        "username": username,
        "message": message,
        "bot_response": bot_response,
        "is_human_transfer": is_human_transfer
    }
    
    log_data = load_ticket_log(channel_id)
    log_data.append(log_entry)
    save_ticket_log(channel_id, log_data)

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

# ==============================================================================
# DISCORD БОТ
# ==============================================================================
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

# Настройка прокси для Discord бота будет выполнена в setup_hook
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def setup_hook():
    """Настройка прокси при запуске бота."""
    if USE_PROXY:
        proxy_url = get_proxy_url()
        # Пересоздаём сессию с прокси
        if hasattr(bot, 'http') and hasattr(bot.http, '_HTTPClient__session'):
            await bot.http.close()
        
        connector = ProxyConnector.from_url(proxy_url)
        import aiohttp
        bot.http._HTTPClient__session = aiohttp.ClientSession(
            connector=connector,
            headers=bot.http._HTTPClient__session.headers if hasattr(bot.http, '_HTTPClient__session') else {}
        )
        print(f"🔗 Прокси настроен: {PROXY_HOST}:{PROXY_PORT}")

conversation_histories = {}

# Rate limiting: общий лимит для всех пользователей
global_message_times = deque()

def check_rate_limit():
    global global_message_times
    current_time = time.time()
    
    while global_message_times and current_time - global_message_times[0] > RATE_WINDOW:
        global_message_times.popleft()
    
    if len(global_message_times) >= RATE_LIMIT:
        return False
    
    global_message_times.append(current_time)
    return True

def check_bot_has_role(guild):
    if BOT_ROLE_ID is None:
        return True
    
    bot_member = guild.get_member(bot.user.id)
    if bot_member is None:
        return False
    
    for role in bot_member.roles:
        if role.id == BOT_ROLE_ID:
            return True
    return False

def check_channel_cooldown(channel_data):
    current_time = time.time()
    last_time = channel_data.get("last_answer_time", 0)
    elapsed = current_time - last_time
    
    if elapsed < CHANNEL_COOLDOWN:
        return int(CHANNEL_COOLDOWN - elapsed)
    return 0

def check_duplicate_message(channel_data, message_content):
    current_time = time.time()
    last_msg = channel_data.get("last_message", "")
    last_time = channel_data.get("last_message_time", 0)
    
    if last_msg == message_content and (current_time - last_time) < DUPLICATE_CHECK_TIME:
        return True
    return False

def is_human_transfer(text):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in HUMAN_TRANSFER_PHRASES)

@bot.event
async def on_ready():
    print(f'✅ Бот запущен: {bot.user}')
    print(f'ID бота: {bot.user.id}')
    print('─────────────────────────')

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if TICKET_CATEGORY_ID and message.channel.category_id != TICKET_CATEGORY_ID:
        await bot.process_commands(message)
        return

    channel_id = message.channel.id
    
    if channel_id not in conversation_histories:
        conversation_histories[channel_id] = {
            "history": [],
            "human_mode": False,
            "last_message": "",
            "last_message_time": 0,
            "last_answer_time": 0,
            "user_messages": deque()
        }
    
    channel_data = conversation_histories[channel_id]
    
    # Проверка флуда пользователя (более 3 сообщений за 10 секунд)
    current_time = time.time()
    user_times = channel_data.get("user_messages", deque())
    while user_times and current_time - user_times[0] > 10:
        user_times.popleft()
    
    user_times.append(current_time)
    channel_data["user_messages"] = user_times
    
    if len(user_times) > 3:
        await message.channel.send("⏳ Не флудите! Подождите перед следующим вопросом.")
        return
    
    log_message(
        channel_id,
        message.author.id,
        str(message.author),
        message.content
    )
    
    if channel_data["human_mode"]:
        return
    
    if check_duplicate_message(channel_data, message.content):
        return
    
    cooldown_remaining = check_channel_cooldown(channel_data)
    if cooldown_remaining > 0:
        await message.channel.send(f"⏳ Подождите {cooldown_remaining} секунд перед следующим вопросом.")
        return
    
    if not check_rate_limit():
        await message.channel.send("⏳ Слишком много сообщений. Подожди минуту.")
        return
    
    if not check_bot_has_role(message.guild):
        return

    # Проверка: если игрок просит перевести на человека
    if is_human_transfer(message.content):
        transfer_answer = "Я передам ваш тикет старшему специалисту. Пожалуйста, ожидайте, в ближайшее свободное время вам ответят."
        await message.channel.send(transfer_answer)
        
        channel_data["last_message"] = message.content
        channel_data["last_message_time"] = time.time()
        channel_data["last_answer_time"] = time.time()
        channel_data["human_mode"] = True
        
        log_message(
            channel_id,
            message.author.id,
            str(message.author),
            message.content,
            bot_response=transfer_answer,
            is_human_transfer=True
        )
        
        channel_data["history"].append(f"Пользователь: {message.content}")
        channel_data["history"].append(f"Бот: {transfer_answer}")
        if len(channel_data["history"]) > MAX_HISTORY * 2:
            channel_data["history"] = channel_data["history"][-MAX_HISTORY * 2:]
        return

    async with message.channel.typing():
        answer = generate_answer(message.content, channel_data["history"])
    
    await message.channel.send(answer)
    
    channel_data["last_message"] = message.content
    channel_data["last_message_time"] = time.time()
    
    # Обновляем время ответа ТОЛЬКО если ответ успешный
    if answer and not answer.startswith("⚠️"):
        channel_data["last_answer_time"] = time.time()
    
    log_message(
        channel_id,
        message.author.id,
        str(message.author),
        message.content,
        bot_response=answer
    )
    
    channel_data["history"].append(f"Пользователь: {message.content}")
    channel_data["history"].append(f"Бот: {answer}")
    
    if len(channel_data["history"]) > MAX_HISTORY * 2:
        channel_data["history"] = channel_data["history"][-MAX_HISTORY * 2:]
    
    if is_human_transfer(answer):
        channel_data["human_mode"] = True
        log_message(
            channel_id,
            bot.user.id,
            str(bot.user),
            "Режим передачи человеку активирован",
            is_human_transfer=True
        )

    await bot.process_commands(message)

# ==============================================================================
# КОМАНДЫ
# ==============================================================================
@bot.command()
@commands.has_permissions(administrator=True)
async def clear_history(ctx):
    channel_id = ctx.channel.id
    if channel_id in conversation_histories:
        conversation_histories[channel_id] = {
            "history": [],
            "human_mode": False,
            "last_message": "",
            "last_message_time": 0,
            "last_answer_time": 0
        }
        await ctx.send("✅ История диалога очищена")
    else:
        await ctx.send("История пуста")

@bot.command()
@commands.has_permissions(administrator=True)
async def resume_bot(ctx):
    channel_id = ctx.channel.id
    if channel_id in conversation_histories:
        conversation_histories[channel_id]["human_mode"] = False
        await ctx.send("✅ Бот возобновил работу")
    else:
        await ctx.send("Нет данных о канале")

@bot.command()
async def ping(ctx):
    await ctx.send(f"Pong! Задержка: {round(bot.latency * 1000)}ms")

# ==============================================================================
# ЗАПУСК
# ==============================================================================
if __name__ == "__main__":
    if USE_PROXY:
        print(f"🔗 Discord бот будет использовать прокси: {PROXY_HOST}:{PROXY_PORT}")
    else:
        print("🔗 Discord бот работает без прокси")
    bot.run(DISCORD_TOKEN)
