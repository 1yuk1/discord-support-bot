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
import aiohttp
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
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_MODEL_TYPE = config.EMBEDDING_MODEL_TYPE
USE_PROXY = config.USE_PROXY
PROXY_HOST = config.PROXY_HOST
PROXY_PORT = config.PROXY_PORT
PROXY_USERNAME = config.PROXY_USERNAME
PROXY_PASSWORD = config.PROXY_PASSWORD
MODEL_CACHE_PATH = config.MODEL_CACHE_PATH
DB_PATH = config.DB_PATH
MAX_HISTORY = config.MAX_HISTORY
TICKET_CATEGORY_ID = config.TICKET_CATEGORY_ID
TICKET_CATEGORY_IDS = set(config.TICKET_CATEGORY_IDS)
BOT_ROLE_ID = config.BOT_ROLE_ID
IGNORED_ROLE_IDS = set(config.IGNORED_ROLE_IDS)
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
    """Возвращает HTTP прокси для подключения."""
    if PROXY_USERNAME and PROXY_PASSWORD:
        return f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
    return f"http://{PROXY_HOST}:{PROXY_PORT}"


def format_embedding_text(text, mode):
    if EMBEDDING_MODEL_TYPE == "e5":
        return f"{mode}: {text}"
    return text


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
embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_PATH)

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
    query_embedding = embedder.encode(format_embedding_text(query, "query")).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    context_parts = []
    docs_list = results.get('documents') or [[]]
    metas_list = results.get('metadatas') or [[]]
    docs = docs_list[0] if docs_list else []
    metas = metas_list[0] if metas_list else []

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
Если в КОНТЕКСТЕ есть блок "## Диагностика", задай только следующий самый важный вопрос, которого не хватает для решения. Не задавай весь список сразу.
Если в истории уже есть ответ на вопрос из диагностики, не повторяй его.
Если данных уже достаточно, сразу дай решение или следующий понятный шаг.
Если информации в контексте НЕТ, вежливо уточни детали проблемы.
Никогда не выдумывай IP-адреса, команды, способы оплаты или сроки, которых нет в базе.

ВАЖНО — Передача на человека:
- Если игрок просит перевести на человека/техподдержку/админа/оператора/специалиста — отвечай: "Я передам ваш тикет старшему специалисту. Пожалуйста, ожидайте, в ближайшее свободное время вам ответят."
- Не задавай дополнительных вопросов если игрок явно просит перевести на человека.

ВАЖНО — Донат:
- На сервере доступны только easyDonate и Boosty.
- Купить донат можно только на https://sinussmp.ru или через Boosty https://boosty.to/ingrog
- Никогда не отправляй игрока покупать донат на easyDonate.com, easydonate.com, easydonate.ru или любой другой сайт easyDonate.
- easyDonate — это платежная система внутри сайта https://sinussmp.ru, а не отдельный магазин доната.
- Для России используется сайт https://sinussmp.ru: оплата проходит через easyDonate, а донат выдается автоматически.
- Для другой страны основной вариант — Boosty: https://boosty.to/ingrog
- Донат через Boosty не выдается автоматически: нужна ручная проверка оплаты и ручная выдача администрацией.
- Если донат через easyDonate не пришел, сначала проси перезайти на режим, указанный при оплате: lite1, lite2, lite3 или PRAC.
- Если это не помогло, запрашивай именно чек от easyDonate с электронной почты, указанной при оплате, а не чек банка. Чек нужен для проверки сбоя автоматической выдачи, а не для обещания ручной выдачи.
- Никогда не говори, что донат через easyDonate будет выдан вручную. Ручная выдача относится только к Boosty.
- Если оплата не проходит, не обещай автоматическое решение. Предложи доступные способы оплаты или пользовательский вариант для ручной проверки.

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

# Настройка прокси для Discord бота
if USE_PROXY:
    proxy_url = get_proxy_url()
    proxy_auth = None
    if PROXY_USERNAME and PROXY_PASSWORD:
        proxy_auth = aiohttp.BasicAuth(PROXY_USERNAME, PROXY_PASSWORD)
        # Для прокси с авторизацией передаём URL без credentials
        proxy_url_for_discord = f"http://{PROXY_HOST}:{PROXY_PORT}"
    else:
        proxy_url_for_discord = proxy_url
    bot = commands.Bot(command_prefix='!', intents=intents, proxy=proxy_url_for_discord, proxy_auth=proxy_auth)
    print(f"🔗 Discord прокси настроен: {PROXY_HOST}:{PROXY_PORT}")
else:
    bot = commands.Bot(command_prefix='!', intents=intents)

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


def create_channel_state():
    return {
        "history": [],
        "human_mode": False,
        "last_message": "",
        "last_message_time": 0,
        "last_answer_time": 0,
        "user_messages": deque(),
        "processed_message_ids": set(),
    }


def is_ticket_channel(channel):
    if not TICKET_CATEGORY_IDS:
        return True
    return channel.category_id in TICKET_CATEGORY_IDS


def member_has_ignored_role(member):
    if not IGNORED_ROLE_IDS or member is None:
        return False
    return any(role.id in IGNORED_ROLE_IDS for role in getattr(member, "roles", []))


def should_use_message_as_question(message):
    content = (message.content or "").strip()
    if content:
        return True
    return len(message.embeds) > 0


def extract_message_text(message):
    parts = []
    content = (message.content or "").strip()
    if content:
        parts.append(content)

    for embed in message.embeds:
        if embed.title:
            parts.append(embed.title.strip())
        if embed.description:
            parts.append(embed.description.strip())
        for field in embed.fields:
            field_text = " ".join(part for part in [field.name, field.value] if part)
            field_text = field_text.strip()
            if field_text:
                parts.append(field_text)

    return "\n".join(part for part in parts if part).strip()

def check_bot_has_role(guild):
    if BOT_ROLE_ID in (None, 0):
        return True
    if guild is None or bot.user is None:
        return False
    
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
    if bot.user is not None:
        print(f'ID бота: {bot.user.id}')
    if TICKET_CATEGORY_IDS:
        print(f'📂 Категории тикетов: {sorted(TICKET_CATEGORY_IDS)}')
    else:
        print('📂 Категории тикетов: все категории')
    if IGNORED_ROLE_IDS:
        print(f'🚫 Игнорируемые роли: {sorted(IGNORED_ROLE_IDS)}')
    print('─────────────────────────')

@bot.event
async def on_message(message):
    if not is_ticket_channel(message.channel):
        await bot.process_commands(message)
        return

    if bot.user is not None and message.author.bot and message.author.id == bot.user.id:
        return

    if message.author.bot and not should_use_message_as_question(message):
        return

    if not message.author.bot and member_has_ignored_role(message.author):
        return

    channel_id = message.channel.id
    
    if channel_id not in conversation_histories:
        conversation_histories[channel_id] = create_channel_state()
    
    channel_data = conversation_histories[channel_id]
    if message.id in channel_data["processed_message_ids"]:
        return

    message_text = extract_message_text(message)
    if not message_text:
        return

    channel_data["processed_message_ids"].add(message.id)
    if len(channel_data["processed_message_ids"]) > 200:
        channel_data["processed_message_ids"] = set(list(channel_data["processed_message_ids"])[-100:])
    
    if not message.author.bot:
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
        message_text
    )
    
    if channel_data["human_mode"]:
        return
    
    if check_duplicate_message(channel_data, message_text):
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
    if is_human_transfer(message_text):
        transfer_answer = "Я передам ваш тикет старшему специалисту. Пожалуйста, ожидайте, в ближайшее свободное время вам ответят."
        await message.channel.send(transfer_answer)
        
        channel_data["last_message"] = message_text
        channel_data["last_message_time"] = time.time()
        channel_data["last_answer_time"] = time.time()
        channel_data["human_mode"] = True
        
        log_message(
            channel_id,
            message.author.id,
            str(message.author),
            message_text,
            bot_response=transfer_answer,
            is_human_transfer=True
        )
        
        author_label = "Система" if message.author.bot else "Пользователь"
        channel_data["history"].append(f"{author_label}: {message_text}")
        channel_data["history"].append(f"Бот: {transfer_answer}")
        if len(channel_data["history"]) > MAX_HISTORY * 2:
            channel_data["history"] = channel_data["history"][-MAX_HISTORY * 2:]
        return

    async with message.channel.typing():
        answer = generate_answer(message_text, channel_data["history"])
    
    await message.channel.send(answer)
    
    channel_data["last_message"] = message_text
    channel_data["last_message_time"] = time.time()
    
    # Обновляем время ответа ТОЛЬКО если ответ успешный
    if answer and not answer.startswith("⚠️"):
        channel_data["last_answer_time"] = time.time()
    
    log_message(
        channel_id,
        message.author.id,
        str(message.author),
        message_text,
        bot_response=answer
    )
    
    author_label = "Система" if message.author.bot else "Пользователь"
    channel_data["history"].append(f"{author_label}: {message_text}")
    channel_data["history"].append(f"Бот: {answer}")
    
    if len(channel_data["history"]) > MAX_HISTORY * 2:
        channel_data["history"] = channel_data["history"][-MAX_HISTORY * 2:]
    
    if is_human_transfer(answer):
        channel_data["human_mode"] = True
        log_message(
            channel_id,
            bot.user.id if bot.user is not None else 0,
            str(bot.user) if bot.user is not None else "bot",
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
        conversation_histories[channel_id] = create_channel_state()
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
