"""Формирование текста для эмбеддингов и загрузка модели.

ВАЖНО: этот модуль используется и ботом, и индексатором. Раньше функция
format_embedding_text была скопирована в discord_bot.py и indexer.py с
комментарием «держать в синхроне». Если копии расходятся, индексация идёт
одним префиксом, а поиск другим — качество падает без единой ошибки в логах.
Теперь источник один.
"""

from bot import settings

QUERY_INSTRUCTION = (
    "Найди наиболее релевантный блок базы знаний для вопроса игрока "
    "Minecraft-сервера SinusSMP."
)

MODE_QUERY = "query"
MODE_PASSAGE = "passage"


def format_embedding_text(text: str, mode: str, model_type: str | None = None) -> str:
    """Возвращает текст с префиксом под конкретный тип модели.

    e5-instruct: запросы требуют формата Instruct/Query, документы идут как есть.
    e5: оба вида получают префикс "query: " / "passage: ".
    Остальные модели (bge и т.п.) префиксов не требуют.
    """
    effective_type = model_type or settings.EMBEDDING_MODEL_TYPE

    if effective_type == "e5-instruct":
        if mode == MODE_QUERY:
            return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"
        return text
    if effective_type == "e5":
        return f"{mode}: {text}"
    return text


def load_embedder(model_name: str | None = None, cache_folder: str | None = None):
    """Загружает SentenceTransformer. Импорт внутри — модуль тяжёлый."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name or settings.EMBEDDING_MODEL,
        cache_folder=cache_folder or settings.MODEL_CACHE_PATH,
    )


def embedding_fingerprint(embedder, model_name: str | None = None) -> dict[str, str]:
    """Отпечаток модели для метаданных коллекции.

    Позволяет боту при старте убедиться, что база собрана той же моделью.
    Смена модели без переиндексации не ломает бота явно — он просто начинает
    искать мусор, и это крайне трудно заметить по логам.
    """
    name = model_name or settings.EMBEDDING_MODEL
    dimension = ""
    try:
        value = embedder.get_sentence_embedding_dimension()
        if value:
            dimension = str(value)
    except Exception:
        dimension = ""

    return {
        "embedding_model": name,
        "embedding_model_type": settings.EMBEDDING_MODEL_TYPE,
        "embedding_dimension": dimension,
    }
