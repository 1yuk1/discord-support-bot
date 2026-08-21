"""Точка входа Discord-бота поддержки SinusSMP.

Вся логика разложена по модулям пакета bot/:
  settings   загрузка settings.toml
  rag        поиск по базе знаний в ChromaDB
   llm        генерация ответов через OpenRouter с резервным провайдером
  handlers   обработка сообщений
  commands   административные команды
  state      состояние диалогов
"""

from bot.app import run

if __name__ == "__main__":
    run()
