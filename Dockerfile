FROM python:3.14-slim

WORKDIR /app

ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
ENV USER=root
ENV APP_BASE_DIR=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY discord_bot.py config.py ./
COPY settings.toml ./

RUN mkdir -p chroma_db logs model_cache

CMD ["python", "discord_bot.py"]
