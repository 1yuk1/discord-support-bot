FROM python:3.14-slim

WORKDIR /home/container

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache \
    HF_HOME=/home/container/model_cache \
    USER=root

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-версия torch: GPU в контейнере нет, полный пакет весит на порядок больше.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# В образ попадает только bootstrap. Код бота скачивается из репозитория при
# запуске, поэтому правки логики не требуют пересборки образа.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && sed -i 's/\r$//' /entrypoint.sh

CMD ["bash", "/entrypoint.sh"]
