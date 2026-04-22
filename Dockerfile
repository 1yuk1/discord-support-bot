FROM python:3.14-slim

WORKDIR /home/container

ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
ENV USER=root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p chroma_db logs model_cache

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]