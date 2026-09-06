FROM python:3.11-slim

# Провайдер LLM_PROVIDER=local тянет torch/transformers/accelerate (~5 ГБ колёс nvidia-*),
# и нужен он только для локальной разработки. По умолчанию их не ставим: образ собирается
# за пару минут вместо часа. Для сборки с ними:
#   docker compose build --build-arg WITH_LOCAL_LLM=true graph-rag-svc
# либо GRAPH_RAG_WITH_LOCAL_LLM=true в main/.env.
ARG WITH_LOCAL_LLM=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-local.txt ./
RUN pip install -r requirements.txt && \
    if [ "$WITH_LOCAL_LLM" = "true" ]; then \
        pip install -r requirements-local.txt; \
    fi

COPY app ./app
COPY cli ./cli

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
