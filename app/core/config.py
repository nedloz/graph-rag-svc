from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = "graph-rag-svc"

    internal_auth_mode: str = "mtls"

    app_name: str = "graph-rag-svc"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://graph_user:graph_password@postgres:5432/app_db"

    chat_ws_url: str = "ws://chat-svc:8000/ws/internal/llm"
    embedding_svc_url: str = "http://embedding-svc:8000"

    # Размерность векторов модели эмбеддингов (multilingual-e5-base — 768). Сверяется при
    # запуске и с embedding-svc, и с фактическим типом колонки library.chunk_embeddings
    # (app/core/startup_checks.py) — расхождение раньше обнаруживалось только в момент
    # вставки/поиска. Смена модели требует пересчёта всех чанков (см. README, §13.1).
    embedding_dim: int = 768
    embedding_startup_retries: int = 30
    embedding_startup_retry_wait_sec: int = 5

    # LLM: все вызовы (реранк, переписывание вопроса, ответ) идут в llm-proxy, OpenAI-совместимый
    # API. Провайдера, модель и ключ выбирает прокси (main/infra/llm-proxy/README.md); сервис шлёт
    # model "default" и ключ-заглушку. С mTLS-оверлеем адрес — https://llm-proxy:8443/v1.
    llm_base_url: str = "http://llm-proxy:8080/v1"
    llm_model: str = "default"

    # Как часто сверять индекс базы знаний (BM25 по чанкам) с базой и перестраивать его, если
    # набор проиндексированных чанков изменился (app/pipeline/runtime.py).
    index_refresh_interval_sec: int = 60

    llm_http_concurrency: int = 1
    max_queue_size: int = 100

    # Таймауты вызовов LLM. Для стриминга read-таймаут означает «столько секунд без единого
    # нового фрагмента» — это ловит зависшего провайдера, но не обрывает долгую, однако
    # продвигающуюся генерацию. Без них один зависший вызов вешал воркер навсегда, а при
    # LLM_HTTP_CONCURRENCY=1 — всю очередь.
    llm_request_timeout_sec: int = 180
    llm_connect_timeout_sec: int = 10

    # Сколько ждать подключения WS-моста при отправке события. Раньше ожидание было
    # бесконечным, и недоступность chat-svc вешала очередь целиком.
    bridge_send_timeout_sec: int = 15
    # Сколько ждать первого подключения моста при запуске — только для диагностики,
    # неуспех логируется, но не мешает сервису стартовать (зависимость взаимная).
    bridge_startup_wait_sec: int = 15

    # Сколько ждать готовый ответ в синхронном режиме (response_mode="sync").
    # Соседняя команда подтвердила, что долгий HTTP-запрос их устраивает (README, §12.3).
    sync_response_timeout_sec: int = 300

    # Признаки prompt injection в выбранных чанках всегда пишутся в лог; при true ответ на
    # такой контекст не генерируется (задание завершается ошибкой).
    fail_closed_on_injection: bool = False

    # GraphRAG (DISABLED, app/retrieval/graph_retriever.py). Сколько последних реплик
    # подмешивать в извлечение сущностей и сколько чанков брать из обхода графа.
    history_messages_for_extraction: int = 4
    max_chunks_to_use: int = 8

    # Ontology extraction (offline graph-builder + online query-time extraction)
    graph_extraction_model: str = ""
    graph_window_size: int = 4
    graph_window_step: int = 3
    entity_similarity_threshold: float = 0.86
    graph_traversal_max_hops: int = 2

    # Запуск пересборки графа через HTTP (/internal/graph/build). В прод-среде отключается:
    # пересборка меняет ответы сразу у обеих команд, поэтому кнопка не должна быть доступна
    # случайно (README, §12.3, вопрос 11). CLI-запуск флагом не ограничивается.
    graph_build_api_enabled: bool = False


settings = Settings()
