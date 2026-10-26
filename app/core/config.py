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

    # Размерность векторов модели эмбеддингов. Сверяется при запуске и с embedding-svc,
    # и с фактическим типом колонок в БД (app/core/startup_checks.py) — расхождение
    # раньше обнаруживалось только в момент вставки/поиска. Смена модели требует
    # пересчёта всех чанков и всего графа (см. README, §13.1).
    embedding_dim: int = 1024
    embedding_startup_retries: int = 30
    embedding_startup_retry_wait_sec: int = 5

    # Generation
    llm_provider: str = "stub"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "mistral"
    transformers_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cpu"

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

    # Бюджет истории диалога в промпте. Количество сообщений не ограничено — при превышении
    # бюджета отбрасываются самые старые реплики.
    max_history_chars: int = 24000
    max_history_message_chars: int = 4000
    # Сколько последних реплик подмешивать в извлечение сущностей, чтобы разрешались
    # уточняющие и местоименные вопросы («а что насчёт пересдачи по ней же?»).
    history_messages_for_extraction: int = 4

    max_chunks_to_use: int = 8
    max_chunk_chars: int = 2500
    max_context_chars: int = 12000
    max_question_chars: int = 4000
    fail_closed_on_injection: bool = False

    # Ontology extraction (offline graph-builder + online query-time extraction)
    graph_extraction_model: str = ""
    graph_window_size: int = 4
    graph_window_step: int = 3
    entity_similarity_threshold: float = 0.86
    graph_traversal_max_hops: int = 2

    # Retrieval routing: always_graph | always_vector | heuristic
    judge_mode: str = "always_graph"

    # Запуск пересборки графа через HTTP (/internal/graph/build). В прод-среде отключается:
    # пересборка меняет ответы сразу у обеих команд, поэтому кнопка не должна быть доступна
    # случайно (README, §12.3, вопрос 11). CLI-запуск флагом не ограничивается.
    graph_build_api_enabled: bool = True


settings = Settings()
