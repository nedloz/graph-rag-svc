import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Double, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base

# Единственное место в этом сервисе, объявляющее размерность. Значение приходит из
# EMBEDDING_DIM (общий для всей системы корневой .env) и проверяется на старте против
# embedding-svc и фактического типа колонок — см. app/core/startup_checks.py.
EMBEDDING_DIM = settings.embedding_dim


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"
    __table_args__ = {"schema": "graph"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_type: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_model: Mapped[str | None] = mapped_column(Text)
    ontology_version: Mapped[str] = mapped_column(Text, nullable=False)
    document_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    stats_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = {"schema": "graph"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    name_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(Text)
    attributes_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    aliases: Mapped[list["EntityAlias"]] = relationship(back_populates="entity", cascade="all, delete-orphan")


class EntityAlias(Base):
    __tablename__ = "entity_aliases"
    __table_args__ = {"schema": "graph"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("graph.entities.id", ondelete="CASCADE"), nullable=False)
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    alias_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="extraction")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    entity: Mapped[Entity] = relationship(back_populates="aliases")


class Relation(Base):
    __tablename__ = "relations"
    __table_args__ = {"schema": "graph"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("graph.entities.id", ondelete="CASCADE"), nullable=False)
    object_entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("graph.entities.id", ondelete="CASCADE"), nullable=False)
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(REAL)
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("graph.extraction_runs.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EntitySourceChunk(Base):
    __tablename__ = "entity_source_chunks"
    __table_args__ = {"schema": "graph"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("graph.entities.id", ondelete="CASCADE"))
    relation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("graph.relations.id", ondelete="CASCADE"))
    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("library.chunks.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Read-only views onto `library` — graph-rag-svc never writes here (see
# init.sql grants: graph_user has SELECT-only on schema library).
# ---------------------------------------------------------------------------


class LibraryDocument(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "library"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    ingest_status: Mapped[str | None] = mapped_column(Text)


class LibraryChunk(Base):
    __tablename__ = "chunks"
    __table_args__ = {"schema": "library"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("library.documents.id"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class LibraryChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = {"schema": "library"}

    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("library.chunks.id"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Имя модели, которой посчитан вектор. Поиск обязан фильтровать по нему: векторы разных
    # моделей лежат в разных пространствах и несравнимы между собой (README, §13.1).
    embedding_model: Mapped[str] = mapped_column(Text)


class CurationBlocklist(Base):
    """Имена, которые построение графа не должно создавать заново.

    Без стоп-листа ручное удаление мусорной сущности бессмысленно: билдер работает через
    upsert и создаст её снова на следующей же пересборке.
    """

    __tablename__ = "curation_blocklist"
    __table_args__ = {"schema": "graph"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name_normalized: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetrievalRun(Base):
    """Аналитика одного прогона ретривала — общая для обоих проектов.

    Живёт в нейтральной схеме `rag`, а не в `chat.rag_runs`: у соседнего проекта нет
    ни chat-svc, ни схемы `chat`, а аналитика нужна обоим. Колонка `consumer` (имя
    вызывающего сервиса из X-Service-Name) разделяет трафик команд.
    """

    __tablename__ = "retrieval_runs"
    __table_args__ = {"schema": "rag"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    consumer: Mapped[str | None] = mapped_column(Text)
    session_id: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str | None] = mapped_column(Text)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    route_used: Mapped[str | None] = mapped_column(Text)
    query_embedding_model: Mapped[str | None] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    retrieved_doc_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    scores_json: Mapped[dict | None] = mapped_column(JSONB)
    retrieval_ms: Mapped[float | None] = mapped_column(Double)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="completed")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
