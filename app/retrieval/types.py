import uuid
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    doc_id: uuid.UUID
    text: str
    score: float
    source: str  # "graph" | "vector"
    # Название документа-источника. Собирается здесь, а не потребителем: UUID остаётся
    # обязательным (по нему следующий этап строит ссылки), но человекочитаемое название
    # знает только этот сервис — у него есть доступ к library.documents.
    doc_title: str | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": str(self.chunk_id),
            "doc_id": str(self.doc_id),
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "doc_title": self.doc_title,
        }
