"""Шаги пайплайна ноутбука в порядке ноутбука — от вопроса до готовых сообщений для LLM.

1. Быстрый ответ по нормализованному исходному вопросу; найден — сразу результат, LLM не нужна.
2. Переписывание вопроса с историей (только если история есть).
3. Гибридные кандидаты (pgvector + BM25, RRF) → LLM-реранк → выбор документов и соседей →
   ссылки → сообщения [system, user] для финального ответа.

Ответ модели здесь не генерируется: это делает вызывающий (`process_job`), стримя `messages`.
"""

import dataclasses
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.api.schemas import GenerateRequest
from app.graph.embeddings import embed_texts_with_model
from app.pipeline.context import build_context, select_context
from app.pipeline.hybrid import CHUNK_DENSE_K, CHUNK_LEXICAL_K, hybrid_candidates
from app.pipeline.index import Candidate, KnowledgeIndex, SessionFactory
from app.pipeline.links import find_links
from app.pipeline.llm import LlmChat
from app.pipeline.prompts import build_messages
from app.pipeline.quick_answers import QuickAnswer, find_quick_answer
from app.pipeline.rerank import rerank
from app.pipeline.rewrite import contextualize_question
from app.pipeline.text import normalize_variant

# (тексты, kind) -> (векторы, модель); по умолчанию — клиент embedding-svc.
Embedder = Callable[..., Awaitable[tuple[list[list[float]], str]]]


@dataclass
class RetrievalResult:
    quick_answer: QuickAnswer | None
    messages: list[dict]
    selected: list[Candidate]
    links: list[tuple[str, str]]
    rerank_tokens: int | None
    query_model: str
    retrieval_ms: float
    # Запрос, по которому искали (после переписывания) — для аналитики.
    search_query: str = ""
    # Документы с оценками из aggregate_documents — для аналитики.
    documents: list[dict] = field(default_factory=list)


async def _embed_query(embed: Embedder, text: str) -> tuple[list[float], str]:
    vectors, model = await embed([text], kind="query")
    return vectors[0], model


async def run_retrieval(
    job: GenerateRequest,
    llm: LlmChat,
    index: KnowledgeIndex,
    session_factory: SessionFactory,
    embed: Embedder = embed_texts_with_model,
) -> RetrievalResult:
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    question = job.question
    history = list(job.history or [])
    # Один снимок на весь запрос: позиции из BM25 и pgvector относятся к одному набору чанков,
    # даже если индекс обновится посреди обработки.
    view = index.snapshot()

    # 1. Быстрый ответ (ячейки 33–34): эмбеддинг считается от нормализованного вопроса.
    quick_text = normalize_variant(question)
    if quick_text:
        quick_vector, quick_model = await _embed_query(embed, quick_text)
        async with session_factory() as session:
            quick = await find_quick_answer(session, question, quick_vector, quick_model)
        if quick is not None:
            return RetrievalResult(
                quick_answer=quick, messages=[], selected=[], links=[], rerank_tokens=None,
                query_model=quick_model, retrieval_ms=elapsed_ms(), search_query=question,
            )

    # 2. Переписывание вопроса с историей (ячейка 35).
    search_query = await contextualize_question(llm, question, history)

    # 3. Кандидаты (ячейка 21): dense top-60 + BM25 top-60 → RRF top-40.
    query_vector, query_model = await _embed_query(embed, search_query)
    async with session_factory() as session:
        dense = await view.dense_search(session, query_vector, query_model, k=CHUNK_DENSE_K)
        links = await find_links(session, query_vector, query_model)
    lexical = view.lexical_search(search_query, k=CHUNK_LEXICAL_K)
    candidates = [
        dataclasses.replace(view.candidate(idx), rrf_score=score)
        for idx, score in hybrid_candidates(dense, lexical)
    ]

    # 4. Реранк (ячейка 23). Без кандидатов LLM не вызывается: ранжировать нечего.
    if candidates:
        ranked, rerank_tokens = await rerank(llm, search_query, candidates)
    else:
        ranked, rerank_tokens = [], None

    # 5. Документы, соседи, контекст (ячейка 25) и сообщения (ячейка 27). Язык ответа и вопрос
    # в промпте — исходные, не переписанные.
    selected, documents = select_context(ranked, view.neighbour)
    context = build_context(selected)
    messages = build_messages(question, history, context, links)

    return RetrievalResult(
        quick_answer=None,
        messages=messages,
        selected=selected,
        links=links,
        rerank_tokens=rerank_tokens,
        query_model=query_model,
        retrieval_ms=elapsed_ms(),
        search_query=search_query,
        documents=documents,
    )
