"""run_retrieval: порядок шагов ноутбука, быстрый ответ, пустой индекс, язык ответа."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from app.api.schemas import GenerateRequest, HistoryMessage
from app.pipeline import service
from app.pipeline.index import KnowledgeIndex
from app.pipeline.quick_answers import QuickAnswer
from pipeline_fakes import FakeLlm, FakeSession, chunk_row, session_factory

DOC_A = uuid.UUID(int=1)
DOC_B = uuid.UUID(int=2)
ROWS = [
    chunk_row(DOC_A, 0, "Общие положения о пересдачах.", title="Положение о пересдачах"),
    chunk_row(DOC_A, 1, "Пересдача экзамена допускается не более двух раз.", title="Положение о пересдачах"),
    chunk_row(DOC_A, 2, "Комиссия назначается деканатом.", title="Положение о пересдачах"),
    chunk_row(DOC_B, 0, "Стипендия выплачивается ежемесячно.", title="Стипендии"),
]


class FakeEmbed:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    async def __call__(self, texts: list[str], kind: str = "passage"):
        self.calls.append((list(texts), kind))
        return [[0.1, 0.2] for _ in texts], "e5-base"


class Deps:
    """Подменяет поиск быстрых ответов и ссылок, запоминая, с чем их вызвали."""

    def __init__(self, monkeypatch, quick: QuickAnswer | None = None, links=None) -> None:
        self.quick_calls: list[str] = []
        self.links_calls: list[str] = []

        async def fake_quick(session, question, query_vector, model, **kw):
            self.quick_calls.append(question)
            return quick

        async def fake_links(session, query_vector, model, **kw):
            self.links_calls.append(model)
            return list(links or [])

        monkeypatch.setattr(service, "find_quick_answer", fake_quick)
        monkeypatch.setattr(service, "find_links", fake_links)


def _index(rows) -> KnowledgeIndex:
    async def fetch_signature(session):
        return (len(rows), None)

    async def fetch_chunks(session):
        return list(rows)

    index = KnowledgeIndex(fetch_signature=fetch_signature, fetch_chunks=fetch_chunks)
    asyncio.run(index.load(session_factory(FakeSession())))
    return index


def _run(job, llm, index, dense_rows=(), embed=None):
    embed = embed or FakeEmbed()
    factory = session_factory(FakeSession(list(dense_rows)))
    return asyncio.run(service.run_retrieval(job, llm, index, factory, embed=embed)), embed


def test_quick_answer_returns_without_llm(monkeypatch):
    deps = Deps(monkeypatch, quick=QuickAnswer(answer="Закажите в ЛК.", question="Справка", rrf_score=0.041))
    llm = FakeLlm()
    job = GenerateRequest(question="Как получить справку?", history=[HistoryMessage(role="user", content="привет")])

    result, embed = _run(job, llm, _index(ROWS))

    assert result.quick_answer.answer == "Закажите в ЛК."
    assert result.messages == [] and result.selected == []
    assert llm.calls == []  # ни переписывания, ни реранка
    assert deps.quick_calls == ["Как получить справку?"]
    # быстрые ответы ищутся по нормализованному вопросу с kind=query (ячейка 34)
    assert embed.calls == [(["как получить справку"], "query")]
    assert result.query_model == "e5-base"
    assert result.retrieval_ms >= 0


def test_empty_index_gives_honest_no_materials(monkeypatch):
    Deps(monkeypatch)
    llm = FakeLlm()

    result, _ = _run(GenerateRequest(question="Как пересдать экзамен?"), llm, _index([]))

    assert result.quick_answer is None
    assert result.selected == []
    assert result.rerank_tokens is None
    assert llm.calls == []  # реранкать нечего
    system, user = result.messages
    assert system["role"] == "system" and user["role"] == "user"
    assert "Материалы:\nрелевантные материалы не найдены" in user["content"]


def test_full_path_order_and_result(monkeypatch):
    deps = Deps(monkeypatch, links=[("Расписание", "http://ruz.hse.ru")])
    rerank_reply = '{"ranked": [{"id": 1, "score": 0.9}, {"id": 3, "score": 0.2}]}'
    llm = FakeLlm((rerank_reply, 400))
    dense = [SimpleNamespace(chunk_id=ROWS[1].id, distance=0.1), SimpleNamespace(chunk_id=ROWS[3].id, distance=0.4)]

    result, embed = _run(GenerateRequest(question="Сколько раз можно пересдать экзамен?"), llm, _index(ROWS), dense)

    assert result.quick_answer is None
    # без истории LLM вызывается один раз — реранк, по исходному вопросу
    assert len(llm.calls) == 1
    assert "Вопрос:\nСколько раз можно пересдать экзамен?" in llm.calls[0]["messages"][1]["content"]
    assert result.rerank_tokens == 400
    # выбран документ A (попадание + соседи ±1), документ B слабее порога 0.35
    assert [c.idx for c in result.selected] == [1, 0, 2]
    assert result.selected[0].rerank_score == 0.9
    assert result.links == [("Расписание", "http://ruz.hse.ru")]
    assert result.search_query == "Сколько раз можно пересдать экзамен?"
    assert [kind for _, kind in embed.calls] == ["query", "query"]
    assert embed.calls[1][0] == ["Сколько раз можно пересдать экзамен?"]
    system, user = result.messages
    assert "Пересдача экзамена допускается не более двух раз." in user["content"]
    assert "- Расписание: http://ruz.hse.ru" in user["content"]
    assert "Отвечай полностью на русском языке." in system["content"]


def test_rewrite_changes_search_but_not_answer_language(monkeypatch):
    Deps(monkeypatch)
    rewritten = "Сколько раз можно пересдавать экзамен по положению о пересдачах?"
    llm = FakeLlm((rewritten, 30), ('{"ranked": [{"id": 1, "score": 0.9}]}', 100))
    history = [HistoryMessage(role="user", content="Расскажи о пересдачах"),
               HistoryMessage(role="assistant", content="Пересдачи регулируются положением.")]
    job = GenerateRequest(question="How many times can I retake it?", history=history)
    dense = [SimpleNamespace(chunk_id=ROWS[1].id, distance=0.1)]

    result, embed = _run(job, llm, _index(ROWS), dense)

    # шаги: переписывание (150 токенов) → реранк по переписанному запросу
    assert [c["max_tokens"] for c in llm.calls] == [150, 750]
    assert f"Вопрос:\n{rewritten}" in llm.calls[1]["messages"][1]["content"]
    assert embed.calls[-1] == ([rewritten], "query")
    assert result.search_query == rewritten
    system, user = result.messages
    # язык — по исходному английскому вопросу, а не по переписанному русскому
    assert "Отвечай полностью на английском языке." in system["content"]
    assert "Вопрос пользователя:\nHow many times can I retake it?\n\nОтвет:" in user["content"]
    assert "Пользователь: Расскажи о пересдачах" in user["content"]


def test_reranker_garbage_keeps_rrf_order(monkeypatch):
    Deps(monkeypatch)
    llm = FakeLlm(("Не могу ответить в JSON", 10))
    dense = [SimpleNamespace(chunk_id=ROWS[1].id, distance=0.1)]

    result, _ = _run(GenerateRequest(question="Пересдача экзамена"), llm, _index(ROWS), dense)

    # все rerank_score = 0 → документ A с лучшим RRF-кандидатом впереди, контекст не пуст
    assert result.selected and result.selected[0].idx == 1
    assert all(c.rerank_score == 0.0 for c in result.selected)


def test_empty_normalized_question_skips_quick_answer(monkeypatch):
    deps = Deps(monkeypatch)
    llm = FakeLlm()

    result, embed = _run(GenerateRequest(question="?!"), llm, _index([]))

    assert deps.quick_calls == []
    assert embed.calls == [(["?!"], "query")]
    assert result.quick_answer is None and result.selected == []


def test_pipeline_does_not_load_graph_modules():
    import json
    import subprocess
    import sys

    graph_modules = (
        "app.graph.store", "app.graph.builder", "app.graph.curation", "app.graph.resolution",
        "app.ontology.extractor", "app.ontology.llm_backend", "app.ontology.prompts", "app.ontology.schema",
        "app.api.graph_admin", "app.retrieval.graph_retriever",
    )
    code = (
        "import sys, json\n"
        "import app.pipeline.service, app.pipeline.index, app.pipeline.quick_answers, app.pipeline.links\n"
        f"print(json.dumps([m for m in {graph_modules!r} if m in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []


@pytest.mark.parametrize("kind", ["query", "passage"])
def test_embed_forwards_kind(monkeypatch, kind):
    import httpx

    from app.graph import embeddings

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.5, 0.5]], "model": "e5-base"})

    real_client = httpx.AsyncClient

    def client(**kwargs):
        kwargs.pop("verify", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", client)

    vectors, model = asyncio.run(embeddings.embed_texts_with_model(["вопрос"], kind=kind))
    assert (vectors, model) == ([[0.5, 0.5]], "e5-base")
    assert seen[-1] == {"texts": ["вопрос"], "normalize": True, "kind": kind}

    # одиночные помощники пробрасывают kind, по умолчанию — passage
    asyncio.run(embeddings.embed_text_with_model("вопрос", kind=kind))
    assert seen[-1]["kind"] == kind
    asyncio.run(embeddings.embed_text("вопрос"))
    assert seen[-1]["kind"] == "passage"
    asyncio.run(embeddings.embed_texts(["вопрос"], kind=kind))
    assert seen[-1]["kind"] == kind
