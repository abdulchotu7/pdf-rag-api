from app.rag.query_engine import QueryEngine
from app.rag.retrieval import Retriever
from app.schemas.models import ChatResponse

from conftest import FakeEmbedder, FakeLLM, FakeReranker, FakeVectorStore


def _make_engine(reranker=None, hybrid=False):
    store = FakeVectorStore()
    retriever = Retriever(
        embedding_provider=FakeEmbedder(),
        vector_store=store,
    )
    hybrid_retriever = (
        HybridDuck(store) if hybrid else None  # type: ignore[arg-type]
    )
    return (
        QueryEngine(
            llm=FakeLLM(),
            retriever=retriever,
            reranker=reranker,
            hybrid_retriever=hybrid_retriever,
        ),
        store,
    )


class HybridDuck:
    """Duck-typed hybrid retriever (real one needs Qdrant for BM25)."""

    def __init__(self, store):
        self._store = store
        self.last_embed_ms = 1.0
        self.last_dense_search_ms = 1.0

    async def retrieve(self, query):
        return await self._store.search([0.0], top_k=10)


async def _seed(store, n=3):
    for i in range(n):
        await store.upsert_chunks(
            [[1.0] * 8],
            [
                {
                    "chunk_id": f"c{i}",
                    "document_id": f"doc{i % 2}",
                    "filename": f"doc{i % 2}.pdf",
                    "page_number": i + 1,
                    "text": f"Chunk {i} about OAuth tokens.",
                }
            ],
            [f"c{i}"],
        )


async def test_query_no_results_returns_friendly_answer():
    engine, _ = _make_engine()
    resp = await engine.query("anything")
    assert isinstance(resp, ChatResponse)
    assert "No relevant documents" in resp.answer
    assert resp.sources == []


async def test_query_with_results_answers_and_cites_sources():
    engine, store = _make_engine(reranker=FakeReranker())
    await _seed(store)
    resp = await engine.query("what about tokens?")
    assert resp.answer == "Fake answer"
    assert len(resp.sources) > 0
    # Sources are deduped by (filename, page).
    keys = {(s.filename, s.page) for s in resp.sources}
    assert len(keys) == len(resp.sources)


async def test_query_dense_path_without_reranker():
    engine, store = _make_engine(reranker=None)
    await _seed(store)
    resp = await engine.query("what about tokens?")
    assert resp.answer == "Fake answer"
    assert len(resp.sources) > 0


async def test_query_hybrid_path():
    engine, store = _make_engine(reranker=FakeReranker(), hybrid=True)
    await _seed(store)
    resp = await engine.query("what about tokens?")
    assert resp.answer == "Fake answer"
    assert len(resp.sources) > 0


async def test_stream_emits_done_on_no_results():
    engine, _ = _make_engine()
    events = [e async for e in engine.query_stream("nothing here")]
    assert any('"type": "done"' in e for e in events)
    assert any('"type": "error"' in e for e in events)


async def test_stream_emits_status_before_rerank():
    engine, store = _make_engine(reranker=FakeReranker())
    await _seed(store)
    events = [e async for e in engine.query_stream("tokens")]
    joined = "".join(events)
    # "Reranking results..." must appear before any token event.
    rerank_pos = joined.find("Reranking results")
    token_pos = joined.find('"type": "token"')
    assert rerank_pos != -1
    assert token_pos != -1
    assert rerank_pos < token_pos
    assert '"type": "done"' in joined
