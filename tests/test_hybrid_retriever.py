from types import SimpleNamespace

from app.rag.hybrid_retriever import HybridRetriever

from conftest import FakeEmbedder, FakeVectorStore


class _FakeScrollClient:
    """Raw QdrantClient stand-in: scroll() returns canned nodes."""

    def __init__(self, points):
        self._points = points

    def scroll(self, collection_name, limit, with_payload=True):
        return (self._points, None)


def _fake_points(n=3):
    return [
        SimpleNamespace(
            id=i,
            payload={
                "chunk_id": f"chunk{i}",
                "document_id": f"doc{i}",
                "filename": f"doc{i}.pdf",
                "page_number": i + 1,
                "text": f"oauth token flow page {i}",
            },
        )
        for i in range(n)
    ]


async def _make_hybrid(monkeypatch, points):
    monkeypatch.setattr(
        "app.rag.hybrid_retriever.QdrantClient",
        lambda host, port: _FakeScrollClient(points),
    )
    return HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )


async def test_bm25_builds_on_first_retrieve(monkeypatch):
    hybrid = await _make_hybrid(monkeypatch, _fake_points())
    assert hybrid._bm25_retriever is None

    results = await hybrid.retrieve("oauth token flow")
    assert results
    assert hybrid._bm25_built is True
    assert hybrid._bm25_retriever is not None


async def test_invalidate_bm25_forces_rebuild(monkeypatch):
    hybrid = await _make_hybrid(monkeypatch, _fake_points())
    await hybrid.retrieve("oauth token flow")
    assert hybrid._bm25_built is True

    hybrid.invalidate_bm25()
    assert hybrid._bm25_built is False
    assert hybrid._bm25_retriever is None

    # Next retrieve rebuilds from scratch.
    results = await hybrid.retrieve("oauth token flow")
    assert results
    assert hybrid._bm25_built is True


async def test_invalidate_under_concurrent_build_is_safe(monkeypatch):
    hybrid = await _make_hybrid(monkeypatch, _fake_points())
    await hybrid.retrieve("oauth token flow")

    # Invalidate while another retrieval may be in flight — must not deadlock.
    hybrid.invalidate_bm25()
    results = await hybrid.retrieve("oauth token flow")
    assert results
