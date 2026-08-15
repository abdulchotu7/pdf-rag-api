"""Integration tests against a real Qdrant — no models, no API keys.

Skipped unless RUN_INTEGRATION=1 (CI sets it and runs a Qdrant service).
"""

import os
import uuid

import pytest

from app.core.config import settings
from app.vectordb.qdrant import QdrantVectorStore

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 and run a Qdrant instance",
)


def _dummy_vector(dim: int) -> list[float]:
    return [1.0] * dim


@pytest.fixture()
async def store():
    s = QdrantVectorStore()
    await s.ensure_collection()
    yield s
    await s.delete_collection()


async def test_upsert_search_dedup_delete_cycle(store):
    dim = settings.embedding_dimension
    chunk_ids = [str(uuid.uuid4()) for _ in range(3)]  # Qdrant requires UUID/int ids

    # Upsert 3 chunks of one document, sharing a content hash.
    payloads = [
        {
            "chunk_id": chunk_ids[i],
            "document_id": "doc1",
            "filename": "doc1.pdf",
            "page_number": i + 1,
            "text": f"OAuth authorization code flow, part {i}.",
            "content_hash": "abc123",
        }
        for i in range(3)
    ]
    await store.upsert_chunks(
        [_dummy_vector(dim) for _ in payloads],
        payloads,
        chunk_ids,
    )

    # Search finds them.
    hits = await store.search(_dummy_vector(dim), top_k=3)
    assert len(hits) == 3
    assert all(h.document_id == "doc1" for h in hits)

    # Dedup lookup hits.
    assert await store.find_by_hash("abc123") == "doc1"
    assert await store.find_by_hash("nope") is None

    # Listing aggregates by document.
    docs = await store.list_documents()
    assert len(docs) == 1
    assert docs[0]["chunks"] == 3
    assert docs[0]["pages"] == 3

    # Delete-by-filter removes every chunk (no scroll truncation).
    await store.delete_document("doc1")
    assert await store.list_documents() == []
    hits = await store.search(_dummy_vector(dim), top_k=3)
    assert hits == []


async def test_list_documents_missing_collection_returns_empty(store):
    await store.delete_collection()
    assert await store.list_documents() == []
