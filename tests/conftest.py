"""Shared fakes — no torch, no models, no network (unit tests)."""

from dataclasses import dataclass

import pytest
from llama_index.core import Settings as LlamaSettings

from app.embeddings.base import BaseEmbedding
from app.llm.base import BaseLLM
from app.reranker.base import BaseReranker
from app.schemas.models import SearchResult
from app.vectordb.base import BaseVectorStore


@pytest.fixture(autouse=True)
def _disable_llama_llm():
    """Mirror app startup: LlamaIndex's built-in LLM is explicitly disabled.

    Otherwise llama-index's fusion retriever tries to resolve an OpenAI
    default LLM and fails without an OPENAI_API_KEY.
    """
    LlamaSettings.llm = None
    yield


class FakeEmbedder(BaseEmbedding):
    """Deterministic embedding provider for tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    @property
    def dimension(self) -> int:
        return self.dim

    async def embed_text(self, text: str) -> list[float]:
        return [1.0] * self.dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self.dim for _ in texts]


class FakeVectorStore(BaseVectorStore):
    """In-memory stand-in for QdrantVectorStore."""

    def __init__(self) -> None:
        self.chunks: dict[str, tuple[list[float], dict]] = {}

    async def collection_exists(self) -> bool:
        return bool(self.chunks)

    async def create_collection(self) -> None:
        pass

    async def ensure_collection(self) -> None:
        pass

    async def upsert_chunks(
        self,
        vectors: list[list[float]],
        payloads: list[dict],
        ids: list[str],
    ) -> None:
        for v, p, i in zip(vectors, payloads, ids):
            self.chunks[i] = (v, p)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        items = list(self.chunks.items())[:top_k]
        return [
            SearchResult(
                chunk_id=i,
                document_id=p["document_id"],
                filename=p["filename"],
                page_number=p["page_number"],
                text=p["text"],
                score=1.0,
            )
            for i, (_, p) in items
        ]

    async def find_by_hash(self, content_hash: str) -> str | None:
        for _, p in self.chunks.values():
            if p.get("content_hash") == content_hash:
                return p["document_id"]
        return None

    async def list_documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for _, p in self.chunks.values():
            pid = p["document_id"]
            if pid not in seen:
                seen[pid] = {
                    "document_id": pid,
                    "filename": p["filename"],
                    "chunks": 0,
                    "pages": 0,
                }
            seen[pid]["chunks"] += 1
            seen[pid]["pages"] = max(seen[pid]["pages"], p["page_number"])
        return list(seen.values())

    async def delete_document(self, document_id: str) -> None:
        self.chunks = {
            i: (v, p)
            for i, (v, p) in self.chunks.items()
            if p["document_id"] != document_id
        }

    async def delete_collection(self) -> None:
        self.chunks = {}


class FakeLLM(BaseLLM):
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        return "Fake answer"

    async def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        yield "Fake"
        yield " answer"


class FakeReranker(BaseReranker):
    async def rerank(self, query, nodes, top_n):
        return sorted(nodes, key=lambda n: n.score, reverse=True)[:top_n]


@dataclass
class FakePipeline:
    """StreamingIngestionPipeline stand-in that records calls."""

    result: dict

    async def ingest_pdf(self, file_path, filename, content_hash=None, task_id=None):
        return dict(self.result)


class FakeHybridRetriever:
    """Spy for BM25 invalidation calls."""

    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate_bm25(self) -> None:
        self.invalidations += 1
