import time

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.base import BaseEmbedding
from app.schemas.models import SearchResult
from app.vectordb.base import BaseVectorStore

logger = get_logger("retrieval")


class Retriever:
    def __init__(
        self,
        embedding_provider: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedder = embedding_provider
        self._vector_store = vector_store
        self.last_embed_ms: float = 0.0
        self.last_search_ms: float = 0.0

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        logger.info("Retrieving context", extra={"query": query})

        if top_k is None:
            top_k = settings.top_k

        t0 = time.perf_counter()
        query_vector = await self._embedder.embed_text(query)
        self.last_embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        results = await self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )
        self.last_search_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Retrieved chunks",
            extra={"count": len(results)},
        )

        return results
