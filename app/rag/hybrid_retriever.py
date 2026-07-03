import asyncio
import threading
import time

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.base import BaseEmbedding
from app.schemas.models import SearchResult
from app.vectordb.base import BaseVectorStore

logger = get_logger("hybrid_retriever")


class _DenseRetrieverAdapter(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore,
        top_k: int,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self.last_embed_ms: float = 0.0
        self.last_search_ms: float = 0.0

    async def _aretrieve(
        self, query_bundle: QueryBundle
    ) -> list[NodeWithScore]:
        t0 = time.perf_counter()
        query_vector = await self._embedder.embed_text(
            query_bundle.query_str
        )
        self.last_embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        results = await self._vector_store.search(
            query_vector, top_k=self._top_k
        )
        self.last_search_ms = (time.perf_counter() - t0) * 1000

        return [
            NodeWithScore(
                node=TextNode(
                    text=r.text,
                    id_=r.chunk_id,
                    metadata={
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "filename": r.filename,
                        "page_number": r.page_number,
                    },
                ),
                score=r.score,
            )
            for r in results
        ]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        raise NotImplementedError("Use async retrieve")


class HybridRetriever:
    def __init__(
        self,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_retriever: BM25Retriever | None = None
        self._lock = threading.Lock()
        self._bm25_built = False
        self.last_embed_ms: float = 0.0
        self.last_dense_search_ms: float = 0.0
        self.last_retrieval_ms: float = 0.0

    async def _build_bm25(self) -> None:
        if self._bm25_built:
            return

        with self._lock:
            if self._bm25_built:
                return

            nodes = await self._load_nodes_from_qdrant()
            if not nodes:
                logger.warning("No nodes found for BM25 index")
                self._bm25_built = True
                return

            self._bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=settings.bm25_top_k,
            )
            self._bm25_built = True
            logger.info(
                "BM25 index built",
                extra={"node_count": len(nodes)},
            )

    async def _load_nodes_from_qdrant(self) -> list[TextNode]:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        scroll_result, _ = await asyncio.to_thread(
            client.scroll,
            collection_name=settings.qdrant_collection,
            limit=100000,
            with_payload=True,
        )
        nodes = []
        for point in scroll_result:
            payload = point.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            nodes.append(
                TextNode(
                    text=text,
                    id_=str(point.id),
                    metadata={
                        "chunk_id": payload.get("chunk_id", str(point.id)),
                        "document_id": payload.get("document_id", ""),
                        "filename": payload.get("filename", ""),
                        "page_number": payload.get("page_number", 0),
                    },
                )
            )
        return nodes

    async def retrieve(self, query: str) -> list[SearchResult]:
        await self._build_bm25()

        dense = _DenseRetrieverAdapter(
            embedder=self._embedder,
            vector_store=self._vector_store,
            top_k=settings.vector_top_k,
        )

        retrievers: list[BaseRetriever] = [dense]
        if self._bm25_retriever is not None:
            retrievers.append(self._bm25_retriever)

        fusion = QueryFusionRetriever(
            retrievers,
            similarity_top_k=settings.fusion_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
        )

        t0 = time.perf_counter()
        nodes = await fusion.aretrieve(QueryBundle(query_str=query))
        self.last_retrieval_ms = (time.perf_counter() - t0) * 1000
        self.last_embed_ms = dense.last_embed_ms
        self.last_dense_search_ms = dense.last_search_ms

        return [
            SearchResult(
                chunk_id=node.node.node_id,
                document_id=node.node.metadata.get("document_id", ""),
                filename=node.node.metadata.get("filename", ""),
                page_number=node.node.metadata.get("page_number", 0),
                text=node.node.text or "",
                score=node.score or 0.0,
            )
            for node in nodes
        ]
