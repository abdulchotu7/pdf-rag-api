import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from llama_index.core import Settings as LlamaSettings

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.bge import BGEEmbeddingProvider
from app.embeddings.base import BaseEmbedding
from app.llm.base import BaseLLM
from app.llm.deepseek import DeepSeekProvider
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_engine import QueryEngine
from app.rag.retrieval import Retriever
from app.rag.streaming_ingestion import StreamingIngestionPipeline
from app.reranker.base import BaseReranker
from app.reranker.factory import RerankerFactory
from app.services.chat_service import ChatService
from app.services.ingest_service import IngestService
from app.utils.device import resolve_device
from app.vectordb.base import BaseVectorStore
from app.vectordb.qdrant import QdrantVectorStore

logger = get_logger("dependencies")


class AppContainer:
    def __init__(self, device: str) -> None:
        # Disable LlamaIndex's built-in LLM — all LLM calls go through our
        # DeepSeekProvider; all embeddings through BGEEmbeddingProvider.
        # No OPENAI_API_KEY is required or set.
        LlamaSettings.llm = None

        self.device = device
        self.llm: BaseLLM = DeepSeekProvider()
        self.embedding: BaseEmbedding = BGEEmbeddingProvider(
            model_name=settings.embedding_model,
            device=device,
        )
        self.vector_store: BaseVectorStore = QdrantVectorStore()

        self.streaming_pipeline = StreamingIngestionPipeline(
            embedding_provider=self.embedding,
            vector_store=self.vector_store,
        )
        self.retriever = Retriever(
            embedding_provider=self.embedding,
            vector_store=self.vector_store,
        )
        self.reranker: BaseReranker | None = RerankerFactory.create(
            device=device,
        )
        self.hybrid_retriever: HybridRetriever | None = (
            HybridRetriever(
                embedder=self.embedding,
                vector_store=self.vector_store,
            )
            if settings.hybrid_enabled
            else None
        )
        self.query_engine = QueryEngine(
            llm=self.llm,
            retriever=self.retriever,
            reranker=self.reranker,
            hybrid_retriever=self.hybrid_retriever,
        )

        self.ingest_service = IngestService(
            pipeline=self.streaming_pipeline,
            vector_store=self.vector_store,
            hybrid_retriever=self.hybrid_retriever,
        )
        self.chat_service = ChatService(
            query_engine=self.query_engine,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting application")

    device = resolve_device(settings.embedding_device)
    logger.info(
        f"Embedding Device: {device}",
        extra={"device": device, "config_setting": settings.embedding_device},
    )
    logger.info(
        f"Reranker Device: {device}",
        extra={"device": device},
    )

    container = AppContainer(device=device)

    logger.info("Loading embedding model...")
    t0 = time.monotonic()
    await container.embedding.warmup()
    logger.info(
        f"Embedding model ready in {(time.monotonic() - t0) * 1000:.0f} ms"
    )

    if container.reranker:
        logger.info("Loading reranker...")
        t0 = time.monotonic()
        await container.reranker.warmup()
        logger.info(
            f"Reranker ready in {(time.monotonic() - t0) * 1000:.0f} ms"
        )

    app.state.container = container
    logger.info("Application ready.")
    yield
    logger.info("Shutting down application")


def get_ingest_service(request: Request) -> IngestService:
    return request.app.state.container.ingest_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.container.chat_service
