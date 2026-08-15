import json
import time
from typing import AsyncIterator

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import BaseLLM
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.retrieval import Retriever
from app.reranker.base import BaseReranker
from app.schemas.models import ChatResponse, SourceNode

logger = get_logger("query_engine")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use only the information from the context to answer. "
    "If the context does not contain enough information, say so. "
    "Cite sources with numbered markers like [1], [2], etc. "
    "Each marker corresponds to a source listed below. "
    "After your answer, include a 'Sources' section listing each source: "
    "[1] filename.pdf Page X, [2] filename.pdf Page Y, etc."
)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _status(msg: str) -> str:
    return f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"


class QueryEngine:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        reranker: BaseReranker | None = None,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._reranker = reranker
        self._hybrid_retriever = hybrid_retriever

    async def _retrieve_and_rerank(
        self,
        question: str,
    ) -> tuple[list[SourceNode], str, dict]:
        t_start = time.perf_counter()
        embed_ms = 0.0
        dense_search_ms = 0.0
        retrieval_ms = 0.0
        rerank_ms = 0.0
        retrieved_count = 0

        if self._hybrid_retriever:
            results = await self._hybrid_retriever.retrieve(question)
            retrieval_ms = (time.perf_counter() - t_start) * 1000
            embed_ms = self._hybrid_retriever.last_embed_ms
            dense_search_ms = self._hybrid_retriever.last_dense_search_ms
            retrieved_count = len(results)
        else:
            top_k = (
                settings.retrieval_top_k
                if self._reranker
                else settings.top_k
            )
            results = await self._retriever.retrieve(question, top_k=top_k)
            retrieval_ms = (time.perf_counter() - t_start) * 1000
            embed_ms = self._retriever.last_embed_ms
            dense_search_ms = self._retriever.last_search_ms
            retrieved_count = len(results)

        if not results:
            return [], "", {}

        if self._reranker:
            results = await self._reranker.rerank(
                query=question,
                nodes=results,
                top_n=settings.final_top_k,
            )
            rerank_ms = (time.perf_counter() - t_start) * 1000 - retrieval_ms

        final_count = len(results)

        context_parts = []
        seen = set()
        sources = []

        for idx, r in enumerate(results, 1):
            context_parts.append(
                f"[{idx}] Source: {r.filename}, Page {r.page_number}\n{r.text}"
            )
            key = (r.filename, r.page_number)
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceNode(
                        page=r.page_number,
                        filename=r.filename,
                        score=r.score,
                    )
                )

        context = "\n\n".join(context_parts)

        log_extra = {
            "embed_ms": round(embed_ms, 2),
            "dense_search_ms": round(dense_search_ms, 2),
            "retrieval_ms": round(retrieval_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
            "retrieved_count": retrieved_count,
            "final_count": final_count,
        }
        if self._hybrid_retriever:
            bm25_ms = max(0.0, retrieval_ms - embed_ms - dense_search_ms)
            log_extra["bm25_ms"] = round(bm25_ms, 2)
            log_extra["mode"] = "hybrid"
        else:
            log_extra["bm25_ms"] = 0.0
            log_extra["mode"] = "dense"

        return sources, context, log_extra

    async def query(self, question: str) -> ChatResponse:
        logger.info("Processing query", extra={"question": question})

        t_start = time.perf_counter()
        sources, context, log_extra = await self._retrieve_and_rerank(question)

        if not sources:
            return ChatResponse(
                answer="No relevant documents found to answer your question.",
                sources=[],
            )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the context above."
        )

        t0 = time.perf_counter()
        answer = await self._llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        log_extra["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        t_end = time.perf_counter()
        log_extra["total_ms"] = round((t_end - t_start) * 1000, 1)

        msg = (
            f"mode={log_extra['mode']} "
            f"embed={log_extra['embed_ms']}ms "
            f"dense_search={log_extra['dense_search_ms']}ms "
            f"bm25={log_extra['bm25_ms']}ms "
            f"rerank={log_extra['rerank_ms']}ms "
            f"llm={log_extra['llm_ms']}ms "
            f"total={log_extra['total_ms']}ms "
            f"retrieved={log_extra['retrieved_count']} "
            f"final={log_extra['final_count']}"
        )
        logger.info(msg, extra=log_extra)

        return ChatResponse(answer=answer, sources=sources)

    async def query_stream(
        self,
        question: str,
    ) -> AsyncIterator[str]:
        logger.info("Processing streaming query", extra={"question": question})

        t_start = time.perf_counter()

        yield _status("Embedding query...")

        if self._hybrid_retriever:
            yield _status("Hybrid search (dense + BM25)...")
        else:
            yield _status("Searching documents...")

        sources, context, log_extra = await self._retrieve_and_rerank(question)

        if not sources:
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        if self._reranker:
            yield _status("Reranking results...")

        yield _status("Generating answer...")
        yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources]})}\n\n"

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the context above."
        )

        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        tokens_in = _estimate_tokens(full_prompt)

        t_llm = time.perf_counter()
        tokens_out = 0
        async for token in self._llm.chat_stream(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ):
            tokens_out += _estimate_tokens(token)
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        log_extra["llm_ms"] = round((time.perf_counter() - t_llm) * 1000, 1)
        log_extra["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        msg = (
            f"mode={log_extra['mode']} "
            f"embed={log_extra['embed_ms']}ms "
            f"dense_search={log_extra['dense_search_ms']}ms "
            f"bm25={log_extra['bm25_ms']}ms "
            f"rerank={log_extra['rerank_ms']}ms "
            f"llm={log_extra['llm_ms']}ms "
            f"total={log_extra['total_ms']}ms "
            f"retrieved={log_extra['retrieved_count']} "
            f"final={log_extra['final_count']}"
        )
        logger.info(msg, extra=log_extra)

        metrics = {
            "type": "metrics",
            "latency_ms": round(log_extra["total_ms"]),
            "retrieval_ms": round(log_extra["retrieval_ms"]),
            "rerank_ms": round(log_extra["rerank_ms"]),
            "llm_ms": round(log_extra["llm_ms"]),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        yield f"data: {json.dumps(metrics)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"
