import asyncio
import threading
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings
from app.core.exceptions import EmbeddingModelError
from app.core.logging import get_logger
from app.reranker.base import BaseReranker
from app.schemas.models import SearchResult

logger = get_logger("bge_reranker")


class BGEReranker(BaseReranker):
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    @lru_cache(maxsize=1)
    def _load(self) -> tuple:
        logger.info(
            "Loading reranker model",
            extra={"model": self._model_name, "device": self._device},
        )

        torch_dtype = self._pick_dtype()

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype=torch_dtype,
        )
        model.config.torch_dtype = torch_dtype
        model.config.use_cache = False
        model.eval()
        model.to(self._device)
        return tokenizer, model

    def _pick_dtype(self) -> torch.dtype:
        if self._device in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    async def warmup(self) -> None:
        tokenizer, model = await asyncio.to_thread(self._load)
        inputs = tokenizer(
            [["warmup", "warmup"]],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=settings.rerank_max_length,
        ).to(self._device)
        with torch.inference_mode():
            model(**inputs)

    async def rerank(
        self,
        query: str,
        nodes: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        if not nodes:
            return []

        if top_n <= 0:
            return []

        try:
            tokenizer, model = await asyncio.to_thread(self._load)

            pairs = [[query, node.text] for node in nodes]

            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=settings.rerank_max_length,
            ).to(self._device)

            with self._lock:
                with torch.inference_mode():
                    scores = model(**inputs).logits.squeeze(-1).tolist()

            if isinstance(scores, float):
                scores = [scores]

            scored = list(zip(nodes, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = scored[:top_n]

            return [
                SearchResult(
                    chunk_id=node.chunk_id,
                    document_id=node.document_id,
                    filename=node.filename,
                    page_number=node.page_number,
                    text=node.text,
                    score=round(score, 6),
                )
                for node, score in reranked
            ]
        except Exception as exc:
            logger.error("Reranking failed", exc_info=exc)
            raise EmbeddingModelError(
                f"Reranker inference failed: {exc}"
            ) from exc
