import asyncio
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.core.exceptions import EmbeddingModelError
from app.core.logging import get_logger
from app.embeddings.base import BaseEmbedding

logger = get_logger("bge")


class BGEEmbeddingProvider(BaseEmbedding):
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension() or 1024

    @lru_cache(maxsize=1)
    def _load_model(self) -> SentenceTransformer:
        logger.info(
            "Loading embedding model",
            extra={"model": self._model_name, "device": self._device},
        )
        torch_dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
        return SentenceTransformer(
            self._model_name,
            device=self._device,
            model_kwargs={"torch_dtype": torch_dtype},
        )

    async def warmup(self) -> None:
        model = await asyncio.to_thread(self._load_model)
        await asyncio.to_thread(
            model.encode,
            ["warmup"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            model = self._load_model()
            embeddings = await asyncio.to_thread(
                model.encode,
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Embedding generation failed", exc_info=exc)
            raise EmbeddingModelError(
                f"Failed to generate embeddings: {exc}"
            ) from exc
