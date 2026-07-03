from app.core.config import settings
from app.core.logging import get_logger
from app.reranker.base import BaseReranker
from app.reranker.bge import BGEReranker

logger = get_logger("reranker_factory")


class RerankerFactory:
    @staticmethod
    def create(device: str) -> BaseReranker | None:
        if not settings.rerank_enabled:
            logger.info("Reranker disabled by configuration")
            return None

        logger.info(
            "Creating reranker",
            extra={"model": settings.rerank_model, "device": device},
        )
        return BGEReranker(
            model_name=settings.rerank_model,
            device=device,
        )
