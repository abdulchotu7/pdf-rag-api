from abc import ABC, abstractmethod

from app.schemas.models import SearchResult


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        nodes: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        ...

    async def warmup(self) -> None:
        pass
