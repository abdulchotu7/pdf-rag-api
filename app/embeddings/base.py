from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    async def warmup(self) -> None:
        pass
