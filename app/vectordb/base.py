from abc import ABC, abstractmethod

from app.schemas.models import SearchResult


class BaseVectorStore(ABC):
    @abstractmethod
    async def create_collection(self) -> None:
        ...

    @abstractmethod
    async def collection_exists(self) -> bool:
        ...

    @abstractmethod
    async def ensure_collection(self) -> None:
        ...

    @abstractmethod
    async def upsert_chunks(
        self,
        vectors: list[list[float]],
        payloads: list[dict],
        ids: list[str],
    ) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def find_by_hash(self, content_hash: str) -> str | None:
        ...

    @abstractmethod
    async def list_documents(self) -> list[dict]:
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    async def delete_collection(self) -> None:
        ...
