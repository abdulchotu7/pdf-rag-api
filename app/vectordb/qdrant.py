import asyncio

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.core.exceptions import VectorDBUnavailableError
from app.core.logging import get_logger
from app.schemas.models import SearchResult
from app.vectordb.base import BaseVectorStore

logger = get_logger("qdrant")


class QdrantVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
        return self._client

    async def collection_exists(self) -> bool:
        try:
            client = self._get_client()
            collections = await asyncio.to_thread(client.get_collections)
            return any(
                c.name == settings.qdrant_collection
                for c in collections.collections
            )
        except UnexpectedResponse as exc:
            logger.error("Qdrant connection failed", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Cannot connect to Qdrant: {exc}"
            ) from exc

    async def create_collection(self) -> None:
        try:
            client = self._get_client()
            await asyncio.to_thread(
                client.create_collection,
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            # Payload indexes so filters (dedup by hash, delete/list by
            # document_id) don't fall back to full collection scans.
            for field in ("document_id", "content_hash"):
                try:
                    await asyncio.to_thread(
                        client.create_payload_index,
                        collection_name=settings.qdrant_collection,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except UnexpectedResponse:
                    # Index already exists (e.g. collection pre-created manually).
                    logger.debug(
                        "Payload index already exists",
                        extra={"field": field},
                    )
            logger.info(
                "Created Qdrant collection",
                extra={"collection": settings.qdrant_collection},
            )
        except UnexpectedResponse as exc:
            logger.error("Failed to create Qdrant collection", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Cannot create collection: {exc}"
            ) from exc

    async def ensure_collection(self) -> None:
        if not await self.collection_exists():
            await self.create_collection()

    async def upsert_chunks(
        self,
        vectors: list[list[float]],
        payloads: list[dict],
        ids: list[str],
    ) -> None:
        try:
            client = self._get_client()
            points = [
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload=payloads[i],
                )
                for i in range(len(ids))
            ]
            await asyncio.to_thread(
                client.upsert,
                collection_name=settings.qdrant_collection,
                points=points,
            )
            logger.info(
                "Upserted chunks to Qdrant",
                extra={"count": len(points)},
            )
        except UnexpectedResponse as exc:
            logger.error("Qdrant upsert failed", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Failed to upsert vectors: {exc}"
            ) from exc

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        try:
            client = self._get_client()
            response = await asyncio.to_thread(
                client.query_points,
                collection_name=settings.qdrant_collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            return [
                SearchResult(
                    chunk_id=str(hit.id),
                    document_id=hit.payload.get("document_id", ""),
                    filename=hit.payload.get("filename", ""),
                    page_number=hit.payload.get("page_number", 0),
                    text=hit.payload.get("text", ""),
                    score=hit.score,
                )
                for hit in response.points
            ]
        except UnexpectedResponse as exc:
            logger.error("Qdrant search failed", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Failed to search vectors: {exc}"
            ) from exc

    async def find_by_hash(self, content_hash: str) -> str | None:
        try:
            client = self._get_client()
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="content_hash",
                        match=MatchValue(value=content_hash),
                    )
                ]
            )
            result, _ = await asyncio.to_thread(
                client.scroll,
                collection_name=settings.qdrant_collection,
                scroll_filter=qdrant_filter,
                limit=1,
                with_payload=True,
            )
            if result:
                return result[0].payload.get("document_id")
            return None
        except UnexpectedResponse as exc:
            # Qdrant being down is not "no duplicate" — fail fast so the
            # caller gets a clear 503 instead of a confusing upsert failure.
            logger.error("Qdrant dedup lookup failed", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Cannot check for duplicates: {exc}"
            ) from exc

    async def list_documents(self) -> list[dict]:
        try:
            client = self._get_client()
            result, _ = await asyncio.to_thread(
                client.scroll,
                collection_name=settings.qdrant_collection,
                limit=100000,
                with_payload=True,
                with_vectors=False,
            )
            seen: dict[str, dict] = {}
            for p in result:
                pid = p.payload.get("document_id", "")
                fname = p.payload.get("filename", "")
                pn = p.payload.get("page_number", 0) or 0
                if pid not in seen:
                    seen[pid] = {
                        "document_id": pid,
                        "filename": fname,
                        "chunks": 0,
                        "pages": 0,
                    }
                seen[pid]["chunks"] += 1
                if pn > seen[pid]["pages"]:
                    seen[pid]["pages"] = pn
            return list(seen.values())
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                # Collection doesn't exist yet — nothing to list.
                return []
            logger.error("Failed to list documents", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Cannot list documents: {exc}"
            ) from exc

    async def delete_document(self, document_id: str) -> None:
        try:
            client = self._get_client()
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )
            # Delete by filter — Qdrant matches all points in one operation.
            # (A scroll-based approach would truncate at the scroll limit.)
            deleted = await asyncio.to_thread(
                client.delete,
                collection_name=settings.qdrant_collection,
                points_selector=qdrant_filter,
            )
            logger.info(
                "Deleted document from Qdrant",
                extra={
                    "document_id": document_id,
                    "delete_status": getattr(deleted, "status", None),
                },
            )
        except UnexpectedResponse as exc:
            logger.error("Qdrant delete failed", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Failed to delete document: {exc}"
            ) from exc

    async def delete_collection(self) -> None:
        try:
            client = self._get_client()
            await asyncio.to_thread(
                client.delete_collection,
                collection_name=settings.qdrant_collection,
            )
            logger.info(
                "Deleted Qdrant collection",
                extra={"collection": settings.qdrant_collection},
            )
        except UnexpectedResponse as exc:
            logger.error("Failed to delete Qdrant collection", exc_info=exc)
            raise VectorDBUnavailableError(
                f"Cannot delete collection: {exc}"
            ) from exc
