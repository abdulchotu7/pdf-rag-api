import asyncio
import hashlib
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import EmptyUploadError, InvalidPDFError
from app.core.logging import get_logger
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.streaming_ingestion import StreamingIngestionPipeline
from app.schemas.tasks import TaskStatus
from app.schemas.models import UploadResponse
from app.services.task_manager import task_manager
from app.vectordb.base import BaseVectorStore

logger = get_logger("ingest_service")


class IngestService:
    def __init__(
        self,
        pipeline: StreamingIngestionPipeline,
        vector_store: BaseVectorStore,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._vector_store = vector_store
        self._hybrid_retriever = hybrid_retriever
        self._background_tasks: set[asyncio.Task] = set()

    def _invalidate_bm25(self) -> None:
        if self._hybrid_retriever is not None:
            self._hybrid_retriever.invalidate_bm25()

    def _spawn(self, coro) -> asyncio.Task:
        """Create a background task and keep a strong reference until done.

        Without a reference, the task can be garbage-collected mid-flight.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def list_documents(self) -> list[dict]:
        return await self._vector_store.list_documents()

    async def delete_document(self, document_id: str) -> None:
        await self._vector_store.delete_document(document_id)
        self._invalidate_bm25()

    async def upload_pdf(
        self,
        filename: str,
        content: bytes,
    ) -> UploadResponse:
        if not filename.lower().endswith(".pdf"):
            raise InvalidPDFError("Only PDF files are supported")

        if not content:
            raise EmptyUploadError("Uploaded file is empty")

        if len(content) > settings.max_upload_size:
            raise InvalidPDFError(
                f"File exceeds maximum size of {settings.max_upload_size // (1024 * 1024)} MB"
            )

        content_hash = hashlib.sha256(content).hexdigest()
        file_path = self._save_file(content, filename)
        try:
            result = await self._pipeline.ingest_pdf(
                file_path=str(file_path),
                filename=filename,
                content_hash=content_hash,
            )
        finally:
            # Always clean up the temp file — success, duplicate, or failure.
            file_path.unlink(missing_ok=True)

        if not result.get("duplicate"):
            self._invalidate_bm25()

        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunks=result["chunks"],
            pages=result["pages"],
        )

    async def upload_pdf_background(
        self,
        file_path: str,
        filename: str,
    ) -> str:
        if not filename.lower().endswith(".pdf"):
            raise InvalidPDFError("Only PDF files are supported")

        path = Path(file_path)
        if not path.exists():
            raise InvalidPDFError(f"File not found: {file_path}")

        if path.stat().st_size > settings.max_upload_size:
            raise InvalidPDFError(
                f"File exceeds maximum size of {settings.max_upload_size // (1024 * 1024)} MB"
            )

        content_hash = await asyncio.to_thread(self._hash_file, path)
        task_id = task_manager.create_task()

        self._spawn(
            self._process_background(
                task_id,
                str(path),
                filename,
                content_hash=content_hash,
            )
        )

        return task_id

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()

    async def _process_background(
        self,
        task_id: str,
        file_path: str,
        filename: str,
        content_hash: str | None = None,
    ) -> None:
        try:
            result = await self._pipeline.ingest_pdf(
                file_path=file_path,
                filename=filename,
                content_hash=content_hash,
                task_id=task_id,
            )
            if not result.get("duplicate"):
                self._invalidate_bm25()
            task_manager.set_status(
                task_id,
                TaskStatus.completed,
                progress=result["pages"],
                total=result["pages"],
                result=result,
            )
        except Exception as exc:
            logger.error("Background ingestion failed", exc_info=exc)
            task_manager.set_status(
                task_id,
                TaskStatus.failed,
                error=str(exc),
            )
        finally:
            Path(file_path).unlink(missing_ok=True)

    def _save_file(self, content: bytes, filename: str) -> Path:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        unique_name = f"{uuid.uuid4()}_{filename}"
        file_path = upload_dir / unique_name
        file_path.write_bytes(content)
        return file_path
