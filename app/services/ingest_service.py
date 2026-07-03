import asyncio
import hashlib
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import EmptyUploadError, InvalidPDFError
from app.core.logging import get_logger
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
    ) -> None:
        self._pipeline = pipeline
        self._vector_store = vector_store

    async def list_documents(self) -> list[dict]:
        return await self._vector_store.list_documents()

    async def delete_document(self, document_id: str) -> None:
        await self._vector_store.delete_document(document_id)

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
        result = await self._pipeline.ingest_pdf(
            file_path=str(file_path),
            filename=filename,
            content_hash=content_hash,
        )

        if result.get("duplicate"):
            file_path.unlink(missing_ok=True)

        return UploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunks=result["chunks"],
            pages=result["pages"],
        )

    async def upload_pdf_background(
        self,
        filename: str,
        content: bytes,
    ) -> str:
        if not filename.lower().endswith(".pdf"):
            raise InvalidPDFError("Only PDF files are supported")

        if not content:
            raise EmptyUploadError("Uploaded file is empty")

        if len(content) > settings.max_upload_size:
            raise InvalidPDFError(
                f"File exceeds maximum size of {settings.max_upload_size // (1024 * 1024)} MB"
            )

        content_hash = hashlib.sha256(content).hexdigest()
        task_id = task_manager.create_task()
        file_path = self._save_file(content, filename)

        asyncio.create_task(
            self._process_background(
                task_id,
                str(file_path),
                filename,
                content_hash=content_hash,
            )
        )

        return task_id

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
