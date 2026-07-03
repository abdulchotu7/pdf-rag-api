import uuid
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader

from app.core.config import settings
from app.core.exceptions import InvalidPDFError
from app.core.logging import get_logger
from app.embeddings.base import BaseEmbedding
from app.schemas.tasks import TaskStatus
from app.services.task_manager import task_manager
from app.vectordb.base import BaseVectorStore

logger = get_logger("streaming_ingestion")

EMBED_BATCH_SIZE = 16


class StreamingIngestionPipeline:
    def __init__(
        self,
        embedding_provider: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedder = embedding_provider
        self._vector_store = vector_store

    async def ingest_pdf(
        self,
        file_path: str,
        filename: str,
        content_hash: str | None = None,
        task_id: str | None = None,
    ) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise InvalidPDFError(f"File not found: {file_path}")

        if content_hash:
            existing = await self._vector_store.find_by_hash(content_hash)
            if existing:
                logger.info(
                    "Duplicate PDF detected, skipping",
                    extra={
                        "doc_filename": filename,
                        "existing_document_id": existing,
                    },
                )
                return {
                    "document_id": existing,
                    "filename": filename,
                    "chunks": 0,
                    "pages": 0,
                    "duplicate": True,
                }

        document_id = str(uuid.uuid4())
        logger.info(
            "Starting streaming ingestion",
            extra={
                "doc_filename": filename,
                "document_id": document_id,
            },
        )

        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        await self._vector_store.ensure_collection()

        pdf_reader = PdfReader(str(path))
        total_pages = len(pdf_reader.pages)

        if task_id:
            task_manager.set_status(
                task_id,
                TaskStatus.processing,
                progress=0,
                total=total_pages,
            )

        all_chunks = []
        total_chunks = 0

        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""

            if not page_text.strip():
                if task_id:
                    task_manager.set_status(
                        task_id,
                        TaskStatus.processing,
                        progress=page_num + 1,
                    )
                continue

            page_doc = Document(
                text=page_text,
                metadata={
                    "page_label": str(page_num + 1),
                    "file_name": filename,
                },
            )

            nodes = splitter.get_nodes_from_documents([page_doc])

            for node in nodes:
                chunk_uuid = str(uuid.uuid4())
                payload = {
                    "chunk_id": chunk_uuid,
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_num + 1,
                    "text": node.get_content(),
                }
                if content_hash:
                    payload["content_hash"] = content_hash
                all_chunks.append(payload)

            if len(all_chunks) >= EMBED_BATCH_SIZE:
                total_chunks += await self._flush_batch(all_chunks, document_id)
                all_chunks = []

            if task_id:
                task_manager.set_status(
                    task_id,
                    TaskStatus.processing,
                    progress=page_num + 1,
                )

        if all_chunks:
            total_chunks += await self._flush_batch(all_chunks, document_id)

        result = {
            "document_id": document_id,
            "filename": filename,
            "chunks": total_chunks,
            "pages": total_pages,
        }

        logger.info(
            "Streaming ingestion complete",
            extra={
                "document_id": document_id,
                "doc_filename": filename,
                "chunks": total_chunks,
                "pages": total_pages,
            },
        )

        return result

    async def _flush_batch(
        self,
        chunks: list[dict],
        document_id: str,
    ) -> int:
        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]
        payloads = [
            {
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "filename": c["filename"],
                "page_number": c["page_number"],
                "text": c["text"],
                **({"content_hash": c["content_hash"]} if "content_hash" in c else {}),
            }
            for c in chunks
        ]

        vectors = await self._embedder.embed_texts(texts)
        await self._vector_store.upsert_chunks(vectors, payloads, chunk_ids)

        logger.debug(
            "Flushed embedding batch",
            extra={"count": len(chunks)},
        )

        return len(chunks)
