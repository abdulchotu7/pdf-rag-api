import asyncio
import uuid
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader

from app.core.config import settings
from app.core.exceptions import InvalidPDFError
from app.core.logging import get_logger
from app.embeddings.base import BaseEmbedding
from app.vectordb.base import BaseVectorStore

logger = get_logger("ingestion")


class IngestionPipeline:
    def __init__(
        self,
        embedding_provider: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedder = embedding_provider
        self._vector_store = vector_store

    async def ingest_pdf(self, file_path: str, filename: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise InvalidPDFError(f"File not found: {file_path}")

        logger.info("Reading PDF", extra={"doc_filename": filename})

        try:
            documents = await asyncio.to_thread(
                SimpleDirectoryReader(
                    input_files=[str(path)],
                ).load_data,
            )
        except Exception as exc:
            logger.error("Failed to read PDF", exc_info=exc)
            raise InvalidPDFError(f"Cannot parse PDF: {exc}") from exc

        if not documents:
            raise InvalidPDFError("PDF file contains no parseable content")

        document_id = str(uuid.uuid4())

        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        nodes = await asyncio.to_thread(
            splitter.get_nodes_from_documents,
            documents,
        )
        logger.info(
            "Split PDF into nodes",
            extra={"node_count": len(nodes), "doc_filename": filename},
        )

        texts = []
        payloads = []
        chunk_ids = []

        for node in nodes:
            page_number = 1
            if node.metadata and "page_label" in node.metadata:
                try:
                    page_number = int(node.metadata["page_label"])
                except (ValueError, TypeError):
                    page_number = 1

            chunk_uuid = str(uuid.uuid4())
            texts.append(node.get_content())
            payloads.append(
                {
                    "chunk_id": chunk_uuid,
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "text": node.get_content(),
                }
            )
            chunk_ids.append(chunk_uuid)

        logger.info(
            "Generating embeddings",
            extra={"count": len(texts), "doc_filename": filename},
        )
        vectors = await self._embedder.embed_texts(texts)

        max_page = max(
            p["page_number"] for p in payloads
        ) if payloads else 0

        await self._vector_store.ensure_collection()

        await self._vector_store.upsert_chunks(vectors, payloads, chunk_ids)

        logger.info(
            "Ingested PDF successfully",
            extra={
                "document_id": document_id,
                "doc_filename": filename,
                "chunks": len(chunk_ids),
                "pages": max_page,
            },
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks": len(chunk_ids),
            "pages": max_page,
        }
