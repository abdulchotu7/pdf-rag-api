from pathlib import Path

import pytest

from app.core.exceptions import InvalidPDFError
from app.services.ingest_service import IngestService

from conftest import FakeHybridRetriever, FakePipeline, FakeVectorStore


def _make_service(result=None):
    spy = FakeHybridRetriever()
    service = IngestService(
        pipeline=FakePipeline(result=result or {}),
        vector_store=FakeVectorStore(),
        hybrid_retriever=spy,
    )
    return service, spy


async def test_upload_pdf_from_file_cleans_up_and_invalidates(tmp_path):
    service, spy = _make_service(
        result={
            "document_id": "doc1",
            "filename": "x.pdf",
            "chunks": 5,
            "pages": 2,
        }
    )
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    resp = await service.upload_pdf_from_file(str(f), "x.pdf")

    assert resp.document_id == "doc1"
    assert resp.chunks == 5
    # Temp file is removed even on success.
    assert not f.exists()
    # Real ingest invalidates the BM25 index.
    assert spy.invalidations == 1


async def test_upload_pdf_from_file_duplicate_skips_invalidation(tmp_path):
    service, spy = _make_service(
        result={
            "document_id": "existing",
            "filename": "x.pdf",
            "chunks": 0,
            "pages": 0,
            "duplicate": True,
        }
    )
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    resp = await service.upload_pdf_from_file(str(f), "x.pdf")

    assert resp.document_id == "existing"
    assert not f.exists()
    assert spy.invalidations == 0


async def test_upload_pdf_from_file_rejects_non_pdf(tmp_path):
    service, _ = _make_service()
    f = tmp_path / "notes.txt"
    f.write_text("hello")

    with pytest.raises(InvalidPDFError):
        await service.upload_pdf_from_file(str(f), "notes.txt")

    # Rejected before processing — but file cleanup still handled by caller.
    assert f.exists()


async def test_upload_pdf_from_file_missing_file():
    service, _ = _make_service()
    with pytest.raises(InvalidPDFError):
        await service.upload_pdf_from_file("/nonexistent/x.pdf", "x.pdf")


async def test_upload_pdf_from_file_cleans_up_on_pipeline_failure(tmp_path):
    class BoomPipeline:
        async def ingest_pdf(self, file_path, filename, content_hash=None, task_id=None):
            raise RuntimeError("qdrant exploded")

    spy = FakeHybridRetriever()
    service = IngestService(
        pipeline=BoomPipeline(),
        vector_store=FakeVectorStore(),
        hybrid_retriever=spy,
    )
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(RuntimeError):
        await service.upload_pdf_from_file(str(f), "x.pdf")

    # Temp file cleaned up even when ingestion blows up.
    assert not f.exists()
    assert spy.invalidations == 0


async def test_delete_document_invalidates_bm25():
    spy = FakeHybridRetriever()
    store = FakeVectorStore()
    service = IngestService(
        pipeline=FakePipeline(result={}),
        vector_store=store,
        hybrid_retriever=spy,
    )
    await service.delete_document("doc1")
    assert spy.invalidations == 1


async def test_delete_document_no_hybrid_no_crash():
    store = FakeVectorStore()
    service = IngestService(
        pipeline=FakePipeline(result={}),
        vector_store=store,
        hybrid_retriever=None,
    )
    await service.delete_document("doc1")  # must not raise
