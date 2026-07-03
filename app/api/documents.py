from fastapi import APIRouter, Depends

from app.api.dependencies import get_ingest_service
from app.schemas.models import DocumentInfo
from app.services.ingest_service import IngestService

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    service: IngestService = Depends(get_ingest_service),
) -> list[DocumentInfo]:
    docs = await service.list_documents()
    return [DocumentInfo(**d) for d in docs]


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    service: IngestService = Depends(get_ingest_service),
) -> None:
    await service.delete_document(document_id)
