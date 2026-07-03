import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile

from app.api.dependencies import get_ingest_service
from app.core.config import settings
from app.core.exceptions import EmptyUploadError
from app.schemas.models import UploadResponse
from app.services.ingest_service import IngestService
from app.services.task_manager import task_manager

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=list[UploadResponse])
async def upload_pdf(
    files: list[UploadFile],
    service: IngestService = Depends(get_ingest_service),
) -> list[UploadResponse]:
    if not files:
        raise EmptyUploadError("No files provided")

    results = []
    for file in files:
        if not file.filename:
            raise EmptyUploadError("No filename provided")

        content = await file.read()
        result = await service.upload_pdf(
            filename=file.filename,
            content=content,
        )
        results.append(result)

    return results


@router.post("/upload/async", response_model=dict)
async def upload_pdf_async(
    files: list[UploadFile],
    service: IngestService = Depends(get_ingest_service),
) -> dict:
    if not files:
        raise EmptyUploadError("No files provided")

    task_ids = []
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if not file.filename:
            raise EmptyUploadError("No filename provided")

        temp_name = f"upload_{uuid.uuid4()}_{file.filename}"
        temp_path = upload_dir / temp_name

        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)

        task_id = task_manager.create_task()
        task_ids.append(
            {
                "filename": file.filename,
                "task_id": task_id,
            }
        )

        import asyncio
        asyncio.create_task(
            service._process_background(
                task_id,
                str(temp_path),
                file.filename,
            )
        )

    return {"tasks": task_ids}
