from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.dependencies import lifespan
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.tasks import router as tasks_router
from app.api.upload import router as upload_router
from app.core.config import settings
from app.core.exceptions import RAGException
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(tasks_router)
app.include_router(documents_router)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))


@app.exception_handler(RAGException)
async def rag_exception_handler(
    request: Request,
    exc: RAGException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "code": type(exc).__name__,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "code": "InternalError",
        },
    )
