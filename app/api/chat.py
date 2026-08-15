from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.schemas.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.answer(request)


@router.post("/chat/async", response_model=dict)
async def chat_async(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    task_id = await service.answer_async(request)
    return {"task_id": task_id}


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    return StreamingResponse(
        service.answer_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
