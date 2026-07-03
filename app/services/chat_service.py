from typing import AsyncIterator

from app.core.logging import get_logger
from app.rag.query_engine import QueryEngine
from app.schemas.models import ChatRequest, ChatResponse

logger = get_logger("chat_service")


class ChatService:
    def __init__(self, query_engine: QueryEngine) -> None:
        self._query_engine = query_engine

    async def answer(self, request: ChatRequest) -> ChatResponse:
        logger.info(
            "Chat request received",
            extra={"question": request.question},
        )
        return await self._query_engine.query(request.question)

    async def answer_stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[str]:
        logger.info(
            "Streaming chat request received",
            extra={"question": request.question},
        )
        async for event in self._query_engine.query_stream(
            request.question,
        ):
            yield event
