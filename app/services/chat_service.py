import asyncio
from typing import AsyncIterator

from app.core.logging import get_logger
from app.rag.query_engine import QueryEngine
from app.schemas.models import ChatRequest, ChatResponse
from app.schemas.tasks import TaskStatus
from app.services.task_manager import task_manager

logger = get_logger("chat_service")


class ChatService:
    def __init__(self, query_engine: QueryEngine) -> None:
        self._query_engine = query_engine
        self._background_tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> asyncio.Task:
        """Create a background task and keep a strong reference until done.

        Without a reference, the task can be garbage-collected mid-flight.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def answer(self, request: ChatRequest) -> ChatResponse:
        logger.info(
            "Chat request received",
            extra={"question": request.question},
        )
        return await self._query_engine.query(request.question)

    async def answer_async(self, request: ChatRequest) -> str:
        task_id = task_manager.create_task()
        logger.info(
            "Async chat request created",
            extra={"question": request.question, "task_id": task_id},
        )
        self._spawn(self._process_answer(task_id, request.question))
        return task_id

    async def _process_answer(self, task_id: str, question: str) -> None:
        try:
            response = await self._query_engine.query(question)
            task_manager.set_status(
                task_id,
                TaskStatus.completed,
                result=response.model_dump(),
            )
        except Exception as exc:
            logger.error("Async answer failed", exc_info=exc)
            task_manager.set_status(
                task_id,
                TaskStatus.failed,
                error=str(exc),
            )

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
