from fastapi import APIRouter, HTTPException

from app.schemas.tasks import TaskInfo
from app.services.task_manager import task_manager

router = APIRouter(tags=["tasks"])


@router.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str) -> TaskInfo:
    info = task_manager.get_task(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="Task not found")
    return info
