from enum import Enum

from pydantic import BaseModel


class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = 0
    total: int = 0
    result: dict | None = None
    error: str | None = None
