import uuid
from threading import Lock

from app.schemas.tasks import TaskInfo, TaskStatus


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = Lock()

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = TaskInfo(
                task_id=task_id,
                status=TaskStatus.pending,
            )
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def set_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: int | None = None,
        total: int | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            info = self._tasks[task_id]
            info.status = status
            if progress is not None:
                info.progress = progress
            if total is not None:
                info.total = total
            if result is not None:
                info.result = result
            if error is not None:
                info.error = error


task_manager = TaskManager()
