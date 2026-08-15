from datetime import datetime, timedelta, timezone

from app.schemas.tasks import TaskStatus
from app.services.task_manager import TaskManager


async def test_create_get_status_cycle():
    tm = TaskManager()
    task_id = tm.create_task()
    info = tm.get_task(task_id)
    assert info is not None
    assert info.status == TaskStatus.pending

    tm.set_status(task_id, TaskStatus.processing, progress=3, total=10)
    info = tm.get_task(task_id)
    assert info is not None
    assert info.status == TaskStatus.processing
    assert info.progress == 3
    assert info.total == 10

    tm.set_status(task_id, TaskStatus.completed, result={"answer": "x"})
    info = tm.get_task(task_id)
    assert info is not None
    assert info.status == TaskStatus.completed
    assert info.result == {"answer": "x"}


async def test_get_unknown_task_returns_none():
    tm = TaskManager()
    assert tm.get_task("nope") is None


async def test_ttl_prunes_stale_tasks():
    tm = TaskManager(ttl_seconds=3600)
    old_id = tm.create_task()

    # Age the old task beyond the TTL.
    stale_created = datetime.now(timezone.utc) - timedelta(hours=2)
    tm._tasks[old_id].created_at = stale_created

    # Creating a new task triggers pruning.
    tm.create_task()

    assert tm.get_task(old_id) is None
