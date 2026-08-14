from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field

from backend.clock import utc_stamp


class RunConflictError(RuntimeError):
    """同一会话已有任务运行时拒绝再次启动。"""


class RunNotFoundError(LookupError):
    """目标运行不存在或已经结束。"""


@dataclass(frozen=True)
class GuidanceMessage:
    """用户在 Agent 运行中追加的一条指导。"""

    id: str
    text: str
    created_at: str

    def model_dump(self) -> dict[str, str]:
        """返回可直接写入状态或接口响应的普通字典。"""

        return {"id": self.id, "text": self.text, "created_at": self.created_at}


@dataclass
class ActiveRun:
    """一次正在执行的 Agent 运行及其待处理指导队列。"""

    id: str
    session_id: str
    workdir: str
    event_queue: queue.Queue[dict | None]
    guidance_queue: queue.Queue[GuidanceMessage] = field(default_factory=queue.Queue)

    def drain_guidance(self) -> list[dict[str, str]]:
        """非阻塞取出当前已经到达的全部指导，并保持发送顺序。"""

        messages: list[dict[str, str]] = []
        while True:
            try:
                messages.append(self.guidance_queue.get_nowait().model_dump())
            except queue.Empty:
                return messages


class RunRegistry:
    """线程安全地管理当前进程中的活动运行。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.by_id: dict[str, ActiveRun] = {}
        self.by_session: dict[str, str] = {}

    def start(
        self,
        session_id: str,
        workdir: str,
        event_queue: queue.Queue[dict | None],
    ) -> ActiveRun:
        """注册新运行；同一会话同时只允许一个运行。"""

        with self.lock:
            current_id = self.by_session.get(session_id)
            if current_id and current_id in self.by_id:
                raise RunConflictError(f"会话已有任务正在执行：{current_id}")
            run = ActiveRun(
                id=uuid.uuid4().hex[:16],
                session_id=session_id,
                workdir=workdir,
                event_queue=event_queue,
            )
            self.by_id[run.id] = run
            self.by_session[session_id] = run.id
            return run

    def submit_guidance(self, run_id: str, text: str) -> tuple[ActiveRun, GuidanceMessage]:
        """向指定活动运行追加指导。"""

        with self.lock:
            run = self.by_id.get(run_id)
            if run is None:
                raise RunNotFoundError("任务已经结束或运行编号不存在")
            message = GuidanceMessage(
                id=uuid.uuid4().hex[:12],
                text=text,
                created_at=utc_stamp(),
            )
            run.guidance_queue.put(message)
            return run, message

    def finish(self, run_id: str) -> None:
        """移除已结束运行，使后续指导明确返回过期状态。"""

        with self.lock:
            run = self.by_id.pop(run_id, None)
            if run and self.by_session.get(run.session_id) == run_id:
                self.by_session.pop(run.session_id, None)
