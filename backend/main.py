from __future__ import annotations

import json
import platform
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schema import (
    AgentModelMap,
    ChatRequest,
    HistoryResponse,
    ModelConfig,
    SessionCreate,
    SessionInfo,
    SessionUpdate,
)
from backend.agents.graph import AgentGraph, new_session_id
from backend.memory import MemoryStore
from llm import LlmClient, LlmError, ModelStore

# app 是 FastAPI 应用实例，所有 HTTP 路由都挂在它上面。
app = FastAPI(title="多智能体编程系统", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    # 当前是本地开发工具，允许任意来源访问，方便 Vite 前端连接动态后端端口。
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model_store 是模型配置仓库，读写 memory/config/models.json。
model_store = ModelStore()
# memory_store 是记忆仓库，读写 memory/data 下的会话、项目和全局记忆。
memory_store = MemoryStore()


@app.get("/api/health")
def health() -> dict[str, Any]:
    """健康检查接口，前端或用户可用它确认后端是否启动。"""

    return {"ok": True, "msg": "后端运行中"}


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    """返回模型列表和每个智能体的模型映射。"""

    return {"models": [model.model_dump() for model in model_store.all()], "agent_models": model_store.model_map().model_dump()}


@app.post("/api/models")
def save_model(cfg: ModelConfig) -> ModelConfig:
    """新增或更新模型配置。"""

    return model_store.upsert(cfg)


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str) -> dict[str, Any]:
    """删除一个模型配置，并自动修复智能体模型映射。"""

    model_store.delete(model_id)
    return {"ok": True}


@app.post("/api/models/{model_id}/test")
def test_model(model_id: str) -> dict[str, Any]:
    """测试指定模型是否能正常返回内容。"""

    try:
        # cfg 是待测试的模型配置。
        cfg = model_store.get(model_id)
        return LlmClient(cfg).test()
    except (KeyError, LlmError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/model-map")
def save_model_map(model_map: AgentModelMap) -> AgentModelMap:
    """保存每个智能体对应的模型 id。"""

    return model_store.set_model_map(model_map)


@app.post("/api/sessions")
def create_session(req: SessionCreate) -> SessionInfo:
    """创建新会话，并写入一条 session/start 记忆。"""

    # workdir 是用户选择的真实项目目录，统一解析成绝对路径。
    workdir = str(Path(req.workdir).expanduser().resolve())
    # 如果目录不存在就创建，便于从空项目开始。
    Path(workdir).mkdir(parents=True, exist_ok=True)
    # session_id 是短 uuid，用于 jsonl 文件名和前端状态。
    session_id = new_session_id()
    # title 是会话展示名；没有传标题时默认用 session_id。
    title = " ".join((req.title or "").strip().split())[:60] or session_id
    # 写入会话开始记录，后续历史会话列表从这里读取初始标题。
    memory_store.append(workdir, session_id, "manager", "session", "start", f"新会话：{title}", {"title": title, "custom": bool(req.title)})
    return SessionInfo(id=session_id, workdir=workdir, title=title, interrupted=False)


@app.post("/api/pick-dir")
def pick_dir() -> dict[str, str]:
    """拉起本机系统目录选择器，返回可供后端访问的绝对路径。"""
    # selected 是系统原生目录选择器返回的路径。
    selected = _pick_dir_by_system_dialog()
    # 系统选择器不可用时，尝试 tkinter 兜底。
    if not selected:
        selected = _pick_dir_by_tkinter()
    if not selected:
        raise HTTPException(status_code=400, detail="未选择项目目录")
    return {"workdir": str(Path(selected).expanduser().resolve())}


def _pick_dir_by_system_dialog() -> str:
    """按操作系统选择原生目录选择器命令。"""

    # system 是当前操作系统名称，用于区分 macOS 和 Windows。
    system = platform.system()
    if system == "Darwin":
        # macOS 使用 osascript 拉起 Finder 目录选择器。
        cmd = [
            "osascript",
            "-e",
            'POSIX path of (choose folder with prompt "选择项目目录")',
        ]
    elif system == "Windows":
        # Windows 使用 PowerShell 加载 WinForms FolderBrowserDialog。
        cmd = [
            "powershell",
            "-NoProfile",
            "-STA",
            "-Command",
            (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog;"
                "$dialog.Description = '选择项目目录';"
                "if ($dialog.ShowDialog() -eq 'OK') { $dialog.SelectedPath }"
            ),
        ]
    else:
        return ""
    try:
        # result 保存系统目录选择器命令的输出；stdout 是用户选择的目录。
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _pick_dir_by_tkinter() -> str:
    """使用 tkinter 目录选择器作为跨平台兜底方案。"""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"当前环境无法打开系统目录选择器：{exc}") from exc

    # root 是 tkinter 临时根窗口，隐藏后只显示目录选择弹窗。
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        # selected 是用户选择的目录路径，取消时为空字符串。
        selected = filedialog.askdirectory(title="选择项目目录")
    finally:
        root.destroy()
    return selected


@app.get("/api/history")
def list_history() -> HistoryResponse:
    """返回所有项目和会话历史，供前端历史弹窗展示。"""

    return HistoryResponse(projects=memory_store.list_history())


@app.delete("/api/projects")
def delete_project(workdir: str) -> dict[str, Any]:
    """删除某个项目对应的 agent memory，不删除真实项目目录。"""

    # resolved 是规范化后的真实项目路径，用于定位项目 memory。
    resolved = str(Path(workdir).expanduser().resolve())
    # deleted 表示 memory 目录是否实际存在并被删除。
    deleted = memory_store.delete_project(resolved)
    return {"ok": True, "deleted": deleted}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, workdir: str) -> SessionInfo:
    """读取当前会话基本信息。"""

    # resolved 是规范化后的项目路径。
    resolved = str(Path(workdir).expanduser().resolve())
    return SessionInfo(
        id=session_id,
        workdir=resolved,
        title=memory_store.session_title(resolved, session_id) or session_id,
        interrupted=memory_store.interrupted(resolved, session_id),
    )


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, req: SessionUpdate) -> SessionInfo:
    """重命名会话，并返回更新后的会话信息。"""

    # resolved 是规范化后的项目路径。
    resolved = str(Path(req.workdir).expanduser().resolve())
    try:
        memory_store.rename_session(resolved, session_id, req.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SessionInfo(
        id=session_id,
        workdir=resolved,
        title=memory_store.session_title(resolved, session_id) or session_id,
        interrupted=memory_store.interrupted(resolved, session_id),
    )


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, workdir: str) -> dict[str, Any]:
    """删除指定会话 memory 文件。"""

    # resolved 是规范化后的项目路径。
    resolved = str(Path(workdir).expanduser().resolve())
    # deleted 表示会话 jsonl 是否实际存在并被删除。
    deleted = memory_store.delete_session(resolved, session_id)
    return {"ok": True, "deleted": deleted}


@app.get("/api/memory")
def read_memory(workdir: str, session_id: str | None = None) -> dict[str, Any]:
    """调试接口：读取全局、项目和可选会话记忆。"""

    # resolved 是规范化后的项目路径。
    resolved = str(Path(workdir).expanduser().resolve())
    return {
        "global": memory_store.global_memory(),
        "project": memory_store.project_memory(resolved),
        "session": memory_store.read_session(resolved, session_id, limit=200) if session_id else [],
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """执行一次 agent 任务，并以 NDJSON 流式返回事件和最终结果。"""

    # event_queue 是后台执行线程和 StreamingResponse 生成器之间的通信队列。
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
    # workdir 通过 session_id 反查，避免前端伪造任意项目路径执行任务。
    workdir = _session_workdir(req.session_id)
    # 先记录用户输入，这样即使任务中途异常，也能在历史会话中看到用户发过什么。
    memory_store.append(workdir, req.session_id, "user", "input", "message", req.text)

    def emit(event: Any) -> None:
        """AgentGraph 调用的事件回调，把事件放入流式队列。"""

        event_queue.put({"type": "event", "data": event.model_dump()})

    def worker() -> None:
        """后台线程函数，负责真正运行 AgentGraph。"""

        try:
            # graph 是本次任务的 LangGraph 编排实例，emit 用于实时输出事件。
            graph = AgentGraph(model_store, memory_store, emit=emit)
            # result 是任务最终结果，会以 type=result 写入队列。
            result = graph.run(
                session_id=req.session_id,
                workdir=workdir,
                text=req.text,
                plan_mode=req.plan_mode,
                execute_plan=req.execute_plan,
                model_id=req.model_id,
            )
            event_queue.put({"type": "result", "data": result.model_dump()})
        except Exception as exc:
            # 捕获异常并转成流式 error，避免前端一直等待。
            event_queue.put({"type": "error", "data": {"msg": str(exc)}})
        finally:
            # None 是流结束哨兵，stream() 读到后停止 yield。
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> Any:
        """StreamingResponse 使用的生成器，把队列数据逐行编码为 NDJSON。"""

        while True:
            # item 是后台线程放入队列的一条事件、结果、错误或结束哨兵。
            item = event_queue.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _session_workdir(session_id: str) -> str:
    """从 memory/data 里反查会话工作目录。

    会话文件分散在项目记忆目录中，因此这里扫描 sessions 文件。数量很大时可换成索引文件。
    """
    # workdir 是 MemoryStore 根据 session_id 找到的真实项目目录。
    workdir = memory_store.find_session_workdir(session_id)
    if workdir:
        return workdir
    raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
