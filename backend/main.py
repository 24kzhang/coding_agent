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

app = FastAPI(title="多智能体编程系统", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_store = ModelStore()
memory_store = MemoryStore()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "msg": "后端运行中"}


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"models": [model.model_dump() for model in model_store.all()], "agent_models": model_store.model_map().model_dump()}


@app.post("/api/models")
def save_model(cfg: ModelConfig) -> ModelConfig:
    return model_store.upsert(cfg)


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str) -> dict[str, Any]:
    model_store.delete(model_id)
    return {"ok": True}


@app.post("/api/models/{model_id}/test")
def test_model(model_id: str) -> dict[str, Any]:
    try:
        cfg = model_store.get(model_id)
        return LlmClient(cfg).test()
    except (KeyError, LlmError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/model-map")
def save_model_map(model_map: AgentModelMap) -> AgentModelMap:
    return model_store.set_model_map(model_map)


@app.post("/api/sessions")
def create_session(req: SessionCreate) -> SessionInfo:
    workdir = str(Path(req.workdir).expanduser().resolve())
    Path(workdir).mkdir(parents=True, exist_ok=True)
    session_id = new_session_id()
    title = " ".join((req.title or "").strip().split())[:60] or session_id
    memory_store.append(workdir, session_id, "manager", "session", "start", f"新会话：{title}", {"title": title, "custom": bool(req.title)})
    return SessionInfo(id=session_id, workdir=workdir, title=title, interrupted=False)


@app.post("/api/pick-dir")
def pick_dir() -> dict[str, str]:
    """拉起本机系统目录选择器，返回可供后端访问的绝对路径。"""
    selected = _pick_dir_by_system_dialog()
    if not selected:
        selected = _pick_dir_by_tkinter()
    if not selected:
        raise HTTPException(status_code=400, detail="未选择项目目录")
    return {"workdir": str(Path(selected).expanduser().resolve())}


def _pick_dir_by_system_dialog() -> str:
    system = platform.system()
    if system == "Darwin":
        cmd = [
            "osascript",
            "-e",
            'POSIX path of (choose folder with prompt "选择项目目录")',
        ]
    elif system == "Windows":
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
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _pick_dir_by_tkinter() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"当前环境无法打开系统目录选择器：{exc}") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title="选择项目目录")
    finally:
        root.destroy()
    return selected


@app.get("/api/history")
def list_history() -> HistoryResponse:
    return HistoryResponse(projects=memory_store.list_history())


@app.delete("/api/projects")
def delete_project(workdir: str) -> dict[str, Any]:
    resolved = str(Path(workdir).expanduser().resolve())
    deleted = memory_store.delete_project(resolved)
    return {"ok": True, "deleted": deleted}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, workdir: str) -> SessionInfo:
    resolved = str(Path(workdir).expanduser().resolve())
    return SessionInfo(
        id=session_id,
        workdir=resolved,
        title=memory_store.session_title(resolved, session_id) or session_id,
        interrupted=memory_store.interrupted(resolved, session_id),
    )


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, req: SessionUpdate) -> SessionInfo:
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
    resolved = str(Path(workdir).expanduser().resolve())
    deleted = memory_store.delete_session(resolved, session_id)
    return {"ok": True, "deleted": deleted}


@app.get("/api/memory")
def read_memory(workdir: str, session_id: str | None = None) -> dict[str, Any]:
    resolved = str(Path(workdir).expanduser().resolve())
    return {
        "global": memory_store.global_memory(),
        "project": memory_store.project_memory(resolved),
        "session": memory_store.read_session(resolved, session_id, limit=200) if session_id else [],
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
    workdir = _session_workdir(req.session_id)
    memory_store.append(workdir, req.session_id, "user", "input", "message", req.text)

    def emit(event: Any) -> None:
        event_queue.put({"type": "event", "data": event.model_dump()})

    def worker() -> None:
        try:
            graph = AgentGraph(model_store, memory_store, emit=emit)
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
            event_queue.put({"type": "error", "data": {"msg": str(exc)}})
        finally:
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> Any:
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _session_workdir(session_id: str) -> str:
    """从 memory/data 里反查会话工作目录。

    会话文件分散在项目记忆目录中，因此这里扫描 sessions 文件。数量很大时可换成索引文件。
    """
    workdir = memory_store.find_session_workdir(session_id)
    if workdir:
        return workdir
    raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
