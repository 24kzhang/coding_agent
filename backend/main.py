from __future__ import annotations

import json
import platform
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from api.schema import (
    AgentModelMap,
    AgentProfileMap,
    AttachmentInfo,
    CapabilitiesResponse,
    ChatRequest,
    GuidanceRequest,
    GuidanceResponse,
    HistoryResponse,
    McpAuthStatus,
    ModelConfig,
    ModelPublic,
    SessionCreate,
    SessionInfo,
    SessionUpdate,
    TaskResult,
)
from backend.agents.graph import AgentGraph, new_session_id
from backend.capabilities import CapabilityStore, McpClient
from backend.memory import MemoryStore
from backend.runs import RunConflictError, RunNotFoundError, RunRegistry
from llm import LlmClient, LlmError, ModelStore

# app 是 FastAPI 应用实例，所有 HTTP 路由都挂在它上面。
app = FastAPI(title="多智能体编程系统", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    # 仅允许本机动态端口访问，避免其他网页借助浏览器调用本地文件工具接口。
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Run-ID"],
)

# model_store 是模型配置仓库，读写 memory/config/models.json。
model_store = ModelStore()
# memory_store 是记忆仓库，读写 memory/data 下的会话、项目和全局记忆。
memory_store = MemoryStore()
# capability_store 负责发现项目根目录 skills 和读取 MCP/servers.json。
capability_store = CapabilityStore()
# mcp_client 通过官方 MCP SDK 测试、发现和调用用户选择的服务。
mcp_client = McpClient()
# run_registry 保存当前进程中的活动任务和运行中指导队列。
run_registry = RunRegistry()


@app.get("/api/health")
def health() -> dict[str, Any]:
    """健康检查接口，前端或用户可用它确认后端是否启动。"""

    return {"ok": True, "msg": "后端运行中"}


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    """返回模型列表和每个智能体的模型映射。"""

    return {
        "models": [model.model_dump() for model in model_store.public()],
        "agent_models": model_store.model_map().model_dump(),
        "agent_profiles": model_store.profiles().model_dump(),
    }


@app.post("/api/models")
def save_model(cfg: ModelConfig) -> ModelPublic:
    """新增或更新模型配置。"""

    try:
        saved = model_store.upsert(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return next(item for item in model_store.public() if item.id == saved.id)


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


@app.post("/api/models/{model_id}/probe")
def probe_model(model_id: str) -> dict[str, Any]:
    """探测模型连通性并返回当前声明的协议能力。"""

    try:
        cfg = model_store.get(model_id)
        result = LlmClient(cfg).test()
    except (KeyError, LlmError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **result,
        "protocol": cfg.protocol,
        "capabilities": cfg.capabilities,
        "efforts": cfg.efforts,
        "default_effort": cfg.default_effort,
    }


@app.post("/api/model-map")
def save_model_map(model_map: AgentModelMap) -> AgentModelMap:
    """保存每个智能体对应的模型 id。"""

    try:
        return model_store.set_model_map(model_map)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/agent-profiles")
def save_agent_profiles(profiles: AgentProfileMap) -> AgentProfileMap:
    """保存模型型 Agent 的模型与思考等级。"""

    try:
        return model_store.set_profiles(profiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/capabilities")
def list_capabilities() -> CapabilitiesResponse:
    """返回前端可手动选择的 Skills 和已启用 MCP 服务。"""

    try:
        return CapabilitiesResponse(
            skills=capability_store.list_skills(),
            mcps=capability_store.list_mcps(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/mcps/{server_id}/test")
def test_mcp(server_id: str, workdir: str) -> dict[str, Any]:
    """测试 MCP 初始化和工具发现，不执行任何 MCP 工具。"""

    # resolved 是占位符 ${workspace} 使用的真实项目目录。
    resolved = _resolve_workdir(workdir)
    try:
        # config 必须来自已启用配置，不能由接口临时注入命令或 URL。
        config = capability_store.select_mcps([server_id])[0]
        return mcp_client.test(config, resolved)
    except Exception as exc:
        # 官方 MCP SDK 的异步上下文在子进程失败时可能抛出 ExceptionGroup，
        # API 统一转成可读 400，避免前端只能看到内部服务器错误。
        raise HTTPException(status_code=400, detail=f"MCP 连接失败：{exc}") from exc


@app.post("/api/mcps/{server_id}/auth")
def start_mcp_auth(server_id: str, workdir: str) -> McpAuthStatus:
    """发起 MCP 浏览器授权，并返回官方服务提供的一次性地址。"""

    resolved = _resolve_workdir(workdir)
    try:
        config = capability_store.select_mcps([server_id])[0]
        return mcp_client.start_auth(config, resolved)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"MCP 授权启动失败：{exc}") from exc


@app.get("/api/mcps/{server_id}/auth")
def get_mcp_auth(server_id: str, workdir: str) -> McpAuthStatus:
    """轮询 MCP 浏览器授权状态，不返回令牌或身份工具结果。"""

    resolved = _resolve_workdir(workdir)
    try:
        config = capability_store.select_mcps([server_id])[0]
        return mcp_client.auth_status(config, resolved)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"MCP 授权状态读取失败：{exc}") from exc


@app.post("/api/sessions")
def create_session(req: SessionCreate) -> SessionInfo:
    """创建新会话，并写入一条 session/start 记忆。"""

    # workdir 是用户选择的真实项目目录，统一解析成绝对路径。
    workdir = _resolve_workdir(req.workdir)
    # session_id 是短 uuid，用于 jsonl 文件名和前端状态。
    session_id = new_session_id()
    # title 是会话展示名；没有传标题时默认用 session_id。
    title = " ".join((req.title or "").strip().split())[:60] or session_id
    # 写入会话开始记录，后续历史会话列表从这里读取初始标题。
    memory_store.append(workdir, session_id, "manager", "session", "start", f"新会话：{title}", {"title": title, "custom": bool(req.title)})
    return SessionInfo(
        id=session_id,
        workdir=workdir,
        title=title,
        interrupted=False,
        skills=[],
        mcps=[],
        container_shell=True,
    )


@app.post("/api/sessions/{session_id}/files")
async def upload_session_file(session_id: str, file: UploadFile = File(...)) -> AttachmentInfo:
    """上传一张会话图片，正文只保存到附件目录。"""

    workdir = _session_workdir(session_id)
    content = await file.read(20 * 1024 * 1024 + 1)
    try:
        info = memory_store.save_file(
            workdir,
            session_id,
            file.filename or "image",
            file.content_type or "application/octet-stream",
            content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AttachmentInfo(
        id=info["id"],
        name=info["name"],
        mime=info["mime"],
        size=info["size"],
        url=f"/api/sessions/{session_id}/files/{info['id']}",
    )


@app.get("/api/sessions/{session_id}/files/{file_id}")
def get_session_file(session_id: str, file_id: str) -> FileResponse:
    """返回会话图片，仅通过哈希 id 访问。"""

    workdir = _session_workdir(session_id)
    try:
        info = memory_store.file_info(workdir, session_id, file_id)
    except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="附件不存在") from exc
    return FileResponse(info["path"], media_type=info["mime"], filename=info["name"])


@app.delete("/api/sessions/{session_id}/files/{file_id}")
def delete_session_file(session_id: str, file_id: str) -> dict[str, Any]:
    """删除当前会话的一张附件。"""

    workdir = _session_workdir(session_id)
    try:
        deleted = memory_store.delete_file(workdir, session_id, file_id)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "deleted": deleted}


@app.post("/api/pick-dir")
def pick_dir() -> dict[str, str]:
    """拉起本机系统目录选择器，返回可供后端访问的绝对路径。"""
    # selected 是系统原生目录选择器返回的路径。
    selected = _pick_dir_by_system_dialog()
    if not selected:
        # FastAPI 的同步路由运行在线程池中，不能在这里创建 tkinter 窗口。
        # macOS 会因此抛出 “NSWindow should only be instantiated on the main thread” 并终止进程。
        raise HTTPException(status_code=400, detail="未选择项目目录或系统目录选择器不可用")
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


@app.get("/api/history")
def list_history() -> HistoryResponse:
    """返回所有项目和会话历史，供前端历史弹窗展示。"""

    return HistoryResponse(projects=memory_store.list_history())


@app.delete("/api/projects")
def delete_project(workdir: str) -> dict[str, Any]:
    """删除某个项目对应的 agent memory，不删除真实项目目录。"""

    # resolved 是规范化后的真实项目路径，用于定位项目 memory。
    resolved = _resolve_workdir(workdir, require_exists=False)
    # deleted 表示 memory 目录是否实际存在并被删除。
    deleted = memory_store.delete_project(resolved)
    return {"ok": True, "deleted": deleted}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, workdir: str) -> SessionInfo:
    """读取当前会话基本信息。"""

    # resolved 是规范化后的项目路径。
    resolved = _resolve_workdir(workdir, require_exists=False)
    # capabilities 是该会话最近一次请求选择的 Skills 和 MCP 服务。
    capabilities = memory_store.session_capabilities(resolved, session_id)
    return SessionInfo(
        id=session_id,
        workdir=resolved,
        title=memory_store.session_title(resolved, session_id) or session_id,
        interrupted=memory_store.interrupted(resolved, session_id),
        **capabilities,
    )


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, req: SessionUpdate) -> SessionInfo:
    """重命名会话，并返回更新后的会话信息。"""

    # resolved 是规范化后的项目路径。
    resolved = _resolve_workdir(req.workdir, require_exists=False)
    try:
        memory_store.rename_session(resolved, session_id, req.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # capabilities 在重命名时保持不变，一并返回便于前端合同稳定。
    capabilities = memory_store.session_capabilities(resolved, session_id)
    return SessionInfo(
        id=session_id,
        workdir=resolved,
        title=memory_store.session_title(resolved, session_id) or session_id,
        interrupted=memory_store.interrupted(resolved, session_id),
        **capabilities,
    )


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, workdir: str) -> dict[str, Any]:
    """删除指定会话 memory 文件。"""

    # resolved 是规范化后的项目路径。
    resolved = _resolve_workdir(workdir, require_exists=False)
    # deleted 表示会话 jsonl 是否实际存在并被删除。
    deleted = memory_store.delete_session(resolved, session_id)
    return {"ok": True, "deleted": deleted}


@app.get("/api/memory")
def read_memory(workdir: str, session_id: str | None = None) -> dict[str, Any]:
    """调试接口：读取全局、项目和可选会话记忆。"""

    # resolved 是规范化后的项目路径。
    resolved = _resolve_workdir(workdir, require_exists=False)
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
    attachments: list[dict[str, Any]] = []
    total_attachment_size = 0
    for file_id in req.attachments:
        try:
            info = memory_store.file_info(workdir, req.session_id, file_id)
        except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"附件不存在或已损坏：{file_id}") from exc
        total_attachment_size += int(info["size"])
        if total_attachment_size > 64 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="本轮附件总大小不能超过 64 MiB")
        attachments.append(
            {
                "id": file_id,
                "name": info["name"],
                "mime": info["mime"],
                "size": info["size"],
                "path": str(info["path"]),
            }
        )
    try:
        # skill_contexts 是受字符预算限制的 SKILL.md 正文。
        skill_contexts = capability_store.load_skills(req.skills)
        # mcp_configs 仅能来自 MCP/servers.json 中已启用服务。
        mcp_configs = capability_store.select_mcps(req.mcps)
    except (OSError, UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        # active_run 让后续指导能够定位当前 Agent，并阻止同一会话并发改动项目。
        active_run = run_registry.start(req.session_id, workdir, event_queue)
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        # resuming 必须在写入本轮记录前判断，否则当前用户输入会被误认为上轮中断。
        resuming = memory_store.interrupted(workdir, req.session_id)
        # 能力与 Shell 选择独立写入 session memory，恢复历史会话时前端会同步还原。
        memory_store.append(
            workdir,
            req.session_id,
            "manager",
            "capability",
            "selection",
            f"本轮启用 {len(skill_contexts)} 个 Skill、{len(mcp_configs)} 个 MCP，"
            + ("容器 Shell" if req.container_shell else "原生 Shell"),
            {
                "skills": [item.id for item in skill_contexts],
                # 会话只恢复用户手动选择；always_on 服务由后端在每轮重新合并。
                "mcps": list(dict.fromkeys(req.mcps)),
                "container_shell": req.container_shell,
            },
        )
        # 先记录用户输入，这样即使任务中途异常，也能在历史会话中看到用户发过什么。
        memory_store.append(
            workdir,
            req.session_id,
            "user",
            "input",
            "message",
            model_store.redact_text(req.text),
            {"attachments": [item["id"] for item in attachments]},
        )
        # run/start 是中断恢复的明确边界；正常、失败两种出口都会写对应终态。
        memory_store.append(workdir, req.session_id, "manager", "run", "start", "本轮任务开始")
    except Exception:
        run_registry.finish(active_run.id)
        raise

    def emit(event: Any) -> None:
        """AgentGraph 调用的事件回调，把事件放入流式队列。"""

        active_run.publish({"type": "event", "data": event.model_dump()})

    def worker() -> None:
        """后台线程函数，负责真正运行 AgentGraph。"""

        try:
            # agent 是本次任务的 LangGraph 编排实例，emit 用于实时输出事件。
            graph = AgentGraph(
                model_store,
                memory_store,
                emit=emit,
                mcp_client=mcp_client,
                guidance_source=active_run.drain_guidance,
            )
            # result 是任务最终结果，会以 type=result 写入队列。
            result = graph.run(
                session_id=req.session_id,
                workdir=workdir,
                text=req.text,
                plan_mode=req.plan_mode,
                execute_plan=req.execute_plan,
                model_id=req.model_id,
                effort=req.effort,
                resuming=resuming,
                skills=skill_contexts,
                mcp_servers=[item.model_dump() for item in mcp_configs],
                container_shell=req.container_shell,
                attachments=attachments,
                required_mcps=req.mcps,
            )
            memory_store.append(workdir, req.session_id, "manager", "run", "done", "本轮任务正常结束")
            active_run.publish({"type": "result", "data": result.model_dump()})
        except Exception as exc:
            # failure 会写入最终会话回复，历史恢复后仍能看到异常，而不是只留半轮对话。
            failure = TaskResult(
                ok=False,
                summary=model_store.redact_text(f"任务执行失败：{exc}"),
            )
            memory_store.append(
                workdir,
                req.session_id,
                "manager",
                "final",
                "result",
                failure.summary,
                failure.model_dump(),
            )
            memory_store.append(workdir, req.session_id, "manager", "run", "error", failure.summary)
            # 捕获异常并转成流式 error，避免前端一直等待。
            active_run.publish({"type": "error", "data": {"msg": model_store.redact_text(str(exc)), "result": failure.model_dump()}})
        finally:
            # None 是流结束哨兵，同时让运行进入可恢复的最近完成列表。
            active_run.publish(None)
            run_registry.finish(active_run.id)

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> Any:
        """StreamingResponse 使用的生成器，把队列数据逐行编码为 NDJSON。"""

        for item in active_run.replay():
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Run-ID": active_run.id,
        },
    )


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, cursor: int = 0) -> StreamingResponse:
    """从事件游标恢复中断的运行流；最近完成的运行也可以完整重放。"""

    try:
        active_run = run_registry.get(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def stream() -> Any:
        for item in active_run.replay(cursor):
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Run-ID": active_run.id,
        },
    )


@app.post("/api/runs/{run_id}/guidance")
def submit_guidance(run_id: str, req: GuidanceRequest) -> GuidanceResponse:
    """把用户指导加入当前 Agent 运行，不启动新的任务。"""

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="指导内容不能为空")
    try:
        active_run, guidance = run_registry.submit_guidance(run_id, text)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    memory_store.append(
        active_run.workdir,
        active_run.session_id,
        "user",
        "guidance",
        "message",
        model_store.redact_text(text),
        {"run_id": run_id, "guidance_id": guidance.id},
    )
    return GuidanceResponse(
        run_id=run_id,
        guidance_id=guidance.id,
        accepted_at=guidance.created_at,
    )


def _session_workdir(session_id: str) -> str:
    """从 memory/data 里反查会话工作目录。

    会话文件分散在项目记忆目录中，因此这里扫描 sessions 文件。数量很大时可换成索引文件。
    """
    # workdir 是 MemoryStore 根据 session_id 找到的真实项目目录。
    try:
        workdir = memory_store.find_session_workdir(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if workdir:
        return workdir
    raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")


def _resolve_workdir(value: str, *, require_exists: bool = True) -> str:
    """规范化项目目录，并在需要时验证目录真实存在。"""

    # path 是用户选择目录展开 ``~`` 后的绝对路径。
    path = Path(value).expanduser().resolve()
    if require_exists and not path.is_dir():
        raise HTTPException(status_code=400, detail=f"项目目录不存在或不是文件夹：{path}")
    return str(path)
