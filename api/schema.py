from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 定义前后端数据传输格式

class ModelConfig(BaseModel):
    """模型配置。字段名保持短小，便于前端和 json 文件共同使用。"""

    id: str
    name: str
    base_url: str
    api_key: str
    model: str
    ctx: int = Field(default=128000, description="模型上下文窗口，单位为 token")
    enabled: bool = True
    timeout: int = 120


AgentName = Literal["manager", "planner", "repo", "coder", "verifier", "doc"]


class AgentModelMap(BaseModel):
    """每个智能体可以单独绑定模型。"""

    manager: str = "longcat"
    planner: str = "longcat"
    repo: str = "longcat"
    coder: str = "longcat"
    verifier: str = "longcat"
    doc: str = "longcat"


class ChatRequest(BaseModel):
    session_id: str
    text: str
    plan_mode: bool = False
    execute_plan: bool = False
    model_id: str | None = None


class SessionCreate(BaseModel):
    workdir: str
    title: str | None = None


class SessionUpdate(BaseModel):
    workdir: str
    title: str


class SessionInfo(BaseModel):
    id: str
    workdir: str
    title: str
    interrupted: bool = False


class HistoryMessage(BaseModel):
    id: str
    role: Literal["user", "agent"]
    content: str


class HistorySession(BaseModel):
    id: str
    title: str
    workdir: str
    updated_at: str
    interrupted: bool = False
    messages: list[HistoryMessage] = Field(default_factory=list)


class HistoryProject(BaseModel):
    id: str
    name: str
    workdir: str
    sessions: list[HistorySession] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    projects: list[HistoryProject] = Field(default_factory=list)


class AgentEvent(BaseModel):
    id: int
    ts: str
    agent: str
    kind: str
    msg: str
    tokens: int = 0
    data: dict[str, Any] = Field(default_factory=dict)


class ContextPackage(BaseModel):
    goal: str
    task_type: str
    workdir: str
    plan_mode: bool
    relevant_files: list[str] = Field(default_factory=list)
    project_memory: str = ""
    global_memory: str = ""
    constraints: list[str] = Field(default_factory=list)
    recent: list[str] = Field(default_factory=list)


class TaskResult(BaseModel):
    ok: bool
    summary: str
    files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    plan_path: str | None = None
    doc_path: str | None = None
