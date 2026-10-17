# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ModelProtocol = Literal["auto", "responses", "chat_completions"]
ReasoningEffort = Literal["inherit", "none", "minimal", "low", "medium", "high", "max"]


class ModelConfig(BaseModel):
    """模型配置。字段名保持短小，便于前端和 json 文件共同使用。"""

    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    name: str = Field(min_length=1, max_length=120)
    base_url: str = Field(min_length=8, max_length=500)
    api_key: str = Field(default="", max_length=1000)
    model: str = Field(min_length=1, max_length=200)
    ctx: int = Field(default=128000, ge=1024, le=10_000_000, description="模型上下文窗口，单位为 token")
    enabled: bool = True
    timeout: int = Field(default=120, ge=5, le=1800)
    protocol: ModelProtocol = "auto"
    capabilities: list[Literal["text", "vision", "tools", "reasoning"]] = Field(
        default_factory=lambda: ["text", "vision", "tools", "reasoning"]
    )
    efforts: list[ReasoningEffort] = Field(
        default_factory=lambda: ["minimal", "low", "medium", "high", "max"]
    )
    default_effort: ReasoningEffort = "low"
    max_output_tokens: int = Field(default=16_384, ge=256, le=262_144)
    output_token_field: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    reasoning_style: Literal["auto", "effort", "thinking", "both", "none"] = "auto"
    effort_map: dict[str, str] = Field(default_factory=dict)
    effort_budgets: dict[str, int] = Field(default_factory=dict)
    stream_tools: bool = False


class ModelPublic(BaseModel):
    """可安全返回给浏览器的模型配置，不包含真实密钥。"""

    id: str
    name: str
    base_url: str
    model: str
    ctx: int
    enabled: bool
    timeout: int
    protocol: ModelProtocol
    capabilities: list[str]
    efforts: list[ReasoningEffort]
    default_effort: ReasoningEffort
    max_output_tokens: int
    output_token_field: str = "auto"
    reasoning_style: str = "auto"
    effort_map: dict[str, str] = Field(default_factory=dict)
    effort_budgets: dict[str, int] = Field(default_factory=dict)
    stream_tools: bool = False
    has_key: bool
    key_hint: str = ""


AgentName = Literal["manager", "planner", "coder", "verifier", "answer", "doc", "compact"]


class AgentModelMap(BaseModel):
    """每个智能体可以单独绑定模型。"""

    manager: str = "longcat"
    planner: str = "longcat"
    coder: str = "longcat"
    verifier: str = "longcat"
    answer: str = "longcat"
    doc: str = "longcat"
    compact: str = "longcat"


class AgentProfile(BaseModel):
    """一个模型型 Agent 的运行配置。"""

    model_id: str = ""
    effort: ReasoningEffort = "inherit"


class AgentProfileMap(BaseModel):
    """所有模型型 Agent 的模型与思考等级设置。"""

    manager: AgentProfile = Field(default_factory=AgentProfile)
    planner: AgentProfile = Field(default_factory=AgentProfile)
    coder: AgentProfile = Field(default_factory=AgentProfile)
    verifier: AgentProfile = Field(default_factory=AgentProfile)
    answer: AgentProfile = Field(default_factory=AgentProfile)
    doc: AgentProfile = Field(default_factory=AgentProfile)
    compact: AgentProfile = Field(default_factory=AgentProfile)


class ChatRequest(BaseModel):
    """前端点击发送后提交给 /api/chat/stream 的请求体。"""

    session_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=200_000)
    plan_mode: bool = False
    execute_plan: bool = False
    model_id: str | None = None
    skills: list[str] = Field(default_factory=list, max_length=20)
    mcps: list[str] = Field(default_factory=list, max_length=20)
    container_shell: bool = False
    attachments: list[str] = Field(default_factory=list, max_length=8)
    effort: ReasoningEffort | None = None


class AttachmentInfo(BaseModel):
    """上传后返回给前端的安全附件元数据。"""

    id: str
    name: str
    mime: str
    size: int
    url: str


class TaskEnvelope(BaseModel):
    """父 Agent 派发给子 Agent 的最小任务合同。"""

    id: str
    parent_id: str | None = None
    sender: str
    receiver: str
    goal: str
    input_refs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    budget: dict[str, int] = Field(default_factory=dict)


class TaskOutcome(BaseModel):
    """子 Agent 返回给父 Agent 的结果合同。"""

    task_id: str
    status: Literal["ok", "failed", "needs_clarification"]
    summary: str
    facts: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    question: str | None = None
    error: str | None = None


class GuidanceRequest(BaseModel):
    """Agent 运行中追加指导的请求体。"""

    text: str = Field(min_length=1, max_length=20_000)


class GuidanceResponse(BaseModel):
    """指导进入当前运行队列后的确认信息。"""

    run_id: str
    guidance_id: str
    status: Literal["queued"] = "queued"
    accepted_at: str


class SessionCreate(BaseModel):
    """创建新会话时的请求体。"""

    workdir: str = Field(min_length=1, max_length=4000)
    title: str | None = None


class SessionUpdate(BaseModel):
    """修改会话信息时的请求体，目前只支持修改会话名称。"""

    workdir: str = Field(min_length=1, max_length=4000)
    title: str = Field(min_length=1, max_length=60)


class SessionInfo(BaseModel):
    """后端返回给前端的当前会话基本信息。"""

    id: str
    workdir: str
    title: str
    interrupted: bool = False
    skills: list[str] = Field(default_factory=list)
    mcps: list[str] = Field(default_factory=list)
    container_shell: bool = True


class HistoryMessage(BaseModel):
    """历史会话弹窗中恢复到对话窗口的单条消息。"""

    id: str
    role: Literal["user", "agent"]
    content: str
    attachments: list[AttachmentInfo] = Field(default_factory=list)


class HistorySession(BaseModel):
    """历史项目下的一个会话摘要。"""

    id: str
    title: str
    workdir: str
    updated_at: str
    interrupted: bool = False
    result: TaskResult | None = None
    messages: list[HistoryMessage] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    mcps: list[str] = Field(default_factory=list)
    container_shell: bool = True


class HistoryProject(BaseModel):
    """历史会话弹窗中的项目分组。"""

    id: str
    name: str
    workdir: str
    sessions: list[HistorySession] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """GET /api/history 的响应体。"""

    projects: list[HistoryProject] = Field(default_factory=list)


class AgentEvent(BaseModel):
    """后端流式返回给前端事件流的单条事件。"""

    id: int
    ts: str
    agent: str
    kind: str
    msg: str
    tokens: int = 0
    data: dict[str, Any] = Field(default_factory=dict)


class SkillInfo(BaseModel):
    """前端能力选择器展示的 Skill 摘要。"""

    id: str
    name: str
    description: str
    path: str


class SkillContext(BaseModel):
    """用户选择后真正传给下游智能体的 Skill 内容。"""

    id: str
    name: str
    description: str
    content: str


class McpToolInfo(BaseModel):
    """MCP 服务通过 tools/list 暴露的单个工具定义。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class McpInfo(BaseModel):
    """前端能力选择器展示的 MCP 服务安全摘要。"""

    id: str
    name: str
    description: str = ""
    transport: Literal["stdio", "streamable_http"]
    enabled: bool = True
    auth: bool = False
    always_on: bool = False
    shared: bool = False


class McpAuthStatus(BaseModel):
    """MCP 浏览器授权流程的当前状态。"""

    status: Literal["starting", "authorization_required", "authorized", "error", "idle"]
    msg: str
    url: str | None = None
    code: str | None = None


class CapabilitiesResponse(BaseModel):
    """GET /api/capabilities 返回的可选 Skills 和 MCP 服务。"""

    skills: list[SkillInfo] = Field(default_factory=list)
    mcps: list[McpInfo] = Field(default_factory=list)


class ContextPackage(BaseModel):
    """管理者派发给下游智能体的结构化上下文包。"""

    goal: str
    task_type: str
    workdir: str
    plan_mode: bool
    relevant_files: list[str] = Field(default_factory=list)
    project_memory: str = ""
    global_memory: str = ""
    constraints: list[str] = Field(default_factory=list)
    recent: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    skills: list[SkillContext] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class TaskResult(BaseModel):
    """一次任务结束后返回给前端的最终结果。"""

    ok: bool
    summary: str
    files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    plan_path: str | None = None
    doc_path: str | None = None
    tokens: int = 0
    duration_ms: int = 0
    metrics_estimated: bool = False
