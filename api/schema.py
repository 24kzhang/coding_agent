from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 这个文件只放“前后端通信合同”，不要放业务逻辑。
# 后端 FastAPI、前端 TypeScript、LangGraph 返回值都围绕这些结构传输。

ModelProtocol = Literal["auto", "responses", "chat_completions"]
ReasoningEffort = Literal["inherit", "minimal", "low", "medium", "high", "max"]


class ModelConfig(BaseModel):
    """模型配置。字段名保持短小，便于前端和 json 文件共同使用。"""

    # 模型在本地配置文件中的唯一 id，前端下拉框和智能体映射都用它引用模型。
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    # 给用户看的模型名称，例如“LongCat 默认模型”。
    name: str = Field(min_length=1, max_length=120)
    # OpenAI-compatible 服务的基础地址，LlmClient 会自动拼接 chat completions 路径。
    base_url: str = Field(min_length=8, max_length=500)
    # 模型 API key，当前项目按需求明文保存到 memory/config/models.json。
    api_key: str = Field(default="", max_length=1000)
    # 供应商实际要求的模型名，例如 LongCat-2.0。
    model: str = Field(min_length=1, max_length=200)
    # 模型上下文窗口大小，MemoryStore 会用它判断会话记忆是否需要压缩。
    ctx: int = Field(default=128000, ge=1024, le=10_000_000, description="模型上下文窗口，单位为 token")
    # 模型是否启用；没有显式选择模型时，ModelStore 会优先使用启用模型。
    enabled: bool = True
    # 单次 HTTP 请求超时时间，避免模型接口长时间无响应导致后端卡死。
    timeout: int = Field(default=120, ge=5, le=1800)
    # protocol 决定优先使用 Responses 还是 Chat Completions；auto 会先探测并缓存。
    protocol: ModelProtocol = "auto"
    # capabilities 用于前端和路由器判断模型是否支持视觉、原生工具和推理。
    capabilities: list[Literal["text", "vision", "tools", "reasoning"]] = Field(
        default_factory=lambda: ["text", "vision", "tools", "reasoning"]
    )
    # efforts 是该模型允许用户选择的思考等级。
    efforts: list[ReasoningEffort] = Field(
        default_factory=lambda: ["minimal", "low", "medium", "high", "max"]
    )
    # default_effort 是没有 Agent 或单次请求覆盖时使用的等级。
    # 低思考足以覆盖日常工具编排，也能避免推理 token 挤占工具参数输出预算。
    default_effort: ReasoningEffort = "low"
    # max_output_tokens 为各 Agent 的输出预算提供模型级上限。
    max_output_tokens: int = Field(default=16_384, ge=256, le=262_144)


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
    has_key: bool
    key_hint: str = ""


# 系统内置智能体名称；如果新增智能体，要同步修改这里、AgentModelMap、前端 agents 数组和 AgentGraph。
AgentName = Literal["manager", "planner", "coder", "verifier", "answer", "doc", "compact"]


class AgentModelMap(BaseModel):
    """每个智能体可以单独绑定模型。"""

    # 上下文管理智能体使用的模型 id，负责分类、路由和上下文包构造。
    manager: str = "longcat"
    # Plan 生成智能体使用的模型 id，负责需求澄清和计划生成。
    planner: str = "longcat"
    # Coding 智能体使用的模型 id，负责 ReAct 写代码。
    coder: str = "longcat"
    # 验证测试智能体使用的模型 id；当前 verifier 主要规则执行，保留映射便于后续扩展。
    verifier: str = "longcat"
    # 普通问答智能体使用的模型 id。
    answer: str = "longcat"
    # 文档生成智能体使用的模型 id，负责生成 README 或 docs 文档。
    doc: str = "longcat"
    # 隐藏的上下文压缩智能体使用的模型 id。
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

    # 当前会话 id；后端会通过它反查该会话所属项目目录。
    session_id: str = Field(min_length=1, max_length=80)
    # 用户本轮输入的原始文本。
    text: str = Field(min_length=1, max_length=200_000)
    # 是否开启 Plan 模式；这是前端开关，不完全依赖自然语言判断。
    plan_mode: bool = False
    # 是否直接执行已有计划；当前前端不展示按钮，但保留字段支持 API 和后续快捷操作。
    execute_plan: bool = False
    # 临时覆盖模型 id；为空时按 AgentModelMap 选择每个智能体的模型。
    model_id: str | None = None
    # 本轮任务启用的 Skill id 列表；后端会从项目根目录 skills 中读取并校验。
    skills: list[str] = Field(default_factory=list, max_length=20)
    # 本轮任务启用的 MCP 服务 id 列表；后端只连接 MCP/servers.json 中已启用的服务。
    mcps: list[str] = Field(default_factory=list, max_length=20)
    # 是否让 Coder 和 Verifier 在本轮共享 Docker 容器；API 默认保留旧的原生行为。
    container_shell: bool = False
    # attachments 是通过会话文件接口上传后的附件 id，不接受客户端文件路径。
    attachments: list[str] = Field(default_factory=list, max_length=8)
    # effort 临时覆盖本轮所有模型型 Agent 的思考等级。
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

    # 指导会在下一个安全检查点进入 Agent 上下文，不会启动第二个任务。
    text: str = Field(min_length=1, max_length=20_000)


class GuidanceResponse(BaseModel):
    """指导进入当前运行队列后的确认信息。"""

    # 当前运行 id，前端用它避免把旧指导发送到下一轮任务。
    run_id: str
    # 单条指导 id，用于事件和历史记录关联。
    guidance_id: str
    # queued 表示后端已接收，真正应用时会通过原任务事件流通知前端。
    status: Literal["queued"] = "queued"
    # 后端接收指导的 UTC 时间。
    accepted_at: str


class SessionCreate(BaseModel):
    """创建新会话时的请求体。"""

    # 用户选择的项目工作目录，agent 的文件工具只能在这个目录内操作。
    workdir: str = Field(min_length=1, max_length=4000)
    # 可选会话名称；为空时前端展示会话 id。
    title: str | None = None


class SessionUpdate(BaseModel):
    """修改会话信息时的请求体，目前只支持修改会话名称。"""

    # 会话所属项目目录，用于定位 memory/data/projects/<项目哈希>/sessions/<会话id>.jsonl。
    workdir: str = Field(min_length=1, max_length=4000)
    # 用户输入的新会话名称，会写入会话 jsonl 的 rename 记录。
    title: str = Field(min_length=1, max_length=60)


class SessionInfo(BaseModel):
    """后端返回给前端的当前会话基本信息。"""

    # 会话 id，同时也是会话 jsonl 文件名。
    id: str
    # 会话对应的项目绝对路径。
    workdir: str
    # 前端展示用会话名称；没有自定义名称时通常等于 id。
    title: str
    # 上次会话是否可能异常中断，判断依据是最后一条 memory 是否为 manager/final/result。
    interrupted: bool = False
    # 该会话最近一次任务选择的 Skills，恢复历史会话时同步还原前端选择。
    skills: list[str] = Field(default_factory=list)
    # 该会话最近一次任务选择的 MCP 服务，恢复历史会话时同步还原前端选择。
    mcps: list[str] = Field(default_factory=list)
    # 该会话最近一次任务是否使用容器 Shell；旧会话按前端新默认值开启。
    container_shell: bool = True


class HistoryMessage(BaseModel):
    """历史会话弹窗中恢复到对话窗口的单条消息。"""

    # 消息来源 memory 记录 id，用于前端 key。
    id: str
    # 消息角色；只恢复用户输入和 agent 最终回复，不恢复全部事件流。
    role: Literal["user", "agent"]
    # 显示在对话窗口里的消息正文。
    content: str


class HistorySession(BaseModel):
    """历史项目下的一个会话摘要。"""

    # 会话 id，也是 jsonl 文件名。
    id: str
    # 会话展示名，优先读取 rename 记录，没有自定义名称时展示 id。
    title: str
    # 该会话所属项目目录。
    workdir: str
    # 会话最后一条 memory 的时间，用于历史列表排序。
    updated_at: str
    # 该会话是否疑似中断。
    interrupted: bool = False
    # 可恢复到前端对话窗口的用户消息和 agent 最终回复。
    messages: list[HistoryMessage] = Field(default_factory=list)
    # 整个会话的事件流；历史恢复时按 JSONL 中的原始顺序显示在右侧事件面板。
    events: list[AgentEvent] = Field(default_factory=list)
    # 会话最近一次任务选择的 Skills。
    skills: list[str] = Field(default_factory=list)
    # 会话最近一次任务选择的 MCP 服务。
    mcps: list[str] = Field(default_factory=list)
    # 会话最近一次任务是否使用容器 Shell。
    container_shell: bool = True


class HistoryProject(BaseModel):
    """历史会话弹窗中的项目分组。"""

    # 项目路径哈希，对应 memory/data/projects/<id>。
    id: str
    # 项目展示名，默认取项目目录 basename。
    name: str
    # 项目绝对路径。
    workdir: str
    # 该项目下的所有历史会话摘要。
    sessions: list[HistorySession] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """GET /api/history 的响应体。"""

    # 所有有 memory 记录的项目，按最近会话时间排序。
    projects: list[HistoryProject] = Field(default_factory=list)


class AgentEvent(BaseModel):
    """后端流式返回给前端事件流的单条事件。"""

    # 本次任务内递增事件 id，只用于前端渲染和排序。
    id: int
    # 事件产生时间。
    ts: str
    # 产生事件的智能体或模块名，例如 manager、coder、llm。
    agent: str
    # 事件类型，例如 start、tool、test、result。
    kind: str
    # 前端事件流显示的中文消息。
    msg: str
    # 本事件关联的 token 估算值，主要由模型调用事件使用。
    tokens: int = 0
    # 额外结构化数据，例如工具执行结果、测试结果等。
    data: dict[str, Any] = Field(default_factory=dict)


class SkillInfo(BaseModel):
    """前端能力选择器展示的 Skill 摘要。"""

    # Skill 唯一 id，来自 SKILL.md front matter 的 name。
    id: str
    # Skill 展示名称。
    name: str
    # Skill 用途摘要，来自 SKILL.md front matter 的 description。
    description: str
    # Skill 在本 Agent 项目中的相对路径。
    path: str


class SkillContext(BaseModel):
    """用户选择后真正传给下游智能体的 Skill 内容。"""

    # Skill 唯一 id。
    id: str
    # Skill 展示名称。
    name: str
    # Skill 用途摘要。
    description: str
    # SKILL.md 完整正文，受能力加载器的单文件和总长度预算限制。
    content: str


class McpToolInfo(BaseModel):
    """MCP 服务通过 tools/list 暴露的单个工具定义。"""

    # MCP 原始工具名，调用 tools/call 时必须原样使用。
    name: str
    # 工具用途说明。
    description: str = ""
    # MCP 标准 JSON Schema 输入定义。
    input_schema: dict[str, Any] = Field(default_factory=dict)


class McpInfo(BaseModel):
    """前端能力选择器展示的 MCP 服务安全摘要。"""

    # MCP 服务唯一 id。
    id: str
    # MCP 服务展示名。
    name: str
    # MCP 服务用途说明。
    description: str = ""
    # 连接类型；前端只展示，不参与连接。
    transport: Literal["stdio", "streamable_http"]
    # 是否启用；当前接口只返回启用服务，保留字段便于后续配置管理。
    enabled: bool = True
    # auth 表示该服务配置了独立授权检查工具，前端据此显示授权按钮。
    auth: bool = False
    # always_on 表示该服务自动进入每轮任务，模型仍按需决定是否调用。
    always_on: bool = False
    # shared 表示服务可安全提供给所有调用模型的智能节点。
    shared: bool = False


class McpAuthStatus(BaseModel):
    """MCP 浏览器授权流程的当前状态。"""

    # status 只描述授权生命周期，不包含 OAuth token 或授权码。
    status: Literal["starting", "authorization_required", "authorized", "error", "idle"]
    # msg 是前端可直接展示的简短中文状态。
    msg: str
    # url 是官方服务提供的一次性授权地址，仅在等待浏览器操作时返回。
    url: str | None = None
    # code 是设备授权时需要在官方页面输入的短期用户码。
    code: str | None = None


class CapabilitiesResponse(BaseModel):
    """GET /api/capabilities 返回的可选 Skills 和 MCP 服务。"""

    # 当前 skills 目录中所有格式有效的 Skill。
    skills: list[SkillInfo] = Field(default_factory=list)
    # 当前 MCP 配置中所有已启用服务。
    mcps: list[McpInfo] = Field(default_factory=list)


class ContextPackage(BaseModel):
    """管理者派发给下游智能体的结构化上下文包。"""

    # 当前任务目标，可能是用户原始输入，也可能是合并 Plan 回答后的任务文本。
    goal: str
    # 管理者分类后的任务类型。
    task_type: str
    # 当前项目工作目录。
    workdir: str
    # 当前任务是否处于 Plan 模式流程。
    plan_mode: bool
    # 仓库读取智能体认为和当前任务相关的文件列表。
    relevant_files: list[str] = Field(default_factory=list)
    # 当前项目长期记忆，来自 memory/data/projects/<项目哈希>/project.md。
    project_memory: str = ""
    # 全局长期记忆，来自 memory/data/global.md。
    global_memory: str = ""
    # 本轮任务必须遵守的约束，例如文件命名、中文文档、安全规则。
    constraints: list[str] = Field(default_factory=list)
    # 最近会话摘要，不是完整历史，用于避免下游上下文过载。
    recent: list[str] = Field(default_factory=list)
    # Agent 执行期间用户追加的指导，越靠后优先级越高。
    guidance: list[str] = Field(default_factory=list)
    # 用户本轮手动选择的 Skill 内容；未选择时为空，不会自动加载全部 Skill。
    skills: list[SkillContext] = Field(default_factory=list)
    # attachments 只保存附件 id、名称、类型和大小，不保存本地路径或正文。
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class TaskResult(BaseModel):
    """一次任务结束后返回给前端的最终结果。"""

    # 任务整体是否成功；验证失败时通常为 False。
    ok: bool
    # 给用户看的最终摘要，也是对话窗口中 Agent 回复的第一段。
    summary: str
    # 本轮任务修改或生成的文件相对路径列表。
    files: list[str] = Field(default_factory=list)
    # 本轮 Coding 智能体执行过的命令列表。
    commands: list[str] = Field(default_factory=list)
    # 验证智能体产生的测试结果列表。
    tests: list[dict[str, Any]] = Field(default_factory=list)
    # Plan 模式生成的计划文件路径；非 Plan 任务为空。
    plan_path: str | None = None
    # 文档智能体写入的文档路径；未生成文档时为空。
    doc_path: str | None = None
    # tokens 是本轮所有模型调用累计 token 数或估算值。
    tokens: int = 0
    # duration_ms 是从 AgentGraph 开始到 final 节点的耗时。
    duration_ms: int = 0
