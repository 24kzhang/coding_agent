from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 这个文件只放“前后端通信合同”，不要放业务逻辑。
# 后端 FastAPI、前端 TypeScript、LangGraph 返回值都围绕这些结构传输。

class ModelConfig(BaseModel):
    """模型配置。字段名保持短小，便于前端和 json 文件共同使用。"""

    # 模型在本地配置文件中的唯一 id，前端下拉框和智能体映射都用它引用模型。
    id: str
    # 给用户看的模型名称，例如“LongCat 默认模型”。
    name: str
    # OpenAI-compatible 服务的基础地址，LlmClient 会自动拼接 chat completions 路径。
    base_url: str
    # 模型 API key，当前项目按需求明文保存到 memory/config/models.json。
    api_key: str
    # 供应商实际要求的模型名，例如 LongCat-2.0。
    model: str
    # 模型上下文窗口大小，MemoryStore 会用它判断会话记忆是否需要压缩。
    ctx: int = Field(default=128000, description="模型上下文窗口，单位为 token")
    # 模型是否启用；没有显式选择模型时，ModelStore 会优先使用启用模型。
    enabled: bool = True
    # 单次 HTTP 请求超时时间，避免模型接口长时间无响应导致后端卡死。
    timeout: int = 120


# 系统内置智能体名称；如果新增智能体，要同步修改这里、AgentModelMap、前端 agents 数组和 AgentGraph。
AgentName = Literal["manager", "planner", "repo", "coder", "verifier", "doc"]


class AgentModelMap(BaseModel):
    """每个智能体可以单独绑定模型。"""

    # 上下文管理智能体使用的模型 id，负责分类、路由和上下文包构造。
    manager: str = "longcat"
    # Plan 生成智能体使用的模型 id，负责需求澄清和计划生成。
    planner: str = "longcat"
    # 仓库读取智能体使用的模型 id；当前 repo 节点主要是规则读取，保留映射便于后续扩展。
    repo: str = "longcat"
    # Coding 智能体使用的模型 id，负责 ReAct 写代码。
    coder: str = "longcat"
    # 验证测试智能体使用的模型 id；当前 verifier 主要规则执行，保留映射便于后续扩展。
    verifier: str = "longcat"
    # 文档生成智能体使用的模型 id，负责生成 README 或 docs 文档。
    doc: str = "longcat"


class ChatRequest(BaseModel):
    """前端点击发送后提交给 /api/chat/stream 的请求体。"""

    # 当前会话 id；后端会通过它反查该会话所属项目目录。
    session_id: str
    # 用户本轮输入的原始文本。
    text: str
    # 是否开启 Plan 模式；这是前端开关，不完全依赖自然语言判断。
    plan_mode: bool = False
    # 是否直接执行已有计划；当前前端不展示按钮，但保留字段支持 API 和后续快捷操作。
    execute_plan: bool = False
    # 临时覆盖模型 id；为空时按 AgentModelMap 选择每个智能体的模型。
    model_id: str | None = None


class SessionCreate(BaseModel):
    """创建新会话时的请求体。"""

    # 用户选择的项目工作目录，agent 的文件工具只能在这个目录内操作。
    workdir: str
    # 可选会话名称；为空时前端展示会话 id。
    title: str | None = None


class SessionUpdate(BaseModel):
    """修改会话信息时的请求体，目前只支持修改会话名称。"""

    # 会话所属项目目录，用于定位 memory/data/projects/<项目哈希>/sessions/<会话id>.jsonl。
    workdir: str
    # 用户输入的新会话名称，会写入会话 jsonl 的 rename 记录。
    title: str


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
