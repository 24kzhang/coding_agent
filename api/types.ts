// 作者：张又升
// 邮箱：1041586605@qq.com
// 单位：西北大学 电子信息学院(人工智能学院)

// AgentName 必须与 api/schema.py 中的 AgentName 保持一致。
export type AgentName = "manager" | "planner" | "coder" | "verifier" | "answer" | "doc" | "compact";
export type ModelProtocol = "auto" | "responses" | "chat_completions";
export type ReasoningEffort = "inherit" | "none" | "minimal" | "low" | "medium" | "high" | "max";

export interface ModelConfig {
  // 模型本地唯一 id。
  id: string;
  // 前端展示名称。
  name: string;
  // OpenAI-compatible 服务基础地址。
  base_url: string;
  // 模型 API key。
  api_key?: string;
  // 供应商模型名。
  model: string;
  // 上下文窗口，单位 token。
  ctx: number;
  // 是否启用。
  enabled: boolean;
  // 请求超时时间，单位秒。
  timeout: number;
  protocol: ModelProtocol;
  capabilities: Array<"text" | "vision" | "tools" | "reasoning">;
  efforts: ReasoningEffort[];
  default_effort: ReasoningEffort;
  max_output_tokens: number;
  // 中转服务的实际参数映射，保存模型时完整保留。
  output_token_field?: "auto" | "max_tokens" | "max_completion_tokens";
  reasoning_style?: "auto" | "effort" | "thinking" | "both" | "none";
  effort_map?: Record<string, string>;
  effort_budgets?: Record<string, number>;
  stream_tools?: boolean;
  has_key?: boolean;
  key_hint?: string;
}

export interface AgentModelMap {
  // 管理者智能体模型 id。
  manager: string;
  // Plan 智能体模型 id。
  planner: string;
  // Coding 智能体模型 id。
  coder: string;
  // 验证测试智能体模型 id。
  verifier: string;
  // 普通问答智能体模型 id。
  answer: string;
  // 文档智能体模型 id。
  doc: string;
  // 隐藏的上下文压缩智能体模型 id。
  compact: string;
}

export interface AgentProfile {
  model_id: string;
  effort: ReasoningEffort;
}

export type AgentProfileMap = Record<AgentName, AgentProfile>;

export interface SessionInfo {
  // 会话 id。
  id: string;
  // 会话所属项目目录。
  workdir: string;
  // 会话展示名称。
  title: string;
  // 是否疑似中断。
  interrupted: boolean;
  // 会话最近一次任务选择的 Skills。
  skills: string[];
  // 会话最近一次任务选择的 MCP 服务。
  mcps: string[];
  // 会话最近一次任务是否使用容器 Shell。
  container_shell: boolean;
}

export interface SessionUpdate {
  // 会话所属项目目录。
  workdir: string;
  // 新会话名称。
  title: string;
}

export interface HistoryMessage {
  // 历史消息 id。
  id: string;
  // 消息角色。
  role: "user" | "agent";
  // 消息正文。
  content: string;
  // 原始图片附件，可在实时与恢复后的消息中查看。
  attachments?: AttachmentInfo[];
}

export interface HistorySession {
  // 恢复最新最终结果中的用量和耗时。
  result?: TaskResult | null;
  // 会话 id。
  id: string;
  // 会话展示名称。
  title: string;
  // 会话所属项目目录。
  workdir: string;
  // 最后更新时间。
  updated_at: string;
  // 是否疑似中断。
  interrupted: boolean;
  // 可恢复到对话窗口的消息。
  messages: HistoryMessage[];
  // 整个会话的事件流摘要。
  events: AgentEvent[];
  // 会话最近一次任务选择的 Skills。
  skills: string[];
  // 会话最近一次任务选择的 MCP 服务。
  mcps: string[];
  // 会话最近一次任务是否使用容器 Shell。
  container_shell: boolean;
}

export interface HistoryProject {
  // 项目 memory id。
  id: string;
  // 项目展示名称。
  name: string;
  // 项目真实目录。
  workdir: string;
  // 项目下的历史会话。
  sessions: HistorySession[];
}

export interface AgentEvent {
  // 本次任务内递增事件 id。
  id: number;
  // 事件时间。
  ts: string;
  // 产生事件的智能体。
  agent: string;
  // 事件类型。
  kind: string;
  // 事件显示文本。
  msg: string;
  // 事件关联 token 数。
  tokens: number;
  // 事件附加结构化数据。
  data: Record<string, unknown>;
}

export interface ChatRequest {
  // 当前会话 id。
  session_id: string;
  // 用户输入。
  text: string;
  // 是否开启 Plan 模式。
  plan_mode: boolean;
  // 是否执行已有计划。
  execute_plan: boolean;
  // 临时覆盖模型 id。
  model_id?: string | null;
  // 本轮手动启用的 Skill id。
  skills: string[];
  // 本轮手动启用的 MCP 服务 id。
  mcps: string[];
  // 是否让 Coder 和 Verifier 在本轮共享 Docker 容器。
  container_shell: boolean;
  // 已上传到本会话的附件 id。
  attachments: string[];
  // 本轮临时思考等级。
  effort?: ReasoningEffort | null;
}

export interface AttachmentInfo {
  id: string;
  name: string;
  mime: string;
  size: number;
  url: string;
}

export interface GuidanceRequest {
  // 指导会进入当前运行，而不是启动第二个任务。
  text: string;
}

export interface GuidanceResponse {
  // 当前活动运行 id。
  run_id: string;
  // 单条指导 id。
  guidance_id: string;
  // queued 表示已经排队等待 Agent 安全应用。
  status: "queued";
  // 后端接收时间。
  accepted_at: string;
}

export interface SkillInfo {
  // Skill 唯一 id。
  id: string;
  // Skill 展示名。
  name: string;
  // SKILL.md 中的用途摘要。
  description: string;
  // Skill 在本 Agent 项目中的相对路径。
  path: string;
}

export interface McpInfo {
  // MCP 服务唯一 id。
  id: string;
  // MCP 服务展示名。
  name: string;
  // MCP 服务用途说明。
  description: string;
  // MCP 连接类型。
  transport: "stdio" | "streamable_http";
  // 服务是否启用。
  enabled: boolean;
  // 是否需要并支持通过浏览器完成独立授权。
  auth: boolean;
  // 是否自动进入每轮任务；模型仍可选择不调用。
  always_on: boolean;
  // 是否可供所有调用模型的智能节点使用。
  shared: boolean;
}

export interface McpAuthStatus {
  // 当前授权流程状态。
  status: "starting" | "authorization_required" | "authorized" | "error" | "idle";
  // 可直接展示给用户的状态说明。
  msg: string;
  // 等待授权时由官方 MCP 返回的一次性 OAuth 地址。
  url?: string | null;
  // GitHub 设备授权页面需要输入的短期用户码。
  code?: string | null;
}

export interface McpToolInfo {
  // MCP 原始工具名。
  name: string;
  // 工具用途说明。
  description: string;
  // 工具输入 JSON Schema。
  input_schema: Record<string, unknown>;
}

export interface CapabilitiesResponse {
  // 可手动选择的 Skills。
  skills: SkillInfo[];
  // 可手动选择的 MCP 服务。
  mcps: McpInfo[];
}

export interface TaskResult {
  // 任务整体是否成功。
  ok: boolean;
  // 最终摘要。
  summary: string;
  // 变更文件列表。
  files: string[];
  // 执行命令列表。
  commands: string[];
  // 验证结果列表。
  tests: Array<Record<string, unknown>>;
  // 计划文件路径。
  plan_path?: string | null;
  // 文档文件路径。
  doc_path?: string | null;
  // 本轮累计 token 用量或估算值。
  tokens: number;
  // 本轮执行总耗时，单位毫秒。
  duration_ms: number;
  // 估算值在界面单独标记。
  metrics_estimated?: boolean;
}
