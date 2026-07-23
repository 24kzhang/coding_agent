// AgentName 必须与 api/schema.py 中的 AgentName 保持一致。
export type AgentName = "manager" | "planner" | "repo" | "coder" | "verifier" | "doc";

export interface ModelConfig {
  // 模型本地唯一 id。
  id: string;
  // 前端展示名称。
  name: string;
  // OpenAI-compatible 服务基础地址。
  base_url: string;
  // 模型 API key。
  api_key: string;
  // 供应商模型名。
  model: string;
  // 上下文窗口，单位 token。
  ctx: number;
  // 是否启用。
  enabled: boolean;
  // 请求超时时间，单位秒。
  timeout: number;
}

export interface AgentModelMap {
  // 管理者智能体模型 id。
  manager: string;
  // Plan 智能体模型 id。
  planner: string;
  // 仓库读取智能体模型 id。
  repo: string;
  // Coding 智能体模型 id。
  coder: string;
  // 验证测试智能体模型 id。
  verifier: string;
  // 文档智能体模型 id。
  doc: string;
}

export interface SessionInfo {
  // 会话 id。
  id: string;
  // 会话所属项目目录。
  workdir: string;
  // 会话展示名称。
  title: string;
  // 是否疑似中断。
  interrupted: boolean;
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
}

export interface HistorySession {
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
}
