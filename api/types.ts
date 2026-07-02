export type AgentName = "manager" | "planner" | "repo" | "coder" | "verifier" | "doc";

export interface ModelConfig {
  id: string;
  name: string;
  base_url: string;
  api_key: string;
  model: string;
  ctx: number;
  enabled: boolean;
  timeout: number;
}

export interface AgentModelMap {
  manager: string;
  planner: string;
  repo: string;
  coder: string;
  verifier: string;
  doc: string;
}

export interface SessionInfo {
  id: string;
  workdir: string;
  title: string;
  interrupted: boolean;
}

export interface SessionUpdate {
  workdir: string;
  title: string;
}

export interface HistoryMessage {
  id: string;
  role: "user" | "agent";
  content: string;
}

export interface HistorySession {
  id: string;
  title: string;
  workdir: string;
  updated_at: string;
  interrupted: boolean;
  messages: HistoryMessage[];
}

export interface HistoryProject {
  id: string;
  name: string;
  workdir: string;
  sessions: HistorySession[];
}

export interface AgentEvent {
  id: number;
  ts: string;
  agent: string;
  kind: string;
  msg: string;
  tokens: number;
  data: Record<string, unknown>;
}

export interface ChatRequest {
  session_id: string;
  text: string;
  plan_mode: boolean;
  execute_plan: boolean;
  model_id?: string | null;
}

export interface TaskResult {
  ok: boolean;
  summary: string;
  files: string[];
  commands: string[];
  tests: Array<Record<string, unknown>>;
  plan_path?: string | null;
  doc_path?: string | null;
}
