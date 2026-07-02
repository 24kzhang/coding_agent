import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  Copy,
  Cpu,
  FileText,
  FolderOpen,
  GitBranch,
  History,
  Loader2,
  Play,
  Plus,
  RadioTower,
  Save,
  Settings,
  Trash2,
  XCircle
} from "lucide-react";
import "./style.css";

type AgentName = "manager" | "planner" | "repo" | "coder" | "verifier" | "doc";

interface ModelConfig {
  id: string;
  name: string;
  base_url: string;
  api_key: string;
  model: string;
  ctx: number;
  enabled: boolean;
  timeout: number;
}

interface AgentModelMap {
  manager: string;
  planner: string;
  repo: string;
  coder: string;
  verifier: string;
  doc: string;
}

interface AgentEvent {
  id: number;
  ts: string;
  agent: string;
  kind: string;
  msg: string;
  tokens: number;
  data: Record<string, unknown>;
}

interface TaskResult {
  ok: boolean;
  summary: string;
  files: string[];
  commands: string[];
  tests: Array<Record<string, unknown>>;
  plan_path?: string | null;
  doc_path?: string | null;
}

interface ChatMessage {
  id: string;
  role: "user" | "agent";
  content: string;
}

interface HistorySession {
  id: string;
  title: string;
  workdir: string;
  updated_at: string;
  interrupted: boolean;
  messages: ChatMessage[];
}

interface HistoryProject {
  id: string;
  name: string;
  workdir: string;
  sessions: HistorySession[];
}

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8710";
const agents: Array<{ id: AgentName; name: string }> = [
  { id: "manager", name: "管理者" },
  { id: "planner", name: "Plan" },
  { id: "repo", name: "仓库读取" },
  { id: "coder", name: "Coding" },
  { id: "verifier", name: "验证测试" },
  { id: "doc", name: "文档" }
];

function App() {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [modelMap, setModelMap] = useState<AgentModelMap>({
    manager: "longcat",
    planner: "longcat",
    repo: "longcat",
    coder: "longcat",
    verifier: "longcat",
    doc: "longcat"
  });
  const [draft, setDraft] = useState<ModelConfig>(emptyModel());
  const [workdir, setWorkdir] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [sessionTitle, setSessionTitle] = useState("");
  const [task, setTask] = useState("");
  const [planMode, setPlanMode] = useState(false);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [result, setResult] = useState<TaskResult | null>(null);
  const [notice, setNotice] = useState("等待会话启动");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [modelModalOpen, setModelModalOpen] = useState(false);
  const [historyModalOpen, setHistoryModalOpen] = useState(false);
  const [historyProjects, setHistoryProjects] = useState<HistoryProject[]>([]);

  useEffect(() => {
    void loadModels();
  }, []);

  const tokenTotal = useMemo(() => events.reduce((sum, event) => sum + event.tokens, 0), [events]);
  const activeAgent = events.at(-1)?.agent ?? "manager";

  async function loadModels() {
    const res = await fetch(`${API}/api/models`);
    const data = await res.json();
    setModels(data.models);
    setModelMap(data.agent_models);
    setDraft(data.models[0] ?? emptyModel());
  }

  async function saveModel(event: FormEvent) {
    event.preventDefault();
    await fetch(`${API}/api/models`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(draft)
    });
    setNotice(`模型已保存：${draft.name}`);
    setModelModalOpen(false);
    await loadModels();
  }

  async function deleteModel(id: string) {
    await fetch(`${API}/api/models/${id}`, { method: "DELETE" });
    setNotice(`模型已删除：${id}`);
    await loadModels();
  }

  async function testModel(id: string) {
    setNotice("正在测试模型连通性");
    const res = await fetch(`${API}/api/models/${id}/test`, { method: "POST" });
    const data = await res.json();
    setNotice(res.ok ? `连通成功：${data.reply}` : `连通失败：${data.detail}`);
  }

  async function saveMap(next: AgentModelMap) {
    setModelMap(next);
    await fetch(`${API}/api/model-map`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(next)
    });
  }

  async function createSession() {
    if (!workdir) {
      setNotice("请先选择项目目录");
      return;
    }
    const res = await fetch(`${API}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workdir })
    });
    const data = await res.json();
    setSessionId(data.id);
    setSessionTitle(data.title);
    setEvents([]);
    setResult(null);
    setMessages([]);
    setNotice(`会话已创建：${data.title}`);
  }

  async function pickProject() {
    setNotice("正在打开目录选择器");
    try {
      const res = await fetch(`${API}/api/pick-dir`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail === "Not Found" ? "后端未加载目录选择接口，请重启项目" : data.detail;
        setNotice(`选择目录失败：${detail}`);
        return;
      }
      setWorkdir(data.workdir);
      setSessionId("");
      setSessionTitle("");
      setEvents([]);
      setResult(null);
      setMessages([]);
      setNotice(`已选择项目：${data.workdir}`);
    } catch (error) {
      setNotice(`选择目录失败：${String(error)}`);
    }
  }

  async function openHistory() {
    setNotice("正在读取历史会话");
    const res = await fetch(`${API}/api/history`);
    const data = await res.json();
    if (!res.ok) {
      const detail = data.detail === "Not Found" ? "后端未加载历史会话接口，请重启项目" : data.detail;
      setNotice(`读取历史失败：${detail}`);
      return;
    }
    setHistoryProjects(data.projects ?? []);
    setHistoryModalOpen(true);
    setNotice("历史会话已加载");
  }

  function restoreSession(session: HistorySession) {
    setWorkdir(session.workdir);
    setSessionId(session.id);
    setSessionTitle(session.title);
    setMessages(session.messages);
    setEvents([]);
    setResult(null);
    setHistoryModalOpen(false);
    setNotice(`已恢复会话：${session.id}`);
  }

  async function renameCurrentSession() {
    if (!sessionId || !workdir) {
      setNotice("当前没有可重命名的会话");
      return;
    }
    const nextTitle = window.prompt("请输入新的会话名称", sessionTitle || sessionId);
    if (nextTitle === null) return;
    const cleanTitle = nextTitle.trim();
    if (!cleanTitle) {
      setNotice("会话名称不能为空");
      return;
    }
    const res = await fetch(`${API}/api/sessions/${sessionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workdir, title: cleanTitle })
    });
    const data = await res.json();
    if (!res.ok) {
      setNotice(`重命名失败：${data.detail}`);
      return;
    }
    setSessionTitle(data.title);
    setNotice(`会话已重命名：${data.title}`);
  }

  async function deleteHistorySession(session: HistorySession) {
    if (!window.confirm(`删除会话「${session.title}」？只会删除 agent memory 中的会话记录。`)) return;
    const res = await fetch(`${API}/api/sessions/${session.id}?workdir=${encodeURIComponent(session.workdir)}`, { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) {
      setNotice(`删除会话失败：${data.detail}`);
      return;
    }
    setHistoryProjects((prev) =>
      prev.map((project) => ({ ...project, sessions: project.sessions.filter((item) => item.id !== session.id) }))
    );
    if (sessionId === session.id) {
      setSessionId("");
      setSessionTitle("");
      setMessages([]);
      setEvents([]);
      setResult(null);
    }
    setNotice(data.deleted ? "会话已删除" : "会话不存在");
  }

  async function deleteHistoryProject(project: HistoryProject) {
    if (!window.confirm(`删除项目「${project.name}」的全部 memory 记录？不会删除真实项目目录。`)) return;
    const res = await fetch(`${API}/api/projects?workdir=${encodeURIComponent(project.workdir)}`, { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) {
      setNotice(`删除项目失败：${data.detail}`);
      return;
    }
    setHistoryProjects((prev) => prev.filter((item) => item.id !== project.id));
    if (workdir === project.workdir) {
      setSessionId("");
      setSessionTitle("");
      setMessages([]);
      setEvents([]);
      setResult(null);
    }
    setNotice(data.deleted ? "项目 memory 已删除" : "项目 memory 不存在");
  }

  async function copyMessage(content: string) {
    try {
      await navigator.clipboard.writeText(content);
      setNotice("消息已复制");
    } catch (error) {
      setNotice(`复制失败：${String(error)}`);
    }
  }

  async function runTask(executePlan = false) {
    const trimmedTask = task.trim();
    if (!trimmedTask) {
      setNotice("请输入任务内容");
      return;
    }
    if (!workdir && !sessionId) {
      setNotice("请先选择项目目录或恢复历史会话");
      return;
    }
    const id = sessionId || (await createSessionAndReturnId());
    setRunning(true);
    setResult(null);
    setMessages((prev) => [...prev, { id: `${Date.now()}-user`, role: "user", content: trimmedTask }]);
    setTask("");
    setNotice("任务执行中");
    const res = await fetch(`${API}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: id, text: trimmedTask, plan_mode: planMode, execute_plan: executePlan })
    });
    const reader = res.body?.getReader();
    if (!reader) {
      setRunning(false);
      setNotice("浏览器不支持流式读取");
      return;
    }
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const item = JSON.parse(line);
        if (item.type === "event") {
          setEvents((prev) => [...prev, item.data]);
        }
        if (item.type === "result") {
          setResult(item.data);
          setMessages((prev) => [...prev, { id: `${Date.now()}-agent`, role: "agent", content: formatResultMessage(item.data) }]);
          setNotice(item.data.ok ? "任务完成" : "任务完成但验证失败");
        }
        if (item.type === "error") {
          setMessages((prev) => [...prev, { id: `${Date.now()}-error`, role: "agent", content: `执行失败：${item.data.msg}` }]);
          setNotice(`执行失败：${item.data.msg}`);
        }
      }
    }
    setRunning(false);
  }

  async function createSessionAndReturnId() {
    if (!workdir) {
      throw new Error("请先选择项目目录");
    }
    const res = await fetch(`${API}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workdir })
    });
    const data = await res.json();
    setSessionId(data.id);
    setSessionTitle(data.title);
    return data.id as string;
  }

  return (
    <main className="shell">
      <section className="grid">
        <aside className="panel side">
          <SectionTitle icon={<Settings size={18} />} text="模型配置" />
          <div className="model-list">
            {models.map((model) => (
              <div className="line" key={model.id}>
                <button
                  className="ghost text-left"
                  onClick={() => {
                    setDraft(model);
                    setModelModalOpen(true);
                  }}
                  title="编辑模型"
                >
                  <Cpu size={16} />
                  <span>{model.name}</span>
                </button>
                <button className="icon" onClick={() => testModel(model.id)} title="测试连通性">
                  <RadioTower size={16} />
                </button>
                <button className="icon danger" onClick={() => deleteModel(model.id)} title="删除模型">
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
            <button
              className="ghost"
              onClick={() => {
                setDraft(emptyModel());
                setModelModalOpen(true);
              }}
            >
              <Plus size={16} /> 新增模型
            </button>
          </div>

          <SectionTitle icon={<Bot size={18} />} text="智能体模型" />
          <div className="map-list">
            {agents.map((agent) => (
              <label key={agent.id}>
                <span>{agent.name}</span>
                <select
                  value={modelMap[agent.id]}
                  onChange={(e) => void saveMap({ ...modelMap, [agent.id]: e.target.value })}
                >
                  {models.map((model) => (
                    <option value={model.id} key={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
        </aside>

        <section className="panel chat-panel">
          <div className="chat-window">
            {messages.length === 0 && (
              <div className="empty-chat">
                <FileText size={28} />
                <p>这里会展示你和 agent 的多轮对话。</p>
              </div>
            )}
            {messages.map((msg) => (
              <div className={`message-row ${msg.role}`} key={msg.id}>
                <article className={`bubble ${msg.role}`}>
                  <span>{msg.role === "user" ? "用户" : "Agent"}</span>
                  <p>{msg.content}</p>
                </article>
                <button className="copy-message" type="button" onClick={() => void copyMessage(msg.content)} title="复制消息">
                  <Copy size={12} />
                </button>
              </div>
            ))}
          </div>
        </section>

        <aside className="right-col">
          <section className="panel status">
            <div className="status-title">
              <SectionTitle icon={<Activity size={18} />} text="运行状态" />
              <div className="health">
                <Activity size={16} />
                <span>{notice}</span>
              </div>
            </div>
            <div className="metrics">
              <Metric label="当前智能体" value={agentName(activeAgent)} />
              <Metric label="事件数" value={String(events.length)} />
              <Metric label="Token 估算" value={String(tokenTotal)} />
              <button className="metric clickable" type="button" disabled={!sessionId} onClick={() => void renameCurrentSession()} title={sessionId ? `会话 ID：${sessionId}` : "未创建会话"}>
                <span>会话</span>
                <strong>{sessionId ? sessionTitle || sessionId : "未创建"}</strong>
              </button>
              <Metric label="项目" value={workdir ? shortPath(workdir) : "未选择"} />
            </div>
            <div className="agent-strip">
              {agents.map((agent) => (
                <span className={agent.id === activeAgent ? "agent active" : "agent"} key={agent.id}>
                  {agent.name}
                </span>
              ))}
            </div>
          </section>

          <section className="panel log">
            <SectionTitle icon={<RadioTower size={18} />} text="事件流" />
            <div className="events">
              {events.map((event) => (
                <article key={`${event.id}-${event.ts}`} className="event">
                  <div>
                    <span className="tag">{agentName(event.agent)}</span>
                    <span className="kind">{event.kind}</span>
                  </div>
                  <p>{event.msg}</p>
                </article>
              ))}
            </div>
          </section>
        </aside>

        <section className="composer">
          <textarea value={task} onChange={(e) => setTask(e.target.value)} placeholder="输入任务或问题..." />
          <div className="actions">
            <button className="secondary" type="button" onClick={() => void pickProject()}>
              <FolderOpen size={16} /> 选择项目
            </button>
            <button className="ghost" type="button" onClick={() => void openHistory()}>
              <History size={16} /> 历史会话
            </button>
            <button className="ghost" type="button" disabled={!workdir} onClick={() => void createSession()}>
              <GitBranch size={16} /> 新建会话
            </button>
            <button className="primary" disabled={running} onClick={() => void runTask(false)}>
              {running ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
              发送
            </button>
            <label className="toggle plan-toggle">
              <input type="checkbox" checked={planMode} onChange={(e) => setPlanMode(e.target.checked)} />
              <span>Plan 模式</span>
            </label>
          </div>
        </section>

        {modelModalOpen && (
          <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="模型配置表单">
            <form className="modal" onSubmit={saveModel}>
              <div className="modal-head">
                <h2>
                  <Settings size={18} />
                  模型配置
                </h2>
                <button className="icon" type="button" onClick={() => setModelModalOpen(false)} title="关闭">
                  <XCircle size={18} />
                </button>
              </div>
              <input value={draft.id} onChange={(e) => setDraft({ ...draft, id: e.target.value })} placeholder="id" />
              <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="显示名称" />
              <input value={draft.base_url} onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} placeholder="base url" />
              <input value={draft.model} onChange={(e) => setDraft({ ...draft, model: e.target.value })} placeholder="model name" />
              <input value={draft.api_key} onChange={(e) => setDraft({ ...draft, api_key: e.target.value })} placeholder="api key" type="password" />
              <div className="row">
                <input value={draft.ctx} onChange={(e) => setDraft({ ...draft, ctx: Number(e.target.value) })} placeholder="上下文窗口" type="number" />
                <input value={draft.timeout} onChange={(e) => setDraft({ ...draft, timeout: Number(e.target.value) })} placeholder="超时秒" type="number" />
              </div>
              <div className="modal-actions">
                {draft.id && (
                  <button className="secondary" type="button" onClick={() => void testModel(draft.id)}>
                    <RadioTower size={16} /> 测试连通性
                  </button>
                )}
                <button className="primary" type="submit">
                  <Save size={16} /> 保存模型
                </button>
              </div>
            </form>
          </div>
        )}

        {historyModalOpen && (
          <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="历史会话">
            <section className="modal history-modal">
              <div className="modal-head">
                <h2>
                  <History size={18} />
                  历史会话
                </h2>
                <button className="icon" type="button" onClick={() => setHistoryModalOpen(false)} title="关闭">
                  <XCircle size={18} />
                </button>
              </div>
              <div className="history-list">
                {historyProjects.length === 0 && <p className="muted">暂无历史项目。</p>}
                {historyProjects.map((project) => (
                  <section className="history-project" key={project.id}>
                    <div className="history-project-head">
                      <div>
                        <strong>{project.name}</strong>
                        <button className="icon danger compact-icon" type="button" onClick={() => void deleteHistoryProject(project)} title="删除项目 memory">
                          <Trash2 size={14} />
                        </button>
                      </div>
                      <span>{project.workdir}</span>
                    </div>
                    <div className="history-sessions">
                      {project.sessions.length === 0 && <p className="muted">这个项目还没有会话。</p>}
                      {project.sessions.map((session) => (
                        <div className="session-row" key={session.id}>
                          <button className="session-item" onClick={() => restoreSession(session)} title={`会话 ID：${session.id}`}>
                            <span>{session.title || session.id}</span>
                            <small>
                              ID：{session.id} · {formatTime(session.updated_at)} · {session.messages.length} 条消息
                              {session.interrupted ? " · 上次中断" : ""}
                            </small>
                          </button>
                          <button className="icon danger compact-icon" type="button" onClick={() => void deleteHistorySession(session)} title="删除会话 memory">
                            <Trash2 size={14} />
                          </button>
                        </div>
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            </section>
          </div>
        )}
      </section>
    </main>
  );
}

function SectionTitle({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <h2>
      {icon}
      {text}
    </h2>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function emptyModel(): ModelConfig {
  return {
    id: "",
    name: "",
    base_url: "",
    api_key: "",
    model: "",
    ctx: 128000,
    enabled: true,
    timeout: 120
  };
}

function agentName(id: string) {
  return agents.find((agent) => agent.id === id)?.name ?? id;
}

function shortPath(path: string) {
  const parts = path.split("/");
  return parts.slice(-2).join("/") || path;
}

function formatTime(value: string) {
  if (!value) return "未知时间";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function formatResultMessage(result: TaskResult) {
  const lines = [result.summary];
  if (result.files.length > 0) {
    lines.push(`文件变更：${result.files.join("、")}`);
  }
  if (result.tests.length > 0) {
    const testSummary = result.tests
      .map((item) => `${String(item.cmd ?? "检查")}：${item.ok ? "通过" : "失败"}`)
      .join("；");
    lines.push(`验证结果：${testSummary}`);
  }
  if (result.doc_path) {
    lines.push(`文档：${result.doc_path}`);
  }
  if (result.plan_path) {
    lines.push(`计划：${result.plan_path}`);
  }
  return lines.join("\n");
}

createRoot(document.getElementById("root")!).render(<App />);
