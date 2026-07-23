import React, { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  Check,
  Circle,
  Copy,
  Cpu,
  FileText,
  FolderOpen,
  GitBranch,
  History,
  Loader2,
  Plus,
  RadioTower,
  Save,
  Send,
  Settings,
  Trash2,
  XCircle
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  AgentEvent,
  AgentModelMap,
  AgentName,
  HistoryMessage,
  HistoryProject,
  HistorySession,
  ModelConfig,
  TaskResult
} from "../../api/types";
import "./style.css";

// ChatMessage 与后端历史消息合同一致，单独起别名用于组件状态语义。
type ChatMessage = HistoryMessage;

// API 是后端地址，启动脚本会通过 VITE_API_URL 注入动态端口。
const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8710";
// agents 是左侧“智能体模型”和右侧状态条共享的智能体展示列表。
const agents: Array<{ id: AgentName; name: string }> = [
  { id: "manager", name: "管理者" },
  { id: "planner", name: "Plan" },
  { id: "repo", name: "仓库读取" },
  { id: "coder", name: "Coding" },
  { id: "verifier", name: "验证测试" },
  { id: "doc", name: "文档" }
];

function App() {
  // models 保存后端返回的所有模型配置。
  const [models, setModels] = useState<ModelConfig[]>([]);
  // modelMap 保存每个智能体当前绑定的模型 id。
  const [modelMap, setModelMap] = useState<AgentModelMap>({
    manager: "longcat",
    planner: "longcat",
    repo: "longcat",
    coder: "longcat",
    verifier: "longcat",
    doc: "longcat"
  });
  // draft 是模型弹窗中正在编辑的模型配置草稿。
  const [draft, setDraft] = useState<ModelConfig>(emptyModel());
  // workdir 是当前选中的项目目录。
  const [workdir, setWorkdir] = useState("");
  // sessionId 是当前会话 id。
  const [sessionId, setSessionId] = useState("");
  // sessionTitle 是当前会话展示名称。
  const [sessionTitle, setSessionTitle] = useState("");
  // task 是底部输入框内容。
  const [task, setTask] = useState("");
  // planMode 表示 Plan 模式复选框是否开启。
  const [planMode, setPlanMode] = useState(false);
  // running 表示当前是否正在执行任务，用于禁用发送按钮和显示 loading。
  const [running, setRunning] = useState(false);
  // events 是右侧事件流数据。
  const [events, setEvents] = useState<AgentEvent[]>([]);
  // result 保存最近一次任务最终结果。
  const [result, setResult] = useState<TaskResult | null>(null);
  // notice 是右上角运行状态旁边的小提示文本。
  const [notice, setNotice] = useState("等待会话启动");
  // messages 是中间对话窗口展示的用户消息和 agent 最终回复。
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // modelModalOpen 控制新增/编辑模型弹窗是否显示。
  const [modelModalOpen, setModelModalOpen] = useState(false);
  // editingModelId 非空表示正在编辑已有模型，此时唯一 id 不允许改名。
  const [editingModelId, setEditingModelId] = useState("");
  // historyModalOpen 控制历史会话弹窗是否显示。
  const [historyModalOpen, setHistoryModalOpen] = useState(false);
  // historyProjects 保存历史弹窗中的项目和会话树。
  const [historyProjects, setHistoryProjects] = useState<HistoryProject[]>([]);
  // copiedId 保存刚复制成功的消息 id，用于短暂显示勾选反馈。
  const [copiedId, setCopiedId] = useState("");
  // chatRef 指向对话滚动容器，新消息到达后自动滚到最新位置。
  const chatRef = useRef<HTMLDivElement>(null);
  // eventRef 指向事件流滚动容器，新事件到达后自动滚到底部。
  const eventRef = useRef<HTMLDivElement>(null);

  // 组件首次挂载时加载模型配置。
  useEffect(() => {
    void loadModels();
  }, []);

  // 只在消息数量变化时滚动，用户阅读长消息内部滚动条时不会被持续抢焦点。
  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length]);

  // 事件流始终跟随最新运行事件，便于观察当前智能体。
  useEffect(() => {
    eventRef.current?.scrollTo({ top: eventRef.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);

  // tokenTotal 是右侧运行状态展示的累计 token 估算。
  const tokenTotal = useMemo(() => events.reduce((sum, event) => sum + event.tokens, 0), [events]);
  // activeAgent 是最后一个产生事件的智能体，没有事件时默认 manager。
  const activeAgent = events.at(-1)?.agent ?? "manager";

  async function loadModels() {
    try {
      // data 包含 models 和 agent_models，requestJson 会统一处理非 2xx 和断网错误。
      const data = await requestJson<{ models: ModelConfig[]; agent_models: AgentModelMap }>(`${API}/api/models`);
      setModels(data.models);
      setModelMap(data.agent_models);
      setDraft(data.models[0] ?? emptyModel());
    } catch (error) {
      setNotice(errorMessage(error));
    }
  }

  async function saveModel(event: FormEvent) {
    // 阻止表单默认刷新页面。
    event.preventDefault();
    try {
      await requestJson<ModelConfig>(`${API}/api/models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft)
      });
      setNotice(`模型已保存：${draft.name}`);
      setModelModalOpen(false);
      setEditingModelId("");
      await loadModels();
    } catch (error) {
      setNotice(errorMessage(error));
    }
  }

  async function deleteModel(id: string) {
    // id 是待删除模型的本地配置 id。
    if (!window.confirm(`删除模型「${id}」？`)) return;
    try {
      await requestJson(`${API}/api/models/${encodeURIComponent(id)}`, { method: "DELETE" });
      setNotice(`模型已删除：${id}`);
      await loadModels();
    } catch (error) {
      setNotice(errorMessage(error));
    }
  }

  async function testModel(id: string) {
    // id 是要测试连通性的模型 id。
    setNotice("正在测试模型连通性");
    try {
      // data 包含模型回复和 token，仅把简短结果放入状态条。
      const data = await requestJson<{ ok: boolean; reply: string }>(`${API}/api/models/${encodeURIComponent(id)}/test`, { method: "POST" });
      setNotice(data.ok ? `连通成功：${data.reply}` : "模型返回异常");
    } catch (error) {
      setNotice(errorMessage(error));
    }
  }

  async function saveMap(next: AgentModelMap) {
    // next 是用户刚选择后的完整智能体模型映射。
    // previous 用于接口失败时回滚下拉框，避免界面和磁盘配置不一致。
    const previous = modelMap;
    setModelMap(next);
    try {
      await requestJson<AgentModelMap>(`${API}/api/model-map`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(next)
      });
      setNotice("智能体模型映射已保存");
    } catch (error) {
      setModelMap(previous);
      setNotice(errorMessage(error));
    }
  }

  async function createSession() {
    // 没有选择项目时不允许新建会话，因为后端需要 workdir 创建 memory。
    if (!workdir) {
      setNotice("请先选择项目目录");
      return;
    }
    try {
      // data 是后端创建出的 SessionInfo。
      const data = await requestJson<{ id: string; title: string }>(`${API}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workdir })
      });
      setSessionId(data.id);
      setSessionTitle(data.title);
      setEvents([]);
      setResult(null);
      setMessages([]);
      setNotice(`会话已创建：${data.title}`);
    } catch (error) {
      setNotice(errorMessage(error));
    }
  }

  async function pickProject() {
    // 这里不手动输入路径，而是调用后端拉起系统目录选择器。
    setNotice("正在打开目录选择器");
    try {
      // data 成功时包含 workdir，requestJson 会把失败 detail 转成异常。
      const data = await requestJson<{ workdir: string }>(`${API}/api/pick-dir`, { method: "POST" });
      // 切换项目后清空当前会话和对话状态，避免不同项目的消息混在一起。
      setWorkdir(data.workdir);
      setSessionId("");
      setSessionTitle("");
      setEvents([]);
      setResult(null);
      setMessages([]);
      setNotice(`已选择项目：${data.workdir}`);
    } catch (error) {
      setNotice(`选择目录失败：${errorMessage(error)}`);
    }
  }

  async function openHistory() {
    // 打开历史会话弹窗前先从后端读取最新 memory 列表。
    setNotice("正在读取历史会话");
    try {
      // data 包含 projects 数组。
      const data = await requestJson<{ projects: HistoryProject[] }>(`${API}/api/history`);
      setHistoryProjects(data.projects ?? []);
      setHistoryModalOpen(true);
      setNotice("历史会话已加载");
    } catch (error) {
      setNotice(`读取历史失败：${errorMessage(error)}`);
    }
  }

  function restoreSession(session: HistorySession) {
    // 恢复会话必须同时恢复 workdir 和 sessionId，否则后端无法定位会话项目。
    setWorkdir(session.workdir);
    setSessionId(session.id);
    setSessionTitle(session.title);
    setMessages(session.messages);
    // 历史接口携带最近一次运行的事件摘要，恢复后右侧事件流和 token 统计同步还原。
    setEvents(session.events ?? []);
    setResult(null);
    setHistoryModalOpen(false);
    setNotice(`已恢复会话：${session.title || session.id}`);
  }

  async function renameCurrentSession() {
    // 当前没有会话时，不允许重命名。
    if (!sessionId || !workdir) {
      setNotice("当前没有可重命名的会话");
      return;
    }
    // nextTitle 是用户在浏览器 prompt 中输入的新会话名。
    const nextTitle = window.prompt("请输入新的会话名称", sessionTitle || sessionId);
    if (nextTitle === null) return;
    // cleanTitle 去掉首尾空白，避免保存空名字。
    const cleanTitle = nextTitle.trim();
    if (!cleanTitle) {
      setNotice("会话名称不能为空");
      return;
    }
    try {
      const data = await requestJson<{ title: string }>(`${API}/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workdir, title: cleanTitle })
      });
      setSessionTitle(data.title);
      setNotice(`会话已重命名：${data.title}`);
    } catch (error) {
      setNotice(`重命名失败：${errorMessage(error)}`);
    }
  }

  async function deleteHistorySession(session: HistorySession) {
    // 删除会话前二次确认，明确只删除 agent memory，不删除项目文件。
    if (!window.confirm(`删除会话「${session.title}」？只会删除 agent memory 中的会话记录。`)) return;
    try {
      // data.deleted 表示后端是否真的删除了文件。
      const data = await requestJson<{ deleted: boolean }>(
        `${API}/api/sessions/${encodeURIComponent(session.id)}?workdir=${encodeURIComponent(session.workdir)}`,
        { method: "DELETE" }
      );
      // 本地同步移除历史弹窗里的会话项，避免删除后还显示旧数据。
      setHistoryProjects((prev) =>
        prev.map((project) => ({ ...project, sessions: project.sessions.filter((item) => item.id !== session.id) }))
      );
      // 如果删除的是当前正在查看的会话，清空当前对话状态。
      if (sessionId === session.id) {
        setSessionId("");
        setSessionTitle("");
        setMessages([]);
        setEvents([]);
        setResult(null);
      }
      setNotice(data.deleted ? "会话已删除" : "会话不存在");
    } catch (error) {
      setNotice(`删除会话失败：${errorMessage(error)}`);
    }
  }

  async function deleteHistoryProject(project: HistoryProject) {
    // 删除项目 memory 前二次确认，强调不会删除真实项目目录。
    if (!window.confirm(`删除项目「${project.name}」的全部 memory 记录？不会删除真实项目目录。`)) return;
    try {
      // data.deleted 表示后端是否真的删除了 memory 目录。
      const data = await requestJson<{ deleted: boolean }>(`${API}/api/projects?workdir=${encodeURIComponent(project.workdir)}`, {
        method: "DELETE"
      });
      // 本地同步移除项目项。
      setHistoryProjects((prev) => prev.filter((item) => item.id !== project.id));
      // 如果删除的是当前项目 memory，清空当前会话展示状态。
      if (workdir === project.workdir) {
        setSessionId("");
        setSessionTitle("");
        setMessages([]);
        setEvents([]);
        setResult(null);
      }
      setNotice(data.deleted ? "项目 memory 已删除" : "项目 memory 不存在");
    } catch (error) {
      setNotice(`删除项目失败：${errorMessage(error)}`);
    }
  }

  async function copyMessage(id: string, content: string) {
    // content 是当前消息正文，复制按钮位于消息框外部右下角。
    try {
      await navigator.clipboard.writeText(content);
      setNotice("消息已复制");
      setCopiedId(id);
      window.setTimeout(() => setCopiedId((current) => (current === id ? "" : current)), 1400);
    } catch (error) {
      setNotice(`复制失败：${String(error)}`);
    }
  }

  async function runTask() {
    // trimmedTask 是去掉首尾空白后的用户输入。
    const trimmedTask = task.trim();
    if (!trimmedTask) {
      setNotice("请输入任务内容");
      return;
    }
    if (!workdir) {
      setNotice("请先选择项目目录或恢复历史会话");
      return;
    }
    setRunning(true);
    setResult(null);
    try {
      // id 是本次请求使用的会话 id；没有当前会话时自动创建。
      const id = sessionId || (await createSessionAndReturnId());
      // 先把用户消息加入对话窗口，让界面立即响应。
      setMessages((prev) => [...prev, { id: `${Date.now()}-user`, role: "user", content: trimmedTask }]);
      setTask("");
      setNotice("任务执行中");
      // res 是后端 NDJSON 流式接口响应。
      const res = await fetch(`${API}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: id, text: trimmedTask, plan_mode: planMode, execute_plan: false })
      });
      if (!res.ok) {
        // errorData 可能是 FastAPI detail，也可能是 Pydantic 校验错误数组。
        const errorData = await res.json().catch(() => ({}));
        throw new Error(detailText(errorData.detail) || `请求失败：${res.status}`);
      }
      // reader 用于逐块读取响应体。
      const reader = res.body?.getReader();
      if (!reader) throw new Error("浏览器不支持流式读取");
      // decoder 把 Uint8Array 数据块解码成字符串。
      const decoder = new TextDecoder();
      // buf 保存未读完整的一行 NDJSON，防止数据块刚好从中间断开。
      let buf = "";
      // terminalReceived 表示已经收到 result 或 error 终态。
      let terminalReceived = false;

      const consumeLine = (line: string) => {
        if (!line.trim()) return;
        // item 是后端单行 NDJSON 解析后的对象，heartbeat 不进入界面状态。
        const item = JSON.parse(line) as { type: string; data?: AgentEvent | TaskResult | Record<string, unknown> };
        if (item.type === "event") {
          setEvents((prev) => [...prev, item.data as AgentEvent]);
        } else if (item.type === "result") {
          const taskResult = item.data as TaskResult;
          setResult(taskResult);
          setMessages((prev) => [...prev, { id: `${Date.now()}-agent`, role: "agent", content: formatResultMessage(taskResult) }]);
          setNotice(taskResult.ok ? "任务完成" : "任务未完成");
          terminalReceived = true;
        } else if (item.type === "error") {
          const errorData = item.data as { msg?: string; result?: TaskResult };
          const content = errorData.result ? formatResultMessage(errorData.result) : `执行失败：${errorData.msg ?? "未知错误"}`;
          if (errorData.result) setResult(errorData.result);
          setMessages((prev) => [...prev, { id: `${Date.now()}-error`, role: "agent", content }]);
          setNotice(`执行失败：${errorData.msg ?? "未知错误"}`);
          terminalReceived = true;
        }
      };

      while (true) {
        // done 表示流结束，value 是本次读取的数据块。
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // lines 是当前已经完整读取到的多行 NDJSON。
        const lines = buf.split("\n");
        // 最后一段可能是不完整 JSON，留到下一次数据块再解析。
        buf = lines.pop() ?? "";
        lines.forEach(consumeLine);
      }
      // flushText 包含 TextDecoder 最后尚未输出的字符，和 buf 一起做最终解析。
      const flushText = (buf + decoder.decode()).trim();
      if (flushText) consumeLine(flushText);
      if (!terminalReceived) throw new Error("任务流提前结束，没有收到最终结果");
    } catch (error) {
      const message = errorMessage(error);
      setMessages((prev) => [...prev, { id: `${Date.now()}-client-error`, role: "agent", content: message }]);
      setNotice(message);
    } finally {
      setRunning(false);
    }
  }

  async function createSessionAndReturnId() {
    // 自动创建会话也必须依赖 workdir，否则后端无法建立项目 memory。
    if (!workdir) {
      throw new Error("请先选择项目目录");
    }
    // data 是创建会话接口响应。
    const data = await requestJson<{ id: string; title: string }>(`${API}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workdir })
    });
    setSessionId(data.id);
    setSessionTitle(data.title);
    return data.id as string;
  }

  return (
    <main className="shell">
      <section className="grid">
        {/* 左侧栏：模型配置和每个智能体的模型选择。 */}
        <aside className="panel side">
          <div className="side-section">
            <SectionTitle icon={<Settings size={18} />} text="模型配置" />
            <span className="section-count">{models.length}</span>
          </div>
          <div className="model-list">
            {models.length === 0 && <p className="muted empty-models">尚未配置模型</p>}
            {models.map((model) => (
              <div className="line" key={model.id}>
                <button
                  className="ghost text-left"
                  onClick={() => {
                    setDraft(model);
                    setEditingModelId(model.id);
                    setModelModalOpen(true);
                  }}
                  title="编辑模型"
                >
                  <Cpu size={16} />
                  <span>{model.name}</span>
                </button>
                <button className="icon" onClick={() => testModel(model.id)} title="测试连通性" disabled={running}>
                  <RadioTower size={16} />
                </button>
                <button className="icon danger" onClick={() => deleteModel(model.id)} title="删除模型" disabled={running}>
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
            <button
              className="ghost"
              onClick={() => {
                setDraft(emptyModel());
                setEditingModelId("");
                setModelModalOpen(true);
              }}
            >
              <Plus size={16} /> 新增模型
            </button>
          </div>

          <div className="side-section agent-map-title">
            <SectionTitle icon={<Bot size={18} />} text="智能体模型" />
            <span className="section-count">{agents.length}</span>
          </div>
          <div className="map-list">
            {agents.map((agent) => (
              <label key={agent.id}>
                <span>{agent.name}</span>
                <select
                  value={modelMap[agent.id]}
                  onChange={(e) => void saveMap({ ...modelMap, [agent.id]: e.target.value })}
                  disabled={models.length === 0 || running}
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

        {/* 中间栏：只展示用户消息和 agent 最终回复，不展示全部事件。 */}
        <section className="panel chat-panel">
          <div className="chat-window" ref={chatRef} aria-live="polite">
            {messages.length === 0 && (
              <div className="empty-chat">
                <FileText size={28} />
                <p>这里会展示你和 agent 的多轮对话。</p>
              </div>
            )}
            {messages.map((msg) => (
              <div className={`message-row ${msg.role}`} key={msg.id}>
                <article className={`bubble ${msg.role}`}>
                  <span className="message-role">{msg.role === "user" ? "用户" : "Agent"}</span>
                  <div className="message-content">
                    {msg.role === "agent" ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  </div>
                </article>
                <button className="copy-message" type="button" onClick={() => void copyMessage(msg.id, msg.content)} title="复制消息">
                  {copiedId === msg.id ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* 右侧栏：运行状态和事件流。 */}
        <aside className="right-col">
          <section className="panel status">
            <div className="status-title">
              <SectionTitle icon={<Activity size={18} />} text="运行状态" />
              <div className="health">
                {running ? <Loader2 className="spin" size={15} /> : <Circle className="status-dot" size={10} fill="currentColor" />}
                <span>{notice}</span>
              </div>
            </div>
            <div className="metrics">
              <Metric label="当前智能体" value={agentName(activeAgent)} />
              <Metric label="事件数" value={String(events.length)} />
              <Metric label="Token 估算" value={String(tokenTotal || result?.tokens || 0)} />
              <button className="metric clickable" type="button" disabled={!sessionId} onClick={() => void renameCurrentSession()} title={sessionId ? `会话 ID：${sessionId}` : "未创建会话"}>
                <span>会话</span>
                <strong>{sessionId ? sessionTitle || sessionId : "未创建"}</strong>
              </button>
              <Metric label="项目" value={workdir ? shortPath(workdir) : "未选择"} />
              <Metric label="耗时" value={result ? formatDuration(result.duration_ms) : "--"} />
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
            <div className="events" ref={eventRef}>
              {events.length === 0 && <p className="muted empty-events">任务事件将在这里按时间出现</p>}
              {events.map((event) => (
                <article key={`${event.id}-${event.ts}`} className="event">
                  <div>
                    <span className="event-index">{String(event.id).padStart(2, "0")}</span>
                    <span className="tag">{agentName(event.agent)}</span>
                    <span className="kind">{event.kind}</span>
                    <time>{formatEventTime(event.ts)}</time>
                  </div>
                  <p>{event.msg}</p>
                </article>
              ))}
            </div>
          </section>
        </aside>

        {/* 底部输入区：选择项目、历史会话、新建会话、发送和 Plan 模式。 */}
        <section className="composer">
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            onKeyDown={(event) => handleComposerKey(event, running, () => void runTask())}
            placeholder="输入任务或问题..."
            disabled={running}
          />
          <div className="actions">
            <button className="secondary" type="button" onClick={() => void pickProject()} disabled={running} title={workdir || "选择项目目录"}>
              <FolderOpen size={16} /> 选择项目
            </button>
            <button className="ghost" type="button" onClick={() => void openHistory()} disabled={running}>
              <History size={16} /> 历史会话
            </button>
            <button className="ghost" type="button" disabled={!workdir || running} onClick={() => void createSession()}>
              <GitBranch size={16} /> 新建会话
            </button>
            <button className="primary send-button" disabled={running || !task.trim() || !workdir} onClick={() => void runTask()}>
              {running ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              发送
            </button>
            <label className="toggle plan-toggle">
              <input type="checkbox" checked={planMode} onChange={(e) => setPlanMode(e.target.checked)} disabled={running} />
              <span>Plan 模式</span>
            </label>
          </div>
        </section>

        {/* 模型新增/编辑弹窗。 */}
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
              <label>
                <span>配置 ID</span>
                <input value={draft.id} onChange={(e) => setDraft({ ...draft, id: e.target.value })} placeholder="例如 longcat" disabled={Boolean(editingModelId)} />
              </label>
              <label>
                <span>显示名称</span>
                <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="模型显示名称" />
              </label>
              <label>
                <span>接口地址</span>
                <input value={draft.base_url} onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} placeholder="https://..." />
              </label>
              <label>
                <span>模型名称</span>
                <input value={draft.model} onChange={(e) => setDraft({ ...draft, model: e.target.value })} placeholder="供应商模型名称" />
              </label>
              <label>
                <span>API Key</span>
                <input value={draft.api_key} onChange={(e) => setDraft({ ...draft, api_key: e.target.value })} placeholder="API Key" type="password" />
              </label>
              <div className="row">
                <label>
                  <span>上下文窗口</span>
                  <input value={draft.ctx} onChange={(e) => setDraft({ ...draft, ctx: Number(e.target.value) })} type="number" />
                </label>
                <label>
                  <span>超时秒数</span>
                  <input value={draft.timeout} onChange={(e) => setDraft({ ...draft, timeout: Number(e.target.value) })} type="number" />
                </label>
              </div>
              <label className="toggle model-enabled">
                <input type="checkbox" checked={draft.enabled} onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })} />
                <span>启用此模型</span>
              </label>
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

        {/* 历史会话弹窗，支持恢复、删除会话和删除项目 memory。 */}
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
  // SectionTitle 是侧栏、状态栏、事件流复用的小标题组件。
  return (
    <h2>
      {icon}
      {text}
    </h2>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  // Metric 是运行状态中的单个指标卡片。
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function emptyModel(): ModelConfig {
  // emptyModel 返回新增模型弹窗的默认空配置。
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
  // agentName 把后端事件中的 agent id 转成中文显示名。
  return agents.find((agent) => agent.id === id)?.name ?? id;
}

function shortPath(path: string) {
  // shortPath 只展示路径最后两段，避免状态卡片被长路径撑开。
  const parts = path.split("/");
  return parts.slice(-2).join("/") || path;
}

function formatTime(value: string) {
  // value 为空时显示未知时间，避免 Date 解析空字符串产生 Invalid Date。
  if (!value) return "未知时间";
  // 使用中文 24 小时制显示历史会话更新时间。
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function formatEventTime(value: string) {
  // 事件流只需要时分秒，完整日期保留在会话历史中。
  if (!value) return "--:--:--";
  return new Date(value).toLocaleTimeString("zh-CN", { hour12: false });
}

function formatDuration(value: number) {
  // 小于一秒显示毫秒，较长任务显示一位小数秒。
  if (!Number.isFinite(value) || value < 0) return "--";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`;
}

function formatResultMessage(result: TaskResult) {
  // lines 保存最终展示为 Agent 消息的多行文本。
  const lines = [result.summary];
  if (result.files.length > 0) {
    lines.push(`文件变更：${result.files.join("、")}`);
  }
  if (result.tests.length > 0) {
    // testSummary 把多个测试结果压缩成一行中文摘要。
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

function handleComposerKey(event: KeyboardEvent<HTMLTextAreaElement>, running: boolean, send: () => void) {
  // Cmd/Ctrl + Enter 是开发工具中常见的发送方式，普通 Enter 仍然输入换行。
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    if (!running) send();
  }
}

async function requestJson<T = Record<string, unknown>>(url: string, init?: RequestInit): Promise<T> {
  // response 是统一 API 请求结果；所有普通 JSON 接口都通过这里处理错误。
  const response = await fetch(url, init);
  // data 在后端无 JSON 响应时回退为空对象，避免二次解析异常掩盖原始状态码。
  const data = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (!response.ok) {
    throw new Error(detailText(data.detail) || `请求失败：${response.status}`);
  }
  return data as T;
}

function detailText(detail: unknown): string {
  // FastAPI 的 detail 可能是字符串，也可能是 Pydantic 校验错误数组。
  if (typeof detail === "string") return detail === "Not Found" ? "后端接口不存在，请重启项目" : detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (typeof item === "object" && item && "msg" in item ? String((item as { msg: unknown }).msg) : String(item)))
      .join("；");
  }
  return "";
}

function errorMessage(error: unknown): string {
  // Error 保留后端 detail；其他异常统一转成可读字符串。
  return error instanceof Error ? error.message : String(error);
}

createRoot(document.getElementById("root")!).render(<App />);
