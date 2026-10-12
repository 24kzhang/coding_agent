import React, { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Brain,
  Bot,
  Check,
  Circle,
  Copy,
  Cpu,
  ExternalLink,
  FileText,
  FolderOpen,
  GitBranch,
  History,
  ImagePlus,
  KeyRound,
  Loader2,
  PlugZap,
  Plus,
  Puzzle,
  RadioTower,
  RefreshCw,
  Save,
  Send,
  Settings,
  Trash2,
  X,
  XCircle
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  AgentEvent,
  AgentProfileMap,
  AgentModelMap,
  AgentName,
  CapabilitiesResponse,
  GuidanceResponse,
  HistoryMessage,
  HistoryProject,
  HistorySession,
  McpAuthStatus,
  McpInfo,
  ModelConfig,
  ReasoningEffort,
  AttachmentInfo,
  SkillInfo,
  TaskResult
} from "../../api/types";
import "./style.css";

// ChatMessage 与后端历史消息合同一致，单独起别名用于组件状态语义。
type ChatMessage = HistoryMessage;

// McpAuthPrompt 为设备授权弹窗补充当前服务 id，便于隔离并发或已取消的轮询。
type McpAuthPrompt = McpAuthStatus & { id: string };

// ModelTestState 保存模型配置弹窗内的连通性反馈，避免结果只能在主页面状态栏观察。
type ModelTestState = {
  // status 区分尚未测试、请求中、测试成功和测试失败四种界面状态。
  status: "idle" | "testing" | "success" | "error";
  // message 是弹窗内展示的简短结果，错误信息也会保留在主页面右上角。
  message: string;
};

type PendingImage = {
  key: string;
  file: File;
  preview: string;
};

// API 是后端地址，启动脚本会通过 VITE_API_URL 注入动态端口。
const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8710";
// agents 是左侧“智能体模型”和右侧状态条共享的智能体展示列表。
const agents: Array<{ id: AgentName; name: string }> = [
  { id: "manager", name: "管理者" },
  { id: "planner", name: "Plan" },
  { id: "coder", name: "Coding" },
  { id: "verifier", name: "验证测试" },
  { id: "answer", name: "答疑" },
  { id: "doc", name: "文档" },
  { id: "compact", name: "上下文压缩" }
];
const effortOptions: Array<{ value: ReasoningEffort; label: string }> = [
  { value: "inherit", label: "继承" },
  { value: "none", label: "关闭" },
  { value: "minimal", label: "最低" },
  { value: "low", label: "低" },
  { value: "medium", label: "中" },
  { value: "high", label: "高" },
  { value: "max", label: "最高" }
];

function App() {
  // models 保存后端返回的所有模型配置。
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [profiles, setProfiles] = useState<AgentProfileMap>(() => emptyProfiles());
  const [runEffort, setRunEffort] = useState<ReasoningEffort>("inherit");
  const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
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
  // containerShell 表示 Coder 与 Verifier 是否在本轮共享 Docker 容器。
  const [containerShell, setContainerShell] = useState(true);
  // running 表示当前是否正在执行任务，用于禁用发送按钮和显示 loading。
  const [running, setRunning] = useState(false);
  // activeRunId 是后端当前运行编号；运行中发送框用它追加指导而不是启动第二个任务。
  const [activeRunId, setActiveRunId] = useState("");
  // runLockRef 防止快速重复点击在 React 完成重渲染前启动两次任务。
  const runLockRef = useRef(false);
  // guidanceLockRef 防止同一段指导被快速重复提交；瞬时互斥无需触发界面重渲染。
  const guidanceLockRef = useRef(false);
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
  // modelTestState 是模型配置弹窗内独立的连通性结果，不替代主页面 notice。
  const [modelTestState, setModelTestState] = useState<ModelTestState>({ status: "idle", message: "" });
  // historyModalOpen 控制历史会话弹窗是否显示。
  const [historyModalOpen, setHistoryModalOpen] = useState(false);
  // historyProjects 保存历史弹窗中的项目和会话树。
  const [historyProjects, setHistoryProjects] = useState<HistoryProject[]>([]);
  // skills 保存后端从项目根目录 skills 发现的可选 Skill 摘要。
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  // mcps 保存 MCP/servers.json 中已启用服务的安全摘要。
  const [mcps, setMcps] = useState<McpInfo[]>([]);
  // selectedSkills 是本轮请求要注入结构化上下文的 Skill id。
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  // selectedMcps 是本轮请求允许连接和调用的 MCP 服务 id。
  const [selectedMcps, setSelectedMcps] = useState<string[]>([]);
  // capabilityModalOpen 控制 Skills/MCP 多选弹窗。
  const [capabilityModalOpen, setCapabilityModalOpen] = useState(false);
  // mcpAuthPrompt 保存设备码、官方授权地址和当前状态，避免信息只出现在短提示中。
  const [mcpAuthPrompt, setMcpAuthPrompt] = useState<McpAuthPrompt | null>(null);
  // copiedId 保存刚复制成功的消息 id，用于短暂显示勾选反馈。
  const [copiedId, setCopiedId] = useState("");
  // authAttemptRef 标识当前授权轮询；关闭弹窗后旧轮询不会再次把弹窗打开。
  const authAttemptRef = useRef(0);
  // chatRef 指向对话滚动容器，新消息到达后自动滚到最新位置。
  const chatRef = useRef<HTMLDivElement>(null);
  // eventRef 指向事件流滚动容器，新事件到达后自动滚到底部。
  const eventRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // 组件首次挂载时加载模型配置。
  useEffect(() => {
    void loadModels();
    void loadCapabilities();
  }, []);

  // 只在消息数量变化时滚动，用户阅读长消息内部滚动条时不会被持续抢焦点。
  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length]);

  // 事件流始终跟随最新运行事件，便于观察当前智能体。
  useEffect(() => {
    eventRef.current?.scrollTo({ top: eventRef.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);

  // tokenTotal 是整个会话的累计用量；供应商实值与估算不能混为一谈。
  const tokenTotal = useMemo(() => events.reduce((sum, event) => sum + event.tokens, 0), [events]);
  const usage = useMemo(() => {
    const total = { input: 0, cached: 0, output: 0, reasoning: 0, reasoningKnown: 0, reasoningUnknown: 0, estimated: false };
    for (const event of events.filter((item) => item.kind === "usage")) {
      const data = event.data ?? {};
      total.input += typeof data.prompt_tokens === "number" ? data.prompt_tokens : 0;
      total.cached += typeof data.cached_input_tokens === "number" ? data.cached_input_tokens : 0;
      total.output += typeof data.output_tokens === "number" ? data.output_tokens : 0;
      if (typeof data.reasoning_tokens === "number" && (data.reasoning_tokens_known === true || data.reasoning_tokens > 0)) {
        total.reasoning += data.reasoning_tokens;
        total.reasoningKnown += 1;
      } else {
        total.reasoningUnknown += 1;
      }
      total.estimated ||= data.estimated === true;
    }
    return total;
  }, [events]);
  // capabilityCount 同时统计用户手动选择和后端常驻能力。
  const capabilityCount = useMemo(
    () =>
      selectedSkills.length
      + new Set([
        ...selectedMcps,
        ...mcps.filter((item) => item.always_on).map((item) => item.id)
      ]).size,
    [mcps, selectedMcps, selectedSkills]
  );
  // activeAgent 是最后一个产生事件的智能体，没有事件时默认 manager。
  const activeAgent = eventOwner(events.at(-1)) ?? "manager";
  const agentStats = useMemo(
    () =>
      Object.fromEntries(
        agents.map((agent) => {
          const own = events.filter((event) => eventOwner(event) === agent.id);
          const last = own.at(-1);
          const actualModel = own.filter((event) => typeof event.data?.model_id === "string" && event.data.model_id).at(-1)?.data?.model_id;
          const actualEffort = own.filter((event) => typeof event.data?.effort === "string" && event.data.effort).at(-1)?.data?.effort;
          const state = agent.id === activeAgent && running
            ? "运行中"
            : last
              ? last.kind === "error"
                ? "失败"
                : "已完成"
              : "等待";
          return [
            agent.id,
            {
              state,
              task: last?.msg ?? "尚未接收任务",
              tokens: own.reduce((sum, event) => sum + event.tokens, 0),
              modelId: typeof actualModel === "string" ? actualModel : "",
              effort: typeof actualEffort === "string" && effortOptions.some((option) => option.value === actualEffort) ? actualEffort as ReasoningEffort : ""
            }
          ];
        })
      ) as Record<AgentName, { state: string; task: string; tokens: number; modelId: string; effort: ReasoningEffort | "" }>,
    [activeAgent, events, running]
  );

  async function loadModels() {
    try {
      // data 包含 models 和 agent_models，requestJson 会统一处理非 2xx 和断网错误。
      const data = await requestJson<{ models: ModelConfig[]; agent_models: AgentModelMap; agent_profiles: AgentProfileMap }>(`${API}/api/models`);
      setModels(data.models);
      setProfiles(data.agent_profiles ?? emptyProfiles(data.agent_models));
      setDraft(data.models[0] ? { ...data.models[0], api_key: "" } : emptyModel());
    } catch (error) {
      setNotice(errorMessage(error));
    }
  }

  async function loadCapabilities() {
    try {
      // data 只包含 Skill 摘要和 MCP 安全字段，不包含完整 Skill 正文、env 或 headers。
      const data = await requestJson<CapabilitiesResponse>(`${API}/api/capabilities`);
      setSkills(data.skills ?? []);
      setMcps(data.mcps ?? []);
      // 配置文件删除能力后同步清理旧选择，避免发送时由后端拒绝未知 id。
      const skillIds = new Set((data.skills ?? []).map((item) => item.id));
      const mcpIds = new Set((data.mcps ?? []).map((item) => item.id));
      setSelectedSkills((current) => current.filter((id) => skillIds.has(id)));
      setSelectedMcps((current) => current.filter((id) => mcpIds.has(id)));
    } catch (error) {
      setNotice(`能力列表读取失败：${errorMessage(error)}`);
    }
  }

  async function testMcp(id: string) {
    // MCP 的 ${workspace} 占位符依赖当前项目，未选择项目时不允许测试。
    if (!workdir) {
      setNotice("请先选择项目目录，再测试 MCP");
      return;
    }
    setNotice(`正在测试 MCP：${id}`);
    try {
      const data = await requestJson<{ ok: boolean; msg: string }>(
        `${API}/api/mcps/${encodeURIComponent(id)}/test?workdir=${encodeURIComponent(workdir)}`,
        { method: "POST" }
      );
      setNotice(data.ok ? data.msg : `MCP 测试失败：${id}`);
    } catch (error) {
      setNotice(errorMessage(error));
    }
  }

  async function authorizeMcp(id: string) {
    // OAuth 必须绑定当前项目对应的持久 MCP 进程，未选目录时不能发起。
    if (!workdir) {
      setNotice("请先选择项目目录，再授权 MCP");
      return;
    }
    // attemptId 用于识别本次轮询，用户关闭弹窗或重新授权后立即废弃旧任务。
    const attemptId = ++authAttemptRef.current;
    setMcpAuthPrompt({
      id,
      status: "starting",
      msg: "正在生成 GitHub 设备码",
      url: null,
      code: null
    });
    setNotice(`正在准备 MCP 授权：${id}`);
    try {
      let state = await requestJson<McpAuthStatus>(
        `${API}/api/mcps/${encodeURIComponent(id)}/auth?workdir=${encodeURIComponent(workdir)}`,
        { method: "POST" }
      );
      if (authAttemptRef.current !== attemptId) return;
      setMcpAuthPrompt({ id, ...state });
      setNotice(state.msg);

      // GitHub 设备码最长等待十五分钟；轮询只读取后端状态，不重复启动授权流程。
      for (let attempt = 0; attempt < 450 && !["authorized", "error"].includes(state.status); attempt += 1) {
        await delay(2000);
        if (authAttemptRef.current !== attemptId) return;
        state = await requestJson<McpAuthStatus>(
          `${API}/api/mcps/${encodeURIComponent(id)}/auth?workdir=${encodeURIComponent(workdir)}`
        );
        setMcpAuthPrompt({ id, ...state });
        setNotice(state.msg);
      }
    } catch (error) {
      if (authAttemptRef.current !== attemptId) return;
      const msg = errorMessage(error);
      setMcpAuthPrompt({ id, status: "error", msg, url: null, code: null });
      setNotice(msg);
    }
  }

  function closeMcpAuth() {
    // 递增标识使等待中的旧轮询在下一次检查时退出，不会重新显示已关闭弹窗。
    authAttemptRef.current += 1;
    setMcpAuthPrompt(null);
  }

  async function copyMcpCode() {
    // code 是短期设备码，不包含持久 token；复制后仍不写入会话或事件流。
    if (!mcpAuthPrompt?.code) return;
    try {
      await navigator.clipboard.writeText(mcpAuthPrompt.code);
      setNotice("设备码已复制");
    } catch (error) {
      setNotice(`复制设备码失败：${errorMessage(error)}`);
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

  async function testModel(id: string, showInModal = false) {
    // id 是要测试连通性的模型 id。
    // showInModal 为 true 时同步更新弹窗反馈，左侧快捷测试仍只更新主页面状态。
    if (showInModal) {
      setModelTestState({ status: "testing", message: "正在连接模型接口..." });
    }
    setNotice("正在测试模型连通性");
    try {
      // data 包含模型回复和 token，仅把简短结果放入状态条。
      const data = await requestJson<{ ok: boolean; reply: string }>(`${API}/api/models/${encodeURIComponent(id)}/test`, { method: "POST" });
      const message = data.ok ? `连通成功：${data.reply}` : "模型返回异常";
      setNotice(message);
      if (showInModal) {
        setModelTestState({ status: data.ok ? "success" : "error", message });
      }
    } catch (error) {
      const message = errorMessage(error);
      setNotice(message);
      if (showInModal) {
        setModelTestState({ status: "error", message });
      }
    }
  }

  async function saveProfiles(next: AgentProfileMap) {
    const previous = profiles;
    setProfiles(next);
    try {
      await requestJson<AgentProfileMap>(`${API}/api/agent-profiles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(next)
      });
      setNotice("Agent 模型与思考等级已保存");
    } catch (error) {
      setProfiles(previous);
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
      setContainerShell(true);
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
      setContainerShell(true);
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
    // 历史接口携带整个会话的事件摘要，恢复后右侧事件流和 token 统计同步还原。
    setEvents(session.events ?? []);
    // 能力选择属于会话最近一次任务状态，恢复后下一轮默认延续相同选择。
    setSelectedSkills(session.skills ?? []);
    setSelectedMcps(session.mcps ?? []);
    setContainerShell(session.container_shell ?? true);
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

  async function sendGuidance() {
    // 本地条件先检查，避免运行编号尚未就绪时发起无效网络请求。
    const text = task.trim();
    if (!text) {
      setNotice("请输入指导内容");
      return;
    }
    if (!activeRunId) {
      setNotice("当前任务正在建立连接，请稍后发送指导");
      return;
    }
    if (guidanceLockRef.current) return;
    guidanceLockRef.current = true;
    setTask("");
    try {
      const response = await requestJson<GuidanceResponse>(`${API}/api/runs/${activeRunId}/guidance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      setMessages((prev) => [
        ...prev,
        { id: response.guidance_id, role: "user", content: text }
      ]);
      setNotice("指导已收到，将在下一个安全检查点应用");
    } catch (error) {
      setTask((current) => (current ? `${text}\n${current}` : text));
      setNotice(`指导发送失败：${errorMessage(error)}`);
    } finally {
      guidanceLockRef.current = false;
    }
  }

  async function submitComposer() {
    if (running) {
      await sendGuidance();
      return;
    }
    await runTask();
  }

  function queueImages(files: File[]) {
    const allowed = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);
    const valid = files.filter((file) => allowed.has(file.type) && file.size <= 20 * 1024 * 1024);
    if (valid.length !== files.length) {
      setNotice("仅支持单张不超过 20 MiB 的 PNG、JPEG、WebP 或 GIF");
    }
    setPendingImages((current) => {
      const remaining = Math.max(8 - current.length, 0);
      const added = valid.slice(0, remaining).map((file) => ({
        key: `${file.name}-${file.size}-${file.lastModified}-${crypto.randomUUID()}`,
        file,
        preview: URL.createObjectURL(file)
      }));
      if (valid.length > remaining) setNotice("每条消息最多添加 8 张图片");
      return [...current, ...added];
    });
  }

  function removeImage(key: string) {
    setPendingImages((current) => {
      const target = current.find((item) => item.key === key);
      if (target) URL.revokeObjectURL(target.preview);
      return current.filter((item) => item.key !== key);
    });
  }

  async function uploadImages(id: string): Promise<AttachmentInfo[]> {
    return Promise.all(
      pendingImages.map(async (item) => {
        const form = new FormData();
        form.append("file", item.file);
        return requestJson<AttachmentInfo>(`${API}/api/sessions/${encodeURIComponent(id)}/files`, {
          method: "POST",
          body: form
        });
      })
    );
  }

  async function runTask() {
    if (runLockRef.current) return;
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
    runLockRef.current = true;
    setRunning(true);
    setActiveRunId("");
    setResult(null);
    try {
      // id 是本次请求使用的会话 id；没有当前会话时自动创建。
      const id = sessionId || (await createSessionAndReturnId());
      setNotice(pendingImages.length ? "正在安全上传图片" : "任务执行中");
      const uploaded = await uploadImages(id);
      // 先把用户消息加入对话窗口，让界面立即响应。
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-user`,
          role: "user",
          content: trimmedTask + (uploaded.length ? `\n\n已附加 ${uploaded.length} 张图片。` : "")
        }
      ]);
      setTask("");
      pendingImages.forEach((item) => URL.revokeObjectURL(item.preview));
      setPendingImages([]);
      setNotice("任务执行中");
      // res 是后端 NDJSON 流式接口响应。
      const res = await fetch(`${API}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: id,
          text: trimmedTask,
          plan_mode: planMode,
          execute_plan: false,
          skills: selectedSkills,
          mcps: selectedMcps,
          container_shell: containerShell,
          attachments: uploaded.map((item) => item.id),
          effort: runEffort
        })
      });
      if (!res.ok) {
        // errorData 可能是 FastAPI detail，也可能是 Pydantic 校验错误数组。
        const errorData = await res.json().catch(() => ({}));
        throw new Error(detailText(errorData.detail) || `请求失败：${res.status}`);
      }
      // 运行编号通过允许跨域读取的响应头返回，后续输入会定向发送到这个任务。
      const runId = res.headers.get("X-Run-ID") ?? "";
      if (!runId) throw new Error("后端没有返回当前运行编号");
      setActiveRunId(runId);
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
          const event = item.data as AgentEvent;
          setEvents((prev) => [...prev, event]);
          if (event.kind === "guidance") setNotice(event.msg);
        } else if (item.type === "result") {
          const taskResult = item.data as TaskResult;
          setResult(taskResult);
          setMessages((prev) => [...prev, { id: `${Date.now()}-agent`, role: "agent", content: formatResultMessage(taskResult) }]);
          setNotice(taskResult.ok ? "任务完成" : "任务未完成");
          setActiveRunId("");
          terminalReceived = true;
        } else if (item.type === "error") {
          const errorData = item.data as { msg?: string; result?: TaskResult };
          const content = errorData.result ? formatResultMessage(errorData.result) : `执行失败：${errorData.msg ?? "未知错误"}`;
          if (errorData.result) setResult(errorData.result);
          setMessages((prev) => [...prev, { id: `${Date.now()}-error`, role: "agent", content }]);
          setNotice(`执行失败：${errorData.msg ?? "未知错误"}`);
          setActiveRunId("");
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
      runLockRef.current = false;
      setActiveRunId("");
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
                  className="ghost text-left model-name-button"
                  onClick={() => {
                    setDraft({ ...model, api_key: "" });
                    setEditingModelId(model.id);
                    setModelTestState({ status: "idle", message: "" });
                    setModelModalOpen(true);
                  }}
                  title="编辑模型"
                >
                  <Cpu size={14} />
                  <span>{model.name}</span>
                </button>
                <button className="icon model-action-button" onClick={() => testModel(model.id)} title="测试连通性" disabled={running}>
                  <RadioTower size={14} />
                </button>
                <button className="icon danger model-action-button" onClick={() => deleteModel(model.id)} title="删除模型" disabled={running}>
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            <button
              className="ghost"
              onClick={() => {
                setDraft(emptyModel());
                setEditingModelId("");
                setModelTestState({ status: "idle", message: "" });
                setModelModalOpen(true);
              }}
            >
              <Plus size={16} /> 新增模型
            </button>
          </div>

          <div className="side-section agent-map-title">
            <SectionTitle icon={<Bot size={18} />} text="Agent 配置" />
            <span className="section-count">{agents.length}</span>
          </div>
          <div className="map-list">
            {agents.map((agent) => (
              <div className="agent-setting" key={agent.id}>
                <span>{agent.name}</span>
                <select
                  aria-label={`${agent.name}模型`}
                  value={profiles[agent.id]?.model_id ?? ""}
                  onChange={(e) => void saveProfiles({
                    ...profiles,
                    [agent.id]: { ...profiles[agent.id], model_id: e.target.value }
                  })}
                  disabled={models.length === 0 || running}
                >
                  <option value="">首个启用模型</option>
                  {models.map((model) => (
                    <option value={model.id} key={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
                <select
                  aria-label={`${agent.name}思考等级`}
                  value={profiles[agent.id]?.effort ?? "inherit"}
                  onChange={(e) => void saveProfiles({
                    ...profiles,
                    [agent.id]: {
                      ...profiles[agent.id],
                      effort: e.target.value as ReasoningEffort
                    }
                  })}
                  disabled={models.length === 0 || running}
                >
                  {effortOptions.map((item) => (
                    <option value={item.value} key={item.value}>{item.label}</option>
                  ))}
                </select>
              </div>
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
              <Metric label={usage.estimated ? "Token 用量（含估算）" : "Token 用量"} value={String(tokenTotal || result?.tokens || 0)} />
              <button className="metric clickable" type="button" disabled={!sessionId} onClick={() => void renameCurrentSession()} title={sessionId ? `会话 ID：${sessionId}` : "未创建会话"}>
                <span>会话</span>
                <strong>{sessionId ? sessionTitle || sessionId : "未创建"}</strong>
              </button>
              <Metric label="项目" value={workdir ? shortPath(workdir) : "未选择"} />
              <Metric label="耗时" value={result ? formatDuration(result.duration_ms) : "--"} />
            </div>
            <div className="usage-breakdown" aria-label="模型用量明细">
              <span>输入 {usage.input.toLocaleString()}</span>
              <span>缓存 {usage.cached.toLocaleString()}</span>
              <span>输出 {usage.output.toLocaleString()}</span>
              <span>推理 {usage.reasoningKnown ? `${usage.reasoning.toLocaleString()}${usage.reasoningUnknown ? "（部分未提供）" : ""}` : "未提供"}</span>
            </div>
            <div className="agent-board">
              {agents.map((agent) => (
                <article className={agent.id === activeAgent ? "agent-card active" : "agent-card"} key={agent.id}>
                  <div>
                    <span className="agent-mark"><Bot size={14} /></span>
                    <strong>{agent.name}</strong>
                    <span className={`agent-state ${agentStats[agent.id].state}`}>{agentStats[agent.id].state}</span>
                  </div>
                  <p>{agentStats[agent.id].task}</p>
                  <footer>
                    <span>{models.find((model) => model.id === (agentStats[agent.id].modelId || profiles[agent.id]?.model_id))?.name ?? "自动选择"}{agentStats[agent.id].modelId ? "" : "（默认）"}</span>
                    <span><Brain size={12} /> {effortLabel(agentStats[agent.id].effort || profiles[agent.id]?.effort || "inherit")}{agentStats[agent.id].effort ? "" : "（默认）"}</span>
                    <span>{agentStats[agent.id].tokens} tok</span>
                  </footer>
                </article>
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
        <section
          className={pendingImages.length ? "composer has-images" : "composer"}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            if (!running) queueImages(Array.from(event.dataTransfer.files));
          }}
        >
          {pendingImages.length > 0 && (
            <div className="image-tray" aria-label="待发送图片">
              {pendingImages.map((item) => (
                <figure key={item.key}>
                  <img src={item.preview} alt={item.file.name} />
                  <figcaption>{item.file.name}</figcaption>
                  <button type="button" onClick={() => removeImage(item.key)} aria-label={`移除 ${item.file.name}`}>
                    <X size={13} />
                  </button>
                </figure>
              ))}
            </div>
          )}
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            onKeyDown={(event) => handleComposerKey(event, () => void submitComposer())}
            onPaste={(event) => {
              const files = Array.from(event.clipboardData.files);
              if (files.length && !running) queueImages(files);
            }}
            placeholder={running ? "输入对当前任务的补充或指导..." : "输入任务或问题..."}
          />
          <div className="actions">
            <input
              ref={fileRef}
              className="file-input"
              type="file"
              accept="image/png,image/jpeg,image/webp,image/gif"
              multiple
              onChange={(event) => {
                queueImages(Array.from(event.target.files ?? []));
                event.target.value = "";
              }}
            />
            <button className="secondary attach-button" type="button" onClick={() => fileRef.current?.click()} disabled={running}>
              <ImagePlus size={16} /> 图片
            </button>
            <button className="secondary" type="button" onClick={() => void pickProject()} disabled={running} title={workdir || "选择项目目录"}>
              <FolderOpen size={16} /> 选择项目
            </button>
            <button className="ghost" type="button" onClick={() => void openHistory()} disabled={running}>
              <History size={16} /> 历史会话
            </button>
            <button className="ghost" type="button" disabled={!workdir || running} onClick={() => void createSession()}>
              <GitBranch size={16} /> 新建会话
            </button>
            <button
              className={capabilityCount > 0 ? "ghost capability-button active" : "ghost capability-button"}
              type="button"
              disabled={running}
              onClick={() => setCapabilityModalOpen(true)}
              title="选择 Skills 和 MCP"
            >
              <Puzzle size={16} />
              能力
              {capabilityCount > 0 && (
                <span>{capabilityCount}</span>
              )}
            </button>
            <button
              className="primary send-button"
              disabled={!task.trim() || !workdir || (running && !activeRunId)}
              onClick={() => void submitComposer()}
            >
              {running ? <RadioTower size={16} /> : <Send size={16} />}
              {running ? "发送指导" : "发送"}
            </button>
            <label className="toggle shell-toggle">
              <input
                type="checkbox"
                checked={containerShell}
                onChange={(e) => setContainerShell(e.target.checked)}
                disabled={running}
              />
              <span>容器 Shell</span>
            </label>
            <label className="toggle plan-toggle">
              <input type="checkbox" checked={planMode} onChange={(e) => setPlanMode(e.target.checked)} disabled={running} />
              <span>Plan 模式</span>
            </label>
            <label className="effort-control">
              <Brain size={15} />
              <span>本轮思考</span>
              <select
                value={runEffort}
                onChange={(event) => setRunEffort(event.target.value as ReasoningEffort)}
                disabled={running}
              >
                {effortOptions.map((item) => (
                  <option value={item.value} key={item.value}>{item.label}</option>
                ))}
              </select>
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
                <input value={draft.api_key ?? ""} onChange={(e) => setDraft({ ...draft, api_key: e.target.value })} placeholder={draft.has_key ? `${draft.key_hint ?? "已配置"}（留空保持不变）` : "API Key"} type="password" />
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
              <div className="row">
                <label>
                  <span>调用协议</span>
                  <select value={draft.protocol} onChange={(e) => setDraft({ ...draft, protocol: e.target.value as ModelConfig["protocol"] })}>
                    <option value="auto">自动探测</option>
                    <option value="responses">Responses API</option>
                    <option value="chat_completions">Chat Completions</option>
                  </select>
                </label>
                <label>
                  <span>默认思考等级</span>
                  <select value={draft.default_effort} onChange={(e) => setDraft({ ...draft, default_effort: e.target.value as ReasoningEffort })}>
                    {effortOptions.filter((item) => item.value !== "inherit").map((item) => (
                      <option value={item.value} key={item.value}>{item.label}</option>
                    ))}
                  </select>
                </label>
              </div>
              <label>
                <span>最大输出 Token</span>
                <input value={draft.max_output_tokens} onChange={(e) => setDraft({ ...draft, max_output_tokens: Number(e.target.value) })} type="number" />
              </label>
              <fieldset className="capability-flags">
                <legend>模型能力</legend>
                {(["text", "vision", "tools", "reasoning"] as const).map((capability) => (
                  <label key={capability}>
                    <input
                      type="checkbox"
                      checked={draft.capabilities.includes(capability)}
                      onChange={() => setDraft({
                        ...draft,
                        capabilities: draft.capabilities.includes(capability)
                          ? draft.capabilities.filter((item) => item !== capability)
                          : [...draft.capabilities, capability]
                      })}
                    />
                    <span>{{ text: "文本", vision: "视觉", tools: "工具", reasoning: "推理" }[capability]}</span>
                  </label>
                ))}
              </fieldset>
              <label className="toggle model-enabled">
                <input type="checkbox" checked={draft.enabled} onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })} />
                <span>启用此模型</span>
              </label>
              {modelTestState.status !== "idle" && (
                <div
                  className={`model-test-result ${modelTestState.status}`}
                  role="status"
                  aria-live="polite"
                >
                  {modelTestState.status === "testing" && <Loader2 className="spin" size={16} />}
                  {modelTestState.status === "success" && <Check size={16} />}
                  {modelTestState.status === "error" && <XCircle size={16} />}
                  <span>{modelTestState.message}</span>
                </div>
              )}
              <div className="modal-actions">
                {draft.id && (
                  <button
                    className="secondary"
                    type="button"
                    onClick={() => void testModel(draft.id, true)}
                    disabled={modelTestState.status === "testing"}
                  >
                    {modelTestState.status === "testing" ? <Loader2 className="spin" size={16} /> : <RadioTower size={16} />}
                    测试连通性
                  </button>
                )}
                <button className="primary" type="submit">
                  <Save size={16} /> 保存模型
                </button>
              </div>
            </form>
          </div>
        )}

        {/* 能力弹窗：手动多选本轮使用的 Skills 和 MCP 服务。 */}
        {capabilityModalOpen && (
          <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="能力选择">
            <section className="modal capability-modal">
              <div className="modal-head">
                <h2>
                  <Puzzle size={18} />
                  能力选择
                </h2>
                <div className="modal-head-actions">
                  <button className="icon" type="button" onClick={() => void loadCapabilities()} title="刷新能力列表">
                    <RefreshCw size={17} />
                  </button>
                  <button className="icon" type="button" onClick={() => setCapabilityModalOpen(false)} title="关闭">
                    <XCircle size={18} />
                  </button>
                </div>
              </div>
              <div className="capability-grid">
                <section className="capability-section">
                  <div className="capability-section-head">
                    <h3>
                      <Puzzle size={16} /> Skills
                    </h3>
                    <span>{selectedSkills.length}/{skills.length}</span>
                  </div>
                  <div className="capability-list">
                    {skills.length === 0 && <p className="muted capability-empty">skills 目录中暂无可用 Skill</p>}
                    {skills.map((skill) => (
                      <label className="capability-row" key={skill.id}>
                        <input
                          type="checkbox"
                          checked={selectedSkills.includes(skill.id)}
                          onChange={() => setSelectedSkills(toggleSelected(selectedSkills, skill.id))}
                        />
                        <span>
                          <strong>{skill.name}</strong>
                          <small>{skill.description}</small>
                        </span>
                      </label>
                    ))}
                  </div>
                </section>
                <section className="capability-section">
                  <div className="capability-section-head">
                    <h3>
                      <PlugZap size={16} /> MCP
                    </h3>
                    <span>
                      {
                        new Set([
                          ...selectedMcps,
                          ...mcps.filter((item) => item.always_on).map((item) => item.id)
                        ]).size
                      }
                      /{mcps.length}
                    </span>
                  </div>
                  <div className="capability-list">
                    {mcps.length === 0 && <p className="muted capability-empty">MCP/servers.json 中暂无已启用服务</p>}
                    {mcps.map((mcp) => (
                      <div className="capability-row mcp-row" key={mcp.id}>
                        <label>
                          <input
                            type="checkbox"
                            checked={mcp.always_on || selectedMcps.includes(mcp.id)}
                            disabled={mcp.always_on}
                            onChange={() => setSelectedMcps(toggleSelected(selectedMcps, mcp.id))}
                          />
                          <span>
                            <strong>
                              {mcp.name}
                              {mcp.always_on ? " · 自动启用" : ""}
                            </strong>
                            <small>{mcp.description}</small>
                          </span>
                        </label>
                        <div className="mcp-actions">
                          {mcp.auth && (
                            <button
                              className="icon compact-icon"
                              type="button"
                              disabled={!workdir}
                              onClick={() => void authorizeMcp(mcp.id)}
                              title={workdir ? "通过浏览器授权 MCP" : "选择项目后可授权"}
                            >
                              <KeyRound size={14} />
                            </button>
                          )}
                          <button
                            className="icon compact-icon"
                            type="button"
                            disabled={!workdir}
                            onClick={() => void testMcp(mcp.id)}
                            title={workdir ? "测试 MCP 连通性" : "选择项目后可测试"}
                          >
                            <RadioTower size={14} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              </div>
              <div className="modal-actions capability-actions">
                <button
                  className="ghost"
                  type="button"
                  disabled={selectedSkills.length + selectedMcps.length === 0}
                  onClick={() => {
                    setSelectedSkills([]);
                    setSelectedMcps([]);
                  }}
                >
                  清空选择
                </button>
                <button className="primary" type="button" onClick={() => setCapabilityModalOpen(false)}>
                  <Check size={16} /> 完成
                </button>
              </div>
            </section>
          </div>
        )}

        {/* MCP 授权弹窗始终保留设备码，用户主动点击后再打开 GitHub 官方页面。 */}
        {mcpAuthPrompt && (
          <div className="modal-backdrop auth-backdrop" role="dialog" aria-modal="true" aria-label="GitHub MCP 授权">
            <section className="modal auth-modal">
              <div className="modal-head">
                <h2>
                  <KeyRound size={18} />
                  GitHub MCP 授权
                </h2>
                <button className="icon" type="button" onClick={closeMcpAuth} title="关闭授权窗口">
                  <XCircle size={18} />
                </button>
              </div>
              <div className={`auth-state ${mcpAuthPrompt.status}`}>
                {mcpAuthPrompt.status === "authorized" ? <Check size={18} /> : <KeyRound size={18} />}
                <span>{mcpAuthPrompt.msg}</span>
              </div>
              {mcpAuthPrompt.status === "authorization_required" && (
                <>
                  <div className="auth-code-block">
                    <span>在 GitHub 页面输入此设备码</span>
                    <code>{mcpAuthPrompt.code || "正在生成..."}</code>
                  </div>
                  <div className="auth-help">设备码是一次性的。GitHub 授权成功后，本机会保存登录状态。</div>
                </>
              )}
              <div className="modal-actions">
                {mcpAuthPrompt.code && (
                  <button className="secondary" type="button" onClick={() => void copyMcpCode()}>
                    <Copy size={16} /> 复制设备码
                  </button>
                )}
                {mcpAuthPrompt.url && (
                  <button
                    className="primary"
                    type="button"
                    onClick={() => window.open(mcpAuthPrompt.url || "", "_blank", "noopener,noreferrer")}
                  >
                    <ExternalLink size={16} /> 打开 GitHub 授权页
                  </button>
                )}
                {["authorized", "error"].includes(mcpAuthPrompt.status) && (
                  <button className="primary" type="button" onClick={closeMcpAuth}>
                    <Check size={16} /> 完成
                  </button>
                )}
              </div>
            </section>
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
    timeout: 120,
    protocol: "auto",
    capabilities: ["text", "vision", "tools", "reasoning"],
    efforts: ["minimal", "low", "medium", "high", "max"],
    default_effort: "low",
    max_output_tokens: 16384
  };
}

function emptyProfiles(map?: AgentModelMap): AgentProfileMap {
  return Object.fromEntries(
    agents.map((agent) => [
      agent.id,
      { model_id: map?.[agent.id] ?? "", effort: "inherit" as ReasoningEffort }
    ])
  ) as AgentProfileMap;
}

function effortLabel(value: ReasoningEffort): string {
  return effortOptions.find((item) => item.value === value)?.label ?? value;
}

function toggleSelected(current: string[], id: string): string[] {
  // 已选中时移除，未选中时追加；保持用户点击顺序，后端也按这个顺序注入上下文。
  return current.includes(id) ? current.filter((item) => item !== id) : [...current, id];
}

function delay(milliseconds: number): Promise<void> {
  // 授权状态轮询间隔使用 Promise 定时器，避免阻塞界面交互。
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function eventOwner(event: AgentEvent | undefined): string | undefined {
  const source = event?.data?.source_agent;
  return event?.agent === "llm" && typeof source === "string" && source ? source : event?.agent;
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

function handleComposerKey(event: KeyboardEvent<HTMLTextAreaElement>, send: () => void) {
  // Cmd/Ctrl + Enter 是开发工具中常见的发送方式，普通 Enter 仍然输入换行。
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    send();
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
