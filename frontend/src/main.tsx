// 作者：张又升
// 邮箱：1041586605@qq.com
// 单位：西北大学 电子信息学院(人工智能学院)

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

type ChatMessage = HistoryMessage;

type McpAuthPrompt = McpAuthStatus & { id: string };

type ModelTestState = {
  status: "idle" | "testing" | "success" | "error";
  message: string;
};

type PendingImage = {
  key: string;
  file: File;
  preview: string;
};

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8710";
const agents: Array<{ id: AgentName; name: string }> = [
  { id: "manager", name: "管理者" },
  { id: "planner", name: "规划" },
  { id: "coder", name: "编程" },
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
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [profiles, setProfiles] = useState<AgentProfileMap>(() => emptyProfiles());
  const [runEffort, setRunEffort] = useState<ReasoningEffort>("inherit");
  const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
  const [draft, setDraft] = useState<ModelConfig>(emptyModel());
  const [workdir, setWorkdir] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [sessionTitle, setSessionTitle] = useState("");
  const [task, setTask] = useState("");
  const [planMode, setPlanMode] = useState(false);
  const [containerShell, setContainerShell] = useState(true);
  const [running, setRunning] = useState(false);
  const [activeRunId, setActiveRunId] = useState("");
  const runLockRef = useRef(false);
  const guidanceLockRef = useRef(false);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [result, setResult] = useState<TaskResult | null>(null);
  const [notice, setNotice] = useState("等待会话启动");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDialogElement>(null);
  const settingsButtonRef = useRef<HTMLButtonElement>(null);
  const [profileSaving, setProfileSaving] = useState(false);
  const [modelSaving, setModelSaving] = useState(false);
  const [modelError, setModelError] = useState("");
  const [modelModalOpen, setModelModalOpen] = useState(false);
  const [editingModelId, setEditingModelId] = useState("");
  const [modelTestState, setModelTestState] = useState<ModelTestState>({ status: "idle", message: "" });
  const [historyModalOpen, setHistoryModalOpen] = useState(false);
  const [historyProjects, setHistoryProjects] = useState<HistoryProject[]>([]);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [mcps, setMcps] = useState<McpInfo[]>([]);
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [selectedMcps, setSelectedMcps] = useState<string[]>([]);
  const [capabilityModalOpen, setCapabilityModalOpen] = useState(false);
  const [mcpAuthPrompt, setMcpAuthPrompt] = useState<McpAuthPrompt | null>(null);
  const [copiedId, setCopiedId] = useState("");
  const authAttemptRef = useRef(0);
  const chatRef = useRef<HTMLDivElement>(null);
  const eventRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (settingsOpen && !settingsRef.current?.open) settingsRef.current?.showModal();
  }, [settingsOpen]);

  function closeSettings() {
    settingsRef.current?.close();
    setSettingsOpen(false);
    setModelModalOpen(false);
    settingsButtonRef.current?.focus();
  }

  useEffect(() => {
    void loadModels();
    void loadCapabilities();
  }, []);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length]);

  useEffect(() => {
    eventRef.current?.scrollTo({ top: eventRef.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);

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
  const capabilityCount = useMemo(
    () =>
      selectedSkills.length
      + new Set([
        ...selectedMcps,
        ...mcps.filter((item) => item.always_on).map((item) => item.id)
      ]).size,
    [mcps, selectedMcps, selectedSkills]
  );
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
      const data = await requestJson<CapabilitiesResponse>(`${API}/api/capabilities`);
      setSkills(data.skills ?? []);
      setMcps(data.mcps ?? []);
      const skillIds = new Set((data.skills ?? []).map((item) => item.id));
      const mcpIds = new Set((data.mcps ?? []).map((item) => item.id));
      setSelectedSkills((current) => current.filter((id) => skillIds.has(id)));
      setSelectedMcps((current) => current.filter((id) => mcpIds.has(id)));
    } catch (error) {
      setNotice(`能力列表读取失败：${errorMessage(error)}`);
    }
  }

  async function testMcp(id: string) {
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
    if (!workdir) {
      setNotice("请先选择项目目录，再授权 MCP");
      return;
    }
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
    authAttemptRef.current += 1;
    setMcpAuthPrompt(null);
  }

  async function copyMcpCode() {
    if (!mcpAuthPrompt?.code) return;
    try {
      await navigator.clipboard.writeText(mcpAuthPrompt.code);
      setNotice("设备码已复制");
    } catch (error) {
      setNotice(`复制设备码失败：${errorMessage(error)}`);
    }
  }

  async function saveModel(event: FormEvent) {
    event.preventDefault();
    setModelSaving(true);
    setModelError("");
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
      setModelError(errorMessage(error));
      setNotice(errorMessage(error));
    } finally {
      setModelSaving(false);
    }
  }

  async function deleteModel(id: string) {
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
    if (showInModal) {
      setModelTestState({ status: "testing", message: "正在连接模型接口..." });
    }
    setNotice("正在测试模型连通性");
    try {
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
    setProfileSaving(true);
    setProfiles(next);
    try {
      await requestJson<AgentProfileMap>(`${API}/api/agent-profiles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(next)
      });
      setNotice("智能体模型与思考等级已保存");
    } catch (error) {
      setProfiles(previous);
      setNotice(errorMessage(error));
    } finally {
      setProfileSaving(false);
    }
  }

  async function createSession() {
    if (!workdir) {
      setNotice("请先选择项目目录");
      return;
    }
    try {
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
    setNotice("正在打开目录选择器");
    try {
      const data = await requestJson<{ workdir: string }>(`${API}/api/pick-dir`, { method: "POST" });
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
    setNotice("正在读取历史会话");
    try {
      const data = await requestJson<{ projects: HistoryProject[] }>(`${API}/api/history`);
      setHistoryProjects(data.projects ?? []);
      setHistoryModalOpen(true);
      setNotice("历史会话已加载");
    } catch (error) {
      setNotice(`读取历史失败：${errorMessage(error)}`);
    }
  }

  function restoreSession(session: HistorySession) {
    setWorkdir(session.workdir);
    setSessionId(session.id);
    setSessionTitle(session.title);
    setMessages(session.messages);
    setEvents(session.events ?? []);
    setSelectedSkills(session.skills ?? []);
    setSelectedMcps(session.mcps ?? []);
    setContainerShell(session.container_shell ?? true);
    setResult(session.result ?? null);
    setHistoryModalOpen(false);
    setNotice(`已恢复会话：${session.title || session.id}`);
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
    if (!window.confirm(`删除会话「${session.title}」？只会删除 agent memory 中的会话记录。`)) return;
    try {
      const data = await requestJson<{ deleted: boolean }>(
        `${API}/api/sessions/${encodeURIComponent(session.id)}?workdir=${encodeURIComponent(session.workdir)}`,
        { method: "DELETE" }
      );
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
    } catch (error) {
      setNotice(`删除会话失败：${errorMessage(error)}`);
    }
  }

  async function deleteHistoryProject(project: HistoryProject) {
    if (!window.confirm(`删除项目「${project.name}」的全部 memory 记录？不会删除真实项目目录。`)) return;
    try {
      const data = await requestJson<{ deleted: boolean }>(`${API}/api/projects?workdir=${encodeURIComponent(project.workdir)}`, {
        method: "DELETE"
      });
      setHistoryProjects((prev) => prev.filter((item) => item.id !== project.id));
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
      const id = sessionId || (await createSessionAndReturnId());
      setNotice(pendingImages.length ? "正在安全上传图片" : "任务执行中");
      const uploaded = await uploadImages(id);
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-user`,
          role: "user",
          content: trimmedTask,
          attachments: uploaded
        }
      ]);
      setTask("");
      pendingImages.forEach((item) => URL.revokeObjectURL(item.preview));
      setPendingImages([]);
      setNotice("任务执行中");
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
        const errorData = await res.json().catch(() => ({}));
        throw new Error(detailText(errorData.detail) || `请求失败：${res.status}`);
      }
      const runId = res.headers.get("X-Run-ID") ?? "";
      if (!runId) throw new Error("后端没有返回当前运行编号");
      setActiveRunId(runId);
      const reader = res.body?.getReader();
      if (!reader) throw new Error("浏览器不支持流式读取");
      const decoder = new TextDecoder();
      let buf = "";
      let terminalReceived = false;

      const consumeLine = (line: string) => {
        if (!line.trim()) return;
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
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        lines.forEach(consumeLine);
      }
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
    if (!workdir) {
      throw new Error("请先选择项目目录");
    }
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
        <aside className="panel log event-panel" aria-label="事件流">
          <div className="event-head">
            <SectionTitle icon={<RadioTower size={18} />} text="事件流" />
            <div className="event-head-actions">
              <span className="section-count" aria-label="事件数量">{events.length}</span>
              <button ref={settingsButtonRef} className="ghost settings-button" type="button" onClick={() => setSettingsOpen(true)} aria-haspopup="dialog">
                <Settings size={15} /> 设置
              </button>
            </div>
          </div>
          <div className="events" ref={eventRef}>
            {events.length === 0 && <p className="muted empty-events">执行任务后，事件将在这里按时间展示</p>}
            {events.map((event) => (
              <article key={`${event.id}-${event.ts}`} className="event">
                <div>
                  <span className="event-index">{String(event.id).padStart(2, "0")}</span>
                  <span className="tag">{agentName(event.agent)}</span>
                  <span className="kind">{event.kind}</span>
                  <time>{formatEventTime(event.ts)}</time>
                </div>
                <p className="event-message" tabIndex={0}>{event.msg}</p>
                {event.data && Object.keys(event.data).length > 0 && (
                  <details className="event-detail">
                    <summary>查看详细信息</summary>
                    <pre>{JSON.stringify(event.data, null, 2)}</pre>
                  </details>
                )}
              </article>
            ))}
          </div>
        </aside>

        <section className="panel chat-panel">
          <div className="chat-window" ref={chatRef} aria-live="polite">
            {messages.length === 0 && (
              <div className="empty-chat">
                <FileText size={28} />
                <p>选择项目，开始与智能体协作。</p>
              </div>
            )}
            {messages.map((msg) => (
              <div className={`message-row ${msg.role}`} key={msg.id}>
                <article className={`bubble ${msg.role}`}>
                  <span className="message-role">{msg.role === "user" ? "用户" : "智能体"}</span>
                  {!!msg.attachments?.length && (
                    <div className="message-images">
                      {msg.attachments.map((attachment) => (
                        <a key={attachment.id} href={`${API}${attachment.url}`} target="_blank" rel="noreferrer" title={`查看图片：${attachment.name}`}>
                          <img src={`${API}${attachment.url}`} alt={attachment.name} loading="lazy" />
                          <span>{attachment.name}</span>
                        </a>
                      ))}
                    </div>
                  )}
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

        <aside className="right-col">
          <section className="panel status" aria-label="运行状态" tabIndex={0}>
            <div className="status-title">
              <SectionTitle icon={<Activity size={18} />} text="运行状态" />
              <div className="health">
                {running ? <Loader2 className="spin" size={15} /> : <Circle className="status-dot" size={10} fill="currentColor" />}
                <ScrollText text={notice} className="notice-text" />
              </div>
            </div>
            <div className="metrics">
              <Metric label="当前智能体" value={agentName(activeAgent)} />
              <Metric label="事件数" value={String(events.length)} />
              <Metric label={usage.estimated ? "Token 用量（含估算）" : "Token 用量"} value={String(tokenTotal || result?.tokens || 0)} />
              <button className="metric clickable" type="button" disabled={!sessionId} onClick={() => void renameCurrentSession()} title={sessionId ? `会话 ID：${sessionId}` : "未创建会话"}>
                <span>会话</span>
                <strong><ScrollText text={sessionId ? sessionTitle || sessionId : "未创建"} focusable={false} /></strong>
              </button>
              <Metric label="项目" value={workdir || "未选择"} />
              <Metric label={result?.metrics_estimated ? "耗时（估算）" : "耗时"} value={result ? formatDuration(result.duration_ms) : "--"} />
            </div>
            <div className="usage-breakdown" aria-label="模型用量明细">
              <span>输入 {usage.input.toLocaleString()}</span>
              <span>缓存 {usage.cached.toLocaleString()}</span>
              <span>输出 {usage.output.toLocaleString()}</span>
              <span>推理 {usage.reasoningKnown ? `${usage.reasoning.toLocaleString()}${usage.reasoningUnknown ? "（部分未提供）" : ""}` : "未提供"}</span>
            </div>
            <div className="agent-board">
              {agents.map((agent) => (
                <article className={agent.id === activeAgent ? "agent-card active" : "agent-card"} data-agent={agent.id} key={agent.id}>
                  <div>
                    <span className="agent-mark"><Bot size={14} /></span>
                    <strong>{agent.name}</strong>
                    <span className={`agent-state ${agentStats[agent.id].state}`}>{agentStats[agent.id].state}</span>
                  </div>
                  <p className="agent-task" tabIndex={0} aria-label={`${agent.name}当前任务`}>{agentStats[agent.id].task}</p>
                  <footer>
                    <ScrollText text={`${models.find((model) => model.id === (agentStats[agent.id].modelId || profiles[agent.id]?.model_id))?.name ?? "自动选择"}${agentStats[agent.id].modelId ? "" : "（默认）"}`} />
                    <span><Brain size={12} /> {effortLabel(agentStats[agent.id].effort || profiles[agent.id]?.effort || "inherit")}{agentStats[agent.id].effort ? "" : "（默认）"}</span>
                    <span>{agentStats[agent.id].tokens} tok</span>
                  </footer>
                </article>
              ))}
            </div>
          </section>
        </aside>

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

        {settingsOpen && (
          <dialog
            ref={settingsRef}
            className={`modal settings-modal${modelModalOpen ? " editing" : ""}`}
            aria-label="设置"
            onCancel={(event) => {
              event.preventDefault();
              if (modelModalOpen) setModelModalOpen(false);
              else closeSettings();
            }}
            onClose={() => setSettingsOpen(false)}
            onClick={(event) => {
              if (event.target === event.currentTarget) {
                const bounds = event.currentTarget.getBoundingClientRect();
                if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
                  closeSettings();
                }
              }
            }}
          >
            {modelModalOpen ? (
              <form className="settings-editor" onSubmit={saveModel}>
                <div className="modal-head">
                  <h2>
                    <Settings size={18} />
                    模型配置
                  </h2>
                  <button className="icon" type="button" onClick={() => setModelModalOpen(false)} title="返回设置">
                    <X size={18} />
                  </button>
                </div>
                <label>
                  <span>配置 ID</span>
                  <input value={draft.id} onChange={(e) => setDraft({ ...draft, id: e.target.value })} placeholder="例如 longcat" disabled={Boolean(editingModelId)} />
                </label>
                <label>
                  <span>显示名称</span>
                  <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="模型显示名称" autoFocus />
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
                  <span>接口密钥</span>
                  <input value={draft.api_key ?? ""} onChange={(e) => setDraft({ ...draft, api_key: e.target.value })} placeholder={draft.has_key ? `${draft.key_hint ?? "已配置"}（留空保持不变）` : "请输入接口密钥"} type="password" />
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
                  <span>最大输出词元数</span>
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
                {modelError && <p className="model-error" role="alert">{modelError}</p>}
                <div className="modal-actions">
                  <button className="ghost" type="button" onClick={() => setModelModalOpen(false)}>返回设置</button>
                  {editingModelId && (
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
                  <button className="primary" type="submit" disabled={modelSaving}>
                    {modelSaving ? <Loader2 className="spin" size={16} /> : <Save size={16} />} {modelSaving ? "正在保存…" : "保存模型"}
                  </button>
                </div>
              </form>
            ) : (
              <>
                <div className="modal-head settings-head">
                  <div>
                    <h2><Settings size={19} /> 设置</h2>
                    <p className="muted">统一管理模型连接与智能体分工</p>
                  </div>
                  <button className="icon ghost" type="button" onClick={closeSettings} title="关闭设置" autoFocus>
                    <X size={18} />
                  </button>
                </div>
                <div className="settings-grid">
                  <section className="settings-section" aria-label="模型配置">
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
                              setModelError("");
                              setModelModalOpen(true);
                            }}
                            disabled={running || profileSaving}
                            title="编辑模型"
                          >
                            <Cpu size={14} />
                            <span>{model.name}</span>
                          </button>
                          <button className="icon model-action-button" onClick={() => testModel(model.id)} title="测试连通性" disabled={running || profileSaving}>
                            <RadioTower size={14} />
                          </button>
                          <button className="icon danger model-action-button" onClick={() => deleteModel(model.id)} title="删除模型" disabled={running || profileSaving}>
                            <Trash2 size={14} />
                          </button>
                        </div>
                      ))}
                      <button
                        className="ghost"
                        disabled={running || profileSaving}
                        onClick={() => {
                          setDraft(emptyModel());
                          setEditingModelId("");
                          setModelTestState({ status: "idle", message: "" });
                          setModelError("");
                          setModelModalOpen(true);
                        }}
                      >
                        <Plus size={16} /> 新增模型
                      </button>
                    </div>
                  </section>
                  <section className="settings-section" aria-label="智能体配置">
                    <div className="side-section agent-map-title">
                      <SectionTitle icon={<Bot size={18} />} text="智能体配置" />
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
                            disabled={models.length === 0 || running || profileSaving}
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
                            disabled={models.length === 0 || running || profileSaving}
                          >
                            {effortOptions.filter((item) => item.value === "inherit" || (models.find((model) => model.id === profiles[agent.id]?.model_id) ?? models.find((model) => model.enabled))?.efforts.includes(item.value)).map((item) => (
                              <option value={item.value} key={item.value}>{item.label}</option>
                            ))}
                          </select>
                        </div>
                      ))}
                    </div>
                  </section>
                </div>
                <div className="settings-footer">
                  <p role="status">{running ? "任务执行中，配置暂不可修改。" : profileSaving ? "正在保存智能体配置…" : notice}</p>
                  <button className="primary" type="button" onClick={closeSettings}>完成</button>
                </div>
              </>
            )}
          </dialog>
        )}

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
      <strong><ScrollText text={value} /></strong>
    </div>
  );
}

function ScrollText({ text, className = "", focusable = true }: { text: string; className?: string; focusable?: boolean }) {
  const viewportRef = useRef<HTMLSpanElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const [distance, setDistance] = useState(0);

  useEffect(() => {
    const viewport = viewportRef.current;
    const content = textRef.current;
    if (!viewport || !content) return;
    const measure = () => setDistance(Math.max(0, content.scrollWidth - viewport.clientWidth));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    observer.observe(content);
    return () => observer.disconnect();
  }, [text]);

  return (
    <span
      ref={viewportRef}
      className={`scroll-text ${distance > 0 ? "is-overflowing" : ""} ${className}`}
      title={text}
      tabIndex={distance > 0 && focusable ? 0 : undefined}
      style={{ "--scroll-distance": `${-distance}px`, "--scroll-duration": `${Math.max(6, distance / 26)}s` } as React.CSSProperties}
    >
      <span className="scroll-text-content" ref={textRef}>{text}</span>
    </span>
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
  return current.includes(id) ? current.filter((item) => item !== id) : [...current, id];
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function eventOwner(event: AgentEvent | undefined): string | undefined {
  const source = event?.data?.source_agent;
  return event?.agent === "llm" && typeof source === "string" && source ? source : event?.agent;
}

function agentName(id: string) {
  return agents.find((agent) => agent.id === id)?.name ?? id;
}

function formatTime(value: string) {
  if (!value) return "未知时间";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function formatEventTime(value: string) {
  if (!value) return "--:--:--";
  return new Date(value).toLocaleTimeString("zh-CN", { hour12: false });
}

function formatDuration(value: number) {
  if (!Number.isFinite(value) || value < 0) return "--";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`;
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

function handleComposerKey(event: KeyboardEvent<HTMLTextAreaElement>, send: () => void) {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    send();
  }
}

async function requestJson<T = Record<string, unknown>>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const data = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (!response.ok) {
    throw new Error(detailText(data.detail) || `请求失败：${response.status}`);
  }
  return data as T;
}

function detailText(detail: unknown): string {
  if (typeof detail === "string") return detail === "Not Found" ? "后端接口不存在，请重启项目" : detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (typeof item === "object" && item && "msg" in item ? String((item as { msg: unknown }).msg) : String(item)))
      .join("；");
  }
  return "";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

createRoot(document.getElementById("root")!).render(<App />);
