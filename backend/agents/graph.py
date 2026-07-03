from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from api.schema import AgentEvent, ContextPackage, TaskResult
from backend.agents.prompts import (
    ANSWER_PROMPT,
    CODER_PROMPT,
    DOC_PROMPT,
    MANAGER_PROMPT,
    PLANNER_PROMPT,
)
from backend.agents.types import AgentState
from backend.memory import MemoryStore
from backend.tools import FsTool, GitTool, ShellTool
from llm import LlmClient, LlmError, ModelStore

# Emit 是事件回调类型，AgentGraph 通过它把 AgentEvent 推给 FastAPI 流式接口。
Emit = Callable[[AgentEvent], None]


class AgentGraph:
    """多智能体编排图。

    每次任务创建一个实例，这样事件回调、token 统计和重试次数不会串到别的会话。
    """

    def __init__(self, model_store: ModelStore, memory: MemoryStore, emit: Emit | None = None):
        # model_store 负责根据智能体名称找到对应模型配置。
        self.model_store = model_store
        # memory 负责读写会话记忆、项目长期记忆和全局长期记忆。
        self.memory = memory
        # emit_cb 是外部传入的事件回调；CLI 和 Web 流式接口都会用它实时显示事件。
        self.emit_cb = emit
        # event_id 是本次任务内递增事件编号，每个 AgentGraph 实例独立计数。
        self.event_id = 0
        # graph 是编译后的 LangGraph 状态机。
        self.graph = self._build()

    def run(
        self,
        session_id: str,
        workdir: str,
        text: str,
        plan_mode: bool = False,
        execute_plan: bool = False,
        model_id: str | None = None,
    ) -> TaskResult:
        """运行一次用户任务，并返回最终 TaskResult。"""

        # state 是 LangGraph 节点间传递的共享状态，所有节点都通过它读写任务上下文。
        state: AgentState = {
            # 当前会话 id，用于 memory jsonl 文件定位。
            "session_id": session_id,
            # 当前项目目录，文件和命令工具都受它限制。
            "workdir": workdir,
            # 用户本轮输入，Plan 流程中可能被 manager 改写。
            "text": text,
            # 是否开启 Plan 模式。
            "plan_mode": plan_mode,
            # 是否直接执行已保存计划。
            "execute_plan": execute_plan,
            # 本次请求是否覆盖默认模型选择。
            "model_id": model_id,
            # changes 保存本轮修改过的文件。
            "changes": [],
            # commands 保存本轮执行过的命令。
            "commands": [],
            # tests 保存验证智能体产生的测试结果。
            "tests": [],
            # tests_ok 表示验证是否整体通过。
            "tests_ok": True,
            # retry 是 Coding 修复重试次数。
            "retry": 0,
            # tokens 是本轮模型调用累计 token 估算。
            "tokens": 0,
        }
        # final 是 LangGraph 执行完后返回的最终状态。
        final = self.graph.invoke(state)
        # result 优先使用 final 节点写入的结构化结果；兜底用于异常情况下仍能返回。
        result = final.get("result") or {
            "ok": bool(final.get("tests_ok", True)),
            "summary": final.get("final", "任务已结束"),
            "files": final.get("changes", []),
            "commands": final.get("commands", []),
            "tests": final.get("tests", []),
            "plan_path": (final.get("plan") or {}).get("path"),
            "doc_path": (final.get("repo") or {}).get("doc_path"),
        }
        return TaskResult(**result)

    def _build(self) -> Any:
        """构建 LangGraph 节点和条件路由。"""

        # graph 是以 AgentState 为状态类型的 LangGraph 状态图。
        graph = StateGraph(AgentState)
        # manager 是入口节点，负责分类、构造 Context Package 和决定第一跳。
        graph.add_node("manager", self.manager)
        # planner 负责 Plan 模式下的选择题和计划文件生成。
        graph.add_node("planner", self.planner)
        # repo 负责读取仓库结构和关键文件片段。
        graph.add_node("repo", self.repo)
        # answer 负责解释和普通答疑，不写磁盘。
        graph.add_node("answer", self.answer)
        # coder 负责 ReAct 写代码和执行工具动作。
        graph.add_node("coder", self.coder)
        # verifier 负责选择并运行测试。
        graph.add_node("verifier", self.verifier)
        # doc 负责生成或更新中文文档。
        graph.add_node("doc", self.doc)
        # final 负责整理 TaskResult、写最终 memory、输出最终事件。
        graph.add_node("final", self.final)

        # 所有任务都从 manager 进入。
        graph.add_edge(START, "manager")
        # manager 根据 route 字段决定下一跳，避免所有任务都走固定流水线。
        graph.add_conditional_edges(
            "manager",
            self.route_after_manager,
            {"planner": "planner", "repo": "repo", "answer": "answer", "final": "final"},
        )
        # planner 生成问题时直接 final，确认执行计划时可以进入 repo。
        graph.add_conditional_edges(
            "planner",
            self.route_after_planner,
            {"repo": "repo", "final": "final"},
        )
        # repo 读取仓库后根据任务类型去 coder、doc、answer 或 final。
        graph.add_conditional_edges(
            "repo",
            self.route_after_repo,
            {"coder": "coder", "doc": "doc", "answer": "answer", "final": "final"},
        )
        # answer 是只读答疑节点，结束后直接 final。
        graph.add_edge("answer", "final")
        # coder 写完代码后必须进入 verifier。
        graph.add_edge("coder", "verifier")
        # verifier 失败可回 coder 修复，成功后可去 doc 或 final。
        graph.add_conditional_edges(
            "verifier",
            self.route_after_verifier,
            {"coder": "coder", "doc": "doc", "final": "final"},
        )
        # doc 写完文档后进入 final。
        graph.add_edge("doc", "final")
        # final 是终点。
        graph.add_edge("final", END)
        return graph.compile()

    def manager(self, state: AgentState) -> AgentState:
        """上下文管理智能体：分类任务、处理 Plan 状态、构造 Context Package。"""

        # workdir 是当前会话对应的项目目录。
        workdir = state["workdir"]
        # session_id 是当前会话 id。
        session_id = state["session_id"]
        # interrupted 表示上次会话最后一条记录是否不是 final/result。
        interrupted = self.memory.interrupted(workdir, session_id)
        self._emit(state, "manager", "start", "管理者正在分类任务并构造上下文包")

        # pending_plan 是上一轮 Plan 选择题状态；存在时用户本轮可能是在回复选项。
        pending_plan = self._latest_pending_plan(workdir, session_id)
        # saved_plan 是最近生成但等待确认执行的计划。
        saved_plan = self._latest_saved_plan(workdir, session_id)
        # classification 保存管理者分类结果，后续会交给 _flow_for 转成路由。
        classification: dict[str, Any]
        if pending_plan and self._is_plan_cancel(state["text"]):
            # 用户明确取消 Plan 时写入 plan_cancelled，避免下一轮继续读取旧 pending_plan。
            self.memory.append(workdir, session_id, "planner", "state", "plan_cancelled", "用户取消了待确认计划", {"goal": pending_plan.get("goal", "")})
            classification = {
                "task_type": "direct",
                "need_repo": False,
                "need_code": False,
                "need_doc": False,
                "need_clarify": False,
                "reason": "用户取消了上一轮 Plan 流程",
                "direct_reply": "已取消上一轮 Plan 流程。你可以重新描述新的需求。",
            }
        elif pending_plan and self._should_treat_as_plan_reply(state["text"], pending_plan, state):
            # reply 是把用户的 1A/2B 等回复解析成结构化答案后的结果。
            reply = self._build_plan_reply(state["text"], pending_plan)
            # 把原始需求、上一轮问题和用户答案合并成新的任务文本，避免模型只看到“1A,2A”。
            state = {
                **state,
                "text": self._compose_pending_plan_text(pending_plan, reply),
                "plan_mode": True,
                "pending_plan": pending_plan,
                "plan_answers": reply["answers"],
            }
            classification = {
                "task_type": "plan_gen",
                "need_repo": True,
                "need_code": False,
                "need_doc": False,
                "need_clarify": False,
                "reason": "用户正在回答上一轮 Plan 澄清问题，继续同一个 Plan 流程",
            }
        elif saved_plan and (state.get("execute_plan") or self._is_execute_plan_text(state["text"])):
            # 用户确认执行计划时，把计划正文合并进任务文本，后续 repo/coder 能看到完整计划。
            state = {
                **state,
                "text": self._compose_saved_plan_text(saved_plan, state["text"]),
                "plan": saved_plan,
            }
            classification = {
                "task_type": "code_gen",
                "need_repo": True,
                "need_code": True,
                "need_doc": True,
                "need_clarify": False,
                "reason": "用户确认执行已保存计划",
            }
        else:
            # 没有 Plan 特殊状态时，进入普通任务分类。
            classification = self._classify(state)
        # task_type 是标准任务类型，兜底为 code_gen。
        task_type = classification.get("task_type", "code_gen")
        # route/after_repo/after_verify 是 LangGraph 后续路由需要的三个方向字段。
        route, after_repo, after_verify = self._flow_for(state, classification)

        # ctx 是管理者给下游智能体的结构化上下文包，不直接塞完整历史。
        ctx = ContextPackage(
            goal=state["text"],
            task_type=task_type,
            workdir=workdir,
            plan_mode=bool(state.get("plan_mode")),
            project_memory=self._trim(self.memory.project_memory(workdir), 5000),
            global_memory=self._trim(self.memory.global_memory(), 3000),
            constraints=[
                "代码文件名使用简短英文",
                "面向用户内容、注释和文档使用简体中文",
                "优先最小改动，避免无关重构",
                "危险命令需要用户确认",
            ],
            recent=[rec.get("out", "") for rec in self.memory.read_session(workdir, session_id, limit=8)],
        )
        if interrupted:
            # 会话疑似中断时，在最近上下文里提醒下游以磁盘当前状态为准。
            ctx.recent.append("检测到上次会话可能异常中断，本次会继续以当前磁盘状态为准。")
        # 记录本次分类结果，供历史排查和中断恢复使用。
        self.memory.append(workdir, session_id, "manager", "classify", "context", f"任务分类：{task_type}")
        return {
            **state,
            "task_type": task_type,
            "route": route,
            "after_repo": after_repo,
            "after_verify": after_verify,
            "need_doc": after_verify == "doc" or after_repo == "doc",
            "context": ctx,
            "final": classification.get("direct_reply") or classification.get("reason", ""),
        }

    def planner(self, state: AgentState) -> AgentState:
        """Plan 智能体：生成澄清问题或可执行 Markdown 计划。"""

        self._emit(state, "planner", "start", "Plan 智能体正在生成问题或可执行计划")
        # client 是 planner 智能体对应的模型客户端。
        client = self._client("planner", state)
        # messages 是发给模型的系统提示词和上下文包。
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": self._ctx_text(state)},
        ]
        try:
            # data 应该是模型返回的 JSON，status 为 questions 或 plan。
            data = client.chat_json(messages)
            self._add_tokens(state, client.last_usage.total)
        except LlmError as exc:
            # 模型返回非 JSON 时使用兜底问题或兜底计划，避免 Plan 流程卡死。
            self._emit(state, "planner", "error", f"Plan 模型返回异常，已使用默认澄清问题：{str(exc)[:120]}")
            if state.get("pending_plan"):
                # 如果已经在回答上一轮问题，模型异常时直接生成默认计划。
                data = {"status": "plan", "title": "默认执行计划", "markdown": self._fallback_plan_markdown(state)}
            else:
                # 第一轮 Plan 模型异常时，返回一个默认选择题让用户继续。
                data = {
                    "status": "questions",
                    "questions": [
                        {
                            "question": "是否直接按默认工程架构执行？",
                            "options": ["直接执行", "继续细化计划"],
                            "recommended": "直接执行",
                            "reason": "模型返回格式异常，先用默认问题继续推进，避免阻塞后续执行。",
                            "allow_custom": True,
                        }
                    ],
                }
        if data.get("status") == "questions":
            # 清洗模型输出的问题，防止 JSON 错误信息或异常长文本展示给用户。
            data = {**data, "questions": self._normalize_plan_questions(data.get("questions"))}
        if data.get("status") == "plan" or state.get("execute_plan"):
            # md 是最终计划正文；缺失时使用兜底计划。
            md = data.get("markdown") or self._fallback_plan_markdown(state)
            # plan_dir 是用户项目内保存计划的目录。
            plan_dir = Path(state["workdir"]) / "docs" / "plans"
            plan_dir.mkdir(parents=True, exist_ok=True)
            # path 是本会话对应的计划文件路径。
            path = plan_dir / f"{state['session_id']}.md"
            path.write_text(md.strip() + "\n", encoding="utf-8")
            self._emit(state, "planner", "plan", f"计划已保存：{path}")
            self.memory.append(
                state["workdir"],
                state["session_id"],
                "planner",
                "state",
                "plan_done",
                "计划已生成",
                {"status": "plan", "path": str(path), "markdown": md, "goal": self._plan_goal(state)},
            )
            # execute_plan=True 时直接进入 repo，否则停在 final 等用户确认。
            route = "repo" if state.get("execute_plan") else "final"
            # final 是返回给用户的确认提示；直接执行时保持空字符串。
            final = "" if route == "repo" else "计划已生成，等待你确认执行。\n计划文件：" + str(path) + "\n确认后点击“执行计划”，或回复“执行计划”。"
            return {
                **state,
                "plan": {"status": "plan", "path": str(path), "markdown": md},
                "route": route,
                "after_repo": "coder",
                "after_verify": "doc",
                "need_doc": True,
                "final": final,
            }
        # summary 是格式化后的选择题文本，直接展示给用户。
        summary = self._format_plan_questions(data.get("questions", []))
        # 保存 pending_plan，下一轮用户回复选项时 manager 可以找回上下文。
        self.memory.append(
            state["workdir"],
            state["session_id"],
            "planner",
            "state",
            "pending_plan",
            "Plan 问题等待用户回答",
            {
                "status": "questions",
                "goal": self._plan_goal(state),
                "questions": data.get("questions", []),
                "answers": state.get("plan_answers", []),
            },
        )
        self._emit(state, "planner", "questions", "已生成澄清问题，等待用户选择")
        return {**state, "plan": data, "route": "final", "final": summary}

    def repo(self, state: AgentState) -> AgentState:
        """仓库读取智能体：扫描项目结构、读取关键文件、识别技术栈。"""

        self._emit(state, "repo", "start", "仓库读取智能体正在识别目录结构和技术栈")
        # fs 是受 workdir 限制的文件工具。
        fs = FsTool(state["workdir"])
        # files 是项目内最多 200 个文件的相对路径列表。
        files = fs.list()
        # snippets 保存关键文件的片段，供下游模型理解项目。
        snippets: dict[str, str] = {}
        # priority 是优先读取的文件列表，包含项目说明、依赖配置和常见入口。
        priority = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "src/main.py",
            "main.py",
            "app.py",
            "index.html",
        ]
        # 先读取优先级文件，每个最多 8000 字符。
        for rel in priority:
            if rel in files:
                snippets[rel] = fs.read(rel, 8000)
        # 再读取前 30 个常见代码/配置/文档文件，避免完全空上下文。
        for rel in files[:30]:
            if rel not in snippets and rel.endswith((".py", ".ts", ".tsx", ".js", ".md", ".json", ".html", ".css")):
                snippets[rel] = fs.read(rel, 4000)
        # stack 是根据文件名粗略识别的技术栈。
        stack = self._detect_stack(files)
        # repo 是传给下游节点的仓库摘要。
        repo = {"files": files, "snippets": snippets, "stack": stack, "empty": len(files) == 0}
        self._emit(state, "repo", "summary", f"识别到 {len(files)} 个文件，技术栈：{', '.join(stack) or '空项目'}")
        self.memory.append(state["workdir"], state["session_id"], "repo", "fs", "summary", f"文件数：{len(files)}，技术栈：{stack}")
        # ctx 是管理者构造的上下文包，这里补充相关文件列表。
        ctx = state["context"]
        ctx.relevant_files = files[:80]
        return {**state, "repo": repo, "context": ctx}

    def answer(self, state: AgentState) -> AgentState:
        """答疑智能体：只回答问题，不写入磁盘。"""

        self._emit(state, "answer", "start", "答疑智能体正在生成直接回答")
        # answer 复用 manager 模型配置，因为它也是轻量回答类任务。
        client = self._client("manager", state)
        # messages 包含答疑 prompt、Context Package 和可选仓库摘要。
        messages = [
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": self._ctx_text(state)
                + "\n\n仓库摘要：\n"
                + json.dumps(state.get("repo", {}), ensure_ascii=False)[:16000],
            },
        ]
        try:
            # text 是模型生成的最终回答。
            text = client.chat(messages, temperature=0.2).strip()
            self._add_tokens(state, client.last_usage.total)
        except LlmError as exc:
            # 模型失败时仍返回明确错误，不进入 Coding。
            text = f"我现在无法完成解释，因为模型调用失败：{exc}"
        self.memory.append(state["workdir"], state["session_id"], "answer", "llm", "summary", text)
        return {**state, "final": text}

    def coder(self, state: AgentState) -> AgentState:
        """Coding 智能体：通过 ReAct 循环调用工具，真实修改项目文件。"""

        # retry 是当前 Coding 轮次，验证失败回到 coder 时会递增。
        retry = int(state.get("retry", 0))
        self._emit(state, "coder", "start", f"Coding 智能体开始 ReAct 执行，第 {retry + 1} 轮")
        # fs/shell/git 是 Coding 智能体可调用的工具集合。
        fs = FsTool(state["workdir"])
        shell = ShellTool(state["workdir"])
        git = GitTool(state["workdir"])
        # client 是 coder 智能体对应的模型客户端。
        client = self._client("coder", state)
        # observations 保存工具执行结果，下一步会回填给模型作为观察。
        observations: list[str] = []
        # changes 继承已有变更列表，验证失败重试时不会丢掉前一轮变更记录。
        changes = list(state.get("changes", []))
        # commands 继承已有命令列表。
        commands = list(state.get("commands", []))
        if state.get("tests") and not state.get("tests_ok", True):
            # 如果上一轮验证失败，把测试结果作为观察反馈给模型修复。
            observations.append("上一轮测试失败：\n" + json.dumps(state["tests"], ensure_ascii=False, indent=2))

        # 单轮 Coding 最多执行 6 次模型-工具循环，避免模型无限调用工具。
        for _step in range(1, 7):
            # messages 是本轮发给 Coding 模型的上下文。
            messages = [
                {"role": "system", "content": CODER_PROMPT},
                {
                    "role": "user",
                    "content": self._ctx_text(state)
                    + "\n\n仓库摘要：\n"
                    + json.dumps(state.get("repo", {}), ensure_ascii=False)[:25000]
                    + "\n\n已有观察：\n"
                    + "\n".join(observations[-12:]),
                },
            ]
            try:
                # data 是模型返回的 ReAct JSON。
                data = client.chat_json(messages)
                self._add_tokens(state, client.last_usage.total)
            except LlmError as exc:
                self._emit(state, "coder", "error", f"模型返回失败：{exc}")
                break
            # thought 是模型本轮判断，会进入事件流方便用户观察。
            thought = data.get("thought", "")
            if thought:
                self._emit(state, "coder", "thought", thought)
            # actions 是模型请求执行的工具动作列表。
            actions = data.get("actions") or []
            if not actions and data.get("done"):
                break
            for action in actions:
                # obs 是工具执行后的结构化观察。
                obs = self._do_action(action, fs, shell, git)
                observations.append(obs["text"])
                if obs.get("file"):
                    changes.append(str(obs["file"]))
                if obs.get("cmd"):
                    commands.append(str(obs["cmd"]))
                self._emit(state, "coder", "tool", obs["text"], data=obs)
            if data.get("done"):
                self._emit(state, "coder", "done", data.get("summary") or "Coding 阶段完成")
                break
        # unique_changes 去重并排序，保证最终结果稳定。
        unique_changes = sorted(dict.fromkeys(changes))
        self.memory.append(
            state["workdir"],
            state["session_id"],
            "coder",
            "react",
            "summary",
            f"变更文件：{unique_changes}",
            {"commands": commands},
        )
        return {**state, "changes": unique_changes, "commands": commands, "retry": retry + 1}

    def verifier(self, state: AgentState) -> AgentState:
        """验证智能体：根据项目文件选择测试命令并执行。"""

        self._emit(state, "verifier", "start", "验证智能体正在选择并运行测试")
        # fs 用于列出文件并做静态 Web 检查。
        fs = FsTool(state["workdir"])
        # shell 用于执行测试命令。
        shell = ShellTool(state["workdir"])
        # files 是当前项目文件列表。
        files = fs.list()
        # commands 保存待执行测试命令。
        commands: list[str] = []
        # tests 保存每条测试或静态检查的结构化结果。
        tests: list[dict[str, Any]] = []
        # ok 是整体验证状态，任意关键检查失败都会变为 False。
        ok = True

        # static_check 是静态 Web 项目的轻量接线检查结果。
        static_check = self._static_web_check(fs, files)
        if static_check:
            tests.append(static_check)
            ok = ok and bool(static_check.get("ok"))

        # Python/uv 项目优先跑 pytest。
        if "pyproject.toml" in files:
            commands.append("uv run pytest")
        elif any(file.endswith(".py") for file in files):
            # py_files 是所有 Python 文件拼接后的命令参数。
            py_files = " ".join(file for file in files if file.endswith(".py"))
            commands.append(f"python -m py_compile {py_files}")
        # Node 项目优先跑 npm test。
        if "package.json" in files:
            commands.append("npm test -- --run")
        elif any(file.endswith((".html", ".js", ".css")) for file in files):
            # 静态 Web 项目不真正启动阻塞服务器，只记录可用本地 HTTP 服务打开。
            commands.append("python -m http.server 0")

        if not commands:
            if not static_check:
                tests.append({"cmd": "静态检查", "ok": True, "out": "没有识别到可运行测试，已跳过。"})
        # 最多执行前三条命令，避免一次任务跑太多验证导致响应过慢。
        for cmd in commands[:3]:
            if cmd == "python -m http.server 0":
                tests.append({"cmd": cmd, "ok": True, "out": "检测到静态 Web 文件，可用本地 HTTP 服务打开。"})
                continue
            # res 是命令执行结果。
            res = shell.run(cmd, timeout=240)
            tests.append(res)
            ok = ok and bool(res.get("ok"))
            self._emit(state, "verifier", "test", f"{cmd} -> {'通过' if res.get('ok') else '失败'}", data=res)
        self.memory.append(state["workdir"], state["session_id"], "verifier", "test", "summary", f"测试结果：{ok}", {"tests": tests})
        return {**state, "tests": tests, "tests_ok": ok}

    def doc(self, state: AgentState) -> AgentState:
        """文档智能体：根据任务、仓库摘要、变更和测试结果写中文文档。"""

        self._emit(state, "doc", "start", "文档智能体正在生成或更新中文文档")
        # fs 是受 workdir 限制的文件写入工具。
        fs = FsTool(state["workdir"])
        # client 是 doc 智能体对应的模型客户端。
        client = self._client("doc", state)
        try:
            # data 应该是模型返回的文档 JSON，包含 path/content/summary。
            data = client.chat_json(
                [
                    {"role": "system", "content": DOC_PROMPT},
                    {
                        "role": "user",
                        "content": self._ctx_text(state)
                        + "\n\n仓库："
                        + json.dumps(state.get("repo", {}), ensure_ascii=False)[:15000]
                        + "\n\n变更："
                        + json.dumps(state.get("changes", []), ensure_ascii=False)
                        + "\n\n测试："
                        + json.dumps(state.get("tests", []), ensure_ascii=False),
                    },
                ]
            )
            self._add_tokens(state, client.last_usage.total)
        except LlmError:
            # 模型失败时使用基础 README 兜底，保证文档任务不完全失败。
            data = self._fallback_doc(state)
        # path 是模型建议写入的文档相对路径，默认 README.md。
        path = data.get("path") or "README.md"
        if Path(path).is_absolute():
            # 绝对路径会被强制改为 README.md，防止模型写出项目目录。
            path = "README.md"
        # written 是实际写入的相对路径。
        written = fs.write(path, data.get("content", "# 项目说明\n\n暂无内容。\n"))
        # repo 复制已有仓库摘要，并补充 doc_path 供最终结果展示。
        repo = dict(state.get("repo", {}))
        repo["doc_path"] = str((Path(state["workdir"]) / written).resolve())
        # changes 合并文档文件，并去重排序。
        changes = sorted(dict.fromkeys(list(state.get("changes", [])) + [written]))
        self._emit(state, "doc", "done", data.get("summary") or f"文档已写入 {written}")
        return {**state, "repo": repo, "changes": changes}

    def final(self, state: AgentState) -> AgentState:
        """最终节点：整理返回结果、写入最终 memory、触发记忆压缩。"""

        # ok 是任务最终成功状态，默认取 tests_ok。
        ok = bool(state.get("tests_ok", True))
        # summary 是给用户看的最终摘要；没有显式 final 时根据 ok 生成。
        summary = state.get("final") or ("任务完成" if ok else "任务完成，但验证存在失败")
        # result 是返回给前端的结构化 TaskResult 字典。
        result = {
            "ok": ok,
            "summary": summary,
            "files": state.get("changes", []),
            "commands": state.get("commands", []),
            "tests": state.get("tests", []),
            "plan_path": (state.get("plan") or {}).get("path"),
            "doc_path": (state.get("repo") or {}).get("doc_path"),
        }
        # 写入最终结果，历史会话恢复时会读取这条记录作为 Agent 回复。
        self.memory.append(state["workdir"], state["session_id"], "manager", "final", "result", summary, result)
        # ctx 存在说明 manager 正常构造过上下文，可以尝试按模型窗口压缩会话记忆。
        ctx = state.get("context")
        if ctx:
            # cfg 是 manager 对应模型配置，ctx 字段用于压缩阈值。
            cfg = self.model_store.for_agent("manager", state.get("model_id"))
            self.memory.maybe_compress(state["workdir"], state["session_id"], cfg.ctx)
        self._emit(state, "manager", "result", summary, data=result)
        return {**state, "result": result}

    def route_after_manager(self, state: AgentState) -> str:
        """返回 manager 后的下一跳节点。"""

        return state.get("route", "repo")

    def route_after_planner(self, state: AgentState) -> str:
        """返回 planner 后的下一跳，只允许 repo 或 final。"""

        # route 是 planner 节点写入的下一跳。
        route = state.get("route", "final")
        return route if route in {"repo", "final"} else "final"

    def _normalize_plan_questions(self, raw: Any) -> list[dict[str, Any]]:
        """把模型返回的 Plan 问题清洗成稳定格式。"""

        # questions 是模型返回的原始问题列表；不是列表就按空列表处理。
        questions = raw if isinstance(raw, list) else []
        # normalized 保存清洗后的问题，每个问题都包含 question/options/recommended/reason/allow_custom。
        normalized: list[dict[str, Any]] = []
        for item in questions[:3]:
            if not isinstance(item, dict):
                continue
            # question 是问题正文，异常时使用兜底问题。
            question = self._clean_plan_text(item.get("question"), "需要确认哪一项执行策略？")
            # options_raw 是模型返回的原始选项列表。
            options_raw = item.get("options")
            # options 是清洗后的选项，最多保留 4 个。
            options = [self._clean_plan_text(option, "") for option in options_raw] if isinstance(options_raw, list) else []
            options = [option for option in options if option][:4]
            if len(options) < 2:
                options = ["按推荐方案执行", "继续补充细节"]
            # recommended 是推荐选项；如果不在选项列表里，就强制使用第一个选项。
            recommended = self._clean_plan_text(item.get("recommended"), options[0])
            if recommended not in options:
                recommended = options[0]
            # reason 是推荐理由，异常时给出通用理由。
            reason = self._clean_plan_text(item.get("reason"), "该选项能减少来回确认，便于先形成可执行计划。")
            normalized.append(
                {
                    "question": question,
                    "options": options,
                    "recommended": recommended,
                    "reason": reason,
                    "allow_custom": bool(item.get("allow_custom", True)),
                }
            )
        if normalized:
            return normalized
        # 模型没有返回任何有效问题时，返回一个默认问题。
        return [
            {
                "question": "是否直接按默认工程架构执行？",
                "options": ["直接执行", "继续细化计划"],
                "recommended": "直接执行",
                "reason": "当前信息已经足够先形成初版计划，后续仍可根据你的补充调整。",
                "allow_custom": True,
            }
        ]

    def _format_plan_questions(self, questions: list[dict[str, Any]]) -> str:
        """把结构化 Plan 问题格式化成用户可读文本。"""

        # lines 保存最终输出的每一行。
        lines = [
            "Plan 模式需要你先做几个选择：",
            "",
            "你可以直接回复选项编号，例如：1A，2B；也可以写自定义回答。",
        ]
        # letters 用于把选项编号转换成 A/B/C/D。
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # idx 是第几个问题，item 是问题结构。
        for idx, item in enumerate(questions, start=1):
            lines.extend(["", f"{idx}. {item['question']}"])
            # option_idx 是选项下标，option 是选项文本。
            for option_idx, option in enumerate(item["options"]):
                # mark 标记推荐选项。
                mark = "（推荐）" if option == item["recommended"] else ""
                lines.append(f"   {letters[option_idx]}. {option}{mark}")
            lines.append(f"   推荐理由：{item['reason']}")
            if item.get("allow_custom", True):
                lines.append("   允许自定义回答。")
        return "\n".join(lines)

    def _clean_plan_text(self, value: Any, fallback: str) -> str:
        """清洗 Plan 问题中的单个文本字段。"""

        # text 是模型返回字段转成字符串后的内容。
        text = str(value or "").strip()
        # bad_markers 是不应该展示给用户的错误或 JSON 残片特征。
        bad_markers = ["Expecting", "JSONDecodeError", "模型没有返回合法 JSON", "\\n", "{", "}", "[", "]"]
        if not text or any(marker in text for marker in bad_markers):
            text = fallback
        # 合并多余空白，避免选项展示很乱。
        text = re.sub(r"\s+", " ", text)
        return self._trim(text, 120)

    def _latest_pending_plan(self, workdir: str, session_id: str) -> dict[str, Any] | None:
        """从会话记忆中查找最近仍未完成的 pending_plan。"""

        # 倒序查找最近 200 条记录，优先看到最新 Plan 状态。
        for rec in reversed(self.memory.read_session(workdir, session_id, limit=200)):
            # 如果已经有 plan_done 或 plan_cancelled，说明旧 pending_plan 不再有效。
            if rec.get("ag") == "planner" and rec.get("tl") == "state" and rec.get("k") in {"plan_done", "plan_cancelled"}:
                return None
            if rec.get("ag") == "planner" and rec.get("tl") == "state" and rec.get("k") == "pending_plan":
                # meta 是 pending_plan 的元数据副本，避免直接修改原记录。
                meta = dict(rec.get("m") or {})
                # memory_id 记录 pending_plan 来自哪条 memory，便于调试。
                meta["memory_id"] = rec.get("id")
                return meta
        return None

    def _latest_saved_plan(self, workdir: str, session_id: str) -> dict[str, Any] | None:
        """从会话记忆中查找最近生成的计划。"""

        for rec in reversed(self.memory.read_session(workdir, session_id, limit=200)):
            if rec.get("ag") == "planner" and rec.get("tl") == "state" and rec.get("k") == "plan_done":
                # meta 是计划元数据，包含 path、markdown、goal。
                meta = dict(rec.get("m") or {})
                meta["status"] = "plan"
                return meta
        return None

    def _is_plan_cancel(self, text: str) -> bool:
        """判断用户是否想取消当前 Plan 流程。"""

        # cleaned 去掉空白并转小写，兼容“取消 plan”等写法。
        cleaned = re.sub(r"\s+", "", text).lower()
        return any(word in cleaned for word in ["取消plan", "取消计划", "退出plan", "退出计划", "不做了", "重新开始"])

    def _is_execute_plan_text(self, text: str) -> bool:
        """判断用户是否在用自然语言确认执行计划。"""

        # cleaned 去掉空白并转小写，兼容中文和英文触发词。
        cleaned = re.sub(r"\s+", "", text).lower()
        return cleaned in {"执行计划", "开始执行", "按计划执行", "executeplan", "runplan"}

    def _should_treat_as_plan_reply(self, text: str, pending: dict[str, Any], state: AgentState) -> bool:
        """判断用户输入是否应该视为上一轮 Plan 问题的回答。"""

        # Plan 模式仍开启或显式执行计划时，优先认为它属于 Plan 流程。
        if state.get("plan_mode") or state.get("execute_plan"):
            return True
        # 输入看起来像 1A/2B 或包含选项文本时，认为是 Plan 回答。
        if self._looks_like_plan_answer(text, pending):
            return True
        # 输入明显像新任务时，不要误当成 Plan 回答。
        if self._looks_like_new_task(text):
            return False
        # 短输入一般更可能是回答上一轮选择题。
        return len(text.strip()) <= 500

    def _looks_like_plan_answer(self, text: str, pending: dict[str, Any]) -> bool:
        """用规则判断文本是否像 Plan 选择题回答。"""

        # cleaned 是去除两端空白后的原始输入。
        cleaned = text.strip()
        if re.search(r"(?<!\d)\d+\s*[\.\-:：]?\s*[A-Za-z]", cleaned):
            return True
        if re.search(r"(都|全部|全都).*(推荐|默认)", cleaned):
            return True
        # questions 是 pending_plan 中保存的上一轮问题。
        questions = pending.get("questions") or []
        # options 展平所有选项文本，用于判断用户是否直接复制了选项内容。
        options = [str(option) for item in questions if isinstance(item, dict) for option in item.get("options", [])]
        return any(option and option in cleaned for option in options)

    def _looks_like_new_task(self, text: str) -> bool:
        """判断输入是否更像一个新任务，而不是 Plan 选项回答。"""

        # lower 是小写文本，兼容英文关键词。
        lower = text.lower()
        # keywords 是创建、修改、解释等新任务常见动词。
        keywords = ["创建", "实现", "开发", "新建", "编写", "修改", "修复", "解释", "生成文档", "重构", "build", "create", "fix"]
        return len(text.strip()) > 8 and any(word in lower for word in keywords)

    def _build_plan_reply(self, text: str, pending: dict[str, Any]) -> dict[str, Any]:
        """把用户 Plan 回复解析成结构化答案。"""

        # questions 是上一轮保存的 Plan 问题列表。
        questions = pending.get("questions") or []
        # answers 保存每个问题解析出的答案。
        answers: list[dict[str, str]] = []
        # use_recommended 表示用户是否说“都选推荐/默认”。
        use_recommended = bool(re.search(r"(都|全部|全都).*(推荐|默认)", text))
        for idx, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                continue
            # options 是当前问题的选项文本列表。
            options = [str(option) for option in item.get("options", [])]
            # selected 保存最终选中的选项文本。
            selected = ""
            # source 记录答案来源，便于调试是 option、recommended、default 还是 custom。
            source = "custom"
            if use_recommended:
                selected = str(item.get("recommended") or (options[0] if options else ""))
                source = "recommended"
            else:
                # match 识别类似 1A、1.A、1：A 的回答。
                match = re.search(rf"(?<!\d){idx}\s*[\.\-:：]?\s*([A-Za-z])", text)
                if match:
                    # option_idx 是 A/B/C/D 转成的选项下标。
                    option_idx = ord(match.group(1).upper()) - ord("A")
                    if 0 <= option_idx < len(options):
                        selected = options[option_idx]
                        source = "option"
                if not selected:
                    # 如果没有编号，就尝试直接匹配选项文本。
                    selected = next((option for option in options if option and option in text), "")
            if not selected:
                # 仍未识别时使用推荐项或第一个选项兜底。
                selected = str(item.get("recommended") or (options[0] if options else text.strip()))
                source = "default"
            answers.append({"question": str(item.get("question", "")), "answer": selected, "source": source})
        if not answers:
            # 没有可解析的问题时，把整段输入作为自定义补充。
            answers.append({"question": "自定义补充", "answer": text.strip(), "source": "custom"})
        return {"raw": text, "answers": answers}

    def _compose_pending_plan_text(self, pending: dict[str, Any], reply: dict[str, Any]) -> str:
        """把原始需求、上一轮问题和用户答案合成新的 Plan 输入。"""

        # lines 保存最终拼接给 Plan 模型的多行文本。
        lines = [
            "原始需求：",
            str(pending.get("goal") or ""),
            "",
            "上一轮 Plan 问题与用户回答：",
        ]
        # idx 是回答序号，item 是单个问题的结构化回答。
        for idx, item in enumerate(reply.get("answers", []), start=1):
            lines.append(f"{idx}. 问题：{item.get('question', '')}")
            lines.append(f"   用户回答：{item.get('answer', '')}")
        if reply.get("raw"):
            # raw 保留用户原始回复，避免解析过程丢失自定义信息。
            lines.extend(["", f"用户原始回复：{reply['raw']}"])
        lines.append("")
        lines.append("请基于原始需求和用户回答继续 Plan 流程；信息足够时生成可执行计划。")
        return "\n".join(lines)

    def _compose_saved_plan_text(self, saved_plan: dict[str, Any], text: str) -> str:
        """把已保存计划和用户确认输入合并成 Coding 可执行任务文本。"""

        return "\n".join(
            [
                "用户确认执行已保存计划。",
                f"用户本轮输入：{text}",
                "",
                "计划内容：",
                str(saved_plan.get("markdown") or ""),
            ]
        )

    def _plan_goal(self, state: AgentState) -> str:
        """获取当前 Plan 流程的原始目标。"""

        # pending 是当前状态里的 pending_plan，优先使用它保存的原始 goal。
        pending = state.get("pending_plan") or {}
        if pending.get("goal"):
            return str(pending["goal"])
        # ctx 是管理者构造的上下文包，其 goal 通常是当前任务目标。
        ctx = state.get("context")
        if ctx:
            return ctx.goal
        return state["text"]

    def _fallback_plan_markdown(self, state: AgentState) -> str:
        """生成模型异常时使用的兜底 Markdown 计划。"""

        # answers 是用户已确认的 Plan 选项。
        answers = state.get("plan_answers", [])
        # answer_lines 是写入计划中的“已确认选择”列表。
        answer_lines = "\n".join(f"- {item.get('question', '')}：{item.get('answer', '')}" for item in answers) or "- 暂无补充选择"
        return f"""# 执行计划

## 目标

{self._plan_goal(state)}

## 已确认选择

{answer_lines}

## 执行步骤

1. 读取项目目录，识别技术栈、入口文件、脚本和现有约束。
2. 根据目标拆分实现任务，优先最小改动并复用项目已有结构。
3. 编写或修改代码，并把变更写入磁盘。
4. 运行适合当前项目的测试或静态检查。
5. 根据验证结果修复问题。
6. 生成或更新中文文档，说明启动方式、主要功能和验证结果。
"""

    def route_after_repo(self, state: AgentState) -> str:
        """返回 repo 节点之后的下一跳。"""

        return state.get("after_repo", "coder")

    def route_after_verifier(self, state: AgentState) -> str:
        """返回 verifier 节点之后的下一跳。"""

        # 测试失败且重试次数不足时回到 coder 修复。
        if not state.get("tests_ok", True) and int(state.get("retry", 0)) < 2:
            return "coder"
        # 测试通过且任务需要文档时进入 doc。
        if state.get("tests_ok", True) and state.get("after_verify") == "doc":
            return "doc"
        return "final"

    def _classify(self, state: AgentState) -> dict[str, Any]:
        """对用户输入进行任务分类，优先用确定性规则，兜底才调用模型。"""

        # text 是当前任务文本。
        text = state["text"]
        # lower 是小写文本，便于英文关键词匹配。
        lower = text.lower()
        if state.get("plan_mode"):
            return {"task_type": "plan_gen", "need_repo": True, "need_clarify": False, "reason": "Plan 模式已开启"}
        # direct_reply 处理问候、感谢等无需读仓库的输入。
        direct_reply = self._direct_reply(text)
        if direct_reply:
            return {
                "task_type": "direct",
                "need_repo": False,
                "need_code": False,
                "need_doc": False,
                "need_clarify": False,
                "reason": "普通对话，不需要调用仓库、Coding、验证或文档智能体",
                "direct_reply": direct_reply,
            }
        # code_intent 表示用户有创建/实现/开发倾向。
        code_intent = any(word in lower for word in ["创建", "实现", "开发", "新建", "编写", "做一个", "build", "create"])
        # product_intent 表示用户目标是系统、应用、页面、接口等可交付物。
        product_intent = any(word in lower for word in ["系统", "应用", "app", "网页", "页面", "接口", "功能", "模块", "工具"])
        # doc_intent 表示用户明确要求文档。
        doc_intent = any(word in lower for word in ["文档", "readme", "说明文档", "教程", "部署说明", "接口说明"])
        if code_intent and product_intent:
            return {
                "task_type": "code_gen",
                "need_repo": True,
                "need_code": True,
                "need_doc": doc_intent,
                "need_clarify": False,
                "reason": "用户需要创建可运行代码",
            }
        if any(word in lower for word in ["解释", "为什么", "报错", "error", "traceback"]):
            return {"task_type": "code_explain", "need_repo": True, "need_code": False, "need_doc": False, "need_clarify": False, "reason": "用户需要解释或排错"}
        if doc_intent:
            return {"task_type": "doc_gen", "need_repo": True, "need_code": False, "need_doc": True, "need_clarify": False, "reason": "用户需要生成文档"}
        if any(word in lower for word in ["修改", "修复", "bug", "重构", "适配"]):
            return {"task_type": "code_mod", "need_repo": True, "need_code": True, "need_doc": doc_intent, "need_clarify": False, "reason": "用户需要修改代码"}
        try:
            # client 是管理者模型，用于规则无法覆盖的模糊输入。
            client = self._client("manager", state)
            # data 是模型返回的分类 JSON。
            data = client.chat_json(
                [{"role": "system", "content": MANAGER_PROMPT}, {"role": "user", "content": text}],
                temperature=0,
            )
            self._add_tokens(state, client.last_usage.total)
            return data
        except Exception:
            # 管理者模型异常时降级为普通回答，避免误触发仓库读取和写代码。
            return {"task_type": "general_answer", "need_repo": False, "need_code": False, "need_doc": False, "need_clarify": False, "reason": "我在，可以继续告诉我你要处理的代码任务或问题。"}

    def _flow_for(self, state: AgentState, classification: dict[str, Any]) -> tuple[str, str, str]:
        """把分类结果转换成 LangGraph 三段路由。"""

        # task_type 是管理者分类出的任务类型。
        task_type = classification.get("task_type", "general_answer")
        if task_type == "plan_gen":
            return "planner", "final", "final"
        if state.get("plan_mode") and not state.get("execute_plan"):
            return "planner", "final", "final"
        if task_type == "direct":
            return "final", "final", "final"
        if task_type == "general_answer":
            return "answer", "final", "final"
        if task_type == "doc_gen":
            return "repo", "doc", "final"
        if task_type == "code_explain":
            return ("repo", "answer", "final") if classification.get("need_repo", True) else ("answer", "final", "final")
        if task_type in {"code_gen", "code_mod"}:
            # after_verify 决定验证通过后是否进入文档智能体。
            after_verify = "doc" if bool(classification.get("need_doc")) else "final"
            return "repo", "coder", after_verify
        return "final", "final", "final"

    def _direct_reply(self, text: str) -> str | None:
        """处理不需要调用模型、不需要读仓库的直接回复。"""

        # cleaned 去掉空白并转小写，方便匹配简短口语输入。
        cleaned = re.sub(r"\s+", "", text).lower()
        if not cleaned:
            return "我在。你可以直接告诉我要创建、修改、解释还是生成文档。"
        # greetings 是普通问候集合。
        greetings = {"你好", "您好", "hi", "hello", "在吗", "在不在", "嗨", "hey"}
        # thanks 是感谢类输入集合。
        thanks = {"谢谢", "谢了", "感谢", "多谢", "thanks", "thankyou"}
        if cleaned in greetings:
            return "你好，我在。你可以直接描述要创建、修改、解释或生成文档的任务；普通问候不会触发仓库读取或写文件。"
        if cleaned in thanks:
            return "不客气。"
        if cleaned in {"你是谁", "你能做什么", "能做什么"}:
            return "我是这个本地编程 agent 的管理入口。简单聊天我会直接回复；涉及代码、文档、计划或排错时，我会按需调用对应智能体。"
        return None

    def _detect_stack(self, files: list[str]) -> list[str]:
        """根据文件列表粗略识别项目技术栈。"""

        # stack 保存识别出的技术栈标签。
        stack: list[str] = []
        if "pyproject.toml" in files:
            stack.append("Python/uv")
        if "package.json" in files:
            stack.append("Node")
        if any(file.endswith(".tsx") for file in files):
            stack.append("React/TypeScript")
        if any(file.endswith(".py") for file in files):
            stack.append("Python")
        if any(file.endswith(".html") for file in files):
            stack.append("静态 Web")
        return stack

    def _do_action(self, action: dict[str, Any], fs: FsTool, shell: ShellTool, git: GitTool) -> dict[str, Any]:
        """执行 Coding 智能体返回的单个工具动作。"""

        # tool 是模型请求调用的工具名称。
        tool = action.get("tool")
        try:
            if tool == "write_file":
                # rel 是实际写入文件的项目相对路径。
                rel = fs.write(str(action["path"]), str(action.get("content", "")))
                return {"ok": True, "text": f"写入文件：{rel}", "file": rel}
            if tool == "append_file":
                # rel 是实际追加文件的项目相对路径。
                rel = fs.append(str(action["path"]), str(action.get("content", "")))
                return {"ok": True, "text": f"追加文件：{rel}", "file": rel}
            if tool == "read_file":
                # rel 是模型要求读取的项目相对路径。
                rel = str(action["path"])
                return {"ok": True, "text": f"读取文件：{rel}\n{fs.read(rel, 8000)}"}
            if tool == "list_files":
                return {"ok": True, "text": "文件列表：\n" + "\n".join(fs.list())}
            if tool == "run_command":
                # cmd 是模型要求执行的命令字符串，ShellTool 会做危险命令检查。
                cmd = str(action.get("cmd", ""))
                # res 是命令执行结果。
                res = shell.run(cmd)
                return {"ok": res.get("ok"), "text": f"运行命令：{cmd}\n{res.get('out','')}\n{res.get('err','')}", "cmd": cmd, "result": res}
            if tool == "git_status":
                return {"ok": True, "text": "Git 状态：\n" + git.status()}
            return {"ok": False, "text": f"未知工具：{tool}"}
        except Exception as exc:
            return {"ok": False, "text": f"工具执行失败：{tool} -> {exc}"}

    def _static_web_check(self, fs: FsTool, files: list[str]) -> dict[str, Any] | None:
        """轻量检查静态 Web 项目中常见的 HTML/JS 接线错误。

        它不是浏览器测试的替代品，但能抓住 `getElementById` 指向不存在元素、
        以及内联 `onclick` 调用未定义函数这类高频错误。
        """
        if "index.html" not in files:
            return None
        # html 是入口 HTML 内容。
        html = fs.read("index.html", 40000)
        # js_text 是所有 JS 文件拼接后的内容，用于静态搜索 DOM 引用和函数定义。
        js_text = "\n".join(fs.read(file, 40000) for file in files if file.endswith(".js"))
        # ids 收集 HTML 和 JS 动态创建出的元素 id。
        ids = set(re.findall(r'id=["\']([^"\']+)["\']', html))
        ids.update(re.findall(r"\.id\s*=\s*[\"']([^\"']+)[\"']", js_text))
        ids.update(re.findall(r"setAttribute\([\"']id[\"']\s*,\s*[\"']([^\"']+)[\"']\)", js_text))
        # requested_ids 收集 JS 中 getElementById 访问的 id。
        requested_ids = set(re.findall(r"getElementById\([\"']([^\"']+)[\"']\)", js_text))
        # missing_ids 是 JS 访问了但 HTML/JS 中没有定义的 id。
        missing_ids = sorted(requested_ids - ids)

        # inline_calls 收集 HTML 内联事件里调用的函数名，例如 onclick="foo()"。
        inline_calls = set(re.findall(r'on\w+=["\']([A-Za-z_$][\w$]*)\s*\(', html))
        # defined_funcs 收集 JS 中通过 function 定义的函数名。
        defined_funcs = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", js_text))
        # 同时把挂到 window 上的函数视为可被内联事件调用。
        defined_funcs.update(re.findall(r"window\.([A-Za-z_$][\w$]*)\s*=", js_text))
        # missing_funcs 是 HTML 调用了但 JS 中没有定义的函数。
        missing_funcs = sorted(inline_calls - defined_funcs)

        # issues 保存所有静态检查问题文本。
        issues = []
        if missing_ids:
            issues.append("缺失元素 id：" + "、".join(missing_ids))
        if missing_funcs:
            issues.append("缺失内联事件函数：" + "、".join(missing_funcs))
        return {
            "cmd": "静态 Web 接线检查",
            "ok": not issues,
            "out": "通过" if not issues else "；".join(issues),
            "missing_ids": missing_ids,
            "missing_funcs": missing_funcs,
        }

    def _client(self, agent: str, state: AgentState) -> LlmClient:
        """根据智能体名称创建对应模型客户端。"""

        return LlmClient(self.model_store.for_agent(agent, state.get("model_id")))

    def _ctx_text(self, state: AgentState) -> str:
        """把当前上下文包序列化成模型输入文本。"""

        # ctx 是管理者构造的 ContextPackage。
        ctx = state.get("context")
        if not ctx:
            return state["text"]
        return ctx.model_dump_json(indent=2)

    def _fallback_doc(self, state: AgentState) -> dict[str, str]:
        """模型文档生成失败时使用的基础 README 内容。"""

        # files 是变更文件列表的 Markdown 项。
        files = "\n".join(f"- `{file}`" for file in state.get("changes", [])) or "- 暂无文件变更"
        # tests 是验证结果列表的 Markdown 项。
        tests = "\n".join(f"- `{item.get('cmd', '检查')}`：{'通过' if item.get('ok') else '失败'}" for item in state.get("tests", []))
        return {
            "path": "README.md",
            "summary": "已生成基础 README",
            "content": f"""# 项目说明

本项目由多智能体编程系统生成或维护。

## 功能

{state['text']}

## 文件变更

{files}

## 验证

{tests or '- 暂无可运行测试'}
""",
        }

    def _emit(self, state: AgentState, agent: str, kind: str, msg: str, tokens: int = 0, data: dict[str, Any] | None = None) -> None:
        """记录并向外发送一条 AgentEvent。"""

        # event_id 是本次任务内递增编号。
        self.event_id += 1
        # event 是前端事件流消费的结构化事件。
        event = AgentEvent(
            id=self.event_id,
            ts=datetime.now(UTC).isoformat(),
            agent=agent,
            kind=kind,
            msg=msg,
            tokens=tokens,
            data=data or {},
        )
        # 同步把事件写入会话记忆，便于中断后查看执行到哪一步。
        self.memory.append(state["workdir"], state["session_id"], agent, "event", kind, msg, {"tokens": tokens, "data": data or {}})
        if self.emit_cb:
            self.emit_cb(event)

    def _add_tokens(self, state: AgentState, tokens: int) -> None:
        """累计 token 估算，并发出 usage 事件。"""

        state["tokens"] = int(state.get("tokens", 0)) + tokens
        self._emit(state, "llm", "usage", f"本次模型调用约消耗 {tokens} token", tokens=tokens)

    def _trim(self, text: str, max_chars: int) -> str:
        """保留文本最后 max_chars 个字符，常用于限制 memory 进入上下文的长度。"""

        return text if len(text) <= max_chars else text[-max_chars:]


def new_session_id() -> str:
    """生成短会话 id，便于文件名和前端展示。"""
    # uuid4().hex[:12] 生成 12 位短 id，足够本地会话使用且比完整 uuid 更易展示。
    return uuid.uuid4().hex[:12]
