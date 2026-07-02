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

Emit = Callable[[AgentEvent], None]


class AgentGraph:
    """多智能体编排图。

    每次任务创建一个实例，这样事件回调、token 统计和重试次数不会串到别的会话。
    """

    def __init__(self, model_store: ModelStore, memory: MemoryStore, emit: Emit | None = None):
        self.model_store = model_store
        self.memory = memory
        self.emit_cb = emit
        self.event_id = 0
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
        state: AgentState = {
            "session_id": session_id,
            "workdir": workdir,
            "text": text,
            "plan_mode": plan_mode,
            "execute_plan": execute_plan,
            "model_id": model_id,
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        }
        final = self.graph.invoke(state)
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
        graph = StateGraph(AgentState)
        graph.add_node("manager", self.manager)
        graph.add_node("planner", self.planner)
        graph.add_node("repo", self.repo)
        graph.add_node("answer", self.answer)
        graph.add_node("coder", self.coder)
        graph.add_node("verifier", self.verifier)
        graph.add_node("doc", self.doc)
        graph.add_node("final", self.final)

        graph.add_edge(START, "manager")
        graph.add_conditional_edges(
            "manager",
            self.route_after_manager,
            {"planner": "planner", "repo": "repo", "answer": "answer", "final": "final"},
        )
        graph.add_conditional_edges(
            "planner",
            self.route_after_planner,
            {"repo": "repo", "final": "final"},
        )
        graph.add_conditional_edges(
            "repo",
            self.route_after_repo,
            {"coder": "coder", "doc": "doc", "answer": "answer", "final": "final"},
        )
        graph.add_edge("answer", "final")
        graph.add_edge("coder", "verifier")
        graph.add_conditional_edges(
            "verifier",
            self.route_after_verifier,
            {"coder": "coder", "doc": "doc", "final": "final"},
        )
        graph.add_edge("doc", "final")
        graph.add_edge("final", END)
        return graph.compile()

    def manager(self, state: AgentState) -> AgentState:
        workdir = state["workdir"]
        session_id = state["session_id"]
        interrupted = self.memory.interrupted(workdir, session_id)
        self._emit(state, "manager", "start", "管理者正在分类任务并构造上下文包")

        pending_plan = self._latest_pending_plan(workdir, session_id)
        saved_plan = self._latest_saved_plan(workdir, session_id)
        classification: dict[str, Any]
        if pending_plan and self._is_plan_cancel(state["text"]):
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
            reply = self._build_plan_reply(state["text"], pending_plan)
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
            classification = self._classify(state)
        task_type = classification.get("task_type", "code_gen")
        route, after_repo, after_verify = self._flow_for(state, classification)

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
            ctx.recent.append("检测到上次会话可能异常中断，本次会继续以当前磁盘状态为准。")
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
        self._emit(state, "planner", "start", "Plan 智能体正在生成问题或可执行计划")
        client = self._client("planner", state)
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": self._ctx_text(state)},
        ]
        try:
            data = client.chat_json(messages)
            self._add_tokens(state, client.last_usage.total)
        except LlmError as exc:
            self._emit(state, "planner", "error", f"Plan 模型返回异常，已使用默认澄清问题：{str(exc)[:120]}")
            if state.get("pending_plan"):
                data = {"status": "plan", "title": "默认执行计划", "markdown": self._fallback_plan_markdown(state)}
            else:
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
            data = {**data, "questions": self._normalize_plan_questions(data.get("questions"))}
        if data.get("status") == "plan" or state.get("execute_plan"):
            md = data.get("markdown") or self._fallback_plan_markdown(state)
            plan_dir = Path(state["workdir"]) / "docs" / "plans"
            plan_dir.mkdir(parents=True, exist_ok=True)
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
            route = "repo" if state.get("execute_plan") else "final"
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
        summary = self._format_plan_questions(data.get("questions", []))
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
        self._emit(state, "repo", "start", "仓库读取智能体正在识别目录结构和技术栈")
        fs = FsTool(state["workdir"])
        files = fs.list()
        snippets: dict[str, str] = {}
        priority = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "src/main.py",
            "main.py",
            "app.py",
            "index.html",
        ]
        for rel in priority:
            if rel in files:
                snippets[rel] = fs.read(rel, 8000)
        for rel in files[:30]:
            if rel not in snippets and rel.endswith((".py", ".ts", ".tsx", ".js", ".md", ".json", ".html", ".css")):
                snippets[rel] = fs.read(rel, 4000)
        stack = self._detect_stack(files)
        repo = {"files": files, "snippets": snippets, "stack": stack, "empty": len(files) == 0}
        self._emit(state, "repo", "summary", f"识别到 {len(files)} 个文件，技术栈：{', '.join(stack) or '空项目'}")
        self.memory.append(state["workdir"], state["session_id"], "repo", "fs", "summary", f"文件数：{len(files)}，技术栈：{stack}")
        ctx = state["context"]
        ctx.relevant_files = files[:80]
        return {**state, "repo": repo, "context": ctx}

    def answer(self, state: AgentState) -> AgentState:
        self._emit(state, "answer", "start", "答疑智能体正在生成直接回答")
        client = self._client("manager", state)
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
            text = client.chat(messages, temperature=0.2).strip()
            self._add_tokens(state, client.last_usage.total)
        except LlmError as exc:
            text = f"我现在无法完成解释，因为模型调用失败：{exc}"
        self.memory.append(state["workdir"], state["session_id"], "answer", "llm", "summary", text)
        return {**state, "final": text}

    def coder(self, state: AgentState) -> AgentState:
        retry = int(state.get("retry", 0))
        self._emit(state, "coder", "start", f"Coding 智能体开始 ReAct 执行，第 {retry + 1} 轮")
        fs = FsTool(state["workdir"])
        shell = ShellTool(state["workdir"])
        git = GitTool(state["workdir"])
        client = self._client("coder", state)
        observations: list[str] = []
        changes = list(state.get("changes", []))
        commands = list(state.get("commands", []))
        if state.get("tests") and not state.get("tests_ok", True):
            observations.append("上一轮测试失败：\n" + json.dumps(state["tests"], ensure_ascii=False, indent=2))

        for _step in range(1, 7):
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
                data = client.chat_json(messages)
                self._add_tokens(state, client.last_usage.total)
            except LlmError as exc:
                self._emit(state, "coder", "error", f"模型返回失败：{exc}")
                break
            thought = data.get("thought", "")
            if thought:
                self._emit(state, "coder", "thought", thought)
            actions = data.get("actions") or []
            if not actions and data.get("done"):
                break
            for action in actions:
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
        self._emit(state, "verifier", "start", "验证智能体正在选择并运行测试")
        fs = FsTool(state["workdir"])
        shell = ShellTool(state["workdir"])
        files = fs.list()
        commands: list[str] = []
        tests: list[dict[str, Any]] = []
        ok = True

        static_check = self._static_web_check(fs, files)
        if static_check:
            tests.append(static_check)
            ok = ok and bool(static_check.get("ok"))

        if "pyproject.toml" in files:
            commands.append("uv run pytest")
        elif any(file.endswith(".py") for file in files):
            py_files = " ".join(file for file in files if file.endswith(".py"))
            commands.append(f"python -m py_compile {py_files}")
        if "package.json" in files:
            commands.append("npm test -- --run")
        elif any(file.endswith((".html", ".js", ".css")) for file in files):
            commands.append("python -m http.server 0")

        if not commands:
            if not static_check:
                tests.append({"cmd": "静态检查", "ok": True, "out": "没有识别到可运行测试，已跳过。"})
        for cmd in commands[:3]:
            if cmd == "python -m http.server 0":
                tests.append({"cmd": cmd, "ok": True, "out": "检测到静态 Web 文件，可用本地 HTTP 服务打开。"})
                continue
            res = shell.run(cmd, timeout=240)
            tests.append(res)
            ok = ok and bool(res.get("ok"))
            self._emit(state, "verifier", "test", f"{cmd} -> {'通过' if res.get('ok') else '失败'}", data=res)
        self.memory.append(state["workdir"], state["session_id"], "verifier", "test", "summary", f"测试结果：{ok}", {"tests": tests})
        return {**state, "tests": tests, "tests_ok": ok}

    def doc(self, state: AgentState) -> AgentState:
        self._emit(state, "doc", "start", "文档智能体正在生成或更新中文文档")
        fs = FsTool(state["workdir"])
        client = self._client("doc", state)
        try:
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
            data = self._fallback_doc(state)
        path = data.get("path") or "README.md"
        if Path(path).is_absolute():
            path = "README.md"
        written = fs.write(path, data.get("content", "# 项目说明\n\n暂无内容。\n"))
        repo = dict(state.get("repo", {}))
        repo["doc_path"] = str((Path(state["workdir"]) / written).resolve())
        changes = sorted(dict.fromkeys(list(state.get("changes", [])) + [written]))
        self._emit(state, "doc", "done", data.get("summary") or f"文档已写入 {written}")
        return {**state, "repo": repo, "changes": changes}

    def final(self, state: AgentState) -> AgentState:
        ok = bool(state.get("tests_ok", True))
        summary = state.get("final") or ("任务完成" if ok else "任务完成，但验证存在失败")
        result = {
            "ok": ok,
            "summary": summary,
            "files": state.get("changes", []),
            "commands": state.get("commands", []),
            "tests": state.get("tests", []),
            "plan_path": (state.get("plan") or {}).get("path"),
            "doc_path": (state.get("repo") or {}).get("doc_path"),
        }
        self.memory.append(state["workdir"], state["session_id"], "manager", "final", "result", summary, result)
        ctx = state.get("context")
        if ctx:
            cfg = self.model_store.for_agent("manager", state.get("model_id"))
            self.memory.maybe_compress(state["workdir"], state["session_id"], cfg.ctx)
        self._emit(state, "manager", "result", summary, data=result)
        return {**state, "result": result}

    def route_after_manager(self, state: AgentState) -> str:
        return state.get("route", "repo")

    def route_after_planner(self, state: AgentState) -> str:
        route = state.get("route", "final")
        return route if route in {"repo", "final"} else "final"

    def _normalize_plan_questions(self, raw: Any) -> list[dict[str, Any]]:
        questions = raw if isinstance(raw, list) else []
        normalized: list[dict[str, Any]] = []
        for item in questions[:3]:
            if not isinstance(item, dict):
                continue
            question = self._clean_plan_text(item.get("question"), "需要确认哪一项执行策略？")
            options_raw = item.get("options")
            options = [self._clean_plan_text(option, "") for option in options_raw] if isinstance(options_raw, list) else []
            options = [option for option in options if option][:4]
            if len(options) < 2:
                options = ["按推荐方案执行", "继续补充细节"]
            recommended = self._clean_plan_text(item.get("recommended"), options[0])
            if recommended not in options:
                recommended = options[0]
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
        lines = [
            "Plan 模式需要你先做几个选择：",
            "",
            "你可以直接回复选项编号，例如：1A，2B；也可以写自定义回答。",
        ]
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx, item in enumerate(questions, start=1):
            lines.extend(["", f"{idx}. {item['question']}"])
            for option_idx, option in enumerate(item["options"]):
                mark = "（推荐）" if option == item["recommended"] else ""
                lines.append(f"   {letters[option_idx]}. {option}{mark}")
            lines.append(f"   推荐理由：{item['reason']}")
            if item.get("allow_custom", True):
                lines.append("   允许自定义回答。")
        return "\n".join(lines)

    def _clean_plan_text(self, value: Any, fallback: str) -> str:
        text = str(value or "").strip()
        bad_markers = ["Expecting", "JSONDecodeError", "模型没有返回合法 JSON", "\\n", "{", "}", "[", "]"]
        if not text or any(marker in text for marker in bad_markers):
            text = fallback
        text = re.sub(r"\s+", " ", text)
        return self._trim(text, 120)

    def _latest_pending_plan(self, workdir: str, session_id: str) -> dict[str, Any] | None:
        for rec in reversed(self.memory.read_session(workdir, session_id, limit=200)):
            if rec.get("ag") == "planner" and rec.get("tl") == "state" and rec.get("k") in {"plan_done", "plan_cancelled"}:
                return None
            if rec.get("ag") == "planner" and rec.get("tl") == "state" and rec.get("k") == "pending_plan":
                meta = dict(rec.get("m") or {})
                meta["memory_id"] = rec.get("id")
                return meta
        return None

    def _latest_saved_plan(self, workdir: str, session_id: str) -> dict[str, Any] | None:
        for rec in reversed(self.memory.read_session(workdir, session_id, limit=200)):
            if rec.get("ag") == "planner" and rec.get("tl") == "state" and rec.get("k") == "plan_done":
                meta = dict(rec.get("m") or {})
                meta["status"] = "plan"
                return meta
        return None

    def _is_plan_cancel(self, text: str) -> bool:
        cleaned = re.sub(r"\s+", "", text).lower()
        return any(word in cleaned for word in ["取消plan", "取消计划", "退出plan", "退出计划", "不做了", "重新开始"])

    def _is_execute_plan_text(self, text: str) -> bool:
        cleaned = re.sub(r"\s+", "", text).lower()
        return cleaned in {"执行计划", "开始执行", "按计划执行", "executeplan", "runplan"}

    def _should_treat_as_plan_reply(self, text: str, pending: dict[str, Any], state: AgentState) -> bool:
        if state.get("plan_mode") or state.get("execute_plan"):
            return True
        if self._looks_like_plan_answer(text, pending):
            return True
        if self._looks_like_new_task(text):
            return False
        return len(text.strip()) <= 500

    def _looks_like_plan_answer(self, text: str, pending: dict[str, Any]) -> bool:
        cleaned = text.strip()
        if re.search(r"(?<!\d)\d+\s*[\.\-:：]?\s*[A-Za-z]", cleaned):
            return True
        if re.search(r"(都|全部|全都).*(推荐|默认)", cleaned):
            return True
        questions = pending.get("questions") or []
        options = [str(option) for item in questions if isinstance(item, dict) for option in item.get("options", [])]
        return any(option and option in cleaned for option in options)

    def _looks_like_new_task(self, text: str) -> bool:
        lower = text.lower()
        keywords = ["创建", "实现", "开发", "新建", "编写", "修改", "修复", "解释", "生成文档", "重构", "build", "create", "fix"]
        return len(text.strip()) > 8 and any(word in lower for word in keywords)

    def _build_plan_reply(self, text: str, pending: dict[str, Any]) -> dict[str, Any]:
        questions = pending.get("questions") or []
        answers: list[dict[str, str]] = []
        use_recommended = bool(re.search(r"(都|全部|全都).*(推荐|默认)", text))
        for idx, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                continue
            options = [str(option) for option in item.get("options", [])]
            selected = ""
            source = "custom"
            if use_recommended:
                selected = str(item.get("recommended") or (options[0] if options else ""))
                source = "recommended"
            else:
                match = re.search(rf"(?<!\d){idx}\s*[\.\-:：]?\s*([A-Za-z])", text)
                if match:
                    option_idx = ord(match.group(1).upper()) - ord("A")
                    if 0 <= option_idx < len(options):
                        selected = options[option_idx]
                        source = "option"
                if not selected:
                    selected = next((option for option in options if option and option in text), "")
            if not selected:
                selected = str(item.get("recommended") or (options[0] if options else text.strip()))
                source = "default"
            answers.append({"question": str(item.get("question", "")), "answer": selected, "source": source})
        if not answers:
            answers.append({"question": "自定义补充", "answer": text.strip(), "source": "custom"})
        return {"raw": text, "answers": answers}

    def _compose_pending_plan_text(self, pending: dict[str, Any], reply: dict[str, Any]) -> str:
        lines = [
            "原始需求：",
            str(pending.get("goal") or ""),
            "",
            "上一轮 Plan 问题与用户回答：",
        ]
        for idx, item in enumerate(reply.get("answers", []), start=1):
            lines.append(f"{idx}. 问题：{item.get('question', '')}")
            lines.append(f"   用户回答：{item.get('answer', '')}")
        if reply.get("raw"):
            lines.extend(["", f"用户原始回复：{reply['raw']}"])
        lines.append("")
        lines.append("请基于原始需求和用户回答继续 Plan 流程；信息足够时生成可执行计划。")
        return "\n".join(lines)

    def _compose_saved_plan_text(self, saved_plan: dict[str, Any], text: str) -> str:
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
        pending = state.get("pending_plan") or {}
        if pending.get("goal"):
            return str(pending["goal"])
        ctx = state.get("context")
        if ctx:
            return ctx.goal
        return state["text"]

    def _fallback_plan_markdown(self, state: AgentState) -> str:
        answers = state.get("plan_answers", [])
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
        return state.get("after_repo", "coder")

    def route_after_verifier(self, state: AgentState) -> str:
        if not state.get("tests_ok", True) and int(state.get("retry", 0)) < 2:
            return "coder"
        if state.get("tests_ok", True) and state.get("after_verify") == "doc":
            return "doc"
        return "final"

    def _classify(self, state: AgentState) -> dict[str, Any]:
        text = state["text"]
        lower = text.lower()
        if state.get("plan_mode"):
            return {"task_type": "plan_gen", "need_repo": True, "need_clarify": False, "reason": "Plan 模式已开启"}
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
        code_intent = any(word in lower for word in ["创建", "实现", "开发", "新建", "编写", "做一个", "build", "create"])
        product_intent = any(word in lower for word in ["系统", "应用", "app", "网页", "页面", "接口", "功能", "模块", "工具"])
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
            client = self._client("manager", state)
            data = client.chat_json(
                [{"role": "system", "content": MANAGER_PROMPT}, {"role": "user", "content": text}],
                temperature=0,
            )
            self._add_tokens(state, client.last_usage.total)
            return data
        except Exception:
            return {"task_type": "general_answer", "need_repo": False, "need_code": False, "need_doc": False, "need_clarify": False, "reason": "我在，可以继续告诉我你要处理的代码任务或问题。"}

    def _flow_for(self, state: AgentState, classification: dict[str, Any]) -> tuple[str, str, str]:
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
            after_verify = "doc" if bool(classification.get("need_doc")) else "final"
            return "repo", "coder", after_verify
        return "final", "final", "final"

    def _direct_reply(self, text: str) -> str | None:
        cleaned = re.sub(r"\s+", "", text).lower()
        if not cleaned:
            return "我在。你可以直接告诉我要创建、修改、解释还是生成文档。"
        greetings = {"你好", "您好", "hi", "hello", "在吗", "在不在", "嗨", "hey"}
        thanks = {"谢谢", "谢了", "感谢", "多谢", "thanks", "thankyou"}
        if cleaned in greetings:
            return "你好，我在。你可以直接描述要创建、修改、解释或生成文档的任务；普通问候不会触发仓库读取或写文件。"
        if cleaned in thanks:
            return "不客气。"
        if cleaned in {"你是谁", "你能做什么", "能做什么"}:
            return "我是这个本地编程 agent 的管理入口。简单聊天我会直接回复；涉及代码、文档、计划或排错时，我会按需调用对应智能体。"
        return None

    def _detect_stack(self, files: list[str]) -> list[str]:
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
        tool = action.get("tool")
        try:
            if tool == "write_file":
                rel = fs.write(str(action["path"]), str(action.get("content", "")))
                return {"ok": True, "text": f"写入文件：{rel}", "file": rel}
            if tool == "append_file":
                rel = fs.append(str(action["path"]), str(action.get("content", "")))
                return {"ok": True, "text": f"追加文件：{rel}", "file": rel}
            if tool == "read_file":
                rel = str(action["path"])
                return {"ok": True, "text": f"读取文件：{rel}\n{fs.read(rel, 8000)}"}
            if tool == "list_files":
                return {"ok": True, "text": "文件列表：\n" + "\n".join(fs.list())}
            if tool == "run_command":
                cmd = str(action.get("cmd", ""))
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
        html = fs.read("index.html", 40000)
        js_text = "\n".join(fs.read(file, 40000) for file in files if file.endswith(".js"))
        ids = set(re.findall(r'id=["\']([^"\']+)["\']', html))
        ids.update(re.findall(r"\.id\s*=\s*[\"']([^\"']+)[\"']", js_text))
        ids.update(re.findall(r"setAttribute\([\"']id[\"']\s*,\s*[\"']([^\"']+)[\"']\)", js_text))
        requested_ids = set(re.findall(r"getElementById\([\"']([^\"']+)[\"']\)", js_text))
        missing_ids = sorted(requested_ids - ids)

        inline_calls = set(re.findall(r'on\w+=["\']([A-Za-z_$][\w$]*)\s*\(', html))
        defined_funcs = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", js_text))
        defined_funcs.update(re.findall(r"window\.([A-Za-z_$][\w$]*)\s*=", js_text))
        missing_funcs = sorted(inline_calls - defined_funcs)

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
        return LlmClient(self.model_store.for_agent(agent, state.get("model_id")))

    def _ctx_text(self, state: AgentState) -> str:
        ctx = state.get("context")
        if not ctx:
            return state["text"]
        return ctx.model_dump_json(indent=2)

    def _fallback_doc(self, state: AgentState) -> dict[str, str]:
        files = "\n".join(f"- `{file}`" for file in state.get("changes", [])) or "- 暂无文件变更"
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
        self.event_id += 1
        event = AgentEvent(
            id=self.event_id,
            ts=datetime.now(UTC).isoformat(),
            agent=agent,
            kind=kind,
            msg=msg,
            tokens=tokens,
            data=data or {},
        )
        self.memory.append(state["workdir"], state["session_id"], agent, "event", kind, msg, {"tokens": tokens, "data": data or {}})
        if self.emit_cb:
            self.emit_cb(event)

    def _add_tokens(self, state: AgentState, tokens: int) -> None:
        state["tokens"] = int(state.get("tokens", 0)) + tokens
        self._emit(state, "llm", "usage", f"本次模型调用约消耗 {tokens} token", tokens=tokens)

    def _trim(self, text: str, max_chars: int) -> str:
        return text if len(text) <= max_chars else text[-max_chars:]


def new_session_id() -> str:
    """生成短会话 id，便于文件名和前端展示。"""
    return uuid.uuid4().hex[:12]
