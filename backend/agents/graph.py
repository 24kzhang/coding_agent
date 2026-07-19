from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from api.schema import AgentEvent, TaskResult
from backend.agents.answer import answer
from backend.agents.coder import coder
from backend.agents.doc import doc
from backend.agents.final import final
from backend.agents.manager import manager
from backend.agents.planner import planner
from backend.agents.prompts import MANAGER_PROMPT
from backend.agents.repo import repo
from backend.agents.types import AgentState
from backend.agents.verifier import verifier
from backend.clock import utc_stamp
from backend.memory import MemoryStore
from backend.tools import FsTool, GitTool, ShellTool
from llm import LlmClient, ModelStore

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
        resuming: bool = False,
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
            # resuming 由 HTTP 层在写入本轮 run/start 前计算，避免把当前任务误判成旧中断。
            "resuming": resuming,
            # started_at 使用单调时钟，只用于计算耗时，不受系统时间调整影响。
            "started_at": time.monotonic(),
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
            # coding_ok 初始为 True，使非 Coding 路径不被误判失败；coder 会主动重置。
            "coding_ok": True,
            # coding_summary 在 Coding 明确结束后写入。
            "coding_summary": "",
            # error 保存跨节点传播的失败原因。
            "error": "",
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
            "tokens": int(final.get("tokens", 0)),
            "duration_ms": int((time.monotonic() - final.get("started_at", time.monotonic())) * 1000),
        }
        return TaskResult(**result)

    def _build(self) -> Any:
        """构建 LangGraph 节点和条件路由。"""

        # graph 是以 AgentState 为状态类型的 LangGraph 状态图。
        graph = StateGraph(AgentState)
        # manager 是入口节点，负责分类、构造 Context Package 和决定第一跳。
        graph.add_node("manager", partial(manager, self))
        # planner 负责 Plan 模式下的选择题和计划文件生成。
        graph.add_node("planner", partial(planner, self))
        # repo 负责读取仓库结构和关键文件片段。
        graph.add_node("repo", partial(repo, self))
        # answer 负责解释和普通答疑，不写磁盘。
        graph.add_node("answer", partial(answer, self))
        # coder 负责 ReAct 写代码和执行工具动作。
        graph.add_node("coder", partial(coder, self))
        # verifier 负责选择并运行测试。
        graph.add_node("verifier", partial(verifier, self))
        # doc 负责生成或更新中文文档。
        graph.add_node("doc", partial(doc, self))
        # final 负责整理 TaskResult、写最终 memory、输出最终事件。
        graph.add_node("final", partial(final, self))

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
            if rec.get("ag") == "planner" and rec.get("tl") == "state" and rec.get("k") in {"plan_executed", "plan_cancelled"}:
                return None
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
        # 普通问候和感谢属于新对话，不应被旧 Plan 状态解释成自定义选项。
        if self._direct_reply(text):
            return False
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
            "Plan 问题与用户回答：",
        ]
        # previous_answers 是更早轮次已经确认的选择。
        previous_answers = pending.get("answers") or []
        for idx, item in enumerate(previous_answers, start=1):
            lines.append(f"{idx}. 问题：{item.get('question', '')}")
            lines.append(f"   用户回答：{item.get('answer', '')}")
        # idx 是回答序号，item 是单个问题的结构化回答。
        for idx, item in enumerate(reply.get("answers", []), start=len(previous_answers) + 1):
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
        if not state.get("tests_ok", True) and int(state.get("retry", 0)) < 3:
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
        # code_intent 表示用户有从零创建或实现可交付物的倾向。
        code_intent = any(word in lower for word in ["创建", "实现", "开发", "新建", "编写", "做一个", "搭建", "build", "create"])
        # product_intent 表示用户目标是代码、系统、页面、接口或其他工程交付物。
        product_intent = any(word in lower for word in ["代码", "项目", "系统", "应用", "app", "网页", "页面", "接口", "功能", "模块", "工具", "agent"])
        # doc_intent 表示用户明确要求文档。
        doc_intent = any(word in lower for word in ["文档", "readme", "说明文档", "教程", "部署说明", "接口说明"])
        # modify_intent 覆盖真实迭代中常见的完善、优化、添加和更新表达。
        modify_intent = any(
            word in lower
            for word in ["修改", "修复", "完善", "优化", "调整", "增加", "添加", "升级", "删除", "更新", "改成", "改为", "bug", "fix", "refactor"]
        )
        # code_subject 表示修改对象明显属于代码仓库，而不是纯文档措辞。
        code_subject = product_intent or any(word in lower for word in ["函数", "类", "变量", "样式", "前端", "后端", "数据库", "配置", "依赖"])
        if modify_intent and code_subject:
            return {
                "task_type": "code_mod",
                "need_repo": True,
                "need_code": True,
                "need_doc": doc_intent,
                "need_clarify": False,
                "reason": "用户需要迭代或修复现有项目",
            }
        if code_intent and product_intent:
            return {
                "task_type": "code_gen",
                "need_repo": True,
                "need_code": True,
                "need_doc": doc_intent,
                "need_clarify": False,
                "reason": "用户需要创建可运行代码",
            }
        if any(word in lower for word in ["解释", "为什么", "原因", "报错", "审查", "检查", "review", "error", "traceback"]):
            return {"task_type": "code_explain", "need_repo": True, "need_code": False, "need_doc": False, "need_clarify": False, "reason": "用户需要解释或排错"}
        if doc_intent:
            return {"task_type": "doc_gen", "need_repo": True, "need_code": False, "need_doc": True, "need_clarify": False, "reason": "用户需要生成文档"}
        try:
            # client 是管理者模型，用于规则无法覆盖的模糊输入。
            client = self._client("manager", state)
            # data 是模型返回的分类 JSON。
            # classify_context 同时包含本轮输入和最近真实对话，支持“继续改”“还是不行”等追问。
            classify_context = {
                "current": text,
                "recent": self.memory.conversation_context(
                    state["workdir"],
                    state["session_id"],
                    limit=8,
                    exclude_latest_user=True,
                ),
            }
            data = client.chat_json(
                [
                    {"role": "system", "content": MANAGER_PROMPT},
                    {"role": "user", "content": json.dumps(classify_context, ensure_ascii=False)},
                ],
                temperature=0,
            )
            self._add_tokens(state, client.last_usage.total)
            return data
        except Exception:
            # 管理者模型异常时降级为普通回答，避免误触发仓库读取和写代码。
            return {"task_type": "general_answer", "need_repo": False, "need_code": False, "need_doc": False, "need_clarify": False, "reason": "我在，可以继续告诉我你要处理的代码任务或问题。"}

    def _normalize_classification(self, raw: dict[str, Any]) -> dict[str, Any]:
        """把规则或模型分类规范化为受控路由字段。"""

        # allowed 是 LangGraph 已实现的全部任务类型，未知值不能直接参与路由。
        allowed = {"direct", "general_answer", "code_gen", "code_mod", "code_explain", "doc_gen", "plan_gen"}
        # data 是输入副本，避免修改模型响应原对象。
        data = dict(raw) if isinstance(raw, dict) else {}
        # task_type 无效时回退到只读回答，避免误写代码或无声结束。
        task_type = str(data.get("task_type") or "general_answer")
        if task_type not in allowed:
            task_type = "general_answer"
        # reason 始终是短中文说明，异常长模型文本不会进入最终回复。
        reason = self._clean_summary(data.get("reason"), "管理者已完成任务分类。")
        # normalized 显式转换所有布尔字段，字符串 "false" 不会被 bool() 误判为 True。
        normalized = {
            **data,
            "task_type": task_type,
            "need_repo": self._as_bool(data.get("need_repo"), task_type in {"code_gen", "code_mod", "code_explain", "doc_gen"}),
            "need_code": self._as_bool(data.get("need_code"), task_type in {"code_gen", "code_mod"}),
            "need_doc": self._as_bool(data.get("need_doc"), task_type == "doc_gen"),
            "need_clarify": self._as_bool(data.get("need_clarify"), False),
            "reason": reason,
        }
        if normalized["need_clarify"]:
            # clarification 是模型可选的具体问题；缺失时使用 reason，确保用户知道要补什么。
            normalized["direct_reply"] = self._clean_summary(data.get("clarification"), reason)
        if task_type == "direct" and not data.get("direct_reply"):
            # 规则未命中的普通对话仍交给 answer 模型，不能把内部分类理由直接回复用户。
            normalized["task_type"] = "general_answer"
        return normalized

    def _as_bool(self, value: Any, fallback: bool) -> bool:
        """安全解析模型返回的布尔值。"""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "是"}:
                return True
            if lowered in {"false", "0", "no", "否"}:
                return False
        return fallback

    def _memory_request(self, text: str) -> tuple[str, bool] | None:
        """识别用户明确要求长期记住的偏好，并判断作用域。"""

        # clean 是合并空白后的原始指令，便于提取“记住”之后的正文。
        clean = re.sub(r"\s+", " ", text).strip()
        # explicit_markers 只包含明确长期意图，并按更具体的短语优先匹配。
        explicit_markers = ["请记住我", "请记住", "记住我", "以后都", "今后都"]
        marker = next((item for item in explicit_markers if item in clean), "")
        if not marker:
            return None
        # content 只取触发词之后的正文，避免把“请记住我”等控制语句写进记忆。
        content = clean.split(marker, 1)[1].strip(" ：:，,")
        # 用户常说“请记住我以后都……”，第二个长期语气词也不属于偏好正文。
        for prefix in ["以后都", "今后都"]:
            if content.startswith(prefix):
                content = content[len(prefix) :].strip(" ：:，,")
                break
        if not content:
            return None
        # global_scope 只有明确提到全局、所有项目或跨项目时才为 True。
        global_scope = any(word in clean for word in ["全局", "所有项目", "跨项目"])
        return self._trim(content, 500), global_scope

    def _clean_summary(self, value: Any, fallback: str) -> str:
        """清洗模型生成的短摘要，移除多余空白并限制长度。"""

        # text 保留摘要中的正常换行，但压缩连续空格和三个以上空行。
        text = str(value or "").strip()
        if not text:
            return fallback
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return self._trim(text, 1200)

    def _flow_for(self, state: AgentState, classification: dict[str, Any]) -> tuple[str, str, str]:
        """把分类结果转换成 LangGraph 三段路由。"""

        # task_type 是管理者分类出的任务类型。
        task_type = classification.get("task_type", "general_answer")
        if task_type == "plan_gen":
            return "planner", "final", "final"
        if state.get("plan_mode") and not state.get("execute_plan"):
            return "planner", "final", "final"
        if classification.get("need_clarify"):
            return "final", "final", "final"
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

    def _repo_candidates(self, fs: FsTool, files: list[str], goal: str) -> list[str]:
        """根据任务和工程惯例选择值得放入初始上下文的文件。"""

        # priority_names 是依赖清单、入口、协作规则和项目说明的常见文件名。
        priority_names = {
            "AGENTS.md",
            "README.md",
            "Cargo.toml",
            "go.mod",
            "package.json",
            "pnpm-workspace.yaml",
            "pyproject.toml",
            "requirements.txt",
            "tsconfig.json",
            "uv.lock",
        }
        # entry_names 是跨技术栈常见入口文件。
        entry_names = {"app.py", "main.py", "main.ts", "main.tsx", "index.html", "index.js", "index.ts", "server.py"}
        # tokens 提取用户明确写出的英文路径片段和标识符，用于文件名与内容检索。
        tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", goal)]
        # scores 保存每个候选文件的相关性分数。
        scores: dict[str, int] = {}
        for rel in files:
            # path 和 name 分别用于完整路径与 basename 评分。
            path = Path(rel)
            name = path.name
            if fs.is_sensitive(fs.safe(rel)):
                continue
            score = 0
            if name in priority_names:
                score += 120
            if name in entry_names:
                score += 100
            if path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".go", ".rs", ".java"}:
                score += 25
            if path.parts and path.parts[0] in {"src", "app", "backend", "frontend", "server", "api"}:
                score += 20
            # tests 文件对修改后的验证有价值，但优先级低于生产入口和目标文件。
            if "test" in name.lower() or "tests" in path.parts:
                score += 8
            for token in tokens:
                if token in rel.lower():
                    score += 80
            if score:
                scores[rel] = score

        # 对最多 5 个用户标识符做内容检索，把定义或引用文件纳入候选集合。
        for token in list(dict.fromkeys(tokens))[:5]:
            try:
                hits = fs.search(token, max_results=12)
            except ValueError:
                hits = []
            for hit in hits:
                rel = str(hit["path"])
                scores[rel] = scores.get(rel, 0) + 90

        # 空项目直接返回空列表；有文件但没有得分时选择少量根级代码文件兜底。
        if not scores:
            fallback = [rel for rel in files if len(Path(rel).parts) <= 2 and Path(rel).suffix.lower() in FsTool.text_suffixes]
            return fallback[:30]
        # ordered 先按分数倒序，再按路径排序，结果稳定且最相关文件在前。
        ordered = sorted(scores, key=lambda rel: (-scores[rel], rel))
        return ordered[:50]

    def _normalize_actions(self, raw: Any) -> list[dict[str, Any]]:
        """过滤并限制 Coding 模型返回的工具动作。"""

        # allowed 是 Coding 智能体当前真实实现的工具协议。
        allowed = {
            "append_file",
            "git_diff",
            "git_status",
            "list_files",
            "read_file",
            "replace_file",
            "run_command",
            "search_files",
            "write_file",
        }
        # actions 不是列表时视为无动作；每轮最多执行 6 个，避免模型一次失控调用。
        actions = raw if isinstance(raw, list) else []
        normalized: list[dict[str, Any]] = []
        for item in actions[:6]:
            if not isinstance(item, dict):
                continue
            # tool 必须命中白名单，未知工具会作为无效动作被忽略。
            tool = str(item.get("tool") or "")
            if tool not in allowed:
                continue
            normalized.append({**item, "tool": tool})
        return normalized

    def _verification_commands(self, fs: FsTool, files: list[str]) -> list[str]:
        """读取项目配置，选择已有且适合非交互执行的验证命令。"""

        # commands 按轻量检查、测试、构建的顺序保存，并在返回前去重。
        commands: list[str] = []
        # python_files 用于判断是否存在 Python 代码和测试目录。
        python_files = [rel for rel in files if rel.endswith(".py")]
        if python_files:
            # compile_targets 优先选择顶层代码目录，避免 compileall 扫描依赖环境。
            top_dirs = sorted({Path(rel).parts[0] for rel in python_files if len(Path(rel).parts) > 1})
            root_files = [rel for rel in python_files if len(Path(rel).parts) == 1]
            compile_targets = top_dirs[:8] + root_files[:12]
            if compile_targets:
                prefix = "uv run " if "pyproject.toml" in files else ""
                commands.append(prefix + "python -m compileall -q " + " ".join(compile_targets))
            # 只有仓库确实存在 pytest 风格测试时才运行 pytest，避免“无测试”被当成失败。
            has_pytest = any("tests" in Path(rel).parts or Path(rel).name.startswith("test_") for rel in python_files)
            if has_pytest:
                commands.append(("uv run " if "pyproject.toml" in files else "") + "pytest -q")

        if "package.json" in files:
            try:
                # package_data 用结构化 JSON 读取 scripts，不猜测项目使用哪个 Node 测试框架。
                package_data = json.loads(fs.read("package.json", 80_000))
                scripts = package_data.get("scripts") if isinstance(package_data, dict) else {}
            except (json.JSONDecodeError, OSError, ValueError):
                scripts = {}
            if isinstance(scripts, dict):
                for script in ["lint", "test", "typecheck", "build"]:
                    if script in scripts:
                        commands.append(f"npm run {script}")
        return list(dict.fromkeys(commands))[:4]

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
            if tool == "replace_file":
                # expected 要求旧文本匹配数量准确，默认只允许唯一匹配。
                expected = int(action.get("expected", 1))
                rel = fs.replace(
                    str(action["path"]),
                    str(action.get("old", "")),
                    str(action.get("new", "")),
                    expected=expected,
                )
                return {"ok": True, "text": f"精确修改文件：{rel}", "file": rel}
            if tool == "read_file":
                # rel 是模型要求读取的项目相对路径。
                rel = str(action["path"])
                # start/max_chars 支持读取大文件的后续片段。
                start = int(action.get("start", 0))
                max_chars = int(action.get("max_chars", 12000))
                return {"ok": True, "text": f"读取文件：{rel}（从字符 {start} 开始）\n{fs.read(rel, max_chars, start)}"}
            if tool == "search_files":
                # results 是结构化内容命中列表，方便模型继续精确读取目标文件。
                results = fs.search(str(action.get("query", "")), int(action.get("max_results", 50)))
                return {"ok": True, "text": "搜索结果：\n" + json.dumps(results, ensure_ascii=False, indent=2)}
            if tool == "list_files":
                return {"ok": True, "text": "文件列表：\n" + self._trim("\n".join(fs.list()), 30_000)}
            if tool == "run_command":
                # cmd 是模型要求执行的命令字符串，ShellTool 会做危险命令检查。
                cmd = str(action.get("cmd", ""))
                # res 是命令执行结果。
                res = shell.run(cmd)
                return {"ok": res.get("ok"), "text": f"运行命令：{cmd}\n{res.get('out','')}\n{res.get('err','')}", "cmd": cmd, "result": res}
            if tool == "git_status":
                return {"ok": True, "text": "Git 状态：\n" + git.status()}
            if tool == "git_diff":
                return {"ok": True, "text": "Git 差异：\n" + git.diff()}
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
            ts=utc_stamp(),
            agent=agent,
            kind=kind,
            msg=self._trim(str(msg), 3000),
            tokens=tokens,
            data=data or {},
        )
        # 同步把事件写入会话记忆，便于中断后查看执行到哪一步。
        # JSONL 只保存事件摘要和 token，不重复保存可能很大的命令输出结构。
        self.memory.append(
            state["workdir"],
            state["session_id"],
            agent,
            "event",
            kind,
            self._trim(str(msg), 3000),
            {"tokens": tokens},
        )
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
