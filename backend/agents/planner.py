from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backend.agents.prompts import PLANNER_PROMPT
from backend.agents.types import AgentState
from llm import LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def planner(graph: AgentGraph, state: AgentState) -> AgentState:
    """Plan 智能体：生成澄清问题或可执行 Markdown 计划。"""

    graph._emit(state, "planner", "start", "Plan 智能体正在生成问题或可执行计划")
    # client 是 planner 智能体对应的模型客户端。
    client = graph._client("planner", state)
    # messages 是发给模型的系统提示词和上下文包。
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": graph._ctx_text(state)},
    ]
    try:
        # data 应该是模型返回的 JSON，status 为 questions 或 plan。
        data = client.chat_json(messages)
        graph._add_tokens(state, client.last_usage.total)
    except LlmError as exc:
        # 模型返回非 JSON 时使用兜底问题或兜底计划，避免 Plan 流程卡死。
        graph._emit(state, "planner", "error", f"Plan 模型返回异常，已使用默认澄清问题：{str(exc)[:120]}")
        if state.get("pending_plan"):
            # 如果已经在回答上一轮问题，模型异常时直接生成默认计划。
            data = {"status": "plan", "title": "默认执行计划", "markdown": graph._fallback_plan_markdown(state)}
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
        data = {**data, "questions": graph._normalize_plan_questions(data.get("questions"))}
    if data.get("status") == "plan" or state.get("execute_plan"):
        # md 是最终计划正文；缺失时使用兜底计划。
        md = data.get("markdown") or graph._fallback_plan_markdown(state)
        # plan_dir 是用户项目内保存计划的目录。
        plan_dir = Path(state["workdir"]) / "docs" / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        # path 是本会话对应的计划文件路径。
        path = plan_dir / f"{state['session_id']}.md"
        path.write_text(md.strip() + "\n", encoding="utf-8")
        graph._emit(state, "planner", "plan", f"计划已保存：{path}")
        graph.memory.append(
            state["workdir"],
            state["session_id"],
            "planner",
            "state",
            "plan_done",
            "计划已生成",
            {"status": "plan", "path": str(path), "markdown": md, "goal": graph._plan_goal(state)},
        )
        # execute_plan=True 时直接进入 repo，否则停在 final 等用户确认。
        route = "repo" if state.get("execute_plan") else "final"
        # final 是返回给用户的确认提示；直接执行时保持空字符串。
        final = "" if route == "repo" else "计划已生成，等待你确认执行。\n计划文件：" + str(path) + "\n确认后回复“执行计划”。"
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
    summary = graph._format_plan_questions(data.get("questions", []))
    # 保存 pending_plan，下一轮用户回复选项时 manager 可以找回上下文。
    graph.memory.append(
        state["workdir"],
        state["session_id"],
        "planner",
        "state",
        "pending_plan",
        "Plan 问题等待用户回答",
        {
            "status": "questions",
            "goal": graph._plan_goal(state),
            "questions": data.get("questions", []),
            "answers": state.get("plan_answers", []),
        },
    )
    graph._emit(state, "planner", "questions", "已生成澄清问题，等待用户选择")
    return {**state, "plan": data, "route": "final", "final": summary}
