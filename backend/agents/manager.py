# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from api.schema import ContextPackage
from backend.agents.types import AgentState

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def manager(graph: AgentGraph, state: AgentState) -> AgentState:
    """上下文管理智能体：分类任务、处理 Plan 状态、构造 Context Package。"""

    workdir = state["workdir"]
    session_id = state["session_id"]
    interrupted = bool(state.get("resuming"))
    graph._emit(state, "manager", "start", "管理者正在分类任务并构造上下文包")

    pending_plan = graph._latest_pending_plan(workdir, session_id)
    saved_plan = graph._latest_saved_plan(workdir, session_id)
    memory_request = graph._memory_request(state["text"])
    classification: dict[str, Any]
    if memory_request:
        memory_text, global_scope = memory_request
        scope_name = graph.memory.remember(workdir, memory_text, global_scope=global_scope)
        classification = {
            "task_type": "direct",
            "need_repo": False,
            "need_code": False,
            "need_doc": False,
            "need_clarify": False,
            "reason": "用户明确要求写入长期记忆",
            "direct_reply": f"已写入{scope_name}：{memory_text}",
        }
    elif pending_plan and graph._is_plan_cancel(state["text"]):
        graph.memory.append(
            workdir,
            session_id,
            "planner",
            "state",
            "plan_cancelled",
            "用户取消了待确认计划",
            {"goal": pending_plan.get("goal", "")},
        )
        classification = {
            "task_type": "direct",
            "need_repo": False,
            "need_code": False,
            "need_doc": False,
            "need_clarify": False,
            "reason": "用户取消了上一轮 Plan 流程",
            "direct_reply": "已取消上一轮 Plan 流程。你可以重新描述新的需求。",
        }
    elif pending_plan and graph._should_treat_as_plan_reply(state["text"], pending_plan, state):
        reply = graph._build_plan_reply(state["text"], pending_plan)
        all_answers = list(pending_plan.get("answers") or []) + list(reply["answers"])
        state = {
            **state,
            "text": graph._compose_pending_plan_text(pending_plan, reply),
            "plan_mode": True,
            "pending_plan": pending_plan,
            "plan_answers": all_answers,
        }
        classification = {
            "task_type": "plan_gen",
            "need_repo": True,
            "need_code": False,
            "need_doc": False,
            "need_clarify": False,
            "reason": "用户正在回答上一轮 Plan 澄清问题，继续同一个 Plan 流程",
        }
    elif saved_plan and (state.get("execute_plan") or graph._is_execute_plan_text(state["text"])):
        state = {
            **state,
            "text": graph._compose_saved_plan_text(saved_plan, state["text"]),
            "original_goal": str(saved_plan.get("goal") or ""),
            "plan": saved_plan,
            "execute_plan": True,
            "plan_mode": False,
            "executing_plan": True,
        }
        classification = {
            "task_type": "code_gen",
            "need_repo": True,
            "need_code": True,
            "need_doc": True,
            "need_clarify": False,
            "reason": "用户确认执行已保存计划",
        }
    elif pending_plan:
        graph.memory.append(
            workdir,
            session_id,
            "planner",
            "state",
            "plan_cancelled",
            "新任务已替代待回答计划",
            {"goal": pending_plan.get("goal", "")},
        )
        classification = graph._classify(state)
    else:
        classification = graph._classify(state)
    classification = graph._normalize_classification(classification)
    task_type = classification.get("task_type", "code_gen")
    route, after_repo, after_verify = graph._flow_for(state, classification)

    ctx = ContextPackage(
        goal=state["text"],
        task_type=task_type,
        workdir=workdir,
        plan_mode=bool(state.get("plan_mode")),
        project_memory=graph._trim(graph.memory.project_memory(workdir), 2000),
        global_memory=graph._trim(graph.memory.global_memory(), 2000),
        constraints=[
            "代码文件名使用简短英文",
            "面向用户内容、注释和文档使用简体中文",
            "优先最小改动，避免无关重构",
            "当前用户明确指令优先于项目记忆和全局记忆",
            "项目事实优先于默认偏好，安全规则优先级最高",
        ],
        recent=graph.memory.conversation_context(workdir, session_id, limit=12, exclude_latest_user=True),
        skills=list(state.get("skills") or []),
        attachments=[
            {key: item[key] for key in ("id", "name", "mime", "size") if key in item}
            for item in state.get("attachments") or []
        ],
    )
    if ctx.skills or state.get("mcp_servers"):
        selected_names = [f"Skill:{item.id}" for item in ctx.skills]
        selected_names.extend(f"MCP:{item.get('id')}" for item in state.get("mcp_servers", []))
        graph._emit(state, "manager", "capability", "本轮已启用：" + "、".join(selected_names))
    for skill in ctx.skills:
        graph._emit(state, "manager", "skill", f"已读取技能 {skill.name}（{len(skill.content)} 字符）", data={
            "技能": skill.id, "说明": skill.description, "读取内容": skill.content,
        })
    if interrupted:
        ctx.recent.append("检测到上次会话可能异常中断，本次会继续以当前磁盘状态为准。")
    graph.memory.append(workdir, session_id, "manager", "classify", "context", f"任务分类：{task_type}")
    return {
        **state,
        "task_type": task_type,
        "route": route,
        "after_repo": after_repo,
        "after_verify": after_verify,
        "need_doc": after_verify == "doc" or after_repo == "doc",
        "context": ctx,
        "final": classification.get("direct_reply") or "",
    }
