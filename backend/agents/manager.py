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
    interrupted = bool(state.get("resuming")) # 上次会话是否可能中断
    graph._emit(state, "manager", "start", "管理者正在分类任务并构造上下文包")

    # pending_plan 是上一轮 Plan 选择题状态；存在时用户本轮可能是在回复选项。
    pending_plan = graph._latest_pending_plan(workdir, session_id)
    # saved_plan 是最近生成但等待确认执行的计划。
    saved_plan = graph._latest_saved_plan(workdir, session_id)
    # memory_request 只响应用户明确的长期记忆指令，不从普通任务中猜测偏好。
    memory_request = graph._memory_request(state["text"])
    # classification 保存管理者分类结果，后续会交给 _flow_for 转成路由。
    classification: dict[str, Any]
    if memory_request:
        # memory_text 是去掉“记住”等触发词后的实际偏好内容。
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
        # 用户明确取消 Plan 时写入 plan_cancelled，避免下一轮继续读取旧 pending_plan。
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
        # reply 是把用户的 1A/2B 等回复解析成结构化答案后的结果。
        reply = graph._build_plan_reply(state["text"], pending_plan)
        # all_answers 合并前几轮和本轮回答，支持 Plan 多轮追问而不丢选择上下文。
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
        # 用户确认执行计划时，把计划正文合并进任务文本，后续 repo/coder 能看到完整计划。
        state = {
            **state,
            "text": graph._compose_saved_plan_text(saved_plan, state["text"]),
            "plan": saved_plan,
            # 自然语言“执行计划”与旧 execute_plan 按钮具有相同语义，必须进入执行态。
            "execute_plan": True,
            # 即使前端 Plan 开关仍保持开启，也不能再次路由回 planner 重复生成计划。
            "plan_mode": False,
            # 最终节点仅在 Coding 和验证均成功后才把计划标记为已执行。
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
        # 用户输入明显是新任务时终止旧 pending_plan，避免它在以后劫持普通短回复。
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
        # 没有 Plan 特殊状态时，进入普通任务分类。
        classification = graph._classify(state)
    # 任何模型分类都必须经过白名单和布尔字段规范化，防止未知 task_type 直接结束。
    classification = graph._normalize_classification(classification)
    # task_type 是标准任务类型，兜底为 code_gen。
    task_type = classification.get("task_type", "code_gen")
    # route/after_repo/after_verify 是 LangGraph 后续路由需要的三个方向字段。
    route, after_repo, after_verify = graph._flow_for(state, classification)

    # ctx 是管理者给下游智能体的结构化上下文包，不直接塞完整历史。
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
    )
    if interrupted:
        # 会话疑似中断时，在最近上下文里提醒下游以磁盘当前状态为准。
        ctx.recent.append("检测到上次会话可能异常中断，本次会继续以当前磁盘状态为准。")
    # 记录本次分类结果，供历史排查和中断恢复使用。
    graph.memory.append(workdir, session_id, "manager", "classify", "context", f"任务分类：{task_type}")
    return {
        **state,
        "task_type": task_type,
        "route": route,
        "after_repo": after_repo,
        "after_verify": after_verify,
        "need_doc": after_verify == "doc" or after_repo == "doc",
        "context": ctx,
        # final 只保存真正要回复用户的文本；分类 reason 是内部路由依据，不能覆盖后续执行结果。
        "final": classification.get("direct_reply") or "",
    }
