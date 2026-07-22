from __future__ import annotations

import time
from typing import TYPE_CHECKING

from backend.agents.types import AgentState

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def final(graph: AgentGraph, state: AgentState) -> AgentState:
    """最终节点：整理返回结果、写入最终 memory、触发记忆压缩。"""

    # ok 同时要求 Coding 明确完成和验证通过，不能用空测试掩盖 Coding 失败。
    ok = bool(state.get("tests_ok", True)) and bool(state.get("coding_ok", True))
    # summary 是给用户看的最终摘要；没有显式 final 时根据 ok 生成。
    if not ok:
        # failed_tests 优先解释真实验证失败，不能用早先的 Coding 成功摘要掩盖测试问题。
        failed_tests = [item for item in state.get("tests", []) if not item.get("ok")]
        if failed_tests:
            first = failed_tests[0]
            detail = str(first.get("out") or first.get("err") or "验证失败")
            summary = f"任务未通过验证：{first.get('cmd', '检查')} - {detail}"
        else:
            summary = str(state.get("error") or state.get("coding_summary") or "任务未完成，验证或执行存在失败。")
    elif state.get("task_type") == "verify":
        # 只读验收应把验证智能体的具体结论返回给用户，不能退化成泛化的“任务已完成”。
        reviews = [item for item in state.get("tests", []) if item.get("cmd") == "需求实现审查" and item.get("ok")]
        summary = str(reviews[-1].get("out") if reviews else "项目验收通过。")
    elif (
        state.get("task_type") in {"code_gen", "code_mod"}
        and state.get("coding_summary")
        and not (state.get("repo") or {}).get("doc_path")
    ):
        # 代码任务没有进入文档节点时，Coding 完成摘要必须覆盖管理者早期分类文本。
        summary = str(state["coding_summary"])
    elif state.get("final"):
        summary = str(state["final"])
    elif ok and state.get("coding_summary"):
        summary = str(state["coding_summary"])
    elif ok:
        summary = "任务已完成。"
    # duration_ms 使用 run() 写入的单调时钟起点计算，不受系统时间变化影响。
    duration_ms = int((time.monotonic() - float(state.get("started_at", time.monotonic()))) * 1000)
    # result 是返回给前端的结构化 TaskResult 字典。
    result = {
        "ok": ok,
        "summary": summary,
        "files": state.get("changes", []),
        "commands": state.get("commands", []),
        "tests": state.get("tests", []),
        "plan_path": (state.get("plan") or {}).get("path"),
        "doc_path": (state.get("repo") or {}).get("doc_path"),
        "tokens": int(state.get("tokens", 0)),
        "duration_ms": duration_ms,
    }
    if state.get("executing_plan") and ok:
        # 只有执行和验证都成功才结束计划生命周期；失败或中断时保留计划供用户重试。
        graph.memory.append(
            state["workdir"],
            state["session_id"],
            "planner",
            "state",
            "plan_executed",
            "计划已成功执行",
            {"path": (state.get("plan") or {}).get("path")},
        )
    # 写入最终结果，历史会话恢复时会读取这条记录作为 Agent 回复。
    graph.memory.append(state["workdir"], state["session_id"], "manager", "final", "result", summary, result)
    # ctx 存在说明 manager 正常构造过上下文，可以尝试按模型窗口压缩会话记忆。
    ctx = state.get("context")
    if ctx:
        # cfg 是 manager 对应模型配置，ctx 字段用于压缩阈值。
        try:
            # cfg 的 ctx 决定会话压缩阈值；模型配置缺失不应推翻已经完成的任务。
            config = graph.model_store.for_agent("manager", state.get("model_id"))
            graph.memory.maybe_compress(state["workdir"], state["session_id"], config.ctx)
        except KeyError:
            pass
    graph._emit(state, "manager", "result", summary, data=result)
    return {**state, "result": result}
