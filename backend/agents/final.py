# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from backend.agents.types import AgentState

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def final(graph: AgentGraph, state: AgentState) -> AgentState:
    """最终节点：整理返回结果、写入最终 memory、触发记忆压缩。"""

    ok = bool(state.get("tests_ok", True)) and bool(state.get("coding_ok", True))
    if not ok:
        failed_tests = [item for item in state.get("tests", []) if not item.get("ok")]
        if failed_tests:
            first = failed_tests[0]
            detail = str(first.get("out") or first.get("err") or "验证失败")
            summary = f"任务未通过验证：{first.get('cmd', '检查')} - {detail}"
        else:
            summary = str(state.get("error") or state.get("coding_summary") or "任务未完成，验证或执行存在失败。")
    elif state.get("task_type") == "verify":
        reviews = [item for item in state.get("tests", []) if item.get("cmd") == "需求实现审查" and item.get("ok")]
        summary = str(reviews[-1].get("out") if reviews else "项目验收通过。")
    elif (
        state.get("task_type") in {"code_gen", "code_mod"}
        and state.get("coding_summary")
        and not (state.get("repo") or {}).get("doc_path")
    ):
        summary = str(state["coding_summary"])
    elif state.get("final"):
        summary = str(state["final"])
    elif ok and state.get("coding_summary"):
        summary = str(state["coding_summary"])
    elif ok:
        summary = "任务已完成。"
    duration_ms = int((time.monotonic() - float(state.get("started_at", time.monotonic()))) * 1000)
    result = graph.model_store.redact({
        "ok": ok,
        "summary": summary,
        "files": state.get("changes", []),
        "commands": state.get("commands", []),
        "tests": state.get("tests", []),
        "plan_path": (state.get("plan") or {}).get("path"),
        "doc_path": (state.get("repo") or {}).get("doc_path"),
        "tokens": int(state.get("tokens", 0)),
        "duration_ms": duration_ms,
    })
    summary = graph.model_store.redact_text(summary)
    if state.get("executing_plan") and ok:
        graph.memory.append(
            state["workdir"],
            state["session_id"],
            "planner",
            "state",
            "plan_executed",
            "计划已成功执行",
            {"path": (state.get("plan") or {}).get("path")},
        )
    graph.memory.append(state["workdir"], state["session_id"], "manager", "final", "result", summary, result)
    ctx = state.get("context")
    if ctx:
        try:
            config = graph.model_store.for_agent("manager", state.get("model_id"))
            graph.memory.maybe_compress(state["workdir"], state["session_id"], config.ctx)
        except KeyError:
            pass
    graph._emit(state, "manager", "result", summary, data=result)
    return {**state, "result": result}
