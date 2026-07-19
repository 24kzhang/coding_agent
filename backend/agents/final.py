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
    if state.get("final"):
        summary = str(state["final"])
    elif ok and state.get("coding_summary"):
        summary = str(state["coding_summary"])
    elif ok:
        summary = "任务已完成。"
    else:
        summary = str(state.get("error") or state.get("coding_summary") or "任务未完成，验证或执行存在失败。")
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
