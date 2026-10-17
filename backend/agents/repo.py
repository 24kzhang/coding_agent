# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.agents.types import AgentState
from backend.tools import FsTool

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def repo(graph: AgentGraph, state: AgentState) -> AgentState:
    """仓库读取智能体：扫描项目结构、读取关键文件、识别技术栈。"""

    graph._emit(state, "repo", "start", "仓库读取智能体正在识别目录结构和技术栈")
    fs = FsTool(state["workdir"])
    files = fs.list()
    snippets: dict[str, str] = {}
    candidates = graph._repo_candidates(fs, files, graph._effective_text(state))
    budget = 24_000
    for rel in candidates:
        if budget <= 0:
            break
        chunk = graph.model_store.redact_text(fs.read(rel, min(7000, budget)))
        snippets[rel] = chunk
        budget -= len(chunk)
    stack = graph._detect_stack(files)
    symbol_map = graph.repo_map.build(fs, files, graph._effective_text(state))
    repo_context = {
        "files": files,
        "snippets": snippets,
        "stack": stack,
        "empty": len(files) == 0,
        "selected": candidates,
        "symbol_map": symbol_map,
    }
    graph._emit(state, "repo", "summary", f"识别到 {len(files)} 个文件，技术栈：{', '.join(stack) or '空项目'}")
    graph.memory.append(state["workdir"], state["session_id"], "repo", "fs", "summary", f"文件数：{len(files)}，技术栈：{stack}")
    ctx = state["context"]
    ctx.relevant_files = candidates[:80]
    return {**state, "repo": repo_context, "context": ctx}
