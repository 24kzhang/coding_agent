from __future__ import annotations

from typing import TYPE_CHECKING

from backend.agents.types import AgentState
from backend.tools import FsTool

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def repo(graph: AgentGraph, state: AgentState) -> AgentState:
    """仓库读取智能体：扫描项目结构、读取关键文件、识别技术栈。"""

    graph._emit(state, "repo", "start", "仓库读取智能体正在识别目录结构和技术栈")
    # fs 是受 workdir 限制的文件工具。
    fs = FsTool(state["workdir"])
    # files 是过滤依赖和构建目录后的项目文件索引。
    files = fs.list()
    # snippets 保存关键文件的片段，供下游模型理解项目。
    snippets: dict[str, str] = {}
    # candidates 根据入口文件、依赖清单、任务关键词和文件类型计算相关性。
    candidates = graph._repo_candidates(fs, files, state["text"])
    # budget 控制仓库摘要总字符数，避免一开始就把模型窗口塞满。
    budget = 24_000
    for rel in candidates:
        if budget <= 0:
            break
        # chunk 是文件开头片段；后续 Coder 可通过 read_file 分段读取完整内容。
        chunk = fs.read(rel, min(7000, budget))
        snippets[rel] = chunk
        budget -= len(chunk)
    # stack 是根据文件名粗略识别的技术栈。
    stack = graph._detect_stack(files)
    # repo_context 是传给下游节点的仓库摘要。
    repo_context = {
        "files": files,
        "snippets": snippets,
        "stack": stack,
        "empty": len(files) == 0,
        "selected": candidates,
    }
    graph._emit(state, "repo", "summary", f"识别到 {len(files)} 个文件，技术栈：{', '.join(stack) or '空项目'}")
    graph.memory.append(state["workdir"], state["session_id"], "repo", "fs", "summary", f"文件数：{len(files)}，技术栈：{stack}")
    # ctx 是管理者构造的上下文包，这里补充相关文件列表。
    ctx = state["context"]
    ctx.relevant_files = candidates[:80]
    return {**state, "repo": repo_context, "context": ctx}
