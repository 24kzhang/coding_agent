from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from backend.agents.prompts import DOC_PROMPT
from backend.agents.types import AgentState
from backend.tools import FsTool
from llm import LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def doc(graph: AgentGraph, state: AgentState) -> AgentState:
    """文档智能体：根据任务、仓库摘要、变更和测试结果写中文文档。"""

    graph._emit(state, "doc", "start", "文档智能体正在生成或更新中文文档")
    # fs 是受 workdir 限制的文件写入工具。
    fs = FsTool(state["workdir"])
    # client 是 doc 智能体对应的模型客户端。
    client = graph._client("doc", state)
    try:
        # data 应该是模型返回的文档 JSON，包含 path/content/summary。
        data = client.chat_json(
            [
                {"role": "system", "content": DOC_PROMPT},
                {
                    "role": "user",
                    "content": graph._ctx_text(state)
                    + "\n\n仓库："
                    + json.dumps(state.get("repo", {}), ensure_ascii=False)[:15000]
                    + "\n\n变更："
                    + json.dumps(state.get("changes", []), ensure_ascii=False)
                    + "\n\n测试："
                    + json.dumps(state.get("tests", []), ensure_ascii=False),
                },
            ]
        )
        graph._add_tokens(state, client.last_usage.total)
    except LlmError:
        # 模型失败时使用基础 README 兜底，保证文档任务不完全失败。
        data = graph._fallback_doc(state)
    # path 是模型建议写入的文档相对路径，默认 README.md。
    path = data.get("path") or "README.md"
    if Path(path).is_absolute():
        # 绝对路径会被强制改为 README.md，防止模型写出项目目录。
        path = "README.md"
    # written 是实际写入的相对路径。
    written = fs.write(path, data.get("content", "# 项目说明\n\n暂无内容。\n"))
    # repo_context 复制已有仓库摘要，并补充 doc_path 供最终结果展示。
    repo_context = dict(state.get("repo", {}))
    repo_context["doc_path"] = str((Path(state["workdir"]) / written).resolve())
    # changes 合并文档文件，并去重排序。
    changes = sorted(dict.fromkeys(list(state.get("changes", [])) + [written]))
    # doc_summary 是面向用户的短文档结果，不使用完整 Markdown 正文。
    doc_summary = graph._clean_summary(data.get("summary"), f"文档已写入 {written}")
    graph._emit(state, "doc", "done", doc_summary)
    # final_summary 优先保留已有回答或计划提示；代码任务则合并 Coding 与文档结果。
    final_summary = state.get("final") or state.get("coding_summary") or doc_summary
    if state.get("coding_summary") and doc_summary not in final_summary:
        final_summary = f"{final_summary}\n{doc_summary}"
    return {**state, "repo": repo_context, "changes": changes, "final": final_summary}
