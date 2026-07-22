from __future__ import annotations

import json
import re
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
        # content 直接接收 Markdown 正文，避免长文档被 JSON 转义后触发兼容模型格式和超时问题。
        content = client.chat(
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
        ).strip()
        graph._add_tokens(state, client.last_usage.total)
    except LlmError as exc:
        # 文档模型不可用时绝不能用通用模板覆盖已有 README；保留磁盘并明确报告失败。
        failure = {
            "cmd": "文档生成",
            "ok": False,
            "infra": True,
            "out": f"文档模型不可用：{graph._trim(str(exc), 160)}",
        }
        tests = list(state.get("tests", [])) + [failure]
        graph._emit(state, "doc", "error", failure["out"], data=failure)
        return {**state, "tests": tests, "tests_ok": False, "error": failure["out"]}
    if content.startswith("```") and content.endswith("```"):
        # 少数模型仍会包裹代码围栏；仅去掉最外层围栏，不改动正文内部代码块。
        content = re.sub(r"^```(?:markdown|md)?\s*\n", "", content, count=1, flags=re.IGNORECASE)
        content = re.sub(r"\n```\s*$", "", content, count=1)
    if not content.strip():
        failure = {"cmd": "文档生成", "ok": False, "out": "文档模型返回空内容"}
        tests = list(state.get("tests", [])) + [failure]
        return {**state, "tests": tests, "tests_ok": False, "error": failure["out"]}
    # path 从用户任务中的明确 Markdown 路径确定；未指定时使用项目根目录 README.md。
    path_match = re.search(r"(?<![\w.-])([\w./-]+\.md)(?![\w.-])", state.get("text", ""), re.IGNORECASE)
    path = path_match.group(1) if path_match else "README.md"
    if Path(path).is_absolute() or ".." in Path(path).parts:
        # 绝对路径和父级跳转会被强制改为 README.md，防止模型任务文本写出项目目录。
        path = "README.md"
    # written 是实际写入的相对路径。
    written = fs.write(path, content.rstrip() + "\n")
    # repo_context 复制已有仓库摘要，并补充 doc_path 供最终结果展示。
    repo_context = dict(state.get("repo", {}))
    repo_context["doc_path"] = str((Path(state["workdir"]) / written).resolve())
    # changes 合并文档文件，并去重排序。
    changes = sorted(dict.fromkeys(list(state.get("changes", [])) + [written]))
    # doc_summary 是面向用户的短文档结果，不使用完整 Markdown 正文。
    doc_summary = f"文档已写入 {written}"
    graph._emit(state, "doc", "done", doc_summary)
    # 代码任务优先使用真实 Coding 完成摘要；管理者早期分类文本不能成为最终结果。
    final_summary = state.get("coding_summary") or state.get("final") or doc_summary
    if state.get("coding_summary") and doc_summary not in final_summary:
        final_summary = f"{final_summary}\n{doc_summary}"
    return {**state, "repo": repo_context, "changes": changes, "final": final_summary}
