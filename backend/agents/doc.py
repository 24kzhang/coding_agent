# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from backend.agents.prompts import DOC_PROMPT
from backend.agents.types import AgentState
from backend.tools import FsTool
from backend.tools.scope import delivery_root, require_delivery_path
from llm import LlmBalanceError, LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def _document_path(graph: AgentGraph, state: AgentState, fs: FsTool) -> str:
    """按显式路径或真实交付项目定位文档，不能默认污染父工作区。"""

    text = graph._effective_text(state)
    explicit = None
    for match in re.finditer(r'[`“"]([^`”"]+\.md)[`”"]|(?<![\w.-])([\w./-]+\.md)(?![\w.-])', text, re.IGNORECASE):
        prefix = text[max(0, match.start() - 40):match.start()]
        if not re.search(r"(?:写入|生成|更新|保存|输出|创建|修改|恢复|编写|完善|补充|修订)[^\n。；]{0,40}$", prefix):
            continue
        candidate = match.group(1) or match.group(2)
        explicit = candidate if "/" in candidate or candidate != "README.md" else None
        break
    if explicit:
        rel = fs.safe(explicit).relative_to(fs.root).as_posix()
        require_delivery_path(fs, delivery_root(fs, state), rel)
        return rel
    changes = list(state.get("changes", []))
    readmes = [rel for rel in changes if Path(rel).name.lower() == "readme.md"]
    if len(readmes) == 1:
        require_delivery_path(fs, delivery_root(fs, state), readmes[0])
        return readmes[0]
    output_root = delivery_root(fs, state)
    if output_root != fs.root:
        return (output_root.relative_to(fs.root) / "README.md").as_posix()
    projects = [Path(rel).parent for rel in fs.list() if Path(rel).name in {"pyproject.toml", "package.json"}]
    affected = [path.as_posix() for path in projects if any(path == Path(".") or path in Path(rel).parents for rel in changes)]
    parent = Path(os.path.commonpath(affected)) if affected else Path(".")
    return (parent / "README.md").as_posix()


def doc(graph: AgentGraph, state: AgentState) -> AgentState:
    """文档智能体：根据任务、仓库摘要、变更和测试结果写中文文档。"""

    graph._emit(state, "doc", "start", "文档智能体正在生成或更新中文文档")
    if graph._guidance_requires_coding(state):
        graph._emit(state, "doc", "guidance", "文档生成前收到新指导，返回 Coding 重新规划")
        return state
    fs = FsTool(state["workdir"])
    try:
        path = _document_path(graph, state, fs)
        existing = fs.read(path, 80_000) if fs.safe(path).is_file() else ""
    except (OSError, ValueError) as exc:
        failure = {"cmd": "文档目标检查", "ok": False, "out": str(exc)}
        return {**state, "tests_ok": False, "tests": list(state.get("tests", [])) + [failure], "error": str(exc)}
    if existing and state.get("task_type") in {"code_gen", "code_mod"} and path in state.get("changes", []) and state.get("tests_ok"):
        repo_context = {**(state.get("repo") or {}), "doc_path": str(fs.safe(path))}
        graph._emit(state, "doc", "done", f"保留已交付并通过验证的文档：{path}")
        return {**state, "repo": repo_context}
    if existing and not fs.last_reads[path]["complete"]:
        failure = {"cmd": "文档完整性检查", "ok": False, "out": "已有文档超过读取上限，禁止使用局部上下文整体覆盖"}
        return {**state, "tests_ok": False, "tests": list(state.get("tests", [])) + [failure], "error": failure["out"]}
    client = graph._client("doc", state)
    try:
        planned_guidance_revision = int(state.get("guidance_revision", 0))
        content = graph._model_text(
            state,
            "doc",
            client,
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
                    + json.dumps(state.get("tests", []), ensure_ascii=False)
                    + f"\n\n文档目标：{path}\n\n已有文档（保留仍有效的信息）：\n{existing}",
                },
            ]
        ).strip()
    except LlmBalanceError:
        raise
    except LlmError as exc:
        state = graph._apply_guidance(state, checkpoint="文档模型失败后")
        if int(state.get("guidance_revision", 0)) != planned_guidance_revision:
            graph._emit(state, "doc", "guidance", "文档生成期间收到新指导，返回 Coding 重新规划")
            return state
        failure = {
            "cmd": "文档生成",
            "ok": False,
            "infra": True,
            "out": f"文档模型不可用：{graph._trim(str(exc), 160)}",
        }
        tests = list(state.get("tests", [])) + [failure]
        graph._emit(state, "doc", "error", failure["out"], data=failure)
        return {**state, "tests": tests, "tests_ok": False, "error": failure["out"]}
    state = graph._apply_guidance(state, checkpoint="文档模型返回后")
    if int(state.get("guidance_revision", 0)) != planned_guidance_revision:
        graph._emit(state, "doc", "guidance", "文档生成期间收到新指导，丢弃旧文档草稿并返回 Coding")
        return state
    content = graph.model_store.redact_text(content)
    if content.startswith("```") and content.endswith("```"):
        content = re.sub(r"^```(?:markdown|md)?\s*\n", "", content, count=1, flags=re.IGNORECASE)
        content = re.sub(r"\n```\s*$", "", content, count=1)
    if not content.strip():
        failure = {"cmd": "文档生成", "ok": False, "out": "文档模型返回空内容"}
        tests = list(state.get("tests", [])) + [failure]
        return {**state, "tests": tests, "tests_ok": False, "error": failure["out"]}
    written = fs.write(path, content.rstrip() + "\n")
    repo_context = dict(state.get("repo", {}))
    repo_context["doc_path"] = str((Path(state["workdir"]) / written).resolve())
    changes = sorted(dict.fromkeys(list(state.get("changes", [])) + [written]))
    doc_summary = f"文档已写入 {written}"
    graph._emit(state, "doc", "done", doc_summary)
    final_summary = state.get("coding_summary") or state.get("final") or doc_summary
    if state.get("coding_summary") and doc_summary not in final_summary:
        final_summary = f"{final_summary}\n{doc_summary}"
    return {**state, "repo": repo_context, "changes": changes, "final": final_summary}
