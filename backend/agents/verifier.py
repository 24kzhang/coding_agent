from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.agents.prompts import VERIFIER_PROMPT
from backend.agents.types import AgentState
from backend.tools import FsTool, ShellTool
from llm import LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


# 这些警告说明异步代码或测试替身实际没有执行完成，不能和普通弃用提示一样放行。
_CRITICAL_WARNING_MARKERS = (
    "RuntimeWarning: coroutine",
    "was never awaited",
    "Task was destroyed but it is pending",
)


def _reject_critical_warnings(result: dict[str, Any]) -> dict[str, Any]:
    """把会掩盖真实异步失败的运行时警告转换为验证失败。"""

    output = str(result.get("out") or "") + "\n" + str(result.get("err") or "")
    matched = [marker for marker in _CRITICAL_WARNING_MARKERS if marker in output]
    if not matched:
        return result
    return {
        **result,
        "ok": False,
        "issues": [f"检测到严重运行时警告：{marker}" for marker in matched],
    }


def verifier(graph: AgentGraph, state: AgentState) -> AgentState:
    """验证智能体：根据项目文件选择测试命令并执行。"""

    graph._emit(state, "verifier", "start", "验证智能体正在选择并运行测试")
    # fs 用于列出文件并做静态 Web 检查。
    fs = FsTool(state["workdir"])
    # shell 用于执行测试命令。
    shell = ShellTool(state["workdir"])
    # files 是当前项目文件列表。
    files = fs.list()
    # commands 根据项目真实配置生成，不把某个测试框架硬编码为所有项目默认值。
    commands = graph._verification_commands(fs, files)
    # tests 保存每条测试或静态检查的结构化结果。
    tests: list[dict[str, Any]] = []
    # ok 是整体验证状态，任意关键检查失败都会变为 False。
    ok = bool(state.get("coding_ok", True))

    if not state.get("coding_ok", True):
        tests.append(
            {
                "cmd": "Coding 完成状态",
                "ok": False,
                "out": state.get("coding_summary") or "Coding 智能体没有明确完成任务。",
            }
        )

    # static_check 是静态 Web 项目的轻量接线检查结果。
    static_check = graph._static_web_check(fs, files)
    if static_check:
        tests.append(static_check)
        ok = ok and bool(static_check.get("ok"))

    if not commands and not static_check:
        tests.append({"cmd": "静态检查", "ok": True, "out": "没有识别到可运行测试，已跳过。"})
    # 最多执行四条项目已有验证命令，避免无限扩展任务时间。
    for command in commands[:4]:
        # result 是命令执行结果。
        result = shell.run(command, timeout=240)
        result = _reject_critical_warnings(result)
        tests.append(result)
        ok = ok and bool(result.get("ok"))
        graph._emit(state, "verifier", "test", f"{command} -> {'通过' if result.get('ok') else '失败'}", data=result)

    # 代码变更和显式只读验收都追加需求对照审查，弥补编译和静态正则无法理解业务语义的问题。
    task_type = state.get("task_type")
    needs_review = bool(state.get("changes")) and task_type in {"code_gen", "code_mod"}
    needs_review = needs_review or task_type == "verify"
    if needs_review:
        review_files: list[str] = []
        budget = 60_000
        # verify 没有本轮 changes，改用仓库读取智能体选择的关键文件作为只读审查依据。
        review_targets = list(state.get("changes", [])) or list((state.get("repo") or {}).get("selected") or [])
        for rel in review_targets[:16]:
            if budget <= 0:
                break
            try:
                # 普通源码优先完整读取；过去固定截到 12K 会让模型把“上下文结尾”误判成
                # “磁盘文件被截断”。多文件总预算仍限制为 60K，单文件最多占 48K。
                read_limit = min(48_000, budget)
                observed = fs.read(rel, read_limit + 1)
            except (OSError, ValueError, UnicodeError):
                continue
            truncated = len(observed) > read_limit
            content = observed[:read_limit]
            if truncated:
                content += "\n[审查上下文在此截断，磁盘文件仍有后续内容；不得据此断言源码不完整。]"
            review_files.append(f"## {rel}\n{content}")
            budget -= len(content)
        try:
            client = graph._client("verifier", state)
            review = client.chat_json(
                [
                    {"role": "system", "content": VERIFIER_PROMPT},
                    {
                        "role": "user",
                        "content": graph._ctx_text(state)
                        + "\n\n仓库事实摘要（字段和路径以这里及变更文件为准，优先级高于计划假设）：\n"
                        + str((state.get("repo") or {}).get("snippets") or {})[:24_000]
                        + "\n\n变更文件：\n"
                        + "\n\n".join(review_files)
                        + "\n\n已运行检查：\n"
                        + str(tests),
                    },
                ],
                temperature=0,
                plain_text=True,
            )
            graph._add_tokens(state, client.last_usage.total)
            issues = [str(item).strip() for item in review.get("issues", []) if str(item).strip()][:12]
            review_ok = bool(review.get("ok")) and not issues
            review_result = {
                "cmd": "需求实现审查",
                "ok": review_ok,
                "out": str(review.get("summary") or ("通过" if review_ok else "发现需求实现问题")),
                "issues": issues,
            }
            tests.append(review_result)
            ok = ok and review_ok
            graph._emit(state, "verifier", "test", f"需求实现审查 -> {'通过' if review_ok else '失败'}", data=review_result)
        except (LlmError, KeyError) as exc:
            # 语义审查是代码任务的必要验收项；不可用时不能伪装为通过，也不能让 Coding 修复基础设施。
            tests.append(
                {
                    "cmd": "需求实现审查",
                    "ok": False,
                    "infra": True,
                    "out": f"审查模型不可用：{graph._trim(str(exc), 160)}",
                }
            )
            ok = False
    graph.memory.append(
        state["workdir"],
        state["session_id"],
        "verifier",
        "test",
        "summary",
        f"测试结果：{ok}",
        {"tests": tests},
    )
    return {**state, "tests": tests, "tests_ok": ok}
