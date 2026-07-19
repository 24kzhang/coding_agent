from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.agents.types import AgentState
from backend.tools import FsTool, ShellTool

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


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
        tests.append(result)
        ok = ok and bool(result.get("ok"))
        graph._emit(state, "verifier", "test", f"{command} -> {'通过' if result.get('ok') else '失败'}", data=result)
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
