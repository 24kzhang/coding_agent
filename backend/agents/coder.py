from __future__ import annotations

import json
from typing import TYPE_CHECKING

from backend.agents.prompts import CODER_PROMPT
from backend.agents.types import AgentState
from backend.tools import FsTool, GitTool, ShellTool
from llm import LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def coder(graph: AgentGraph, state: AgentState) -> AgentState:
    """Coding 智能体：通过 ReAct 循环调用工具，真实修改项目文件。"""

    # retry 是当前 Coding 轮次，验证失败回到 coder 时会递增。
    retry = int(state.get("retry", 0))
    graph._emit(state, "coder", "start", f"Coding 智能体开始 ReAct 执行，第 {retry + 1} 轮")
    # fs/shell/git 是 Coding 智能体可调用的工具集合。
    fs = FsTool(state["workdir"])
    shell = ShellTool(state["workdir"])
    git = GitTool(state["workdir"])
    # client 是 coder 智能体对应的模型客户端。
    client = graph._client("coder", state)
    # observations 保存工具执行结果，下一步会回填给模型作为观察。
    observations: list[str] = []
    # changes 继承已有变更列表，验证失败重试时不会丢掉前一轮变更记录。
    changes = list(state.get("changes", []))
    # commands 继承已有命令列表。
    commands = list(state.get("commands", []))
    if state.get("tests") and not state.get("tests_ok", True):
        # 如果上一轮验证失败，把测试结果作为观察反馈给模型修复。
        observations.append("上一轮测试失败：\n" + json.dumps(state["tests"], ensure_ascii=False, indent=2))

    # coding_ok 只有模型明确 done 且最后一批动作都成功时才会变为 True。
    coding_ok = False
    # coding_summary 保存模型明确给出的完成摘要或循环失败原因。
    coding_summary = ""
    # parse_failures 统计连续结构化输出失败次数，偶发格式错误不会立刻终止任务。
    parse_failures = 0
    # 单轮 Coding 最多执行 10 次模型-工具循环，兼顾真实项目能力和失控保护。
    for step in range(1, 11):
        # messages 是本轮发给 Coding 模型的上下文。
        messages = [
            {"role": "system", "content": CODER_PROMPT},
            {
                "role": "user",
                "content": graph._ctx_text(state)
                + "\n\n仓库摘要：\n"
                + json.dumps(state.get("repo", {}), ensure_ascii=False)[:25000]
                + "\n\n已有观察：\n"
                + "\n".join(observations[-12:]),
            },
        ]
        try:
            # data 是模型返回的 ReAct JSON。
            data = client.chat_json(messages)
            graph._add_tokens(state, client.last_usage.total)
        except LlmError as exc:
            parse_failures += 1
            # error_text 只保留错误前 240 字符，完整模型残片不会进入前端和记忆。
            error_text = graph._trim(str(exc), 240)
            observations.append(f"第 {step} 轮输出格式无效：{error_text}。请严格返回约定 JSON。")
            graph._emit(state, "coder", "error", f"模型输出格式无效，正在重试（{parse_failures}/2）")
            if parse_failures >= 2:
                coding_summary = "Coding 模型连续两次没有返回有效执行指令。"
                break
            continue
        parse_failures = 0
        # thought 是模型本轮判断，会进入事件流方便用户观察。
        thought = data.get("thought", "")
        if thought:
            graph._emit(state, "coder", "thought", thought)
        # actions 是模型请求执行的工具动作列表。
        actions = graph._normalize_actions(data.get("actions"))
        if not actions and data.get("done"):
            coding_ok = True
            coding_summary = graph._clean_summary(data.get("summary"), "Coding 阶段已完成。")
            break
        if not actions:
            observations.append("本轮既没有工具动作也没有完成标记，请读取、修改或验证后再继续。")
            continue
        # action_failed 表示本批动作至少有一个失败，模型必须观察并修复后才能 done。
        action_failed = False
        for action in actions:
            # observation 是工具执行后的结构化观察。
            observation = graph._do_action(action, fs, shell, git)
            observations.append(observation["text"])
            action_failed = action_failed or not bool(observation.get("ok"))
            if observation.get("file"):
                changes.append(str(observation["file"]))
            if observation.get("cmd"):
                commands.append(str(observation["cmd"]))
            graph._emit(state, "coder", "tool", observation["text"], data=observation)
        if data.get("done"):
            if action_failed:
                observations.append("本轮存在失败动作，不能标记完成；请根据观察修复。")
                continue
            coding_ok = True
            coding_summary = graph._clean_summary(data.get("summary"), "Coding 阶段已完成。")
            graph._emit(state, "coder", "done", coding_summary)
            break
    if not coding_ok and not coding_summary:
        coding_summary = "Coding 在最大 ReAct 步数内没有明确完成任务。"
    # unique_changes 去重并排序，保证最终结果稳定。
    unique_changes = sorted(dict.fromkeys(changes))
    graph.memory.append(
        state["workdir"],
        state["session_id"],
        "coder",
        "react",
        "summary",
        f"变更文件：{unique_changes}",
        {"commands": commands, "ok": coding_ok, "summary": coding_summary},
    )
    return {
        **state,
        "changes": unique_changes,
        "commands": commands,
        "retry": retry + 1,
        "coding_ok": coding_ok,
        "coding_summary": coding_summary,
        "error": "" if coding_ok else coding_summary,
    }
