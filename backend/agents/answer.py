from __future__ import annotations

import json
from typing import TYPE_CHECKING

from backend.agents.prompts import ANSWER_PROMPT, MCP_ANSWER_PROMPT
from backend.agents.types import AgentState
from llm import LlmClient, LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def answer(graph: AgentGraph, state: AgentState) -> AgentState:
    """答疑智能体：可按需调用所选 MCP，但不写入用户项目磁盘。"""

    graph._emit(state, "answer", "start", "答疑智能体正在生成回答")
    # answer 复用 manager 模型配置，因为它也是轻量回答类任务。
    client = graph._client("manager", state)
    # 用户没有选择 MCP 时保持原来的单轮答疑，不连接任何外部服务。
    if not state.get("mcp_servers"):
        text = _plain_answer(graph, state, client)
        graph.memory.append(state["workdir"], state["session_id"], "answer", "llm", "summary", text)
        return {**state, "final": text}

    # 用户选择 MCP 后先执行真实 tools/list；结果写入 state，供答疑 ReAct 使用。
    mcp_tools = graph._ensure_mcp_tools(state)
    # mcp_prompt 只包含安全动态工具名、说明和输入 Schema，不暴露服务命令或 token。
    mcp_prompt = graph._mcp_prompt(state)
    # observations 保存最近的 MCP 调用结果，防止模型忘记已经发生的外部事实。
    observations: list[str] = []
    # seen_actions 保存已经执行的工具和参数签名，防止模型重复调用并浪费 token。
    seen_actions: set[str] = set()
    # final_text 是模型完成 ReAct 后给用户的最终回答。
    final_text = ""
    # model_name 用于兼容 LongCat 原生工具标签协议。
    model_name = str(getattr(getattr(client, "cfg", None), "model", "")).lower()

    # 最多 6 轮足以完成查询、读取详情和总结，同时避免外部工具失控循环。
    for step in range(1, 7):
        # messages 每轮都携带当前任务、真实工具定义和最近观察，不发送完整会话历史。
        messages = [
            {
                "role": "system",
                "content": ANSWER_PROMPT + "\n\n" + MCP_ANSWER_PROMPT + "\n\n" + mcp_prompt,
            },
            {
                "role": "user",
                "content": graph._ctx_text(state)
                + "\n\n仓库摘要：\n"
                + json.dumps(state.get("repo", {}), ensure_ascii=False)[:12000]
                + f"\n\n当前是 MCP ReAct 第 {step}/6 轮。最近工具观察：\n"
                + ("\n\n".join(observations[-4:]) if observations else "尚未调用 MCP 工具。"),
            },
        ]
        try:
            # LongCat 普通文本模式可稳定返回原生工具标签，其他模型优先使用 JSON 输出模式。
            if isinstance(client, LlmClient):
                data = client.chat_json(
                    messages,
                    plain_text="longcat" in model_name,
                    max_tokens=3000,
                )
            else:
                data = client.chat_json(messages, plain_text="longcat" in model_name)
            graph._add_tokens(state, client.last_usage.total)
        except LlmError as exc:
            final_text = f"已选择的 MCP 工具发现完成，但答疑模型无法继续处理：{exc}"
            break

        # 只保留本轮 tools/list 发现的 mcp__ 动态工具，拒绝文件、shell 和 Git 动作。
        actions = [
            item
            for item in graph._normalize_actions(data.get("actions"), mcp_tools)
            if str(item.get("tool") or "").startswith("mcp__")
        ][:2]
        if not actions and data.get("done"):
            final_text = _answer_summary(data.get("summary"), "MCP 查询已完成。")
            break
        if not actions:
            observations.append("模型没有给出 MCP 动作或完成回答；请直接总结 tools/list，或调用一个已列出的 MCP 工具。")
            continue

        for action in actions:
            # signature 使用稳定 JSON，同一工具和参数即使字段顺序不同也视为重复。
            signature = json.dumps(action, ensure_ascii=False, sort_keys=True)
            if signature in seen_actions:
                # LongCat 偶尔会忽略已有观察并重复同一工具，此时切换普通文本模式强制总结。
                final_text = _synthesize_mcp_answer(graph, state, client, observations)
                break
            seen_actions.add(signature)
            # result 已经过服务白名单、动态工具白名单和输入 Schema 类型恢复。
            result = graph._call_mcp_action(action, state)
            # observation 限长后反馈下一轮，完整 token 和服务配置不会进入模型上下文。
            observation = graph._trim(str(result.get("text") or "MCP 工具没有返回文本"), 12000)
            observations.append(observation)
            graph._emit(
                state,
                "mcp",
                "tool",
                f"已调用 {action['tool']}：{'成功' if result.get('ok') else '失败'}",
            )
        if final_text:
            break

    if not final_text:
        # 达到轮次上限时明确返回现状，不能继续凭空生成外部数据。
        final_text = (
            "MCP 工具已连接，但答疑智能体在 6 轮内没有形成最终回答。"
            + (f"\n\n最近结果：{observations[-1]}" if observations else "")
        )
    graph.memory.append(state["workdir"], state["session_id"], "answer", "llm", "summary", final_text)
    return {**state, "final": final_text}


def _plain_answer(graph: AgentGraph, state: AgentState, client: object) -> str:
    """处理未选择 MCP 的普通答疑，保持原有单轮、无工具行为。"""

    # messages 包含答疑 prompt、Context Package 和可选仓库摘要。
    messages = [
        {"role": "system", "content": ANSWER_PROMPT},
        {
            "role": "user",
            "content": graph._ctx_text(state)
            + "\n\n仓库摘要：\n"
            + json.dumps(state.get("repo", {}), ensure_ascii=False)[:16000],
        },
    ]
    try:
        # text 是模型生成的最终回答。
        text = client.chat(messages, temperature=0.2).strip()
        graph._add_tokens(state, client.last_usage.total)
    except LlmError as exc:
        # 模型失败时仍返回明确错误，不进入 Coding。
        text = f"我现在无法完成解释，因为模型调用失败：{exc}"
    return text


def _answer_summary(value: object, fallback: str) -> str:
    """清洗 MCP 最终回答并保留开头；通用 _trim 取末尾，不能用于面向用户的长回复。"""

    # text 保留 Markdown 结构，只压缩过多空行并设置宽松的最终回答上限。
    text = str(value or "").strip()
    if not text:
        return fallback
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    if len(text) > 12000:
        return text[:12000] + "\n\n[回答已按长度上限截断]"
    return text


def _synthesize_mcp_answer(
    graph: AgentGraph,
    state: AgentState,
    client: object,
    observations: list[str],
) -> str:
    """工具阶段结束后用普通文本模式总结，避免结构化 ReAct 再次请求同一动作。"""

    messages = [
        {
            "role": "system",
            "content": (
                ANSWER_PROMPT
                + "\n工具调用阶段已经结束。只根据下面的真实工具结果回答用户，"
                "不要输出 JSON、工具标签或新的工具调用。"
            ),
        },
        {
            "role": "user",
            "content": graph._ctx_text(state)
            + "\n\n真实 MCP 工具结果：\n"
            + "\n\n".join(observations[-4:]),
        },
    ]
    try:
        # 普通 chat 用于最终自然语言回答，不再解析任何工具协议。
        text = client.chat(messages, temperature=0.2, max_tokens=3000)
        graph._add_tokens(state, client.last_usage.total)
        return _answer_summary(text, "MCP 查询已完成，但模型没有返回总结。")
    except LlmError as exc:
        return f"MCP 工具调用成功，但最终总结模型失败：{exc}"
