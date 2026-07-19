from __future__ import annotations

import json
from typing import TYPE_CHECKING

from backend.agents.prompts import ANSWER_PROMPT
from backend.agents.types import AgentState
from llm import LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


def answer(graph: AgentGraph, state: AgentState) -> AgentState:
    """答疑智能体：只回答问题，不写入磁盘。"""

    graph._emit(state, "answer", "start", "答疑智能体正在生成回答")
    # answer 复用 manager 模型配置，因为它也是轻量回答类任务。
    client = graph._client("manager", state)
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
    graph.memory.append(state["workdir"], state["session_id"], "answer", "llm", "summary", text)
    return {**state, "final": text}
