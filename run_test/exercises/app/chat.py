"""聊天层：模型配置加载、会话存储、上游流式请求与 SSE 转发。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

BASE = Path(__file__).resolve().parent
CONFIG_FILE = BASE / ".local" / "model.json"

# 会话存储：{session_id: {"exercise_id": str, "messages": [...], "created_at": float}}
SESSIONS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()

MAX_MESSAGE_CHARS = 2000
# 短问答输出预算，最终受配置 max_output_tokens 钳制
CHAT_OUTPUT_BUDGET = 1024

SYSTEM_TEMPLATE = (
    "你是一名专业、耐心的健身教练，正在为用户解答动作相关问题。"
    "当前讨论的动作是《{name}》。\n"
    "动作描述：{desc}\n"
    "目标肌肉：{target}；协同肌肉：{secondary}；所需器械：{equipment}；分类：{category}。\n"
    "用户可能用“这个动作”指代它。回答使用简体中文，简洁分点，"
    "涉及伤病或疼痛时提醒用户咨询医生。"
)


class ConfigError(Exception):
    """模型配置缺失或字段不全。"""


def load_config() -> dict[str, Any]:
    """读取私密模型配置；只在内存中使用，不写日志、不回显。"""
    if not CONFIG_FILE.is_file():
        raise ConfigError("模型配置缺失，请检查 app/.local/model.json")
    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("模型配置缺失，请检查 app/.local/model.json") from exc
    required = ("base_url", "model", "api_key", "protocol")
    if any(not cfg.get(key) for key in required):
        raise ConfigError("模型配置缺失，请检查 app/.local/model.json")
    return cfg


def build_system_message(exercise: dict[str, Any]) -> dict[str, str]:
    desc = (exercise.get("instructions") or {}).get("zh") or "暂无数据"
    secondary = "、".join(exercise.get("secondary_muscles") or []) or "暂无数据"
    content = SYSTEM_TEMPLATE.format(
        name=exercise.get("name", ""),
        desc=desc,
        target=exercise.get("target") or "暂无数据",
        secondary=secondary,
        equipment=exercise.get("equipment") or "暂无数据",
        category=exercise.get("category") or "暂无数据",
    )
    return {"role": "system", "content": content}


def get_session(session_id: str, exercise_id: str, exercise: dict[str, Any]) -> dict[str, Any]:
    """取会话；不存在或动作不一致时重建，保证动作间上下文隔离。

    动作上下文（system 消息）在首次提问时由 append_user 注入，
    因此这里只登记会话骨架，messages 保持为空。
    """
    with LOCK:
        session = SESSIONS.get(session_id)
        if session is None or session.get("exercise_id") != exercise_id:
            session = {
                "exercise_id": exercise_id,
                "messages": [],
                "created_at": time.time(),
            }
            SESSIONS[session_id] = session
        return session


def ensure_session(session_id: str, exercise_id: str, exercise: dict[str, Any]) -> dict[str, Any]:
    """只读地确保会话存在（用于查询历史），不注入动作上下文。"""
    return get_session(session_id, exercise_id, exercise)


def history(session_id: str, exercise_id: str) -> list[dict[str, str]]:
    """返回用户可见的历史（不含 system 动作上下文），供前端恢复聊天记录。"""
    with LOCK:
        session = SESSIONS.get(session_id)
        if not session or session.get("exercise_id") != exercise_id:
            return []
        return [dict(m) for m in session["messages"] if m.get("role") in ("user", "assistant")]


def snapshot(session_id: str) -> list[dict[str, str]]:
    """返回发送给上游的消息快照（深拷贝，避免并发修改）。"""
    with LOCK:
        session = SESSIONS.get(session_id)
        if not session:
            return []
        return [dict(m) for m in session["messages"]]


def append_user(session_id: str, message: str, exercise: dict[str, Any] | None = None) -> None:
    """追加用户消息；会话首轮时先注入动作上下文（system），此后不再重复。"""
    with LOCK:
        session = SESSIONS[session_id]
        if not session["messages"] and exercise is not None:
            session["messages"].append(build_system_message(exercise))
        session["messages"].append({"role": "user", "content": message})


def drop_last_user(session_id: str, message: str) -> None:
    """回滚未被回答的用户消息（上游失败或客户端中断时调用），避免历史错位。"""
    with LOCK:
        session = SESSIONS.get(session_id)
        if not session:
            return
        msgs = session["messages"]
        if msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == message:
            msgs.pop()


def append_assistant(session_id: str, text: str) -> None:
    with LOCK:
        session = SESSIONS.get(session_id)
        if session is not None and text:
            session["messages"].append({"role": "assistant", "content": text})


def reset_session(session_id: str) -> None:
    with LOCK:
        SESSIONS.pop(session_id, None)


def turn_count(session_id: str) -> int:
    with LOCK:
        session = SESSIONS.get(session_id)
        if not session:
            return 0
        return sum(1 for m in session["messages"] if m["role"] == "user")


def build_payload(cfg: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    """按配置组装上游请求体：request_options 原样合入，预算受上限钳制。"""
    payload: dict[str, Any] = {"model": cfg["model"], "messages": messages, "stream": True}
    budget = CHAT_OUTPUT_BUDGET
    cap = cfg.get("max_output_tokens")
    if isinstance(cap, int) and cap > 0:
        budget = min(budget, cap)
    field = cfg.get("output_token_field") or "max_tokens"
    payload[field] = budget
    options = cfg.get("request_options")
    if isinstance(options, dict):
        payload.update(options)
    return payload


def endpoint(cfg: dict[str, Any]) -> str:
    base = str(cfg["base_url"]).rstrip("/")
    protocol = cfg.get("protocol") or "chat_completions"
    if protocol == "responses":
        return f"{base}/responses"
    return f"{base}/chat/completions"


def _extract_delta(chunk: dict[str, Any], protocol: str) -> str:
    if protocol == "responses":
        if chunk.get("type") in ("response.output_text.delta", "response.text.delta"):
            return chunk.get("delta") or ""
        return ""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


async def stream_upstream(
    cfg: dict[str, Any], messages: list[dict[str, str]]
) -> AsyncIterator[str]:
    """向上游发起流式请求，逐段产出文本增量。"""
    protocol = cfg.get("protocol") or "chat_completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(float(cfg.get("timeout") or 120), connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", endpoint(cfg), json=build_payload(cfg, messages), headers=headers
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise RuntimeError(f"上游返回 {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = _extract_delta(chunk, protocol)
                if text:
                    yield text


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
