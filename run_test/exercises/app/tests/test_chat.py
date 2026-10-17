"""聊天上下文与流式转发测试（替换上游传输边界）。"""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

import chat
import data
import main

client = TestClient(main.app)

CFG = {
    "base_url": "https://example.invalid/v1",
    "model": "TEST_ONLY_MODEL",
    "api_key": "TEST_ONLY_KEY",
    "protocol": "chat_completions",
    "timeout": 30,
    "max_output_tokens": 16384,
    "request_options": {"temperature": 0.3},
    "output_token_field": "max_tokens",
}


@pytest.fixture(autouse=True)
def clean_sessions(monkeypatch):
    chat.SESSIONS.clear()
    monkeypatch.setattr(chat, "load_config", lambda: dict(CFG))
    yield
    chat.SESSIONS.clear()


def fake_upstream(captured: list, chunks: list[str]):
    async def gen(cfg, messages):
        captured.append(copy.deepcopy(messages))
        for text in chunks:
            yield text
    return gen


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        event = "message"
        data_line = ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_line += line[5:].strip()
        events.append((event, json.loads(data_line)))
    return events


def test_first_turn_injects_system_with_name_and_desc(monkeypatch) -> None:
    captured: list = []
    monkeypatch.setattr(chat, "stream_upstream", fake_upstream(captured, ["主要", "练腹肌"]))
    res = client.post(
        "/api/chat",
        json={"session_id": "s-1", "exercise_id": "0001", "message": "这个动作主要练哪里？"},
    )
    assert res.status_code == 200
    events = parse_sse(res.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta"
    assert "delta" in kinds and kinds[-1] == "done"
    assert events[0][1]["turn"] == 1

    sent = captured[0]
    assert sent[0]["role"] == "system"
    assert "3/4仰卧起坐" in sent[0]["content"]
    assert "平躺，膝盖弯曲" in sent[0]["content"]
    assert sent[-1] == {"role": "user", "content": "这个动作主要练哪里？"}


def test_second_turn_sends_full_history_without_repeat(monkeypatch) -> None:
    captured: list = []
    monkeypatch.setattr(chat, "stream_upstream", fake_upstream(captured, ["第一轮回答"]))
    client.post("/api/chat", json={"session_id": "s-2", "exercise_id": "0001", "message": "第一问"})
    monkeypatch.setattr(chat, "stream_upstream", fake_upstream(captured, ["第二轮回答"]))
    res = client.post("/api/chat", json={"session_id": "s-2", "exercise_id": "0001", "message": "那新手要注意什么？"})
    assert res.status_code == 200

    first, second = captured[0], captured[1]
    assert sum(1 for m in second if m["role"] == "system") == 1
    assert second[0] == first[0]
    assert second[1:] == [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "那新手要注意什么？"},
    ]
    assert second[: len(first)] == first


def test_switching_exercise_rebuilds_session(monkeypatch) -> None:
    captured: list = []
    monkeypatch.setattr(chat, "stream_upstream", fake_upstream(captured, ["好的"]))
    client.post("/api/chat", json={"session_id": "s-3", "exercise_id": "0001", "message": "问一"})
    client.post("/api/chat", json={"session_id": "s-3", "exercise_id": "0002", "message": "问二"})
    second = captured[1]
    assert len(second) == 2
    assert "45°侧弯" in second[0]["content"]
    assert "3/4仰卧起坐" not in second[0]["content"]


def test_upstream_failure_does_not_store_assistant(monkeypatch) -> None:
    async def boom(cfg, messages):
        yield "半截"
        raise RuntimeError("上游异常")

    monkeypatch.setattr(chat, "stream_upstream", boom)
    res = client.post("/api/chat", json={"session_id": "s-4", "exercise_id": "0001", "message": "问"})
    events = parse_sse(res.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["message"] == "模型服务暂时不可用，请稍后重试"
    # 失败轮次回滚未被回答的用户消息，历史保持首轮注入后的干净状态
    roles = [m["role"] for m in chat.snapshot("s-4")]
    assert roles == ["system"]
    assert chat.history("s-4", "0001") == []


def test_reset_clears_session(monkeypatch) -> None:
    captured: list = []
    monkeypatch.setattr(chat, "stream_upstream", fake_upstream(captured, ["好"]))
    client.post("/api/chat", json={"session_id": "s-5", "exercise_id": "0001", "message": "问"})
    assert client.post("/api/chat/reset", json={"session_id": "s-5"}).json() == {"ok": True}
    assert chat.snapshot("s-5") == []


def test_payload_budget_and_options() -> None:
    payload = chat.build_payload(CFG, [{"role": "user", "content": "hi"}])
    assert payload["model"] == "TEST_ONLY_MODEL"
    assert payload["stream"] is True
    assert payload["max_tokens"] == chat.CHAT_OUTPUT_BUDGET
    assert payload["temperature"] == 0.3
    tight = dict(CFG, max_output_tokens=128)
    assert chat.build_payload(tight, [])["max_tokens"] == 128


def test_endpoint_by_protocol() -> None:
    assert chat.endpoint(CFG).endswith("/chat/completions")
    assert chat.endpoint(dict(CFG, protocol="responses")).endswith("/responses")


def test_missing_config_returns_error_event(monkeypatch) -> None:
    def missing():
        raise chat.ConfigError("模型配置缺失，请检查 app/.local/model.json")

    monkeypatch.setattr(chat, "load_config", missing)
    res = client.post("/api/chat", json={"session_id": "s-6", "exercise_id": "0001", "message": "问"})
    events = parse_sse(res.text)
    assert events[0][0] == "error"
    assert "模型配置缺失" in events[0][1]["message"]


def test_unknown_exercise_and_long_message(monkeypatch) -> None:
    assert client.post("/api/chat", json={"session_id": "s", "exercise_id": "9999", "message": "问"}).status_code == 404
    long_msg = "字" * (chat.MAX_MESSAGE_CHARS + 1)
    res = client.post("/api/chat", json={"session_id": "s", "exercise_id": "0001", "message": long_msg})
    assert res.status_code == 422


def test_system_message_uses_real_data() -> None:
    item = data.get_exercise("0001")
    msg = chat.build_system_message(item)
    assert item["name"] in msg["content"]
    assert item["instructions"]["zh"] in msg["content"]
