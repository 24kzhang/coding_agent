"""原生函数流必须完整收齐且正确终止，任何残片都不能成为可执行动作。"""
import json

import httpx
import pytest

from api.schema import ModelConfig
from llm.client import LlmClient, LlmError, LlmOutputLimitError


def make_client(monkeypatch, frames):
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        body = "\n\n".join("data: " + (frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)) for frame in frames) + "\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    original = httpx.Client
    monkeypatch.setattr("llm.client.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    cfg = ModelConfig(id="test", name="测试", model="generic", base_url="https://example.test/v1", api_key="TEST_ONLY_KEY", protocol="chat_completions", stream_tools=True)
    return LlmClient(cfg), payloads


def choice(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def test_native_stream_reassembles_identity_arguments_reasoning_and_usage(monkeypatch):
    frames = [
        choice({"reasoning_content": "内部判断"}),
        choice({"tool_calls": [{"index": 0, "id": "call_real", "function": {"name": "write_", "arguments": '{"path":"main.py",'}}]}),
        choice({"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": '"content":"完成"}'}}]}),
        choice({}, "tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 4}}},
        "[DONE]",
    ]
    client, payloads = make_client(monkeypatch, frames)
    result = client.chat_tools([], [])
    assert result["actions"] == [{"tool": "write_file", "path": "main.py", "content": "完成"}]
    assert result["native_message"]["tool_calls"][0]["id"] == "call_real"
    assert result["native_message"]["reasoning_content"] == "内部判断"
    assert client.last_usage.total == 70 and client.last_usage.reasoning == 4
    assert payloads[0]["stream"] is True and payloads[0]["stream_options"]["include_usage"] is True


@pytest.mark.parametrize("ending,error", [("missing", LlmError), ("length", LlmOutputLimitError), ("upstream", LlmError)])
def test_partial_native_stream_is_never_returned_as_actions(monkeypatch, ending, error):
    frames = [choice({"tool_calls": [{"index": 0, "id": "partial", "function": {"name": "write_file", "arguments": '{"path":"main.py"'}}]})]
    if ending == "length":
        frames.append(choice({}, "length"))
    elif ending == "upstream":
        frames.append({"error": {"message": "上游失败", "type": "server_error"}})
    frames.append("[DONE]")
    client, _payloads = make_client(monkeypatch, frames)
    with pytest.raises(error):
        client.chat_tools([], [])
