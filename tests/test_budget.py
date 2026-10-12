"""当供应商把思考等级用作总生成预算时，恢复不能把可用空间反向缩小。"""
import json

import httpx
import pytest

from api.schema import ModelConfig
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.client import LlmClient, LlmOutputLimitError
from llm.store import ModelStore


def mapped_config():
    return ModelConfig(id="mapped", name="预算模型", model="generic", base_url="https://example.test/v1", api_key="TEST_ONLY_KEY", protocol="chat_completions", efforts=["low", "medium", "high"], default_effort="low", reasoning_style="effort", effort_map={"low": "light", "medium": "balanced", "high": "deep"}, effort_budgets={"low": 4096, "medium": 16384, "high": 65536})


def test_shared_budget_recovers_with_real_wire_budget(tmp_path, monkeypatch):
    cfg = mapped_config()
    store = ModelStore(tmp_path / "models.json")
    store.upsert(cfg)
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if body["reasoning_effort"] == "light":
            return httpx.Response(200, json={"choices": [{"message": {"tool_calls": []}, "finish_reason": "length"}], "usage": {"prompt_tokens": 30, "completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}}})
        args = json.dumps({"result": {"ok": True}})
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"id": "json1", "function": {"name": "return_json", "arguments": args}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 30, "completion_tokens": 50}})

    original = httpx.Client
    monkeypatch.setattr("llm.client.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    graph = AgentGraph(store, MemoryStore(tmp_path / "mem"))
    graph._ensure_mcp_tools = lambda *_args, **_kwargs: []
    state = {"session_id": "s1", "workdir": str(tmp_path)}
    assert graph._model_json(state, "verifier", LlmClient(cfg), [{"role": "user", "content": "审查当前合同"}])["ok"]
    assert [(body["reasoning_effort"], body["max_tokens"]) for body in sent] == [("light", 6000), ("balanced", 12000)]
    assert state["output_tokens"] == 4146


@pytest.mark.parametrize("arguments", ['{"result":', '{}'])
def test_invalid_structured_call_at_effective_budget_is_output_limit(monkeypatch, arguments):
    cfg = mapped_config()
    original = httpx.Client
    response = {"choices": [{"message": {"tool_calls": [{"id": "partial", "function": {"name": "return_json", "arguments": arguments}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 30, "completion_tokens": 4096}}
    monkeypatch.setattr("llm.client.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=response)), **kwargs))
    with pytest.raises(LlmOutputLimitError, match="有效生成预算"):
        LlmClient(cfg).chat_json([], structured_tools=True, max_tokens=6000)
