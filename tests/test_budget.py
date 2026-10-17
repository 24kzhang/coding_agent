"""当供应商把思考等级用作总生成预算时，恢复不能把可用空间反向缩小。"""
import json

import httpx
import pytest

from api.schema import ModelConfig
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.client import LlmClient, LlmError, LlmOutputLimitError, LlmUsage
from llm.store import ModelStore


def test_unconfigured_cost_limits_do_not_interrupt_useful_work(tmp_path, monkeypatch):
    """累计用量再大也只是统计，不能悄悄启用未配置的成本上限。"""
    from backend.agents import graph as module
    for name in ("MAX_MODEL_CALLS", "MAX_UNCACHED_INPUT_TOKENS", "MAX_OUTPUT_TOKENS"):
        monkeypatch.setattr(module, name, 0)
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    state = {"session_id": "s1", "workdir": str(tmp_path), "model_calls": 100,
             "uncached_input_tokens": 2_000_000, "output_tokens": 200_000}
    assert graph._check_model_budget(state, 12_000) == 12_000
    graph._add_tokens(state, LlmUsage(prompt=80_000, completion=2_000, cache_miss=80_000))
    assert state["uncached_input_tokens"] == 2_080_000
    assert state["output_tokens"] == 202_000


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("1200", 1200)])
def test_cost_limit_requires_explicit_configuration(monkeypatch, value, expected):
    from backend.agents.graph import _cost_limit
    name = "CODING_AGENT_TEST_LIMIT"
    monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(name, value)
    assert _cost_limit(name) == expected


@pytest.mark.parametrize("value", ["invalid", "-1"])
def test_invalid_cost_limit_is_not_silently_enabled(monkeypatch, value):
    from backend.agents.graph import _cost_limit
    monkeypatch.setenv("CODING_AGENT_TEST_LIMIT", value)
    with pytest.raises(ValueError, match="非负整数"):
        _cost_limit("CODING_AGENT_TEST_LIMIT")


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


def test_recovery_does_call_third_model_after_format_and_budget_failures(tmp_path, monkeypatch):
    """格式、预算恢复消耗的次数不能让已宣布的第三个备用模型从未被调用。"""

    store = ModelStore(tmp_path / "models.json")
    for ident in ["first", "second", "third"]:
        store.upsert(ModelConfig(id=ident, name="测试", model=ident, base_url="https://example.test", api_key="TEST_ONLY_KEY"))
    calls = []
    counts = {}

    class Client(LlmClient):
        def chat_json(self, _messages, **kwargs):
            ident = self.cfg.id
            calls.append(ident)
            counts[ident] = counts.get(ident, 0) + 1
            if ident == "third":
                self.last_usage = LlmUsage(prompt=10, completion=50)
                return {"ok": True}
            if counts[ident] == 1:
                self.last_usage = LlmUsage(prompt=10, completion=50)
                raise LlmError("结构化函数没有返回唯一的 result 对象")
            self.last_usage = LlmUsage(prompt=10, completion=kwargs["max_tokens"])
            raise LlmOutputLimitError("输出预算触顶")

    monkeypatch.setattr("backend.agents.graph.LlmClient", Client)
    graph = AgentGraph(store, MemoryStore(tmp_path / "mem"))
    graph._ensure_mcp_tools = lambda *_args, **_kwargs: []
    state = {"session_id": "s1", "workdir": str(tmp_path)}
    assert graph._model_json(state, "verifier", Client(store.get("first")), [{"role": "user", "content": "返回审查结论"}])["ok"]
    assert calls == ["first", "first", "first", "second", "second", "second", "third"]
