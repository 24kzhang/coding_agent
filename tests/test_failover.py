"""模型不可用时继续当前合同，并保留视觉能力与实际换模记录。"""
import json

import httpx
import pytest

from api.schema import ModelConfig
from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.client import LlmBalanceError
from llm.store import ModelStore


def setup_graph(tmp_path, monkeypatch, mode, status=503):
    store = ModelStore(tmp_path / "models.json")
    for ident, caps in [("first", ["text", "tools", "vision"]), ("textonly", ["text"]), ("healthy", ["text", "tools", "vision"])]:
        store.upsert(ModelConfig(id=ident, name="测试模型", model=ident, base_url="https://example.test/v1", api_key="TEST_ONLY_KEY", protocol="chat_completions", capabilities=caps, default_effort="medium"))
    calls = []

    def handler(request):
        data = json.loads(request.content)
        calls.append(data)
        if data["model"] == "first":
            if status == "timeout":
                raise httpx.ReadTimeout("读取超时", request=request)
            return httpx.Response(status, json={"error": "服务不可用"})
        if mode == "json":
            message = {"tool_calls": [{"id": "json1", "type": "function", "function": {"name": "return_json", "arguments": json.dumps({"result": {"ok": True}})}}]}
        elif mode == "coder":
            message = {"tool_calls": [{"id": "finish1", "type": "function", "function": {"name": "finish", "arguments": json.dumps({"summary": "检查完成"})}}]}
        else:
            message = {"content": "已完成原合同"}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    original = httpx.Client
    monkeypatch.setattr("llm.client.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    events = []
    graph = AgentGraph(store, MemoryStore(tmp_path / "mem"), emit=events.append)
    graph._ensure_mcp_tools = lambda *_args, **_kwargs: []
    return graph, calls, events


@pytest.mark.parametrize("mode", ["json", "text", "coder"])
@pytest.mark.parametrize("status", [401, 503, "timeout"])
def test_explicit_primary_fails_over_without_losing_contract(tmp_path, monkeypatch, mode, status):
    graph, calls, events = setup_graph(tmp_path, monkeypatch, mode, status)
    state = {"session_id": "s1", "workdir": str(tmp_path / "project"), "text": "检查仓库", "model_id": "first", "attachments": [{"id": "pic"}]}
    messages = [{"role": "user", "content": "原始任务合同"}]
    if mode == "coder":
        graph._multimodal_content = lambda text, *_args: text
        assert coder(graph, state)["coding_ok"]
    elif mode == "json":
        assert graph._model_json(state, "planner", graph._client("planner", state), messages) == {"ok": True}
    else:
        assert graph._model_text(state, "answer", graph._client("answer", state), messages) == "已完成原合同"
    expected = ["first", "first", "healthy"] if mode == "text" and status == 503 else ["first", "healthy"]
    assert [call["model"] for call in calls] == expected
    assert calls[-1]["reasoning_effort"] == "medium"
    if mode == "coder":
        assert calls[-1]["messages"][:2] == calls[0]["messages"][:2]
        assert "尚未执行成功工具" in calls[-1]["messages"][2]["content"]
    else:
        assert calls[-1]["messages"] == calls[0]["messages"]
    assert any(event.kind == "model" and "healthy" in event.msg for event in events)
    assert not any(event.kind == "error" for event in events)
    assert graph._client("planner", state).cfg.id == "healthy"


def test_balance_error_preserves_stop_semantics(tmp_path, monkeypatch):
    graph, calls, _events = setup_graph(tmp_path, monkeypatch, "text", 402)
    with pytest.raises(LlmBalanceError):
        state = {"model_id": "first", "workdir": str(tmp_path), "session_id": "balance"}
        graph._model_text(state, "answer", graph._client("answer", state), [{"role": "user", "content": "测试"}])
    assert len(calls) == 1


def test_reused_node_client_does_not_retry_known_unavailable_model(tmp_path, monkeypatch):
    """调用方保留旧客户端进行计划修订时，也应沿用本轮换模结果。"""

    graph, calls, _events = setup_graph(tmp_path, monkeypatch, "json")
    state = {"model_id": "first", "workdir": str(tmp_path), "session_id": "reuse", "attachments": [{"id": "pic"}]}
    client = graph._client("planner", state)
    messages = [{"role": "user", "content": "修订原合同"}]
    assert graph._model_json(state, "planner", client, messages)["ok"]
    assert graph._model_json(state, "planner", client, messages)["ok"]
    assert [row["model"] for row in calls] == ["first", "healthy", "healthy"]
