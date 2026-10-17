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
            if status in {"invalid", "recover_json"}:
                message = {"tool_calls": [{"id": "broken", "type": "function", "function": {"name": "return_json", "arguments": "{"}}]} if data.get("tools") else {"content": '{"ok": true}' if status == "recover_json" else '{"broken":'}
                return httpx.Response(200, json={"choices": [{"message": message}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
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
@pytest.mark.parametrize("status", [401, 402, 503, "timeout"])
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


def test_balance_error_stops_when_no_capable_backup(tmp_path, monkeypatch):
    graph, calls, _events = setup_graph(tmp_path, monkeypatch, "text", 402)
    graph.unavailable_models.add("healthy")
    with pytest.raises(LlmBalanceError):
        state = {"model_id": "first", "workdir": str(tmp_path), "session_id": "balance", "attachments": [{"id": "pic"}]}
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


@pytest.mark.parametrize('role', ['manager', 'planner', 'coder', 'answer', 'verifier'])
def test_all_multimodal_roles_route_text_model_to_vision(tmp_path, role):
    store = ModelStore(tmp_path / 'models.json')
    for ident, caps in [('text', ['text', 'tools']), ('vision', ['text', 'tools', 'vision'])]:
        store.upsert(ModelConfig(id=ident, name='测试模型', model=ident, base_url='https://example.test/v1', api_key='TEST_ONLY_KEY', capabilities=caps))
    graph = AgentGraph(store, MemoryStore(tmp_path / 'mem'))
    state = {'workdir': str(tmp_path / 'work'), 'session_id': 'vision', 'attachments': [{'id': 'picture'}]}
    assert graph._client(role, state).cfg.id == 'vision'


def test_plan_revision_never_switches_image_task_to_text_model(tmp_path, monkeypatch):
    from backend.agents.planner import planner

    graph, _requests, _events = setup_graph(tmp_path, monkeypatch, 'json')
    calls = []
    monkeypatch.setattr(graph, '_ctx_text', lambda *_: '创建可验证的项目')
    monkeypatch.setattr('backend.agents.planner._plan_quality_issues', lambda _g, _s, md: [] if md == '完整合同' else ['合同未完整'])
    monkeypatch.setattr('backend.agents.planner._semantic_plan_issues', lambda *_: [])
    def reply(_state, _role, client, _messages, **_kwargs):
        calls.append(client.cfg.id)
        return {'status': 'plan', 'markdown': '完整合同' if client.cfg.id == 'healthy' else '初稿'}
    monkeypatch.setattr(graph, '_model_json', reply)
    result = planner(graph, {'session_id': 'imageplan', 'workdir': str(tmp_path / 'work'), 'text': '创建项目', 'attachments': [{'id': 'pic', 'name': '参考图'}]})
    assert result['plan']['markdown'] == '完整合同'
    assert calls == ['first', 'first', 'healthy']


def test_invalid_json_switches_after_one_repair_without_losing_task(tmp_path, monkeypatch):
    graph, calls, events = setup_graph(tmp_path, monkeypatch, 'json', 'invalid')
    state = {'workdir': str(tmp_path / 'work'), 'session_id': 'jsonfallback', 'model_id': 'first', 'attachments': [{'id': 'pic'}]}
    messages = [{'role': 'user', 'content': '必须保留的原始任务'}]
    assert graph._model_json(state, 'planner', graph._client('planner', state), messages) == {'ok': True}
    assert [item['model'] for item in calls] == ['first', 'first', 'healthy']
    assert calls[-1]['messages'][0] == messages[0]
    assert graph._client('planner', state).cfg.id == 'healthy'
    assert not any(event.kind == 'error' for event in events)


def test_readonly_json_recovery_changes_protocol_and_reuses_it(tmp_path, monkeypatch):
    graph, calls, events = setup_graph(tmp_path, monkeypatch, 'json', 'recover_json')
    state = {'workdir': str(tmp_path / 'work'), 'session_id': 'protocol', 'model_id': 'first'}
    messages = [{'role': 'user', 'content': '完整原始合同'}]
    client = graph._client('planner', state)
    assert graph._model_json(state, 'planner', client, messages) == {'ok': True}
    assert graph._model_json(state, 'planner', client, messages) == {'ok': True}
    assert len(calls) == 3 and calls[0].get('tools')
    assert all(call.get('response_format') == {'type': 'json_object'} and not call.get('tools') for call in calls[1:])
    assert all(call['messages'][0] == messages[0] for call in calls)
    assert not any(event.kind == 'error' for event in events)
