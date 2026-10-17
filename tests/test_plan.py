"""执行失败后复用原 Plan，不依赖最近事件条数。"""

import pytest

from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.store import ModelStore


@pytest.mark.parametrize("kind", ["pending_plan", "plan_done"])
def test_plan_state_survives_execution_events_compression_and_restart(tmp_path, kind):
    root = tmp_path / "mem"
    work = str(tmp_path / "work")
    store = MemoryStore(root)
    plan = {"goal": "实现通用项目", "markdown": "已通过的完整计划", "path": "计划.md"}
    store.append(work, "s1", "planner", "state", kind, "计划状态", plan)
    for index in range(230):
        store.append(work, "s1", "coder", "event", "tool", f"执行事件 {index}")
    store.append(work, "s1", "manager", "run", "error", "执行失败")
    store.maybe_compress(work, "s1", ctx=100)
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(root))
    saved = graph._latest_saved_plan(work, "s1") if kind == "plan_done" else graph._latest_pending_plan(work, "s1")
    assert saved["markdown"] == plan["markdown"]
    assert saved["goal"] == plan["goal"]
    graph.memory.append(work, "s1", "planner", "state", "plan_cancelled", "已取消")
    assert graph._latest_saved_plan(work, "s1") is None
    assert graph._latest_pending_plan(work, "s1") is None


def test_executed_plan_cannot_be_replayed_as_unexecuted(tmp_path):
    store = MemoryStore(tmp_path / "mem")
    work = str(tmp_path / "work")
    store.append(work, "s1", "planner", "state", "plan_done", "计划", {"markdown": "内容"})
    store.append(work, "s1", "planner", "state", "plan_executed", "执行通过")
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), store)
    assert graph._latest_saved_plan(work, "s1") is None


def test_state_handoff_cannot_be_approved_when_required_value_is_missing(tmp_path, monkeypatch):
    """临时生成的任务标识必须实际保存给下一次无状态请求。"""
    from backend.agents.planner import _semantic_plan_issues

    graph = AgentGraph(ModelStore(tmp_path / 'models.json'), MemoryStore(tmp_path / 'mem'))
    monkeypatch.setattr(graph, '_client', lambda *_: object())
    monkeypatch.setattr(graph, '_model_json', lambda *_args, **_kwargs: {
        'ok': True, 'issues': [], 'handoffs': [{
            'state': '任务标识', 'persisted_as': '', 'next_input': '', 'preserved': True,
            'evidence': '首轮临时生成任务标识，但返回与保存时只留下结果。',
        }],
    })
    issues = _semantic_plan_issues(graph, {'text': '后续请求继续原任务'}, '原始合同', '无状态接口方案')
    assert len(issues) == 1 and '任务标识' in issues[0]


def test_semantic_review_receives_actual_scope_resolution(tmp_path, monkeypatch):
    from backend.agents.planner import _semantic_plan_issues

    graph = AgentGraph(ModelStore(tmp_path / 'models.json'), MemoryStore(tmp_path / 'mem'))
    monkeypatch.setattr(graph, '_client', lambda *_: object())
    messages = []
    def review(_state, _agent, _client, value, **_kwargs):
        messages.extend(value)
        return {'ok': True, 'issues': [], 'handoffs': []}
    monkeypatch.setattr(graph, '_model_json', review)
    work = tmp_path / 'work'
    _semantic_plan_issues(graph, {'text': '项目输出到 portal。创建入口', 'workdir': str(work)}, '原始合同', '计划')
    content = str(messages[-1]['content'])
    assert str(work / 'portal') in content
    assert 'uv run --directory portal python -m pytest' in content
    assert '显式交付前缀不会再次拼接' in content


def test_handoff_accepts_different_names_for_same_persisted_value(tmp_path, monkeypatch):
    from backend.agents.planner import _semantic_plan_issues

    graph = AgentGraph(ModelStore(tmp_path / 'models.json'), MemoryStore(tmp_path / 'mem'))
    monkeypatch.setattr(graph, '_client', lambda *_: object())
    monkeypatch.setattr(graph, '_model_json', lambda *_args, **_kwargs: {
        'ok': True, 'issues': [], 'handoffs': [{
            'state': '任务标识', 'persisted_as': '浏览器记录中的 job_id',
            'next_input': '下一次请求携带 job_id', 'preserved': True,
            'evidence': '首轮返回 job_id，客户端保存后随下一轮请求发送。',
        }],
    })
    assert _semantic_plan_issues(graph, {'text': '继续原任务'}, '原始合同', '状态交接计划') == []


def test_second_model_review_removes_invented_issues_but_keeps_real_loss(tmp_path, monkeypatch):
    from api.schema import ModelConfig
    from backend.agents.planner import _semantic_plan_issues
    from llm.client import LlmClient

    models = ModelStore(tmp_path / 'models.json')
    for name in ('author', 'reviewer'):
        models.upsert(ModelConfig(id=name, name='测试模型', model=name, base_url='https://example.test', api_key='TEST_ONLY_KEY'))
    graph = AgentGraph(models, MemoryStore(tmp_path / 'mem'))
    monkeypatch.setattr(graph, '_client', lambda *_: LlmClient(models.get('reviewer')))
    calls = []
    def review(_state, _agent, client, _messages, **_kwargs):
        calls.append(client.cfg.id)
        if client.cfg.id == 'reviewer':
            return {'ok': False, 'issues': ['读取只读输入等于写入', '首轮任务标识未保存给第二轮'], 'handoffs': []}
        return {'issues': ['首轮任务标识未保存给第二轮'], 'discarded': ['读取操作不修改输入']}
    monkeypatch.setattr(graph, '_model_json', review)
    issues = _semantic_plan_issues(graph, {'text': '保留任务标识', 'workdir': str(tmp_path / 'work'), 'session_id': 'review'}, '原始合同', '读取输入但未保存标识')
    assert issues == ['首轮任务标识未保存给第二轮']
    assert calls == ['reviewer', 'author']
