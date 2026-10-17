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
            'required': ['任务标识', '上次结果'], 'available_next': ['上次结果'],
            'evidence': '首轮临时生成任务标识，但返回与保存时只留下结果。',
        }],
    })
    issues = _semantic_plan_issues(graph, {'text': '后续请求继续原任务'}, '原始合同', '无状态接口方案')
    assert len(issues) == 1 and '任务标识' in issues[0]
