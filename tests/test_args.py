"""模型参数在工具执行前修订，真实失败与协议诊断保持区分。"""

from types import SimpleNamespace

from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.client import LlmUsage
from llm.store import ModelStore


def test_invalid_argument_batch_is_not_partially_executed_or_hidden_by_done(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    replies = iter([
        {"actions": [{"tool": "write_file", "path": "unexpected.py", "content": "禁止执行"}, {"tool": "read_file", "command": "错误字段"}], "done": True},
        {"actions": [{"tool": "write_file", "path": "main.py", "content": "ready = True"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    work = tmp_path / "work"
    result = coder(graph, {"workdir": str(work), "session_id": "s1", "text": "创建入口"})
    assert result["coding_ok"] and (work / "main.py").exists()
    assert not (work / "unexpected.py").exists()
    assert any(event.kind == "model" and event.data.get("reason") == "invalid_action_protocol" for event in events)
    assert not any(event.kind == "error" or event.data.get("ok") is False for event in events)


def test_repeated_invalid_arguments_stop_with_bounded_cost(tmp_path):
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    calls = []
    def reply(*_args, **_kwargs):
        calls.append(1)
        return {"actions": [{"tool": "write_file", "content": "缺少路径"}], "done": True}
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    result = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建入口"})
    assert not result["coding_ok"] and len(calls) == 3
