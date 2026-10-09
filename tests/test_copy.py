"""受控配置重复复制是幂等操作，变更仍必须安全刷新。"""

from types import SimpleNamespace

from api.schema import ModelConfig
from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from backend.tools import FsTool, GitTool, ShellTool
from llm.client import LlmUsage
from llm.store import ModelStore


def setup(tmp_path, emit=None):
    store = ModelStore(tmp_path / "models.json")
    store.upsert(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="TEST_ONLY_KEY", model="generic"))
    return AgentGraph(store, MemoryStore(tmp_path / "mem"), emit=emit)


def test_copy_cache_tracks_configuration_and_ignore_rule_changes(tmp_path):
    graph = setup(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    shell, git = ShellTool(str(fs.root)), GitTool(str(fs.root))
    action = {"tool": "model_copy", "model_id": "test"}
    first = graph._do_action(action, fs, shell, git)
    second = graph._do_action(action, fs, shell, git)
    assert first["ok"] and second["cached"] and "file" not in second
    absolute = graph._do_action({**action, "target": str(fs.root / ".local/model.json")}, fs, shell, git)
    assert absolute["cached"]
    graph.model_store.upsert(graph.model_store.get("test").model_copy(update={"default_effort": "none"}))
    refreshed = graph._do_action(action, fs, shell, git)
    assert refreshed["ok"] and not refreshed.get("cached")
    (fs.root / ".gitignore").write_text("")
    repaired = graph._do_action(action, fs, shell, git)
    assert repaired["ok"] and not repaired.get("cached")
    assert "TEST_ONLY_KEY" not in str([first, second, refreshed, repaired])


def test_coder_does_not_reject_idempotent_copy_as_failed_tool(tmp_path):
    events = []
    graph = setup(tmp_path, emit=events.append)
    replies = iter([
        {"actions": [{"tool": "model_copy", "model_id": "test"}]},
        {"actions": [{"tool": "model_copy", "model_id": "test"}, {"tool": "write_file", "path": "main.py", "content": "ready = True"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    result = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "复用配置实现入口"})
    assert result["coding_ok"]
    assert any(event.kind == "tool" and event.data.get("cached") for event in events)
    assert not any(event.data.get("ok") is False for event in events)
