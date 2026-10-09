"""包装命令不能探查私密正文，复合语法在执行前诊断。"""

from types import SimpleNamespace

from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from backend.tools import FsTool, GitTool, ShellTool
from backend.tools.reads import inspection_read
from backend.tools.shell import prepare_command
from llm.client import LlmUsage
from llm.store import ModelStore


def test_private_json_inspection_returns_metadata_and_shell_rejects_bypass(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    path = fs.root / "portal/.local/model.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"api_key":"TEST_ONLY_PRIVATE"}')
    state = {"text": "项目输出到 portal"}
    command = "uv run --project portal python -m json.tool portal/.local/model.json"
    prepared = prepare_command(fs.root, command, resolve_executable=False)
    assert isinstance(prepared, dict) and not prepared["ok"]
    for wrapped in ["uv run cat portal/.local/model.json", "env cat portal/.local/model.json"]:
        assert not prepare_command(fs.root, wrapped, resolve_executable=False)["ok"]
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    result = graph._do_action(inspection_read(fs, state, command), fs, ShellTool(str(fs.root)), GitTool(str(fs.root)), state)
    assert result["ok"] and result["content_available"] is False
    assert "TEST_ONLY_PRIVATE" not in str(result) and not fs.last_reads
    assert FsTool.sensitive_path(fs.root, fs.root / ".LOCAL/unknown.json")


def test_composite_command_is_revised_before_shell_and_cannot_finish_early(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    replies = iter([
        {"actions": [{"tool": "run_command", "cmd": "echo x | sort"}], "done": True},
        {"actions": [{"tool": "run_command", "cmd": "node --version"}, {"tool": "write_file", "path": "main.py", "content": "ready = True"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    calls = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: calls.append(command) or {"ok": True, "out": "版本", "cmd": command})
    result = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建项目"})
    assert result["coding_ok"] and calls == ["node --version"]
    assert any(event.kind == "review" and "尚未执行" in event.msg for event in events)
    assert not any(event.kind == "error" or event.data.get("ok") is False for event in events)


def test_duplicate_version_query_reuses_real_result_without_subprocess_or_failure(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    replies = iter([
        {"actions": [{"tool": "run_command", "cmd": "uv --version"}]},
        {"actions": [{"tool": "run_command", "cmd": "uv --version"}, {"tool": "write_file", "path": "main.py", "content": "ready = True"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    calls = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: calls.append(command) or {"ok": True, "out": "工具版本", "cmd": command})
    result = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建项目"})
    assert result["coding_ok"] and calls == ["uv --version"]
    assert any(event.data.get("cached") for event in events)
    assert not any(event.data.get("ok") is False for event in events)


def test_environment_change_invalidates_version_cache_even_in_same_batch(tmp_path):
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    replies = iter([
        {"actions": [{"tool": "run_command", "cmd": "python --version"}]},
        {"actions": [{"tool": "run_command", "cmd": "uv sync"}, {"tool": "run_command", "cmd": "python --version"}, {"tool": "write_file", "path": "main.py", "content": "ready = True"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    calls = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: calls.append(command) or {"ok": True, "out": "版本", "cmd": command})
    result = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建项目"})
    assert result["coding_ok"] and calls == ["python --version", "uv sync", "python --version"]
