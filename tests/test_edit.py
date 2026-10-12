"""定位错误在写入前修订，文件内容与安全边界保持不变。"""

from types import SimpleNamespace

import pytest

from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from backend.tools import FsTool
from llm.client import LlmUsage
from llm.store import ModelStore


def test_preview_validates_exact_anchors_without_writing(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    original = "def last():\n    return 1\n"
    fs.write("main.py", original)
    fs.read("main.py")
    assert fs.preview_block("main.py", "def last():", "", "def last():\n    return 2\n").endswith("return 2\n")
    with pytest.raises(ValueError):
        fs.preview_block("main.py", "def last():", "不存在的边界", "new")
    with pytest.raises(ValueError):
        fs.preview_replace("main.py", "不存在的旧文本", "new")
    assert (fs.root / "main.py").read_text() == original


def test_invalid_edit_reloads_source_and_does_not_execute_or_claim_done(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("main.py", "def last():\n    return 1\n")
    replies = iter([
        {"actions": [{"tool": "read_file", "path": "main.py"}]},
        {"actions": [{"tool": "replace_block", "path": "main.py", "start_marker": "def last():", "end_marker": "不存在的边界", "content": "错误"}], "done": True},
        {"actions": [{"tool": "replace_block", "path": "main.py", "start_marker": "def last():", "end_marker": "", "content": "def last():\n    return 2\n"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    result = coder(graph, {"workdir": str(fs.root), "session_id": "s1", "text": "更新函数返回值"})
    assert result["coding_ok"] and "return 2" in (fs.root / "main.py").read_text()
    assert any(event.kind == "review" and event.data.get("reason") == "edit_precondition" for event in events)
    assert not any(event.kind == "error" or event.data.get("ok") is False for event in events)


def test_repeated_invalid_edits_stop_without_writing(tmp_path):
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    fs = FsTool(str(tmp_path / "work"))
    original = "ready = True\n"
    fs.write("main.py", original)
    calls = []
    def reply(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"actions": [{"tool": "read_file", "path": "main.py"}]}
        return {"actions": [{"tool": "replace_file", "path": "main.py", "old": "不存在", "new": "错误"}], "done": True}
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    result = coder(graph, {"workdir": str(fs.root), "session_id": "s1", "text": "更新入口"})
    assert not result["coding_ok"] and len(calls) == 4
    assert (fs.root / "main.py").read_text() == original


def test_unvalidated_whole_rewrite_is_revised_without_tool_execution(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    replies = iter([
        {"actions": [{"tool": "write_file", "path": "main.py", "content": "ready = True\n"}]},
        {"actions": [{"tool": "write_file", "path": "main.py", "content": "ready = False\n"}]},
        {"actions": [{"tool": "replace_file", "path": "main.py", "old": "True", "new": "False"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    state = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建模块"})
    assert state["coding_ok"] and (tmp_path / "work/main.py").read_text() == "ready = False\n"
    assert any(event.data.get("reason") == "unchecked_rewrite" for event in events)
    assert sum(event.kind == "tool" and event.data.get("tool") == "write_file" for event in events) == 1
    assert not any(event.kind == "error" or event.data.get("ok") is False for event in events)
