"""只读命令适配的空范围、原始换行和私密边界。"""

import pytest

from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from backend.tools import FsTool, GitTool, ShellTool
from backend.tools.reads import inspection_read
from llm.store import ModelStore


@pytest.mark.parametrize("command,expected", [
    ("head -n1 main.py", "first\r\n"),
    ("tail -n1 main.py", "last\r\n"),
    ("tail -c4 main.py", "st\r\n"),
])
def test_crlf_ranges_match_actual_file_offsets(tmp_path, command, expected):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("main.py", "first\r\nlast\r\n")
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    result = graph._do_action(inspection_read(fs, {}, command), fs, ShellTool(str(fs.root)), GitTool(str(fs.root)))
    assert result["text"].endswith(expected)
    assert result["read"]["total_chars"] == len("first\r\nlast\r\n")


def test_only_line_feed_counts_as_command_line_separator(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("main.py", "one\rtwo\nthree\n")
    action = inspection_read(fs, {}, "head -n1 main.py")
    assert fs.read(action["path"], action["max_chars"], action["start"]) == "one\rtwo\n"


@pytest.mark.parametrize("command", ["head -n0 main.py", "tail -n0 main.py", "head -c0 main.py", "tail -c0 main.py"])
def test_empty_ranges_do_not_read_a_character_or_create_snapshot(tmp_path, command):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("main.py", "不应返回的正文")
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    result = graph._do_action(inspection_read(fs, {}, command), fs, ShellTool(str(fs.root)), GitTool(str(fs.root)))
    assert result["ok"] and result["content_available"] is False
    assert "不应返回的正文" not in result["text"]
    assert not fs.last_reads and "read" not in result
