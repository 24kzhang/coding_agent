from __future__ import annotations

from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from backend.tools import FsTool
from llm import ModelStore


def test_static_web_check_detects_missing_dom_wiring(tmp_path) -> None:
    workdir = tmp_path / "web"
    fs = FsTool(str(workdir))
    fs.write("index.html", "<button onclick=\"sendMsg()\"></button><div id=\"chat\"></div>")
    fs.write("app.js", "document.getElementById('missing')")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    result = graph._static_web_check(fs, fs.list())

    assert result is not None
    assert result["ok"] is False
    assert result["missing_ids"] == ["missing"]
    assert result["missing_funcs"] == ["sendMsg"]
