from __future__ import annotations

import json

from backend.agents.graph import AgentGraph
from backend.agents.verifier import verifier
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


def test_verifier_keeps_coding_failure_as_a_failure(tmp_path) -> None:
    """没有可运行测试时，也不能把未完成的 Coding 阶段改判为成功。"""

    workdir = tmp_path / "project"
    workdir.mkdir()
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    result = verifier(
        graph,
        {
            "session_id": "s1",
            "workdir": str(workdir),
            "coding_ok": False,
            "tests": [],
            "tests_ok": True,
        }
    )

    assert result["tests_ok"] is False


def test_verifier_uses_only_existing_node_scripts(tmp_path) -> None:
    """Node 验证命令来自 package.json，不猜测测试框架参数。"""

    fs = FsTool(str(tmp_path / "web"))
    fs.write(
        "package.json",
        json.dumps(
            {
                "scripts": {
                    "dev": "vite",
                    "test": "vitest run",
                    "typecheck": "tsc --noEmit",
                    "build": "vite build",
                }
            }
        ),
    )
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert commands == ["npm run test", "npm run typecheck", "npm run build"]
