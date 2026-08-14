from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from api.schema import ContextPackage, GuidanceRequest
from backend import main as main_module
from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from backend.runs import RunConflictError, RunNotFoundError, RunRegistry
from llm import ModelStore


def test_run_registry_queues_guidance_and_blocks_parallel_session(tmp_path) -> None:
    registry = RunRegistry()
    events: queue.Queue[dict | None] = queue.Queue()
    active = registry.start("s1", str(tmp_path), events)

    with pytest.raises(RunConflictError):
        registry.start("s1", str(tmp_path), queue.Queue())

    _, first = registry.submit_guidance(active.id, "先不要修改样式")
    _, second = registry.submit_guidance(active.id, "改用蓝色主题")

    assert [item["id"] for item in active.drain_guidance()] == [first.id, second.id]
    assert active.drain_guidance() == []

    registry.finish(active.id)
    with pytest.raises(RunNotFoundError):
        registry.submit_guidance(active.id, "已经太晚")


def test_guidance_endpoint_persists_message_and_queues_it(tmp_path, monkeypatch) -> None:
    registry = RunRegistry()
    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "project")
    active = registry.start("s1", workdir, queue.Queue())
    monkeypatch.setattr(main_module, "run_registry", registry)
    monkeypatch.setattr(main_module, "memory_store", memory)

    response = main_module.submit_guidance(active.id, GuidanceRequest(text="  优先修复接口  "))

    assert response.status == "queued"
    assert active.drain_guidance()[0]["text"] == "优先修复接口"
    records = memory.read_session(workdir, "s1")
    assert records[-1]["tl"] == "guidance"
    assert records[-1]["out"] == "优先修复接口"


def test_graph_applies_guidance_to_context_and_emits_event(tmp_path) -> None:
    pending = [{"id": "g1", "text": "只修改后端", "created_at": "2026-08-14T00:00:00Z"}]
    events = []

    def guidance_source() -> list[dict[str, str]]:
        messages = list(pending)
        pending.clear()
        return messages

    graph = AgentGraph(
        ModelStore(),
        MemoryStore(tmp_path / "mem"),
        emit=events.append,
        guidance_source=guidance_source,
    )
    state = {
        "session_id": "s1",
        "workdir": str(tmp_path / "project"),
        "text": "实现功能",
        "guidance": [],
        "guidance_revision": 0,
        "context": ContextPackage(
            goal="实现功能",
            task_type="code_mod",
            workdir=str(tmp_path / "project"),
            plan_mode=False,
        ),
    }

    result = graph._apply_guidance(state, checkpoint="测试检查点")

    assert result["guidance"] == ["只修改后端"]
    assert result["guidance_revision"] == 1
    assert result["context"].guidance == ["只修改后端"]
    assert "只修改后端" in graph._ctx_text(result)
    assert events[-1].kind == "guidance"
    assert "测试检查点" in events[-1].msg


def test_new_guidance_after_coding_routes_back_from_verifier_and_doc(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "task_type": "code_mod",
        "guidance_revision": 2,
        "coded_guidance_revision": 1,
        "tests": [],
        "tests_ok": True,
        "after_verify": "doc",
    }

    assert graph.route_after_verifier(state) == "coder"
    assert graph.route_after_doc(state) == "coder"

    state["coded_guidance_revision"] = 2
    assert graph.route_after_verifier(state) == "doc"
    assert graph.route_after_doc(state) == "final"


def test_coder_discards_stale_actions_when_guidance_arrives_during_model_call(tmp_path) -> None:
    pending: list[dict[str, str]] = []

    def guidance_source() -> list[dict[str, str]]:
        messages = list(pending)
        pending.clear()
        return messages

    graph = AgentGraph(
        ModelStore(),
        MemoryStore(tmp_path / "mem"),
        guidance_source=guidance_source,
    )

    class GuidedClient:
        def __init__(self) -> None:
            self.calls = 0
            self.last_usage = SimpleNamespace(total=0)

        def chat_json(self, messages, temperature=0.1):
            self.calls += 1
            if self.calls == 1:
                pending.append(
                    {
                        "id": "g1",
                        "text": "不要创建 old.py，改为创建 new.py",
                        "created_at": "2026-08-14T00:00:00Z",
                    }
                )
                return {
                    "actions": [{"tool": "write_file", "path": "old.py", "content": "old = True\n"}],
                    "done": False,
                }
            assert "不要创建 old.py，改为创建 new.py" in messages[-1]["content"]
            return {
                "actions": [{"tool": "write_file", "path": "new.py", "content": "new = True\n"}],
                "done": True,
                "summary": "已按最新指导完成",
            }

    client = GuidedClient()
    graph._client = lambda agent, state: client
    project = tmp_path / "project"
    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "创建 old.py",
            "guidance": [],
            "guidance_revision": 0,
            "context": ContextPackage(
                goal="创建 old.py",
                task_type="code_gen",
                workdir=str(project),
                plan_mode=False,
            ),
            "repo": {"files": [], "snippets": {}, "stack": []},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert client.calls == 2
    assert result["coding_ok"] is True
    assert not (project / "old.py").exists()
    assert (project / "new.py").read_text(encoding="utf-8") == "new = True\n"
