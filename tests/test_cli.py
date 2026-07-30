from __future__ import annotations

import sys
from types import SimpleNamespace

from backend import cli
from backend.memory import MemoryStore


def test_cli_reuses_session_for_plan_flow(tmp_path, monkeypatch, capsys) -> None:
    """CLI 多次调用同一 session id 时必须保留会话，而不是每步新建。"""

    workdir = tmp_path / "project"
    memory = MemoryStore(tmp_path / "memory")
    calls: list[dict] = []

    class FakeGraph:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda: {"ok": True, "summary": "完成"})

    monkeypatch.setattr(cli, "MemoryStore", lambda: memory)
    monkeypatch.setattr(cli, "AgentGraph", FakeGraph)
    monkeypatch.setattr(cli, "ModelStore", object)
    monkeypatch.setattr(cli, "McpClient", object)

    common = [
        "coding-agent",
        "--workdir",
        str(workdir),
        "--session-id",
        "cli-plan-1",
    ]
    monkeypatch.setattr(sys, "argv", common + ["--task", "生成计划", "--plan-mode"])
    cli.main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", common + ["--task", "执行计划", "--execute-plan"])
    cli.main()
    output = capsys.readouterr().out

    assert [item["session_id"] for item in calls] == ["cli-plan-1", "cli-plan-1"]
    assert calls[0]["plan_mode"] is True
    assert calls[1]["execute_plan"] is True
    assert calls[1]["resuming"] is False
    assert '"session_id": "cli-plan-1"' in output

    records = memory.read_session(str(workdir), "cli-plan-1")
    assert sum(item["tl"] == "cli" and item["k"] == "start" for item in records) == 1
    assert [
        item["out"]
        for item in records
        if item["ag"] == "user" and item["tl"] == "input" and item["k"] == "message"
    ] == ["生成计划", "执行计划"]
    assert memory.interrupted(str(workdir), "cli-plan-1") is False


def test_cli_reads_long_task_from_utf8_file(tmp_path, monkeypatch, capsys) -> None:
    """长任务可从文件传入，避免凭据和大段提示暴露在进程参数中。"""

    workdir = tmp_path / "project"
    task_file = tmp_path / "task.txt"
    task_file.write_text("创建中文健身网站", encoding="utf-8")
    captured: dict = {}
    selected: list[str] = []

    class FakeCapabilities:
        def load_skills(self, skill_ids):
            selected.extend(skill_ids)
            return [SimpleNamespace(id=item) for item in skill_ids]

        def select_mcps(self, _mcp_ids):
            return []

    class FakeGraph:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(model_dump=lambda: {"ok": True, "summary": "完成"})

    monkeypatch.setattr(cli, "MemoryStore", lambda: MemoryStore(tmp_path / "memory"))
    monkeypatch.setattr(cli, "AgentGraph", FakeGraph)
    monkeypatch.setattr(cli, "ModelStore", object)
    monkeypatch.setattr(cli, "McpClient", object)
    monkeypatch.setattr(cli, "CapabilityStore", FakeCapabilities)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coding-agent",
            "--workdir",
            str(workdir),
            "--task-file",
            str(task_file),
            "--skill",
            "web-design-guidelines",
        ],
    )

    cli.main()
    capsys.readouterr()

    assert captured["text"] == "创建中文健身网站"
    assert selected == ["web-design-guidelines"]
    assert [item.id for item in captured["skills"]] == ["web-design-guidelines"]
