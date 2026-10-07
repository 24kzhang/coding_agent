from __future__ import annotations

import json
import os
import queue
import shlex
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.coder import coder
from backend.agents.graph import AgentGraph, RunBudgetError
from backend.agents.loop import LoopGuard
from backend.agents.planner import _normalize_chat_contract, _plan_quality_issues, _schema_fields
from backend.agents.verifier import _reject_empty_tests, verifier
from backend.memory import MemoryStore
from backend.runs import RunConflictError, RunRegistry
from backend.tools import FsTool, ShellTool
from backend.tools.shell import prepare_command
from llm import LlmError, ModelStore
from llm.client import LlmUsage


def make_graph(tmp_path):
    return AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))


def test_atomic_edits_preserve_mode_and_failed_append_keeps_original(tmp_path, monkeypatch):
    """失败追加不留下部分内容，精确修改保留脚本执行位。"""
    fs = FsTool(str(tmp_path))
    fs.write("run.sh", "echo old\n")
    os.chmod(tmp_path / "run.sh", 0o755)
    fs.read("run.sh")
    fs.replace("run.sh", "old", "new")
    assert (tmp_path / "run.sh").stat().st_mode & 0o777 == 0o755
    original = (tmp_path / "run.sh").read_text()

    def fail_replace(*_args):
        raise OSError("模拟原子替换失败")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        fs.append("run.sh", "echo tail\n")
    assert (tmp_path / "run.sh").read_text() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_stale_snapshot_edit_is_rejected_without_losing_user_changes(tmp_path):
    fs = FsTool(str(tmp_path))
    fs.write("app.py", "value = 1\n")
    fs.read("app.py")
    (tmp_path / "app.py").write_text("value = 1\nuser_change = True\n")
    with pytest.raises(ValueError, match="外部修改"):
        fs.replace("app.py", "value = 1", "value = 2")
    assert "user_change" in (tmp_path / "app.py").read_text()
    fs.read("app.py")
    fs.replace("app.py", "value = 1", "value = 2")
    assert "user_change" in fs.read("app.py")


def test_fs_index_excludes_secrets_and_external_symlinks(tmp_path):
    root = tmp_path / "work"
    fs = FsTool(str(root))
    private = root / ".local/model.json"
    private.parent.mkdir()
    private.write_text('{"api_key":"假密钥"}')
    external = tmp_path / "outside.py"
    external.write_text("private = True\n")
    (root / "linked.py").symlink_to(external)
    fs.write("app.py", "public = True\n")
    assert fs.list() == ["app.py"]
    assert fs.search("private") == []
    with pytest.raises(ValueError, match="敏感配置"):
        fs.read(".local/model.json")


@pytest.mark.parametrize("old,expected", [("", 1), ("value", 0), ("value", -1)])
def test_invalid_replace_cannot_insert_or_succeed_without_mutation(tmp_path, old, expected):
    fs = FsTool(str(tmp_path))
    fs.write("app.py", "value = 1\n")
    with pytest.raises(ValueError):
        fs.replace("app.py", old, "broken", expected)
    assert fs.read("app.py") == "value = 1\n"


def test_partial_read_cannot_authorize_whole_file_overwrite(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("large.py", "# " + "x" * 30_000 + "\nkeep = True\n")
    graph = make_graph(tmp_path)
    observations = iter([
        {"actions": [{"tool": "read_file", "path": "large.py"}]},
        {"actions": [{"tool": "write_file", "path": "large.py", "content": "lost = True\n"}], "done": True},
        {"actions": [], "done": True, "summary": "保留原文件"},
    ])
    client = SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(observations))
    graph._client = lambda *_args: client
    result = coder(graph, {"session_id": "s1", "workdir": str(fs.root), "text": "审查大文件", "repo": {"files": ["large.py"]}})
    assert "keep = True" in (fs.root / "large.py").read_text()
    assert result["changes"] == []


def test_invalid_actions_are_observable_and_do_not_finish_a_batch(tmp_path):
    graph = make_graph(tmp_path)
    actions = graph._normalize_actions([
        {"tool": "read_file", "path": "x", "start": "invalid"},
        {"tool": "write_file", "path": "x"},
        {"tool": "unknown"},
    ])
    assert len(actions) == 3
    assert all(action.get("_error") for action in actions)
    replies = iter([
        {"actions": [{"tool": "unknown"}], "done": True},
        {"actions": [{"tool": "write_file", "path": "app.py", "content": "ready = True\n"}], "done": True},
    ])
    client = SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    graph._client = lambda *_args: client
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "创建入口"})
    assert result["coding_ok"]
    assert (tmp_path / "work/app.py").exists()


def test_budget_stops_before_call_and_caps_remaining_output(tmp_path):
    graph = make_graph(tmp_path)
    state = {"model_calls": 72, "workdir": str(tmp_path / "work"), "session_id": "s1"}
    called = []
    client = SimpleNamespace(chat_json=lambda *_args: called.append(True))
    with pytest.raises(RunBudgetError):
        graph._model_json(state, "planner", client, [])
    assert called == []
    assert graph._check_model_budget({"output_tokens": 159_900}, 6_000) == 100


def test_failed_parse_records_paid_usage_once(tmp_path):
    graph = make_graph(tmp_path)
    client = SimpleNamespace(last_usage=LlmUsage())

    def fail(*_args, **_kwargs):
        client.last_usage = LlmUsage(prompt=20, completion=30)
        raise LlmError("模拟解析失败")

    client.chat_json = fail
    state = {"workdir": str(tmp_path / "work"), "session_id": "s1"}
    with pytest.raises(LlmError):
        graph._model_json(state, "planner", client, [])
    assert state["tokens"] == 50
    assert state["model_requests"] == 1


def test_usage_cannot_reduce_budget_or_fake_cache_hits():
    usage = LlmUsage(prompt=100, completion=20, cached=200, cache_miss=-5, reasoning=100)
    assert usage.cached == 100
    assert usage.cache_miss == 0
    assert usage.reasoning == 20
    assert LlmUsage(prompt=-2, completion=-3).total == 0


@pytest.mark.parametrize("cmd", ["uv run rm file", "env rm file", "git --no-pager reset --hard", "git -C . status", "npm --prefix=../outside test", "tool --output=/tmp/outside"])
def test_command_flags_and_wrappers_cannot_bypass_workspace_policy(tmp_path, cmd):
    result = prepare_command(tmp_path, cmd, resolve_executable=False)
    assert isinstance(result, dict)
    assert result["code"] == 126


def test_command_log_is_bounded_and_keeps_both_error_and_summary(tmp_path):
    script = tmp_path / "output.py"
    script.write_text("print('根因')\nprint('x' * 1000000)\nprint('最后结论')\n")
    result = ShellTool(str(tmp_path)).run(f"{shlex.quote(sys.executable)} output.py")
    assert result["ok"]
    assert result["output_truncated"]
    assert result["output_bytes"] > 1_000_000
    assert result["out"].startswith("根因")
    assert result["out"].rstrip().endswith("最后结论")
    assert len(result["out"]) < 13_000


@pytest.mark.skipif(os.name == "nt", reason="此场景检验 POSIX 进程组")
def test_timeout_terminates_spawned_server_child(tmp_path):
    child = tmp_path / "child.py"
    child.write_text("import socket,time\ns=socket.socket()\ns.bind(('127.0.0.1',0))\ns.listen()\nprint(s.getsockname()[1], flush=True)\ntime.sleep(30)\n")
    parent = tmp_path / "parent.py"
    parent.write_text("import subprocess,sys,time\nsubprocess.Popen([sys.executable,'child.py'])\ntime.sleep(30)\n")
    result = ShellTool(str(tmp_path)).run(f"{shlex.quote(sys.executable)} parent.py", timeout=1)
    assert result["code"] == 124
    assert result["timed_out"]
    port = int(result["out"].strip())
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_verification_covers_all_projects_and_quotes_paths(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    for index in range(4):
        fs.write(f"web {index}/package.json", json.dumps({"scripts": {"test": "vitest run", "typecheck": "tsc", "build": "vite build"}}))
    fs.write("service/pyproject.toml", "[project]\nname='demo'\nversion='1'\n")
    fs.write("service/src/tests/test_api.py", "def test_ready(): assert True\n")
    commands = make_graph(tmp_path)._verification_commands(fs, fs.list())
    assert len(commands) > 8
    assert "npm --prefix 'web 3' run build" in commands
    assert "uv run --project service pytest -q service/src/tests" in commands


def test_zero_tests_are_not_a_pass_and_empty_code_has_no_evidence(tmp_path):
    assert not _reject_empty_tests({"cmd": "npm run test", "ok": True, "out": "0 passing"})["ok"]
    assert _reject_empty_tests({"cmd": "npm run build", "ok": True, "out": "0 tests"})["ok"]
    graph = make_graph(tmp_path)
    result = verifier(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "task_type": "code_gen", "coding_ok": True})
    assert not result["tests_ok"]


def test_alternating_no_progress_cycles_stop_and_progress_resets():
    guard = LoopGuard()
    for signature in ["a", "b", "a", "b", "a"]:
        assert not guard.record(signature, progress=False)
    assert guard.record("b", progress=False)
    assert not guard.record("write", progress=True)
    assert not guard.record("a", progress=False)


def test_overlapping_workspaces_are_serialized_but_siblings_can_run(tmp_path):
    registry = RunRegistry()
    first = registry.start("s1", str(tmp_path / "work"), queue.Queue())
    with pytest.raises(RunConflictError, match="重叠运行"):
        registry.start("s2", str(tmp_path / "work/src"), queue.Queue())
    sibling = registry.start("s3", str(tmp_path / "other"), queue.Queue())
    registry.finish(first.id)
    registry.finish(sibling.id)
    registry.start("s2", str(tmp_path / "work/src"), queue.Queue())


def test_schema_and_plan_contracts_work_for_unrelated_domains(tmp_path):
    """商品领域、任意引用目标、非 Python 测试框架都不依赖健身样例常量。"""
    schema = {"type": "array", "items": {"$ref": "#/$defs/product"}, "$defs": {"product": {"type": "object", "properties": {"sku": {"type": "string"}, "price": {"type": "number"}}}}}
    repo = {"snippets": {"data/catalog.schema.json": json.dumps(schema)}}
    assert _schema_fields(repo) == ["sku", "price"]
    graph = make_graph(tmp_path)
    issues = _plan_quality_issues(graph, {"text": "基于商品数据集开发网站，展示所有详细信息", "repo": repo}, "GET /api/products 返回商品列表，GET /api/products/{sku} 返回完整对象。逐字段展示 `sku`、`price`。使用 Vitest 自动化测试。")
    assert issues == []
    normalized = _normalize_chat_contract("前端初始化 system", "第一次提问传商品信息，之后传历史对话")
    assert "exercise" not in normalized
    assert "业务对象标识" in normalized


def test_plan_accepts_generic_chat_fields_and_unmarked_design_image(tmp_path):
    graph = make_graph(tmp_path)
    markdown = '客户端保存但不渲染 system，服务端根据商品构造和验证 system。响应返回完整 messages（system/user/assistant）。请求体为 {"sku":"abc","prompt":"问题","messages":[]}。商品.png 采用双栏布局和绿色配色。'
    issues = _plan_quality_issues(graph, {"text": "第一次提问传商品信息，之后传历史对话", "attachments": [{"name": "商品.png"}]}, markdown)
    assert issues == []


def test_private_copy_refuses_source_targets_and_ignore_symlinks(tmp_path):
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="私密目录"):
        graph.model_store.copy_private("unused", str(work), "src/settings.json")
    outside = tmp_path / "ignore"
    outside.write_text("用户内容\n")
    (work / ".gitignore").symlink_to(outside)
    with pytest.raises(ValueError, match="符号链接"):
        graph.model_store.copy_private("unused", str(work))
    assert outside.read_text() == "用户内容\n"
