from __future__ import annotations

import json
import os
import queue
import shlex
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from api.schema import ModelConfig
from backend.agents.coder import coder
from backend.agents.doc import doc
from backend.agents.graph import AgentGraph, RunBudgetError
from backend.agents.loop import LoopGuard
from backend.agents.planner import _normalize_chat_contract, _plan_quality_issues, _schema_fields
from backend.agents.verifier import _reject_empty_tests, verifier
from backend.memory import MemoryStore
from backend.runs import RunConflictError, RunRegistry
from backend.tools import FsTool, GitTool, ShellTool
from backend.tools.shell import prepare_command
from llm import LlmClient, LlmError, ModelStore
from llm.client import LlmOutputLimitError, LlmUsage


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


def test_structured_output_limit_recovers_once_with_accounted_usage(tmp_path):
    graph = make_graph(tmp_path)

    class LimitedClient(LlmClient):
        def __init__(self):
            self.cfg = SimpleNamespace(max_output_tokens=16_384, efforts=["low", "minimal"])
            self.default_effort = "low"
            self.last_usage = LlmUsage()
            self.calls = 0

        def chat_json(self, _messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs["max_tokens"] == 12_000
                self.last_usage = LlmUsage(prompt=100, completion=12_000, reasoning=11_900)
                raise LlmOutputLimitError("输出触顶")
            assert self.default_effort == "minimal"
            assert kwargs["max_tokens"] == 16_384
            self.last_usage = LlmUsage(prompt=100, completion=50)
            return {"status": "plan", "markdown": "完整计划"}

    state = {"session_id": "s1", "workdir": str(tmp_path / "work")}
    client = LimitedClient()
    result = graph._model_json(state, "planner", client, [])
    assert result["status"] == "plan"
    assert client.calls == 2
    assert state["tokens"] == 12_250


def test_string_done_cannot_claim_completion(tmp_path):
    graph = make_graph(tmp_path)
    replies = iter([
        {"actions": [], "done": "false"},
        {"actions": [{"tool": "write_file", "path": "done.py", "content": "ready = True\n"}], "done": True},
    ])
    client = SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    graph._client = lambda *_args: client
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "创建文件"})
    assert result["coding_ok"]
    assert (tmp_path / "work/done.py").exists()


def test_native_text_is_not_completion_and_identity_cannot_be_overridden():
    assert LlmClient._tool_result([], "接下来我会创建文件")["done"] is False
    assert LlmClient._tool_result([{"tool": "finish", "summary": "已完成"}], "")["done"] is True
    with pytest.raises(LlmError, match="不得覆盖"):
        LlmClient._native_action("read_file", {"tool": "write_file", "path": "x"})
    with pytest.raises(LlmError):
        LlmClient._native_action("read_file", [])


def test_empty_native_calls_fall_back_to_structured_actions(tmp_path):
    graph = make_graph(tmp_path)

    class EmptyNativeClient(LlmClient):
        def __init__(self):
            self.cfg = SimpleNamespace(id="native", model="generic", capabilities=["tools"])
            self.default_effort = "low"
            self.last_usage = LlmUsage()
            self.native_calls = 0

        def chat_tools(self, *_args, **_kwargs):
            self.native_calls += 1
            return self._tool_result([], "准备开始写入")

        def chat_json(self, *_args, **kwargs):
            assert kwargs["plain_text"] is True
            return {"actions": [{"tool": "write_file", "path": "ready.py", "content": "ready = True\n"}], "done": True}

    client = EmptyNativeClient()
    graph._client = lambda *_args: client
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "创建文件", "model_id": "native"})
    assert client.native_calls == 2
    assert result["coding_ok"]
    assert (tmp_path / "work/ready.py").exists()


def test_nested_private_copy_stays_in_project_and_reports_companion_file(tmp_path):
    graph = make_graph(tmp_path)
    graph.model_store.upsert(ModelConfig(id="generic", name="测试模型", base_url="https://example.test", api_key="假密钥", model="generic"))
    fs = FsTool(str(tmp_path / "work"))
    result = graph._do_action({"tool": "model_copy", "model_id": "generic", "target": "service/.local/model.json"}, fs, ShellTool(str(fs.root)), GitTool(str(fs.root)))
    assert result["ok"]
    assert result["files"] == ["service/.local/model.json", "service/.gitignore"]
    assert not (fs.root / ".gitignore").exists()
    assert (fs.root / "service/.gitignore").read_text() == ".local/\n"


def test_document_node_preserves_verified_nested_readme_without_model_call(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("service/README.md", "# 已验收的用户说明\n")
    graph._client = lambda *_args: pytest.fail("已验收的说明不应再次调用模型覆盖")
    result = doc(graph, {"session_id": "s1", "workdir": str(fs.root), "text": "生成项目和 README.md", "task_type": "code_gen", "changes": ["service/README.md"], "tests_ok": True})
    assert result["repo"]["doc_path"] == str(fs.root / "service/README.md")
    assert not (fs.root / "README.md").exists()
    assert fs.read("service/README.md") == "# 已验收的用户说明\n"


def test_document_model_receives_existing_content_and_nested_target(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("service/pyproject.toml", "[project]\nname='demo'\nversion='1'\n")
    fs.write("service/README.md", "# 用户保留信息\n")

    def generate(_state, _agent, _client, messages):
        assert "用户保留信息" in messages[1]["content"]
        assert "文档目标：service/README.md" in messages[1]["content"]
        return "# 用户保留信息\n\n新增说明。"

    graph._client = lambda *_args: object()
    graph._model_text = generate
    result = doc(graph, {"session_id": "s1", "workdir": str(fs.root), "text": "更新项目说明", "task_type": "doc_gen", "changes": ["service/main.py"]})
    assert "service/README.md" in result["changes"]
    assert not (fs.root / "README.md").exists()


def test_real_verification_commands_are_handed_off_and_independently_repeated(tmp_path):
    graph = make_graph(tmp_path)
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return {"cmd": command, "ok": True, "code": 0, "out": "校验通过", "err": ""}

    graph._shell = lambda *_args: SimpleNamespace(run=run)
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: {"actions": [{"tool": "run_command", "cmd": "uv run python tools/build_bundle.py"}], "done": True})
    state = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "运行构建"})
    assert state["verification_commands"] == ["uv run python tools/build_bundle.py"]
    result = verifier(graph, state)
    assert result["tests_ok"]
    assert calls == ["uv run python tools/build_bundle.py"] * 2


def test_stream_output_limit_retains_paid_usage_before_raising():
    client = LlmClient(ModelConfig(id="generic", name="测试", base_url="https://example.test", api_key="假密钥", model="generic"))
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, text='data: {"choices":[{"delta":{"content":"片段"},"finish_reason":"length"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":6000}}\n\ndata: [DONE]\n\n')
    with pytest.raises(LlmOutputLimitError):
        client._stream_text(response, [{"role": "user", "content": "测试"}])
    assert client.last_usage.total == 6100
