from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from api.schema import ModelConfig
from backend.agents.coder import _guidance_forbids_reads, _is_verification_command, coder
from backend.agents.doc import _document_path, doc
from backend.agents.graph import AgentGraph, RunBudgetError
from backend.agents.loop import LoopGuard
from backend.agents.planner import _plan_quality_issues, _schema_fields, _semantic_plan_issues
from backend.agents.precheck import import_issues, inline_issues, manifest_issues, server_issues
from backend.agents.verifier import _reject_empty_tests, verifier
from backend.memory import MemoryStore
from backend.runs import RunConflictError, RunRegistry
from backend.tools import FsTool, GitTool, ShellTool
from backend.tools.reads import inspection_read
from backend.tools.scope import command_cwd, delivery_root
from backend.tools.shell import prepare_command
from llm import LlmBalanceError, LlmClient, LlmError, ModelStore
from llm.client import LlmOutputLimitError, LlmUsage


def make_graph(tmp_path):
    return AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))


def test_failed_tool_aborts_remaining_actions_before_replanning(tmp_path):
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    replies = iter([
        {"actions": [{"tool": "read_file", "path": "../../outside.py"}, {"tool": "write_file", "path": "unexpected.py", "content": "bad = True"}]},
        {"actions": [{"tool": "write_file", "path": "done.py", "content": "ready = True"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    result = coder(graph, {"session_id": "s1", "workdir": str(work), "text": "创建通用入口"})
    assert result["coding_ok"] is True
    assert not (work / "unexpected.py").exists()
    assert (work / "done.py").exists()


def test_missing_delivery_file_is_not_fabricated_as_a_read_snapshot(tmp_path):
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    calls = []

    def reply(messages, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"actions": [{"tool": "read_file", "path": "README.md"}]}
        prompt = str(messages[-1]["content"])
        assert "文件尚不存在" in prompt
        assert "[README.md 当前内容" not in prompt
        return {"actions": [{"tool": "write_file", "path": "README.md", "content": "# 使用说明"}], "done": True}

    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    assert coder(graph, {"session_id": "s1", "workdir": str(work), "text": "创建使用说明"})["coding_ok"]
    assert (work / "README.md").read_text() == "# 使用说明"
    fs = FsTool(str(work))
    result = graph._do_action({"tool": "read_file", "path": ".local/unknown.json"}, fs, ShellTool(str(work)), GitTool(str(work)))
    assert result["ok"] and result["content_available"] is False
    assert result["exists"] is False


def test_same_batch_can_append_and_precisely_edit_its_new_file(tmp_path):
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    events = []
    graph._emit = lambda *_args, **kwargs: events.append(kwargs.get("data", {}))
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: {
        "actions": [
            {"tool": "write_file", "path": "style.css", "content": ".card { color: red; }\n"},
            {"tool": "append_file", "path": "portal/style.css", "content": ".panel { display: grid; }\n"},
            {"tool": "replace_file", "path": "style.css", "old": "color: red", "new": "color: green"},
        ], "done": True,
    })
    result = coder(graph, {"session_id": "s1", "workdir": str(work), "text": "项目输出到 portal"})
    assert result["coding_ok"] is True
    assert (work / "portal/style.css").read_text() == ".card { color: green; }\n.panel { display: grid; }\n"
    assert not any(event.get("ok") is False for event in events)


def test_private_metadata_never_becomes_content_or_writable_snapshot(tmp_path):
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    private = work / ".local/model.json"
    private.parent.mkdir(parents=True)
    secret = "测试私密值不可公开"
    private.write_text(json.dumps({"api_key": secret, "model": "私密字段值"}))
    fs = FsTool(str(work))
    result = graph._do_action({"tool": "read_file", "path": ".local/model.json"}, fs, ShellTool(str(work)), GitTool(str(work)))
    assert result["ok"] and result["exists"] and result["content_available"] is False
    assert secret not in str(result) and "私密字段值" not in str(result)
    assert fs.last_reads == {}
    with pytest.raises(ValueError, match="敏感"):
        fs.read(".local/model.json")
    assert not graph._do_action({"tool": "write_file", "path": ".local/model.json", "content": "覆盖"}, fs, ShellTool(str(work)), GitTool(str(work)))["ok"]
    calls = []
    def reply(messages, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"actions": [{"tool": "read_file", "path": ".local/model.json"}]}
        prompt = str(messages[-1]["content"])
        assert secret not in prompt and "[.local/model.json 当前内容" not in prompt
        assert "已读取且当前可直接修改的文件：无" in prompt
        return {"actions": [{"tool": "write_file", "path": "main.py", "content": "ready = True\n"}], "done": True}
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    assert coder(graph, {"session_id": "s1", "workdir": str(work), "text": "生成入口"})["coding_ok"]


def test_local_import_contract_handles_exports_modules_and_dynamic_names(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    state = {"text": "项目输出到 portal"}
    fs.write("portal/pkg/__init__.py", "")
    fs.write("portal/pkg/data.py", "CODE = True\nclass Item: pass\n")
    fs.write("portal/pkg/dynamic.py", "def __getattr__(name): return 1\n")
    fs.write("portal/pkg/client.py", "from .data import CODE, Item\nfrom pkg import data\nfrom math import pi\nfrom pkg.dynamic import MAGIC\nfrom pkg.data import MISSING\n")
    issues = import_issues(fs, state, fs.list())
    assert len(issues) == 1 and "MISSING" in issues[0]
    fs.write("portal/pkg/data.py", "CODE = True\nMISSING = True\nclass Item: pass\n")
    assert import_issues(fs, state, fs.list()) == []
    fs.write("portal/src/extra/data.py", "VALUE = 1\n")
    fs.write("portal/tests/test_src.py", "from extra.data import VALUE\n")
    assert import_issues(fs, state, fs.list()) == []


def test_uv_build_metadata_files_exist_before_environment_creation(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    state = {"text": "项目输出到 portal"}
    content = '[project]\nname="portal"\nversion="1"\nreadme="README.md"\n[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n[tool.hatch.build.targets.wheel]\npackages=["pkg"]\n'
    fs.write("portal/pyproject.toml", content)
    issues = manifest_issues(fs, state, "uv run --directory portal python tools/inspect.py")
    assert len(issues) == 2 and "README.md" in issues[0]
    assert manifest_issues(fs, state, "npm --prefix portal install") == []
    assert manifest_issues(fs, state, "uv sync --no-install-project") == []
    fs.write("portal/README.md", "# 项目说明\n")
    fs.write("portal/pkg/__init__.py", "")
    assert manifest_issues(fs, state, "uv sync") == []
    fs.write("portal/pyproject.toml", '[project]\nname="portal"\nversion="1"\nreadme="missing.md"\n')
    assert manifest_issues(fs, state, "uv sync") == []


def test_metadata_repair_allows_uv_sync_without_stale_pending_issue(tmp_path):
    graph = make_graph(tmp_path)
    replies = iter([
        {"actions": [{"tool": "write_file", "path": "pyproject.toml", "content": '[project]\nname="demo"\nversion="1"\nreadme="README.md"\n[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n'},
                     {"tool": "run_command", "cmd": "uv sync"}], "done": True},
        {"actions": [{"tool": "write_file", "path": "README.md", "content": "# 项目说明\n"},
                     {"tool": "run_command", "cmd": "uv sync"}], "done": True},
    ])
    client = LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic", capabilities=[]))
    client.chat_json = lambda *_args, **_kwargs: next(replies)
    graph._client = lambda *_args: client
    commands = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: commands.append(command) or {"cmd": command, "ok": True, "code": 0, "out": "环境就绪", "err": ""})
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "准备项目环境", "task_type": "code_gen"})
    assert result["coding_ok"] and commands == ["uv sync"]


def test_inline_request_becomes_script_workflow_before_shell(tmp_path):
    graph = make_graph(tmp_path)
    replies = iter([
        {"actions": [{"tool": "run_command", "cmd": "python3 -c 'print(1);print(2)'"}], "done": True},
        {"actions": [{"tool": "write_file", "path": "probe.py", "content": "print(1)\nprint(2)\n"},
                     {"tool": "run_command", "cmd": "python3 probe.py"}], "done": True},
    ])
    client = LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic", capabilities=[]))
    client.chat_json = lambda *_args, **_kwargs: next(replies)
    graph._client = lambda *_args: client
    commands = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: commands.append(command) or {"cmd": command, "ok": True, "code": 0, "out": "1\n2", "err": ""})
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "执行检查脚本", "task_type": "code_gen"})
    assert result["coding_ok"] and commands == ["python3 probe.py"]
    assert inline_issues("uv run python -c 'print(1)'")
    assert inline_issues("python3.13 -c 'print(1)'")
    assert inline_issues("python3 probe.py") == []


def test_server_command_is_deferred_until_bounded_smoke_script(tmp_path):
    graph = make_graph(tmp_path)
    replies = iter([
        {"actions": [{"tool": "run_command", "cmd": "uv run --directory portal python -m uvicorn core.main:app --port 9012"}], "done": True},
        {"actions": [{"tool": "write_file", "path": "probe.py", "content": "print('启动与关闭已由脚本管理')\n"},
                     {"tool": "run_command", "cmd": "python probe.py"}], "done": True},
    ])
    client = LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic", capabilities=[]))
    client.chat_json = lambda *_args, **_kwargs: next(replies)
    graph._client = lambda *_args: client
    commands = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: commands.append(command) or {"cmd": command, "ok": True, "code": 0, "out": "已关闭", "err": ""})
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "真实启动并验收", "task_type": "code_gen"})
    assert result["coding_ok"] and commands == ["python probe.py"]
    assert server_issues("uv run --with uvicorn python probe.py") == []
    assert server_issues("uv run --directory portal uvicorn core.main:app")
    assert server_issues("uvicorn --help") == []


def test_proven_import_conflict_cannot_run_tests_or_finish_before_repair(tmp_path, monkeypatch):
    from backend.agents import coder as module
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    replies = iter([
        {"actions": [{"tool": "write_file", "path": "defs.py", "content": "CODE = True\n"},
                     {"tool": "write_file", "path": "test_main.py", "content": "from defs import MISSING\n"},
                     {"tool": "run_command", "cmd": "python -m pytest"}], "done": True},
        {"actions": [], "done": True},
        {"actions": [{"tool": "replace_file", "path": "defs.py", "old": "CODE = True", "new": "CODE = True\nMISSING = True"},
                     {"tool": "run_command", "cmd": "python -m pytest"}], "done": True},
    ])
    client = LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic", capabilities=[]))
    client.chat_json = lambda *_args, **_kwargs: next(replies)
    graph._client = lambda *_args: client
    monkeypatch.setattr(module, "review_before_verification", lambda *_args: [])
    commands = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: commands.append(command) or {"cmd": command, "ok": True, "code": 0, "out": "1 passed", "err": ""})
    result = coder(graph, {"session_id": "s1", "workdir": str(work), "text": "创建模块并测试", "task_type": "code_gen"})
    assert result["coding_ok"] and commands == ["python -m pytest"]
    assert "MISSING" in (work / "defs.py").read_text()


@pytest.mark.parametrize("repair", [True, False])
def test_preflight_suggestions_allow_repair_or_actual_verification(tmp_path, monkeypatch, repair):
    from backend.agents import coder as module

    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    replies = iter([
        {"actions": [
            {"tool": "write_file", "path": "main.py", "content": "ready = False\n"},
            {"tool": "run_command", "cmd": "python -m pytest"},
        ], "done": True},
        {"actions": [
            *([{"tool": "replace_file", "path": "main.py", "old": "False", "new": "True"}] if repair else []),
            {"tool": "run_command", "cmd": "python -m pytest"},
        ], "done": True},
    ])
    client = LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic", capabilities=[]))
    client.chat_json = lambda *_args, **_kwargs: next(replies)
    graph._client = lambda *_args: client
    reviews = []
    def review(*_args):
        reviews.append(1)
        return ["main.py 的 ready 必须为真"] if len(reviews) == 1 else []
    monkeypatch.setattr(module, "review_before_verification", review)
    commands = []
    graph._shell = lambda *_args: SimpleNamespace(run=lambda command, **_kwargs: commands.append(command) or {"cmd": command, "ok": True, "code": 0, "out": "1 passed", "err": ""})
    result = coder(graph, {"session_id": "s1", "workdir": str(work), "text": "创建入口并测试", "task_type": "code_gen"})
    assert result["coding_ok"]
    assert len(reviews) == (2 if repair else 1)
    assert commands == ["python -m pytest"]
    assert (work / "main.py").read_text() == f"ready = {repair}\n"


def test_distinct_integration_reads_do_not_end_unfinished_coding(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    for name in ("one.py", "two.py", "three.py"):
        fs.write(name, f"name = '{name}'\n")
    replies = iter([
        {"actions": [{"tool": "write_file", "path": "entry.js", "content": "const ready = false;"}]},
        *({"actions": [{"tool": "read_file", "path": name}]} for name in ("one.py", "two.py", "three.py")),
        {"actions": [{"tool": "replace_file", "path": "entry.js", "old": "false", "new": "true"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    result = coder(graph, {"session_id": "s1", "workdir": str(fs.root), "text": "完成入口集成", "repo": {"files": fs.list()}})
    assert result["coding_ok"] is True
    assert fs.read("entry.js") == "const ready = true;"


def test_interface_contract_remains_when_full_snapshot_is_evicted(tmp_path, monkeypatch):
    from backend.agents import coder as module

    monkeypatch.setattr(module, "SNAPSHOT_BUDGET", 300)
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    calls = []

    def reply(messages, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"actions": [{"tool": "write_file", "path": "api.js", "content": "export function submitItem({sku, quantity}) { return sku; }"}]}
        if len(calls) == 2:
            return {"actions": [{"tool": "write_file", "path": "other.js", "content": "const x = 1;\n" + "// 说明\n" * 100}]}
        prompt = str(messages[-1]["content"])
        assert "api.js: submitItem({sku, quantity})" in prompt
        assert "[api.js 当前内容" not in prompt
        return {"actions": [{"tool": "write_file", "path": "caller.js", "content": "export const ready = true;"}], "done": True}

    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    assert coder(graph, {"session_id": "s1", "workdir": str(work), "text": "创建商品调用模块"})["coding_ok"]


@pytest.mark.parametrize("project", ["portal", "service"])
def test_delivery_directory_drives_script_cwd_without_losing_workspace_inputs(tmp_path, project):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    output = fs.root / project
    state = {"text": f"项目输出到 {output}"}
    fs.write(f"{project}/tools/probe.py", "from pathlib import Path\nprint(Path.cwd().name)\nprint((Path.cwd().parent / 'catalog.json').read_text())\n")
    fs.write("catalog.json", "只读输入")
    shell = ShellTool(str(fs.root))
    command = f"{shlex.quote(sys.executable)} tools/probe.py"
    result = graph._do_action({"tool": "run_command", "cmd": command}, fs, shell, GitTool(str(fs.root)), state)
    assert result["ok"] is True
    assert result["result"]["cwd"] == str(output)
    assert result["result"]["out"].splitlines() == [project, "只读输入"]
    assert command_cwd(fs, state, f"uv run --project {project} pytest {project}/tests") == fs.root
    assert command_cwd(fs, state, "uv sync") == output
    assert prepare_command(fs.root, "python3 tools/probe.py", resolve_executable=True, cwd=output)[1][0] == sys.executable
    with pytest.raises(ValueError, match="禁止写入"):
        from backend.tools.scope import require_delivery_path
        require_delivery_path(fs, delivery_root(fs, state), "outside.py")


def test_delivery_path_quotes_and_symlinks_are_checked(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    assert delivery_root(fs, {"text": '交付目录："my portal"'}) == fs.root / "my portal"
    assert delivery_root(fs, {"text": "输出目录不得覆盖现有数据"}) == fs.root
    outside = tmp_path / "outside"
    outside.mkdir()
    (fs.root / "portal").symlink_to(outside)
    with pytest.raises(ValueError, match="工作目录外"):
        delivery_root(fs, {"text": "项目输出到 portal"})


def test_project_relative_file_paths_preserve_delivery_boundary_and_inputs(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("catalog.json", "只读输入")
    state = {"text": "项目输出到 portal"}
    shell, git = ShellTool(str(fs.root)), GitTool(str(fs.root))
    result = graph._do_action({"tool": "write_file", "path": "main.py", "content": "ready = True\n"}, fs, shell, git, state)
    assert result["ok"]
    assert result["file"] == "portal/main.py"
    assert not (fs.root / "main.py").exists()
    assert graph._do_action({"tool": "read_file", "path": "main.py"}, fs, shell, git, state)["exists"]
    assert graph._do_action({"tool": "write_file", "path": "work/portal/main.py", "content": "ready = False\n"}, fs, shell, git, state)["file"] == "portal/main.py"
    assert not (fs.root / "portal/work").exists()
    assert "只读输入" in graph._do_action({"tool": "read_file", "path": "catalog.json"}, fs, shell, git, state)["text"]
    assert not graph._do_action({"tool": "write_file", "path": "../outside.py", "content": "bad"}, fs, shell, git, state)["ok"]


def test_plan_semantic_review_preserves_real_conflicts_and_tool_contracts(tmp_path, monkeypatch):
    """跨步骤合同矛盾进入修订反馈，不通过固定业务规则选择架构。"""

    graph = make_graph(tmp_path)
    monkeypatch.setattr(graph, "_client", lambda *_args: object())
    captured = []

    def review(_state, agent, _client, messages, **_kwargs):
        captured.append(messages)
        assert agent == "verifier"
        return {"ok": False, "issues": ["步骤一清空库存索引，步骤二却依赖该索引结算"]}

    monkeypatch.setattr(graph, "_model_json", review)
    issues = _semantic_plan_issues(graph, {"text": "结算必须保留已有库存索引"}, "原始需求及真实工具合同", "步骤一清空库存索引；步骤二按索引结算")
    assert issues == ["步骤一清空库存索引，步骤二却依赖该索引结算"]
    assert "原始需求及真实工具合同" in str(captured)
    assert "不强制字段名" in captured[0][0]["content"]


def test_plan_semantic_review_balance_failure_propagates(tmp_path, monkeypatch):
    graph = make_graph(tmp_path)
    monkeypatch.setattr(graph, "_client", lambda *_args: object())

    def fail(*_args, **_kwargs):
        raise LlmBalanceError("余额不足")

    monkeypatch.setattr(graph, "_model_json", fail)
    with pytest.raises(LlmBalanceError):
        _semantic_plan_issues(graph, {"text": "任务"}, "合同", "计划")


@pytest.mark.parametrize("explicit", [False, True])
def test_structured_node_fails_over_after_exhausted_recovery_unless_explicit(tmp_path, monkeypatch, explicit):
    graph = make_graph(tmp_path)
    for model in ("primary", "backup"):
        graph.model_store.upsert(ModelConfig(id=model, name="测试模型", model=model, base_url="https://example.test", api_key="TEST_ONLY_KEY"))
    calls = []

    class Client(LlmClient):
        def chat_json(self, _messages, **kwargs):
            calls.append(self.cfg.id)
            if self.cfg.id == "primary":
                self.last_usage = LlmUsage(prompt=10, completion=kwargs["max_tokens"])
                raise LlmOutputLimitError("输出预算耗尽")
            self.last_usage = LlmUsage(prompt=10, completion=50)
            return {"ok": True, "issues": []}

    monkeypatch.setattr("backend.agents.graph.LlmClient", Client)
    state = {"session_id": "s1", "workdir": str(tmp_path / "work"), "model_id": "primary" if explicit else None}
    client = Client(graph.model_store.get("primary"))
    if explicit:
        with pytest.raises(LlmOutputLimitError):
            graph._model_json(state, "verifier", client, [])
        assert calls == ["primary", "primary"]
    else:
        assert graph._model_json(state, "verifier", client, [])["ok"] is True
        assert calls == ["primary", "primary", "backup"]
        assert state["output_tokens"] == 18_050


@pytest.mark.parametrize("balance", [False, True])
def test_structured_node_timeout_fails_over_but_balance_stops(tmp_path, monkeypatch, balance):
    graph = make_graph(tmp_path)
    for name in ("primary", "backup"):
        graph.model_store.upsert(ModelConfig(id=name, name="测试模型", model=name, base_url="https://example.test", api_key="TEST_ONLY_KEY"))
    calls = []

    class Client(LlmClient):
        def chat_json(self, _messages, **_kwargs):
            calls.append(self.cfg.id)
            if self.cfg.id == "primary":
                if balance:
                    raise LlmBalanceError("模型账户余额不足")
                raise LlmError("模型响应超时")
            self.last_usage = LlmUsage(prompt=10, completion=50)
            return {"ok": True}

    monkeypatch.setattr("backend.agents.graph.LlmClient", Client)
    state = {"session_id": "s1", "workdir": str(tmp_path / "work")}
    client = Client(graph.model_store.get("primary"))
    if balance:
        with pytest.raises(LlmBalanceError):
            graph._model_json(state, "verifier", client, [])
        assert calls == ["primary"]
    else:
        assert graph._model_json(state, "verifier", client, [])["ok"] is True
        assert calls == ["primary", "backup"]


def test_structured_node_repairs_json_with_same_explicit_model(tmp_path, monkeypatch):
    graph = make_graph(tmp_path)
    cfg = ModelConfig(id="only", name="测试模型", model="only", base_url="https://example.test", api_key="TEST_ONLY_KEY")
    graph.model_store.upsert(cfg)
    calls = []

    class Client(LlmClient):
        def chat_json(self, messages, **_kwargs):
            calls.append(self.cfg.id)
            self.last_usage = LlmUsage(prompt=10, completion=50)
            if len(calls) == 1:
                raise LlmError("模型没有返回合法 JSON：字符串未转义")
            assert "正确转义" in str(messages)
            return {"ok": True}

    monkeypatch.setattr("backend.agents.graph.LlmClient", Client)
    state = {"session_id": "s1", "workdir": str(tmp_path / "work"), "model_id": "only"}
    assert graph._model_json(state, "planner", Client(cfg), [])["ok"]
    assert calls == ["only", "only"]
    assert state["output_tokens"] == 100


@pytest.mark.parametrize("selected", [False, True])
def test_optional_mcp_unavailability_does_not_become_required_failure(tmp_path, selected):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)

    class Unavailable:
        calls = 0

        def list_tools(self, *_args):
            self.calls += 1
            raise RuntimeError("服务暂不可用")

    client = Unavailable()
    graph.mcp_client = client
    state = {"session_id": "s1", "workdir": str(tmp_path / "work"), "required_mcps": ["addon"] if selected else [],
             "mcp_servers": [{"id": "addon", "name": "可选扩展", "always_on": True, "transport": "streamable_http", "url": "https://example.test/mcp"}]}
    assert graph._ensure_mcp_tools(state) == []
    assert graph._ensure_mcp_tools(state) == []
    assert client.calls == 1
    assert bool(state["mcp_errors"])
    assert events[0].kind == ("error" if selected else "capability")


def test_plan_checks_actual_field_section_after_image_description(tmp_path):
    """商品详情的完整字段合同可以位于图像解读之后，不依赖固定措辞。"""

    graph = make_graph(tmp_path)
    schema = {"type": "object", "properties": {"sku": {}, "price": {}}}
    state = {"text": "商品详情展示所有详细信息", "repo": {"snippets": {"schema.json": json.dumps(schema)}}}
    markdown = "图片左侧是商品封面。\n" + "图像说明\n" * 700
    markdown += "详情左栏展示全部 2 个原始字段：`sku`、`price`。GET /api/products/{sku} 返回完整对象。"
    assert _plan_quality_issues(graph, state, markdown) == []


def test_plan_checks_endpoint_definition_after_earlier_reference(tmp_path):
    graph = make_graph(tmp_path)
    schema = {"type": "object", "properties": {"sku": {}, "price": {}}}
    state = {"text": "商品详情展示所有详细信息", "repo": {"snippets": {"schema.json": json.dumps(schema)}}}
    markdown = "媒体存在性由 /api/products/{sku} 检查。\n### 健康接口\nGET /api/health\n"
    markdown += "### 详情接口\nGET /api/products/{sku} 返回完整对象。左栏展示全部 2 个原始字段：`sku`、`price`。"
    assert _plan_quality_issues(graph, state, markdown) == []


def test_complete_contract_rejects_field_exclusion_despite_complete_claim(tmp_path):
    graph = make_graph(tmp_path)
    schema = {"type": "object", "properties": {"sku": {}, "thumb": {}, "price": {}}}
    state = {"text": "商品详情展示全部详细信息，禁止加载缩略图资源", "repo": {"snippets": {"schema.json": json.dumps(schema)}}}
    base = "左栏逐字段展示全部原始字段：`sku`、`thumb`、`price`。GET /api/products/{sku} 返回完整对象。\n"
    assert _plan_quality_issues(graph, state, base + "`thumb` 仅作文本展示，不加载该资源。") == []
    issues = _plan_quality_issues(graph, state, base + "`thumb` 接口不返回，仅用于校验。全部 2 个字段。")
    assert any("数量" in issue for issue in issues)
    assert any("明确排除" in issue and "thumb" in issue for issue in issues)
    issues = _plan_quality_issues(graph, state, base + "前端不渲染 `thumb`，只在响应保留。")
    assert any("界面" in issue and "thumb" in issue for issue in issues)
    assert _plan_quality_issues(graph, state, base + "前端不渲染 `thumb` 图片，但字段值作为文本展示。") == []


def test_repair_can_read_distinct_files_before_first_new_mutation(tmp_path):
    """上轮产物不能消耗新修复轮的实现前诊断机会。"""
    graph = make_graph(tmp_path)
    work = tmp_path / "work"
    fs = FsTool(str(work))
    for name in ("one.py", "two.py", "three.py", "four.py"):
        fs.write(name, "value = 1\n")
    replies = iter([
        *({"actions": [{"tool": "read_file", "path": name}]} for name in ("one.py", "two.py", "three.py", "four.py")),
        {"actions": [{"tool": "write_file", "path": "fix.py", "content": "ready = True\n"}], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(
        last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies),
    )
    result = coder(graph, {
        "session_id": "repair", "workdir": str(work), "text": "诊断四个相关模块并补充实现",
        "changes": ["one.py"], "retry": 1,
    })
    assert result["coding_ok"]
    assert (work / "fix.py").read_text() == "ready = True\n"


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
    assert "uv run --project service python -m pytest -q service/src/tests" in commands


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


@pytest.mark.parametrize("supports_none", [False, True])
def test_structured_output_limit_recovers_once_with_accounted_usage(tmp_path, supports_none):
    graph = make_graph(tmp_path)

    class LimitedClient(LlmClient):
        def __init__(self):
            self.cfg = SimpleNamespace(max_output_tokens=16_384, efforts=["low", "minimal"] + (["none"] if supports_none else []))
            self.default_effort = "low"
            self.last_usage = LlmUsage()
            self.calls = 0

        def chat_json(self, _messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs["max_tokens"] == 12_000
                self.last_usage = LlmUsage(prompt=100, completion=12_000, reasoning=11_900)
                raise LlmOutputLimitError("输出触顶")
            assert self.default_effort == ("none" if supports_none else "minimal")
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


def test_private_copy_resolves_unique_name_and_rejects_ambiguous_alias(tmp_path):
    store = ModelStore(tmp_path / "models.json")
    store.upsert(ModelConfig(id="service-a", name="服务甲", base_url="https://example.test", api_key="假密钥", model="generic-model"))
    work = tmp_path / "work"
    work.mkdir()
    for alias in ["service-a", "服务甲", "generic-model"]:
        path = store.copy_private(alias, str(work))
        assert json.loads(path.read_text())["id"] == "service-a"
    store.upsert(ModelConfig(id="service-b", name="服务乙", base_url="https://other.test", api_key="假密钥", model="generic-model"))
    with pytest.raises(ValueError, match="唯一"):
        store.copy_private("generic-model", str(work))
    assert json.loads(store.copy_private("service-a", str(work)).read_text())["id"] == "service-a"


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


def test_native_output_limit_recovers_with_small_structured_action(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)

    class LimitedNativeClient(LlmClient):
        def __init__(self):
            self.cfg = SimpleNamespace(id="native", model="generic", capabilities=["tools"], max_output_tokens=16_384, efforts=["minimal"])
            self.default_effort = "minimal"
            self.last_usage = LlmUsage()
            self.native_calls = 0

        def chat_tools(self, *_args, **_kwargs):
            self.native_calls += 1
            self.last_usage = LlmUsage(prompt=100, completion=6000, reasoning=6000)
            raise LlmOutputLimitError("原生工具输出触顶")

        def chat_json(self, messages, **kwargs):
            assert kwargs["plain_text"] is True
            assert kwargs["max_tokens"] == 8000
            assert "4000 字符" in messages[-1]["content"]
            self.last_usage = LlmUsage(prompt=100, completion=50)
            return {"actions": [{"tool": "write_file", "path": "done.py", "content": "ready = True\n"}], "done": True}

    client = LimitedNativeClient()
    graph._client = lambda *_args: client
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "创建文件", "model_id": "native"})
    assert client.native_calls == 1
    assert result["coding_ok"]
    assert result["tokens"] == 6250
    assert any(event.kind == "model" for event in events)
    assert not any(event.kind == "error" for event in events)


def test_exhausted_format_recovery_still_reports_terminal_error(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    client = LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic", capabilities=[]))
    def invalid(*_args, **_kwargs):
        raise LlmError("模型没有返回合法 JSON")
    client.chat_json = invalid
    graph._client = lambda *_args: client
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "创建入口", "model_id": "test"})
    assert result["coding_ok"] is False
    assert len([event for event in events if event.kind == "error"]) == 1


def test_search_rechecks_changed_files_and_reuses_unchanged_results(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    query = {"tool": "search_files", "query": "ready"}
    replies = iter([
        {"actions": [query]},
        {"actions": [{"tool": "write_file", "path": "main.py", "content": "ready = True\n"}, query]},
        {"actions": [query], "done": True},
    ])
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=lambda *_args, **_kwargs: next(replies))
    result = coder(graph, {"session_id": "s1", "workdir": str(tmp_path / "work"), "text": "创建入口并定位内容"})
    assert result["coding_ok"]
    searches = [event.data for event in events if event.kind == "tool" and event.data.get("tool") == "search_files"]
    assert len(searches) == 3
    assert searches[0]["hits"] == []
    assert searches[1]["hits"][0]["path"] == "main.py"
    assert searches[2]["hits"] == searches[1]["hits"] and searches[2]["cached"]
    assert all(event.get("ok") for event in searches)


@pytest.mark.parametrize("changed", [False, True])
def test_cached_discovery_and_segment_reads_preserve_current_version(tmp_path, changed):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    work = tmp_path / "work"
    calls = []
    def reply(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"actions": [{"tool": "list_files"}, {"tool": "run_command", "cmd": "ls"}]}
        if len(calls) == 2:
            return {"actions": [{"tool": "write_file", "path": "main.py", "content": "ready = True\n"}]}
        if len(calls) == 4 and changed:
            (work / "main.py").write_text("fresh = True\n")
        return {"actions": [{"tool": "list_files"}, {"tool": "run_command", "cmd": "ls"},
                            {"tool": "read_file", "path": "main.py", "start": 1, "max_chars": 5}],
                "done": len(calls) == 4}
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    result = coder(graph, {"session_id": "s1", "workdir": str(work), "text": "创建入口", "repo": {"files": []}})
    assert result["coding_ok"]
    tools = [event.data for event in events if event.kind == "tool"]
    assert all(tool.get("ok") for tool in tools)
    reads = [tool for tool in tools if tool.get("tool") == "read_file"]
    assert reads[-1].get("cached", False) is not changed
    assert reads[-1]["read"]["complete"] is False
    assert reads[-1]["text"].endswith("resh " if changed else "eady ")


def test_read_cache_refreshes_visibility_after_context_eviction(tmp_path, monkeypatch):
    """被上下文预算裁掉的文件必须可重新读取，最近读取的文件优先完整展示。"""
    from backend.agents import coder as coder_module

    monkeypatch.setattr(coder_module, "SNAPSHOT_BUDGET", 3_000)
    fs = FsTool(str(tmp_path / "work"))
    for name in ("a.py", "b.py", "c.py"):
        fs.write(name, "value = 1\n# " + "x" * 2000 + "\n")
    graph = make_graph(tmp_path)

    class CacheClient:
        def __init__(self):
            self.calls = 0
            self.last_usage = LlmUsage()

        def chat_json(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"actions": [{"tool": "read_file", "path": "a.py"}]}
            if self.calls == 2:
                assert "[a.py 当前内容]" in messages[-1]["content"]
                return {"actions": [{"tool": "read_file", "path": "b.py"}, {"tool": "read_file", "path": "c.py"}]}
            if self.calls == 3:
                return {"actions": [{"tool": "read_file", "path": "a.py"}]}
            assert "[a.py 当前内容]" in messages[-1]["content"]
            assert "拒绝重复读取已知文件：a.py" not in messages[-1]["content"]
            return {"actions": [{"tool": "replace_file", "path": "a.py", "old": "value = 1", "new": "value = 2"}], "done": True}

    client = CacheClient()
    graph._client = lambda *_args: client
    result = coder(graph, {"session_id": "s1", "workdir": str(fs.root), "text": "精确修改 a.py", "repo": {"files": fs.list()}})
    assert result["coding_ok"]
    assert fs.read("a.py").startswith("value = 2")


def test_reference_markdown_does_not_become_document_write_target(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("service/pyproject.toml", "[project]\nname='demo'\nversion='1'\n")
    state = {"text": "参考 docs/spec.md 实现项目，并生成 README.md", "changes": ["service/main.py"]}
    assert _document_path(graph, state, fs) == "service/README.md"
    state["text"] = "参考 docs/spec.md，更新 docs/guide.md"
    assert _document_path(graph, state, fs) == "docs/guide.md"


def test_module_verification_entrypoints_are_not_confused_with_servers():
    assert _is_verification_command("uv run --directory service python -m py_compile package/main.py")
    assert _is_verification_command("uv run --project service python -m package.build")
    assert _is_verification_command("uv run --project service python -m package.smoke")
    assert not _is_verification_command("uv run --project service python -m package.server")


@pytest.mark.parametrize("project", ["service", "portal"])
def test_explicit_delivery_prefix_restricts_writes_and_private_copy(tmp_path, project):
    graph = make_graph(tmp_path)
    graph.model_store.upsert(ModelConfig(id="generic", name="测试模型", base_url="https://example.test", api_key="假密钥", model="generic"))
    fs = FsTool(str(tmp_path / "work"))
    state = {"text": f"所有项目文件必须使用 `{project}/...` 相对路径", "workdir": str(fs.root)}
    shell, git = ShellTool(str(fs.root)), GitTool(str(fs.root))
    denied = graph._do_action({"tool": "write_file", "path": "README.md", "content": "不应落盘"}, fs, shell, git, state)
    assert not denied["ok"]
    assert not (fs.root / "README.md").exists()
    accepted = graph._do_action({"tool": "write_file", "path": f"{project}/main.py", "content": "ready = True\n"}, fs, shell, git, state)
    assert accepted["ok"]
    copied = graph._do_action({"tool": "model_copy", "model_id": "generic"}, fs, shell, git, state)
    assert copied["file"] == f"{project}/.local/model.json"
    assert copied["files"][-1] == f"{project}/.gitignore"
    assert not (fs.root / ".local").exists()
    assert not (fs.root / ".gitignore").exists()
    assert _document_path(graph, state, fs) == f"{project}/README.md"


@pytest.mark.skipif(shutil.which("uv") is None or os.name == "nt", reason="需要 uv 与 POSIX 可执行测试替身")
def test_project_python_rejects_borrowed_host_test_runner(tmp_path, monkeypatch):
    """旧命令可错误借用宿主零退出程序，新命令必须暴露项目缺少依赖。"""
    fs = FsTool(str(tmp_path / "work"))
    fs.write("service/pyproject.toml", "[project]\nname='isolated-demo'\nversion='1.0.0'\nrequires-python='>=3.11'\ndependencies=[]\n[tool.uv]\npackage=false\n")
    fs.write("service/tests/test_core.py", "def test_failure(): assert False\n")
    host = tmp_path / "host"
    host.mkdir()
    runner = host / "pytest"
    runner.write_text("#!/bin/sh\nprintf '宿主测试替身被借用\\n'\n")
    runner.chmod(0o755)
    monkeypatch.setenv("PATH", f"{host}{os.pathsep}{os.environ['PATH']}")
    shell = ShellTool(str(fs.root))
    borrowed = shell.run("uv run --project service pytest -q service/tests")
    assert borrowed["ok"]
    assert "宿主测试替身" in borrowed["out"]
    isolated = shell.run("uv run --project service python -m pytest -q service/tests")
    assert not isolated["ok"]
    assert "No module named pytest" in isolated["err"]


@pytest.mark.parametrize("marker,container,space", [("｜DSML｜", "calls", " "), ("｜｜DSML｜｜", "calls", " "), ("|DSML|", "tool_calls", ""), ("｜DSML｜", "function_calls", "")])
def test_dsml_variants_preserve_source_and_typed_arguments(marker, container, space):
    client = LlmClient(ModelConfig(id="generic", name="测试", base_url="https://example.test", api_key="假密钥", model="generic"))
    content = "  value = '&lt;'\n\n"
    text = f'<{marker}{space}{container}>\n<{marker}{space}invoke name="write_file">\n<{marker}{space}parameter name="path" string="true">service/main.py</{marker}{space}parameter>\n<{marker}{space}parameter name="content" string="true">{content}</{marker}{space}parameter>\n</{marker}{space}invoke>\n<{marker}{space}invoke name="read_file">\n<{marker}{space}parameter name="path" string="true">service/main.py</{marker}{space}parameter>\n<{marker}{space}parameter name="start" string="false">12</{marker}{space}parameter>\n</{marker}{space}invoke>\n</{marker}{space}{container}>'
    client.chat = lambda *_args, **_kwargs: text
    result = client.chat_json([], plain_text=True)
    assert result["actions"][0]["content"] == content
    assert result["actions"][1]["start"] == 12
    assert result["done"] is False


def test_dsml_truncated_batch_duplicate_parameters_and_prose_do_not_execute():
    client = LlmClient(ModelConfig(id="generic", name="测试", base_url="https://example.test", api_key="假密钥", model="generic"))
    call = '<｜DSML｜ invoke name="read_file"><｜DSML｜ parameter name="path" string="true">main.py</｜DSML｜ parameter></｜DSML｜ invoke>'
    with pytest.raises(LlmError, match="边界不完整"):
        client._parse_dsml_tool_calls('<｜DSML｜ calls>' + call)
    duplicate = call.replace('</｜DSML｜ invoke>', '<｜DSML｜ parameter name="path" string="true">other.py</｜DSML｜ parameter></｜DSML｜ invoke>')
    with pytest.raises(LlmError, match="参数重复"):
        client._parse_dsml_tool_calls('<｜DSML｜ calls>' + duplicate + '</｜DSML｜ calls>')
    assert client._parse_dsml_tool_calls('示例：<｜DSML｜ calls>' + call + '</｜DSML｜ calls>') is None


def test_inspection_adapter_keeps_scope_ranges_and_private_metadata(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    state = {"text": "项目输出到 portal"}
    fs.write("portal/main.py", "# 数据\nready = True\n")
    shell, git = ShellTool(str(fs.root)), GitTool(str(fs.root))
    for command, expected in [("head -n 1 portal/main.py", "# 数据\n"), ("tail -n1 portal/main.py", "ready = True\n"), ("tail -c 5 portal/main.py", "True\n")]:
        action = inspection_read(fs, state, command)
        assert action and action["tool"] == "read_file"
        assert graph._do_action(action, fs, shell, git, state)["text"].endswith(expected)
    assert inspection_read(fs, state, "cat main.py; echo unsafe") is None
    assert inspection_read(fs, state, "cat ../outside.py") is None
    private = fs.root / "portal/.local/model.json"
    private.parent.mkdir()
    private.write_text('{"api_key":"TEST_ONLY_PRIVATE"}')
    action = inspection_read(fs, state, "cat portal/.local/model.json")
    result = graph._do_action(action, fs, shell, git, state)
    assert result["content_available"] is False and "TEST_ONLY_PRIVATE" not in str(result)
    for command, expected in [("wc -l portal/main.py", 2), ("wc -m portal/main.py", len("# 数据\nready = True\n")), ("wc -c portal/main.py", len("# 数据\nready = True\n".encode()))]:
        result = graph._do_action(inspection_read(fs, state, command), fs, shell, git, state)
        assert result["value"] == expected and result["content_available"] is False
    result = graph._do_action(inspection_read(fs, state, "wc -l portal/.local/model.json"), fs, shell, git, state)
    assert result["content_available"] is False and "value" not in result


def test_dsml_tail_request_uses_file_tool_without_shell_execution(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    work = tmp_path / "work"
    FsTool(str(work)).write("main.py", "ready = True\n")
    client = LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic", capabilities=[]))
    replies = iter([
        '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="run_command"><｜｜DSML｜｜ parameter name="cmd" string="true">tail -c 5 main.py</｜｜DSML｜｜ parameter></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>',
        '{"actions":[{"tool":"write_file","path":"done.py","content":"ready = True"}],"done":true}',
    ])
    client.chat = lambda *_args, **_kwargs: next(replies)
    graph._client = lambda *_args: client
    graph._shell = lambda *_args: SimpleNamespace(run=lambda *_args, **_kwargs: pytest.fail("只读命令不应进入 shell"))
    result = coder(graph, {"session_id": "s1", "workdir": str(work), "text": "查看末尾并生成入口", "model_id": "test"})
    assert result["coding_ok"]
    read = next(event.data for event in events if event.kind == "tool" and event.data.get("tool") == "read_file")
    assert read["adapted_from"] == "tail -c 5 main.py" and read["text"].endswith("True\n")


def test_nested_custom_modal_cannot_pass_with_missing_styles(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/web/index.html", '<div id="overlay" class="overlay" role="dialog" aria-modal="true" hidden><p>详情</p></div>')
    fs.write("portal/web/main.js", 'document.getElementById("overlay");')
    fs.write("portal/web/style.css", '.card { color: black; }')
    result = graph._static_web_check(fs, fs.list())
    assert not result["ok"]
    assert "自定义模态样式缺失" in result["out"]
    fs.write("portal/web/style.css", '#overlay { position: fixed; inset: 0; display: grid; }')
    assert graph._static_web_check(fs, fs.list())["ok"]


def test_native_dialog_and_script_only_classes_do_not_require_extra_styles(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("web/index.html", '<dialog aria-modal="true" role="dialog"><span class="script-hook">内容</span></dialog>')
    assert graph._static_web_check(fs, fs.list())["ok"]


def test_inline_modal_styles_are_valid_and_input_pages_are_not_delivery_checks(tmp_path):
    graph = make_graph(tmp_path)
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/index.html", '<style>#overlay { position:fixed;inset:0; }</style><div id="overlay" role="dialog" aria-modal="true"></div>')
    assert graph._static_web_check(fs, fs.list())["ok"]
    fs.write("input/index.html", '<div id="reference" role="dialog" aria-modal="true"></div>')
    graph._verification_commands = lambda *_args: []
    graph._ctx_text = lambda *_args: "只检查交付目录"
    result = verifier(graph, {"session_id": "s1", "workdir": str(fs.root), "text": "所有项目文件必须使用 `portal/...` 相对路径"})
    assert result["tests_ok"]


def test_latest_explicit_read_permission_overrides_earlier_stop():
    assert _guidance_forbids_reads({"original_text": "不要再搜索或读取，直接实现"})
    assert not _guidance_forbids_reads({"guidance": ["不要再读取", "允许读取失败涉及的文件"]})
    assert _guidance_forbids_reads({"guidance": ["允许读取文件", "不要再搜索或读取"]})


def test_qwen_xml_tools_preserve_code_and_keep_whitelist_enforcement(tmp_path):
    client = LlmClient(ModelConfig(id="generic", name="测试", base_url="https://example.test", api_key="假密钥", model="generic"))
    source = '  value = "中文"\n'
    text = f'直接修改：\n<tool_call>\n<function=replace_file>\n<parameter=path>\nservice/main.py\n</parameter>\n<parameter=old>\nold\n</parameter>\n<parameter=new>\n{source}\n</parameter>\n<parameter=expected>\n1\n</parameter>\n</function>\n</tool_call>'
    client.chat = lambda *_args, **_kwargs: text
    result = client.chat_json([], plain_text=True)
    assert result["actions"][0]["new"] == source
    assert result["actions"][0]["expected"] == 1
    assert result["done"] is False
    with pytest.raises(LlmError, match="边界不完整"):
        client._parse_qwen_tool_calls(text[:-12])
    invalid = client._parse_qwen_tool_calls('<tool_call><function=unknown></function></tool_call>')
    assert make_graph(tmp_path)._normalize_actions(invalid["actions"])[0]["_error"]
