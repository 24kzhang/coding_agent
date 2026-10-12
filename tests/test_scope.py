"""命令目录与预检查使用同一套项目路径语义。"""

from types import SimpleNamespace

import pytest

from backend.agents.graph import AgentGraph
from backend.agents.precheck import command_issues, import_issues, manifest_issues
from backend.tools import FsTool
from backend.tools.scope import command_scope, process_cwd


def test_workspace_alias_is_normalized_before_execution(tmp_path):
    fs = FsTool(str(tmp_path / "workspace"))
    state = {"text": "项目输出到 portal"}
    command = 'uv run --project=workspace/portal python "workspace/portal/check file.py"'
    cwd, actual = command_scope(fs, state, command)
    assert cwd == fs.root
    assert actual == "uv run --project=portal python 'portal/check file.py'"
    captured = []
    shell = SimpleNamespace(run=lambda cmd, **kwargs: captured.append((cmd, kwargs)) or {"ok": True})
    result = AgentGraph._run_command(None, fs, shell, state, command)
    assert captured == [(actual, {"timeout": 180})]
    assert result["requested_cmd"] == command


def test_json_input_is_checked_in_effective_uv_directory(tmp_path):
    fs = FsTool(str(tmp_path / "workspace"))
    state = {"text": "项目输出到 portal"}
    fs.write("inputs/source.json", "[]")
    (fs.root / "portal").mkdir()
    assert command_issues(fs, state, "uv --directory portal run python -m json.tool inputs/source.json")
    assert not command_issues(fs, state, f"uv --directory portal run python -m json.tool {fs.root / 'inputs/source.json'}")
    assert not command_issues(fs, state, "uv run --project portal python -m json.tool inputs/source.json")
    assert process_cwd(fs.root, ["uv", "run", "--directory=portal", "python", "--directory", "ignored"]) == fs.root / "portal"


def test_uv_dependency_inspection_uses_uv_without_requiring_pip_module(tmp_path):
    fs = FsTool(str(tmp_path / "workspace"))
    assert command_issues(fs, {}, "uv run --frozen python -m pip list")
    assert not command_issues(fs, {}, "uv pip list")
    assert not command_issues(fs, {}, "python -m pip list")


@pytest.mark.parametrize("command,expected", [
    ("uv run --project portal pytest -q", True),
    ("uv run --project=portal python -m pytest -q", True),
    ("uv run --directory portal python -m pytest -q", False),
    ("uv run --project portal pytest -c portal/pyproject.toml", False),
    ("uv run --project portal pytest portal/tests", False),
])
def test_uv_pytest_project_selection_does_not_replace_import_directory(tmp_path, command, expected):
    fs = FsTool(str(tmp_path / "workspace"))
    fs.write("portal/pyproject.toml", "[project]\nname='portal'\nversion='1'\n[tool.pytest.ini_options]\npythonpath=['.']\n")
    assert bool(manifest_issues(fs, {"text": "项目输出到 portal"}, command)) is expected


def test_script_and_explicit_directory_must_exist_before_launch(tmp_path):
    fs = FsTool(str(tmp_path / "workspace"))
    state = {"text": "项目输出到 portal"}
    assert command_issues(fs, state, "uv run --directory portal python tools/probe.py")
    fs.write("portal/pyproject.toml", "[project]\nname='portal'\nversion='0.1'\n")
    assert command_issues(fs, state, "uv run --directory portal python tools/probe.py")
    fs.write("portal/tools/probe.py", "print('检查')")
    assert not command_issues(fs, state, "uv run --directory portal python tools/probe.py")
    assert not command_issues(fs, state, "python tools/probe.py")


@pytest.mark.parametrize("definition,expected", [
    ("def fake_send():\n        return ('回复', {})", True),
    ("async def fake_send():\n        return ('回复', {})", False),
    ("def fake_send():\n        return make_awaitable()", False),
])
def test_async_patch_contract_checks_only_proven_nonawaitable_returns(tmp_path, definition, expected):
    fs = FsTool(str(tmp_path / "workspace"))
    fs.write("portal/service.py", "async def send():\n    return '完成'\n")
    fs.write("portal/test_api.py", "import service as mod\ndef env(monkeypatch):\n    " + definition + "\n    monkeypatch.setattr(mod, 'send', fake_send)\n")
    issues = import_issues(fs, {"text": "项目输出到 portal"}, ["portal/service.py", "portal/test_api.py"])
    assert bool(issues) is expected


@pytest.mark.parametrize("command", [
    "uv --directory=workspace/portal sync",
    "uv sync --project=portal",
    "uv --directory portal run --project child pytest",
    "uv run --project child --directory portal pytest",
])
def test_manifest_precheck_resolves_equals_flags_and_directory_before_project(tmp_path, command):
    fs = FsTool(str(tmp_path / "workspace"))
    state = {"text": "项目输出到 portal"}
    fs.write("portal/pyproject.toml", '[project]\nreadme = "missing.md"\n[build-system]\nrequires = []\n')
    fs.write("portal/child/pyproject.toml", '[project]\nreadme = "child.md"\n[build-system]\nrequires = []\n')
    # 显式 portal 前缀令命令在父工作区执行；程序后的同名选项不得改动预检查目录。
    issues = manifest_issues(fs, state, command)
    assert len(issues) == 1
    assert ("child.md" if "child" in command else "missing.md") in issues[0]
    assert manifest_issues(fs, state, "uv run pytest --project ignored")


@pytest.mark.parametrize("malformed", ['tool = "invalid"', 'tool = {uv = "invalid", hatch = {build = "invalid"}}', 'tool = {hatch = {build = {targets = {wheel = {packages = 1}}}}}', 'project = "invalid"'])
def test_malformed_manifest_tables_do_not_crash_precheck(tmp_path, malformed):
    fs = FsTool(str(tmp_path / "workspace"))
    fs.write("portal/pyproject.toml", malformed + '\n[build-system]\nrequires = []\n')
    assert manifest_issues(fs, {"text": "项目输出到 portal"}, "uv sync") == []
