"""命令目录与预检查使用同一套项目路径语义。"""

from types import SimpleNamespace

import pytest

from backend.agents.graph import AgentGraph
from backend.agents.precheck import command_issues, manifest_issues
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
    assert command_issues(fs, state, "uv --directory portal run python -m json.tool inputs/source.json")
    assert not command_issues(fs, state, f"uv --directory portal run python -m json.tool {fs.root / 'inputs/source.json'}")
    assert not command_issues(fs, state, "uv run --project portal python -m json.tool inputs/source.json")
    assert process_cwd(fs.root, ["uv", "run", "--directory=portal", "python", "--directory", "ignored"]) == fs.root / "portal"


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
