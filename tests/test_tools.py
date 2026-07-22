from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.tools import FsTool, GitTool, ShellTool


def test_fs_skips_dependencies_and_supports_precise_replace(tmp_path) -> None:
    workdir = tmp_path / "project"
    fs = FsTool(str(workdir))
    fs.write("src/app.py", "name = 'old'\n")
    dependency = workdir / "node_modules" / "pkg" / "index.js"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("ignored", encoding="utf-8")
    media = workdir / "assets" / "demo.gif"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"GIF89a")
    archive = workdir / "dataset.zip"
    archive.write_bytes(b"zip")

    assert fs.list() == ["src/app.py"]
    assert fs.replace("src/app.py", "'old'", "'new'") == "src/app.py"
    assert fs.read("src/app.py") == "name = 'new'\n"


def test_fs_replaces_long_block_between_unique_markers(tmp_path) -> None:
    """长代码块可按唯一边界替换，模型无需回传整段旧内容。"""

    fs = FsTool(str(tmp_path / "project"))
    fs.write("app.js", "// before\nfunction old() {\n    return 1;\n}\n// after\nconst ready = true;\n")

    result = fs.replace_block(
        "app.js",
        "function old() {",
        "// after",
        "function current() {\n    return 2;\n}\n",
    )

    assert result == "app.js"
    assert fs.read("app.js") == "// before\nfunction current() {\n    return 2;\n}\n// after\nconst ready = true;\n"


def test_fs_replace_block_supports_end_of_file_and_rejects_ambiguous_start(tmp_path) -> None:
    """文件尾替换使用空结束标记，重复开始标记必须拒绝。"""

    fs = FsTool(str(tmp_path / "project"))
    fs.write("tail.js", "const head = 1;\n// tail\nold\n")
    fs.replace_block("tail.js", "// tail", "", "// tail\nnew\n")
    assert fs.read("tail.js") == "const head = 1;\n// tail\nnew\n"

    fs.write("duplicate.js", "// same\na\n// same\nb\n")
    with pytest.raises(ValueError, match="开始标记必须唯一"):
        fs.replace_block("duplicate.js", "// same", "", "replacement\n")


def test_fs_searches_text_and_blocks_sensitive_files(tmp_path) -> None:
    fs = FsTool(str(tmp_path / "project"))
    fs.write("src/service.py", "def load_session():\n    return None\n")

    assert fs.search("load_session")[0]["line"] == 1
    with pytest.raises(ValueError, match="敏感配置"):
        fs.write(".env", "TOKEN=secret")


def test_fs_finds_forbidden_text_in_selected_files(tmp_path) -> None:
    """不存在断言只检查指定文件，并返回可定位的命中信息。"""

    fs = FsTool(str(tmp_path / "project"))
    fs.write("README.md", "第一行\n禁止示例\n")
    fs.write("start.sh", "echo ready\n")

    assert fs.find_text(["README.md", "start.sh"], ["禁止示例", "your_key"]) == [
        {"path": "README.md", "text": "禁止示例", "line": 2}
    ]


def test_shell_rejects_compound_and_inline_commands(tmp_path) -> None:
    shell = ShellTool(str(tmp_path))

    assert shell.run("python --version")["ok"] is True
    assert shell.run("python --version && python --version")["code"] == 126
    assert shell.run("python -c pass")["code"] == 126
    assert shell.run("uv run python -c pass")["code"] == 126


def test_shell_forces_uv_init_to_stay_out_of_parent_workspace(tmp_path, monkeypatch) -> None:
    """子项目初始化不能让 uv 修改用户所选目录之外的父项目。"""

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("backend.tools.shell.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("backend.tools.shell.subprocess.run", fake_run)

    result = ShellTool(str(tmp_path)).run("uv init --package")

    assert result["ok"] is True
    assert captured["argv"] == ["/usr/bin/uv", "init", "--no-workspace", "--package"]


def test_shell_drops_virtualenv_outside_selected_project(tmp_path, monkeypatch) -> None:
    """Agent 自身虚拟环境不能让用户项目中的 uv 产生环境错配警告。"""

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path.parent / "agent" / ".venv"))
    monkeypatch.setattr("backend.tools.shell.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("backend.tools.shell.subprocess.run", fake_run)

    result = ShellTool(str(tmp_path / "user-project")).run("uv run pytest -q")

    assert result["ok"] is True
    assert "VIRTUAL_ENV" not in captured["env"]


def test_git_does_not_walk_into_parent_repository(tmp_path) -> None:
    """子项目未初始化 Git 时不能泄露父仓库的状态和差异。"""

    (tmp_path / ".git").mkdir()
    child = tmp_path / "child"
    child.mkdir()
    git = GitTool(str(child))

    assert git.status() == "当前项目目录未初始化 Git 仓库"
    assert git.diff() == "当前项目目录未初始化 Git 仓库"


def test_shell_git_command_does_not_walk_into_parent_repository(tmp_path) -> None:
    """模型通过 run_command 调 Git 时也不能绕过 GitTool 的项目边界。"""

    (tmp_path / ".git").mkdir()
    child = tmp_path / "child"
    child.mkdir()

    result = ShellTool(str(child)).run("git diff --stat")

    assert result["ok"] is False
    assert result["code"] == 126
    assert "父级仓库" in result["err"]
