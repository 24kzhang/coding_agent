from __future__ import annotations

from subprocess import CompletedProcess, TimeoutExpired
from types import SimpleNamespace

import pytest

from backend.tools import ContainerShellTool, FsTool, GitTool, ShellTool
from backend.tools import container as container_module


def test_fs_skips_dependencies_and_supports_precise_replace(tmp_path) -> None:
    workdir = tmp_path / "project"
    fs = FsTool(str(workdir))
    fs.write("src/app.py", "name = 'old'\n")
    dependency = workdir / "node_modules" / "pkg" / "index.js"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("ignored", encoding="utf-8")
    uv_cache = workdir / ".uv-cache" / "archive-v0" / "pkg" / "module.py"
    uv_cache.parent.mkdir(parents=True)
    uv_cache.write_text("def ignored_cache_symbol(): ...\n", encoding="utf-8")
    media = workdir / "assets" / "demo.gif"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"GIF89a")
    archive = workdir / "dataset.zip"
    archive.write_bytes(b"zip")

    assert fs.list() == ["src/app.py"]
    assert fs.search("ignored_cache_symbol") == []
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


def test_fs_replace_block_collapses_exact_adjacent_boundary_overlap(tmp_path) -> None:
    fs = FsTool(str(tmp_path))
    fs.write("main.py", "# 开始\nold = False\ndef next_item():\n    return 1\n")
    fs.read("main.py")
    fs.replace_block("main.py", "# 开始", "def next_item():", "# 开始\nready = True\ndef next_item():\n")
    result = fs.read("main.py")
    assert result == "# 开始\nready = True\ndef next_item():\n    return 1\n"
    compile(result, "main.py", "exec")


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


def test_container_subdirectory_cwd_preserves_mount_boundary(tmp_path, monkeypatch):
    """命令在交付子目录执行，输入工作区仍挂载且不能通过链接逃逸。"""

    project = tmp_path / "portal"
    project.mkdir()
    calls = []
    tool = ContainerShellTool(str(tmp_path), image="test")
    tool.docker = "/usr/bin/docker"
    tool.container_id = "container"
    monkeypatch.setattr(tool, "ensure_ready", lambda: None)

    def run(argv, **_kwargs):
        calls.append(argv)
        return CompletedProcess(argv, 0, "完成", "")

    monkeypatch.setattr(container_module.subprocess, "run", run)
    result = tool.run("python tools/probe.py", cwd=project)
    assert result["ok"] is True
    assert calls[0][2:4] == ["--workdir", "/workspace/portal"]
    assert result["cwd"] == str(project)
    assert tool.run("python --version", cwd=tmp_path.parent)["code"] == 126
    (project / "link.py").symlink_to(tmp_path.parent / "outside.py")
    assert tool.run("python link.py", cwd=project)["code"] == 126
    assert len(calls) == 1


def test_container_shell_starts_once_executes_and_cleans_up(tmp_path, monkeypatch) -> None:
    """同一个容器执行器必须复用容器，并带上挂载和最小权限参数。"""

    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        args = [str(item) for item in argv]
        calls.append(args)
        if args[1:3] == ["version", "--format"]:
            return CompletedProcess(args, 0, "27.0.0\n", "")
        if args[1:3] == ["image", "inspect"]:
            return CompletedProcess(args, 0, "[]", "")
        if args[1] == "run":
            return CompletedProcess(args, 0, "container-123\n", "")
        if args[1] == "exec":
            return CompletedProcess(args, 0, "Python 3.12.0\n", "")
        if args[1] == "stop":
            return CompletedProcess(args, 0, "container-123\n", "")
        raise AssertionError(args)

    monkeypatch.setattr(container_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(container_module.subprocess, "run", fake_run)
    tool = ContainerShellTool(str(tmp_path), image="coding-agent-shell:test")

    first = tool.run("python --version")
    second = tool.run("node --version")
    tool.close()

    assert first["ok"] is True
    assert second["ok"] is True
    run_calls = [args for args in calls if args[1] == "run"]
    exec_calls = [args for args in calls if args[1] == "exec"]
    assert len(run_calls) == 1
    assert len(exec_calls) == 2
    assert f"{tmp_path.resolve()}:/workspace:rw" in run_calls[0]
    assert "no-new-privileges:true" in run_calls[0]
    assert "ALL" in run_calls[0]
    assert "/var/run/docker.sock" not in " ".join(run_calls[0])
    assert [args[1] for args in calls].count("stop") == 1


def test_container_shell_builds_versioned_default_image_when_missing(tmp_path, monkeypatch) -> None:
    """默认镜像缺失时按 Dockerfile 哈希构建，之后再启动任务容器。"""

    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        args = [str(item) for item in argv]
        calls.append(args)
        if args[1:3] == ["version", "--format"]:
            return CompletedProcess(args, 0, "27.0.0\n", "")
        if args[1:3] == ["image", "inspect"]:
            return CompletedProcess(args, 1, "", "No such image")
        if args[1] == "build":
            return CompletedProcess(args, 0, "built", "")
        if args[1] == "run":
            return CompletedProcess(args, 0, "container-123\n", "")
        if args[1] == "stop":
            return CompletedProcess(args, 0, "", "")
        raise AssertionError(args)

    monkeypatch.setattr(container_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(container_module.subprocess, "run", fake_run)
    tool = ContainerShellTool(str(tmp_path))

    tool.ensure_ready()
    tool.close()

    build_calls = [args for args in calls if args[1] == "build"]
    assert len(build_calls) == 1
    assert tool.image.startswith("coding-agent-shell:")
    assert build_calls[0][build_calls[0].index("--tag") + 1] == tool.image


def test_container_shell_never_falls_back_without_docker(tmp_path, monkeypatch) -> None:
    """Docker 缺失时返回基础设施失败，且不会执行任何宿主命令。"""

    monkeypatch.setattr(container_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        container_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Docker 缺失时不应启动任何进程"),
    )
    result = ContainerShellTool(str(tmp_path), image="coding-agent-shell:test").run("python --version")

    assert result["ok"] is False
    assert result["code"] == 125
    assert result["infrastructure"] is True
    assert "未找到 Docker CLI" in result["err"]


def test_container_shell_reuses_policy_and_maps_project_absolute_paths(tmp_path, monkeypatch) -> None:
    """容器模式复用危险命令策略，并把宿主项目绝对路径映射到 /workspace。"""

    project_file = tmp_path / "src" / "app.py"
    project_file.parent.mkdir()
    project_file.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(container_module.shutil, "which", lambda _name: None)
    tool = ContainerShellTool(str(tmp_path), image="coding-agent-shell:test")

    assert tool.run("rm app.py")["code"] == 126
    assert tool.run("python -c pass")["code"] == 126
    prepared = tool.run(f"python {project_file}")

    assert prepared["code"] == 125
    assert prepared["argv"] == ["python", "/workspace/src/app.py"]


def test_container_shell_timeout_stops_container(tmp_path, monkeypatch) -> None:
    """docker exec 超时后必须停止容器，不能留下后台命令。"""

    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        args = [str(item) for item in argv]
        calls.append(args)
        if args[1:3] == ["version", "--format"]:
            return CompletedProcess(args, 0, "27.0.0\n", "")
        if args[1:3] == ["image", "inspect"]:
            return CompletedProcess(args, 0, "[]", "")
        if args[1] == "run":
            return CompletedProcess(args, 0, "container-123\n", "")
        if args[1] == "exec":
            raise TimeoutExpired(args, 1, output="partial")
        if args[1] == "stop":
            return CompletedProcess(args, 0, "", "")
        raise AssertionError(args)

    monkeypatch.setattr(container_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(container_module.subprocess, "run", fake_run)
    result = ContainerShellTool(str(tmp_path), image="coding-agent-shell:test").run("pytest -q", timeout=1)

    assert result["code"] == 124
    assert "已停止任务容器" in result["err"]
    assert [args[1] for args in calls].count("stop") == 1


def test_shell_forces_uv_init_to_stay_out_of_parent_workspace(tmp_path, monkeypatch) -> None:
    """子项目初始化不能让 uv 修改用户所选目录之外的父项目。"""

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return SimpleNamespace(wait=lambda **_options: 0)

    monkeypatch.setattr("backend.tools.shell.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("backend.tools.shell.subprocess.Popen", fake_run)

    result = ShellTool(str(tmp_path)).run("uv init --package")

    assert result["ok"] is True
    assert captured["argv"] == ["/usr/bin/uv", "init", "--no-workspace", "--package"]


def test_shell_drops_virtualenv_outside_selected_project(tmp_path, monkeypatch) -> None:
    """Agent 自身虚拟环境不能让用户项目中的 uv 产生环境错配警告。"""

    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(wait=lambda **_options: 0)

    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path.parent / "agent" / ".venv"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path.parent / "agent" / "packages"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path.parent / "agent"))
    monkeypatch.setattr("backend.tools.shell.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr("backend.tools.shell.subprocess.Popen", fake_run)

    result = ShellTool(str(tmp_path / "user-project")).run("uv run pytest -q")

    assert result["ok"] is True
    assert "VIRTUAL_ENV" not in captured["env"]
    assert "PYTHONPATH" not in captured["env"]
    assert "PYTHONHOME" not in captured["env"]
    assert "coding-agent-cache" in captured["env"]["UV_CACHE_DIR"]
    assert "coding-agent-cache" in captured["env"]["NPM_CONFIG_CACHE"]
    for key in ("UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR", "PLAYWRIGHT_BROWSERS_PATH"):
        assert "coding-agent-cache" in captured["env"][key]
        assert str(tmp_path / "user-project") not in captured["env"][key]
    assert str(tmp_path / "user-project") not in captured["env"]["UV_CACHE_DIR"]


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
