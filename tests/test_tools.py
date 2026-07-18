from __future__ import annotations

import pytest

from backend.tools import FsTool, ShellTool


def test_fs_skips_dependencies_and_supports_precise_replace(tmp_path) -> None:
    workdir = tmp_path / "project"
    fs = FsTool(str(workdir))
    fs.write("src/app.py", "name = 'old'\n")
    dependency = workdir / "node_modules" / "pkg" / "index.js"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("ignored", encoding="utf-8")

    assert fs.list() == ["src/app.py"]
    assert fs.replace("src/app.py", "'old'", "'new'") == "src/app.py"
    assert fs.read("src/app.py") == "name = 'new'\n"


def test_fs_searches_text_and_blocks_sensitive_files(tmp_path) -> None:
    fs = FsTool(str(tmp_path / "project"))
    fs.write("src/service.py", "def load_session():\n    return None\n")

    assert fs.search("load_session")[0]["line"] == 1
    with pytest.raises(ValueError, match="敏感配置"):
        fs.write(".env", "TOKEN=secret")


def test_shell_rejects_compound_and_inline_commands(tmp_path) -> None:
    shell = ShellTool(str(tmp_path))

    assert shell.run("python --version")["ok"] is True
    assert shell.run("python --version && python --version")["code"] == 126
    assert shell.run("python -c pass")["code"] == 126
    assert shell.run("uv run python -c pass")["code"] == 126
