"""独立脚本导入路径检查区分模块执行、安装型项目、显式路径与可选回退。"""

import pytest

from backend.agents.precheck import command_issues
from backend.tools import FsTool


@pytest.mark.parametrize("source,expected", [
    ("from pkg.api import ping", True),
    ("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nfrom pkg.api import ping", False),
    ("try:\n    from pkg.api import ping\nexcept ImportError:\n    pass", False),
])
def test_script_import_root_must_be_declared(tmp_path, source, expected):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/pyproject.toml", "[project]\nname='portal'\nversion='1'\n")
    fs.write("portal/pkg/api.py", "def ping():\n    return True\n")
    fs.write("portal/tools/check.py", source)
    state = {"text": "项目输出到 portal"}
    assert bool(command_issues(fs, state, "python tools/check.py")) is expected
    assert not command_issues(fs, state, "python -m tools.check")
    fs.write("portal/check.py", source)
    assert not command_issues(fs, state, "python check.py")
    fs.write("portal/pyproject.toml", "[project]\nname='portal'\nversion='1'\n[build-system]\nrequires=[]\n")
    assert not command_issues(fs, state, "python tools/check.py")
