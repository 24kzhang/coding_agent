"""脚本输入预检区分确定缺失、异常回退、动态路径与先写后读。"""

import pytest

from backend.agents.precheck import command_issues
from backend.tools import FsTool


@pytest.mark.parametrize("source,expected", [
    ("open('inputs/data.json').read()", True),
    ("from pathlib import Path\nPath('inputs/data.json').read_text()", True),
    ("try:\n    open('inputs/data.json').read()\nexcept FileNotFoundError:\n    pass", False),
    ("open('inputs/data.json', 'w').write('[]')\nopen('inputs/data.json').read()", False),
    ("path='inputs/data.json'\nopen(path).read()", False),
    ("def open(path):\n    return []\nopen('inputs/data.json')", False),
    ("from external import bootstrap\nbootstrap()\nopen('inputs/data.json').read()", False),
])
def test_literal_script_input_is_checked_in_actual_directory(tmp_path, source, expected):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/probe.py", source)
    fs.write("inputs/data.json", "[]")
    state = {"text": "项目输出到 portal"}
    assert bool(command_issues(fs, state, "python probe.py")) is expected
    fs.write("portal/inputs/data.json", "[]")
    assert not command_issues(fs, state, "python probe.py")
