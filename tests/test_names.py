"""编辑后静态发现未定义名称，避免付费审查遗漏后再触发测试收集失败。"""

from backend.agents.precheck import name_issues
from backend.tools import FsTool


def test_undefined_names_are_detected_without_importing_source(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/main.py", 'from pathlib import Path\n\ndef create():\n    return Path(PROJECT_DIR)\n\nraise RuntimeError("不允许导入项目代码")\n')
    fs.write("input.py", "outside = MISSING_INPUT\n")
    issues = name_issues(fs, {"text": "项目输出到 portal"}, ["portal/main.py", "input.py"])
    assert len(issues) == 1 and "PROJECT_DIR" in issues[0]
    fs.read("portal/main.py")
    fs.replace("portal/main.py", "from pathlib import Path\n", 'from pathlib import Path\nPROJECT_DIR = "."\n')
    assert not name_issues(fs, {"text": "项目输出到 portal"}, ["portal/main.py"])


def test_imports_arguments_and_inline_exemptions_are_respected(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("main.py", 'import os\ndef value(parameter):\n    return os.fspath(parameter)\n\nexample = DYNAMIC_VALUE  # noqa: F821\n')
    assert name_issues(fs, {}, ["main.py"]) == []
