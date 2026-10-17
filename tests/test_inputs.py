"""脚本输入预检区分确定缺失、异常回退、动态路径与先写后读。"""

import pytest

from backend.agents.precheck import command_issues, imported_inputs
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


def test_static_file_relative_path_chain_is_resolved_before_execution(tmp_path):
    fs = FsTool(str(tmp_path / 'work'))
    fs.write('inputs/data.json', '[]')
    source = '''import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'inputs' / 'data.json'
def main():
    return json.loads(DATA.read_text())
if __name__ == '__main__':
    main()
'''
    fs.write('portal/tools/probe.py', source)
    state = {'text': '项目输出到 portal'}
    assert command_issues(fs, state, 'python tools/probe.py')
    fs.write('portal/tools/probe.py', source.replace('.parent.parent', '.parents[2]'))
    assert not command_issues(fs, state, 'python tools/probe.py')
    fs.write('portal/tools/probe.py', source.replace("if __name__ == '__main__':\n    main()", ''))
    assert not command_issues(fs, state, 'python tools/probe.py')


def test_static_paths_do_not_assume_unknown_bootstrap_has_no_effect(tmp_path):
    fs = FsTool(str(tmp_path / 'work'))
    fs.write('portal/probe.py', '''from pathlib import Path
from external import prepare
DATA = Path(__file__).resolve().parent / 'later.json'
def main():
    return DATA.read_text()
def bootstrap():
    prepare()
bootstrap()
main()
''')
    assert not command_issues(fs, {'text': '项目输出到 portal'}, 'python probe.py')


def test_later_logging_cannot_create_input_for_earlier_module_read(tmp_path):
    fs = FsTool(str(tmp_path / 'work'))
    fs.write('inputs/data.json', '[]')
    source = '''import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'inputs' / 'data.json'
rows = json.loads(DATA.read_text())
print('数量', len(rows))
'''
    fs.write('portal/tools/probe.py', source)
    state = {'text': '项目输出到 portal'}
    assert command_issues(fs, state, 'python tools/probe.py')
    fs.write('portal/tools/probe.py', source.replace('.parent.parent', '.parents[2]'))
    assert not command_issues(fs, state, 'python tools/probe.py')


def test_imported_module_input_is_checked_before_pytest_collection(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("inputs/rows.json", "[]")
    fs.write("portal/tests/conftest.py", "import entry\n")
    fs.write("portal/entry.py", "import loader\n")
    source = '''import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'inputs' / 'rows.json'
def load():
    return json.loads(DATA.read_text())
ROWS = load()
'''
    fs.write("portal/loader.py", source)
    files = ["portal/tests/conftest.py", "portal/entry.py", "portal/loader.py"]
    issues = imported_inputs(fs, {"text": "项目输出到 portal"}, files, "uv run --directory portal python -m pytest tests -q")
    assert len(issues) == 1 and "portal/loader.py" in issues[0]
    fs.write("portal/loader.py", source.replace("ROOT / 'inputs'", "ROOT.parent / 'inputs'"))
    assert not imported_inputs(fs, {"text": "项目输出到 portal"}, files, "uv run --directory portal python -m pytest tests -q")


@pytest.mark.parametrize("statement", ["if False:\n    import optional", "try:\n    import optional\nexcept OSError:\n    pass", "def later():\n    import optional"])
def test_conditional_or_unused_modules_are_not_treated_as_required_inputs(tmp_path, statement):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/test_main.py", statement)
    fs.write("portal/optional.py", "open('absent.json').read()")
    assert not imported_inputs(fs, {"text": "项目输出到 portal"}, ["portal/test_main.py", "portal/optional.py"], "python -m pytest")


def test_relative_package_import_input_is_checked_without_executing_code(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/test_main.py", "import pkg\n")
    fs.write("portal/pkg/__init__.py", "from . import loader\n")
    fs.write("portal/pkg/loader.py", "from pathlib import Path\nPath('absent.json').read_text()\n")
    issues = imported_inputs(fs, {"text": "项目输出到 portal"}, fs.list(), "python -m pytest")
    assert len(issues) == 1 and "pkg/loader.py" in issues[0]


def test_pytest_selection_does_not_import_unselected_tests(tmp_path):
    fs = FsTool(str(tmp_path / "work"))
    fs.write("portal/tests/test_unit.py", "def test_ok():\n    assert True\n")
    fs.write("portal/integration/test_remote.py", "import absent\n")
    fs.write("portal/absent.py", "open('not_configured.json').read()")
    assert not imported_inputs(fs, {"text": "项目输出到 portal"}, fs.list(), "python -m pytest tests/test_unit.py::test_ok -q")
