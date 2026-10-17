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
