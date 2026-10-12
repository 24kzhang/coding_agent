"""语法检查应真正执行，但不重复做整份需求的模型审查。"""
import subprocess
import sys

import pytest

from backend.agents.verifier import _syntax_only, review_before_verification
from backend.tools import FsTool


@pytest.mark.parametrize("command", ["python -m compileall -q .", "uv run python -m py_compile main.py", "node --check main.js"])
def test_syntax_checks_do_not_request_model_review(tmp_path, command):
    class Graph:
        def _client(self, *_args):
            raise AssertionError("语法检查不应调用模型审查")
    fs = FsTool(str(tmp_path))
    assert review_before_verification(Graph(), {}, fs, [], command) == []


def test_local_module_shadow_and_runtime_commands_still_need_review(tmp_path):
    fs = FsTool(str(tmp_path))
    (tmp_path / "compileall.py").write_text("raise RuntimeError('业务副作用')")
    assert not _syntax_only(fs, {}, "python -m compileall")
    assert not _syntax_only(fs, {}, "python -m pytest tests")
    assert not _syntax_only(fs, {}, "node --require ./boot.js --check main.js")


def test_skipping_semantic_review_does_not_mask_actual_syntax_failure(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("def broken(:\n")
    assert _syntax_only(FsTool(str(tmp_path)), {}, "python -m py_compile bad.py")
    result = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True)
    assert result.returncode != 0 and b"SyntaxError" in result.stderr
