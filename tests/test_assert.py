"""内部实现标识与用户明确的文字禁止约束必须区分。"""

from types import SimpleNamespace

import pytest

from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.agents.precheck import assertion_issues
from backend.memory import MemoryStore
from backend.tools import FsTool, GitTool, ShellTool
from llm.client import LlmUsage
from llm.store import ModelStore


def test_ad_hoc_source_identifier_check_requires_behavioral_validation():
    action = {"paths": ["service.py"], "texts": ["cfg.host", "host_of"]}
    assert assertion_issues({"text": "实现模型调用"}, action)
    assert not assertion_issues({"text": "删除 cfg.host，禁止保留 host_of"}, action)


def test_explicit_resource_constraints_and_document_text_are_not_blocked():
    assert not assertion_issues({"text": "实现接口"}, {"paths": ["page.js"], "texts": ["/private/images/", "unsafe-token-value"]})
    assert not assertion_issues({"text": "更新说明"}, {"paths": ["README.md"], "texts": ["旧版本"]})


def test_positive_source_snippet_requires_behavior_but_explicit_format_is_allowed():
    action = {"tool": "assert_text_present", "paths": ["page.js"], "texts": ["items.forEach((item) => pane.appendChild(item));"]}
    assert assertion_issues({"text": "显示全部条目"}, action)
    assert not assertion_issues({"text": "源码必须包含 " + action["texts"][0]}, action)
    assert not assertion_issues({"text": "忽略环境目录"}, {"tool": "assert_text_present", "paths": [".gitignore"], "texts": [".venv/"]})


def test_ad_hoc_check_is_not_executed_and_has_bounded_revisions(tmp_path):
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    calls = []
    def reply(*_args, **_kwargs):
        calls.append(1)
        return {"actions": [{"tool": "assert_text_absent", "paths": ["main.py"], "texts": ["helper_name"]}], "done": True}
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    graph._do_action = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("临时标识断言不应执行"))
    result = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建项目"})
    assert not result["coding_ok"] and len(calls) == 3


def test_ignore_rule_requires_presence_and_explicit_removal_remains_supported(tmp_path):
    action = {"paths": [".gitignore"], "texts": [".venv"]}
    assert assertion_issues({"text": "将 .venv 写入忽略文件"}, action)
    assert not assertion_issues({"text": "移除 .venv 忽略规则"}, action)
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    fs = FsTool(str(tmp_path / "work"))
    fs.write(".gitignore", ".venv/\n.local/\n")
    shell, git = ShellTool(str(fs.root)), GitTool(str(fs.root))
    assert graph._do_action({"tool": "assert_text_present", **action}, fs, shell, git)["ok"]
    assert not graph._do_action({"tool": "assert_text_present", "paths": [".gitignore"], "texts": ["missing/"]}, fs, shell, git)["ok"]
    fs.write("portal/.gitignore", ".venv/\n")
    scoped = {"tool": "assert_text_present", "paths": [".gitignore"], "texts": [".venv"]}
    assert graph._do_action(scoped, fs, shell, git, {"text": "项目输出到 portal"})["ok"]


@pytest.mark.parametrize("tool", ["assert_text_present", "assert_text_absent"])
def test_private_assertions_are_revised_without_reading_body(tmp_path, tool):
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    fs = FsTool(str(tmp_path / "work"))
    path = fs.root / ".local/model.json"
    path.parent.mkdir()
    path.write_text('{"api_key":"TEST_ONLY_PRIVATE"}')
    calls = []
    def reply(messages, **_kwargs):
        calls.append(messages)
        assert "TEST_ONLY_PRIVATE" not in str(messages)
        return {"actions": [{"tool": tool, "paths": [".local/model.json"], "texts": ["api_key"]}], "done": True}
    graph._client = lambda *_args: SimpleNamespace(last_usage=LlmUsage(), chat_json=reply)
    graph._do_action = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("私密断言不得执行"))
    result = coder(graph, {"workdir": str(fs.root), "session_id": "s1", "text": "核对 api_key 字段合同"})
    assert not result["coding_ok"] and len(calls) == 3
