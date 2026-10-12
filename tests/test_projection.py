"""列表投影不能被误判为完整详情删字段，用户要求完整列表时仍须保护。"""
import json

from backend.agents.graph import AgentGraph
from backend.agents.planner import _plan_quality_issues
from backend.memory import MemoryStore
from llm.store import ModelStore


def fixture(tmp_path, goal):
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    state = {"text": goal, "repo": {"snippets": {"record.schema.json": json.dumps({"type": "object", "properties": {"id": {"type": "string"}, "cover": {"type": "string"}}})}}}
    text = '''字段为 `id`、`cover`。
列表不返回 `cover` 加载地址，避免误用外部媒体。
左侧动态遍历所有原始字段，`cover` 仅作为路径文本展示。
详情接口 `GET /api/records/{id}` 返回全部原始字段，不删除任何字段。
用 pytest 做自动化测试，静态资源用于页面。
'''
    return graph, state, text


def test_compact_list_with_full_detail_is_allowed(tmp_path):
    graph, state, text = fixture(tmp_path, "详情左侧展示所有详细信息。")
    assert _plan_quality_issues(graph, state, text) == []


def test_full_list_requirement_still_rejects_omitted_fields(tmp_path):
    graph, state, text = fixture(tmp_path, "列表必须展示全部原始字段，详情展示所有详细信息。")
    assert any("明确排除了原始字段：cover" in issue for issue in _plan_quality_issues(graph, state, text))


def test_detail_omission_cannot_hide_in_list_sentence(tmp_path):
    graph, state, text = fixture(tmp_path, "详情展示所有详细信息。")
    text += '\n列表和详情都不返回 `cover`。'
    assert any("明确排除了原始字段：cover" in issue for issue in _plan_quality_issues(graph, state, text))
