"""列表投影不能被误判为完整详情删字段，用户要求完整列表时仍须保护。"""
import json

from backend.agents.graph import AgentGraph
from backend.agents.planner import _mentions_field, _plan_quality_issues
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


def test_list_and_detail_sentences_on_same_line_have_distinct_scopes(tmp_path):
    graph, state, text = fixture(tmp_path, "详情展示所有详细信息。")
    text += '\n列表不返回 `cover` 加载地址。详情提供全部原始字段，`cover` 字段禁止删除。'
    assert _plan_quality_issues(graph, state, text) == []


def test_resource_ban_does_not_mean_symbolic_field_deletion(tmp_path):
    graph, state, text = fixture(tmp_path, "详情展示所有详细信息。")
    text += '\n`cover` 是原始字段，禁止加载其资源 ≠ 删除该字段：详情接口必须原样返回 `cover` 的原始路径文本。'
    assert _plan_quality_issues(graph, state, text) == []


def test_plain_and_grouped_identifiers_are_real_field_mentions():
    assert _mentions_field('字段：`record_id/cover/details.zh`。', 'cover')
    assert _mentions_field('字段record_id是编号。', 'record_id')
    assert not _mentions_field('record_id_backup 或 xrecord_id 不是该字段。', 'record_id')
