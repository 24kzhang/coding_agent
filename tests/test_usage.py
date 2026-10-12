"""成本诊断保留真实计费用量，区分正文、推理与失败响应。"""

import pytest

from api.schema import ModelConfig
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.client import LlmClient, LlmUsage
from llm.store import ModelStore


def test_usage_preserves_billed_tokens_and_failure_attribution(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    state = {"workdir": str(tmp_path / "work"), "session_id": "s1"}
    graph._add_tokens(state, LlmUsage(prompt=500, cached=400, completion=100, reasoning=80), agent="coder", model_id="selected", effort="low", failed=True)
    data = events[-1].data
    assert data["visible_output_tokens"] == 20
    assert data["output_tokens"] == 100 and data["reasoning_tokens"] == 80
    assert data["source_agent"] == "coder" and data["model_id"] == "selected" and data["failed_response"]
    assert data["effort"] == "low"
    assert state["output_tokens"] == 100 and state["tokens"] == 600
    restored = MemoryStore(tmp_path / "mem").list_history()[0]["sessions"][0]["events"][-1]
    assert restored["data"] == data


@pytest.mark.parametrize("responses", [False, True])
def test_missing_reasoning_details_are_not_supplier_zero(tmp_path, responses):
    client = LlmClient(ModelConfig(id="test", name="测试", model="generic", base_url="https://example.test", api_key="TEST_ONLY_KEY"))
    key = "output_tokens_details" if responses else "completion_tokens_details"
    body = {"usage": {("input_tokens" if responses else "prompt_tokens"): 20, ("output_tokens" if responses else "completion_tokens"): 5}}
    parse = client._responses_usage if responses else client._usage
    missing = parse(body, [])
    assert missing.reasoning_known is False
    body["usage"][key] = {"reasoning_tokens": 0}
    assert parse(body, []).reasoning_known is True
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    graph._add_tokens({"workdir": str(tmp_path / "work"), "session_id": "s1"}, missing)
    assert events[-1].data["reasoning_tokens"] is None
    assert events[-1].data["visible_output_tokens"] is None
    assert events[-1].data["output_tokens"] == 5 and events[-1].tokens == 25


def test_usage_positional_estimate_flag_keeps_previous_contract():
    usage = LlmUsage(20, 5, 0, 20, 0, True)
    assert usage.estimated is True and usage.reasoning_known is False
