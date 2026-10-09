"""成本诊断保留真实计费用量，区分正文、推理与失败响应。"""

from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.client import LlmUsage
from llm.store import ModelStore


def test_usage_preserves_billed_tokens_and_failure_attribution(tmp_path):
    events = []
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"), emit=events.append)
    state = {"workdir": str(tmp_path / "work"), "session_id": "s1"}
    graph._add_tokens(state, LlmUsage(prompt=500, cached=400, completion=100, reasoning=80), agent="coder", model_id="selected", failed=True)
    data = events[-1].data
    assert data["visible_output_tokens"] == 20
    assert data["output_tokens"] == 100 and data["reasoning_tokens"] == 80
    assert data["source_agent"] == "coder" and data["model_id"] == "selected" and data["failed_response"]
    assert state["output_tokens"] == 100 and state["tokens"] == 600
    restored = MemoryStore(tmp_path / "mem").list_history()[0]["sessions"][0]["events"][-1]
    assert restored["data"] == data
