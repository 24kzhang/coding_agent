"""文本与原生工具请求的供应商参数边界。"""

import httpx
import pytest

from api.schema import AgentProfileMap, ModelConfig
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm.client import LlmClient
from llm.store import ModelStore


@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("model,capabilities,effort,expected", [
    ("generic", [], "high", None),
    ("deepseek-chat", [], "none", {"type": "disabled"}),
    ("deepseek-chat", [], "medium", {"type": "enabled"}),
    ("custom", ["reasoning"], "low", None),
])
def test_chat_reasoning_fields_follow_model_capabilities(monkeypatch, tools, model, capabilities, effort, expected):
    captured = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, _url, **kwargs):
            captured.append(kwargs["json"])
            return httpx.Response(200, json={"choices": [{"message": {"content": "完成", "tool_calls": []}}]})

    monkeypatch.setattr("llm.client.httpx.Client", Client)
    cfg = ModelConfig(id="test", name="测试", base_url="https://example.test/v1", api_key="假密钥", model=model,
                      capabilities=capabilities, protocol="chat_completions", efforts=["none", "low", "medium", "high"])
    client = LlmClient(cfg)
    if tools:
        client.chat_tools([], [], effort=effort)
    else:
        client.chat([], effort=effort)
    assert len(captured) == 1
    assert captured[0].get("thinking") == expected
    assert captured[0]["reasoning_effort"] == ("high" if model.startswith("deepseek") and effort == "medium" else effort)


@pytest.mark.parametrize("override,profile,expected", [(None, "inherit", "none"), ("high", "inherit", "none"), (None, "low", "low"), ("low", "inherit", "low")])
def test_selected_model_effort_does_not_require_primary_configuration(tmp_path, override, profile, expected):
    store = ModelStore(tmp_path / "models.json")
    profiles = AgentProfileMap()
    profiles.coder.effort = profile
    store.set_profiles(profiles)
    selected = ModelConfig(id="fallback", name="备用模型", base_url="https://example.test", api_key="假密钥", model="generic",
                           default_effort="none", efforts=["none", "low"])
    assert store.effort_for_agent("coder", override, model=selected) == expected


def test_explicit_model_inherits_its_own_default_instead_of_primary(tmp_path):
    store = ModelStore(tmp_path / "models.json")
    for name, effort in [("primary", "high"), ("selected", "none")]:
        store.upsert(ModelConfig(id=name, name="测试模型", base_url="https://example.test", api_key="假密钥", model="generic",
                                 default_effort=effort, efforts=["none", "high"]))
    graph = AgentGraph(store, MemoryStore(tmp_path / "mem"))
    assert graph._client("coder", {"model_id": "selected"}).default_effort == "none"
