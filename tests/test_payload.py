"""文本与原生工具请求的供应商参数边界。"""

import httpx
import pytest

from api.schema import ModelConfig
from llm.client import LlmClient


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
