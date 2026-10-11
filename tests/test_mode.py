"""原生执行与 JSON 兼容输出只接收各自一致的协议。"""

import httpx
import pytest

from api.schema import ModelConfig
from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.agents.prompts import NATIVE_CODER_PROMPT
from backend.memory import MemoryStore
from llm.client import LlmClient, LlmUsage
from llm.store import ModelStore


def test_native_coder_receives_no_react_json_example_and_requires_call(tmp_path):
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    class Client(LlmClient):
        def __init__(self):
            self.cfg = ModelConfig(id="test", name="测试", model="generic", base_url="https://example.test", api_key="TEST_ONLY_KEY")
            self.last_usage = LlmUsage()
        def chat_tools(self, messages, _tools, **kwargs):
            assert kwargs["require_call"] is True
            assert messages[0]["content"].startswith(NATIVE_CODER_PROMPT)
            assert '"actions": [' not in messages[0]["content"]
            assert "actions=[]、done=true" not in str(messages)
            return {"actions": [{"tool": "write_file", "path": "main.py", "content": "ready = True"}], "done": True}
    graph._client = lambda *_args: Client()
    result = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建入口"})
    assert result["coding_ok"] and (tmp_path / "work/main.py").exists()


@pytest.mark.parametrize("protocol", ["responses", "chat_completions"])
def test_required_tool_choice_is_sent_without_changing_optional_default(monkeypatch, protocol):
    requests = []
    class Client:
        def __init__(self, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def post(self, _url, **kwargs):
            requests.append(kwargs["json"])
            return httpx.Response(200, json={"choices": [{"message": {"content": "完成", "tool_calls": []}}], "output": [], "output_text": "完成"})
    monkeypatch.setattr("llm.client.httpx.Client", Client)
    model = LlmClient(ModelConfig(id="test", name="测试", model="generic", base_url="https://example.test", api_key="TEST_ONLY_KEY", protocol=protocol))
    model.chat_tools([], [], require_call=True)
    model.chat_tools([], [])
    assert [request["tool_choice"] for request in requests] == ["required", "auto"]
