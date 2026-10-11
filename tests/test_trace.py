"""原生工具往返保留调用身份、真实结果与取消结果，裁剪不拆散调用组。"""

import json

import httpx

from api.schema import ModelConfig
from backend.agents.coder import coder
from backend.agents.graph import AgentGraph
from backend.agents.trace import ToolTrace
from backend.memory import MemoryStore
from llm.client import LlmClient
from llm.store import ModelStore


def message(*names):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": "{}"}}
        for index, name in enumerate(names)
    ]}


def test_cancelled_batch_has_one_result_per_original_call_and_group_eviction():
    trace = ToolTrace(limit=1)
    trace.start(message("write_file", "run_command", "finish"))
    trace.record(0, {"ok": True, "text": "写入完成"})
    trace.complete(["命令参数需修订"])
    result = trace.messages()
    assert len(result) == 4
    assert json.loads(result[1]["content"])["ok"] is True
    assert json.loads(result[2]["content"])["executed"] is False
    assert [entry["tool_call_id"] for entry in result[1:]] == ["call-0", "call-1", "call-2"]
    trace.start(message("read_file"))
    trace.record(0, {"ok": False})
    trace.complete([])
    assert len(trace.messages()) == 2
    assert trace.messages()[0]["tool_calls"][0]["function"]["name"] == "read_file"


def test_coder_replays_real_native_call_and_tool_result_before_finish(tmp_path, monkeypatch):
    requests = []
    class Transport:
        def __init__(self, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def post(self, _url, **kwargs):
            payload = kwargs["json"]
            requests.append(payload)
            if len(requests) == 1:
                msg = message("write_file")
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({"path": "main.py", "content": "ready = True\n"})
            else:
                assert [item["role"] for item in payload["messages"]] == ["system", "user", "assistant", "tool", "user"]
                assistant, tool = payload["messages"][2:4]
                assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == "call-0"
                assert json.loads(tool["content"])["ok"] is True
                assert json.loads(assistant["tool_calls"][0]["function"]["arguments"])["path"] == "main.py"
                msg = message("finish")
                msg["tool_calls"][0]["function"]["arguments"] = '{"summary":"已完成"}'
            return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {"prompt_tokens": 40, "completion_tokens": 10}})
    monkeypatch.setattr("llm.client.httpx.Client", Transport)
    client = LlmClient(ModelConfig(id="test", name="测试", model="generic", base_url="https://example.test", api_key="TEST_ONLY_KEY", protocol="chat_completions"))
    graph = AgentGraph(ModelStore(tmp_path / "models.json"), MemoryStore(tmp_path / "mem"))
    graph._client = lambda *_args: client
    state = coder(graph, {"workdir": str(tmp_path / "work"), "session_id": "s1", "text": "创建模块"})
    assert state["coding_ok"] and (tmp_path / "work/main.py").read_text() == "ready = True\n"
    assert len(requests) == 2
