"""只读结构化节点使用原生函数参数，减少付费格式修订。"""

import json

import httpx
import pytest

from api.schema import ModelConfig
from llm.client import LlmClient, LlmError


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_json_function_returns_nested_quotes_in_one_request(monkeypatch, protocol):
    result = {"ok": False, "issues": ['接口返回 {"detail":"错误"}，正则为 \\d+'], "actions": []}
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
            args = json.dumps({"result": result}, ensure_ascii=False)
            return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"function": {"name": "return_json", "arguments": args}}]}}],
                                           "output": [{"type": "function_call", "name": "return_json", "arguments": args}],
                                           "usage": {"prompt_tokens": 30, "completion_tokens": 20, "input_tokens": 30, "output_tokens": 20}})

    monkeypatch.setattr("llm.client.httpx.Client", Client)
    cfg = ModelConfig(id="test", name="测试", base_url="https://example.test/v1", api_key="假密钥", model="generic", protocol=protocol)
    client = LlmClient(cfg)
    assert client.chat_json([], structured_tools=True, max_tokens=100) == result
    assert len(captured) == 1 and client.last_usage.completion == 20
    assert captured[0]["temperature"] == 0.1
    assert captured[0]["tool_choice"] == ({"type": "function", "name": "return_json"} if protocol == "responses" else {"type": "function", "function": {"name": "return_json"}})


def test_json_function_cannot_execute_an_unexpected_tool():
    cfg = ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic")
    client = LlmClient(cfg)
    client.chat_tools = lambda *_args, **_kwargs: {"actions": [{"tool": "write_file", "path": "main.py", "content": "错误"}]}
    with pytest.raises(LlmError, match="未执行任何动作"):
        client.chat_json([], structured_tools=True)


def test_tool_arguments_accept_decoded_objects_and_never_repair_truncated_actions():
    result = {"path": "main.py", "content": "中文源码"}
    assert LlmClient._tool_arguments(result) == result
    assert LlmClient._tool_arguments(json.dumps(result)) == result
    for value in [None, [], '[]', '{"path":"main.py"', '{"secret":"TEST_ONLY_PRIVATE"坏}']:
        with pytest.raises(LlmError) as captured:
            LlmClient._tool_arguments(value)
        assert "TEST_ONLY_PRIVATE" not in str(captured.value)
