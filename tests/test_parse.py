"""结构化响应解析的通用边界回归。"""

import pytest

from api.schema import ModelConfig
from llm.client import LlmClient, LlmError


def client():
    return LlmClient(ModelConfig(id="test", name="测试", base_url="https://example.test", api_key="假密钥", model="generic"))


def test_json_skips_code_braces_and_preserves_nested_strings():
    model = client()
    response = '先检查路径 {id} 和样式 { color: red; }。\n{"items":[{"name":"括号 { 和 \\\"引号\\\" }"}],"ok":true}'
    model.chat = lambda *_args, **_kwargs: response
    assert model.chat_json([]) == {"items": [{"name": '括号 { 和 "引号" }'}], "ok": True}


@pytest.mark.parametrize("response", [
    '{"ok":false}\n{"ok":true}',
    '{"actions":[{"tool":"write_file","path":"main.py","content":"部分结果"}]',
    '{"ok":true}\n{"unfinished":',
])
def test_ambiguous_or_truncated_objects_never_become_success(response):
    model = client()
    model.chat = lambda *_args, **_kwargs: response
    with pytest.raises(LlmError, match="合法 JSON"):
        model.chat_json([])
