from __future__ import annotations

import json

import httpx
import pytest

from api.schema import ModelConfig
from llm.client import LlmBalanceError, LlmClient, LlmError
from llm.store import ModelStore


def test_model_store_fallbacks_skip_disabled_and_keep_config_order(tmp_path) -> None:
    """故障转移只能选择启用模型，并保留 models.json 中的优先级顺序。"""

    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "longcat",
                        "name": "LongCat",
                        "base_url": "https://example.com",
                        "api_key": "k",
                        "model": "LongCat-2.0",
                        "enabled": True,
                    },
                    {
                        "id": "qwen",
                        "name": "Qwen",
                        "base_url": "https://example.com",
                        "api_key": "k",
                        "model": "qwen",
                        "enabled": True,
                    },
                    {
                        "id": "disabled",
                        "name": "Disabled",
                        "base_url": "https://example.com",
                        "api_key": "k",
                        "model": "disabled",
                        "enabled": False,
                    },
                ],
                "agent_models": {},
            }
        ),
        encoding="utf-8",
    )

    fallbacks = ModelStore(path).fallbacks("missing")

    assert [model.id for model in fallbacks] == ["longcat", "qwen"]


def test_model_store_public_view_and_private_copy(tmp_path) -> None:
    """浏览器接口不能拿到密钥，受控复制必须设置权限并加入忽略。"""

    config = tmp_path / "models.json"
    store = ModelStore(config)
    store.upsert(
        ModelConfig(
            id="DeepSeek-V4.1-Flash",
            name="DeepSeek",
            base_url="https://api.deepseek.com",
            api_key="sk-private-value",
            model="deepseek-flash",
            protocol="responses",
        )
    )
    project = tmp_path / "project"
    project.mkdir()
    destination = store.copy_private("DeepSeek-V4.1-Flash", str(project))
    public = store.public()[0].model_dump()

    assert "api_key" not in public
    assert public["has_key"] is True
    assert destination.stat().st_mode & 0o777 == 0o600
    assert destination.parent.stat().st_mode & 0o777 == 0o700
    assert ".local/" in (project / ".gitignore").read_text(encoding="utf-8")
    assert json.loads(destination.read_text(encoding="utf-8"))["api_key"] == "sk-private-value"


def test_llm_url_candidates_for_plain_openai_base() -> None:
    cfg = ModelConfig(
        id="x",
        name="测试",
        base_url="https://api.example.com/openai",
        api_key="k",
        model="m",
    )
    client = LlmClient(cfg)

    assert client._urls() == [
        "https://api.example.com/openai/v1/chat/completions",
        "https://api.example.com/openai/chat/completions",
    ]


def test_llm_prefers_previously_working_url() -> None:
    """同一服务已验证过的接口地址应排在候选列表首位。"""

    cfg = ModelConfig(
        id="cached",
        name="测试",
        base_url="https://cached.example.com/openai",
        api_key="k",
        model="m",
    )
    LlmClient.working_urls[cfg.base_url] = "https://cached.example.com/openai/chat/completions"

    assert LlmClient(cfg)._urls()[0] == "https://cached.example.com/openai/chat/completions"


def test_llm_read_timeout_is_not_retried(monkeypatch) -> None:
    """模型读取超时应立即失败，不能重复提交请求并放大等待时间和额度消耗。"""

    calls: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, **kwargs):
            calls.append(url)
            request = httpx.Request("POST", url)
            raise httpx.ReadTimeout("读取超时", request=request)

    cfg = ModelConfig(
        id="timeout",
        name="测试",
        base_url="https://timeout.example.com/openai",
        api_key="k",
        model="m",
    )
    monkeypatch.setattr("llm.client.httpx.Client", FakeClient)

    try:
        LlmClient(cfg).chat([{"role": "user", "content": "测试"}])
    except LlmError as exc:
        assert "模型响应超时" in str(exc)
    else:
        raise AssertionError("读取超时必须抛出 LlmError")

    assert calls == ["https://timeout.example.com/openai/v1/chat/completions"]


def test_longcat_chat_streams_and_joins_sse_content(monkeypatch) -> None:
    """LongCat 长响应应使用 SSE 持续读取，并忽略空 choices 控制块。"""

    requests: list[dict] = []

    class StreamContext:
        def __init__(self, response):
            self.response = response

        def __enter__(self):
            return self.response

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def stream(self, method, url, **kwargs):
            requests.append({"method": method, "url": url, "json": kwargs["json"]})
            body = "\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"<longcat_tool_"}}]}',
                    'data: {"choices":[]}',
                    'data: {"choices":[{"delta":{"content":"call>finish</longcat_tool_call>"}}]}',
                    "data: [DONE]",
                ]
            )
            response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())
            return StreamContext(response)

    cfg = ModelConfig(
        id="longcat-stream",
        name="LongCat",
        base_url="https://stream.example.com/openai",
        api_key="k",
        model="LongCat-2.0",
    )
    monkeypatch.setattr("llm.client.httpx.Client", FakeClient)

    text = LlmClient(cfg).chat([{"role": "user", "content": "完成任务"}], max_tokens=6000)

    assert text == "<longcat_tool_call>finish</longcat_tool_call>"
    assert requests[0]["json"]["stream"] is True
    assert requests[0]["json"]["max_tokens"] == 6000
    assert requests[0]["url"].endswith("/v1/chat/completions")


def test_llm_token_estimate_handles_chinese() -> None:
    assert LlmClient.estimate_tokens("你好，世界") >= 3


def test_responses_api_sends_effort_image_and_extracts_usage(monkeypatch, tmp_path) -> None:
    """Responses 适配器应映射思考等级与图片，并只返回最终文本。"""

    image = tmp_path / "tiny.png"
    image.write_bytes(
        __import__("base64").b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nS4AAAAASUVORK5CYII="
        )
    )
    captured: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, **kwargs):
            captured.update({"url": url, "json": kwargs["json"]})
            return httpx.Response(
                200,
                json={
                    "output_text": "看到了",
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
            )

    cfg = ModelConfig(
        id="deepseek",
        name="视觉模型",
        base_url="https://api.deepseek.com",
        api_key="k",
        model="deepseek-flash",
        protocol="responses",
    )
    monkeypatch.setattr("llm.client.httpx.Client", FakeClient)
    client = LlmClient(cfg)
    text = client.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "看图"},
                    {"type": "input_image", "path": str(image), "mime": "image/png"},
                ],
            }
        ],
        effort="max",
    )

    assert text == "看到了"
    assert captured["url"].endswith("/v1/responses")
    assert captured["json"]["reasoning"] == {"effort": "max"}
    assert captured["json"]["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert client.last_usage.total == 15


def test_native_responses_tools_are_converted_to_actions(monkeypatch) -> None:
    """原生 function_call 必须转换为内部动作，执行策略字段不得发送给供应商。"""

    captured: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, **kwargs):
            captured.update(kwargs["json"])
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "function_call",
                            "name": "write_file",
                            "call_id": "c1",
                            "arguments": '{"path":"app.py","content":"print(1)\\n"}',
                        }
                    ],
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            )

    cfg = ModelConfig(
        id="deepseek",
        name="工具模型",
        base_url="https://api.deepseek.com",
        api_key="k",
        model="deepseek-flash",
        protocol="responses",
    )
    monkeypatch.setattr("llm.client.httpx.Client", FakeClient)
    result = LlmClient(cfg).chat_tools(
        [{"role": "user", "content": "创建文件"}],
        [
            {
                "name": "write_file",
                "description": "写文件",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                "risk": "medium",
                "permission": "allow",
            }
        ],
    )

    assert result["actions"] == [{"tool": "write_file", "path": "app.py", "content": "print(1)\n"}]
    assert "risk" not in captured["tools"][0]
    assert "permission" not in captured["tools"][0]


def test_balance_error_stops_and_redacts_key(monkeypatch) -> None:
    """明确余额不足时应抛出专用错误，错误中不能出现密钥。"""

    secret = "sk-super-secret-key"

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, **kwargs):
            return httpx.Response(402, text=f"余额不足，key={secret}")

    cfg = ModelConfig(
        id="deepseek",
        name="测试",
        base_url="https://api.deepseek.com",
        api_key=secret,
        model="deepseek-flash",
        protocol="responses",
    )
    monkeypatch.setattr("llm.client.httpx.Client", FakeClient)

    with pytest.raises(LlmBalanceError) as caught:
        LlmClient(cfg).chat([{"role": "user", "content": "测试"}])

    assert secret not in str(caught.value)


def test_llm_extracts_balanced_json_with_braces_inside_string() -> None:
    cfg = ModelConfig(id="x", name="测试", base_url="https://api.example.com", api_key="k", model="m")
    client = LlmClient(cfg)

    text = '说明文字 {"thought":"保留 { 花括号 }", "done":true} 后续文字 {"ignored":true}'

    assert client._extract_json_object(text) == '{"thought":"保留 { 花括号 }", "done":true}'


def test_chat_json_falls_back_when_json_mode_has_no_content() -> None:
    """供应商接受 response_format 但省略 content 时应退回普通 JSON 请求。"""

    cfg = ModelConfig(id="x", name="测试", base_url="https://api.example.com", api_key="k", model="m")
    client = LlmClient(cfg)
    modes: list[bool] = []

    def fake_chat(messages, temperature=0.2, *, json_mode=False):
        modes.append(json_mode)
        if json_mode:
            raise LlmError("模型接口调用失败：'content'")
        return '{"ok": true}'

    client.chat = fake_chat

    assert client.chat_json([{"role": "user", "content": "返回 JSON"}]) == {"ok": True}
    assert modes == [True, False]


def test_chat_json_parses_longcat_native_tool_calls() -> None:
    """LongCat 原生工具标签应转换为统一 actions，避免被当作无效 JSON。"""

    cfg = ModelConfig(id="x", name="测试", base_url="https://api.example.com", api_key="k", model="LongCat-2.0")
    client = LlmClient(cfg)
    client.chat = lambda *args, **kwargs: """先读取两个文件。
<longcat_tool_call>read_file
<longcat_arg_key>path</longcat_arg_key>
<longcat_arg_value>main.py</longcat_arg_value>
<longcat_arg_key>start</longcat_arg_key>
<longcat_arg_value>0</longcat_arg_value>
<longcat_arg_key>max_chars</longcat_arg_key>
<longcat_arg_value>24000</longcat_arg_value>
</longcat_tool_call>
<longcat_tool_call>read_file
<longcat_arg_key>path</longcat_arg_key>
<longcat_arg_value>tests/test_main.py</longcat_arg_value>
</longcat_tool_call>"""

    result = client.chat_json([{"role": "user", "content": "读取文件"}])

    assert result["thought"] == "先读取两个文件。"
    assert result["done"] is False
    assert result["actions"] == [
        {"tool": "read_file", "path": "main.py", "start": 0, "max_chars": 24000},
        {"tool": "read_file", "path": "tests/test_main.py"},
    ]


def test_longcat_finish_call_maps_to_done() -> None:
    """LongCat 用 finish 标签结束任务时也应进入统一完成状态。"""

    cfg = ModelConfig(id="x", name="测试", base_url="https://api.example.com", api_key="k", model="LongCat-2.0")
    client = LlmClient(cfg)
    result = client._parse_longcat_tool_calls(
        """<longcat_tool_call>finish
<longcat_arg_key>summary</longcat_arg_key>
<longcat_arg_value>测试已通过</longcat_arg_value>
</longcat_tool_call>"""
    )

    assert result == {
        "thought": "模型返回 LongCat 原生工具调用。",
        "actions": [],
        "done": True,
        "summary": "测试已通过",
    }


def test_chat_json_plain_text_bypasses_response_format() -> None:
    """主动降级模式必须关闭 response_format，同时继续解析 LongCat 标签。"""

    cfg = ModelConfig(id="x", name="测试", base_url="https://api.example.com", api_key="k", model="LongCat-2.0")
    client = LlmClient(cfg)
    modes: list[bool] = []

    def fake_chat(messages, temperature=0.2, *, json_mode=False):
        modes.append(json_mode)
        return """<longcat_tool_call>read_file
<longcat_arg_key>path</longcat_arg_key>
<longcat_arg_value>app.py</longcat_arg_value>
</longcat_tool_call>"""

    client.chat = fake_chat
    result = client.chat_json([{"role": "user", "content": "读取"}], plain_text=True)

    assert modes == [False]
    assert result["actions"] == [{"tool": "read_file", "path": "app.py"}]
