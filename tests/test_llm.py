from __future__ import annotations

from api.schema import ModelConfig
from llm.client import LlmClient


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


def test_llm_token_estimate_handles_chinese() -> None:
    assert LlmClient.estimate_tokens("你好，世界") >= 3
