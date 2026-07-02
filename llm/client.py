from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from api.schema import ModelConfig


class LlmError(RuntimeError):
    """模型调用失败时抛出的统一异常。"""


@dataclass
class LlmUsage:
    """统一记录 token 用量；部分兼容接口不返回 usage 时使用估算值。"""

    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


class LlmClient:
    """OpenAI-compatible 模型客户端。

    用户可能提供 `.../openai`、`.../v1` 或完整兼容地址，因此这里会按顺序尝试
    常见的 chat completions 路径。这样不会把某个供应商的路径细节写死。
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.last_usage = LlmUsage()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """没有分词器时的保守估算：中文约 1.6 字符/token，英文约 4 字符/token。"""
        if not text:
            return 0
        chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other = max(len(text) - chinese, 0)
        return int(chinese / 1.6 + other / 4) + 1

    def _urls(self) -> list[str]:
        base = self.cfg.base_url.rstrip("/")
        urls: list[str] = []
        if base.endswith("/chat/completions"):
            urls.append(base)
        if base.endswith("/v1"):
            urls.append(f"{base}/chat/completions")
        else:
            urls.append(f"{base}/v1/chat/completions")
            urls.append(f"{base}/chat/completions")
        # 去重但保留顺序，便于错误重试。
        return list(dict.fromkeys(urls))

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        last_error = ""
        for url in self._urls():
            try:
                with httpx.Client(timeout=self.cfg.timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code in {404, 405}:
                    last_error = f"{url} 返回 {resp.status_code}"
                    continue
                if resp.status_code >= 400:
                    raise LlmError(f"模型接口错误 {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                self.last_usage = self._usage(data, messages)
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = f"{url} 调用失败：{exc}"
        raise LlmError(last_error or "没有可用的模型接口地址")

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.1) -> dict[str, Any]:
        """要求模型返回 JSON，并对常见代码块包裹做容错。"""
        text = self.chat(messages, temperature=temperature).strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LlmError(f"模型没有返回合法 JSON：{exc}\n{text[:1000]}") from exc

    def test(self) -> dict[str, Any]:
        """用于前端设置页的连通性测试。"""
        text = self.chat(
            [
                {"role": "system", "content": "你只回答中文。"},
                {"role": "user", "content": "请只回复：连通成功"},
            ],
            temperature=0,
        )
        return {"ok": "连通" in text or "成功" in text, "reply": text, "tokens": self.last_usage.total}

    def _usage(self, data: dict[str, Any], messages: list[dict[str, str]]) -> LlmUsage:
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        if prompt or completion:
            return LlmUsage(prompt=prompt, completion=completion)
        prompt_text = "\n".join(msg.get("content", "") for msg in messages)
        completion_text = json.dumps(data.get("choices", []), ensure_ascii=False)
        return LlmUsage(
            prompt=self.estimate_tokens(prompt_text),
            completion=self.estimate_tokens(completion_text),
        )
