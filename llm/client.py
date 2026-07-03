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

    # 输入侧 token 数，对应 OpenAI usage.prompt_tokens。
    prompt: int = 0
    # 输出侧 token 数，对应 OpenAI usage.completion_tokens。
    completion: int = 0

    @property
    def total(self) -> int:
        """返回本次模型调用总 token 数。"""

        return self.prompt + self.completion


class LlmClient:
    """OpenAI-compatible 模型客户端。

    用户可能提供 `.../openai`、`.../v1` 或完整兼容地址，因此这里会按顺序尝试
    常见的 chat completions 路径。这样不会把某个供应商的路径细节写死。
    """

    def __init__(self, cfg: ModelConfig):
        # cfg 是当前要调用的模型配置，来自 ModelStore。
        self.cfg = cfg
        # last_usage 保存最近一次模型调用的 token 用量，AgentGraph 会把它累加到事件流。
        self.last_usage = LlmUsage()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """没有分词器时的保守估算：中文约 1.6 字符/token，英文约 4 字符/token。"""
        if not text:
            return 0
        # chinese 统计中文字符数，中文通常比英文字符/token 比例更高。
        chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        # other 统计非中文字符数，按英文和符号的大致比例估算。
        other = max(len(text) - chinese, 0)
        return int(chinese / 1.6 + other / 4) + 1

    def _urls(self) -> list[str]:
        """根据用户填写的 base_url 生成候选 chat completions 地址。"""

        # base 去掉末尾斜杠，避免拼接路径时出现双斜杠。
        base = self.cfg.base_url.rstrip("/")
        # urls 按优先级保存候选地址，chat() 会依次尝试。
        urls: list[str] = []
        # 用户如果已经填到完整 chat completions 地址，就直接使用。
        if base.endswith("/chat/completions"):
            urls.append(base)
        # 用户如果填到 /v1，就只需要补 /chat/completions。
        if base.endswith("/v1"):
            urls.append(f"{base}/chat/completions")
        else:
            # 常见 OpenAI-compatible 服务可能需要 /v1/chat/completions。
            urls.append(f"{base}/v1/chat/completions")
            # 也有供应商直接在 base_url 后接 /chat/completions。
            urls.append(f"{base}/chat/completions")
        # 去重但保留顺序，便于错误重试。
        return list(dict.fromkeys(urls))

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """调用模型并返回文本内容。"""

        # payload 是 OpenAI-compatible chat completions 请求体。
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        # headers 放鉴权信息和 JSON 类型声明。
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        # last_error 保存最后一次失败原因，所有候选 URL 都失败后抛给上层。
        last_error = ""
        # 按 _urls() 给出的候选地址依次尝试，兼容不同供应商 URL 规则。
        for url in self._urls():
            try:
                # client 是本次 HTTP 请求客户端，timeout 来自模型配置。
                with httpx.Client(timeout=self.cfg.timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                # 404/405 可能只是路径不匹配，继续尝试下一个候选地址。
                if resp.status_code in {404, 405}:
                    last_error = f"{url} 返回 {resp.status_code}"
                    continue
                # 其他 4xx/5xx 通常表示鉴权、模型名或服务异常，直接抛出。
                if resp.status_code >= 400:
                    raise LlmError(f"模型接口错误 {resp.status_code}: {resp.text[:500]}")
                # data 是供应商返回的 JSON 响应。
                data = resp.json()
                # 记录 usage，若供应商不返回 usage，则 _usage() 会做估算。
                self.last_usage = self._usage(data, messages)
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                # 单个候选地址失败时保存错误，继续尝试其他地址。
                last_error = f"{url} 调用失败：{exc}"
        raise LlmError(last_error or "没有可用的模型接口地址")

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.1) -> dict[str, Any]:
        """要求模型返回 JSON，并对常见代码块包裹做容错。"""
        # text 是模型原始文本输出，后续会尽量清洗成 JSON 字符串。
        text = self.chat(messages, temperature=temperature).strip()
        # 有些模型会返回 ```json 代码块，这里去掉代码围栏。
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        # start/end 用于从解释性文本里截取最外层 JSON 对象。
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # 抛出前带上截断后的模型输出，方便定位 prompt 或模型格式问题。
            raise LlmError(f"模型没有返回合法 JSON：{exc}\n{text[:1000]}") from exc

    def test(self) -> dict[str, Any]:
        """用于前端设置页的连通性测试。"""
        # text 是模型连通性探测结果，只要求模型回复“连通成功”附近的中文。
        text = self.chat(
            [
                {"role": "system", "content": "你只回答中文。"},
                {"role": "user", "content": "请只回复：连通成功"},
            ],
            temperature=0,
        )
        # ok 用宽松判断，兼容模型回复“连接成功”“连通成功”等中文变体。
        return {"ok": "连通" in text or "成功" in text, "reply": text, "tokens": self.last_usage.total}

    def _usage(self, data: dict[str, Any], messages: list[dict[str, str]]) -> LlmUsage:
        """从模型响应中读取 usage；没有 usage 时用文本长度估算。"""

        # usage 是 OpenAI-compatible 响应里的标准 token 用量字段。
        usage = data.get("usage") or {}
        # prompt 是输入 token 数。
        prompt = int(usage.get("prompt_tokens") or 0)
        # completion 是输出 token 数。
        completion = int(usage.get("completion_tokens") or 0)
        if prompt or completion:
            return LlmUsage(prompt=prompt, completion=completion)
        # prompt_text 是所有输入消息拼接后的文本，用于估算输入 token。
        prompt_text = "\n".join(msg.get("content", "") for msg in messages)
        # completion_text 是 choices 的 JSON 字符串，用于估算输出 token。
        completion_text = json.dumps(data.get("choices", []), ensure_ascii=False)
        return LlmUsage(
            prompt=self.estimate_tokens(prompt_text),
            completion=self.estimate_tokens(completion_text),
        )
