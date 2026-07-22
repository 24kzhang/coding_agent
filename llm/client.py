from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from html import unescape
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

    # working_urls 缓存每个模型服务已经成功响应过的接口地址，避免后续请求反复探测路径。
    working_urls: dict[str, str] = {}

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
        # candidates 去重但保留顺序，便于首次调用探测不同供应商的路径规则。
        candidates = list(dict.fromkeys(urls))
        # working_url 是此前已经成功调用过的地址；优先复用可显著减少无效路径探测。
        working_url = self.working_urls.get(base)
        if working_url in candidates:
            candidates.remove(working_url)
            candidates.insert(0, working_url)
        return candidates

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """调用模型并返回文本内容。"""

        # payload 是 OpenAI-compatible chat completions 请求体。
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        # max_tokens 是调用方按智能体职责设置的输出硬上限。Coding 工具参数需要比分类回答
        # 更大，但仍不能允许模型无限生成完整文件并占住会话数分钟。
        if max_tokens is not None:
            payload["max_tokens"] = max(1, int(max_tokens))
        # response_format 请求兼容接口强制返回 JSON；普通对话不携带该字段。
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # headers 放鉴权信息和 JSON 类型声明。
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        # last_error 保存最后一次失败原因，所有候选 URL 都失败后抛给上层。
        last_error = ""
        # json_mode_rejected 记录供应商是否明确拒绝结构化输出参数，供 chat_json 降级重试。
        json_mode_rejected = False
        # LongCat 长工具响应可能持续数分钟。使用 SSE 边生成边读取，读取超时按“相邻分片间隔”
        # 计算，不会因为等待完整大响应而误判超时；其他兼容模型保留普通 JSON 请求。
        use_stream = "longcat" in self.cfg.model.lower()
        if use_stream:
            payload["stream"] = True
        # 一个 client 复用连接池；候选 URL 和短重试不需要重复建立 TLS 连接。
        with httpx.Client(timeout=self.cfg.timeout) as client:
            # 按 _urls() 给出的候选地址依次尝试，兼容不同供应商 URL 规则。
            for url in self._urls():
                for attempt in range(2):
                    try:
                        if use_stream:
                            # stream() 保持连接打开到所有 SSE 分片读取结束。
                            with client.stream("POST", url, headers=headers, json=payload) as resp:
                                status = resp.status_code
                                if status in {404, 405}:
                                    last_error = f"{url} 返回 {status}"
                                    break
                                if status in {429, 500, 502, 503, 504} and attempt == 0:
                                    self._sleep_retry(resp.headers.get("Retry-After"))
                                    continue
                                if status == 400 and json_mode:
                                    json_mode_rejected = True
                                    last_error = f"{url} 不支持 JSON 输出模式"
                                    break
                                if status >= 400:
                                    body = resp.read().decode("utf-8", errors="replace")
                                    raise LlmError(f"模型接口错误 {status}: {body[:500]}")
                                self.working_urls[self.cfg.base_url.rstrip("/")] = url
                                return self._stream_text(resp, messages)

                        # 非流式模型沿用标准 OpenAI-compatible JSON 响应。
                        resp = client.post(url, headers=headers, json=payload)
                        if resp.status_code in {404, 405}:
                            last_error = f"{url} 返回 {resp.status_code}"
                            break
                        if resp.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                            self._sleep_retry(resp.headers.get("Retry-After"))
                            continue
                        if resp.status_code == 400 and json_mode:
                            json_mode_rejected = True
                            last_error = f"{url} 不支持 JSON 输出模式"
                            break
                        if resp.status_code >= 400:
                            raise LlmError(f"模型接口错误 {resp.status_code}: {resp.text[:500]}")
                        self.working_urls[self.cfg.base_url.rstrip("/")] = url
                        data = resp.json()
                        self.last_usage = self._usage(data, messages)
                        content = data["choices"][0]["message"]["content"]
                        return self._content_text(content)
                    except LlmError:
                        raise
                    except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                        # 读取超时通常表示模型生成卡住，重复提交会浪费额度并把等待时间成倍放大。
                        raise LlmError(f"{url} 模型响应超时：{exc}") from exc
                    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        # 只有连接建立阶段的短暂失败值得原地址重试；响应阶段错误不应重放模型请求。
                        last_error = f"{url} 调用失败：{exc}"
                        if attempt == 0 and isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                            time.sleep(0.4)
                            continue
                        break
        if json_mode_rejected:
            raise LlmError("模型接口不支持 JSON 输出模式")
        raise LlmError(last_error or "没有可用的模型接口地址")

    @staticmethod
    def _sleep_retry(value: str | None) -> None:
        """按服务端 Retry-After 做一次有上限的短退避。"""

        try:
            retry_after = min(float(value or "0.6"), 3.0)
        except ValueError:
            retry_after = 0.6
        time.sleep(max(retry_after, 0.1))

    def _stream_text(self, response: httpx.Response, messages: list[dict[str, str]]) -> str:
        """读取 OpenAI-compatible SSE 分片并拼成普通文本。"""

        # 少数网关会忽略 stream=true 并返回普通 JSON，仍按标准响应兼容处理。
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            data = json.loads(response.read())
            self.last_usage = self._usage(data, messages)
            return self._content_text(data["choices"][0]["message"]["content"])

        parts: list[str] = []
        usage_data: dict[str, Any] = {}
        # 流式读取避免把正常长生成误判为单次读取超时，但仍需要总时长和正文上限，
        # 防止供应商持续发送心跳或模型失控生成导致会话永远不结束。
        deadline = time.monotonic() + max(self.cfg.timeout * 2, 300)
        text_size = 0
        for raw_line in response.iter_lines():
            if time.monotonic() > deadline:
                raise LlmError("模型流式响应超过总时长上限")
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            payload = line[5:].strip() if line.startswith("data:") else line
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                # SSE 允许注释和供应商扩展行；无法解析的非数据行不影响后续内容。
                continue
            if isinstance(chunk.get("usage"), dict):
                usage_data = chunk
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            source = choice.get("delta") or choice.get("message") or {}
            if isinstance(source, dict):
                content = source.get("content")
                if content is not None and content != "":
                    piece = self._content_text(content)
                    parts.append(piece)
                    text_size += len(piece)
                    if text_size > 100_000:
                        raise LlmError("模型流式响应正文超过 100000 字符上限")
        text = "".join(parts)
        if not text:
            raise LlmError("模型流式响应中没有可用的文本 content")
        usage_source = usage_data or {"choices": [{"message": {"content": text}}]}
        self.last_usage = self._usage(usage_source, messages)
        return text

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        *,
        plain_text: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """返回结构化对象，并兼容 JSON 代码块与 LongCat 原生工具标签。"""

        # chat_kwargs 只在调用方真的设置预算时携带 max_tokens，保持对旧测试桩和兼容客户端的调用合同。
        chat_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            chat_kwargs["max_tokens"] = max_tokens
        if plain_text:
            # plain_text 用于供应商在 response_format 下连续返回空对象时的主动降级。
            raw_text = self.chat(messages, **chat_kwargs).strip()
        else:
            # raw_text 优先请求供应商约束为 JSON；不支持 response_format 时兼容旧接口降级。
            try:
                raw_text = self.chat(messages, json_mode=True, **chat_kwargs).strip()
            except LlmError as exc:
                # 少数供应商接受 response_format，却把结果放到非标准字段并省略 content；
                # 这种情况与明确返回 400 一样，应自动退回普通文本 JSON 请求。
                error_text = str(exc)
                unsupported_markers = ("不支持 JSON 输出模式", "'content'", "没有可用的文本 content")
                if not any(marker in error_text for marker in unsupported_markers):
                    raise
                raw_text = self.chat(messages, **chat_kwargs).strip()
        # text 是去除 Markdown 代码围栏后的候选 JSON 文本。
        text = raw_text
        # 有些模型会返回 ```json 代码块，这里去掉代码围栏。
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        # LongCat 偶尔会忽略 JSON 要求，改用其原生工具标签；这里统一转换为现有 ReAct actions。
        longcat_result = self._parse_longcat_tool_calls(text)
        if longcat_result:
            return longcat_result
        # candidate 从解释文字中提取第一个括号平衡的 JSON 对象，不会被字符串内的花括号干扰。
        candidate = self._extract_json_object(text)
        if candidate:
            text = candidate
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # 抛出前带上截断后的模型输出，方便定位 prompt 或模型格式问题。
            raise LlmError(f"模型没有返回合法 JSON：{exc}\n{text[:1000]}") from exc

    def _parse_longcat_tool_calls(self, text: str) -> dict[str, Any] | None:
        """把 LongCat 原生工具标签转换为项目统一的 ReAct JSON。"""

        # call_pattern 同时支持一个响应包含多个工具调用；工具名位于开始标签后的首行。
        call_pattern = re.compile(
            r"<longcat_tool_call>\s*([^\r\n<]+)\s*(.*?)</longcat_tool_call>",
            re.DOTALL,
        )
        # arg_pattern 提取键值标签；值允许包含换行和普通代码字符。
        arg_pattern = re.compile(
            r"<longcat_arg_key>\s*(.*?)\s*</longcat_arg_key>\s*"
            r"<longcat_arg_value>\s*(.*?)\s*</longcat_arg_value>",
            re.DOTALL,
        )
        matches = list(call_pattern.finditer(text))
        if not matches:
            return None

        actions: list[dict[str, Any]] = []
        done = False
        summary = ""
        # integer_keys 是 Coding 工具协议中要求整数的参数。
        integer_keys = {"expected", "max_chars", "max_results", "start"}
        for match in matches:
            tool = unescape(match.group(1).strip())
            action: dict[str, Any] = {"tool": tool}
            for arg_match in arg_pattern.finditer(match.group(2)):
                key = unescape(arg_match.group(1).strip())
                value: Any = unescape(arg_match.group(2).strip())
                if key in integer_keys:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                action[key] = value
            # 某些 LongCat 响应会把完成状态也表示成工具调用。
            if tool in {"done", "finish"}:
                done = True
                summary = str(action.get("summary") or action.get("content") or "任务已完成。")
                continue
            actions.append(action)

        # thought 删除工具标签后只保留模型的自然语言判断；纯工具响应使用稳定说明。
        thought = call_pattern.sub("", text).strip() or "模型返回 LongCat 原生工具调用。"
        return {"thought": thought, "actions": actions, "done": done, "summary": summary}

    def _content_text(self, content: Any) -> str:
        """把兼容接口返回的 content 统一转换为纯文本。"""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # parts 只提取字符串项或带 text 字段的内容块。
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts)
        raise LlmError("模型响应中没有可用的文本 content")

    def _extract_json_object(self, text: str) -> str:
        """提取文本中第一个完整 JSON 对象，正确处理字符串和转义字符。"""

        # start 是当前候选对象起始位置；depth 是花括号嵌套深度。
        start = -1
        depth = 0
        # in_string 和 escaped 用于忽略 JSON 字符串内部的花括号及转义引号。
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if start < 0:
                if char == "{":
                    start = index
                    depth = 1
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return ""

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
