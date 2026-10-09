from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

import httpx

from api.schema import ModelConfig


class LlmError(RuntimeError):
    """模型调用失败时抛出的统一异常。"""


class LlmBalanceError(LlmError):
    """账户余额或额度明确不足，调用方必须立即停止。"""


class LlmOutputLimitError(LlmError):
    """已付费输出达到上限，但没有产出完整结构化结果。"""


@dataclass
class LlmUsage:
    """统一记录 token 用量，并拆分缓存与推理开销。"""

    # 输入侧 token 数，对应 OpenAI usage.prompt_tokens。
    prompt: int = 0
    # 输出侧 token 数，对应 OpenAI usage.completion_tokens。
    completion: int = 0
    # cached 是输入中命中供应商前缀缓存的 token 数。
    cached: int = 0
    # cache_miss 是未命中缓存、需要重新计费或计算的输入 token 数。
    cache_miss: int = 0
    # reasoning 是 completion 中用于内部推理的 token 数，不与 completion 重复累加。
    reasoning: int = 0
    # 估算值与供应商返回值分开，缓存和成本结论不能建立在估算命中上。
    estimated: bool = False

    def __post_init__(self) -> None:
        self.prompt = max(0, self.prompt)
        self.completion = max(0, self.completion)
        self.cached = max(0, min(self.cached, self.prompt))
        self.cache_miss = min(self.prompt, max(self.cache_miss, self.prompt - self.cached))
        self.reasoning = max(0, min(self.reasoning, self.completion))

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
        # default_effort 可由 AgentGraph 按 Agent 配置覆盖。
        self.default_effort = cfg.default_effort

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

    def _response_urls(self) -> list[str]:
        """生成 Responses API 候选地址。"""

        base = self.cfg.base_url.rstrip("/")
        if base.endswith("/responses"):
            return [base]
        urls = [f"{base}/responses"] if base.endswith("/v1") else [
            f"{base}/v1/responses",
            f"{base}/responses",
        ]
        key = f"responses:{base}"
        working = self.working_urls.get(key)
        if working in urls:
            urls.remove(working)
            urls.insert(0, working)
        return list(dict.fromkeys(urls))

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> str:
        """按模型配置选择 Responses 或 Chat Completions。"""
        self.last_usage = LlmUsage()
        protocol = self.cfg.protocol
        if protocol == "auto":
            model = self.cfg.model.lower()
            protocol = "responses" if "deepseek" in model or "qwen3.8" in model else "chat_completions"
        if protocol == "responses":
            try:
                return self._responses(
                    messages,
                    temperature=temperature,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    effort=effort,
                )
            except LlmError as exc:
                # auto 只在接口不存在且尚未执行任何工具时回退，认证、余额和模型错误不重放。
                if self.cfg.protocol != "auto" or not any(
                    marker in str(exc) for marker in ("返回 404", "返回 405", "接口不存在")
                ):
                    raise
        return self._chat_completions(
            messages,
            temperature=temperature,
            json_mode=json_mode,
            max_tokens=max_tokens,
            effort=effort,
        )

    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        effort: str | None = None,
        force_tool: str | None = None,
    ) -> dict[str, Any]:
        """使用供应商原生函数工具，并转换为 Agent 内部动作格式。"""
        self.last_usage = LlmUsage()
        if force_tool and force_tool not in {item["name"] for item in tools}:
            raise LlmError("强制结构化函数未包含在当前工具合同中")
        protocol = self.cfg.protocol
        if protocol == "auto":
            model = self.cfg.model.lower()
            protocol = "responses" if "deepseek" in model or "qwen3.8" in model else "chat_completions"
        if protocol == "responses":
            try:
                return self._responses_tools(messages, tools, max_tokens=max_tokens, effort=effort, force_tool=force_tool)
            except LlmError as exc:
                if self.cfg.protocol != "auto" or not any(
                    marker in str(exc) for marker in ("返回 404", "返回 405", "接口不存在")
                ):
                    raise
        return self._chat_tools(messages, tools, max_tokens=max_tokens, effort=effort, force_tool=force_tool)

    @staticmethod
    def _public_tools(tools: list[dict[str, Any]], *, responses: bool) -> list[dict[str, Any]]:
        """只把名称、说明和 JSON Schema 发送给供应商。"""

        result: list[dict[str, Any]] = []
        for item in tools:
            function = {
                "name": item["name"],
                "description": item.get("description", ""),
                "parameters": item.get("parameters") or {"type": "object", "properties": {}},
            }
            result.append({"type": "function", **function} if responses else {"type": "function", "function": function})
        return result

    def _output_limit(self, requested: int | None) -> int:
        """所有协议都携带模型级上限，调用方不能绕过已配置的成本边界。"""

        return max(1, min(int(requested) if requested is not None else self.cfg.max_output_tokens, self.cfg.max_output_tokens))

    @staticmethod
    def _tool_arguments(value: Any) -> dict[str, Any]:
        """兼容字符串或已解码对象，保留严格校验且不回显任何参数正文。"""

        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            raise LlmError("原生工具参数必须是 JSON 对象或对象字符串")
        try:
            result = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LlmError(f"原生工具调用参数不是合法 JSON：{exc.msg}，位置 {exc.pos}") from exc
        if not isinstance(result, dict):
            raise LlmError("原生工具参数 JSON 顶层必须是对象")
        return result

    def _responses_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int | None,
        effort: str | None,
        force_tool: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "input": self._normalize_messages(messages, responses=True),
            "tools": self._public_tools(tools, responses=True),
            "tool_choice": {"type": "function", "name": force_tool} if force_tool else "auto",
        }
        payload["max_output_tokens"] = self._output_limit(max_tokens)
        chosen_effort = self._effort(effort)
        if chosen_effort:
            payload["reasoning"] = {"effort": chosen_effort}
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        last_error = ""
        with httpx.Client(timeout=self.cfg.timeout, trust_env=False) as client:
            for url in self._response_urls():
                for attempt in range(2):
                    try:
                        response = client.post(url, headers=headers, json=payload)
                    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                        last_error = f"{url} 调用失败：{exc}"
                        if attempt == 0:
                            time.sleep(0.4)
                            continue
                        break
                    if response.status_code in {404, 405}:
                        last_error = f"{url} 返回 {response.status_code}"
                        break
                    if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                        self._sleep_retry(response.headers.get("Retry-After"))
                        continue
                    if response.status_code >= 400:
                        self._raise_http_error(response.status_code, response.text)
                    try:
                        data = response.json()
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise LlmError("Responses API 没有返回合法 JSON") from exc
                    self.last_usage = self._responses_usage(data, messages)
                    self._check_completion(data)
                    actions: list[dict[str, Any]] = []
                    summary_parts: list[str] = []
                    for item in data.get("output") or []:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "function_call":
                            arguments = self._tool_arguments(item.get("arguments", "{}"))
                            actions.append(self._native_action(str(item.get("name") or ""), arguments))
                        elif item.get("type") == "message":
                            summary_parts.extend(
                                str(part.get("text") or "")
                                for part in item.get("content") or []
                                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}
                            )
                    return self._tool_result(actions, "".join(summary_parts))
        raise LlmError(last_error or "Responses API 接口不存在")

    def _chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int | None,
        effort: str | None,
        force_tool: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._normalize_messages(messages, responses=False),
            "tools": self._public_tools(tools, responses=False),
            "tool_choice": {"type": "function", "function": {"name": force_tool}} if force_tool else "auto",
        }
        payload["max_tokens"] = self._output_limit(max_tokens)
        payload.update(self._chat_reasoning(effort))
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        last_error = ""
        with httpx.Client(timeout=self.cfg.timeout, trust_env=False) as client:
            for url in self._urls():
                try:
                    response = client.post(url, headers=headers, json=payload)
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    last_error = f"{url} 调用失败：{exc}"
                    continue
                if response.status_code in {404, 405}:
                    last_error = f"{url} 返回 {response.status_code}"
                    continue
                if response.status_code >= 400:
                    self._raise_http_error(response.status_code, response.text)
                try:
                    data = response.json()
                    message = data["choices"][0]["message"]
                except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise LlmError("Chat Completions 没有返回合法工具响应") from exc
                self.last_usage = self._usage(data, messages)
                self._check_completion(data)
                actions: list[dict[str, Any]] = []
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = self._tool_arguments(function.get("arguments", "{}"))
                    actions.append(self._native_action(str(function.get("name") or ""), arguments))
                return self._tool_result(actions, self._content_text(message.get("content") or ""))
        raise LlmError(last_error or "没有可用的模型接口地址")

    @staticmethod
    def _native_action(name: str, arguments: Any) -> dict[str, Any]:
        """原生工具的身份来自供应商结构，不能被参数覆盖。"""

        if not isinstance(arguments, dict) or "tool" in arguments:
            raise LlmError("原生工具调用参数必须是对象，且不得覆盖工具名称")
        return {"tool": name, **arguments}

    @staticmethod
    def _tool_result(actions: list[dict[str, Any]], summary: str) -> dict[str, Any]:
        finish = next((item for item in actions if item.get("tool") == "finish"), None)
        executable = [item for item in actions if item.get("tool") != "finish"]
        if finish and not executable:
            return {
                "thought": "",
                "actions": [],
                "done": True,
                "summary": str(finish.get("summary") or summary or "任务已完成。"),
            }
        return {
            "thought": summary,
            "actions": executable,
            "done": False,
            "summary": "",
        }

    def _chat_completions(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> str:
        """调用模型并返回文本内容。"""

        # payload 是 OpenAI-compatible chat completions 请求体。
        payload = {
            "model": self.cfg.model,
            "messages": self._normalize_messages(messages, responses=False),
            "temperature": temperature,
        }
        payload.update(self._chat_reasoning(effort))
        # max_tokens 是调用方按智能体职责设置的输出硬上限。Coding 工具参数需要比分类回答
        # 更大，但仍不能允许模型无限生成完整文件并占住会话数分钟。
        payload["max_tokens"] = self._output_limit(max_tokens)
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
        with httpx.Client(timeout=self.cfg.timeout, trust_env=False) as client:
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
                                    self._raise_http_error(status, body)
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
                            self._raise_http_error(resp.status_code, resp.text)
                        self.working_urls[self.cfg.base_url.rstrip("/")] = url
                        data = resp.json()
                        self.last_usage = self._usage(data, messages)
                        self._check_completion(data)
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

    def _responses(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        json_mode: bool,
        max_tokens: int | None,
        effort: str | None,
    ) -> str:
        """调用 OpenAI-compatible Responses API 并提取文本。"""

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "input": self._normalize_messages(messages, responses=True),
            "temperature": temperature,
        }
        payload["max_output_tokens"] = self._output_limit(max_tokens)
        chosen_effort = self._effort(effort)
        if chosen_effort:
            payload["reasoning"] = {"effort": chosen_effort}
        if json_mode:
            payload["text"] = {"format": {"type": "json_object"}}
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        last_error = ""
        with httpx.Client(timeout=self.cfg.timeout, trust_env=False) as client:
            for url in self._response_urls():
                for attempt in range(2):
                    try:
                        response = client.post(url, headers=headers, json=payload)
                    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                        last_error = f"{url} 调用失败：{exc}"
                        if attempt == 0:
                            time.sleep(0.4)
                            continue
                        break
                    except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                        raise LlmError(f"{url} 模型响应超时：{exc}") from exc
                    if response.status_code in {404, 405}:
                        last_error = f"{url} 返回 {response.status_code}"
                        break
                    if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                        self._sleep_retry(response.headers.get("Retry-After"))
                        continue
                    if response.status_code >= 400:
                        self._raise_http_error(response.status_code, response.text)
                    try:
                        data = response.json()
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise LlmError("Responses API 没有返回合法 JSON") from exc
                    self.working_urls[f"responses:{self.cfg.base_url.rstrip('/')}"] = url
                    self.last_usage = self._responses_usage(data, messages)
                    self._check_completion(data)
                    text = data.get("output_text")
                    if isinstance(text, str) and text:
                        return text
                    parts: list[str] = []
                    for item in data.get("output") or []:
                        if not isinstance(item, dict) or item.get("type") != "message":
                            continue
                        for part in item.get("content") or []:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                parts.append(part["text"])
                    if parts:
                        return "".join(parts)
                    limit = min(max_tokens or self.cfg.max_output_tokens, self.cfg.max_output_tokens)
                    if self.last_usage.completion >= int(limit * 0.95):
                        raise LlmOutputLimitError("Responses API 推理耗尽输出预算，没有产出正文")
                    raise LlmError("Responses API 响应中没有可用文本")
        raise LlmError(last_error or "Responses API 接口不存在")

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        responses: bool,
    ) -> list[dict[str, Any]]:
        """把文本和本地图片引用转换为供应商内容块。"""

        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                normalized.append({"role": role, "content": str(content)})
                continue
            parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = str(part.get("type") or "")
                if kind in {"text", "input_text"}:
                    text = str(part.get("text") or "")
                    parts.append(
                        {"type": "input_text" if responses else "text", "text": text}
                    )
                elif kind in {"image", "input_image"}:
                    url = str(part.get("image_url") or "")
                    path = str(part.get("path") or "")
                    if path:
                        raw = Path(path).read_bytes()
                        mime = str(part.get("mime") or "image/png")
                        url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
                    if not url:
                        continue
                    parts.append(
                        {"type": "input_image", "image_url": url}
                        if responses
                        else {"type": "image_url", "image_url": {"url": url}}
                    )
            normalized.append({"role": role, "content": parts})
        return normalized

    def _chat_reasoning(self, effort: str | None) -> dict[str, Any]:
        """普通兼容接口不携带供应商专用字段，文本与原生工具请求保持一致。"""

        chosen = self._effort(effort)
        if not chosen:
            return {}
        fields: dict[str, Any] = {"reasoning_effort": chosen}
        if "deepseek" in self.cfg.model.lower():
            fields["thinking"] = {"type": "disabled" if chosen == "none" else "enabled"}
        return fields

    def _effort(self, value: str | None) -> str:
        chosen = value or self.default_effort
        if chosen == "inherit":
            chosen = self.cfg.default_effort
        chosen = chosen if chosen in self.cfg.efforts else self.cfg.default_effort
        # 统一界面的等级映射到供应商真正支持的等级，避免未知值被服务端忽略后启用高推理。
        if "deepseek" in self.cfg.model.lower():
            return {"minimal": "low", "medium": "high"}.get(chosen, chosen)
        return chosen

    def _raise_http_error(self, status: int, body: str) -> None:
        """分类供应商错误并保证密钥不会通过错误正文进入日志。"""

        safe = body[:500]
        if self.cfg.api_key:
            safe = safe.replace(self.cfg.api_key, "[密钥已脱敏]")
        lower = safe.lower()
        balance_markers = (
            "insufficient balance",
            "insufficient quota",
            "balance_not_enough",
            "余额不足",
            "额度不足",
            "账户余额",
        )
        if status == 402 or any(marker in lower for marker in balance_markers):
            raise LlmBalanceError("模型账户余额不足，已停止后续调用并保留现场")
        raise LlmError(f"模型接口错误 {status}: {safe}")

    def _responses_usage(
        self,
        data: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> LlmUsage:
        usage = data.get("usage") or {}
        prompt = int(usage.get("input_tokens") or 0)
        completion = int(usage.get("output_tokens") or 0)
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        cached = int(
            input_details.get("cached_tokens")
            or usage.get("prompt_cache_hit_tokens")
            or 0
        )
        cache_miss = int(
            usage.get("prompt_cache_miss_tokens")
            or max(prompt - cached, 0)
        )
        reasoning = int(output_details.get("reasoning_tokens") or 0)
        if prompt or completion:
            return LlmUsage(
                prompt=prompt,
                completion=completion,
                cached=min(cached, prompt),
                cache_miss=min(cache_miss, prompt),
                reasoning=min(reasoning, completion),
            )
        estimated_prompt = self.estimate_tokens(self._messages_text(messages))
        return LlmUsage(
            prompt=estimated_prompt,
            completion=self.estimate_tokens(str(data.get("output_text") or data.get("output") or "")),
            cache_miss=estimated_prompt,
            estimated=True,
        )

    @staticmethod
    def _check_completion(data: dict[str, Any]) -> None:
        """供应商明确报告截断时，部分正文不能伪装成完整结果。"""

        reason = (data.get("incomplete_details") or {}).get("reason")
        choices = data.get("choices") or []
        length_stop = any(isinstance(item, dict) and item.get("finish_reason") in {"length", "max_tokens"} for item in choices)
        if reason in {"max_output_tokens", "max_tokens"} or length_stop:
            raise LlmOutputLimitError("供应商报告输出预算耗尽，响应尚未完成")

    @staticmethod
    def _messages_text(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    str(item.get("text") or "")
                    for item in content
                    if isinstance(item, dict)
                )
        return "\n".join(parts)

    @staticmethod
    def _sleep_retry(value: str | None) -> None:
        """按服务端 Retry-After 做一次有上限的短退避。"""

        try:
            retry_after = min(float(value or "0.6"), 3.0)
        except ValueError:
            retry_after = 0.6
        time.sleep(max(retry_after, 0.1))

    def _stream_text(self, response: httpx.Response, messages: list[dict[str, Any]]) -> str:
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
        length_stop = False
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
            length_stop = length_stop or choice.get("finish_reason") in {"length", "max_tokens"}
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
        usage_source = usage_data or {"choices": [{"message": {"content": text}}]}
        self.last_usage = self._usage(usage_source, messages)
        if length_stop:
            raise LlmOutputLimitError("模型流式响应输出预算耗尽，正文尚未完成")
        if not text:
            raise LlmError("模型流式响应中没有可用的文本 content")
        return text

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        *,
        plain_text: bool = False,
        max_tokens: int | None = None,
        structured_tools: bool = False,
    ) -> dict[str, Any]:
        """返回结构化对象，并兼容 JSON 代码块与 LongCat 原生工具标签。"""

        if structured_tools and "tools" in self.cfg.capabilities:
            # 分析节点只返回数据，不执行该函数；用原生参数协议避免源码引号污染 JSON 正文。
            spec = {"name": "return_json", "description": "返回调用方要求的完整 JSON 对象，不执行文件或命令操作。",
                    "parameters": {"type": "object", "properties": {"result": {"type": "object", "additionalProperties": True}},
                                   "required": ["result"], "additionalProperties": False}}
            instruction = {"role": "system", "content": "本次只调用 return_json，把原始要求的完整对象放进 result 参数；不要输出 JSON 正文或解释，不改变原始字段合同。"}
            result = self.chat_tools([*messages, instruction], [spec], max_tokens=max_tokens, force_tool="return_json")
            actions = result.get("actions") or []
            if len(actions) != 1 or actions[0].get("tool") != "return_json" or not isinstance(actions[0].get("result"), dict):
                raise LlmError("结构化函数没有返回唯一的 result 对象，未执行任何动作")
            return actions[0]["result"]

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
        # 完整 JSON 中的源码、示例标签都只是数据，不能再次按文本工具协议执行。
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if not isinstance(parsed, dict):
                raise LlmError("模型 JSON 顶层必须是对象")
            return parsed
        # LongCat 偶尔会忽略 JSON 要求，改用其原生工具标签；这里统一转换为现有 ReAct actions。
        qwen_result = self._parse_qwen_tool_calls(text)
        if qwen_result:
            return qwen_result
        dsml_result = self._parse_dsml_tool_calls(text)
        if dsml_result:
            return dsml_result
        longcat_result = self._parse_longcat_tool_calls(text)
        if longcat_result:
            return longcat_result
        # 仅提取唯一合法对象；代码中的花括号和多个相互矛盾的对象不能冒充结果。
        candidate = self._extract_json_object(text)
        if candidate:
            text = candidate
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise LlmError("模型 JSON 顶层必须是对象")
            return parsed
        except json.JSONDecodeError as exc:
            limit = min(max_tokens or self.cfg.max_output_tokens, self.cfg.max_output_tokens)
            if self.last_usage.completion >= int(limit * 0.95):
                raise LlmOutputLimitError("模型输出达到预算上限，未返回完整 JSON") from exc
            # 抛出前带上截断后的模型输出，方便定位 prompt 或模型格式问题。
            raise LlmError(f"模型没有返回合法 JSON：{exc}\n{text[:1000]}") from exc

    def _parse_qwen_tool_calls(self, text: str) -> dict[str, Any] | None:
        """解析 Qwen XML 工具格式；严格校验完整边界，保留代码正文缩进。"""

        start = text.find("<tool_call>")
        if start < 0:
            return None
        suffix = text[start:]
        calls = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        if calls.sub("", suffix).strip():
            raise LlmError("XML 工具调用边界不完整，未执行任何动作")
        actions: list[dict[str, Any]] = []
        for call in calls.finditer(suffix):
            body = call.group(1)
            function = re.fullmatch(r"<function=([\w.-]+)>\s*(.*?)\s*</function>", body, re.DOTALL)
            if function is None:
                raise LlmError("XML 工具函数格式无效")
            name, parameters_text = function.groups()
            parameters = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.DOTALL)
            if parameters.sub("", parameters_text).strip():
                raise LlmError("XML 工具参数边界不完整")
            arguments: dict[str, Any] = {}
            for parameter in parameters.finditer(parameters_text):
                key, value = parameter.groups()
                if key in arguments:
                    raise LlmError("XML 工具参数重复")
                # 模板包裹的一对换行不是正文，正文内部空白和缩进保持不变。
                value = value.removeprefix("\n").removesuffix("\n")
                if key in {"expected", "max_chars", "max_results", "start", "arguments", "paths", "texts"}:
                    try:
                        parsed: Any = json.loads(value)
                    except json.JSONDecodeError as exc:
                        raise LlmError("XML 工具类型参数不是合法 JSON") from exc
                    arguments[key] = parsed
                else:
                    arguments[key] = value
            actions.append(self._native_action(name, arguments))
        if not actions:
            raise LlmError("XML 没有有效工具调用")
        return self._tool_result(actions, text[:start].strip())

    def _parse_dsml_tool_calls(self, text: str) -> dict[str, Any] | None:
        """把供应商 DSML 工具标记转为同一动作协议，参数正文按语法原样保留。"""

        marker = r"[|｜]+DSML[|｜]+"
        if not re.match(rf"\s*<{marker}\s*(?:calls|tool_calls|function_calls)>", text):
            if re.search(rf"<{marker}\s*(?:calls|tool_calls|function_calls)>", text):
                raise LlmError("DSML 工具调用前存在非协议文本，请返回完整 JSON 或纯 DSML；未执行任何动作")
            return None
        outer = re.fullmatch(rf"\s*<{marker}\s*(calls|tool_calls|function_calls)>\s*(.*?)\s*</{marker}\s*\1>\s*", text, re.DOTALL)
        if outer is None:
            raise LlmError("DSML 工具调用边界不完整，未执行任何动作")
        calls = re.compile(rf'<{marker}\s*invoke\s+name="([^\"]+)">\s*(.*?)</{marker}\s*invoke>', re.DOTALL)
        parameters = re.compile(rf'<{marker}\s*parameter\s+name="([^\"]+)"(?:\s+string="(true|false)")?>(.*?)</{marker}\s*parameter>', re.DOTALL)
        actions: list[dict[str, Any]] = []
        body = outer.group(2)
        if calls.sub("", body).strip():
            raise LlmError("DSML 工具调用含无法解析的内容，未执行任何动作")
        for call in calls.finditer(body):
            name, arguments_text = call.groups()
            if parameters.sub("", arguments_text).strip():
                raise LlmError("DSML 工具参数边界不完整，未执行任何动作")
            arguments: dict[str, Any] = {}
            for parameter in parameters.finditer(arguments_text):
                key, is_string, raw = parameter.groups()
                if key in arguments:
                    raise LlmError("DSML 工具参数重复，未执行任何动作")
                if is_string == "false":
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise LlmError("DSML 非字符串参数不是合法 JSON") from exc
                else:
                    value = raw
                arguments[key] = value
            actions.append(self._native_action(name, arguments))
        if not actions:
            raise LlmError("DSML 没有包含有效工具动作")
        return self._tool_result(actions, "")

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
        """提取唯一合法的完整对象，不从截断对象内部抢救出可执行动作。"""

        # start 是当前候选对象起始位置；depth 是花括号嵌套深度。
        start = -1
        depth = 0
        # in_string 和 escaped 用于忽略 JSON 字符串内部的花括号及转义引号。
        in_string = False
        escaped = False
        candidates: list[str] = []
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
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        candidates.append(candidate)
                    start = -1
        # 未闭合的外层对象不能被忽略，否则可能执行残缺批次中的早期动作。
        return candidates[0] if start < 0 and len(candidates) == 1 else ""

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

    def _usage(self, data: dict[str, Any], messages: list[dict[str, Any]]) -> LlmUsage:
        """从模型响应中读取 usage；没有 usage 时用文本长度估算。"""

        # usage 是 OpenAI-compatible 响应里的标准 token 用量字段。
        usage = data.get("usage") or {}
        # prompt 是输入 token 数。
        prompt = int(usage.get("prompt_tokens") or 0)
        # completion 是输出 token 数。
        completion = int(usage.get("completion_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached = int(
            usage.get("prompt_cache_hit_tokens")
            or prompt_details.get("cached_tokens")
            or 0
        )
        cache_miss = int(
            usage.get("prompt_cache_miss_tokens")
            or max(prompt - cached, 0)
        )
        reasoning = int(completion_details.get("reasoning_tokens") or 0)
        if prompt or completion:
            return LlmUsage(
                prompt=prompt,
                completion=completion,
                cached=min(cached, prompt),
                cache_miss=min(cache_miss, prompt),
                reasoning=min(reasoning, completion),
            )
        # prompt_text 是所有输入消息拼接后的文本，用于估算输入 token。
        prompt_text = self._messages_text(messages)
        # completion_text 是 choices 的 JSON 字符串，用于估算输出 token。
        completion_text = json.dumps(data.get("choices", []), ensure_ascii=False)
        estimated_prompt = self.estimate_tokens(prompt_text)
        return LlmUsage(
            prompt=estimated_prompt,
            completion=self.estimate_tokens(completion_text),
            cache_miss=estimated_prompt,
            estimated=True,
        )
