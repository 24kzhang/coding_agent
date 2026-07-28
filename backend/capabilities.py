from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field, ValidationError, model_validator

from api.schema import McpAuthStatus, McpInfo, McpToolInfo, SkillContext, SkillInfo
from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types

# ROOT 是 Coding Agent 自身的项目根目录，不是用户本轮选择的项目工作目录。
ROOT = Path(__file__).resolve().parents[1]
# SKILLS_DIR 保存安装到本 Agent 的 Skill，每个 Skill 使用独立子目录和 SKILL.md。
SKILLS_DIR = ROOT / "skills"
# MCP_DIR 保存 MCP 服务配置；运行时不会把配置复制到用户项目。
MCP_DIR = ROOT / "MCP"
# MCP_CONFIG 是 MCP 服务的主配置文件。
MCP_CONFIG = MCP_DIR / "servers.json"
# MCP_AUTH_CONFIG 是 MCP OAuth 凭据的本机持久化配置；该文件必须被 Git 忽略。
MCP_AUTH_CONFIG = MCP_DIR / "auth.json"
# SKILL_FILE_LIMIT 限制单个 Skill 包进入模型上下文的字符数，防止附属文档占满窗口。
SKILL_FILE_LIMIT = 40_000
# SKILL_TOTAL_LIMIT 限制一次任务选择的所有 Skill 总字符数。
SKILL_TOTAL_LIMIT = 100_000
# MCP_OUTPUT_LIMIT 限制一次 MCP 工具结果反馈给模型和事件流的字符数。
MCP_OUTPUT_LIMIT = 16_000
# ID_PATTERN 是 Skill、MCP 服务和动态工具名可使用的安全字符集合。
ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class McpServerConfig(BaseModel):
    """从 MCP/servers.json 读取的单个 MCP 服务配置。"""

    # id 是服务唯一标识，也是动态工具名的一部分。
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    # name 是前端展示名；为空时使用 id。
    name: str = Field(default="", max_length=120)
    # description 说明服务用途，帮助用户在前端选择。
    description: str = Field(default="", max_length=500)
    # enabled=False 的配置保留在文件中，但不会显示或允许执行。
    enabled: bool = True
    # transport 支持本地 stdio 和 MCP Streamable HTTP。
    transport: Literal["stdio", "streamable_http"] = "stdio"
    # command 是 stdio 服务启动命令，例如 npx、uvx 或 python。
    command: str | None = Field(default=None, max_length=4000)
    # args 是 stdio 命令参数，支持 ${workspace} 和 ${agent_root} 占位符。
    args: list[str] = Field(default_factory=list)
    # env 是只传给 MCP 子进程的环境变量，可在本地配置中保存服务密钥。
    env: dict[str, str] = Field(default_factory=dict)
    # cwd 是可选 stdio 启动目录，默认使用用户当前项目目录。
    cwd: str | None = Field(default=None, max_length=4000)
    # url 是 Streamable HTTP 服务地址。
    url: str | None = Field(default=None, max_length=4000)
    # headers 是 Streamable HTTP 请求头；接口不会把它返回前端。
    headers: dict[str, str] = Field(default_factory=dict)
    # timeout 是初始化、工具发现和工具调用的读取超时秒数。
    timeout: int = Field(default=60, ge=3, le=1800)
    # persistent 保持同一个 MCP 进程和会话，适用于需要连续身份状态的服务。
    persistent: bool = False
    # auth_tool 是无副作用的身份检查工具；配置后前端可以显式发起浏览器授权。
    auth_tool: str | None = Field(default=None, max_length=120, pattern=r"^[A-Za-z0-9._-]+$")
    # auth_mode 指定授权通道；github_device 可绕过内置浏览器无法访问本机回调的问题。
    auth_mode: Literal["mcp", "github_device"] | None = None
    # oauth_client_id 是公开的 OAuth 应用标识，不是客户端密钥。
    oauth_client_id: str | None = Field(default=None, max_length=200)
    # oauth_scopes 是设备授权请求范围，配置时应遵守最小权限原则。
    oauth_scopes: str = Field(default="", max_length=1000)
    # auth_token_env 指定持久化 token 注入 MCP 子进程时使用的环境变量名。
    auth_token_env: str | None = Field(
        default=None,
        max_length=120,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )

    @model_validator(mode="after")
    def validate_transport_fields(self) -> McpServerConfig:
        """保证不同 transport 的必填连接字段完整。"""

        if self.transport == "stdio" and not (self.command or "").strip():
            raise ValueError("stdio MCP 必须配置 command")
        if self.transport == "streamable_http" and not (self.url or "").strip():
            raise ValueError("streamable_http MCP 必须配置 url")
        if self.auth_mode == "github_device" and not (self.oauth_client_id or "").strip():
            raise ValueError("github_device 授权必须配置 oauth_client_id")
        if self.auth_mode == "github_device" and not (self.auth_token_env or "").strip():
            raise ValueError("github_device 授权必须配置 auth_token_env")
        if not self.name.strip():
            self.name = self.id
        return self


class CapabilityStore:
    """发现本 Agent 已安装的 Skills，并读取 MCP 服务配置。"""

    def __init__(self, skills_dir: Path | None = None, mcp_config: Path | None = None):
        # skills_dir 支持测试传临时目录；正式运行时固定为项目根目录 skills。
        self.skills_dir = (skills_dir or SKILLS_DIR).resolve()
        # mcp_config 支持测试隔离；正式运行时固定为 MCP/servers.json。
        self.mcp_config = (mcp_config or MCP_CONFIG).resolve()
        # 首次启动自动创建两个约定目录和空 MCP 配置。
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.mcp_config.parent.mkdir(parents=True, exist_ok=True)
        if not self.mcp_config.exists():
            self.mcp_config.write_text('{\n  "servers": []\n}\n', encoding="utf-8")

    def list_skills(self) -> list[SkillInfo]:
        """扫描 skills/**/SKILL.md，并返回不含完整正文的前端摘要。"""

        # result 按 id 去重，避免嵌套目录中出现同名 Skill 时产生歧义。
        result: dict[str, SkillInfo] = {}
        for path in sorted(self.skills_dir.glob("*/SKILL.md")):
            # 隐藏目录和越界软链接不属于可安装 Skill。
            if path.parent.name.startswith(".") or not self._is_under(path.resolve(), self.skills_dir):
                continue
            try:
                # text 只读取有限字符；元数据必须位于 SKILL.md 开头。
                text = path.read_text(encoding="utf-8")[:SKILL_FILE_LIMIT]
                metadata, _body = self._parse_skill(text)
            except (OSError, UnicodeError, ValueError, yaml.YAMLError):
                continue
            # skill_id 优先使用 YAML name；缺失时使用目录名。
            skill_id = str(metadata.get("name") or path.parent.name).strip()
            if not ID_PATTERN.fullmatch(skill_id) or skill_id in result:
                continue
            # description 用于选择弹窗说明，不把完整 Skill 提前发送给浏览器。
            description = self._clean_text(metadata.get("description"), 500)
            result[skill_id] = SkillInfo(
                id=skill_id,
                name=skill_id,
                description=description or "未提供说明",
                path=str(path.relative_to(self.skills_dir.parent)),
            )
        return sorted(result.values(), key=lambda item: item.id.lower())

    def load_skills(self, selected_ids: list[str]) -> list[SkillContext]:
        """读取用户明确选择的 Skill 正文，并按总预算构造模型上下文。"""

        # available 建立 id 到摘要的映射；未知 id 会明确报错，不静默忽略前端旧状态。
        available = {item.id: item for item in self.list_skills()}
        # unique_ids 保留用户选择顺序并去重。
        unique_ids = list(dict.fromkeys(selected_ids))
        unknown = [skill_id for skill_id in unique_ids if skill_id not in available]
        if unknown:
            raise ValueError("未找到 Skill：" + "、".join(unknown))
        # contexts 保存实际进入 AgentState 的 Skill；remaining 是共享字符预算。
        contexts: list[SkillContext] = []
        remaining = SKILL_TOTAL_LIMIT
        for skill_id in unique_ids:
            info = available[skill_id]
            path = (self.skills_dir.parent / info.path).resolve()
            if not self._is_under(path, self.skills_dir):
                raise ValueError(f"Skill 路径越界：{skill_id}")
            # content 包含入口指令和受预算约束的附属资料，避免安装后只有目录说明可用。
            content = self._skill_content(path)
            if len(content) > remaining:
                if remaining < 1000:
                    raise ValueError("所选 Skills 总内容过大，请减少选择数量")
                content = content[:remaining] + "\n\n[Skill 内容因总预算被截断]"
            remaining -= len(content)
            contexts.append(
                SkillContext(
                    id=info.id,
                    name=info.name,
                    description=info.description,
                    content=content,
                )
            )
        return contexts

    def _skill_content(self, path: Path) -> str:
        """读取 Skill 入口，并按预算附加 references 文档或编译后的 AGENTS.md。"""

        # root 是当前 Skill 的独立目录，附属文件必须仍位于该目录中。
        root = path.parent.resolve()
        # content 始终先保留 SKILL.md，确保触发条件和主流程不会被附属资料挤掉。
        content = path.read_text(encoding="utf-8")
        if len(content) >= SKILL_FILE_LIMIT:
            return content[:SKILL_FILE_LIMIT] + "\n\n[Skill 内容因单包上限被截断]"

        # candidates 优先加载作者显式放入 references 的资料，再加载完整编译文档。
        candidates = sorted(root.glob("references/**/*.md"))
        compiled = root / "AGENTS.md"
        if compiled.is_file():
            candidates.append(compiled)

        for candidate in candidates:
            # resolved 用于拒绝越界软链接；普通目录遍历不能成为读取 Agent 项目外文件的入口。
            resolved = candidate.resolve()
            if not candidate.is_file() or not self._is_under(resolved, root):
                continue
            # relative 会作为模型上下文中的资料来源标签，方便 Skill 正文引用具体文件。
            relative = candidate.relative_to(root).as_posix()
            header = f"\n\n---\n\n## Skill 附属资料：{relative}\n\n"
            # room 是当前 Skill 包剩余字符预算。
            room = SKILL_FILE_LIMIT - len(content)
            if room <= len(header) + 100:
                break
            try:
                auxiliary = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if len(header) + len(auxiliary) <= room:
                content += header + auxiliary
                continue
            # 单个大文档只取预算内前缀并明确标记，不继续加载优先级更低的资料。
            marker = "\n\n[Skill 附属资料因单包上限被截断]"
            available = max(0, room - len(header) - len(marker))
            content += header + auxiliary[:available] + marker
            break
        return content

    def list_mcp_configs(self) -> list[McpServerConfig]:
        """读取并校验 MCP 配置，兼容 servers 数组和 mcpServers 对象格式。"""

        try:
            # raw 是配置文件根对象；JSON 解析失败必须反馈给前端，不能显示成“没有服务”。
            raw = json.loads(self.mcp_config.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"MCP 配置读取失败：{exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("MCP 配置根节点必须是 JSON 对象")

        # entries 统一转换成带 id 的字典列表。
        entries: list[dict[str, Any]] = []
        if isinstance(raw.get("servers"), list):
            entries = [dict(item) for item in raw["servers"] if isinstance(item, dict)]
        elif isinstance(raw.get("mcpServers"), dict):
            for server_id, item in raw["mcpServers"].items():
                if isinstance(item, dict):
                    entries.append({"id": str(server_id), "name": str(server_id), **item})

        # result 只保留启用配置；重复 id 会直接报错，避免调用错误服务。
        result: list[McpServerConfig] = []
        seen: set[str] = set()
        for entry in entries:
            normalized = self._normalize_mcp_entry(entry)
            try:
                config = McpServerConfig.model_validate(normalized)
            except ValidationError as exc:
                server_id = str(entry.get("id") or "未命名")
                raise ValueError(f"MCP 配置 {server_id} 无效：{exc.errors()[0]['msg']}") from exc
            if config.id in seen:
                raise ValueError(f"MCP 服务 id 重复：{config.id}")
            seen.add(config.id)
            if config.enabled:
                result.append(config)
        return sorted(result, key=lambda item: item.id.lower())

    def list_mcps(self) -> list[McpInfo]:
        """返回可安全展示给前端的 MCP 服务摘要，不泄露命令环境变量或请求头。"""

        return [
            McpInfo(
                id=config.id,
                name=config.name,
                description=config.description or "未提供说明",
                transport=config.transport,
                enabled=config.enabled,
                auth=bool(config.auth_tool),
            )
            for config in self.list_mcp_configs()
        ]

    def select_mcps(self, selected_ids: list[str]) -> list[McpServerConfig]:
        """按前端选择顺序返回配置，并拒绝未知或禁用的服务 id。"""

        # available 仅包含启用服务，因此 disabled 配置同样不能绕过接口被执行。
        available = {item.id: item for item in self.list_mcp_configs()}
        unique_ids = list(dict.fromkeys(selected_ids))
        unknown = [server_id for server_id in unique_ids if server_id not in available]
        if unknown:
            raise ValueError("未找到或未启用 MCP：" + "、".join(unknown))
        return [available[server_id] for server_id in unique_ids]

    def _parse_skill(self, text: str) -> tuple[dict[str, Any], str]:
        """使用 YAML 解析 SKILL.md front matter，返回元数据和正文。"""

        if not text.startswith("---"):
            return {}, text
        # parts 最多切成 front matter、正文和多余部分，正文中的 --- 不受影响。
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Skill front matter 没有结束标记")
        metadata = yaml.safe_load(parts[1]) or {}
        if not isinstance(metadata, dict):
            raise ValueError("Skill front matter 必须是 YAML 对象")
        return metadata, parts[2].lstrip()

    def _normalize_mcp_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """把不同客户端常见字段名规范成 McpServerConfig。"""

        normalized = dict(entry)
        # type 是部分 MCP 客户端使用的 transport 字段名。
        if "transport" not in normalized and "type" in normalized:
            normalized["transport"] = normalized.pop("type")
        # url 存在而 command 不存在时，可推断为 Streamable HTTP。
        if "transport" not in normalized:
            normalized["transport"] = "streamable_http" if normalized.get("url") else "stdio"
        # streamable-http 是常见连字符写法。
        if normalized.get("transport") in {"http", "streamable-http", "streamableHttp"}:
            normalized["transport"] = "streamable_http"
        return normalized

    def _clean_text(self, value: Any, limit: int) -> str:
        """压缩摘要中的换行和多余空白。"""

        return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]

    def _is_under(self, path: Path, root: Path) -> bool:
        """判断解析后的路径是否仍在指定根目录中。"""

        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


@dataclass
class _McpAuthFlow:
    """一个仅存在于后端内存中的 MCP 浏览器授权流程。"""

    # status 是前端轮询使用的授权生命周期状态。
    status: Literal["starting", "authorization_required", "authorized", "error"] = "starting"
    # msg 是界面提示，不保存 OAuth token、授权码或工具结果。
    msg: str = "正在准备浏览器授权"
    # url 是 MCP 服务通过 URL elicitation 返回的一次性地址。
    url: str | None = None
    # code 是设备授权页面需要输入的短期用户码。
    code: str | None = None
    # ready 在拿到 URL 或身份检查完成时唤醒首次授权请求。
    ready: threading.Event = field(default_factory=threading.Event)
    # done 标记授权检查已经结束，允许用户重新发起一次检查。
    done: threading.Event = field(default_factory=threading.Event)
    # cancelled 让已经不再需要的设备授权轮询安静退出，不覆盖现有成功状态。
    cancelled: threading.Event = field(default_factory=threading.Event)
    # lock 保证 MCP 事件循环与 FastAPI 请求读取到一致状态。
    lock: threading.Lock = field(default_factory=threading.Lock)

    def require_authorization(self, url: str, message: str, code: str | None = None) -> None:
        """记录官方授权地址并通知等待中的 HTTP 请求。"""

        with self.lock:
            self.status = "authorization_required"
            self.msg = message or "请在浏览器中完成 GitHub 授权"
            self.url = url
            self.code = code
            self.ready.set()

    def finish(self, status: Literal["authorized", "error"], message: str) -> None:
        """记录授权检查终态并清除不再需要展示的一次性地址。"""

        with self.lock:
            self.status = status
            self.msg = message
            self.url = None
            self.code = None
            self.ready.set()
            self.done.set()

    def snapshot(self) -> McpAuthStatus:
        """复制可安全返回前端的状态。"""

        with self.lock:
            return McpAuthStatus(status=self.status, msg=self.msg, url=self.url, code=self.code)


class McpClient:
    """通过官方 MCP Python SDK 发现并调用用户选择的 MCP 服务。"""

    def __init__(self, auth_config: Path | None = None) -> None:
        # auth_config 支持测试隔离；正式运行固定使用被 Git 忽略的 MCP/auth.json。
        self.auth_config = (auth_config or MCP_AUTH_CONFIG).resolve()
        # auth_tokens 保存从本机凭据配置读取的服务 token，不会通过 API 返回前端。
        self.auth_tokens = self._load_auth_tokens()
        # connections 缓存显式启用 persistent 的服务连接，普通服务仍使用短连接。
        self.connections: dict[str, _PersistentMcpSession] = {}
        # connection_lock 保护 FastAPI 请求线程和 Coding 执行线程同时访问连接缓存。
        self.connection_lock = threading.Lock()
        # auth_flows 只保存当前进程中的 OAuth 展示状态，不保存 token。
        self.auth_flows: dict[str, _McpAuthFlow] = {}
        # runtime_env 保存按工作目录展开后、只注入对应 MCP 子进程的凭据环境变量。
        self.runtime_env: dict[str, dict[str, str]] = {}
        # verified_auth 记录当前进程已经通过 auth_tool 验证的连接键。
        self.verified_auth: set[str] = set()
        # auth_lock 保护授权接口线程、MCP 事件循环和后台身份检查线程共享的状态。
        self.auth_lock = threading.Lock()
        # 进程退出时关闭持久 MCP 子进程，避免本地开发停止后残留后台服务。
        atexit.register(self.close)

    def list_tools(self, config: McpServerConfig, workdir: str) -> list[McpToolInfo]:
        """连接 MCP 服务并返回标准化工具定义。"""

        if config.persistent:
            result = self._connection(config, workdir).list_tools()
            return self._tool_infos(result)
        return asyncio.run(self._list_tools(config, workdir))

    def call_tool(
        self,
        config: McpServerConfig,
        workdir: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """调用一个 MCP 工具，并把多模态结果压缩成可反馈给模型的结构。"""

        if config.persistent:
            result = self._connection(config, workdir).call_tool(tool_name, arguments)
            return self._tool_result(result)
        return asyncio.run(self._call_tool(config, workdir, tool_name, arguments))

    def test(self, config: McpServerConfig, workdir: str) -> dict[str, Any]:
        """使用初始化和 tools/list 测试 MCP 连通性，不执行有副作用的工具。"""

        tools = self.list_tools(config, workdir)
        return {
            "ok": True,
            "msg": f"连接成功，发现 {len(tools)} 个工具",
            "tools": [tool.model_dump() for tool in tools],
        }

    def start_auth(self, config: McpServerConfig, workdir: str) -> McpAuthStatus:
        """启动身份检查工具，并等待 MCP 返回浏览器授权地址。"""

        if not config.auth_tool:
            raise ValueError(f"MCP 服务 {config.id} 未配置浏览器授权")
        if not config.persistent:
            raise ValueError(f"MCP 服务 {config.id} 必须启用 persistent 才能保持 OAuth 登录")

        # key 与持久连接键一致，确保授权完成的是后续工具调用复用的服务进程。
        key = self._connection_key(config, workdir)
        with self.auth_lock:
            flow = self.auth_flows.get(key)
            if key in self.verified_auth:
                if flow is None:
                    flow = _McpAuthFlow()
                    self.auth_flows[key] = flow
                flow.cancelled.set()
                flow.finish("authorized", f"{config.name}授权成功")
                return flow.snapshot()
            if flow is None or flow.done.is_set():
                flow = _McpAuthFlow()
                self.auth_flows[key] = flow
                # saved_token 在重启后先经过真实身份工具验证，失效时自动重新进入设备授权。
                saved_token = self.auth_tokens.get(config.id)
                if saved_token and config.auth_token_env:
                    target = self._run_saved_auth
                    args = (config, workdir, flow, saved_token)
                else:
                    target = (
                        self._run_github_device_auth
                        if config.auth_mode == "github_device"
                        else self._run_auth
                    )
                    args = (config, workdir, flow)
                threading.Thread(
                    target=target,
                    args=args,
                    name=f"mcp-auth-{config.id}",
                    daemon=True,
                ).start()

        # 正常情况下初始化后会立即产生提示；超时则让前端继续轮询 starting。
        flow.ready.wait(timeout=min(config.timeout, 15))
        return flow.snapshot()

    def auth_status(self, config: McpServerConfig, workdir: str) -> McpAuthStatus:
        """返回授权状态；未发起过授权时返回 idle。"""

        key = self._connection_key(config, workdir)
        with self.auth_lock:
            flow = self.auth_flows.get(key)
            authorized = key in self.verified_auth
        if authorized:
            if flow is None:
                flow = _McpAuthFlow()
                with self.auth_lock:
                    self.auth_flows[key] = flow
            flow.cancelled.set()
            flow.finish("authorized", f"{config.name}授权成功")
            return flow.snapshot()
        if flow is None:
            return McpAuthStatus(status="idle", msg="尚未发起授权")
        return flow.snapshot()

    def _run_auth(
        self,
        config: McpServerConfig,
        workdir: str,
        flow: _McpAuthFlow,
    ) -> None:
        """后台调用只读身份工具，授权期间不占用 FastAPI 请求线程。"""

        try:
            result = self.call_tool(config, workdir, config.auth_tool or "", {})
            if result["ok"]:
                flow.finish("authorized", f"{config.name}授权成功")
            else:
                flow.finish("error", f"{config.name}身份检查失败")
        except Exception as exc:
            flow.finish("error", f"{config.name}授权失败：{exc}")

    def _run_saved_auth(
        self,
        config: McpServerConfig,
        workdir: str,
        flow: _McpAuthFlow,
        token: str,
    ) -> None:
        """验证重启前保存的 token；无效时清除凭据并自动启动新的设备流。"""

        try:
            # token_env 已在 github_device 配置校验中保证存在，这里仍做防御式检查。
            token_env = config.auth_token_env or ""
            if not token_env:
                raise ValueError("MCP 未配置 token 环境变量")
            self._set_runtime_env(config, workdir, {token_env: token})
            result = self.call_tool(config, workdir, config.auth_tool or "", {})
            if not result["ok"]:
                raise RuntimeError("已保存的身份凭据无效")
            # key 只有通过真实身份工具后才进入 verified_auth。
            key = self._connection_key(config, workdir)
            with self.auth_lock:
                self.verified_auth.add(key)
            flow.finish("authorized", f"{config.name}授权成功")
        except Exception:
            # 不把服务返回的认证错误或 token 写入前端；删除失效凭据后直接申请新设备码。
            self._delete_auth_token(config.id)
            self._clear_runtime_env(config, workdir)
            if config.auth_mode == "github_device":
                self._run_github_device_auth(config, workdir, flow)
            else:
                flow.finish("error", f"{config.name}已保存授权失效，请重新授权")

    def _run_github_device_auth(
        self,
        config: McpServerConfig,
        workdir: str,
        flow: _McpAuthFlow,
    ) -> None:
        """执行 GitHub 设备授权，验证成功后把 token 保存到本机 MCP 凭据配置。"""

        headers = {"Accept": "application/json", "User-Agent": "coding-agent-github-mcp"}
        try:
            start = httpx.post(
                "https://github.com/login/device/code",
                data={
                    "client_id": config.oauth_client_id or "",
                    "scope": config.oauth_scopes,
                },
                headers=headers,
                timeout=30,
            )
            start.raise_for_status()
            payload = start.json()
            device_code = str(payload["device_code"])
            user_code = str(payload["user_code"])
            verification_uri = str(payload["verification_uri"])
            interval = max(5, int(payload.get("interval", 5)))
            expires_at = time.monotonic() + min(int(payload.get("expires_in", 900)), config.timeout)
            flow.require_authorization(
                verification_uri,
                f"请在 GitHub 页面输入设备码：{user_code}",
                user_code,
            )

            token = ""
            while time.monotonic() < expires_at:
                time.sleep(interval)
                if flow.cancelled.is_set():
                    return
                response = httpx.post(
                    "https://github.com/login/oauth/access_token",
                    data={
                        "client_id": config.oauth_client_id or "",
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                if result.get("access_token"):
                    token = str(result["access_token"])
                    break
                error = str(result.get("error") or "")
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    interval += 5
                    continue
                raise RuntimeError(str(result.get("error_description") or error or "设备授权失败"))
            if not token:
                raise TimeoutError("等待 GitHub 设备授权超时")

            # token 只注入配置声明的环境变量，不进入会话 memory、事件流或接口响应。
            token_env = config.auth_token_env or ""
            self._set_runtime_env(config, workdir, {token_env: token})
            result = self.call_tool(config, workdir, config.auth_tool or "", {})
            if not result["ok"]:
                raise RuntimeError("GitHub 身份检查失败")
            # 身份验证通过后再落盘，避免持久化未验证或错误的凭据。
            self._save_auth_token(config.id, token)
            key = self._connection_key(config, workdir)
            with self.auth_lock:
                self.verified_auth.add(key)
            flow.finish("authorized", f"{config.name}授权成功")
        except Exception as exc:
            flow.finish("error", f"{config.name}授权失败：{exc}")

    def _load_auth_tokens(self) -> dict[str, str]:
        """读取 MCP/auth.json 中的 token；配置缺失或损坏时按未授权启动。"""

        try:
            # raw 必须是 {"tokens": {"server-id": "token"}}，其他字段不会参与认证。
            raw = json.loads(self.auth_config.read_text(encoding="utf-8"))
            entries = raw.get("tokens", {}) if isinstance(raw, dict) else {}
            if not isinstance(entries, dict):
                return {}
            # tokens 只接收安全服务 id 和非空字符串，避免异常结构污染子进程环境。
            return {
                str(server_id): token
                for server_id, token in entries.items()
                if ID_PATTERN.fullmatch(str(server_id))
                and isinstance(token, str)
                and token.strip()
            }
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return {}

    def _save_auth_token(self, server_id: str, token: str) -> None:
        """原子写入单个服务 token，并在 POSIX 系统上限制为当前用户可读写。"""

        if not ID_PATTERN.fullmatch(server_id) or not token:
            raise ValueError("MCP OAuth 凭据无效")
        with self.auth_lock:
            self.auth_tokens[server_id] = token
            snapshot = dict(self.auth_tokens)
        self._write_auth_tokens(snapshot)

    def _delete_auth_token(self, server_id: str) -> None:
        """删除失效的服务 token，同时保留其他 MCP 服务的凭据。"""

        with self.auth_lock:
            if server_id not in self.auth_tokens:
                return
            self.auth_tokens.pop(server_id, None)
            snapshot = dict(self.auth_tokens)
        self._write_auth_tokens(snapshot)

    def _write_auth_tokens(self, tokens: dict[str, str]) -> None:
        """把完整凭据快照写入临时文件后替换目标文件，避免中断产生半截 JSON。"""

        self.auth_config.parent.mkdir(parents=True, exist_ok=True)
        # temporary 与目标文件位于同一目录，replace 在常见本地文件系统中是原子操作。
        temporary = self.auth_config.with_suffix(self.auth_config.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"tokens": tokens}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if os.name != "nt":
            temporary.chmod(0o600)
        temporary.replace(self.auth_config)
        if os.name != "nt":
            self.auth_config.chmod(0o600)

    def _set_runtime_env(
        self,
        config: McpServerConfig,
        workdir: str,
        values: dict[str, str],
    ) -> None:
        """更新服务临时环境并关闭旧进程，使下一次连接使用新凭据。"""

        key = self._connection_key(config, workdir)
        with self.auth_lock:
            self.runtime_env[key] = dict(values)
        with self.connection_lock:
            connection = self.connections.pop(key, None)
        if connection is not None:
            connection.close()

    def _clear_runtime_env(self, config: McpServerConfig, workdir: str) -> None:
        """清除指定连接的临时凭据并关闭旧进程，供失效授权重新登录。"""

        key = self._connection_key(config, workdir)
        with self.auth_lock:
            self.runtime_env.pop(key, None)
            self.verified_auth.discard(key)
        with self.connection_lock:
            connection = self.connections.pop(key, None)
        if connection is not None:
            connection.close()

    async def _elicitation(
        self,
        config: McpServerConfig,
        workdir: str,
        _context: Any,
        params: mcp_types.ElicitRequestParams,
    ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
        """接收 MCP 安全带外提示，只处理 URL 模式浏览器授权。"""

        if getattr(params, "mode", "") != "url" or not getattr(params, "url", ""):
            return mcp_types.ErrorData(
                code=mcp_types.INVALID_REQUEST,
                message="当前客户端仅支持 MCP URL 授权提示",
            )
        key = self._connection_key(config, workdir)
        with self.auth_lock:
            flow = self.auth_flows.setdefault(key, _McpAuthFlow())
        flow.require_authorization(str(params.url), str(params.message))
        # accept 只表示客户端已接收 URL；真正结果仍由 GitHub 的 OAuth 回调决定。
        return mcp_types.ElicitResult(action="accept")

    async def _list_tools(self, config: McpServerConfig, workdir: str) -> list[McpToolInfo]:
        """异步执行 MCP initialize 和 tools/list。"""

        async with self._session(config, workdir) as session:
            result = await session.list_tools()
        return self._tool_infos(result)

    def _tool_infos(self, result: Any) -> list[McpToolInfo]:
        """把 MCP SDK 的工具列表转换成项目稳定使用的接口模型。"""

        return [
            McpToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        ]

    async def _call_tool(
        self,
        config: McpServerConfig,
        workdir: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """异步连接服务并执行 tools/call。"""

        async with self._session(config, workdir) as session:
            result = await session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=config.timeout),
            )
        return self._tool_result(result)

    def _tool_result(self, result: Any) -> dict[str, Any]:
        """把 MCP SDK 工具结果转换成可记录、可截断的结构化对象。"""

        # blocks 保存文本、资源和其他内容块的可序列化表示。
        blocks: list[Any] = []
        for item in result.content:
            if getattr(item, "type", "") == "text":
                blocks.append(getattr(item, "text", ""))
            else:
                blocks.append(item.model_dump(mode="json", by_alias=True, exclude_none=True))
        payload = {
            "content": blocks,
            "structured_content": result.structuredContent,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(text) > MCP_OUTPUT_LIMIT:
            text = text[:MCP_OUTPUT_LIMIT] + "\n... MCP 工具结果已按长度上限截断"
        return {
            "ok": not result.isError,
            "text": text,
            "result": payload,
        }

    def _connection(self, config: McpServerConfig, workdir: str) -> _PersistentMcpSession:
        """获取或创建与配置和工作目录绑定的持久 MCP 会话。"""

        # key 包含完整非敏感/敏感配置和值域目录，只保存在进程内且不会写入日志。
        key = self._connection_key(config, workdir)
        with self.connection_lock:
            connection = self.connections.get(key)
            if connection is None or not connection.alive:
                if connection is not None:
                    connection.close()
                connection = _PersistentMcpSession(self, config, workdir)
                self.connections[key] = connection
            return connection

    def _connection_key(self, config: McpServerConfig, workdir: str) -> str:
        """生成持久连接与 OAuth 状态共用的进程内键。"""

        return config.model_dump_json() + "\n" + str(Path(workdir).resolve())

    def close(self) -> None:
        """关闭当前客户端持有的全部持久 MCP 会话。"""

        with self.connection_lock:
            connections = list(self.connections.values())
            self.connections.clear()
        for connection in connections:
            connection.close()

    @asynccontextmanager
    async def _session(
        self,
        config: McpServerConfig,
        workdir: str,
    ) -> AsyncIterator[ClientSession]:
        """按配置 transport 创建并初始化短生命周期 MCP ClientSession。"""

        # timeout 统一用于 ClientSession 的默认读取超时。
        timeout = timedelta(seconds=config.timeout)
        if config.transport == "stdio":
            # credential_env 来自本机 MCP/auth.json，重启后仍可复用且不会进入前端。
            credential_env = self._credential_env(config)
            # temporary_env 允许当前工作目录覆盖刚完成授权后的凭据。
            key = self._connection_key(config, workdir)
            with self.auth_lock:
                temporary_env = dict(self.runtime_env.get(key, {}))
            # command/args 是展开工作目录占位符并完成 Windows 命令包装后的进程参数。
            command, args = self._stdio_process(config, workdir)
            # params 包含展开占位符后的命令、参数、环境变量和启动目录。
            params = StdioServerParameters(
                command=command,
                args=args,
                env={
                    **os.environ,
                    **{key: self._expand(value, workdir) for key, value in config.env.items()},
                    **credential_env,
                    **temporary_env,
                },
                cwd=self._expand(config.cwd, workdir) if config.cwd else workdir,
            )
            # errlog 必须提供真实文件描述符；os.devnull 跨平台丢弃服务 stderr，避免污染启动终端。
            with Path(os.devnull).open("w", encoding="utf-8") as errlog:
                async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timeout,
                        elicitation_callback=lambda context, params: self._elicitation(
                            config,
                            workdir,
                            context,
                            params,
                        ),
                    ) as session:
                        await session.initialize()
                        yield session
            return

        # HTTP transport 使用独立 AsyncClient，把本地配置 headers 只发送给该 MCP。
        async with httpx.AsyncClient(
            headers={key: self._expand(value, workdir) for key, value in config.headers.items()},
            timeout=config.timeout,
        ) as http_client:
            async with streamable_http_client(
                self._expand(config.url or "", workdir),
                http_client=http_client,
            ) as (read_stream, write_stream, _session_id):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timeout,
                    elicitation_callback=lambda context, params: self._elicitation(
                        config,
                        workdir,
                        context,
                        params,
                    ),
                ) as session:
                    await session.initialize()
                    yield session

    def _credential_env(self, config: McpServerConfig) -> dict[str, str]:
        """把持久化 token 转成仅供目标 MCP 子进程使用的环境变量。"""

        if not config.auth_token_env:
            return {}
        with self.auth_lock:
            token = self.auth_tokens.get(config.id, "")
        return {config.auth_token_env: token} if token else {}

    def _stdio_process(
        self,
        config: McpServerConfig,
        workdir: str,
        *,
        windows: bool | None = None,
    ) -> tuple[str, list[str]]:
        """生成跨平台 stdio 启动参数，Windows 自动包装 npm/npx 批处理命令。"""

        # command 和 args 先展开路径占位符，避免包装后漏掉工作目录。
        command = self._expand(config.command or "", workdir)
        args = [self._expand(item, workdir) for item in config.args]
        # is_windows 默认读取当前平台，测试可显式传入验证两个分支。
        is_windows = os.name == "nt" if windows is None else windows
        # Windows CreateProcess 不能像 POSIX 一样直接执行 npx/npm 的 .cmd 包装脚本。
        executable = Path(command).name.lower()
        if is_windows and executable in {"npm", "npm.cmd", "npx", "npx.cmd"}:
            batch_name = executable.removesuffix(".cmd")
            return "cmd.exe", ["/c", batch_name, *args]
        return command, args

    def _expand(self, value: str, workdir: str) -> str:
        """展开 MCP 配置中允许的两个路径占位符。"""

        return (
            str(value)
            .replace("${workspace}", str(Path(workdir).resolve()))
            .replace("${agent_root}", str(ROOT))
        )


class _PersistentMcpSession:
    """在独立事件循环线程中维持一个可重复调用的 MCP ClientSession。"""

    def __init__(self, client: McpClient, config: McpServerConfig, workdir: str) -> None:
        # client 创建底层 stdio/HTTP 连接；config 和 workdir 决定唯一服务实例。
        self.client = client
        self.config = config
        self.workdir = str(Path(workdir).resolve())
        # loop 专门承载 MCP SDK 的异步流，避免跨多个 asyncio.run 复用异步对象。
        self.loop = asyncio.new_event_loop()
        # ready 在 initialize 完成或启动失败时唤醒调用线程。
        self.ready = threading.Event()
        # stop_event 只能在 loop 所在线程创建和触发，用于正常退出上下文管理器。
        self.stop_event: asyncio.Event | None = None
        # session 是初始化完成后的 SDK 会话，所有工具调用都提交回同一个 loop。
        self.session: ClientSession | None = None
        # error 保存连接线程的启动异常，调用方会收到明确失败而不是无限等待。
        self.error: BaseException | None = None
        # thread 使用 daemon 模式作为异常退出兜底，正常退出仍由 McpClient.close() 完成。
        self.thread = threading.Thread(target=self._run, name=f"mcp-{config.id}", daemon=True)
        self.thread.start()

    @property
    def alive(self) -> bool:
        """连接线程是否仍可接受调用。"""

        return self.thread.is_alive() and self.error is None

    def list_tools(self) -> Any:
        """在持久会话的事件循环中执行 tools/list。"""

        session = self._wait_session()
        future = asyncio.run_coroutine_threadsafe(session.list_tools(), self.loop)
        return future.result(timeout=self.config.timeout)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """在同一持久会话中执行工具，保留服务进程内的 OAuth 状态。"""

        session = self._wait_session()
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=self.config.timeout),
            ),
            self.loop,
        )
        return future.result(timeout=self.config.timeout)

    def close(self) -> None:
        """通知连接协程退出并等待子进程清理。"""

        if self.loop.is_running() and self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=5)

    def _wait_session(self) -> ClientSession:
        """等待初始化完成，并把后台异常转换成同步调用异常。"""

        if not self.ready.wait(timeout=self.config.timeout):
            raise TimeoutError(f"MCP 服务 {self.config.id} 初始化超时")
        if self.error is not None:
            raise RuntimeError(f"MCP 服务 {self.config.id} 启动失败") from self.error
        if self.session is None or not self.thread.is_alive():
            raise RuntimeError(f"MCP 服务 {self.config.id} 已断开")
        return self.session

    def _run(self) -> None:
        """运行持久连接专用事件循环。"""

        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._hold())
        finally:
            self.loop.close()

    async def _hold(self) -> None:
        """保持异步上下文打开，直到客户端显式关闭连接。"""

        try:
            async with self.client._session(self.config, self.workdir) as session:
                self.session = session
                self.stop_event = asyncio.Event()
                self.ready.set()
                await self.stop_event.wait()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
