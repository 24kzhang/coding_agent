from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.schema import McpToolInfo, SkillContext
from backend.agents.answer import _answer_summary, answer
from backend.agents.coder import _task_brief
from backend.agents.graph import AgentGraph
from backend.capabilities import CapabilityStore, McpClient, McpServerConfig, _McpAuthFlow
from backend.memory import MemoryStore
from backend.tools import FsTool, GitTool, ShellTool
from llm import ModelStore
from scripts.github_mcp import target


def test_mcp_process_env_removes_only_malformed_proxy_values(monkeypatch) -> None:
    """MCP 子进程不应因格式损坏的代理变量启动失败，也不能无条件清空正常代理。"""

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1::1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")

    env = McpClient._process_env()

    assert "HTTP_PROXY" not in env
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:8080"


def test_skill_discovery_and_selected_content_loading(tmp_path: Path) -> None:
    """只发现规范 Skill，并且只读取用户明确选择的正文。"""

    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: 检查代码缺陷\n---\n\n# 审查\n\n只读检查。\n",
        encoding="utf-8",
    )
    invalid_dir = skills_dir / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "SKILL.md").write_text("---\nname: bad id\n---\n正文\n", encoding="utf-8")
    store = CapabilityStore(skills_dir, tmp_path / "MCP" / "servers.json")

    available = store.list_skills()
    selected = store.load_skills(["review"])

    assert [item.id for item in available] == ["review"]
    assert selected[0].description == "检查代码缺陷"
    assert "只读检查" in selected[0].content
    with pytest.raises(ValueError, match="未找到 Skill"):
        store.load_skills(["missing"])


def test_selected_skill_loads_bundled_reference_documents(tmp_path: Path) -> None:
    """选中 Skill 时应同时加载包内参考资料，并拒绝依赖运行时读取 Agent 目录。"""

    skill_dir = tmp_path / "skills" / "design"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: design\ndescription: 前端设计\n---\n\n先确定设计方向。\n",
        encoding="utf-8",
    )
    (references / "checklist.md").write_text("检查键盘焦点与文本溢出。", encoding="utf-8")
    (skill_dir / "AGENTS.md").write_text("完整规则：不要使用无意义动画。", encoding="utf-8")
    store = CapabilityStore(tmp_path / "skills", tmp_path / "MCP" / "servers.json")

    content = store.load_skills(["design"])[0].content

    assert "先确定设计方向" in content
    assert "Skill 附属资料：references/checklist.md" in content
    assert "检查键盘焦点与文本溢出" in content
    assert "Skill 附属资料：AGENTS.md" in content
    assert "不要使用无意义动画" in content


def test_mcp_config_supports_common_mcp_servers_shape(tmp_path: Path) -> None:
    """兼容 mcpServers 对象格式，且前端摘要不泄露环境变量。"""

    config_path = tmp_path / "MCP" / "servers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": "python",
                        "args": ["server.py", "${workspace}"],
                        "env": {"SECRET": "hidden"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    store = CapabilityStore(tmp_path / "skills", config_path)

    config = store.list_mcp_configs()[0]
    info = store.list_mcps()[0].model_dump()

    assert config.id == "local"
    assert config.transport == "stdio"
    assert config.env == {"SECRET": "hidden"}
    assert "env" not in info
    assert "command" not in info


def test_mcp_stdio_process_wraps_npx_on_windows(tmp_path: Path) -> None:
    """Windows 必须通过 cmd.exe 执行 npm/npx 批处理入口。"""

    client = McpClient()
    config = McpServerConfig(
        id="filesystem",
        command="npx",
        args=["-y", "package-name", "${workspace}"],
    )

    windows_command, windows_args = client._stdio_process(
        config,
        str(tmp_path),
        windows=True,
    )
    mac_command, mac_args = client._stdio_process(
        config,
        str(tmp_path),
        windows=False,
    )

    assert windows_command == "cmd.exe"
    assert windows_args == ["/c", "npx", "-y", "package-name", str(tmp_path.resolve())]
    assert mac_command == "npx"
    assert mac_args == ["-y", "package-name", str(tmp_path.resolve())]


def test_mcp_stdio_client_discovers_and_calls_tool(tmp_path: Path) -> None:
    """用临时标准 MCP 服务验证 initialize、tools/list 和 tools/call 全链路。"""

    server = tmp_path / "server.py"
    server.write_text(
        """
from mcp.server.fastmcp import FastMCP

app = FastMCP("test-server")

@app.tool()
def echo(text: str) -> str:
    return "收到：" + text

if __name__ == "__main__":
    app.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    config = McpServerConfig(
        id="local",
        name="本地测试服务",
        transport="stdio",
        command=sys.executable,
        args=[str(server)],
        timeout=20,
    )
    client = McpClient()

    tools = client.list_tools(config, str(tmp_path))
    result = client.call_tool(config, str(tmp_path), "echo", {"text": "你好"})

    assert [tool.name for tool in tools] == ["echo"]
    assert result["ok"] is True
    assert "收到：你好" in result["text"]


def test_persistent_mcp_reuses_same_server_process(tmp_path: Path) -> None:
    """持久 MCP 的多次工具调用必须复用进程内状态，供 OAuth 令牌持续生效。"""

    server = tmp_path / "persistent_server.py"
    server.write_text(
        """
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

app = FastMCP("persistent-test")
instance_id = uuid4().hex

@app.tool()
def identity() -> str:
    return instance_id

if __name__ == "__main__":
    app.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    config = McpServerConfig(
        id="persistent",
        command=sys.executable,
        args=[str(server)],
        timeout=20,
        persistent=True,
    )
    client = McpClient()

    try:
        first = client.call_tool(config, str(tmp_path), "identity", {})
        second = client.call_tool(config, str(tmp_path), "identity", {})
    finally:
        client.close()

    assert first["ok"] is True
    assert first["text"] == second["text"]


def test_github_mcp_bootstrap_selects_supported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """GitHub MCP 启动器应覆盖 Apple Silicon macOS 和常见 Windows 架构。"""

    monkeypatch.setattr("scripts.github_mcp.platform.system", lambda: "Darwin")
    monkeypatch.setattr("scripts.github_mcp.platform.machine", lambda: "arm64")
    mac_asset, mac_digest = target()

    monkeypatch.setattr("scripts.github_mcp.platform.system", lambda: "Windows")
    monkeypatch.setattr("scripts.github_mcp.platform.machine", lambda: "AMD64")
    windows_asset, windows_digest = target()

    assert mac_asset == "github-mcp-server_Darwin_arm64.tar.gz"
    assert len(mac_digest) == 64
    assert windows_asset == "github-mcp-server_Windows_x86_64.zip"
    assert len(windows_digest) == 64


def test_github_device_auth_persists_verified_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备授权通过身份检查后应持久化，但不能出现在前端状态中。"""

    class Response:
        """模拟 GitHub 两个设备授权接口的最小响应。"""

        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            """模拟成功 HTTP 状态。"""

        def json(self) -> dict[str, object]:
            """返回固定 JSON。"""

            return self.payload

    responses = iter(
        [
            Response(
                {
                    "device_code": "device-secret",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 0,
                    "expires_in": 900,
                }
            ),
            Response({"error": "authorization_pending"}),
            Response({"access_token": "runtime-token"}),
        ]
    )
    monkeypatch.setattr("backend.capabilities.httpx.post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("backend.capabilities.time.sleep", lambda _seconds: None)
    auth_config = tmp_path / "MCP" / "auth.json"
    client = McpClient(auth_config)
    monkeypatch.setattr(
        client,
        "call_tool",
        lambda config, workdir, tool_name, arguments: {"ok": True},
    )
    config = McpServerConfig(
        id="github",
        name="GitHub",
        command="github-mcp-server",
        timeout=900,
        persistent=True,
        auth_tool="get_me",
        auth_mode="github_device",
        oauth_client_id="public-client-id",
        oauth_scopes="repo read:user",
        auth_token_env="GITHUB_PERSONAL_ACCESS_TOKEN",
    )
    flow = _McpAuthFlow()

    client._run_github_device_auth(config, str(tmp_path), flow)

    status = flow.snapshot().model_dump()
    runtime = client.runtime_env[client._connection_key(config, str(tmp_path))]
    persisted = json.loads(auth_config.read_text(encoding="utf-8"))
    assert status == {
        "status": "authorized",
        "msg": "GitHub授权成功",
        "url": None,
        "code": None,
    }
    assert runtime == {"GITHUB_PERSONAL_ACCESS_TOKEN": "runtime-token"}
    assert persisted == {"tokens": {"github": "runtime-token"}}
    assert "runtime-token" not in json.dumps(status)
    if sys.platform != "win32":
        assert auth_config.stat().st_mode & 0o777 == 0o600


def test_authorized_mcp_does_not_start_second_device_flow(tmp_path: Path) -> None:
    """当前进程已经验证身份后，重复点击授权不应生成新的设备码。"""

    client = McpClient()
    config = McpServerConfig(
        id="github",
        name="GitHub",
        command="github-mcp-server",
        timeout=900,
        persistent=True,
        auth_tool="get_me",
        auth_mode="github_device",
        oauth_client_id="public-client-id",
        auth_token_env="GITHUB_PERSONAL_ACCESS_TOKEN",
    )
    key = client._connection_key(config, str(tmp_path))
    client.runtime_env[key] = {"GITHUB_PERSONAL_ACCESS_TOKEN": "runtime-token"}
    client.verified_auth.add(key)

    status = client.start_auth(config, str(tmp_path))

    assert status.status == "authorized"
    assert status.url is None
    assert status.code is None


def test_saved_github_token_is_verified_and_reused_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新客户端应加载本机凭据，真实验证成功后直接恢复授权状态。"""

    auth_config = tmp_path / "MCP" / "auth.json"
    auth_config.parent.mkdir(parents=True)
    auth_config.write_text(
        json.dumps({"tokens": {"github": "saved-token"}}),
        encoding="utf-8",
    )
    client = McpClient(auth_config)
    config = McpServerConfig(
        id="github",
        name="GitHub",
        command="github-mcp-server",
        timeout=900,
        persistent=True,
        auth_tool="get_me",
        auth_mode="github_device",
        oauth_client_id="public-client-id",
        auth_token_env="GITHUB_PERSONAL_ACCESS_TOKEN",
    )
    calls: list[dict[str, str]] = []

    def verify_saved_token(*_args: object, **_kwargs: object) -> dict[str, bool]:
        """记录身份检查时注入当前工作目录的运行时凭据。"""

        key = client._connection_key(config, str(tmp_path))
        calls.append(dict(client.runtime_env[key]))
        return {"ok": True}

    monkeypatch.setattr(client, "call_tool", verify_saved_token)

    status = client.start_auth(config, str(tmp_path))

    assert status.status == "authorized"
    assert calls == [{"GITHUB_PERSONAL_ACCESS_TOKEN": "saved-token"}]
    assert client._credential_env(config) == {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "saved-token"
    }


class FakeMcpClient:
    """记录动态工具调用，避免单元测试启动外部服务。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def list_tools(self, config: McpServerConfig, workdir: str) -> list[McpToolInfo]:
        return [
            McpToolInfo(
                name="lookup",
                description="查询资料",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

    def call_tool(
        self,
        config: McpServerConfig,
        workdir: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((config.id, tool_name, arguments))
        return {"ok": True, "text": "查询完成", "result": {"value": 1}}


class FakeMcpAnswerModel:
    """模拟答疑模型先调用 MCP、再根据观察生成最终回答。"""

    def __init__(self) -> None:
        # cfg.model 让答疑节点走 LongCat 普通文本兼容分支。
        self.cfg = SimpleNamespace(model="LongCat-2.0")
        # last_usage 提供与真实 LlmClient 一致的 token 统计字段。
        self.last_usage = SimpleNamespace(total=10)
        # responses 是两轮 MCP ReAct 的固定模型输出。
        self.responses = iter(
            [
                {
                    "thought": "先查询真实数据",
                    "actions": [
                        {"tool": "write_file", "path": "bad.txt", "content": "禁止执行"},
                        {
                            "tool": "mcp__docs__lookup",
                            "arguments": {"query": "LangGraph"},
                        },
                    ],
                    "done": False,
                    "summary": "",
                },
                {
                    "thought": "工具结果足够回答",
                    "actions": [],
                    "done": True,
                    "summary": "MCP 已连接并成功完成查询。",
                },
            ]
        )

    def chat_json(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        """返回下一轮固定 ReAct 数据。"""

        return next(self.responses)


class FakeRepeatingMcpAnswerModel:
    """模拟连续请求同一个 MCP 工具的模型。"""

    def __init__(self) -> None:
        # cfg 和 last_usage 与真实客户端合同一致。
        self.cfg = SimpleNamespace(model="LongCat-2.0")
        self.last_usage = SimpleNamespace(total=10)

    def chat_json(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        """每轮都错误地返回同一个工具动作。"""

        return {
            "thought": "重复查询",
            "actions": [
                {
                    "tool": "mcp__docs__lookup",
                    "arguments": {"query": "LangGraph"},
                }
            ],
            "done": False,
            "summary": "",
        }

    def chat(self, *_args: object, **_kwargs: object) -> str:
        """纯文本总结阶段返回最终回答。"""

        return "已根据首次 MCP 查询结果完成回答。"


class FakeSharedNodeModel:
    """模拟普通智能节点自主搜索一次后恢复自身结构化输出合同。"""

    def __init__(self, *, search: bool = True) -> None:
        self.last_usage = SimpleNamespace(total=7)
        self.calls = 0
        self.search = search

    def chat_json(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        self.calls += 1
        if self.search and self.calls == 1:
            return {
                "thought": "需要最新外部资料",
                "actions": [
                    {
                        "tool": "mcp__search__lookup",
                        "arguments": {"query": "LangGraph 最新文档"},
                    }
                ],
            }
        return {"status": "plan", "title": "计划", "markdown": "根据资料生成计划。"}


class FakeSharedTextModel:
    """模拟文档智能节点先搜索再返回原有纯文本。"""

    def __init__(self) -> None:
        self.last_usage = SimpleNamespace(total=5)
        self.calls = 0

    def chat(self, *_args: object, **_kwargs: object) -> str:
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "thought": "先确认外部文档",
                    "actions": [
                        {
                            "tool": "mcp__search__lookup",
                            "arguments": {"query": "项目依赖文档"},
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return "# 项目说明\n\n已结合必要的外部资料。"


def test_graph_only_allows_discovered_mcp_tools(tmp_path: Path) -> None:
    """动态工具必须同时命中本轮服务选择和 tools/list 发现结果。"""

    fake = FakeMcpClient()
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "memory"), mcp_client=fake)
    project = tmp_path / "project"
    project.mkdir()
    state = {
        "session_id": "s1",
        "workdir": str(project),
        "mcp_servers": [
            McpServerConfig(
                id="docs",
                name="资料服务",
                command=sys.executable,
                args=["server.py"],
            ).model_dump()
        ],
        "mcp_tools": [],
        "mcp_errors": [],
        "tokens": 0,
    }

    tools = graph._ensure_mcp_tools(state)
    allowed = graph._normalize_actions(
        [{"tool": "mcp__docs__lookup", "arguments": {"query": "LangGraph"}}],
        tools,
    )
    rejected = graph._normalize_actions(
        [{"tool": "mcp__other__lookup", "arguments": {"query": "secret"}}],
        tools,
    )
    result = graph._do_action(
        allowed[0],
        FsTool(str(project)),
        ShellTool(str(project)),
        GitTool(str(project)),
        state,
    )

    assert rejected == []
    assert result["ok"] is True
    assert fake.calls == [("docs", "lookup", {"query": "LangGraph"})]


def test_shared_intelligent_node_can_optionally_search(tmp_path: Path) -> None:
    """非 Answer/Coder 智能节点可以自主调用 shared MCP，并恢复节点原有 JSON 合同。"""

    fake_mcp = FakeMcpClient()
    model = FakeSharedNodeModel(search=True)
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "memory"), mcp_client=fake_mcp)
    project = tmp_path / "project"
    project.mkdir()
    config = McpServerConfig(
        id="search",
        name="网络搜索",
        command=sys.executable,
        args=["server.py"],
        shared=True,
    )
    state = {
        "session_id": "s1",
        "workdir": str(project),
        "mcp_servers": [config.model_dump()],
        "mcp_tools": [],
        "mcp_errors": [],
        "mcp_discovered_servers": [],
        "tokens": 0,
    }

    result = graph._model_json(
        state,
        "planner",
        model,
        [{"role": "system", "content": "返回计划 JSON。"}],
    )

    assert result["status"] == "plan"
    assert model.calls == 2
    assert fake_mcp.calls == [("search", "lookup", {"query": "LangGraph 最新文档"})]


def test_shared_intelligent_node_can_skip_search(tmp_path: Path) -> None:
    """搜索常驻只代表可选；模型不请求动作时不得额外调用 MCP。"""

    fake_mcp = FakeMcpClient()
    model = FakeSharedNodeModel(search=False)
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "memory"), mcp_client=fake_mcp)
    project = tmp_path / "project"
    project.mkdir()
    config = McpServerConfig(
        id="search",
        name="网络搜索",
        command=sys.executable,
        args=["server.py"],
        shared=True,
    )
    state = {
        "session_id": "s1",
        "workdir": str(project),
        "mcp_servers": [config.model_dump()],
        "mcp_tools": [],
        "mcp_errors": [],
        "mcp_discovered_servers": [],
        "tokens": 0,
    }

    result = graph._model_json(
        state,
        "planner",
        model,
        [{"role": "system", "content": "返回计划 JSON。"}],
    )

    assert result["status"] == "plan"
    assert model.calls == 1
    assert fake_mcp.calls == []


def test_shared_text_node_can_search_and_return_plain_text(tmp_path: Path) -> None:
    """Doc 等纯文本智能节点搜索后仍必须返回原有文本合同。"""

    fake_mcp = FakeMcpClient()
    model = FakeSharedTextModel()
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "memory"), mcp_client=fake_mcp)
    project = tmp_path / "project"
    project.mkdir()
    config = McpServerConfig(
        id="search",
        name="网络搜索",
        command=sys.executable,
        args=["server.py"],
        shared=True,
    )
    state = {
        "session_id": "s1",
        "workdir": str(project),
        "mcp_servers": [config.model_dump()],
        "mcp_tools": [],
        "mcp_errors": [],
        "mcp_discovered_servers": [],
        "tokens": 0,
    }

    result = graph._model_text(
        state,
        "doc",
        model,
        [{"role": "system", "content": "返回 Markdown。"}],
    )

    assert result.startswith("# 项目说明")
    assert model.calls == 2
    assert fake_mcp.calls == [("search", "lookup", {"query": "项目依赖文档"})]


def test_always_on_mcp_is_merged_without_manual_selection(tmp_path: Path) -> None:
    """常驻搜索服务应自动进入任务，但前端仍只保存用户手动选择。"""

    config_path = tmp_path / "MCP" / "servers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "id": "search",
                        "name": "网络搜索",
                        "transport": "streamable_http",
                        "url": "https://example.com/mcp",
                        "always_on": True,
                        "shared": True,
                    },
                    {
                        "id": "manual",
                        "name": "手动服务",
                        "command": "python",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    store = CapabilityStore(tmp_path / "skills", config_path)

    selected = store.select_mcps([])
    info = {item.id: item for item in store.list_mcps()}

    assert [item.id for item in selected] == ["search"]
    assert info["search"].always_on is True
    assert info["search"].shared is True


def test_answer_agent_discovers_and_calls_selected_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """答疑任务选择 MCP 后也必须发现和调用工具，不能只在 Coding 节点可见。"""

    fake_mcp = FakeMcpClient()
    fake_model = FakeMcpAnswerModel()
    memory = MemoryStore(tmp_path / "memory")
    graph = AgentGraph(ModelStore(), memory, mcp_client=fake_mcp)
    project = tmp_path / "project"
    project.mkdir()
    config = McpServerConfig(
        id="docs",
        name="资料服务",
        command=sys.executable,
        args=["server.py"],
    )
    state = {
        "session_id": "s1",
        "workdir": str(project),
        "text": "查询 LangGraph 资料",
        "mcp_servers": [config.model_dump()],
        "mcp_tools": [],
        "mcp_errors": [],
        "repo": {},
        "tokens": 0,
    }
    monkeypatch.setattr(graph, "_client", lambda *_args, **_kwargs: fake_model)
    monkeypatch.setattr(graph, "_ctx_text", lambda _state: "用户任务：查询 LangGraph 资料")

    result = answer(graph, state)

    assert result["final"] == "MCP 已连接并成功完成查询。"
    assert [item["action"] for item in result["mcp_tools"]] == ["mcp__docs__lookup"]
    assert fake_mcp.calls == [("docs", "lookup", {"query": "LangGraph"})]
    assert not (project / "bad.txt").exists()


def test_mcp_answer_summary_preserves_long_reply_prefix() -> None:
    """长 MCP 回答截断时必须保留标题开头，不能只留下末尾字符。"""

    text = "GitHub MCP 工具说明\n" + ("详细能力。" * 3000)

    result = _answer_summary(text, "失败")

    assert result.startswith("GitHub MCP 工具说明")
    assert result.endswith("[回答已按长度上限截断]")


def test_answer_agent_stops_repeated_mcp_action_and_summarizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型重复请求同一 MCP 动作时只能执行一次，随后切换纯文本总结。"""

    fake_mcp = FakeMcpClient()
    fake_model = FakeRepeatingMcpAnswerModel()
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "memory"), mcp_client=fake_mcp)
    project = tmp_path / "project"
    project.mkdir()
    config = McpServerConfig(
        id="docs",
        name="资料服务",
        command=sys.executable,
        args=["server.py"],
    )
    state = {
        "session_id": "s1",
        "workdir": str(project),
        "text": "查询 LangGraph 资料",
        "mcp_servers": [config.model_dump()],
        "mcp_tools": [],
        "mcp_errors": [],
        "repo": {},
        "tokens": 0,
    }
    monkeypatch.setattr(graph, "_client", lambda *_args, **_kwargs: fake_model)
    monkeypatch.setattr(graph, "_ctx_text", lambda _state: "用户任务：查询 LangGraph 资料")

    result = answer(graph, state)

    assert result["final"] == "已根据首次 MCP 查询结果完成回答。"
    assert fake_mcp.calls == [("docs", "lookup", {"query": "LangGraph"})]


def test_selected_skill_enters_coder_task_context() -> None:
    """Coding 专用摘要必须包含用户手动选择的 Skill 正文。"""

    state = {
        "context": SimpleNamespace(
            goal="修改接口",
            constraints=["最小改动"],
            skills=[
                SkillContext(
                    id="review",
                    name="review",
                    description="审查",
                    content="必须先检查调用方。",
                )
            ],
        )
    }

    text = _task_brief(state)

    assert "必须先检查调用方" in text
    assert "Skill: review" in text


def test_mcp_arguments_restore_longcat_string_types(tmp_path: Path) -> None:
    """LongCat 标签参数应按 MCP Schema 恢复数字、布尔和数组类型。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "memory"))

    converted = graph._coerce_mcp_arguments(
        {
            "limit": "5",
            "exact": "true",
            "tags": '["python", "agent"]',
            "query": "LangGraph",
        },
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "exact": {"type": "boolean"},
                "tags": {"type": "array"},
                "query": {"type": "string"},
            },
        },
    )

    assert converted == {
        "limit": 5,
        "exact": True,
        "tags": ["python", "agent"],
        "query": "LangGraph",
    }


def test_memory_restores_latest_capability_selection(tmp_path: Path) -> None:
    """历史会话只恢复最近一次 Skills/MCP 选择。"""

    store = MemoryStore(tmp_path / "memory")
    project = tmp_path / "project"
    project.mkdir()
    store.append(
        str(project),
        "s1",
        "manager",
        "session",
        "start",
        "开始",
    )
    store.append(
        str(project),
        "s1",
        "manager",
        "capability",
        "selection",
        "第一次选择",
        {"skills": ["one"], "mcps": []},
    )
    store.append(
        str(project),
        "s1",
        "manager",
        "capability",
        "selection",
        "第二次选择",
        {"skills": ["two"], "mcps": ["docs"], "container_shell": False},
    )

    assert store.session_capabilities(str(project), "s1") == {
        "skills": ["two"],
        "mcps": ["docs"],
        "container_shell": False,
    }


def test_old_session_defaults_to_container_shell(tmp_path: Path) -> None:
    """旧 selection 没有 Shell 字段时，前端恢复为新版本默认开启。"""

    store = MemoryStore(tmp_path / "memory")
    project = tmp_path / "project"
    project.mkdir()
    store.append(
        str(project),
        "legacy",
        "manager",
        "capability",
        "selection",
        "旧记录",
        {"skills": [], "mcps": []},
    )

    assert store.session_capabilities(str(project), "legacy")["container_shell"] is True
