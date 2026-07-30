from __future__ import annotations

import json
from types import SimpleNamespace

from backend.agents.doc import doc
from backend.agents.final import final
from backend.agents.graph import AgentGraph
from backend.agents.manager import manager
from backend.agents.planner import (
    _mentions_field,
    _normalize_chat_contract,
    _plan_quality_issues,
    planner,
)
from backend.memory import MemoryStore
from backend.tools import FsTool
from llm import LlmError, ModelStore


def test_planner_retries_transient_gateway_errors(tmp_path, monkeypatch) -> None:
    """Plan 模型短暂 502 应退避后继续生成真实计划，不能立即降级。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    calls = 0

    class Usage:
        total = 9

    class FlakyClient:
        last_usage = Usage()

        def chat_json(self, _messages):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise LlmError("模型接口错误 502")
            return {"status": "plan", "markdown": "# 真实计划\n\n完整实现。"}

    monkeypatch.setattr(graph, "_client", lambda *_args: FlakyClient())
    monkeypatch.setattr(graph, "_ctx_text", lambda _state: "创建网站")
    monkeypatch.setattr("backend.agents.planner.time.sleep", lambda _delay: None)
    result = planner(
        graph,
        {
            "session_id": "plan-retry",
            "workdir": str(tmp_path / "project"),
            "text": "创建网站",
            "tokens": 0,
        },
    )

    assert calls == 3
    assert result["plan"]["markdown"].startswith("# 真实计划")
    assert result["tokens"] == 9


def test_planner_fails_over_after_model_timeout(tmp_path, monkeypatch) -> None:
    """Planner 单次长读取超时后应切到启用模型，不能继续等待同一模型多次超时。"""

    events = []
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"), emit=events.append)

    class Usage:
        total = 7

    class SlowClient:
        cfg = SimpleNamespace(id="slow")
        last_usage = Usage()

        def chat_json(self, _messages):
            raise LlmError("模型响应超时：The read operation timed out")

    class HealthyClient:
        last_usage = Usage()

        def __init__(self, cfg):
            self.cfg = cfg

        def chat_json(self, _messages):
            return {"status": "plan", "markdown": "# 可执行计划\n\n完整实现。"}

    fallback_cfg = SimpleNamespace(id="healthy")
    monkeypatch.setattr(graph, "_client", lambda *_args: SlowClient())
    monkeypatch.setattr(graph.model_store, "fallbacks", lambda *_args: [fallback_cfg])
    monkeypatch.setattr("backend.agents.planner.LlmClient", HealthyClient)
    monkeypatch.setattr(graph, "_ctx_text", lambda _state: "创建网站")

    result = planner(
        graph,
        {
            "session_id": "plan-failover",
            "workdir": str(tmp_path / "project"),
            "text": "创建网站",
            "tokens": 0,
        },
    )

    assert result["plan"]["markdown"].startswith("# 可执行计划")
    assert any(event.kind == "model" and "healthy" in event.msg for event in events)


def test_planner_rejects_plan_that_stays_invalid_after_repairs(tmp_path, monkeypatch) -> None:
    """自动修订耗尽后不得保存不合格计划或返回成功。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    class Usage:
        total = 3

    class BadClient:
        last_usage = Usage()

        def chat_json(self, _messages):
            return {"status": "plan", "markdown": "# 初稿\n\n前端直接读取数据。"}

    monkeypatch.setattr(graph, "_client", lambda *_args: BadClient())
    monkeypatch.setattr(graph, "_ctx_text", lambda _state: "基于数据集开发网站")
    result = planner(
        graph,
        {
            "session_id": "plan-invalid",
            "workdir": str(tmp_path / "project"),
            "text": "基于数据集开发网站",
            "tokens": 0,
        },
    )

    assert result["route"] == "final"
    assert result["tests_ok"] is False
    assert result["coding_ok"] is False
    assert result["plan"]["status"] == "invalid"
    assert not (tmp_path / "project" / "docs" / "plans" / "plan-invalid.md").exists()


def test_planner_switches_repair_model_when_quality_stagnates(tmp_path, monkeypatch) -> None:
    """同一模型修订后问题数不下降时，应换备用模型继续，而不是重复无效调用。"""

    events = []
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"), emit=events.append)

    class Usage:
        total = 4

    class StagnantClient:
        cfg = SimpleNamespace(id="stagnant")
        last_usage = Usage()

        def chat_json(self, _messages):
            return {"status": "plan", "markdown": "# 初稿\n\n前端直接读取数据。"}

    class RepairClient:
        last_usage = Usage()

        def __init__(self, cfg):
            self.cfg = cfg

        def chat_json(self, _messages):
            return {"status": "plan", "markdown": "# 可执行计划\n\n实现 `/api/exercises`，并用 pytest 覆盖数据接口。"}

    fallback_cfg = SimpleNamespace(id="repair")
    monkeypatch.setattr(graph, "_client", lambda *_args: StagnantClient())
    monkeypatch.setattr(graph.model_store, "fallbacks", lambda *_args: [fallback_cfg])
    monkeypatch.setattr("backend.agents.planner.LlmClient", RepairClient)
    monkeypatch.setattr(graph, "_ctx_text", lambda _state: "基于数据集开发网站")

    result = planner(
        graph,
        {
            "session_id": "plan-repair-switch",
            "workdir": str(tmp_path / "project"),
            "text": "基于数据集开发网站",
            "tokens": 0,
        },
    )

    assert result["plan"]["status"] == "plan"
    assert result["plan"]["markdown"].startswith("# 可执行计划")
    assert any(event.kind == "model" and "repair" in event.msg for event in events)


def test_planner_cross_reviews_after_two_improving_repairs(tmp_path, monkeypatch) -> None:
    """同一模型连续修订两次后即使持续改善，也应把最后检查交给备用模型。"""

    events = []
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"), emit=events.append)

    class Usage:
        total = 5

    class ImprovingClient:
        cfg = SimpleNamespace(id="primary")
        last_usage = Usage()

        def __init__(self):
            self.calls = 0

        def chat_json(self, _messages):
            self.calls += 1
            plans = [
                "# 初稿\n\n开始实现。",
                "# 修订一\n\n配置使用 `model_name`。",
                "# 修订二\n\n配置使用 `model_name`，实现 `/api/exercises`。",
            ]
            return {"status": "plan", "markdown": plans[min(self.calls - 1, 2)]}

    class CrossReviewClient:
        last_usage = Usage()

        def __init__(self, cfg):
            self.cfg = cfg

        def chat_json(self, _messages):
            return {
                "status": "plan",
                "markdown": "# 最终计划\n\n配置使用 `model_name`，实现 `/api/exercises` 并用 pytest 测试。",
            }

    primary = ImprovingClient()
    fallback_cfg = SimpleNamespace(id="cross")
    monkeypatch.setattr(graph, "_client", lambda *_args: primary)
    monkeypatch.setattr(graph.model_store, "fallbacks", lambda *_args: [fallback_cfg])
    monkeypatch.setattr("backend.agents.planner.LlmClient", CrossReviewClient)
    monkeypatch.setattr(graph, "_ctx_text", lambda _state: "基于数据集开发网站，配置字段 model_name")

    result = planner(
        graph,
        {
            "session_id": "plan-cross-review",
            "workdir": str(tmp_path / "project"),
            "text": "基于数据集开发网站，配置字段 model_name",
            "tokens": 0,
        },
    )

    assert result["plan"]["markdown"].startswith("# 最终计划")
    assert any(event.kind == "model" and "交叉审查" in event.msg for event in events)


def test_fallback_plan_redacts_api_key_but_preserves_execution_goal(tmp_path) -> None:
    """兜底 Plan 文档必须脱敏，执行元数据仍可另存原始目标。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    secret = "sk-local-test-secret"
    state = {
        "text": f'创建网站，apikey："{secret}"，并写入 config.json。',
        "plan_answers": [],
    }

    markdown = graph._fallback_plan_markdown(state)

    assert secret not in markdown
    assert "已脱敏" in markdown
    assert graph._plan_goal(state).endswith("并写入 config.json。")


def test_plan_redaction_handles_json_fields_and_bare_tokens(tmp_path) -> None:
    """模型即使把凭据复制进 JSON 代码块或正文，保存计划前也必须彻底脱敏。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    secret = "sk-local-test-secret-123456"
    markdown = f'''# 配置

```json
{{"api_key": "{secret}", "model_name": "demo"}}
```

备用值：{secret}
'''

    redacted = graph._redact_plan_secrets(markdown)

    assert secret not in redacted
    assert redacted.count("已脱敏") == 2
    assert '"model_name": "demo"' in redacted
    config_text = redacted.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(config_text)["api_key"].startswith("[已脱敏")


def test_plan_quality_review_detects_broken_full_stack_contracts(tmp_path) -> None:
    """数据网站计划若改配置字段、漏接口/测试或前端构造 system，必须在保存前触发修订。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "text": (
            "基于数据集开发网站，第一次提问添加动作上下文，之后传历史对话；"
            "展示所有详细信息；api_key、base_url、model_name 写入 JSON；"
            "只用 videos，不用 images。"
        ),
        "skills": [],
    }
    markdown = """# 初稿

使用 React + TypeScript。配置只有 `api_key`、`base_url` 和 `model`。
前端读取 JSON，打开弹窗时初始化 System Prompt，后续请求排除 system，并渲染 system。
currentMessages.push({ role: 'user', content: question });
媒体不使用 images，只使用 videos。
"""

    issues = _plan_quality_issues(graph, state, markdown)

    assert any("model_name" in item for item in issues)
    assert any("/api/exercises" in item for item in issues)
    assert any("pytest" in item for item in issues)
    assert any("typecheck" in item for item in issues)
    assert any("npm run build" in item for item in issues)
    assert any("所有原始字段" in item for item in issues)
    assert any("完整 messages" in item for item in issues)
    assert any("前端初始化 system" in item for item in issues)
    assert any("前后端统一请求体" in item for item in issues)
    assert any("messages 为空数组" in item for item in issues)
    assert any("不得过滤 system" in item for item in issues)
    assert any("界面不得渲染" in item for item in issues)
    assert any("不得先把当前 user" in item for item in issues)


def test_plan_quality_accepts_explicit_video_only_contract(tmp_path) -> None:
    """`/videos/*.gif` 与禁止 `/images/*` 的明确合同不应被审查器误报。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "text": "网页展示只用 videos 文件夹的 gif，不要用 images 文件夹的静图。",
        "skills": [],
    }
    markdown = "所有页面媒体请求只能指向 `/videos/*.gif`，明确禁止引用 `/images/*`。"

    issues = _plan_quality_issues(graph, state, markdown)

    assert not any("媒体合同" in item for item in issues)


def test_plan_field_match_accepts_real_language_subfield() -> None:
    assert _mentions_field("左侧显示 `instructions.zh` 和 `instruction_steps.zh`。", "instructions")
    assert _mentions_field("左侧显示 `instructions.zh` 和 `instruction_steps.zh`。", "instruction_steps")
    assert _mentions_field("| id | 唯一标识 |", "id")


def test_plan_quality_accepts_complete_detail_api_claim(tmp_path) -> None:
    """详情接口明确返回完整动作详情时，无需在同一区段重复整张 schema 表。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "text": "详情展示所有详细信息",
        "skills": [],
        "repo": {
            "snippets": {
                "data/exercise.schema.json": json.dumps(
                    {
                        "$defs": {
                            "exercise": {
                                "properties": {
                                    "id": {"type": "string"},
                                    "image": {"type": "string"},
                                }
                            }
                        }
                    }
                )
            }
        },
    }
    markdown = """| `id` | 标识 |
| `image` | 静态路径 |

`GET /api/exercises/{id}` 返回完整动作详情。
"""

    issues = _plan_quality_issues(graph, state, markdown)

    assert not any("详情 API 合同遗漏" in item for item in issues)


def test_repo_candidates_exclude_generated_plan_files(tmp_path) -> None:
    """历史 docs/plans 不能被下一次 Planner 当成仓库事实再次注入。"""

    workdir = tmp_path / "project"
    (workdir / "docs" / "plans").mkdir(parents=True)
    (workdir / "docs" / "plans" / "old.md").write_text("exercises-dataset 旧计划", encoding="utf-8")
    (workdir / "README.md").write_text("# 项目", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    fs = FsTool(str(workdir))

    candidates = graph._repo_candidates(fs, fs.list(), "exercises-dataset")

    assert "docs/plans/old.md" not in candidates


def test_chat_contract_normalizer_removes_frontend_system_contradictions() -> None:
    """Plan 编排层应确定性消除模型反复产生的前端 system 违约行。"""

    markdown = """- 前端初始化 System Prompt
- UI 渲染所有消息，包括 system
- currentMessages.push({ role: 'user', content: question })
- 后续追加新 user 消息后回传完整 messages
- 页面保持响应式
"""
    normalized = _normalize_chat_contract(
        markdown,
        "第一次提问添加动作信息，之后传历史对话。",
    )

    assert "前端初始化 System" not in normalized
    assert "渲染所有消息" not in normalized
    assert "currentMessages.push" not in normalized
    assert "追加新 user" not in normalized
    assert "前端不创建 system、不初始化 system、不过滤 system，也不显示 system" in normalized
    assert "页面保持响应式" in normalized


def test_direct_greeting_does_not_read_repo_or_write_files(tmp_path) -> None:
    events = []
    workdir = tmp_path / "proj"
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"), emit=events.append)

    result = graph.run(session_id="s1", workdir=str(workdir), text="你好")

    assert result.ok is True
    assert "你好" in result.summary
    assert result.files == []
    assert result.commands == []
    assert result.tests == []
    assert not (workdir / "README.md").exists()
    assert [event.agent for event in events] == ["manager", "manager"]


def test_manager_prefers_code_generation_when_doc_is_a_deliverable(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "session_id": "s1",
        "workdir": str(tmp_path / "proj"),
        "text": "创建一个手机销售店铺智能客服系统，要求包含中文说明文档。",
        "plan_mode": False,
        "execute_plan": False,
        "model_id": None,
    }

    result = graph._classify(state)

    assert result["task_type"] == "code_gen"
    assert result["need_doc"] is True


def test_manager_routes_selected_mcp_question_directly_to_answer(tmp_path) -> None:
    """MCP 能力查询不应因“检查”二字读取用户项目仓库。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "session_id": "s1",
        "workdir": str(tmp_path / "proj"),
        "text": "检查当前 GitHub MCP 有哪些工具和能力",
        "plan_mode": False,
        "mcp_servers": [{"id": "github", "name": "GitHub（只读）"}],
    }

    result = graph._classify(state)

    assert result["task_type"] == "general_answer"
    assert result["need_repo"] is False


def test_manager_keeps_mcp_assisted_code_change_in_coding_flow(tmp_path) -> None:
    """选择 MCP 不能把明确的项目代码修改任务降级成普通答疑。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "session_id": "s1",
        "workdir": str(tmp_path / "proj"),
        "text": "使用 GitHub MCP 修复项目代码并补测试",
        "plan_mode": False,
        "mcp_servers": [{"id": "github", "name": "GitHub（只读）"}],
    }

    result = graph._classify(state)

    assert result["task_type"] == "code_mod"
    assert result["need_code"] is True


def test_code_task_with_doc_routes_to_coder_then_doc(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "session_id": "s1",
        "workdir": str(tmp_path / "proj"),
        "text": "创建一个手机销售店铺智能客服系统，要求包含中文说明文档。",
        "plan_mode": False,
        "execute_plan": False,
        "model_id": None,
        "changes": [],
        "commands": [],
        "tests": [],
        "tests_ok": True,
        "retry": 0,
        "tokens": 0,
    }

    next_state = manager(graph, state)

    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "coder"
    assert next_state["after_verify"] == "doc"


def test_doc_task_routes_to_repo_then_doc_without_coder(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "session_id": "s1",
        "workdir": str(tmp_path / "proj"),
        "text": "根据当前项目生成 README 文档",
        "plan_mode": False,
        "execute_plan": False,
        "model_id": None,
        "changes": [],
        "commands": [],
        "tests": [],
        "tests_ok": True,
        "retry": 0,
        "tokens": 0,
    }

    next_state = manager(graph, state)

    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "doc"
    assert next_state["after_verify"] == "final"


def test_doc_model_failure_preserves_existing_readme(tmp_path, monkeypatch) -> None:
    """文档模型超时必须显式失败，不能用通用模板覆盖现有文档。"""

    workdir = tmp_path / "proj"
    workdir.mkdir()
    readme = workdir / "README.md"
    readme.write_text("# 原有完整文档\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    class BrokenClient:
        def chat(self, messages):
            raise LlmError("模型响应超时")

    monkeypatch.setattr(graph, "_client", lambda agent, state: BrokenClient())
    next_state = doc(
        graph,
        {
            "session_id": "s1",
            "workdir": str(workdir),
            "text": "更新文档",
            "tests": [],
            "tests_ok": True,
            "changes": [],
        },
    )

    assert readme.read_text(encoding="utf-8") == "# 原有完整文档\n"
    assert next_state["tests_ok"] is False
    assert next_state["tests"][-1]["cmd"] == "文档生成"


def test_doc_writes_plain_markdown_to_requested_path(tmp_path, monkeypatch) -> None:
    """文档智能体直接写 Markdown，并由编排确定目标路径。"""

    workdir = tmp_path / "proj"
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    class Usage:
        total = 12

    class MarkdownClient:
        last_usage = Usage()

        def chat(self, messages):
            return "# 使用指南\n\n真实内容。"

    monkeypatch.setattr(graph, "_client", lambda agent, state: MarkdownClient())
    next_state = doc(
        graph,
        {
            "session_id": "s1",
            "workdir": str(workdir),
            "text": "更新 docs/guide.md",
            "tests": [],
            "tests_ok": True,
            "changes": [],
            "tokens": 0,
        },
    )

    assert (workdir / "docs" / "guide.md").read_text(encoding="utf-8") == "# 使用指南\n\n真实内容。\n"
    assert next_state["changes"] == ["docs/guide.md"]


def test_plan_questions_are_formatted_for_user(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    questions = graph._normalize_plan_questions(
        [
            {
                "question": "前端技术栈选择？",
                "options": ["React + TypeScript + Vite", "Vue3 + TypeScript + Vite"],
                "recommended": "React + TypeScript + Vite",
                "reason": "React 生态成熟，TypeScript 提升可维护性。",
                "allow_custom": True,
            }
        ]
    )

    text = graph._format_plan_questions(questions)

    assert "Plan 模式需要你先做几个选择" in text
    assert "A. React + TypeScript + Vite（推荐）" in text
    assert "允许自定义回答" in text
    assert "{" not in text
    assert "}" not in text


def test_plan_questions_hide_json_parse_errors(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    questions = graph._normalize_plan_questions(
        [
            {
                "question": "是否直接按默认工程架构执行？",
                "options": ["直接执行", "继续细化计划"],
                "recommended": "模型暂时不可用：Expecting ',' delimiter: line 43 column 25 { bad json }",
                "reason": "模型没有返回合法 JSON：Expecting ',' delimiter",
                "allow_custom": True,
            }
        ]
    )

    text = graph._format_plan_questions(questions)

    assert "Expecting" not in text
    assert "模型没有返回合法 JSON" not in text
    assert "A. 直接执行（推荐）" in text


def test_manager_continues_pending_plan_from_option_reply(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    memory.append(
        workdir,
        "s1",
        "planner",
        "state",
        "pending_plan",
        "Plan 问题等待用户回答",
        {
            "goal": "创建一个本地多智能体编程系统",
            "questions": [
                {
                    "question": "前端技术栈选择？",
                    "options": ["React + TypeScript + Vite", "Vue3 + TypeScript + Vite"],
                    "recommended": "React + TypeScript + Vite",
                    "reason": "生态成熟",
                    "allow_custom": True,
                },
                {
                    "question": "后端框架选择？",
                    "options": ["FastAPI", "Flask"],
                    "recommended": "FastAPI",
                    "reason": "适合 API 服务",
                    "allow_custom": True,
                },
            ],
        },
    )
    graph = AgentGraph(ModelStore(), memory)
    state = {
        "session_id": "s1",
        "workdir": workdir,
        "text": "1A, 2A",
        "plan_mode": False,
        "execute_plan": False,
        "model_id": None,
        "changes": [],
        "commands": [],
        "tests": [],
        "tests_ok": True,
        "retry": 0,
        "tokens": 0,
    }

    next_state = manager(graph, state)

    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "planner"
    assert next_state["task_type"] == "plan_gen"
    assert "创建一个本地多智能体编程系统" in next_state["context"].goal
    assert "React + TypeScript + Vite" in next_state["context"].goal
    assert "FastAPI" in next_state["context"].goal


def test_manager_can_cancel_pending_plan(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    memory.append(workdir, "s1", "planner", "state", "pending_plan", "Plan 问题等待用户回答", {"goal": "旧计划", "questions": []})
    graph = AgentGraph(ModelStore(), memory)
    state = {
        "session_id": "s1",
        "workdir": workdir,
        "text": "取消计划",
        "plan_mode": False,
        "execute_plan": False,
        "model_id": None,
        "changes": [],
        "commands": [],
        "tests": [],
        "tests_ok": True,
        "retry": 0,
        "tokens": 0,
    }

    next_state = manager(graph, state)

    assert next_state["route"] == "final"
    assert "已取消" in next_state["final"]
    assert graph._latest_pending_plan(workdir, "s1") is None


def test_pending_plan_does_not_hijack_a_new_greeting(tmp_path) -> None:
    """普通问候应按新输入处理，并终止不再继续的旧 Plan。"""

    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    memory.append(
        workdir,
        "s1",
        "planner",
        "state",
        "pending_plan",
        "Plan 问题等待用户回答",
        {"goal": "旧任务", "questions": [{"question": "继续吗", "options": ["继续", "取消"]}]},
    )
    graph = AgentGraph(ModelStore(), memory)

    next_state = manager(
        graph,
        {
            "session_id": "s1",
            "workdir": workdir,
            "text": "你好",
            "plan_mode": False,
            "execute_plan": False,
            "model_id": None,
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        }
    )

    assert next_state["route"] == "final"
    assert next_state["task_type"] == "direct"
    assert "你好" in next_state["final"]
    assert graph._latest_pending_plan(workdir, "s1") is None


def test_manager_executes_latest_saved_plan(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    memory.append(
        workdir,
        "s1",
        "planner",
        "state",
        "plan_done",
        "计划已生成",
        {"status": "plan", "path": str(tmp_path / "plan.md"), "markdown": "# 执行计划\n\n实现客服系统。", "goal": "客服系统"},
    )
    graph = AgentGraph(ModelStore(), memory)
    state = {
        "session_id": "s1",
        "workdir": workdir,
        "text": "执行计划",
        "plan_mode": False,
        "execute_plan": True,
        "model_id": None,
        "changes": [],
        "commands": [],
        "tests": [],
        "tests_ok": True,
        "retry": 0,
        "tokens": 0,
    }

    next_state = manager(graph, state)

    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "coder"
    assert next_state["after_verify"] == "doc"
    assert next_state["execute_plan"] is True
    assert next_state["plan_mode"] is False
    assert next_state["executing_plan"] is True
    assert "实现客服系统" in next_state["context"].goal
    # 管理者只确认开始执行，不能提前关闭计划，否则中断后无法重试。
    assert graph._latest_saved_plan(workdir, "s1") is not None


def test_manager_preserves_original_credentials_when_executing_plan(tmp_path) -> None:
    """计划文档可脱敏，但执行上下文必须保留用户明确要求写入的本地凭据。"""

    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    secret = "test-local-key"
    memory.append(
        workdir,
        "s1",
        "planner",
        "state",
        "plan_done",
        "计划已生成",
        {
            "status": "plan",
            "path": str(tmp_path / "plan.md"),
            "markdown": "# 执行计划\n\n把用户提供的凭据写入本地配置。",
            "goal": f"创建应用，并把 api_key={secret} 写入 config.json。",
        },
    )
    graph = AgentGraph(ModelStore(), memory)

    next_state = manager(
        graph,
        {
            "session_id": "s1",
            "workdir": workdir,
            "text": "执行计划",
            "plan_mode": False,
            "execute_plan": True,
            "model_id": None,
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert secret in next_state["context"].goal
    assert secret not in next_state["plan"]["markdown"]


def test_manager_executes_plan_text_while_plan_toggle_stays_on(tmp_path) -> None:
    """前端 Plan 开关未关闭时，执行指令仍应进入 Repo/Coding。"""

    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    memory.append(
        workdir,
        "s1",
        "planner",
        "state",
        "plan_done",
        "计划已生成",
        {"status": "plan", "path": str(tmp_path / "plan.md"), "markdown": "# 执行计划\n\n实现健身网页。"},
    )
    graph = AgentGraph(ModelStore(), memory)

    next_state = manager(
        graph,
        {
            "session_id": "s1",
            "workdir": workdir,
            "text": "执行计划",
            "plan_mode": True,
            "execute_plan": False,
            "model_id": None,
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert next_state["route"] == "repo"
    assert next_state["execute_plan"] is True
    assert next_state["plan_mode"] is False


def test_final_closes_saved_plan_only_after_success(tmp_path) -> None:
    """执行失败时计划仍可重试，执行成功后才关闭计划。"""

    workdir = str(tmp_path / "proj")
    memory = MemoryStore(tmp_path / "mem")
    graph = AgentGraph(ModelStore(), memory)
    plan = {"status": "plan", "path": str(tmp_path / "plan.md"), "markdown": "# 计划"}

    memory.append(workdir, "failed", "planner", "state", "plan_done", "计划已生成", plan)
    failed_state = final(
        graph,
        {
            "session_id": "failed",
            "workdir": workdir,
            "plan": plan,
            "executing_plan": True,
            "coding_ok": False,
            "tests_ok": True,
            "coding_summary": "执行失败",
            "final": "用户确认执行已保存计划",
        },
    )
    assert graph._latest_saved_plan(workdir, "failed") is not None
    assert failed_state["result"]["summary"] == "执行失败"

    memory.append(workdir, "passed", "planner", "state", "plan_done", "计划已生成", plan)
    final(
        graph,
        {
            "session_id": "passed",
            "workdir": workdir,
            "plan": plan,
            "executing_plan": True,
            "coding_ok": True,
            "tests_ok": True,
            "coding_summary": "执行成功",
        },
    )
    assert graph._latest_saved_plan(workdir, "passed") is None


def test_manager_classifies_project_improvement_as_code_change(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "session_id": "s1",
        "workdir": str(tmp_path / "proj"),
        "text": "请仔细完善这个 agent 项目",
        "plan_mode": False,
        "execute_plan": False,
        "model_id": None,
    }

    result = graph._classify(state)

    assert result["task_type"] == "code_mod"
    assert result["need_code"] is True


def test_manager_does_not_expose_classification_reason_as_code_result(tmp_path) -> None:
    """代码任务的分类理由只用于路由，不能提前占用最终回复字段。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    next_state = manager(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "proj"),
            "text": "修改现有项目中的接口功能",
            "plan_mode": False,
            "execute_plan": False,
            "model_id": None,
        },
    )

    assert next_state["route"] == "repo"
    assert next_state["final"] == ""


def test_manager_routes_no_write_acceptance_to_verifier(tmp_path) -> None:
    """只读验收不能误触发 Coding 或文档生成。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    next_state = manager(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "proj"),
            "text": "不要修改文件，只运行测试并完成验收",
            "plan_mode": False,
            "execute_plan": False,
            "model_id": None,
        },
    )

    assert next_state["task_type"] == "verify"
    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "verifier"
    assert next_state["need_doc"] is False


def test_manager_recognizes_common_read_only_verification_phrases(tmp_path) -> None:
    """只读验证的自然表达必须稳定进入 verifier，不依赖管理者模型猜测。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    phrases = [
        "继续只读验证，禁止修改文件",
        "只读验收当前项目",
        "验证所有功能，禁止修改任何文件",
    ]

    for text in phrases:
        result = graph._classify(
            {
                "session_id": "s1",
                "workdir": str(tmp_path / "proj"),
                "text": text,
                "plan_mode": False,
            }
        )
        assert result["task_type"] == "verify"


def test_manager_does_not_treat_scoped_no_other_files_as_read_only(tmp_path) -> None:
    """“不要修改其他文件”只限制修改范围，不能吞掉明确的 README 写入任务。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    result = graph._classify(
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "proj"),
            "text": "恢复 README.md 并写清验证方式，不要修改其他文件",
            "plan_mode": False,
        }
    )

    assert result["task_type"] == "doc_gen"


def test_manager_routes_readme_recovery_to_doc_only(tmp_path) -> None:
    """项目背景词不能让“恢复 README”重复经过 Coding 和文档两个写入节点。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    result = graph._classify(
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "proj"),
            "text": "恢复健身动作网站被覆盖的 README.md，不要修改其他文件",
            "plan_mode": False,
        }
    )

    assert result["task_type"] == "doc_gen"
    assert result["need_code"] is False


def test_final_prefers_completed_code_summary_over_stale_manager_text(tmp_path) -> None:
    """即使状态中残留分类文本，代码任务也必须返回真实执行摘要。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = final(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "proj"),
            "task_type": "code_mod",
            "coding_ok": True,
            "tests_ok": True,
            "coding_summary": "接口修改完成，测试已通过。",
            "final": "用户需要修改接口，后续应先读取仓库。",
            "repo": {},
        },
    )

    assert state["result"]["summary"] == "接口修改完成，测试已通过。"


def test_final_uses_semantic_review_summary_for_verify_task(tmp_path) -> None:
    """只读验收的最终回复应包含可读的审查结论。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = final(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "proj"),
            "task_type": "verify",
            "coding_ok": True,
            "tests_ok": True,
            "tests": [{"cmd": "需求实现审查", "ok": True, "out": "核心功能和测试均通过。"}],
        },
    )

    assert state["result"]["summary"] == "核心功能和测试均通过。"


def test_manager_only_writes_long_term_memory_on_explicit_request(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    graph = AgentGraph(ModelStore(), memory)
    state = {
        "session_id": "s1",
        "workdir": workdir,
        "text": "请记住我以后都使用中文文档",
        "plan_mode": False,
        "execute_plan": False,
        "model_id": None,
        "changes": [],
        "commands": [],
        "tests": [],
        "tests_ok": True,
        "retry": 0,
        "tokens": 0,
    }

    next_state = manager(graph, state)

    assert next_state["route"] == "final"
    assert "使用中文文档" in memory.project_memory(workdir)
    assert "我以后都" not in memory.project_memory(workdir)
