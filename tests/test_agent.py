from __future__ import annotations

from backend.agents.doc import doc
from backend.agents.final import final
from backend.agents.graph import AgentGraph
from backend.agents.manager import manager
from backend.memory import MemoryStore
from llm import LlmError, ModelStore


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

    assert next_state["route"] == "planner"
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
