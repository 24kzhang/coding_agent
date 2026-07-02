from __future__ import annotations

from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm import ModelStore


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

    next_state = graph.manager(state)

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

    next_state = graph.manager(state)

    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "doc"
    assert next_state["after_verify"] == "final"


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

    next_state = graph.manager(state)

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

    next_state = graph.manager(state)

    assert next_state["route"] == "final"
    assert "已取消" in next_state["final"]
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

    next_state = graph.manager(state)

    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "coder"
    assert next_state["after_verify"] == "doc"
    assert "实现客服系统" in next_state["context"].goal
