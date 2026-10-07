from __future__ import annotations

import json

from backend.agents.graph import AgentGraph
from backend.agents.prompts import CODER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from backend.agents.verifier import (
    _private_model_config_facts,
    _reject_critical_warnings,
    _typescript_project_checks,
    verifier,
)
from backend.memory import MemoryStore
from backend.tools import FsTool
from llm import ModelStore


def test_agent_prompts_require_durable_stateless_chat_context() -> None:
    """Plan、Coding、Verifier 必须一致要求首轮上下文进入后续完整历史。"""

    assert "返回包含该 system、user、assistant 的完整 messages" in PLANNER_PROMPT
    assert "客户端保存但不渲染 system" in CODER_PROMPT
    assert "不得要求把正确的可持续 system 历史删除" in VERIFIER_PROMPT
    assert "客户端请求体、查询参数和响应类型要与服务端模型完全一致" in CODER_PROMPT
    assert "逐字段比较客户端实际发送的请求体" in VERIFIER_PROMPT


def test_static_web_check_detects_missing_dom_wiring(tmp_path) -> None:
    workdir = tmp_path / "web"
    fs = FsTool(str(workdir))
    fs.write("index.html", "<button onclick=\"sendMsg()\"></button><div id=\"chat\"></div>")
    fs.write("app.js", "document.getElementById('missing')")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    result = graph._static_web_check(fs, fs.list())

    assert result is not None
    assert result["ok"] is False
    assert result["missing_ids"] == ["missing"]
    assert result["missing_funcs"] == ["sendMsg"]


def test_static_web_check_supports_flask_template(tmp_path) -> None:
    fs = FsTool(str(tmp_path / "web"))
    fs.write("templates/index.html", '<main id="cards"></main>')
    fs.write("static/js/main.js", "document.getElementById('missing')")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    result = graph._static_web_check(fs, fs.list())

    assert result is not None
    assert result["ok"] is False
    assert result["missing_ids"] == ["missing"]


def test_static_web_check_supports_flask_static_index(tmp_path) -> None:
    """Flask 直接托管 static/index.html 时也必须检查前端接线。"""

    fs = FsTool(str(tmp_path / "web"))
    fs.write("static/index.html", '<main id="cards"></main>')
    fs.write("static/js/main.js", "document.getElementById('missing')")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    result = graph._static_web_check(fs, fs.list())

    assert result is not None
    assert result["ok"] is False
    assert result["missing_ids"] == ["missing"]


def test_static_web_check_reports_unstyled_classes_without_false_failure(tmp_path) -> None:
    """未单独声明的类可能只用于脚本定位，不能据此判定页面不可用。"""

    fs = FsTool(str(tmp_path / "web"))
    fs.write("index.html", '<main class="shell"><div id="cards"></div></main>')
    fs.write("app.js", 'cards.innerHTML = `<article class="card-title">标题</article>`;')
    fs.write("style.css", ".shell { display: block; }")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    result = graph._static_web_check(fs, fs.list())

    assert result is not None
    assert result["ok"] is True
    assert result["missing_classes"] == ["card-title"]


def test_verifier_keeps_coding_failure_as_a_failure(tmp_path) -> None:
    """没有可运行测试时，也不能把未完成的 Coding 阶段改判为成功。"""

    workdir = tmp_path / "project"
    workdir.mkdir()
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    result = verifier(
        graph,
        {
            "session_id": "s1",
            "workdir": str(workdir),
            "coding_ok": False,
            "tests": [],
            "tests_ok": True,
        }
    )

    assert result["tests_ok"] is False
    assert result["tests"][0]["cmd"] == "Coding 完成状态"


def test_verifier_uses_only_existing_node_scripts(tmp_path) -> None:
    """Node 验证命令来自 package.json，不猜测测试框架参数。"""

    fs = FsTool(str(tmp_path / "web"))
    fs.write(
        "package.json",
        json.dumps(
            {
                "scripts": {
                    "dev": "vite",
                    "test": "vitest run",
                    "typecheck": "tsc --noEmit",
                    "build": "vite build",
                }
            }
        ),
    )
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert commands == ["npm run test", "npm run typecheck", "npm run build"]


def test_typescript_project_requires_config_and_typecheck_script(tmp_path) -> None:
    """Vite 构建不做类型检查，TS 项目必须显式提供类型检查配置与脚本。"""

    fs = FsTool(str(tmp_path / "project"))
    fs.write("web/package.json", json.dumps({"scripts": {"build": "vite build"}}))
    fs.write("web/src/app.tsx", "export default function App() { return null }\n")

    checks = _typescript_project_checks(fs, fs.list())

    assert len(checks) == 1
    assert checks[0]["ok"] is False
    assert "缺少 tsconfig.json" in checks[0]["out"]
    assert "缺少可执行的 typecheck 脚本" in checks[0]["out"]

    fs.write("web/tsconfig.json", json.dumps({"compilerOptions": {"strict": True}}))
    fs.write(
        "web/package.json",
        json.dumps({"scripts": {"typecheck": "tsc --noEmit", "build": "vite build"}}),
    )

    assert _typescript_project_checks(fs, fs.list())[0]["ok"] is True


def test_python_verification_separates_compile_and_uv_project_check(tmp_path) -> None:
    """Python 语法检查不触发安装，uv 配置另用 dry-run 验证。"""

    fs = FsTool(str(tmp_path / "python-app"))
    fs.write("pyproject.toml", '[project]\nname = "demo"\nversion = "0.1.0"\n')
    fs.write("server.py", "VALUE = 1\n")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert commands == [
        "uv run --no-project python -m compileall -q server.py",
        "uv sync --dry-run",
    ]


def test_fastapi_entry_gets_real_import_smoke_check(tmp_path) -> None:
    """compileall 无法发现错误导入，常见 Web 入口还必须真实执行一次。"""

    fs = FsTool(str(tmp_path / "project"))
    fs.write(
        "api/pyproject.toml",
        '[project]\nname = "api"\nversion = "0.1.0"\ndependencies = ["fastapi"]\n',
    )
    fs.write("api/app.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert "uv run --project api python api/app.py" in commands


def test_packaged_fastapi_entry_uses_module_import_smoke_check(tmp_path) -> None:
    """包目录中的 FastAPI 入口应通过 -m 导入，避免脚本路径破坏包搜索路径。"""

    fs = FsTool(str(tmp_path / "project"))
    fs.write(
        "pyproject.toml",
        '[project]\nname = "api"\nversion = "0.1.0"\ndependencies = ["fastapi"]\n',
    )
    fs.write("app/__init__.py", "")
    fs.write("app/main.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert "uv run python -m app.main" in commands


def test_nested_frontend_backend_verification_uses_project_scoped_commands(tmp_path) -> None:
    """嵌套前后端项目必须在各自目录解析依赖和测试，不能误用根环境。"""

    fs = FsTool(str(tmp_path / "fullstack"))
    fs.write(
        "backend/pyproject.toml",
        '[project]\nname = "api"\nversion = "0.1.0"\ndependencies = ["pytest"]\n',
    )
    fs.write("backend/app/main.py", "READY = True\n")
    fs.write("backend/tests/test_api.py", "def test_ready():\n    assert True\n")
    fs.write(
        "frontend/package.json",
        json.dumps(
            {
                "scripts": {
                    "test": "vitest run",
                    "typecheck": "tsc --noEmit",
                    "build": "vite build",
                }
            }
        ),
    )
    fs.write("frontend/src/main.tsx", "export const ready = true;\n")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert commands == [
        "uv run --no-project python -m compileall -q backend",
        "uv sync --project backend --dry-run",
        "uv run --project backend python -m pytest -q backend/tests",
        "npm --prefix frontend run test",
        "npm --prefix frontend run typecheck",
        "npm --prefix frontend run build",
    ]


def test_python_project_runs_declared_build_and_smoke_scripts(tmp_path) -> None:
    """项目自带的常规构建与烟测脚本必须由 Verifier 独立重跑。"""

    fs = FsTool(str(tmp_path / "repo"))
    fs.write("app/pyproject.toml", "[project]\nname='demo'\nversion='0.1.0'\n")
    fs.write("app/server/main.py", "print('ok')\n")
    fs.write("app/tests/test_app.py", "def test_ok(): assert True\n")
    fs.write("app/scripts/build.py", "print('build')\n")
    fs.write("app/scripts/smoke.py", "print('smoke')\n")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert "uv run --project app python app/scripts/build.py" in commands
    assert "uv run --project app python app/scripts/smoke.py" in commands


def test_static_javascript_verification_includes_node_syntax_check(tmp_path) -> None:
    """无 package.json 的静态页面也必须验证 JavaScript 入口语法。"""

    fs = FsTool(str(tmp_path / "web"))
    fs.write("static/app.js", "const ready = true;\n")
    fs.write("main.py", "READY = True\n")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    commands = graph._verification_commands(fs, fs.list())

    assert commands[0] == "node --check static/app.js"
    assert "python -m compileall -q main.py" in commands


def test_unawaited_coroutine_warning_fails_verification() -> None:
    """pytest 退出码为零也不能掩盖协程未等待问题。"""

    result = _reject_critical_warnings(
        {
            "ok": True,
            "code": 0,
            "out": "17 passed\nRuntimeWarning: coroutine 'mock' was never awaited",
            "err": "",
        }
    )

    assert result["ok"] is False
    assert any("was never awaited" in issue for issue in result["issues"])


def test_verifier_infrastructure_failure_does_not_retry_coder(tmp_path) -> None:
    """语义审查服务异常应最终报告失败，不能要求 Coding 修改业务代码。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "tests_ok": False,
        "retry": 1,
        "tests": [{"cmd": "需求实现审查", "ok": False, "infra": True, "out": "超时"}],
    }

    assert graph.route_after_verifier(state) == "final"


def test_read_only_verify_failure_never_routes_to_coder(tmp_path) -> None:
    """用户要求只验证时，业务失败也只能报告，不能擅自修改项目。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "task_type": "verify",
        "tests_ok": False,
        "retry": 0,
        "tests": [{"cmd": "node --check app.js", "ok": False, "out": "SyntaxError"}],
    }

    assert graph.route_after_verifier(state) == "final"


def test_complex_project_stops_at_repair_budget(tmp_path) -> None:
    """复杂项目可精确修复，但达到四轮成本上限后必须停止。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    state = {
        "task_type": "code_gen",
        "tests_ok": False,
        "retry": 3,
        "tests": [{"cmd": "需求实现审查", "ok": False, "issues": ["缺少配置文件"]}],
    }

    # 验证失败先回到主管，由主管决定修复、重规划或请求澄清。
    assert graph.route_after_verifier(state) == "manager"

    state["retry"] = 4
    assert graph.route_after_verifier(state) == "final"


def test_verify_task_reviews_selected_files_without_changes(tmp_path, monkeypatch) -> None:
    """只读验收即使没有本轮变更，也必须把关键文件交给语义审查。"""

    workdir = tmp_path / "project"
    fs = FsTool(str(workdir))
    fs.write("README.md", "# 项目\n\n真实说明。\n")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    captured: dict[str, str] = {}

    class Usage:
        total = 10

    class ReviewClient:
        last_usage = Usage()

        def chat_json(self, messages, temperature=0, *, plain_text=False):
            assert plain_text is True
            captured["content"] = messages[1]["content"]
            return {"ok": True, "issues": [], "summary": "验收通过"}

    monkeypatch.setattr(graph, "_client", lambda agent, state: ReviewClient())
    monkeypatch.setattr(graph, "_ctx_text", lambda state: "目标：只读验收")
    result = verifier(
        graph,
        {
            "session_id": "s1",
            "workdir": str(workdir),
            "task_type": "verify",
            "coding_ok": True,
            "changes": [],
            "tests": [],
            "tokens": 0,
            "repo": {"selected": ["README.md"], "snippets": {"README.md": "# 项目"}},
        },
    )

    assert result["tests_ok"] is True
    assert result["tests"][-1]["cmd"] == "需求实现审查"
    assert "真实说明" in captured["content"]


def test_verifier_reads_complete_normal_source_instead_of_false_truncation(tmp_path, monkeypatch) -> None:
    """略大于 12K 的普通源码必须完整交给审查模型，不能误报磁盘文件被截断。"""

    workdir = tmp_path / "project"
    fs = FsTool(str(workdir))
    source = "// 用于填充审查上下文的合法注释\n" * 700 + "const VERIFIED_TAIL = true;\n"
    assert len(source) > 12_000
    fs.write("app.js", source)
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    class Usage:
        total = 10

    class ReviewClient:
        last_usage = Usage()

        def chat_json(self, messages, temperature=0, *, plain_text=False):
            prompt = messages[1]["content"]
            assert "const VERIFIED_TAIL = true;" in prompt
            assert "审查上下文在此截断" not in prompt
            return {"ok": True, "issues": [], "summary": "完整源码审查通过"}

    monkeypatch.setattr(graph, "_client", lambda agent, state: ReviewClient())
    monkeypatch.setattr(graph, "_ctx_text", lambda state: "目标：审查完整源码")

    result = verifier(
        graph,
        {
            "session_id": "s1",
            "workdir": str(workdir),
            "task_type": "code_mod",
            "coding_ok": True,
            "changes": ["app.js"],
            "tests": [],
            "tokens": 0,
            "repo": {"snippets": {}},
        },
    )

    assert result["tests_ok"] is True


def test_private_model_config_facts_verify_secret_without_exposing_it(tmp_path) -> None:
    """语义审查应确认受控复制成功，但提示中不得包含真实密钥。"""

    fs = FsTool(str(tmp_path / "project"))
    private = fs.root / ".local/model.json"
    private.parent.mkdir()
    private.write_text(
        json.dumps(
            {
                "base_url": "https://example.test/v1",
                "api_key": "sk-private-value",
                "model": "deepseek-flash",
            }
        ), encoding="utf-8",
    )

    facts = _private_model_config_facts(fs, [".local/model.json"])

    assert len(facts) == 1
    assert "model='deepseek-flash'" in facts[0]
    assert "api_key_nonempty=True" in facts[0]
    assert "api_key_is_redaction_placeholder=False" in facts[0]
    assert "sk-private-value" not in facts[0]
