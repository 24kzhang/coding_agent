from __future__ import annotations

from types import SimpleNamespace

from backend.agents.coder import _is_discovery_command, _observation_brief, _test_brief, coder
from backend.agents.graph import AgentGraph
from backend.memory import MemoryStore
from llm import LlmError, ModelStore


class ProgressClient:
    """模拟模型先重复扫描仓库，看到拒绝观察后再写入文件。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "先扫描目录", "actions": [{"tool": "list_files"}], "done": False}
        assert "拒绝重复调用 list_files" in messages[-1]["content"]
        return {
            "thought": "仓库已扫描，开始写入",
            "actions": [{"tool": "write_file", "path": "app.py", "content": "print('ok')\n"}],
            "done": True,
            "summary": "已创建入口文件",
        }


def test_coder_rejects_repo_rescan_and_moves_to_write(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = ProgressClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建一个应用",
            "repo": {"files": ["README.md"], "snippets": {}, "stack": []},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert client.calls == 2
    assert result["coding_ok"] is True
    assert result["changes"] == ["app.py"]
    assert (tmp_path / "project" / "app.py").read_text(encoding="utf-8") == "print('ok')\n"


class OverwriteClient:
    """模拟模型尝试直接覆盖已有文件，随后按观察改用精确修改。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {
                "thought": "直接覆盖",
                "actions": [{"tool": "write_file", "path": "app.py", "content": "new\n"}],
                "done": False,
            }
        if self.calls == 2:
            assert "拒绝覆盖现有文件" in messages[-1]["content"]
            return {"thought": "先读取", "actions": [{"tool": "read_file", "path": "app.py"}], "done": False}
        return {
            "thought": "精确修改",
            "actions": [{"tool": "replace_file", "path": "app.py", "old": "old\n", "new": "new\n"}],
            "done": True,
            "summary": "修改完成",
        }


def test_coder_refuses_to_overwrite_existing_file(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("old\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = OverwriteClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "修改应用",
            "repo": {"files": ["app.py"], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert (project / "app.py").read_text(encoding="utf-8") == "new\n"


class RepeatedReadClient:
    """模拟模型写入后持续重复读取同一文件。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {
                "thought": "写入代码",
                "actions": [{"tool": "write_file", "path": "app.py", "content": "print('ok')\n"}],
                "done": False,
            }
        return {"thought": "重复确认", "actions": [{"tool": "read_file", "path": "app.py"}], "done": False}


def test_coder_stops_after_repeated_reads_without_progress(tmp_path) -> None:
    """已有写入后连续空转时应尽快失败，避免进入十轮无效读取。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = RepeatedReadClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建应用",
            "repo": {"files": [], "snippets": {}, "stack": []},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert client.calls == 7
    assert result["coding_ok"] is False
    assert "连续六轮只有失败、读取或状态查询" in result["coding_summary"]


class OwnedRewriteClient:
    """模拟修复轮次整体重写本次任务自己创建的文件。"""

    def __init__(self) -> None:
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        return {
            "thought": "根据最新内容整体修复",
            "actions": [{"tool": "write_file", "path": "app.py", "content": "print('fixed')\n"}],
            "done": True,
            "summary": "已修复入口文件",
        }


def test_coder_allows_rewrite_of_owned_file_after_fresh_read(tmp_path) -> None:
    """修复轮次可重写本任务创建的文件，但仍要求先加载最新磁盘内容。"""

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("print('broken')\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    graph._client = lambda agent, state: OwnedRewriteClient()

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "修复应用",
            "repo": {"files": [], "snippets": {}, "stack": []},
            "changes": ["app.py"],
            "commands": [],
            "tests": [{"cmd": "需求实现审查", "ok": False, "issues": ["入口错误"]}],
            "tests_ok": False,
            "retry": 1,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert (project / "app.py").read_text(encoding="utf-8") == "print('fixed')\n"


class ReadThenRewriteClient:
    """模拟普通修改任务先读取用户文件，再在确有必要时整体重写。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "读取损坏文件", "actions": [{"tool": "read_file", "path": "app.py"}], "done": False}
        assert "已读取且当前可直接修改的文件：app.py" in messages[-1]["content"]
        return {
            "thought": "整体清理重复结构",
            "actions": [{"tool": "write_file", "path": "app.py", "content": "print('clean')\n"}],
            "done": True,
            "summary": "已清理损坏结构",
        }


def test_coder_allows_full_rewrite_after_current_file_read(tmp_path) -> None:
    """已有文件在完整读取后可以整体重写，满足真实项目的大范围修复需求。"""

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("print('broken')\nprint('duplicate')\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    graph._client = lambda agent, state: ReadThenRewriteClient()

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "清理重复代码",
            "repo": {"files": ["app.py"], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert (project / "app.py").read_text(encoding="utf-8") == "print('clean')\n"


class MultiReplaceClient:
    """模拟同一读取快照上的多个互不冲突精确替换。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "读取文件", "actions": [{"tool": "read_file", "path": "app.py"}], "done": False}
        return {
            "thought": "修复两个独立问题",
            "actions": [
                {"tool": "replace_file", "path": "app.py", "old": "OLD_A", "new": "NEW_A"},
                {"tool": "replace_file", "path": "app.py", "old": "OLD_B", "new": "NEW_B"},
            ],
            "done": True,
            "summary": "两个问题均已修复",
        }


def test_coder_allows_multiple_precise_replacements_from_one_snapshot(tmp_path) -> None:
    """同批独立替换共享读取快照，避免每次小改都额外消耗一轮。"""

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("OLD_A\nOLD_B\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    graph._client = lambda agent, state: MultiReplaceClient()

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "修复两个错误",
            "repo": {"files": ["app.py"], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert (project / "app.py").read_text(encoding="utf-8") == "NEW_A\nNEW_B\n"


class ReplaceBlockClient:
    """模拟模型读取文件后用锚点工具替换文件末尾长函数。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "读取文件", "actions": [{"tool": "read_file", "path": "app.js"}], "done": False}
        return {
            "thought": "按唯一锚点替换文件尾函数",
            "actions": [
                {
                    "tool": "replace_block",
                    "path": "app.js",
                    "start_marker": "// send",
                    "end_marker": "",
                    "content": "// send\nfunction send() {\n    return 'new';\n}\n",
                }
            ],
            "done": True,
            "summary": "长函数已修改",
        }


def test_coder_can_replace_long_block_without_copying_old_content(tmp_path) -> None:
    """replace_block 必须通过读取守卫并作为真实文件变更进入 Coding 结果。"""

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.js").write_text("const head = true;\n// send\nfunction send() {\n    return 'old';\n}\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    graph._client = lambda agent, state: ReplaceBlockClient()

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "修改发送逻辑",
            "repo": {"files": ["app.js"], "snippets": {}, "stack": ["JavaScript"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert result["changes"] == ["app.js"]
    assert "return 'new'" in (project / "app.js").read_text(encoding="utf-8")


class RewriteCheckClient:
    """模拟模型整体写入后企图重复写入，收到拒绝后改为运行检查。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {
                "thought": "整体写入",
                "actions": [{"tool": "write_file", "path": "app.py", "content": "print('ok')\n"}],
                "done": False,
            }
        if self.calls == 2:
            return {
                "thought": "再次整体写入",
                "actions": [{"tool": "write_file", "path": "app.py", "content": "print('again')\n"}],
                "done": False,
            }
        assert "拒绝重复整体重写" in messages[-1]["content"]
        return {
            "thought": "先验证已经写入的实现",
            "actions": [{"tool": "run_command", "cmd": "python -m py_compile app.py"}],
            "done": True,
            "summary": "语法检查通过",
        }


def test_coder_requires_validation_before_rewriting_same_file_again(tmp_path) -> None:
    """重复整体生成必须被验证门禁拦截，避免模型持续覆盖已完成文件。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = RewriteCheckClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建应用并验证",
            "repo": {"files": [], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert client.calls == 3
    assert (tmp_path / "project" / "app.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert result["commands"] == ["python -m py_compile app.py"]


class EmptyCommandOutputClient:
    """模拟验证命令成功但没有 stdout，下一轮应能直接确认完成。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {
                "thought": "运行语法检查",
                "actions": [{"tool": "run_command", "cmd": "python -m py_compile app.py"}],
                "done": False,
            }
        content = messages[-1]["content"]
        assert "工具结果 [成功] run_command" in content
        assert "退出码=0；命令无输出，已正常结束" in content
        return {"thought": "验证已经通过", "actions": [], "done": True, "summary": "语法检查通过"}


def test_coder_exposes_empty_successful_command_result_to_next_step(tmp_path) -> None:
    """无输出的成功命令必须作为明确事实进入下一步，防止重复验证空转。"""

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = EmptyCommandOutputClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "验证应用",
            "repo": {"files": ["app.py"], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert client.calls == 2


def test_observation_brief_marks_command_failure_explicitly() -> None:
    """失败命令同样要携带退出码和错误摘要。"""

    brief = _observation_brief(
        {"tool": "run_command", "cmd": "pytest -q"},
        {"ok": False, "result": {"code": 1, "out": "", "err": "1 failed"}},
    )

    assert brief == "工具结果 [失败] run_command：pytest -q。退出码=1；1 failed"


class MiddleTargetClient:
    """模拟待修改代码位于较长文件中部，验证快照不会把它裁掉。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "读取长文件", "actions": [{"tool": "read_file", "path": "app.js"}], "done": False}
        content = messages[-1]["content"]
        assert "const target = 'OLD_MIDDLE';" in content
        return {
            "thought": "精确修改文件中部",
            "actions": [
                {
                    "tool": "replace_file",
                    "path": "app.js",
                    "old": "const target = 'OLD_MIDDLE';",
                    "new": "const target = 'NEW_MIDDLE';",
                }
            ],
            "done": True,
            "summary": "中部代码已修改",
        }


def test_coder_keeps_middle_of_normal_sized_source_in_snapshot(tmp_path) -> None:
    """常见大小源码必须完整进入下一步，避免两个模型都因缺少中部代码反复读取。"""

    project = tmp_path / "project"
    project.mkdir()
    source = "a" * 6_000 + "\nconst target = 'OLD_MIDDLE';\n" + "b" * 5_000
    (project / "app.js").write_text(source, encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = MiddleTargetClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "修改文件中部",
            "repo": {"files": ["app.js"], "snippets": {}, "stack": ["JavaScript"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert "NEW_MIDDLE" in (project / "app.js").read_text(encoding="utf-8")


class ConsecutiveMutationClient:
    """模拟同一文件连续两次精确修改，第二次不再重复读取文件。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "读取当前文件", "actions": [{"tool": "read_file", "path": "app.py"}], "done": False}
        if self.calls == 2:
            return {
                "thought": "完成第一处修改",
                "actions": [{"tool": "replace_file", "path": "app.py", "old": "VALUE = 1", "new": "VALUE = 2"}],
                "done": False,
            }

        prompt = messages[-1]["content"]
        assert "已读取且当前可直接修改的文件：app.py" in prompt
        assert "VALUE = 2" in prompt
        assert "已自动刷新修改后的当前文件：app.py" in prompt
        return {
            "thought": "基于最新磁盘快照继续修改",
            "actions": [{"tool": "replace_file", "path": "app.py", "old": "VALUE = 2", "new": "VALUE = 3"}],
            "done": True,
            "summary": "连续修改完成",
        }


def test_coder_refreshes_snapshot_after_mutation_without_repeated_read(tmp_path) -> None:
    """写入后的磁盘快照应由编排器刷新，避免模型再次读取并放大上下文。"""

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = ConsecutiveMutationClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "连续修改配置",
            "repo": {"files": ["app.py"], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert client.calls == 3
    assert result["coding_ok"] is True
    assert (project / "app.py").read_text(encoding="utf-8") == "VALUE = 3\n"


class EmptyJsonFallbackClient:
    """模拟 LongCat 在 JSON 模式返回空对象，普通文本兼容模式恢复工具动作。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1, *, plain_text=False):
        self.calls += 1
        if not plain_text:
            return {}
        return {
            "thought": "改用普通文本工具协议",
            "actions": [{"tool": "write_file", "path": "app.py", "content": "VALUE = 1\n"}],
            "done": True,
            "summary": "已恢复执行",
        }


def test_coder_falls_back_after_two_empty_json_responses(tmp_path) -> None:
    """连续空 JSON 必须切换协议，而不是静默消耗完整 ReAct 步数。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = EmptyJsonFallbackClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建入口",
            "repo": {"files": [], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert client.calls == 3
    assert (tmp_path / "project" / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


class LongCatNativeClient:
    """模拟 LongCat，验证 Coding 首次调用就使用普通文本工具协议。"""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(model="LongCat-2.0")
        self.calls: list[bool] = []
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1, *, plain_text=False):
        self.calls.append(plain_text)
        assert "<longcat_tool_call>" in messages[0]["content"]
        assert "绝对不要只返回 `{}`" in messages[0]["content"]
        return {
            "thought": "直接使用原生工具协议",
            "actions": [{"tool": "write_file", "path": "app.py", "content": "VALUE = 1\n"}],
            "done": True,
            "summary": "已完成",
        }


def test_coder_uses_plain_text_protocol_for_longcat_immediately(tmp_path) -> None:
    """LongCat 不应先空耗两次强制 JSON 请求才切换到稳定协议。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = LongCatNativeClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建入口",
            "repo": {"files": [], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert client.calls == [True]


def test_normalize_actions_expands_initial_source_read(tmp_path) -> None:
    """首段源码读取至少为 24K，防止常见单文件因截断陷入重复读取。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))

    actions = graph._normalize_actions(
        [{"tool": "read_file", "path": "static/app.js", "start": 0, "max_chars": 12000}]
    )

    assert actions == [{"tool": "read_file", "path": "static/app.js", "start": 0, "max_chars": 24000}]


class LongCatTimeoutClient:
    """模拟 LongCat 首次生成超时，验证下一轮得到缩小动作的恢复指令。"""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(model="LongCat-2.0")
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1, *, plain_text=False):
        self.calls += 1
        if self.calls == 1:
            raise LlmError("模型响应超时：The read operation timed out")
        prompt = messages[-1]["content"]
        assert "下一轮只返回一个小动作" in prompt
        assert "LongCat 原生工具标签" in prompt
        return {
            "thought": "缩小为单个动作",
            "actions": [{"tool": "write_file", "path": "app.py", "content": "VALUE = 1\n"}],
            "done": True,
            "summary": "超时后恢复完成",
        }


def test_coder_recovers_from_longcat_timeout_with_smaller_action(tmp_path) -> None:
    """LongCat 超时后不能错误要求 JSON，也不能原样重放大批写入。"""

    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = LongCatTimeoutClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建入口",
            "repo": {"files": [], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert client.calls == 2


class TimeoutSnapshotClient:
    """模拟读取目标和无关文件后超时，确认下一轮只携带任务点名的目标快照。"""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(model="LongCat-2.0")
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1, *, plain_text=False):
        self.calls += 1
        if self.calls == 1:
            return {
                "thought": "读取相关文件",
                "actions": [
                    {"tool": "read_file", "path": "static/app.js"},
                    {"tool": "read_file", "path": "main.py"},
                ],
                "done": False,
            }
        if self.calls == 2:
            raise LlmError("模型流式响应超过总时长上限")

        prompt = messages[-1]["content"]
        assert "[static/app.js 当前内容]" in prompt
        assert "[main.py 当前内容]" not in prompt
        assert "模型超时后已收缩文件上下文，仅保留：static/app.js" in prompt
        return {"thought": "目标上下文已聚焦", "actions": [], "done": True, "summary": "完成"}


def test_coder_prunes_unmentioned_snapshots_after_timeout(tmp_path) -> None:
    """超时恢复不能把已读过的所有长文件原样重放给模型。"""

    project = tmp_path / "project"
    (project / "static").mkdir(parents=True)
    (project / "static" / "app.js").write_text("const ready = true;\n", encoding="utf-8")
    (project / "main.py").write_text("READY = True\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = TimeoutSnapshotClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "只修改 static/app.js",
            "repo": {"files": ["static/app.js", "main.py"], "snippets": {}, "stack": ["JavaScript", "Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert client.calls == 3
    assert result["coding_ok"] is True


class RerunAfterFixClient:
    """模拟测试失败、修复文件、再执行完全相同测试命令。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "读取文件", "actions": [{"tool": "read_file", "path": "app.py"}], "done": False}
        if self.calls == 2:
            return {
                "thought": "先运行检查",
                "actions": [{"tool": "run_command", "cmd": "python -m py_compile app.py"}],
                "done": False,
            }
        if self.calls == 3:
            return {
                "thought": "根据失败结果修复",
                "actions": [{"tool": "replace_file", "path": "app.py", "old": "VALUE =", "new": "VALUE = 1"}],
                "done": False,
            }
        return {
            "thought": "重跑同一检查",
            "actions": [{"tool": "run_command", "cmd": "python -m py_compile app.py"}],
            "done": True,
            "summary": "修复并验证完成",
        }


def test_coder_allows_same_verification_command_after_file_changes(tmp_path) -> None:
    """文件变更后必须允许重跑原失败命令，否则 ReAct 无法闭环。"""

    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("VALUE =\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = RerunAfterFixClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "修复语法错误",
            "repo": {"files": ["app.py"], "snippets": {}, "stack": ["Python"]},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert result["commands"] == ["python -m py_compile app.py", "python -m py_compile app.py"]


class EmptyPlaceholderClient:
    """模拟初始化工具创建空 README 后，模型读取并写入文档。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "读取占位文件", "actions": [{"tool": "read_file", "path": "README.md"}], "done": False}
        return {
            "thought": "写入项目说明",
            "actions": [{"tool": "write_file", "path": "README.md", "content": "# 项目说明\n"}],
            "done": True,
            "summary": "文档完成",
        }


def test_coder_can_fill_read_empty_placeholder_file(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    graph._client = lambda agent, state: EmptyPlaceholderClient()

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "创建说明文档",
            "repo": {"files": ["README.md"], "snippets": {}, "stack": []},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert (project / "README.md").read_text(encoding="utf-8") == "# 项目说明\n"


def test_failed_test_brief_keeps_issues_without_full_traceback() -> None:
    """验证反馈应保留可修复问题，但不能把超长 traceback 每步重复发送。"""

    huge_traceback = "trace-start\n" + ("stack-line\n" * 2_000) + "trace-end"
    brief = _test_brief(
        [
            {"cmd": "pytest -q", "ok": False, "out": huge_traceback},
            {
                "cmd": "需求实现审查",
                "ok": False,
                "issues": ["server/app.py 与 static/js/app.js 的返回结构不一致"],
            },
        ]
    )

    assert "trace-start" in brief
    assert "trace-end" in brief
    assert "server/app.py" in brief
    assert len(brief) < 3_000


class RepairContextClient:
    """捕获修复轮次提示，确认模型始终看到当前文件和精简失败清单。"""

    def __init__(self) -> None:
        self.last_usage = SimpleNamespace(total=0)
        self.prompt = ""

    def chat_json(self, messages, temperature=0.1):
        self.prompt = messages[-1]["content"]
        return {"thought": "当前文件已满足要求", "actions": [], "done": True, "summary": "修复完成"}


def test_coder_repair_prompt_is_compact_and_grounded_in_current_files(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "server.py").write_text("BROKEN = True\n", encoding="utf-8")
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = RepairContextClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(project),
            "text": "修复服务",
            "repo": {"files": ["server.py"], "snippets": {"server.py": "旧快照"}, "stack": ["Python"]},
            "changes": ["server.py"],
            "commands": [],
            "tests": [
                {
                    "cmd": "pytest -q",
                    "ok": False,
                    "out": "server.py 运行失败\n" + ("无关调用栈\n" * 3_000),
                }
            ],
            "tests_ok": False,
            "retry": 1,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert "当前模式：失败修复" in client.prompt
    assert "当前磁盘文件：server.py" in client.prompt
    assert "[server.py 当前内容]" in client.prompt
    assert "BROKEN = True" in client.prompt
    assert "旧快照" not in client.prompt
    assert len(client.prompt) < 35_000


class TooManyActionsClient:
    """模拟模型一次返回过多写入动作。"""

    def __init__(self) -> None:
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        return {
            "thought": "一次创建多个文件",
            "actions": [
                {"tool": "write_file", "path": f"file_{index}.py", "content": f"VALUE = {index}\n"}
                for index in range(5)
            ],
            "done": True,
            "summary": "已完成写入",
        }


def test_coder_executes_at_most_three_actions_per_step(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    graph._client = lambda agent, state: TooManyActionsClient()

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建文件",
            "repo": {"files": [], "snippets": {}, "stack": []},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert result["coding_ok"] is True
    assert result["changes"] == ["file_0.py", "file_1.py", "file_2.py"]
    assert not (tmp_path / "project" / "file_3.py").exists()


class ShellScanClient:
    """模拟模型用 shell 绕过 list_files 限制后改为直接写入。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls == 1:
            return {"thought": "再次扫描", "actions": [{"tool": "run_command", "cmd": "find . -maxdepth 2"}], "done": False}
        assert "拒绝使用 shell 重复枚举仓库" in messages[-1]["content"]
        return {
            "thought": "直接实现",
            "actions": [{"tool": "write_file", "path": "main.py", "content": "print('ok')\n"}],
            "done": True,
            "summary": "完成",
        }


def test_coder_rejects_shell_repository_scan(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = ShellScanClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建应用",
            "repo": {"files": [], "snippets": {}, "stack": []},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert _is_discovery_command("ls -la") is True
    assert _is_discovery_command("uv run pytest -q") is False
    assert result["coding_ok"] is True
    assert result["commands"] == []


class MixedBatchClient:
    """模拟一批中先成功写入、再失败修改，随后继续完成修复。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = SimpleNamespace(total=0)

    def chat_json(self, messages, temperature=0.1):
        self.calls += 1
        if self.calls <= 4:
            return {
                "thought": "持续分批修复",
                "actions": [
                    {"tool": "write_file", "path": f"part_{self.calls}.py", "content": "VALUE = 1\n"},
                    {"tool": "replace_file", "path": "missing.py", "old": "x", "new": "y"},
                ],
                "done": False,
            }
        return {"thought": "修复完成", "actions": [], "done": True, "summary": "完成"}


def test_coder_does_not_count_mixed_success_batch_as_idle(tmp_path) -> None:
    graph = AgentGraph(ModelStore(), MemoryStore(tmp_path / "mem"))
    client = MixedBatchClient()
    graph._client = lambda agent, state: client

    result = coder(
        graph,
        {
            "session_id": "s1",
            "workdir": str(tmp_path / "project"),
            "text": "创建多个模块",
            "repo": {"files": [], "snippets": {}, "stack": []},
            "changes": [],
            "commands": [],
            "tests": [],
            "tests_ok": True,
            "retry": 0,
            "tokens": 0,
        },
    )

    assert client.calls == 5
    assert result["coding_ok"] is True
    assert len(result["changes"]) == 4
