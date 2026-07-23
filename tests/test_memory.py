from __future__ import annotations

import json
import re

from api.schema import HistoryResponse
from backend.memory import MemoryStore


def test_memory_detects_interrupted_session(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "coder", "react", "tool", "写入文件")

    assert store.interrupted(workdir, "s1") is True

    store.append(workdir, "s1", "manager", "final", "result", "完成")

    assert store.interrupted(workdir, "s1") is False


def test_memory_compresses_large_session(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    for idx in range(80):
        store.append(workdir, "s1", "coder", "shell", "tool", "输出" * 80, {"idx": idx})

    store.maybe_compress(workdir, "s1", ctx=100)

    records = store.read_session(workdir, "s1")
    assert records[0]["k"] == "summary"
    assert len(records) < 80


def test_memory_lists_history_messages(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "manager", "session", "start", "新会话：测试会话")
    store.rename_session(workdir, "s1", "测试会话")
    store.append(workdir, "s1", "user", "input", "message", "你好")
    store.append(
        workdir,
        "s1",
        "manager",
        "final",
        "result",
        "你好，我在。",
        {"summary": "你好，我在。", "files": [], "tests": [], "plan_path": None, "doc_path": None},
    )

    history = store.list_history()

    assert history[0]["name"] == "proj"
    assert history[0]["sessions"][0]["title"] == "测试会话"
    assert history[0]["sessions"][0]["messages"] == [
        {"id": "3", "role": "user", "content": "你好"},
        {"id": "4", "role": "agent", "content": "你好，我在。"},
    ]


def test_memory_history_restores_all_session_events(tmp_path) -> None:
    """恢复会话时应按顺序返回整个会话的全部事件。"""

    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "manager", "session", "start", "新会话：s1")
    store.append(workdir, "s1", "manager", "run", "start", "第一轮开始")
    old = store.append(workdir, "s1", "coder", "event", "tool", "旧事件", {"tokens": 12})
    store.append(workdir, "s1", "manager", "run", "done", "第一轮结束")
    store.append(workdir, "s1", "manager", "run", "start", "第二轮开始")
    latest = store.append(workdir, "s1", "verifier", "event", "test", "最新验证通过", {"tokens": 34})
    store.append(workdir, "s1", "manager", "run", "done", "第二轮结束")

    events = store.list_history()[0]["sessions"][0]["events"]
    response = HistoryResponse(projects=store.list_history())

    assert events == [
        {
            "id": 1,
            "ts": old["ts"],
            "agent": "coder",
            "kind": "tool",
            "msg": "旧事件",
            "tokens": 12,
            "data": {},
        },
        {
            "id": 2,
            "ts": latest["ts"],
            "agent": "verifier",
            "kind": "test",
            "msg": "最新验证通过",
            "tokens": 34,
            "data": {},
        }
    ]
    assert [event.msg for event in response.projects[0].sessions[0].events] == ["旧事件", "最新验证通过"]


def test_memory_uses_session_id_without_custom_title(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "manager", "session", "start", "新会话：默认名称")

    history = store.list_history()

    assert history[0]["sessions"][0]["title"] == "s1"


def test_memory_deletes_session_and_project(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "manager", "session", "start", "新会话：s1")
    store.append(workdir, "s2", "manager", "session", "start", "新会话：s2")

    assert store.delete_session(workdir, "s1") is True
    assert store.read_session(workdir, "s1") == []
    assert len(store.list_history()[0]["sessions"]) == 1

    assert store.delete_project(workdir) is True
    assert store.list_history() == []


def test_memory_uses_short_timestamp_and_migrates_old_value(tmp_path) -> None:
    root = tmp_path / "mem"
    workdir = str(tmp_path / "proj")
    store = MemoryStore(root)
    record = store.append(workdir, "s1", "user", "input", "message", "第一条消息")

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["ts"])

    path = store.session_path(workdir, "s1")
    old = {**record, "ts": "2026-07-18T10:20:30.123456+00:00"}
    path.write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")

    migrated = MemoryStore(root).read_session(workdir, "s1")
    assert migrated[0]["ts"] == "2026-07-18T10:20:30Z"


def test_memory_context_keeps_dialogue_and_drops_events(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "user", "input", "message", "先修改接口")
    store.append(workdir, "s1", "coder", "event", "tool", "很长的工具输出")
    store.append(workdir, "s1", "manager", "final", "result", "接口已修改", {"summary": "接口已修改"})
    store.append(workdir, "s1", "user", "input", "message", "继续完善")

    context = store.conversation_context(workdir, "s1", exclude_latest_user=True)

    assert context == ["用户：先修改接口", "Agent：接口已修改"]
    assert all("工具输出" not in item for item in context)


def test_memory_run_lifecycle_ignores_session_rename(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "manager", "run", "start", "开始")
    assert store.interrupted(workdir, "s1") is True

    store.append(workdir, "s1", "manager", "run", "done", "结束")
    store.rename_session(workdir, "s1", "新名称")

    assert store.interrupted(workdir, "s1") is False


def test_long_term_memory_deduplicates_content_across_writes(tmp_path) -> None:
    """同一条长期偏好重复写入时只保留一份正文。"""

    store = MemoryStore(tmp_path / "memory")
    workdir = str(tmp_path / "project")

    store.remember(workdir, "默认使用 uv")
    store.remember(workdir, "默认使用 uv")

    assert store.project_memory(workdir).count("默认使用 uv") == 1
