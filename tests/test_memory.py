from __future__ import annotations

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
