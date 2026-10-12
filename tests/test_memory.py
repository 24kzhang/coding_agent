from __future__ import annotations

import json
import re
from io import BytesIO

from PIL import Image

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


def test_memory_restores_runtime_guidance_as_user_message(tmp_path) -> None:
    store = MemoryStore(tmp_path / "mem")
    workdir = str(tmp_path / "proj")
    store.append(workdir, "s1", "manager", "session", "start", "新会话：s1")
    store.append(workdir, "s1", "user", "input", "message", "实现首页")
    store.append(workdir, "s1", "manager", "run", "start", "本轮开始")
    store.append(workdir, "s1", "user", "guidance", "message", "改成蓝色主题")
    store.append(workdir, "s1", "manager", "run", "done", "本轮结束")

    messages = store.list_history()[0]["sessions"][0]["messages"]

    assert [item["content"] for item in messages] == ["实现首页", "改成蓝色主题"]
    assert store.interrupted(workdir, "s1") is False


def test_history_restores_image_metadata_after_restart(tmp_path) -> None:
    """原始附件在新进程中可恢复；无效编号不能暴露文件路径或破坏历史。"""

    root = tmp_path / "mem"
    workdir = str(tmp_path / "proj")
    store = MemoryStore(root)
    stream = BytesIO()
    Image.new("RGB", (16, 16), "white").save(stream, format="PNG")
    info = store.save_file(workdir, "s1", "参考图.png", "image/png", stream.getvalue())
    store.append(workdir, "s1", "user", "input", "message", "根据参考图制作页面", {"attachments": [info["id"], "../private"]})
    message = MemoryStore(root).list_history()[0]["sessions"][0]["messages"][0]
    assert message["content"] == "根据参考图制作页面"
    assert message["attachments"] == [{"id": info["id"], "name": "参考图.png", "mime": "image/png", "size": info["size"], "url": f"/api/sessions/s1/files/{info['id']}"}]
    assert "path" not in message["attachments"][0]
    store.delete_file(workdir, "s1", info["id"])
    assert "attachments" not in MemoryStore(root).list_history()[0]["sessions"][0]["messages"][0]


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


def test_history_survives_compression_and_process_restart(tmp_path) -> None:
    """压缩模型上下文后，界面仍应恢复全部消息和结构化事件。"""

    root = tmp_path / "mem"
    workdir = str(tmp_path / "proj")
    store = MemoryStore(root)
    for index in range(90):
        store.append(workdir, "s1", "user", "input", "message", f"需求 {index}")
        store.append(
            workdir, "s1", "verifier", "event", "test", "测试通过" * 80,
            {"tokens": index, "data": {"ok": True, "cmd": f"检查 {index}", "out": "通过"}},
        )
        store.append(workdir, "s1", "manager", "final", "result", f"完成 {index}")
    store.maybe_compress(workdir, "s1", ctx=100)
    assert len(store.read_session(workdir, "s1")) < 270

    session = MemoryStore(root).list_history()[0]["sessions"][0]

    assert len(session["messages"]) == 180
    assert len(session["events"]) == 90
    assert [event["data"]["cmd"] for event in session["events"]] == [f"检查 {index}" for index in range(90)]
    assert [event["tokens"] for event in session["events"]] == list(range(90))
    assert session["interrupted"] is False


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


def test_memory_separates_agent_task_and_chat_logs(tmp_path) -> None:
    """每个 Agent 的过程、交接合同和用户对话必须物理分离。"""

    store = MemoryStore(tmp_path / "memory" / "data")
    workdir = str(tmp_path / "project")
    store.append(workdir, "s1", "user", "input", "message", "开始")
    store.append(workdir, "s1", "coder", "event", "tool", "读取文件")
    store.append_task(workdir, "s1", "manager", "handoff", "派发", {"id": "t1"})
    session = store.session_dir(workdir, "s1")

    assert "开始" in (session / "chat.jsonl").read_text(encoding="utf-8")
    assert "读取文件" in (session / "agents" / "coder" / "log.jsonl").read_text(encoding="utf-8")
    assert "派发" in (session / "tasks.jsonl").read_text(encoding="utf-8")
    assert "读取文件" not in (session / "chat.jsonl").read_text(encoding="utf-8")


def test_memory_archives_legacy_jsonl_without_loading_it(tmp_path) -> None:
    """旧单文件会话只移动和记录哈希，不得成为活动会话。"""

    root = tmp_path / "memory" / "data"
    old = root / "projects" / "p1" / "sessions" / "old.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text('{"id":1,"out":"旧过程"}\n', encoding="utf-8")

    store = MemoryStore(root)
    manifests = list((root.parent / "legacy").glob("*/index.json"))

    assert not old.exists()
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["files"][0]["sha256"]
    assert store.find_session_workdir("old") == ""


def test_memory_validates_and_hashes_real_image(tmp_path) -> None:
    """附件以内容哈希保存，日志不包含 base64。"""

    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    png = buffer.getvalue()
    store = MemoryStore(tmp_path / "memory" / "data")
    workdir = str(tmp_path / "project")
    info = store.save_file(workdir, "s1", "参考图.png", "image/png", png)

    assert info["width"] == info["height"] == 1
    assert re.fullmatch(r"[0-9a-f]{64}", info["id"])
    assert "base64" not in json.dumps(store.read_session(workdir, "s1"), ensure_ascii=False)


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
