"""真实模型两轮流式对话验证（不打印任何凭据）。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
PORT = 8124
BASE = f"http://127.0.0.1:{PORT}"


def wait_ready(timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=10) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("服务未就绪")


def chat(session_id: str, exercise_id: str, message: str) -> tuple[list[str], list[str], list[str]]:
    payload = json.dumps(
        {"session_id": session_id, "exercise_id": exercise_id, "message": message},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deltas: list[str] = []
    events: list[str] = []
    errors: list[str] = []
    with urllib.request.urlopen(req, timeout=180) as resp:
        assert resp.headers.get("content-type", "").startswith("text/event-stream")
        event = "message"
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line[6:].strip()
                events.append(event)
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event == "delta":
                    deltas.append(data.get("text", ""))
                elif event == "error":
                    errors.append(data.get("message", ""))
    return deltas, events, errors


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=APP,
    )
    try:
        wait_ready()
        sid = "live-check-1"
        d1, e1, err1 = chat(sid, "0001", "这个动作主要练哪些肌肉？")
        text1 = "".join(d1)
        assert not err1, err1
        assert "delta" in e1 and e1[-1] == "done", e1
        assert text1.strip(), "第一轮回复为空"
        print("第一轮：事件", e1[0], "→", e1[-1], "增量片段", len(d1), "文本长度", len(text1))

        d2, e2, err2 = chat(sid, "0001", "那新手要注意什么？")
        text2 = "".join(d2)
        assert not err2, err2
        assert "delta" in e2 and e2[-1] == "done", e2
        assert text2.strip(), "第二轮回复为空"
        print("第二轮：事件", e2[0], "→", e2[-1], "增量片段", len(d2), "文本长度", len(text2))
        print("真实模型两轮流式对话通过")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
