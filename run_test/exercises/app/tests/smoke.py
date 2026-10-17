"""真实启动烟测：启动 uvicorn 并请求关键接口。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"


def get(path: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=20) as resp:
            return resp.status, resp.read(), resp.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("content-type", "")


def wait_ready(timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _, _ = get("/api/health")
            if status == 200:
                return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("服务未在预期时间内就绪")


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=APP,
    )
    try:
        wait_ready()
        status, body, ctype = get("/")
        assert status == 200 and "text/html" in ctype, (status, ctype)
        assert "动作库" in body.decode("utf-8")

        status, body, _ = get("/api/exercises")
        data = json.loads(body)
        assert status == 200 and data["total"] == 1324 and data["count"] == 1324
        print("列表接口:", data["total"], "条")

        status, body, _ = get("/api/exercises?category=%E8%85%B0%E9%83%A8&target=%E8%85%B9%E8%82%8C&equipment=%E8%87%AA%E9%87%8D")
        filtered = json.loads(body)
        assert status == 200 and 0 < filtered["count"] < data["total"]
        print("组合筛选:", filtered["count"], "条")

        status, body, _ = get("/api/exercises/0001")
        detail = json.loads(body)
        assert status == 200 and detail["name"] and detail["instructions"]["zh"]
        assert detail["gif"].endswith(".gif")
        print("详情接口:", detail["id"], detail["name"])

        status, body, ctype = get(f"/media/{detail['gif']}")
        assert status == 200 and ctype == "image/gif" and body[:3] == b"GIF"
        print("媒体接口:", detail["gif"], len(body), "字节")

        status, _, _ = get("/api/exercises/9999")
        assert status == 404
        print("未知动作 404 正常")
        print("烟测通过")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
