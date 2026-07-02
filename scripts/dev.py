from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pick_port(start: int, end: int) -> int:
    """从端口范围中挑选第一个可用端口，避免固定端口被占用。"""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"端口范围 {start}-{end} 内没有可用端口")


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    """启动子进程并保留输出，让用户能直接看到后端和前端日志。"""
    return subprocess.Popen(cmd, cwd=cwd, env=env, text=True)


def main() -> None:
    backend_port = pick_port(8710, 8730)
    frontend_port = pick_port(5173, 5199)
    env = os.environ.copy()
    env["BACKEND_PORT"] = str(backend_port)
    env["FRONTEND_PORT"] = str(frontend_port)
    env["VITE_API_URL"] = f"http://127.0.0.1:{backend_port}"

    print(f"后端地址：http://127.0.0.1:{backend_port}")
    print(f"前端地址：http://127.0.0.1:{frontend_port}")

    npm = "npm.cmd" if os.name == "nt" else "npm"
    frontend_dir = ROOT / "frontend"
    if not (frontend_dir / "node_modules").exists():
        subprocess.check_call([npm, "install"], cwd=frontend_dir, env=env)

    backend = run(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
        ROOT,
        env,
    )
    frontend = run([npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(frontend_port)], frontend_dir, env)

    def stop(_sig: int, _frame: object) -> None:
        """同时关闭前后端，避免双击启动后残留进程。"""
        for proc in (frontend, backend):
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        for proc in (frontend, backend):
            if proc.poll() is None:
                proc.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while True:
        if backend.poll() is not None:
            stop(signal.SIGTERM, None)
        if frontend.poll() is not None:
            stop(signal.SIGTERM, None)
        time.sleep(1)


if __name__ == "__main__":
    main()
