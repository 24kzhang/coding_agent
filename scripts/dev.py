from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ROOT 是本 agent 项目根目录，后端和前端启动都以它为基准定位文件。
ROOT = Path(__file__).resolve().parents[1]


def pick_port(start: int, end: int) -> int:
    """从端口范围中挑选第一个可用端口，避免固定端口被占用。"""
    # port 是当前尝试的端口号。
    for port in range(start, end + 1):
        # sock 用于探测 127.0.0.1:port 是否已经有服务监听。
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            # connect_ex 返回非 0 表示连接失败，也就是端口当前可用。
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"端口范围 {start}-{end} 内没有可用端口")


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    """启动子进程并保留输出，让用户能直接看到后端和前端日志。"""
    # cmd 是要启动的命令参数列表，cwd 是子进程工作目录，env 是环境变量。
    return subprocess.Popen(cmd, cwd=cwd, env=env, text=True)


def main() -> None:
    """本地开发启动入口，同时启动 FastAPI 后端和 Vite 前端。"""

    # backend_port 是后端端口，从 8710-8730 中自动选择。
    backend_port = pick_port(8710, 8730)
    # frontend_port 是前端端口，从 5173-5199 中自动选择。
    frontend_port = pick_port(5173, 5199)
    # env 复制当前环境，并注入前后端端口信息。
    env = os.environ.copy()
    env["BACKEND_PORT"] = str(backend_port)
    env["FRONTEND_PORT"] = str(frontend_port)
    # VITE_API_URL 会被 Vite 注入到前端 import.meta.env 中。
    env["VITE_API_URL"] = f"http://127.0.0.1:{backend_port}"

    print(f"后端地址：http://127.0.0.1:{backend_port}")
    print(f"前端地址：http://127.0.0.1:{frontend_port}")

    # npm 是跨平台 npm 命令名，Windows 下需要 npm.cmd。
    npm = "npm.cmd" if os.name == "nt" else "npm"
    # frontend_dir 是前端项目目录。
    frontend_dir = ROOT / "frontend"
    # 首次启动没有 node_modules 时自动安装前端依赖。
    if not (frontend_dir / "node_modules").exists():
        subprocess.check_call([npm, "install"], cwd=frontend_dir, env=env)

    # backend 是 uvicorn 后端进程。
    backend = run(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
        ROOT,
        env,
    )
    # frontend 是 Vite 前端开发服务器进程。
    frontend = run([npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(frontend_port)], frontend_dir, env)

    def stop(_sig: int, _frame: object) -> None:
        """同时关闭前后端，避免双击启动后残留进程。"""
        # 先给前后端进程发送 terminate，让它们有机会正常退出。
        for proc in (frontend, backend):
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        # 仍未退出时强制 kill。
        for proc in (frontend, backend):
            if proc.poll() is None:
                proc.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    # 主循环持续观察子进程状态，任一进程退出都一起收尾。
    while True:
        if backend.poll() is not None:
            stop(signal.SIGTERM, None)
        if frontend.poll() is not None:
            stop(signal.SIGTERM, None)
        time.sleep(1)


if __name__ == "__main__":
    main()
