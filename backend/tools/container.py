from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

from backend.tools.shell import prepare_command


class ContainerShellError(RuntimeError):
    """容器 Shell 基础设施无法使用。"""


class ContainerShellTool:
    """在单轮任务共享的 Docker 容器中执行受控命令。"""

    workspace = "/workspace"
    _build_lock = threading.Lock()

    def __init__(self, workdir: str, image: str | None = None):
        self.root = Path(workdir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.docker = shutil.which("docker")
        self.custom_image = str(image or os.environ.get("CODING_AGENT_SHELL_IMAGE") or "").strip()
        self.image = self.custom_image or self._default_image()
        self.container_id = ""
        self.container_name = f"coding-agent-shell-{uuid.uuid4().hex[:12]}"
        self.image_ready = False

    @staticmethod
    def _dockerfile() -> Path:
        return Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"

    def _default_image(self) -> str:
        dockerfile = self._dockerfile()
        try:
            digest = sha256(dockerfile.read_bytes()).hexdigest()[:12]
        except OSError as exc:
            raise ContainerShellError(f"容器 Shell 镜像定义不可读：{exc}") from exc
        return f"coding-agent-shell:{digest}"

    @staticmethod
    def _output(proc: subprocess.CompletedProcess[str]) -> str:
        return (str(proc.stdout or "") + "\n" + str(proc.stderr or "")).strip()[-4000:]

    def _inspect_image(self) -> subprocess.CompletedProcess[str]:
        assert self.docker is not None
        try:
            return subprocess.run(
                [self.docker, "image", "inspect", self.image],
                text=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContainerShellError(f"容器 Shell 镜像检查失败：{exc}") from exc

    def ensure_ready(self) -> None:
        """确认 Docker、执行镜像和本轮容器均可使用。"""

        if self.container_id:
            return
        if not self.docker:
            raise ContainerShellError("容器 Shell 不可用：未找到 Docker CLI，请安装并启动 Docker Desktop")
        try:
            version = subprocess.run(
                [self.docker, "version", "--format", "{{.Server.Version}}"],
                text=True,
                capture_output=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContainerShellError(f"容器 Shell 不可用：Docker Engine 连接失败：{exc}") from exc
        if version.returncode != 0:
            detail = self._output(version)
            raise ContainerShellError(
                "容器 Shell 不可用：Docker Engine 未启动或无法连接" + (f"：{detail}" if detail else "")
            )
        self._ensure_image()
        self._start()

    def _ensure_image(self) -> None:
        if self.image_ready:
            return
        assert self.docker is not None
        inspect = self._inspect_image()
        if inspect.returncode == 0:
            self.image_ready = True
            return
        if self.custom_image:
            raise ContainerShellError(f"指定的容器 Shell 镜像不存在：{self.image}")
        dockerfile = self._dockerfile()
        with self._build_lock:
            inspect = self._inspect_image()
            if inspect.returncode == 0:
                self.image_ready = True
                return
            try:
                build = subprocess.run(
                    [
                        self.docker,
                        "build",
                        "--file",
                        str(dockerfile),
                        "--tag",
                        self.image,
                        str(dockerfile.parent),
                    ],
                    text=True,
                    capture_output=True,
                    timeout=900,
                )
            except subprocess.TimeoutExpired as exc:
                raise ContainerShellError("容器 Shell 镜像构建超时") from exc
            if build.returncode != 0:
                raise ContainerShellError(f"容器 Shell 镜像构建失败：{self._output(build)}")
        self.image_ready = True

    def _start(self) -> None:
        assert self.docker is not None
        uid = os.getuid() if hasattr(os, "getuid") else 1000
        gid = os.getgid() if hasattr(os, "getgid") else 1000
        try:
            start = subprocess.run(
                [
                    self.docker,
                    "run",
                    "--detach",
                    "--rm",
                    "--name",
                    self.container_name,
                    "--label",
                    "coding-agent.shell=true",
                    "--workdir",
                    self.workspace,
                    "--volume",
                    f"{self.root}:{self.workspace}:rw",
                    "--user",
                    f"{uid}:{gid}",
                    "--security-opt",
                    "no-new-privileges:true",
                    "--cap-drop",
                    "ALL",
                    "--pids-limit",
                    "512",
                    "--stop-timeout",
                    "1",
                    "--env",
                    "HOME=/tmp/coding-agent-home",
                    "--env",
                    "CI=1",
                    "--env",
                    "PYTHONUTF8=1",
                    "--env",
                    "UV_CACHE_DIR=/tmp/coding-agent-cache/uv",
                    "--env",
                    "NPM_CONFIG_CACHE=/tmp/coding-agent-cache/npm",
                    self.image,
                    "/bin/sh",
                    "-c",
                    "mkdir -p \"$HOME\" \"$UV_CACHE_DIR\" \"$NPM_CONFIG_CACHE\" && exec sleep infinity",
                ],
                text=True,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._stop(self.container_name)
            raise ContainerShellError(f"容器 Shell 启动失败：{exc}") from exc
        if start.returncode != 0:
            raise ContainerShellError(f"容器 Shell 启动失败：{self._output(start)}")
        self.container_id = str(start.stdout or "").strip()
        if not self.container_id:
            self._stop(self.container_name)
            raise ContainerShellError("容器 Shell 启动失败：Docker 未返回容器 ID")

    def run(self, cmd: str, timeout: int = 180) -> dict[str, Any]:
        prepared = prepare_command(self.root, cmd, resolve_executable=False, workspace=self.workspace)
        if isinstance(prepared, dict):
            return prepared
        clean, argv = prepared
        try:
            self.ensure_ready()
        except ContainerShellError as exc:
            return {
                "ok": False,
                "code": 125,
                "out": "",
                "err": str(exc),
                "cmd": clean,
                "argv": argv,
                "infrastructure": True,
            }
        assert self.docker is not None
        try:
            proc = subprocess.run(
                [self.docker, "exec", "--workdir", self.workspace, self.container_id, *argv],
                text=True,
                capture_output=True,
                timeout=max(1, min(int(timeout), 900)),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
            self.close()
            return {
                "ok": False,
                "code": 124,
                "out": stdout[-12000:],
                "err": (stderr + "\n命令执行超时，已停止任务容器")[-12000:],
                "cmd": clean,
                "argv": argv,
            }
        except OSError as exc:
            return {"ok": False, "code": 127, "out": "", "err": f"命令启动失败：{exc}", "cmd": clean, "argv": argv}
        return {
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "out": proc.stdout[-12000:],
            "err": proc.stderr[-12000:],
            "cmd": clean,
            "argv": argv,
        }

    def close(self) -> None:
        """停止本轮临时容器；重复调用安全。"""

        container_id = self.container_id
        self.container_id = ""
        if not container_id:
            return
        self._stop(container_id)

    def _stop(self, target: str) -> None:
        """按容器 ID 或唯一名称停止容器，供正常清理和启动异常共同使用。"""

        if not target or not self.docker:
            return
        try:
            subprocess.run(
                [self.docker, "stop", "--time", "1", target],
                text=True,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def __enter__(self) -> ContainerShellTool:
        self.ensure_ready()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
