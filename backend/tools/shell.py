# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from backend.tools.fs import FsTool


def prepare_command(
    root: Path,
    cmd: str,
    *,
    resolve_executable: bool,
    workspace: str | None = None,
    cwd: Path | None = None,
) -> tuple[str, list[str]] | dict[str, Any]:
    """解析并校验一条命令；失败时直接返回 ShellTool 的结构化错误。"""

    clean = str(cmd).strip()
    if not clean:
        return {"ok": False, "code": 2, "out": "", "err": "命令不能为空", "cmd": clean, "argv": []}
    if len(clean) > 4000:
        return {"ok": False, "code": 2, "out": "", "err": "命令长度超过限制", "cmd": clean, "argv": []}
    shell_markers = ["&&", "||", ";", "|", ">", "<", "\n", "\r", "`", "$("]
    if any(marker in clean for marker in shell_markers):
        return {
            "ok": False,
            "code": 126,
            "out": "",
            "err": "不允许复合 shell 语法，请把命令拆成独立工具调用",
            "cmd": clean,
            "argv": [],
        }
    try:
        argv = shlex.split(clean, posix=os.name != "nt")
    except ValueError as exc:
        return {"ok": False, "code": 2, "out": "", "err": f"命令参数解析失败：{exc}", "cmd": clean, "argv": []}
    if not argv:
        return {"ok": False, "code": 2, "out": "", "err": "命令不能为空", "cmd": clean, "argv": []}

    program_name = Path(argv[0]).name.lower()
    if program_name == "uv" and len(argv) > 1 and argv[1] == "init" and "--no-workspace" not in argv[2:]:
        argv.insert(2, "--no-workspace")
    if program_name in ShellTool.dangerous_programs:
        return {"ok": False, "code": 126, "out": "", "err": f"高风险命令需要用户确认：{clean}", "cmd": clean, "argv": argv}
    if program_name in {"uv", "env", "timeout", "npx", "npm"} and any(
        Path(arg).name.lower() in ShellTool.dangerous_programs for arg in argv[1:]
    ):
        return {"ok": False, "code": 126, "out": "", "err": "拒绝通过包装器执行高风险程序", "cmd": clean, "argv": argv}
    if program_name == "git" and any(
        arg in {"-C", "-c", "--git-dir", "--work-tree"} or arg.startswith(("--git-dir=", "--work-tree=", "-C", "-c"))
        for arg in argv[1:]
    ):
        return {"ok": False, "code": 126, "out": "", "err": "Git 命令不得重定向仓库或覆盖执行配置", "cmd": clean, "argv": argv}
    if program_name == "git" and any(arg in ShellTool.dangerous_git for arg in argv[1:]):
        return {"ok": False, "code": 126, "out": "", "err": "拒绝通过 Git 全局参数绕过危险子命令检查", "cmd": clean, "argv": argv}
    if program_name == "git" and len(argv) > 1 and argv[1].lower() in ShellTool.dangerous_git:
        return {"ok": False, "code": 126, "out": "", "err": f"高风险 Git 命令需要用户确认：{clean}", "cmd": clean, "argv": argv}
    if program_name == "git" and len(argv) > 1 and argv[1].lower() != "init" and not ((cwd or root) / ".git").is_dir():
        return {
            "ok": False,
            "code": 126,
            "out": "",
            "err": "当前项目没有独立 .git 目录，已阻止 Git 命令访问父级仓库；需要版本管理时先执行 git init",
            "cmd": clean,
            "argv": argv,
        }

    interpreter_names = {"node", "python", "python3", "ruby", "perl", "sh", "bash", "zsh", "fish", "powershell", "pwsh"}
    has_inline_script = program_name in interpreter_names and any(arg in {"-c", "-e"} for arg in argv[1:])
    if not has_inline_script:
        has_inline_script = any(
            Path(argv[index]).name.lower() in interpreter_names and argv[index + 1] in {"-c", "-e"}
            for index in range(len(argv) - 1)
        )
    if has_inline_script:
        return {"ok": False, "code": 126, "out": "", "err": "不允许执行解释器内联代码，请写入项目脚本后运行", "cmd": clean, "argv": argv}

    container_root = Path(workspace).resolve() if workspace else None
    inspectors = {"cat", "head", "tail", "wc", "grep", "rg", "sed", "awk"}
    inspecting = program_name in inspectors or (program_name in {"uv", "env", "timeout", "npx", "npm"} and any(Path(arg).name.lower() in inspectors for arg in argv[1:])) or any(
        argv[index] == "-m" and argv[index + 1] == "json.tool" for index in range(len(argv) - 1)
    )
    for index, arg in enumerate(argv[1:], start=1):
        option = ""
        candidate = arg
        if arg.startswith("-"):
            if "=" not in arg:
                continue
            option, candidate = arg.split("=", 1)
        if candidate.startswith(("http://", "https://")):
            continue
        if ".." in Path(candidate).parts:
            return {"ok": False, "code": 126, "out": "", "err": f"命令参数越出项目目录：{arg}", "cmd": clean, "argv": argv}
        arg_path = Path(candidate).expanduser()
        inspected_path = ((cwd or root) / arg_path).resolve()
        if inspecting and (root == inspected_path or root in inspected_path.parents) and FsTool.sensitive_path(root, inspected_path):
            return {"ok": False, "code": 126, "out": "", "err": "私密配置不能通过命令探查正文；使用 read_file 获取安全元信息或 model_copy 生成配置", "cmd": clean, "argv": argv}
        if not arg_path.is_absolute():
            resolved_relative = ((cwd or root) / arg_path).resolve()
            if root != resolved_relative and root not in resolved_relative.parents:
                return {"ok": False, "code": 126, "out": "", "err": f"命令参数通过链接越出项目目录：{arg}", "cmd": clean, "argv": argv}
            continue
        resolved_arg = arg_path.resolve()
        if root == resolved_arg or root in resolved_arg.parents:
            if container_root is not None:
                mapped = str(container_root / resolved_arg.relative_to(root))
                argv[index] = f"{option}={mapped}" if option else mapped
            continue
        if container_root is not None and (container_root == resolved_arg or container_root in resolved_arg.parents):
            continue
        return {"ok": False, "code": 126, "out": "", "err": f"命令参数越出项目目录：{arg}", "cmd": clean, "argv": argv}

    if resolve_executable:
        if argv[0] in {"python", "python3"}:
            local_python = (cwd or root) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable = str(local_python) if local_python.is_file() else sys.executable
        else:
            executable = shutil.which(argv[0])
        if not executable:
            return {"ok": False, "code": 127, "out": "", "err": f"找不到命令：{argv[0]}", "cmd": clean, "argv": argv}
        argv[0] = executable
    return clean, argv


class ShellTool:
    """跨平台命令执行工具。

    默认拒绝明显高风险命令。真正产品化时可以把确认流程接到前端，
    这里先保证 agent 不会静默执行破坏性操作。
    """

    dangerous_programs = {
        "rm",
        "rmdir",
        "del",
        "erase",
        "format",
        "mkfs",
        "shutdown",
        "reboot",
        "sudo",
        "su",
    }
    dangerous_git = {"checkout", "clean", "reset", "restore", "rm"}

    def __init__(self, workdir: str):
        """初始化当前对象及其运行状态。"""
        self.root = Path(workdir).resolve()

    def run(self, cmd: str, timeout: int = 180, cwd: Path | None = None) -> dict[str, Any]:
        """在项目目录内执行命令，并返回结构化 stdout/stderr/退出码。"""

        active_dir = Path(cwd).resolve() if cwd is not None else self.root
        if self.root != active_dir and self.root not in active_dir.parents:
            return {"ok": False, "code": 126, "out": "", "err": "命令工作目录越出工作区", "cmd": cmd}
        prepared = prepare_command(self.root, cmd, resolve_executable=True, cwd=active_dir)
        if isinstance(prepared, dict):
            return prepared
        clean, argv = prepared
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        active_venv = env.get("VIRTUAL_ENV")
        if active_venv:
            venv_path = Path(active_venv).expanduser().resolve()
            if self.root != venv_path and self.root not in venv_path.parents:
                env.pop("VIRTUAL_ENV", None)
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("CI", "1")
        cache_key = sha256(str(self.root).encode("utf-8")).hexdigest()[:16]
        cache_root = Path(tempfile.gettempdir()) / "coding-agent-cache" / cache_key
        env["UV_CACHE_DIR"] = str(cache_root / "uv")
        env["UV_PYTHON_INSTALL_DIR"] = str(cache_root / "python")
        env["UV_PYTHON_BIN_DIR"] = str(cache_root / "bin")
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(cache_root / "browsers")
        env["NPM_CONFIG_CACHE"] = str(cache_root / "npm")
        started = time.monotonic()
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                proc = subprocess.Popen(
                    argv, cwd=active_dir, shell=False, stdout=stdout, stderr=stderr,
                    env=env, start_new_session=os.name != "nt",
                )
                timed_out = False
                try:
                    code = proc.wait(timeout=max(1, min(int(timeout), 900)))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name != "nt":
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
                    proc.wait()
                    code = 124
            except OSError as exc:
                return {"ok": False, "code": 127, "out": "", "err": f"命令启动失败：{exc}", "cmd": clean, "argv": argv}
            out, out_size = _bounded_output(stdout)
            err, err_size = _bounded_output(stderr)
        if timed_out:
            err += "\n命令执行超时，已清理命令进程组"
        return {
            "ok": code == 0, "code": code, "out": out, "err": err,
            "cmd": clean, "argv": argv, "timed_out": timed_out,
            "cwd": str(active_dir),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "output_bytes": out_size + err_size,
            "output_truncated": out_size > 12_000 or err_size > 12_000,
        }


def _bounded_output(handle, limit: int = 12_000) -> tuple[str, int]:
    """长输出保留开头错误与末尾结论，并明确标记省略区间。"""

    size = handle.seek(0, os.SEEK_END)
    handle.seek(0)
    if size <= limit:
        return handle.read().decode("utf-8", errors="replace"), size
    head = handle.read(limit // 2)
    handle.seek(-limit // 2, os.SEEK_END)
    tail = handle.read(limit // 2)
    return head.decode("utf-8", errors="replace") + "\n[中间命令输出已省略]\n" + tail.decode("utf-8", errors="replace"), size
