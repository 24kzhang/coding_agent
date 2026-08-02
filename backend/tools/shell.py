from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any


def prepare_command(
    root: Path,
    cmd: str,
    *,
    resolve_executable: bool,
    workspace: str | None = None,
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
    if program_name == "git" and len(argv) > 1 and argv[1].lower() in ShellTool.dangerous_git:
        return {"ok": False, "code": 126, "out": "", "err": f"高风险 Git 命令需要用户确认：{clean}", "cmd": clean, "argv": argv}
    if program_name == "git" and len(argv) > 1 and argv[1].lower() != "init" and not (root / ".git").is_dir():
        return {
            "ok": False,
            "code": 126,
            "out": "",
            "err": "当前项目没有独立 .git 目录，已阻止 Git 命令访问父级仓库；需要版本管理时先执行 git init",
            "cmd": clean,
            "argv": argv,
        }

    interpreter_names = {"node", "python", "python3", "ruby", "perl"}
    has_inline_script = program_name in interpreter_names and any(arg in {"-c", "-e"} for arg in argv[1:])
    if not has_inline_script:
        has_inline_script = any(
            Path(argv[index]).name.lower() in interpreter_names and argv[index + 1] in {"-c", "-e"}
            for index in range(len(argv) - 1)
        )
    if has_inline_script:
        return {"ok": False, "code": 126, "out": "", "err": "不允许执行解释器内联代码，请写入项目脚本后运行", "cmd": clean, "argv": argv}

    container_root = Path(workspace).resolve() if workspace else None
    for index, arg in enumerate(argv[1:], start=1):
        if arg.startswith(("http://", "https://")) or arg.startswith("-"):
            continue
        if ".." in Path(arg).parts:
            return {"ok": False, "code": 126, "out": "", "err": f"命令参数越出项目目录：{arg}", "cmd": clean, "argv": argv}
        arg_path = Path(arg).expanduser()
        if not arg_path.is_absolute():
            continue
        resolved_arg = arg_path.resolve()
        if root == resolved_arg or root in resolved_arg.parents:
            if container_root is not None:
                argv[index] = str(container_root / resolved_arg.relative_to(root))
            continue
        if container_root is not None and (container_root == resolved_arg or container_root in resolved_arg.parents):
            continue
        return {"ok": False, "code": 126, "out": "", "err": f"命令参数越出项目目录：{arg}", "cmd": clean, "argv": argv}

    if resolve_executable:
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
        # 下面这些命令可能删除文件、格式化磁盘或破坏仓库状态，当前版本直接拒绝执行。
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
    # dangerous_git 会覆盖、删除或切换工作区内容，必须由用户在 agent 外明确执行。
    dangerous_git = {"checkout", "clean", "reset", "restore", "rm"}

    def __init__(self, workdir: str):
        # root 是命令执行目录，所有命令都在用户选择的项目目录内运行。
        self.root = Path(workdir).resolve()

    def run(self, cmd: str, timeout: int = 180) -> dict[str, Any]:
        """在项目目录内执行命令，并返回结构化 stdout/stderr/退出码。"""

        prepared = prepare_command(self.root, cmd, resolve_executable=True)
        if isinstance(prepared, dict):
            return prepared
        clean, argv = prepared
        # env 复制当前环境，保留用户已有 PATH 等配置。
        env = os.environ.copy()
        # Agent 自身虚拟环境不应污染另一个所选项目；仅保留位于当前项目内的 VIRTUAL_ENV。
        active_venv = env.get("VIRTUAL_ENV")
        if active_venv:
            venv_path = Path(active_venv).expanduser().resolve()
            if self.root != venv_path and self.root not in venv_path.parents:
                env.pop("VIRTUAL_ENV", None)
        # PYTHONUTF8 让 Python 子进程默认使用 UTF-8，减少中文输出乱码。
        env.setdefault("PYTHONUTF8", "1")
        # CI 避免部分测试运行器进入交互观察模式。
        env.setdefault("CI", "1")
        # uv/npm 的用户级缓存可能不在 Agent 沙箱写入范围内。为每个所选项目分配 /tmp 下的
        # 稳定缓存，既避免权限错误，也不把依赖缓存写入用户项目并污染文件扫描。
        cache_key = sha256(str(self.root).encode("utf-8")).hexdigest()[:16]
        cache_root = Path(tempfile.gettempdir()) / "coding-agent-cache" / cache_key
        env["UV_CACHE_DIR"] = str(cache_root / "uv")
        env["NPM_CONFIG_CACHE"] = str(cache_root / "npm")
        try:
            # proc 不使用 shell，模型输入不会被二次解释为重定向或命令替换。
            proc = subprocess.run(
                argv,
                cwd=self.root,
                shell=False,
                text=True,
                capture_output=True,
                timeout=max(1, min(int(timeout), 900)),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            # stdout/stderr 在超时时可能是 bytes 或 str，统一转换后截断。
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
            return {"ok": False, "code": 124, "out": stdout[-12000:], "err": (stderr + "\n命令执行超时")[-12000:], "cmd": clean, "argv": argv}
        except OSError as exc:
            return {"ok": False, "code": 127, "out": "", "err": f"命令启动失败：{exc}", "cmd": clean, "argv": argv}
        return {
            # ok 表示命令是否以 0 退出码结束。
            "ok": proc.returncode == 0,
            # code 是原始退出码，方便排查具体失败类型。
            "code": proc.returncode,
            # out 只保留最后 12000 字符，避免长日志撑爆上下文。
            "out": proc.stdout[-12000:],
            # err 同样只保留最后 12000 字符。
            "err": proc.stderr[-12000:],
            # cmd 保留原始命令字符串，方便最终结果和 memory 展示。
            "cmd": clean,
            # argv 是按平台规则拆出的参数列表，便于后续审计或测试。
            "argv": argv,
        }
