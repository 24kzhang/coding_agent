from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


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

        # clean 是去除首尾空白后的单条命令；空命令不进入子进程。
        clean = str(cmd).strip()
        if not clean:
            return {"ok": False, "code": 2, "out": "", "err": "命令不能为空", "cmd": clean, "argv": []}
        if len(clean) > 4000:
            return {"ok": False, "code": 2, "out": "", "err": "命令长度超过限制", "cmd": clean, "argv": []}
        # shell_markers 会产生复合命令、重定向或命令替换；ReAct 应拆成多次独立调用。
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
            # argv 是不经过 shell 解释的参数列表，引号只用于参数分组。
            argv = shlex.split(clean, posix=os.name != "nt")
        except ValueError as exc:
            return {"ok": False, "code": 2, "out": "", "err": f"命令参数解析失败：{exc}", "cmd": clean, "argv": []}
        if not argv:
            return {"ok": False, "code": 2, "out": "", "err": "命令不能为空", "cmd": clean, "argv": []}
        # program_name 去掉绝对路径，只按可执行文件名检查危险程序。
        program_name = Path(argv[0]).name.lower()
        if program_name in self.dangerous_programs:
            return {"ok": False, "code": 126, "out": "", "err": f"高风险命令需要用户确认：{clean}", "cmd": clean, "argv": argv}
        if program_name == "git" and len(argv) > 1 and argv[1].lower() in self.dangerous_git:
            return {"ok": False, "code": 126, "out": "", "err": f"高风险 Git 命令需要用户确认：{clean}", "cmd": clean, "argv": argv}
        # 解释器内联代码可以绕过文件和命令工具边界，要求模型改为项目内脚本文件。
        # interpreter_names 同时用于识别直接调用和 `uv run python -c` 这类嵌套调用。
        interpreter_names = {"node", "python", "python3", "ruby", "perl"}
        # has_inline_script 表示参数中出现了“解释器 + -c/-e”组合。
        has_inline_script = program_name in interpreter_names and any(arg in {"-c", "-e"} for arg in argv[1:])
        if not has_inline_script:
            # 遍历相邻参数，防止包管理器或环境工具把内联代码转交给解释器执行。
            has_inline_script = any(
                Path(argv[index]).name.lower() in interpreter_names and argv[index + 1] in {"-c", "-e"}
                for index in range(len(argv) - 1)
            )
        if has_inline_script:
            return {"ok": False, "code": 126, "out": "", "err": "不允许执行解释器内联代码，请写入项目脚本后运行", "cmd": clean, "argv": argv}
        # 参数中的 ../ 或工作目录外绝对路径可能访问用户未选择的区域。
        for arg in argv[1:]:
            if arg.startswith(("http://", "https://")) or arg.startswith("-"):
                continue
            if ".." in Path(arg).parts:
                return {"ok": False, "code": 126, "out": "", "err": f"命令参数越出项目目录：{arg}", "cmd": clean, "argv": argv}
            arg_path = Path(arg).expanduser()
            if arg_path.is_absolute():
                resolved_arg = arg_path.resolve()
                if self.root != resolved_arg and self.root not in resolved_arg.parents:
                    return {"ok": False, "code": 126, "out": "", "err": f"命令参数越出项目目录：{arg}", "cmd": clean, "argv": argv}
        # executable 使用系统 PATH 解析，Windows 下可正确找到 npm.cmd 等入口。
        executable = shutil.which(argv[0])
        if not executable:
            return {"ok": False, "code": 127, "out": "", "err": f"找不到命令：{argv[0]}", "cmd": clean, "argv": argv}
        argv[0] = executable
        # env 复制当前环境，保留用户已有 PATH、虚拟环境等配置。
        env = os.environ.copy()
        # PYTHONUTF8 让 Python 子进程默认使用 UTF-8，减少中文输出乱码。
        env.setdefault("PYTHONUTF8", "1")
        # CI 避免部分测试运行器进入交互观察模式。
        env.setdefault("CI", "1")
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
