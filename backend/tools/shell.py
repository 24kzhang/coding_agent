from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


class ShellTool:
    """跨平台命令执行工具。

    默认拒绝明显高风险命令。真正产品化时可以把确认流程接到前端，
    这里先保证 agent 不会静默执行破坏性操作。
    """

    dangerous = {
        # 下面这些命令可能删除文件、格式化磁盘或破坏仓库状态，当前版本直接拒绝执行。
        "rm",
        "rmdir",
        "del",
        "erase",
        "format",
        "mkfs",
        "shutdown",
        "reboot",
        "git reset",
        "git checkout --",
    }

    def __init__(self, workdir: str):
        # root 是命令执行目录，所有命令都在用户选择的项目目录内运行。
        self.root = Path(workdir).resolve()

    def run(self, cmd: str, timeout: int = 180) -> dict[str, Any]:
        """在项目目录内执行命令，并返回结构化 stdout/stderr/退出码。"""

        # lowered 用于做危险命令前缀匹配，避免大小写差异绕过检查。
        lowered = cmd.strip().lower()
        # 命令以危险前缀开头时直接拒绝；后续产品化可改为前端确认。
        for bad in self.dangerous:
            if lowered.startswith(bad):
                return {"ok": False, "code": 126, "out": "", "err": f"高风险命令需要用户确认：{cmd}"}
        # env 复制当前环境，保留用户已有 PATH、虚拟环境等配置。
        env = os.environ.copy()
        # PYTHONUTF8 让 Python 子进程默认使用 UTF-8，减少中文输出乱码。
        env.setdefault("PYTHONUTF8", "1")
        # proc 是实际命令执行结果；shell=True 是为了兼容 npm、uv 等复合命令字符串。
        proc = subprocess.run(
            cmd,
            cwd=self.root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
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
            "cmd": cmd,
            # argv 是按平台规则拆出的参数列表，便于后续审计或测试。
            "argv": shlex.split(cmd, posix=os.name != "nt") if cmd else [],
        }
