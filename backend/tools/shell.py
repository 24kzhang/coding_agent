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
        self.root = Path(workdir).resolve()

    def run(self, cmd: str, timeout: int = 180) -> dict[str, Any]:
        lowered = cmd.strip().lower()
        for bad in self.dangerous:
            if lowered.startswith(bad):
                return {"ok": False, "code": 126, "out": "", "err": f"高风险命令需要用户确认：{cmd}"}
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
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
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "out": proc.stdout[-12000:],
            "err": proc.stderr[-12000:],
            "cmd": cmd,
            "argv": shlex.split(cmd, posix=os.name != "nt") if cmd else [],
        }
