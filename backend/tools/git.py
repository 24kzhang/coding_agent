from __future__ import annotations

import subprocess
from pathlib import Path


class GitTool:
    """Git 工具只做非破坏性操作，供 agent 记录和查看版本状态。"""

    def __init__(self, workdir: str):
        self.root = Path(workdir).resolve()

    def init(self) -> str:
        if (self.root / ".git").exists():
            return "已存在 Git 仓库"
        proc = subprocess.run(["git", "init"], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr).strip()

    def status(self) -> str:
        proc = subprocess.run(["git", "status", "--short"], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr).strip()

    def diff(self, max_chars: int = 20000) -> str:
        proc = subprocess.run(["git", "diff", "--", "."], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr)[-max_chars:]
