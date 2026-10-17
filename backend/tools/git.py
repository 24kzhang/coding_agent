# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

import subprocess
from pathlib import Path


class GitTool:
    """Git 工具只做非破坏性操作，供 agent 记录和查看版本状态。"""

    def __init__(self, workdir: str):
        """初始化当前对象及其运行状态。"""
        self.root = Path(workdir).resolve()

    def init(self) -> str:
        """初始化 Git 仓库；如果已经存在 .git 就不重复执行。"""

        if (self.root / ".git").exists():
            return "已存在 Git 仓库"
        proc = subprocess.run(["git", "init"], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr).strip()

    def status(self) -> str:
        """读取简短 Git 状态，不修改仓库。"""

        if not (self.root / ".git").exists():
            return "当前项目目录未初始化 Git 仓库"
        proc = subprocess.run(["git", "status", "--short"], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr).strip()

    def diff(self, max_chars: int = 20000) -> str:
        """读取当前工作区 diff，最多返回 max_chars 个字符。"""

        if not (self.root / ".git").exists():
            return "当前项目目录未初始化 Git 仓库"
        proc = subprocess.run(["git", "diff", "--", "."], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr)[-max_chars:]
