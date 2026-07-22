from __future__ import annotations

import subprocess
from pathlib import Path


class GitTool:
    """Git 工具只做非破坏性操作，供 agent 记录和查看版本状态。"""

    def __init__(self, workdir: str):
        # root 是 Git 命令执行目录，固定在用户选择的项目目录。
        self.root = Path(workdir).resolve()

    def init(self) -> str:
        """初始化 Git 仓库；如果已经存在 .git 就不重复执行。"""

        # 已存在 .git 时直接返回，避免重复初始化造成用户困惑。
        if (self.root / ".git").exists():
            return "已存在 Git 仓库"
        # proc 保存 git init 的标准输出和错误输出，用于返回给 agent 观察。
        proc = subprocess.run(["git", "init"], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr).strip()

    def status(self) -> str:
        """读取简短 Git 状态，不修改仓库。"""

        if not (self.root / ".git").exists():
            # Git 默认会向父目录查找仓库；这里必须阻止读取用户未选择的父项目状态。
            return "当前项目目录未初始化 Git 仓库"
        # --short 让输出更短，适合作为事件或模型上下文。
        proc = subprocess.run(["git", "status", "--short"], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr).strip()

    def diff(self, max_chars: int = 20000) -> str:
        """读取当前工作区 diff，最多返回 max_chars 个字符。"""

        if not (self.root / ".git").exists():
            # 未在所选目录显式初始化时不向上穿透父仓库，确保 Git 观察也遵守工作目录边界。
            return "当前项目目录未初始化 Git 仓库"
        # 限制 diff 长度，防止大文件变化把上下文撑爆。
        proc = subprocess.run(["git", "diff", "--", "."], cwd=self.root, text=True, capture_output=True)
        return (proc.stdout + proc.stderr)[-max_chars:]
