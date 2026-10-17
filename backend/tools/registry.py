# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

from dataclasses import dataclass

from backend.tools.fs import FsTool
from backend.tools.git import GitTool
from backend.tools.shell import ShellTool


@dataclass
class ToolRegistry:
    """按公共工具和专有工具划分能力。

    下游智能体不会直接拿到所有工具，而是由管理者或执行器根据智能体职责传入。
    """

    workdir: str

    def public(self) -> dict[str, object]:
        """创建所有智能体可用的公共工具。"""
        return {
            "fs": FsTool(self.workdir),
            "git": GitTool(self.workdir),
        }

    def exclusive(self, agent: str) -> dict[str, object]:
        """按智能体职责添加专用工具。"""
        tools = self.public()
        if agent in {"coder", "verifier"}:
            tools["shell"] = ShellTool(self.workdir)
        return tools
