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
        return {
            "fs": FsTool(self.workdir),
            "git": GitTool(self.workdir),
        }

    def exclusive(self, agent: str) -> dict[str, object]:
        tools = self.public()
        if agent in {"coder", "verifier"}:
            tools["shell"] = ShellTool(self.workdir)
        return tools
