# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

"""有界保存原生工具对话，调用与返回按供应商 ID 成对保留。"""

from __future__ import annotations

import json


class ToolTrace:
    """滚动缓存完整调用组；未执行动作明确返回取消结果，不能伪造成功。"""

    def __init__(self, limit: int = 120_000):
        """初始化当前对象及其运行状态。"""
        self.limit = limit
        self.groups: list[list[dict]] = []
        self.pending: dict | None = None
        self.results: dict[int, dict] = {}

    def start(self, message: dict | None) -> None:
        """开始记录一组原生工具调用。"""
        self.pending = message
        self.results = {}

    def record(self, index: int, result: dict) -> None:
        """记录指定工具调用的执行结果。"""
        self.results[index] = result

    def complete(self, notes: list[str]) -> None:
        """补齐并保存当前工具调用组。"""
        if not self.pending:
            return
        group = [self.pending]
        action_index = 0
        for call in self.pending.get("tool_calls", []):
            name = call.get("function", {}).get("name")
            result = None if name == "finish" else self.results.get(action_index)
            if name != "finish":
                action_index += 1
            if result is None:
                result = {"executed": False, "reason": "本批中止或需要修订，本动作未执行", "notes": notes[-6:]}
            content = json.dumps(result, ensure_ascii=False, default=str)
            if len(content) > 6_000:
                content = json.dumps({"summary": content[:5_500], "truncated": True, "notice": "完整当前文件内容以最新快照为准"}, ensure_ascii=False)
            group.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        self.groups.append(group)
        while len(self.groups) > 1 and sum(len(json.dumps(item, ensure_ascii=False)) for item in self.groups) > self.limit:
            self.groups.pop(0)
        self.pending = None
        self.results = {}

    def messages(self) -> list[dict]:
        """按原始顺序返回缓存的工具对话消息。"""
        return [message for group in self.groups for message in group]
