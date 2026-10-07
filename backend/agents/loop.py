from __future__ import annotations


class LoopGuard:
    """检测单动作和交替动作空转，进展到达时清空旧窗口。"""

    def __init__(self) -> None:
        self.history: list[str] = []

    def record(self, signature: str, *, progress: bool) -> bool:
        """相同的长度 1～3 动作周期连续出现三次时返回真。"""

        if progress:
            self.history.clear()
            return False
        self.history = (self.history + [signature])[-12:]
        for width in (1, 2, 3):
            if len(self.history) < width * 3:
                continue
            if self.history[-width:] == self.history[-2 * width:-width] == self.history[-3 * width:-2 * width]:
                return True
        return False
