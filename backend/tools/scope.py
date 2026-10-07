from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.tools.fs import FsTool


def delivery_root(fs: FsTool, state: dict[str, Any]) -> Path:
    """读取任务明确声明的统一交付前缀，输入文件仍可留在父工作区。"""

    pattern = re.compile(
        r"(?:所有|全部)(?:项目|输出|交付|新增|生成)?文件(?:必须|只能|统一|应当)?(?:使用|写入|放在)\s*[`“\"]?([\w./-]+)/\.\.\."
    )
    texts = [str(state.get("text") or ""), *(str(item) for item in state.get("guidance", []))]
    selected: str | None = None
    for text in texts:
        match = pattern.search(text)
        if match:
            selected = match.group(1)
    return fs.safe(selected) if selected else fs.root


def require_delivery_path(fs: FsTool, root: Path, rel: str) -> None:
    """禁止普通写入和受控配置写入越过用户指定的交付子目录。"""

    target = fs.safe(rel)
    if root != target and root not in target.parents:
        prefix = root.relative_to(fs.root).as_posix()
        raise ValueError(f"任务将交付限制在 {prefix}/，禁止写入该目录之外：{rel}")
