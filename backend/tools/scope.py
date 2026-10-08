from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from backend.tools.fs import FsTool

PREFIX_RULE = re.compile(
    r"(?:所有|全部)(?:项目|输出|交付|新增|生成)?文件(?:必须|只能|统一|应当)?(?:使用|写入|放在)\s*[`“\"]?([\w./-]+)/\.\.\."
)

def delivery_root(fs: FsTool, state: dict[str, Any]) -> Path:
    """读取任务明确声明的统一交付前缀，输入文件仍可留在父工作区。"""

    pattern = PREFIX_RULE
    texts = [str(state.get("text") or ""), *(str(item) for item in state.get("guidance", []))]
    selected: str | None = None
    for text in texts:
        match = pattern.search(text)
        if match:
            selected = match.group(1)
        # 直接理解用户声明的交付目录，不要求用户使用内部约定的「目录/...」句式。
        path = r'(?:`([^`]+)`|“([^”]+)”|"([^"]+)"|([^\s，。；;]+))'
        declared = re.search(r'(?:项目输出到\s*|(?:交付目录|输出目录|项目根目录)(?:为|是|[:：])\s*)' + path, text)
        if declared:
            selected = next(value for value in declared.groups() if value is not None).rstrip("/")
    return fs.safe(selected) if selected else fs.root


def tool_path(fs: FsTool, state: dict[str, Any], rel: str, *, write: bool) -> str:
    """普通交付目录支持项目相对路径，明确参数前缀约束保持严格。"""

    output = delivery_root(fs, state)
    if output == fs.root or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return rel
    texts = [str(state.get("text") or ""), *(str(value) for value in state.get("guidance", []))]
    if any(PREFIX_RULE.search(text) for text in texts):
        return rel
    prefix = output.relative_to(fs.root).as_posix()
    workspace_prefix = fs.root.name + "/"
    if rel.startswith(workspace_prefix + prefix + "/") or rel == workspace_prefix + prefix:
        # 接受带工作区名称的交付路径，避免把工作区/交付前缀再次嵌套到交付目录。
        rel = rel[len(workspace_prefix):]
    if rel == prefix or rel.startswith(prefix + "/"):
        return rel
    target = output / rel
    if write or target.exists() or not fs.safe(rel).exists():
        return target.relative_to(fs.root).as_posix()
    return rel


def command_cwd(fs: FsTool, state: dict[str, Any], command: str) -> Path:
    """未加工作区前缀的项目命令在交付根目录运行；明确前缀的命令保留父目录。"""

    output = delivery_root(fs, state)
    if output == fs.root:
        return output
    prefix = output.relative_to(fs.root).as_posix()
    try:
        argv = shlex.split(command)
    except ValueError:
        return fs.root
    for argument in argv[1:]:
        value = argument.split("=", 1)[-1] if argument.startswith("-") else argument
        if value == prefix or value.startswith(prefix + "/"):
            return fs.root
    return output


def require_delivery_path(fs: FsTool, root: Path, rel: str) -> None:
    """禁止普通写入和受控配置写入越过用户指定的交付子目录。"""

    target = fs.safe(rel)
    if root != target and root not in target.parents:
        prefix = root.relative_to(fs.root).as_posix()
        raise ValueError(f"任务将交付限制在 {prefix}/，禁止写入该目录之外：{rel}")
