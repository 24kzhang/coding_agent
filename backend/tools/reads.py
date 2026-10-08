"""把简单的只读命令请求转为受控文件工具，不执行 shell。"""

from __future__ import annotations

import shlex
from pathlib import Path

from backend.tools.fs import FsTool
from backend.tools.scope import tool_path


def inspection_read(fs: FsTool, state: dict, command: str) -> dict | None:
    """支持单文件 cat/head/tail；私密文件只交给文件工具返回安全元信息。"""

    if any(marker in command for marker in ("&&", "||", ";", "|", ">", "<", "\n", "`", "$(")):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv or Path(argv[0]).name not in {"cat", "head", "tail", "wc"}:
        return None
    program = Path(argv[0]).name
    unit, count = "lines", 10
    if program == "wc":
        if len(argv) != 3 or argv[1] not in {"-l", "-c", "-m"}:
            return None
    elif program == "cat":
        if len(argv) != 2:
            return None
    elif len(argv) == 4 and argv[1] in {"-n", "-c"} and argv[2].isdigit():
        unit, count = ("lines" if argv[1] == "-n" else "chars"), int(argv[2])
    elif len(argv) == 3 and argv[1][:2] in {"-n", "-c"} and argv[1][2:].isdigit():
        unit, count = ("lines" if argv[1][:2] == "-n" else "chars"), int(argv[1][2:])
    elif len(argv) != 2:
        return None
    try:
        rel = tool_path(fs, state, argv[-1], write=False)
        path = fs.safe(rel)
        rel = path.relative_to(fs.root).as_posix()
        action = {"tool": "read_file", "path": rel, "_source_command": command}
        if program == "wc":
            return {**action, "metadata_only": True, "measure": {"-l": "lines", "-c": "bytes", "-m": "chars"}[argv[1]]}
        if program == "cat" or not path.is_file() or fs.is_sensitive(path):
            return action
        if path.stat().st_size > 2_000_000 or path.suffix.lower() not in fs.text_suffixes | {".log", ".out", ".err"}:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        if unit == "lines":
            lines = text.splitlines(keepends=True)
            size = len("".join(lines[:count] if program == "head" else lines[-count:] if count else []))
        else:
            size = count
        size = min(size, 80_000, len(text))
        action.update(start=max(0, len(text) - size) if program == "tail" else 0, max_chars=max(1, size))
        return action
    except (OSError, ValueError):
        return None
