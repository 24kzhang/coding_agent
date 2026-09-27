from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from backend.tools import FsTool


class RepoMap:
    """按文件指纹增量缓存符号，并在字符预算内生成仓库地图。"""

    def __init__(self) -> None:
        self.cache: dict[str, tuple[int, int, list[str], list[str]]] = {}

    def build(self, fs: FsTool, files: list[str], goal: str, budget: int = 8_000) -> dict[str, Any]:
        tokens = {item.lower() for item in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", goal)}
        rows: list[tuple[int, str, list[str], list[str]]] = []
        for rel in files:
            path = fs.safe(rel)
            if path.suffix.lower() not in {".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".go", ".rs", ".java"}:
                continue
            symbols, refs = self._symbols(path)
            score = sum(80 for symbol in symbols if symbol.lower() in tokens)
            score += sum(12 for ref in refs if ref.lower() in tokens)
            score += 10 if symbols else 0
            rows.append((score, rel, symbols, refs))
        rows.sort(key=lambda row: (-row[0], row[1]))
        selected: list[dict[str, Any]] = []
        used = 0
        for score, rel, symbols, refs in rows:
            item = {
                "path": rel,
                "symbols": symbols[:40],
                "references": refs[:20],
                "score": score,
            }
            size = len(str(item))
            if used + size > budget:
                break
            selected.append(item)
            used += size
        return {"files": selected, "cached": len(self.cache), "budget": budget, "used": used}

    def _symbols(self, path: Path) -> tuple[list[str], list[str]]:
        stat = path.stat()
        key = str(path)
        cached = self.cache.get(key)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2], cached[3]
        text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
        symbols: list[str] = []
        refs: list[str] = []
        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(text)
                symbols = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                refs = [
                    node.id
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                ]
            except SyntaxError:
                pass
        else:
            symbols = re.findall(
                r"\b(?:class|function|interface|type|const|let|var|func|struct|enum)\s+([A-Za-z_$][\w$]*)",
                text,
            )
            refs = re.findall(r"\b(?:import|from|require)\s*(?:\(|\{)?\s*[\"']?([^\"'\s;)]+)", text)
        symbols = list(dict.fromkeys(symbols))
        refs = list(dict.fromkeys(refs))
        self.cache[key] = (stat.st_mtime_ns, stat.st_size, symbols, refs)
        return symbols, refs
