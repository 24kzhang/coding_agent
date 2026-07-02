from __future__ import annotations

from pathlib import Path


class FsTool:
    """受工作目录限制的文件工具，避免 agent 写出项目根目录。"""

    def __init__(self, workdir: str):
        self.root = Path(workdir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def safe(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"禁止访问工作目录外的路径：{rel}")
        return path

    def list(self, max_files: int = 200) -> list[str]:
        files: list[str] = []
        for path in self.root.rglob("*"):
            if len(files) >= max_files:
                break
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                files.append(str(path.relative_to(self.root)))
        return sorted(files)

    def read(self, rel: str, max_chars: int = 12000) -> str:
        path = self.safe(rel)
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def write(self, rel: str, content: str) -> str:
        path = self.safe(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.root))

    def append(self, rel: str, content: str) -> str:
        path = self.safe(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return str(path.relative_to(self.root))
