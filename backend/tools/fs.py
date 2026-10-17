# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

import os
import stat
import tempfile
from hashlib import sha256
from pathlib import Path


class FsTool:
    """受工作目录限制的文件工具，避免 agent 写出项目根目录。"""

    ignored_dirs = {
        ".git",
        ".idea",
        ".local",
        ".aws",
        ".ssh",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".uv-cache",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "vendor",
    }
    text_suffixes = {
        "",
        ".bat",
        ".c",
        ".cc",
        ".conf",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".mjs",
        ".py",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
    binary_suffixes = {
        ".7z",
        ".avi",
        ".db",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".sqlite",
        ".tar",
        ".webm",
        ".webp",
        ".woff",
        ".woff2",
        ".zip",
    }
    ignored_names = {".DS_Store", "Thumbs.db"}
    sensitive_names = {".env", "credentials.json", "models.json", "secrets.json"}
    sensitive_suffixes = {".key", ".p12", ".pem", ".pfx"}

    def __init__(self, workdir: str):
        """初始化当前对象及其运行状态。"""
        self.root = Path(workdir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.last_reads: dict[str, dict[str, object]] = {}

    def version(self, rel: str) -> str:
        """计算当前磁盘内容版本，精确编辑不能覆盖读取后的外部修改。"""

        return sha256(self.safe(rel).read_bytes()).hexdigest()

    def assert_unchanged(self, rel: str) -> None:
        """仅在存在读取基线时检查；工具自身的写入会更新基线。"""

        baseline = self.last_reads.get(rel)
        if baseline and self.version(rel) != baseline["version"]:
            raise ValueError(f"文件在读取后被外部修改：{rel}，必须重新读取再编辑")

    def safe(self, rel: str) -> Path:
        """把相对路径转换成受控绝对路径，并阻止访问项目目录外的文件。"""

        path = (self.root / rel).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"禁止访问工作目录外的路径：{rel}")
        return path

    def list(self, max_files: int = 1200) -> list[str]:
        """列出项目内可读代码和文本文件，返回相对路径，最多返回 max_files 个。"""

        files: list[str] = []
        for current, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(directory for directory in dirs if directory not in self.ignored_dirs)
            for name in sorted(names):
                path = Path(current) / name
                if name in self.ignored_names or path.suffix.lower() in self.binary_suffixes:
                    continue
                if self.is_sensitive(path) or path.is_symlink():
                    continue
                files.append(str(path.relative_to(self.root)))
                if len(files) >= max_files:
                    return files
        return sorted(files)

    def read(self, rel: str, max_chars: int = 12000, start: int = 0) -> str:
        """分段读取项目内文本，start 和 max_chars 都按字符计算。"""

        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{rel}")
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        safe_start = max(int(start), 0)
        safe_max = min(max(int(max_chars), 1), 80_000)
        self.last_reads[rel] = {
            "version": sha256(raw).hexdigest(),
            "start": safe_start,
            "end": min(safe_start + safe_max, len(text)),
            "total_chars": len(text),
            "complete": safe_start == 0 and safe_max >= len(text),
        }
        return text[safe_start : safe_start + safe_max]

    def write(self, rel: str, content: str) -> str:
        """写入完整文件内容，返回写入文件的相对路径。"""

        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assert_unchanged(rel)
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temp_path = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, mode)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
        if rel in self.last_reads:
            self.last_reads[rel]["version"] = self.version(rel)
        return str(path.relative_to(self.root))

    def append(self, rel: str, content: str) -> str:
        """向文件末尾追加内容，返回追加文件的相对路径。"""

        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        self.assert_unchanged(rel)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return self.write(rel, text + content)

    def replace(self, rel: str, old: str, new: str, expected: int = 1) -> str:
        """精确替换已有文本；匹配数量不符时拒绝写入。"""

        return self.write(rel, self.preview_replace(rel, old, new, expected))

    def preview_replace(self, rel: str, old: str, new: str, expected: int = 1) -> str:
        """校验精确替换并计算结果，完全不写入文件。"""

        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        if not old or expected < 1:
            raise ValueError("精确替换的旧文本不能为空，匹配数量必须为正整数")
        self.assert_unchanged(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        actual = text.count(old)
        if actual != expected:
            raise ValueError(f"精确替换失败：期望匹配 {expected} 处，实际匹配 {actual} 处")
        updated = text.replace(old, new, expected)
        return updated

    def replace_block(self, rel: str, start_marker: str, end_marker: str, content: str) -> str:
        """从唯一开始标记替换到结束标记之前；结束标记为空时替换到文件末尾。"""

        return self.write(rel, self.preview_block(rel, start_marker, end_marker, content))

    def preview_block(self, rel: str, start_marker: str, end_marker: str, content: str) -> str:
        """定位锚点并计算替换结果，完全不写入文件。"""

        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        self.assert_unchanged(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        if not start_marker or text.count(start_marker) != 1:
            raise ValueError(f"代码块替换失败：开始标记必须唯一，实际匹配 {text.count(start_marker)} 处")
        start_index = text.index(start_marker)
        if end_marker:
            tail = text[start_index + len(start_marker) :]
            if tail.count(end_marker) != 1:
                raise ValueError(f"代码块替换失败：结束标记在开始标记后必须唯一，实际匹配 {tail.count(end_marker)} 处")
            end_index = start_index + len(start_marker) + tail.index(end_marker)
            trimmed = content.rstrip()
            if trimmed.endswith(end_marker):
                content = trimmed[:-len(end_marker)]
        else:
            end_index = len(text)
        updated = text[:start_index] + content + text[end_index:]
        return updated

    def search(self, query: str, max_results: int = 50) -> list[dict[str, object]]:
        """在项目文本文件中搜索字符串并返回路径、行号和行内容。"""

        clean_query = str(query).strip()
        if not clean_query:
            raise ValueError("搜索内容不能为空")
        lowered = clean_query.lower()
        results: list[dict[str, object]] = []
        for rel in self.list():
            path = self.safe(rel)
            if self.is_sensitive(path) or path.suffix.lower() not in self.text_suffixes or path.stat().st_size > 2_000_000:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if lowered in line.lower():
                    results.append({"path": rel, "line": line_number, "text": line.strip()[:500]})
                    if len(results) >= max(1, min(max_results, 200)):
                        return results
        return results

    def find_text(self, paths: list[str], texts: list[str]) -> list[dict[str, object]]:
        """在指定文件中查找禁用文本，供“不存在”断言返回结构化命中。"""

        clean_paths = list(dict.fromkeys(str(path).strip() for path in paths if str(path).strip()))[:20]
        clean_texts = list(dict.fromkeys(str(text) for text in texts if str(text)))[:20]
        if not clean_paths or not clean_texts:
            raise ValueError("文本不存在断言必须提供 paths 和 texts")
        hits: list[dict[str, object]] = []
        for rel in clean_paths:
            path = self.safe(rel)
            self._ensure_not_sensitive(path)
            if not path.is_file():
                raise FileNotFoundError(f"文件不存在：{rel}")
            content = path.read_text(encoding="utf-8", errors="replace")
            for text in clean_texts:
                index = content.find(text)
                if index >= 0:
                    hits.append({"path": rel, "text": text, "line": content.count("\n", 0, index) + 1})
        return hits

    def is_sensitive(self, path: Path) -> bool:
        """判断文件是否可能包含模型密钥、访问凭据或私钥。"""

        return self.sensitive_path(self.root, path)

    @classmethod
    def sensitive_path(cls, root: Path, path: Path) -> bool:
        """命令与文件工具共用同一套私密路径判定，不读取任何正文。"""

        name = path.name.lower()
        if any(part.lower() in {".local", ".aws", ".ssh"} for part in path.relative_to(root).parts[:-1]):
            return True
        if name == ".env.example" or name.endswith(".example"):
            return False
        return name in cls.sensitive_names or name.startswith(".env.") or path.suffix.lower() in cls.sensitive_suffixes

    def _ensure_not_sensitive(self, path: Path) -> None:
        """阻止模型工具读取或写入常见凭据文件。"""

        if self.is_sensitive(path):
            raise ValueError(f"敏感配置文件不允许交给模型工具处理：{path.name}")
