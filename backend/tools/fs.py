from __future__ import annotations

import os
import stat
import tempfile
from hashlib import sha256
from pathlib import Path


class FsTool:
    """受工作目录限制的文件工具，避免 agent 写出项目根目录。"""

    # ignored_dirs 是仓库扫描时不进入的依赖、构建产物、缓存和版本控制目录。
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
    # text_suffixes 限制内容检索范围，避免误读图片、数据库和压缩包。
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
    # binary_suffixes 不进入代码文件索引；Coding 无法直接修改这些资源，枚举它们只会浪费上下文。
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
    # ignored_names 是系统自动生成、对代码理解没有价值的文件。
    ignored_names = {".DS_Store", "Thumbs.db"}
    # sensitive_names 是不应发送给模型或由 Coding 工具改写的常见凭据文件名。
    sensitive_names = {".env", "credentials.json", "models.json", "secrets.json"}
    # sensitive_suffixes 覆盖私钥、证书容器等二进制或文本凭据。
    sensitive_suffixes = {".key", ".p12", ".pem", ".pfx"}

    def __init__(self, workdir: str):
        # root 是所有文件读写操作的安全根目录，后续相对路径都会先拼到它下面再 resolve。
        self.root = Path(workdir).resolve()
        # 如果用户选择的是空目录或还不存在的目录，自动创建，方便 agent 直接开始写项目。
        self.root.mkdir(parents=True, exist_ok=True)
        # 读取元数据用于检测分段快照和外部并发修改，不把全文交给模型。
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

        # path 是用户或模型提供路径解析后的真实绝对路径。
        path = (self.root / rel).resolve()
        # 如果解析后不在 root 内，说明可能使用了 ../ 或绝对路径越界，必须拒绝。
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"禁止访问工作目录外的路径：{rel}")
        return path

    def list(self, max_files: int = 1200) -> list[str]:
        """列出项目内可读代码和文本文件，返回相对路径，最多返回 max_files 个。"""

        # files 保存相对于 root 的文件路径，避免把用户机器的绝对路径塞进模型上下文。
        files: list[str] = []
        # os.walk 允许直接剪枝 ignored_dirs，不会先遍历 node_modules 再逐项丢弃。
        for current, dirs, names in os.walk(self.root, followlinks=False):
            # dirs 原地过滤后，os.walk 不会进入依赖、缓存或构建目录。
            dirs[:] = sorted(directory for directory in dirs if directory not in self.ignored_dirs)
            for name in sorted(names):
                # path 是当前文件绝对路径；只把项目相对路径交给模型。
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

        # path 是经过 safe 校验后的真实文件路径。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{rel}")
        # text 使用替换模式读取，少量非 UTF-8 字节不会让整个任务中断。
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        # safe_start 和 safe_max 阻止模型传入负数或超大读取范围。
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

        # path 是经过 safe 校验后的目标文件路径。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        # 写入嵌套目录文件时自动创建父目录。
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assert_unchanged(rel)
        # 唯一临时文件避免两个操作互相覆盖，保留脚本执行权限；失败不遗留半成品。
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

        # path 是经过 safe 校验后的目标文件路径。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        self.assert_unchanged(rel)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return self.write(rel, text + content)

    def replace(self, rel: str, old: str, new: str, expected: int = 1) -> str:
        """精确替换已有文本；匹配数量不符时拒绝写入。"""

        # path 和 text 分别是受控文件路径及当前完整内容。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        if not old or expected < 1:
            raise ValueError("精确替换的旧文本不能为空，匹配数量必须为正整数")
        self.assert_unchanged(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        # actual 是 old 在当前文件中的真实出现次数，用于防止替错位置。
        actual = text.count(old)
        if actual != expected:
            raise ValueError(f"精确替换失败：期望匹配 {expected} 处，实际匹配 {actual} 处")
        # updated 只替换明确数量，保留文件其他内容和用户已有修改。
        updated = text.replace(old, new, expected)
        return self.write(rel, updated)

    def replace_block(self, rel: str, start_marker: str, end_marker: str, content: str) -> str:
        """从唯一开始标记替换到结束标记之前；结束标记为空时替换到文件末尾。"""

        # path/text 是受工作目录约束的目标文件及当前完整内容，所有定位都在真实磁盘版本上完成。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        self.assert_unchanged(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        # start_marker 必须唯一，避免同名函数或重复标题让工具替换到错误代码块。
        if not start_marker or text.count(start_marker) != 1:
            raise ValueError(f"代码块替换失败：开始标记必须唯一，实际匹配 {text.count(start_marker)} 处")
        start_index = text.index(start_marker)
        if end_marker:
            # end_marker 从开始标记之后查找，且在剩余文本中也必须唯一；边界本身会保留。
            tail = text[start_index + len(start_marker) :]
            if tail.count(end_marker) != 1:
                raise ValueError(f"代码块替换失败：结束标记在开始标记后必须唯一，实际匹配 {tail.count(end_marker)} 处")
            end_index = start_index + len(start_marker) + tail.index(end_marker)
        else:
            # 空结束标记明确表示替换到 EOF，适合安全更新文件最后一个函数或章节。
            end_index = len(text)
        updated = text[:start_index] + content + text[end_index:]
        return self.write(rel, updated)

    def search(self, query: str, max_results: int = 50) -> list[dict[str, object]]:
        """在项目文本文件中搜索字符串并返回路径、行号和行内容。"""

        # clean_query 是模型给出的检索词；空查询没有明确语义，直接拒绝。
        clean_query = str(query).strip()
        if not clean_query:
            raise ValueError("搜索内容不能为空")
        # lowered 用于不区分大小写匹配英文标识符，同时不影响中文。
        lowered = clean_query.lower()
        # results 保存命中结果，达到上限后立即停止以控制上下文体积。
        results: list[dict[str, object]] = []
        for rel in self.list():
            # path 是经过 safe 验证的候选文本文件。
            path = self.safe(rel)
            if self.is_sensitive(path) or path.suffix.lower() not in self.text_suffixes or path.stat().st_size > 2_000_000:
                continue
            # lines 按行读取，错误编码使用替换字符容错。
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if lowered in line.lower():
                    results.append({"path": rel, "line": line_number, "text": line.strip()[:500]})
                    if len(results) >= max(1, min(max_results, 200)):
                        return results
        return results

    def find_text(self, paths: list[str], texts: list[str]) -> list[dict[str, object]]:
        """在指定文件中查找禁用文本，供“不存在”断言返回结构化命中。"""

        # clean_paths/clean_texts 去除空值并限制数量，避免模型用断言工具批量读取整个仓库。
        clean_paths = list(dict.fromkeys(str(path).strip() for path in paths if str(path).strip()))[:20]
        clean_texts = list(dict.fromkeys(str(text) for text in texts if str(text)))[:20]
        if not clean_paths or not clean_texts:
            raise ValueError("文本不存在断言必须提供 paths 和 texts")
        # hits 只返回命中位置和禁用文本，不回传整份文件内容。
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

        # name 是小写文件名；.env.example 明确是模板，不按真实凭据处理。
        name = path.name.lower()
        if any(part in {".local", ".aws", ".ssh"} for part in path.relative_to(self.root).parts[:-1]):
            return True
        if name == ".env.example" or name.endswith(".example"):
            return False
        return name in self.sensitive_names or name.startswith(".env.") or path.suffix.lower() in self.sensitive_suffixes

    def _ensure_not_sensitive(self, path: Path) -> None:
        """阻止模型工具读取或写入常见凭据文件。"""

        if self.is_sensitive(path):
            raise ValueError(f"敏感配置文件不允许交给模型工具处理：{path.name}")
