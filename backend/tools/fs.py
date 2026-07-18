from __future__ import annotations

import os
from pathlib import Path


class FsTool:
    """受工作目录限制的文件工具，避免 agent 写出项目根目录。"""

    # ignored_dirs 是仓库扫描时不进入的依赖、构建产物、缓存和版本控制目录。
    ignored_dirs = {
        ".git",
        ".idea",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
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
    # sensitive_names 是不应发送给模型或由 Coding 工具改写的常见凭据文件名。
    sensitive_names = {".env", "credentials.json", "models.json", "secrets.json"}
    # sensitive_suffixes 覆盖私钥、证书容器等二进制或文本凭据。
    sensitive_suffixes = {".key", ".p12", ".pem", ".pfx"}

    def __init__(self, workdir: str):
        # root 是所有文件读写操作的安全根目录，后续相对路径都会先拼到它下面再 resolve。
        self.root = Path(workdir).resolve()
        # 如果用户选择的是空目录或还不存在的目录，自动创建，方便 agent 直接开始写项目。
        self.root.mkdir(parents=True, exist_ok=True)

    def safe(self, rel: str) -> Path:
        """把相对路径转换成受控绝对路径，并阻止访问项目目录外的文件。"""

        # path 是用户或模型提供路径解析后的真实绝对路径。
        path = (self.root / rel).resolve()
        # 如果解析后不在 root 内，说明可能使用了 ../ 或绝对路径越界，必须拒绝。
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"禁止访问工作目录外的路径：{rel}")
        return path

    def list(self, max_files: int = 1200) -> list[str]:
        """列出项目内文件，返回相对路径，最多返回 max_files 个。"""

        # files 保存相对于 root 的文件路径，避免把用户机器的绝对路径塞进模型上下文。
        files: list[str] = []
        # os.walk 允许直接剪枝 ignored_dirs，不会先遍历 node_modules 再逐项丢弃。
        for current, dirs, names in os.walk(self.root, followlinks=False):
            # dirs 原地过滤后，os.walk 不会进入依赖、缓存或构建目录。
            dirs[:] = sorted(directory for directory in dirs if directory not in self.ignored_dirs)
            for name in sorted(names):
                # path 是当前文件绝对路径；只把项目相对路径交给模型。
                path = Path(current) / name
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
        text = path.read_text(encoding="utf-8", errors="replace")
        # safe_start 和 safe_max 阻止模型传入负数或超大读取范围。
        safe_start = max(int(start), 0)
        safe_max = min(max(int(max_chars), 1), 80_000)
        return text[safe_start : safe_start + safe_max]

    def write(self, rel: str, content: str) -> str:
        """写入完整文件内容，返回写入文件的相对路径。"""

        # path 是经过 safe 校验后的目标文件路径。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        # 写入嵌套目录文件时自动创建父目录。
        path.parent.mkdir(parents=True, exist_ok=True)
        # temp_path 与目标同目录，写完后原子替换，避免中断留下半个代码文件。
        temp_path = path.with_name(f".{path.name}.agent.tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
        return str(path.relative_to(self.root))

    def append(self, rel: str, content: str) -> str:
        """向文件末尾追加内容，返回追加文件的相对路径。"""

        # path 是经过 safe 校验后的目标文件路径。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        # 追加前同样要确保父目录存在。
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return str(path.relative_to(self.root))

    def replace(self, rel: str, old: str, new: str, expected: int = 1) -> str:
        """精确替换已有文本；匹配数量不符时拒绝写入。"""

        # path 和 text 分别是受控文件路径及当前完整内容。
        path = self.safe(rel)
        self._ensure_not_sensitive(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        # actual 是 old 在当前文件中的真实出现次数，用于防止替错位置。
        actual = text.count(old)
        if actual != expected:
            raise ValueError(f"精确替换失败：期望匹配 {expected} 处，实际匹配 {actual} 处")
        # updated 只替换明确数量，保留文件其他内容和用户已有修改。
        updated = text.replace(old, new, expected)
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

    def is_sensitive(self, path: Path) -> bool:
        """判断文件是否可能包含模型密钥、访问凭据或私钥。"""

        # name 是小写文件名；.env.example 明确是模板，不按真实凭据处理。
        name = path.name.lower()
        if name == ".env.example" or name.endswith(".example"):
            return False
        return name in self.sensitive_names or name.startswith(".env.") or path.suffix.lower() in self.sensitive_suffixes

    def _ensure_not_sensitive(self, path: Path) -> None:
        """阻止模型工具读取或写入常见凭据文件。"""

        if self.is_sensitive(path):
            raise ValueError(f"敏感配置文件不允许交给模型工具处理：{path.name}")
