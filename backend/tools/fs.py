from __future__ import annotations

from pathlib import Path


class FsTool:
    """受工作目录限制的文件工具，避免 agent 写出项目根目录。"""

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

    def list(self, max_files: int = 200) -> list[str]:
        """列出项目内文件，返回相对路径，最多返回 max_files 个。"""

        # files 保存相对于 root 的文件路径，避免把用户机器的绝对路径塞进模型上下文。
        files: list[str] = []
        # rglob 会递归扫描项目；这里跳过 .git 和 __pycache__，减少噪音和上下文浪费。
        for path in self.root.rglob("*"):
            if len(files) >= max_files:
                break
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                files.append(str(path.relative_to(self.root)))
        return sorted(files)

    def read(self, rel: str, max_chars: int = 12000) -> str:
        """读取项目内文件内容，并限制最大字符数，避免一次性塞入过大上下文。"""

        # path 是经过 safe 校验后的真实文件路径。
        path = self.safe(rel)
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def write(self, rel: str, content: str) -> str:
        """写入完整文件内容，返回写入文件的相对路径。"""

        # path 是经过 safe 校验后的目标文件路径。
        path = self.safe(rel)
        # 写入嵌套目录文件时自动创建父目录。
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.root))

    def append(self, rel: str, content: str) -> str:
        """向文件末尾追加内容，返回追加文件的相对路径。"""

        # path 是经过 safe 校验后的目标文件路径。
        path = self.safe(rel)
        # 追加前同样要确保父目录存在。
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return str(path.relative_to(self.root))
