"""不执行用户代码的本地 Python 导入合同检查。"""

from __future__ import annotations

import ast
from pathlib import Path

from backend.tools import FsTool
from backend.tools.scope import delivery_root


def import_issues(fs: FsTool, state: dict, files: list[str]) -> list[str]:
    """只报告可由静态本地模块证明的缺失名称，动态导出交给真实测试。"""

    root = delivery_root(fs, state)
    roots = [root, root / "src"]
    modules: dict[Path, ast.Module | None] = {}

    def parse(path: Path) -> ast.Module | None:
        if path not in modules:
            try:
                rel = path.relative_to(fs.root).as_posix()
                safe = fs.safe(rel)
                if fs.is_sensitive(safe) or safe.stat().st_size > 100_000:
                    return None
                modules[path] = ast.parse(safe.read_text(encoding="utf-8"))
            except (OSError, ValueError, SyntaxError, UnicodeError):
                modules[path] = None
        return modules[path]

    def exports(tree: ast.Module) -> set[str] | None:
        names: set[str] = set()
        for node in tree.body:
            # 条件导出、星号导入及动态属性不能由这项保守检查判定缺失。
            if isinstance(node, (ast.If, ast.Try, ast.Match)):
                return None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == "__getattr__":
                    return None
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(alias.name == "*" for alias in node.names):
                    return None
                names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names.update(item.id for item in ast.walk(target) if isinstance(item, ast.Name))
        return names

    issues: list[str] = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        path = fs.safe(rel)
        if path != root and root not in path.parents:
            continue
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                candidates = [base.joinpath(*node.module.split("."))]
            else:
                candidates = [base.joinpath(*node.module.split(".")) for base in roots]
            module = next((candidate.with_suffix(".py") if candidate.with_suffix(".py").is_file()
                           else candidate / "__init__.py" for candidate in candidates
                           if candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file()), None)
            if module is None:
                continue
            source = parse(module)
            names = exports(source) if source else None
            if names is None:
                continue
            for alias in node.names:
                if alias.name == "*" or alias.name in names:
                    continue
                if module.name == "__init__.py" and ((module.parent / (alias.name + ".py")).exists()
                                                      or (module.parent / alias.name).is_dir()):
                    continue
                issues.append(f"{rel}:{node.lineno} 从 {module.relative_to(fs.root)} 导入 `{alias.name}`，但该本地模块未定义或导出此名称")
    return issues[:12]
