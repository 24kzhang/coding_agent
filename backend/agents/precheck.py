"""不执行用户代码的本地 Python 导入合同检查。"""

from __future__ import annotations

import ast
import re
import shlex
import tomllib
from pathlib import Path

from backend.tools import FsTool
from backend.tools.scope import command_cwd, delivery_root


def inline_issues(command: str) -> list[str]:
    """在请求进入 shell 前要求解释器内联内容改为可审查脚本。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return []
    interpreters = {"node", "ruby", "perl", "sh", "bash", "zsh", "fish", "powershell", "pwsh"}
    for index, argument in enumerate(argv[:-1]):
        name = Path(argument).name.lower()
        if (name in interpreters or re.fullmatch(r"python(?:\d+(?:\.\d+)?)?(?:\.exe)?", name)) and argv[index + 1] in {"-c", "-e"}:
            return ["解释器内联命令尚未执行：先用 write_file 将代码保存为独立脚本，再用 run_command 运行脚本路径；不要重试 -c/-e，也不要改用复合 shell"]
    return []


def server_issues(command: str) -> list[str]:
    """有限命令工具不能等待已知常驻服务退出，启动烟测由脚本管理生命周期。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return []
    if not argv or any(flag in argv for flag in {"--help", "--version", "--daemon", "-D"}):
        return []
    names = [Path(value).name.lower() for value in argv]
    servers = {"uvicorn", "gunicorn", "hypercorn"}
    running = names[0] in servers
    for index, value in enumerate(argv[:-1]):
        if value == "-m" and argv[index + 1] in servers | {"http.server"}:
            running = running or any(re.fullmatch(r"python(?:\d+(?:\.\d+)?)?(?:\.exe)?", name) for name in names[:index])
    if names[0] == "uv" and "run" in argv:
        tail = argv[argv.index("run") + 1:]
        valued = {"--directory", "--project", "--python", "--with", "--group", "--package", "--env-file", "--config-file", "--cache-dir"}
        while tail and tail[0].startswith("-"):
            tail = tail[2:] if tail[0] in valued else tail[1:]
        running = running or bool(tail and Path(tail[0]).name.lower() in servers)
    if running:
        return ["常驻服务命令尚未执行：run_command 必须有限结束。请在烟测脚本中用 subprocess.Popen 启动服务、轮询 HTTP 就绪、完成真实检查，并在 finally 中终止和等待子进程；再以单独命令运行这个脚本。不要直接等待 uvicorn/gunicorn/hypercorn 退出，也不要使用 &/nohup 绕过生命周期管理。"]
    return []


def manifest_issues(fs: FsTool, state: dict, command: str) -> list[str]:
    """在 uv 触发项目构建前核对公开元数据文件与显式包目录。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return []
    if not argv or Path(argv[0]).name != "uv" or not any(action in argv for action in {"run", "sync", "build"}):
        return []
    if any(flag in argv for flag in {"--no-project", "--no-sync", "--no-install-project"}):
        return []
    root = command_cwd(fs, state, command)
    for index, argument in enumerate(argv[:-1]):
        if argument in {"--project", "--directory", "-C"}:
            root = (root / argv[index + 1]).resolve()
            break
    manifest = root / "pyproject.toml"
    try:
        path = fs.safe(manifest.relative_to(fs.root).as_posix())
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return []
    uv = data.get("tool", {}).get("uv", {})
    packaged = "build" in argv or uv.get("package", "build-system" in data)
    if not packaged:
        return []
    references = []
    project = data.get("project", {})
    readme = project.get("readme")
    if isinstance(readme, str):
        references.append(readme)
    elif isinstance(readme, dict) and isinstance(readme.get("file"), str):
        references.append(readme["file"])
    license_value = project.get("license")
    if isinstance(license_value, dict) and isinstance(license_value.get("file"), str):
        references.append(license_value["file"])
    wheel = data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    references.extend(value for value in wheel.get("packages", []) if isinstance(value, str) and not any(char in value for char in "*?[]"))
    issues = []
    for reference in references:
        target = root / reference
        if not target.exists():
            issues.append(f"{manifest.relative_to(fs.root)} 的构建元数据引用 {reference}，但文件或包目录尚不存在；请先创建，再运行 uv 构建/安装")
    return issues


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
