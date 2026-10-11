"""不执行用户代码的本地 Python 导入合同检查。"""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

from backend.tools import FsTool, tool_specs
from backend.tools.scope import command_scope, delivery_root, process_cwd
from backend.tools.shell import prepare_command


def name_issues(fs: FsTool, state: dict, files: list[str]) -> list[str]:
    """用独立静态检查器识别确定的未定义名称，不导入或执行项目代码。"""

    from ruff.__main__ import find_ruff_bin

    root = delivery_root(fs, state)
    selected = []
    for rel in files:
        try:
            path = fs.safe(rel)
            if path.is_file() and path.suffix == ".py" and (root == fs.root or root in path.parents) and not fs.is_sensitive(path):
                selected.append(str(path))
        except (OSError, ValueError):
            continue
    if not selected:
        return []
    result = subprocess.run([str(find_ruff_bin()), "check", "--isolated", "--no-cache", "--select", "F821,F822,F823", "--output-format", "json", *selected],
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode not in {0, 1}:
        return []
    try:
        diagnostics = json.loads(result.stdout)
    except ValueError:
        return []
    issues = []
    for item in diagnostics:
        name = re.search(r"`([^`]+)`", item.get("message", ""))
        rel = Path(item["filename"]).relative_to(fs.root).as_posix()
        problem = "使用尚未赋值的局部名称" if item["code"] == "F823" else "引用未定义名称"
        issues.append(f"{rel}:{item['location']['row']} {problem} `{name.group(1) if name else '未知'}`；请补齐定义或导入，再执行测试")
    return issues[:12]


def middleware_issues(fs: FsTool, state: dict, files: list[str]) -> list[str]:
    """框架异常边界的确定性检查：HTTP 中间件应返回响应而非抛路由异常。"""

    root = delivery_root(fs, state)
    issues = []
    for rel in files:
        try:
            path = fs.safe(rel)
            if path.suffix != ".py" or fs.is_sensitive(path) or not (root == fs.root or root in path.parents):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError):
            continue
        names = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in {"fastapi", "starlette.exceptions"}:
                names.update(alias.asname or alias.name for alias in node.names if alias.name == "HTTPException")
        if not names:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not any(
                isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "middleware"
                and decorator.args and isinstance(decorator.args[0], ast.Constant) and decorator.args[0].value == "http"
                for decorator in fn.decorator_list
            ):
                continue
            shadowed = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
            shadowed.update(arg.arg for arg in ast.walk(fn.args) if isinstance(arg, ast.arg))
            candidates = names - shadowed
            def inspect(node, current=fn, aliases=candidates, filename=rel):
                if node is not current and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try, ast.TryStar)):
                    return
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id in aliases:
                    issues.append(f"{filename}:{node.lineno} 在 HTTP 中间件内抛出路由异常，默认框架异常处理边界不能将其转换为预期状态码；请返回带状态码的响应，或用真实测试证明已有自定义异常处理覆盖此路径")
                for child in ast.iter_child_nodes(node):
                    inspect(child)
            inspect(fn)
    return issues[:12]


def native_command_name(command: str) -> str:
    """区分内部工具名称与明确指定路径的真实程序。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return ""
    return argv[0] if argv and argv[0] in {item["name"] for item in tool_specs()} else ""


def command_issues(fs: FsTool, state: dict, command: str) -> list[str]:
    """执行前复用真实命令工具校验，策略错误只作未执行的修订诊断。"""

    cwd, actual = command_scope(fs, state, command)
    contracts = {item["name"]: item for item in tool_specs()}
    program = native_command_name(actual)
    if program in contracts:
        required = "、".join(contracts[program]["parameters"].get("required", []))
        return [f"命令尚未执行：{program} 是 Agent 内部函数工具，不是系统命令。请直接调用该工具并按 JSON 参数合同传入 {required}；不要放入 run_command 或包装器中执行。"]
    prepared = prepare_command(fs.root, actual, resolve_executable=True, cwd=cwd)
    if isinstance(prepared, dict):
        return ["命令尚未执行：" + str(prepared["err"]) + "。请使用受控文件工具或独立可审查脚本，不要重试同一命令。"]
    argv = shlex.split(actual)
    for index in range(len(argv) - 2):
        if argv[index:index + 2] == ["-m", "json.tool"] and len(argv[index + 2:]) == 1 and not argv[-1].startswith("-"):
            target = process_cwd(cwd, argv) / argv[-1]
            if not target.is_file():
                return [f"命令尚未执行：JSON 输入文件不存在于实际进程目录 {target.parent}。请使用已有输入文件的真实路径；交付目录与输入工作区不同，uv --directory 改变进程目录，--project 不改变进程目录。不要为探查输入而重新创建副本。"]
    return []


def assertion_issues(state: dict, action: dict) -> list[str]:
    """源码局部标识不能代替行为断言，用户明确要求的文字禁用约束仍可验证。"""

    requested = str(state.get("text") or "") + "\n" + "\n".join(state.get("guidance") or [])
    if action.get("tool") != "assert_text_present" and any(Path(path).name in {".gitignore", ".dockerignore", ".npmignore"} for path in action.get("paths", [])):
        removal = any(re.search(r"(?:删除|移除|去掉|取消)[^\n。]{0,80}" + re.escape(text), requested) for text in action.get("texts", []))
        if not removal:
            return ["忽略规则检查尚未执行：忽略文件应包含所需规则，请用 assert_text_present 确认存在。只有用户明确要求删除规则时才检查不存在，不能把存在的正确规则当作失败。"]
    if not any(Path(path).suffix in {".py", ".js", ".ts", ".tsx", ".jsx"} for path in action.get("paths", [])):
        return []
    guessed = [text for text in action.get("texts", []) if text not in requested and (
        re.fullmatch(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", text)
        or re.search(r"(?:=>|\b(?:def|function|return|const|let|var)\s|[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\s*\([^\n]*\)\s*;?\s*$)", text)
    )]
    if guessed:
        return ["源码文字断言尚未执行：" + "、".join(guessed) + " 是实现局部拼写，不是用户明确要求的公开格式。请对照接口签名并用真实行为测试验证修改；不能把变量名、代码片段或格式差异当作功能失败。"]
    return []


def source_versions(fs: FsTool, state: dict, files: list[str]) -> dict[str, str]:
    """记录公开源码版本，供后续审查区分真实通过证据与变更后的待验状态。"""

    root = delivery_root(fs, state)
    versions = {}
    for rel in files:
        try:
            path = fs.safe(rel)
            if (root == fs.root or root in path.parents) and not fs.is_sensitive(path) and path.suffix in {".py", ".js", ".ts", ".tsx", ".html", ".css", ".toml"}:
                versions[rel] = fs.version(rel)
        except (OSError, ValueError):
            continue
    return versions


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

    root, actual = command_scope(fs, state, command)
    try:
        argv = shlex.split(actual)
    except ValueError:
        return []
    if not argv or Path(argv[0]).name != "uv" or not any(action in argv for action in {"run", "sync", "build"}):
        return []
    if any(flag in argv for flag in {"--no-project", "--no-sync", "--no-install-project"}):
        return []
    directory, project_path = None, None
    index = 1
    valued = {"--python", "--with", "--with-editable", "--with-requirements", "--group", "--package", "--env-file", "--config-file", "--cache-dir"}
    while index < len(argv):
        argument = argv[index]
        option, separator, value = argument.partition("=")
        if option in {"--project", "--directory", "-C"} | valued:
            if not separator:
                index += 1
                if index >= len(argv):
                    return []
                value = argv[index]
            if option == "--project":
                project_path = value
            elif option in {"--directory", "-C"}:
                directory = value
        elif (argument not in {"run", "sync", "build"} and not argument.startswith("-")) or argument == "--":
            # 程序自己的参数不属于 uv 项目定位参数。
            break
        index += 1
    if directory is not None:
        root = (root / directory).resolve()
    if project_path is not None:
        root = (root / project_path).resolve()
    manifest = root / "pyproject.toml"
    try:
        path = fs.safe(manifest.relative_to(fs.root).as_posix())
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return []
    def table(*keys):
        value = data
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        return value if isinstance(value, dict) else {}

    uv = table("tool", "uv")
    packaged = "build" in argv or uv.get("package", "build-system" in data)
    if not packaged:
        return []
    references = []
    project = table("project")
    readme = project.get("readme")
    if isinstance(readme, str):
        references.append(readme)
    elif isinstance(readme, dict) and isinstance(readme.get("file"), str):
        references.append(readme["file"])
    license_value = project.get("license")
    if isinstance(license_value, dict) and isinstance(license_value.get("file"), str):
        references.append(license_value["file"])
    wheel = table("tool", "hatch", "build", "targets", "wheel")
    packages = wheel.get("packages", [])
    if isinstance(packages, list):
        references.extend(value for value in packages if isinstance(value, str) and not any(char in value for char in "*?[]"))
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
    def locate(candidates: list[Path]) -> Path | None:
        return next((candidate.with_suffix(".py") if candidate.with_suffix(".py").is_file()
                     else candidate / "__init__.py" for candidate in candidates
                     if candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file()), None)

    for rel in files:
        if not rel.endswith(".py"):
            continue
        path = fs.safe(rel)
        if path != root and root not in path.parents:
            continue
        tree = parse(path)
        if tree is None:
            continue
        module_aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name if alias.asname else alias.name.split(".")[0]
                    module = locate([base.joinpath(*name.split(".")) for base in roots])
                    if module:
                        module_aliases[alias.asname or name] = module
                continue
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                candidates = [base.joinpath(*node.module.split("."))]
            else:
                candidates = [base.joinpath(*node.module.split(".")) for base in roots]
            module = locate(candidates)
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
                    child = locate([module.parent / alias.name])
                    if child:
                        module_aliases[alias.asname or alias.name] = child
                    continue
                issues.append(f"{rel}:{node.lineno} 从 {module.relative_to(fs.root)} 导入 `{alias.name}`，但该本地模块未定义或导出此名称")
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "monkeypatch"
                    and node.func.attr == "setattr" and len(node.args) >= 3
                    and isinstance(node.args[0], ast.Name) and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                continue
            if any(keyword.arg == "raising" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False for keyword in node.keywords):
                continue
            if len(node.args) >= 4 and isinstance(node.args[3], ast.Constant) and node.args[3].value is False:
                continue
            module = module_aliases.get(node.args[0].id)
            if module is None:
                continue
            source = parse(module)
            names = exports(source) if source else None
            attribute = node.args[1].value
            if names is not None and attribute not in names and not attribute.startswith("__"):
                if module.name == "__init__.py" and locate([module.parent / attribute]):
                    continue
                issues.append(f"{rel}:{node.lineno} 的 pytest 替身要求替换 `{attribute}`，但本地模块 {module.relative_to(fs.root)} 未定义此属性；请对照实际模块接口打桩")
    return issues[:12]
