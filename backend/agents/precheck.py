"""不执行用户代码的本地 Python 导入合同检查。"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
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


def router_issues(fs: FsTool, state: dict, files: list[str]) -> list[str]:
    """应用工厂不能在多次创建实例时累积闭包路由到同一个全局路由对象。"""

    root = delivery_root(fs, state)
    issues = []
    for rel in files:
        path = fs.safe(rel)
        if path.suffix != ".py" or fs.is_sensitive(path) or not (root == fs.root or root in path.parents):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            continue
        aliases = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "fastapi" for alias in node.names if alias.name == "APIRouter"}
        shared = {target.id for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in aliases for target in node.targets if isinstance(target, ast.Name)}
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            local = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
            local.update(arg.arg for arg in ast.walk(fn.args) if isinstance(arg, ast.arg))
            returned = {node.value.id for node in fn.body if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)}
            included = {argument.id for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "include_router" and isinstance(node.func.value, ast.Name) and node.func.value.id in returned for argument in node.args[:1] if isinstance(argument, ast.Name)}
            candidates = (shared & (returned | included)) - local
            # 明确重置路由表属于幂等注册，不把它误当作累积。
            reset = {node.func.value.value.id for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "clear" and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "routes" and isinstance(node.func.value.value, ast.Name)}
            for child in fn.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in child.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id in candidates - reset and decorator.func.attr in {"get", "post", "put", "patch", "delete", "route", "websocket"}:
                        issues.append(f"{rel}:{decorator.lineno} 工厂 `{fn.name}` 在全局路由对象 `{decorator.func.value.id}` 上新增闭包并返回它或挂载到新应用。重复创建实例会累积同名路由，后建实例可能仍调用旧实例的依赖或配置；请按实例创建路由对象，或用实际隔离测试证明注册幂等")
                        break
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
    effective_cwd = process_cwd(cwd, argv)
    if effective_cwd != cwd and not effective_cwd.is_dir():
        return [f"命令尚未执行：显式进程目录 {effective_cwd} 尚不存在。请先通过真实写入创建项目，再执行已有脚本或验证命令。"]
    for index, argument in enumerate(argv[:-1]):
        name = Path(argument).name
        if (re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name) or name in {"node", "ruby", "perl"}) and Path(argv[index + 1]).suffix in {".py", ".js", ".mjs", ".cjs", ".rb", ".pl"}:
            target = effective_cwd / argv[index + 1]
            if not target.is_file():
                return [f"命令尚未执行：脚本 {target} 尚未创建。请先写入可审查脚本再运行，不得假设计划中提到的文件已经存在。"]
            if target.suffix == ".py":
                imports = script_imports(fs, target, effective_cwd)
                if imports:
                    return imports
                inputs = script_inputs(fs, target, effective_cwd)
                if inputs:
                    return inputs
    if argv and Path(argv[0]).name == "uv":
        for index in range(len(argv) - 2):
            if argv[index:index + 2] == ["-m", "pip"] and argv[index + 2] in {"list", "freeze", "show"}:
                return ["依赖探查尚未执行：uv 的项目环境默认不安装 pip 模块。请使用 uv pip list、uv pip freeze 或 uv pip show，在相同项目目录查询；无需为了探查额外安装 pip，也不要重复确认已成功的 uv sync。"]
    for index in range(len(argv) - 2):
        if argv[index:index + 2] == ["-m", "json.tool"] and len(argv[index + 2:]) == 1 and not argv[-1].startswith("-"):
            target = process_cwd(cwd, argv) / argv[-1]
            if not target.is_file():
                return [f"命令尚未执行：JSON 输入文件不存在于实际进程目录 {target.parent}。请使用已有输入文件的真实路径；交付目录与输入工作区不同，uv --directory 改变进程目录，--project 不改变进程目录。不要为探查输入而重新创建副本。"]
    return []


def script_imports(fs: FsTool, script: Path, cwd: Path) -> list[str]:
    """非安装型项目的子目录脚本须显式暴露项目包，不能把 cwd 当成 sys.path[0]。"""

    try:
        script = fs.safe(script.relative_to(fs.root).as_posix())
        if fs.is_sensitive(script):
            return []
        project = next((parent for parent in script.parents if parent == fs.root or fs.root in parent.parents if (parent / "pyproject.toml").is_file()), None)
        if project is None or project == script.parent:
            return []
        manifest = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
        uv = manifest.get("tool", {}).get("uv", {})
        if uv.get("package", "build-system" in manifest):
            return []
        if any((cwd / value).resolve() == project for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value):
            return []
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError, SyntaxError, TypeError, AttributeError):
        return []
    sys_names = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names if alias.name == "sys"}
    path_setup = []
    for node in tree.body:
        target = node.value.func.value if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in {"insert", "append", "extend"} else None
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        for value in [target, *targets]:
            if isinstance(value, ast.Attribute) and value.attr == "path" and isinstance(value.value, ast.Name) and value.value.id in sys_names:
                path_setup.append(node.lineno)
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or isinstance(node, ast.ImportFrom) and (not node.module or node.level):
            continue
        modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
        for name in modules:
            if name.split(".")[0] in sys.stdlib_module_names:
                continue
            rel = Path(*name.split("."))
            local = (project / rel).with_suffix(".py").is_file() or (project / rel / "__init__.py").is_file()
            in_script_dir = (script.parent / rel).with_suffix(".py").is_file() or (script.parent / rel / "__init__.py").is_file()
            if local and not in_script_dir and not any(line < node.lineno for line in path_setup):
                return [f"脚本尚未执行：{script.relative_to(fs.root)}:{node.lineno} 导入项目包 {name}，但直接运行文件只将 {script.parent} 放到入口搜索路径。该项目未安装为包，也未在导入前设置 sys.path；请从项目目录使用 python -m <脚本模块>，或显式配置导入路径，不要只靠 uv --project/工作目录解决脚本导入。"]
    return []


def script_inputs(fs: FsTool, script: Path, cwd: Path) -> list[str]:
    """核对确定的只读输入；只静态解析 Path 运算，不执行项目代码。"""

    try:
        script = fs.safe(script.relative_to(fs.root).as_posix())
        if fs.is_sensitive(script):
            return []
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError, SyntaxError):
        return []
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    def ancestors(node):
        result = []
        while node in parents:
            node = parents[node]
            result.append(node)
        return result
    def owner(node):
        return next((item for item in ancestors(node) if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    shadowed = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "open" or isinstance(node, ast.Name) and node.id == "open" and isinstance(node.ctx, ast.Store) or isinstance(node, ast.ImportFrom) and any((alias.asname or alias.name) == "open" for alias in node.names) for node in ast.walk(tree))
    path_names = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "pathlib" for alias in node.names if alias.name == "Path"}
    path_names -= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    path_names -= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    known = {"__file__": script}
    pure_calls = set()
    def path_value(node):
        value = None
        if isinstance(node, ast.Name):
            value = known.get(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            value = node.value
        elif isinstance(node, ast.Attribute):
            base = path_value(node.value)
            if isinstance(base, Path) and node.attr in {"parent", "parents"}:
                value = base.parent if node.attr == "parent" else tuple(base.parents)
        elif isinstance(node, ast.Subscript):
            base, index = path_value(node.value), path_value(node.slice)
            if isinstance(base, tuple) and isinstance(index, int) and 0 <= index < len(base):
                value = base[index]
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = path_value(node.left), path_value(node.right)
            if isinstance(left, Path) and isinstance(right, (str, Path)):
                value = left / right
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in path_names and len(node.args) == 1:
                argument = path_value(node.args[0])
                if isinstance(argument, (str, Path)):
                    value = Path(argument)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "absolute"} and not node.args and not node.keywords:
                base = path_value(node.func.value)
                if isinstance(base, Path):
                    value = (cwd / base).resolve()
            if isinstance(value, Path):
                pure_calls.add(node)
        return value
    for statement in tree.body:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
        if targets:
            value = path_value(statement.value)
            for target in targets:
                if isinstance(target, ast.Name):
                    known.pop(target.id, None)
                    if isinstance(value, Path):
                        known[target.id] = value
    local_functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    invoked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in local_functions and owner(node) is None:
            guards = [item for item in ancestors(node) if isinstance(item, (ast.If, ast.Try, ast.TryStar))]
            if not guards or all(isinstance(item, ast.If) and ast.unparse(item.test) == "__name__ == '__main__'" for item in guards):
                invoked.add(node.func.id)
    reads, writes = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path, mode, derived = None, None, False
        if not shadowed and isinstance(node.func, ast.Name) and node.func.id == "open" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            path = node.args[0].value
            mode = node.args[1] if len(node.args) > 1 else next((keyword.value for keyword in node.keywords if keyword.arg == "mode"), ast.Constant(value="r"))
        elif isinstance(node.func, ast.Attribute) and node.func.attr in {"read_text", "read_bytes", "write_text", "write_bytes", "open"}:
            value = path_value(node.func.value)
            if isinstance(value, Path):
                path = str(value)
                derived = not (isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id in path_names and node.func.value.args and isinstance(node.func.value.args[0], ast.Constant))
                if node.func.attr in {"read_text", "read_bytes"}:
                    mode = ast.Constant(value="r")
                elif node.func.attr in {"write_text", "write_bytes"}:
                    mode = ast.Constant(value="w")
                else:
                    mode = node.args[0] if node.args else next((keyword.value for keyword in node.keywords if keyword.arg == "mode"), ast.Constant(value="r"))
        if path is None or not isinstance(mode, ast.Constant) or not isinstance(mode.value, str):
            continue
        target = (cwd / path).resolve()
        if any(char in mode.value for char in "wax+"):
            writes.add(target)
            continue
        chain = ancestors(node)
        if any(isinstance(item, (ast.If, ast.IfExp, ast.Try, ast.TryStar)) for item in chain):
            continue
        fn = owner(node)
        if derived and fn is not None:
            # 未调用或局部遮蔽的函数路径无法证明必然执行，不作阻断。
            local_names = {item.id for item in ast.walk(fn) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)} | {arg.arg for arg in ast.walk(fn.args) if isinstance(arg, ast.arg)}
            used_names = {item.id for item in ast.walk(node.func.value) if isinstance(item, ast.Name)}
            if fn.name not in invoked or invoked - {fn.name} or local_names & used_names:
                continue
        if mode.value.startswith("r"):
            reads.append((target, node, derived, chain))
    opaque_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and node not in pure_calls and not (isinstance(node.func, ast.Name) and node.func.id in {"open", *path_names, *local_functions})]
    for target, node, derived, chain in reads:
        # 参数先于外层调用求值；外层 json.loads 等不能替内层读取创建文件。
        previous = [call for call in opaque_calls if call not in chain and (call.lineno, call.col_offset) < (node.lineno, node.col_offset)]
        if previous or (derived and owner(node) is not None and any(owner(call) is None and call not in chain for call in opaque_calls)):
            continue
        if target not in writes and not target.is_file():
            return [f"脚本尚未执行：{script.relative_to(fs.root)}:{node.lineno} 的静态只读输入 {target} 不存在。请按实际脚本位置逐级解析父目录，交付目录不等于输入工作区；动态路径与异常回退仍由实际运行验证。"]
    return []


def imported_inputs(fs: FsTool, state: dict, files: list[str], command: str) -> list[str]:
    """沿实际 Python 入口的无条件本地导入检查输入，不执行模块或猜测动态导入。"""

    cwd, actual = command_scope(fs, state, command)
    try:
        argv = shlex.split(actual)
    except ValueError:
        return []
    cwd = process_cwd(cwd, argv)
    root = delivery_root(fs, state)
    seeds = []
    is_pytest = any(Path(arg).name == "pytest" for arg in argv)
    if is_pytest:
        start = next(index for index, arg in enumerate(argv) if Path(arg).name == "pytest") + 1
        selected = [cwd / arg.split("::", 1)[0] for arg in argv[start:] if not arg.startswith("-") and (cwd / arg.split("::", 1)[0]).exists()] or [cwd]
        for rel in files:
            path = fs.safe(rel)
            in_selection = any(path == target or target in path.parents or path.name == "conftest.py" and path.parent in target.parents for target in selected)
            if in_selection and path.suffix == ".py" and (path.name == "conftest.py" or path.name.startswith("test_") or path.name.endswith("_test.py")) and (root == fs.root or root in path.parents):
                seeds.append(path)
    else:
        for index, arg in enumerate(argv[:-1]):
            if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(arg).name) and argv[index + 1].endswith(".py"):
                seeds.append(cwd / argv[index + 1])
    search_roots = [cwd, root, root / "src"]
    visited = set()
    while seeds:
        path = seeds.pop()
        if path in visited:
            continue
        visited.add(path)
        try:
            path = fs.safe(path.relative_to(fs.root).as_posix())
            if fs.is_sensitive(path) or path.stat().st_size > 100_000:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError, UnicodeError):
            continue
        issues = script_inputs(fs, path, cwd)
        if issues:
            return issues
        for node in tree.body:
            # 条件/异常回退/函数内部导入不保证运行，留给实际验证。
            if isinstance(node, ast.Import):
                imports = [(alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [(node.module or "", node.level)]
                if not node.module:
                    imports = [(alias.name, node.level) for alias in node.names if alias.name != "*"]
            else:
                continue
            for name, level in imports:
                if level:
                    base = path.parent
                    for _ in range(level - 1):
                        base = base.parent
                    bases = [base]
                else:
                    bases = [path.parent, *search_roots]
                for base in bases:
                    parts = name.split(".")
                    located = False
                    for count in range(1, len(parts) + 1):
                        candidate = base.joinpath(*parts[:count])
                        module = candidate.with_suffix(".py")
                        package = candidate / "__init__.py"
                        if package.is_file():
                            seeds.append(package)
                            located = True
                        if module.is_file():
                            seeds.append(module)
                            located = True
                            break
                    if located:
                        break
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
    command_root = root
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
    tail = argv[index:]
    pytest_call = bool(tail) and (Path(tail[0]).name == "pytest" or len(tail) >= 3 and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(tail[0]).name) and tail[1:3] == ["-m", "pytest"])
    explicit_config = "-c" in tail or "--pyargs" in tail or any(value.startswith("-c=") for value in tail)
    explicit_paths = bool(table("tool", "pytest", "ini_options").get("pythonpath")) and any(not arg.startswith("-") and (root == (command_root / arg).resolve() or root in (command_root / arg).resolve().parents) for arg in tail[1:])
    if project_path is not None and not packaged and pytest_call and process_cwd(command_root, argv) != root and not (explicit_config or explicit_paths):
        return [f"测试尚未执行：uv --project 选择 {root.relative_to(fs.root)} 的环境，但进程目录仍为 {process_cwd(command_root, argv)}。这个项目未作为包安装，无参数 pytest 会从错误位置收集并导入模块；请改用 uv run --directory <项目目录> python -m pytest，或明确指定项目测试路径与带 pythonpath 的配置。"]
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


def import_issues(fs: FsTool, state: dict, files: list[str], contracts: list[dict] | None = None) -> list[str]:
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
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

        def owner(node, ancestors=parents, module_tree=tree):
            while node in ancestors:
                node = ancestors[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node
            return module_tree

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
                if alias.name != "*" and module.name == "__init__.py":
                    child = locate([module.parent / alias.name])
                    reexport = any(isinstance(item, ast.ImportFrom) and item.level == 1 and item.module is None and any(value.name == alias.name and (value.asname or value.name) == alias.name for value in item.names) for item in source.body)
                    shadowed_export = any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == alias.name or isinstance(item, (ast.Assign, ast.AnnAssign)) and any(isinstance(name, ast.Name) and name.id == alias.name and isinstance(name.ctx, ast.Store) for name in ast.walk(item)) for item in source.body)
                    if child and (alias.name not in names or reexport and not shadowed_export):
                        module_aliases[alias.asname or alias.name] = child
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
                    and isinstance(node.args[0], (ast.Name, ast.Attribute)) and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                continue
            target_object = node.args[0]
            if isinstance(target_object, ast.Attribute):
                chain = target_object
                while isinstance(chain.value, ast.Attribute):
                    chain = chain.value
                module = module_aliases.get(chain.value.id) if isinstance(chain.value, ast.Name) else None
                if module is not None:
                    source = parse(module)
                    names = exports(source) if source else None
                    if names is not None and chain.attr not in names and not (module.name == "__init__.py" and locate([module.parent / chain.attr])):
                        issues.append(f"{rel}:{node.lineno} 的 pytest 补丁目标 `{ast.unparse(target_object)}` 先访问了本地模块 {module.relative_to(fs.root)} 未导出的 `{chain.attr}`；函数内部导入不是模块全局属性。请对照实际查找位置注入替身")
                # 已导出的对象可能来自外部模块或动态实例，不猜测其后续成员。
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
            if contracts is not None and source is not None:
                function = next((fn for fn in source.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == attribute), None)
                globals_ = {name.id for item in source.body if isinstance(item, (ast.Assign, ast.AnnAssign)) for target in (item.targets if isinstance(item, ast.Assign) else [item.target]) for name in ast.walk(target) if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)}
                writes = []
                if function:
                    declared = {name for item in ast.walk(function) if isinstance(item, ast.Global) for name in item.names}
                    local = {name.id for name in ast.walk(function) if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)} | {arg.arg for arg in ast.walk(function.args) if isinstance(arg, ast.arg)}
                    globals_ -= local - declared
                    for item in ast.walk(function):
                        targets = item.targets if isinstance(item, ast.Assign) else [item.target] if isinstance(item, (ast.AnnAssign, ast.AugAssign)) else []
                        if any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in globals_ for target in targets):
                            writes.append(ast.unparse(item))
                if writes:
                    contracts.append({"补丁文件": rel, "补丁位置": node.lineno, "被替换函数": f"{module.relative_to(fs.root)}:{attribute}",
                                      "原函数的模块状态写入": writes, "替身表达式": ast.unparse(node.args[2])})
            original = next((fn for fn in source.body if isinstance(fn, ast.AsyncFunctionDef) and fn.name == attribute), None) if source else None
            replacement = node.args[2]
            if original and isinstance(replacement, ast.Name):
                scope = owner(node)
                candidates = [fn for fn in scope.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == replacement.id]
                if not candidates and scope is not tree:
                    candidates = [fn for fn in tree.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == replacement.id]
                if len(candidates) == 1 and isinstance(candidates[0], ast.FunctionDef):
                    fake = candidates[0]
                    returned = [item.value for item in ast.walk(fake) if isinstance(item, ast.Return) and owner(item) is fake]
                    if returned and all(value is None or isinstance(value, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)) for value in returned):
                        issues.append(f"{rel}:{node.lineno} 用同步替身 `{fake.name}` 替换异步函数 `{attribute}`，替身只返回不可 await 的普通值；调用方 await 会失败。请保留异步调用合同，或提供真正的可等待返回值")
            if names is not None and attribute not in names and not attribute.startswith("__"):
                if module.name == "__init__.py" and locate([module.parent / attribute]):
                    continue
                issues.append(f"{rel}:{node.lineno} 的 pytest 替身要求替换 `{attribute}`，但本地模块 {module.relative_to(fs.root)} 未定义此属性；请对照实际模块接口打桩")
    return issues[:12]
