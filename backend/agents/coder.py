# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

import ast
import json
import re
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

from backend.agents.loop import LoopGuard
from backend.agents.precheck import (
    assertion_issues,
    command_issues,
    import_issues,
    imported_inputs,
    inline_issues,
    manifest_issues,
    middleware_issues,
    name_issues,
    native_command_name,
    router_issues,
    server_issues,
    source_versions,
)
from backend.agents.prompts import CODER_PROMPT, LONGCAT_CODER_SUFFIX, NATIVE_CODER_PROMPT
from backend.agents.trace import ToolTrace
from backend.agents.types import AgentState
from backend.agents.verifier import review_before_verification
from backend.tools import FsTool, GitTool, tool_specs
from backend.tools.reads import inspection_read
from backend.tools.scope import delivery_root, tool_path
from llm import LlmBalanceError, LlmClient, LlmError
from llm.client import LlmOutputLimitError, unavailable_error

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


_CODE_FILE_PATTERN = re.compile(
    r"(?<![\w.-])([\w./-]+\.(?:css|html|js|json|md|py|sh|toml|ts|tsx|yaml|yml))(?![\w.-])"
)

FILE_CONTEXT_LIMIT = 24_000
SNAPSHOT_BUDGET = 48_000


def _interface_brief(path: str, content: str) -> str:
    """保留调用签名，避免正文快照被挤出后调用方猜测参数名。"""

    signatures: list[str] = []
    if path.endswith(".py"):
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ""
        exported = []
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            exported.extend(item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name))
        if exported:
            signatures.append("模块变量：" + "、".join(exported[:30]))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                signatures.append(f"{node.name}({ast.unparse(node.args)})")
            elif isinstance(node, ast.ClassDef):
                fields = [f"{item.target.id}: {ast.unparse(item.annotation)}" for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)]
                if fields:
                    signatures.append(f"{node.name} 字段：" + ", ".join(fields))
    elif path.endswith((".js", ".ts", ".tsx")):
        for match in re.finditer(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]{0,500})\)", content):
            signatures.append(f"{match[1]}({match[2]})")
    return "\n".join(signatures[:20])[:2_400]
CODER_OUTPUT_TOKENS = 12_000
CODER_MAX_STEPS = 60
CODER_MAX_TRANSIENT_FAILURES = 6
CODER_TRANSIENT_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 12.0)


def _is_transient_llm_error(error_text: str) -> bool:
    """判断错误是否可能在短暂退避后自行恢复。"""

    lowered = error_text.lower()
    markers = (
        "模型接口错误 429",
        "模型接口错误 500",
        "模型接口错误 502",
        "模型接口错误 503",
        "模型接口错误 504",
        "模型响应超时",
        "模型流式响应超过总时长",
        "调用失败",
        "connecterror",
        "connecttimeout",
        "connection",
    )
    return any(marker.lower() in lowered for marker in markers)


def _version_query(command: str) -> bool:
    """只缓存已知工具的独立版本开关，不把任意项目脚本视为无副作用查询。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return len(argv) == 2 and Path(argv[0]).name in {"uv", "python", "python3", "node", "npm", "npx", "git", "ruff", "pytest"} and argv[1] in {"--version", "-V"}


def _middle_trim(text: str, limit: int) -> str:
    """保留长文本首尾，避免错误根因或文件末尾代码被截掉。"""

    if len(text) <= limit:
        return text
    head_len = max(1, int(limit * 0.7))
    tail_len = max(1, limit - head_len)
    return text[:head_len] + "\n... 中间内容已省略 ...\n" + text[-tail_len:]


def _test_brief(
    tests: list[dict],
    addressed_paths: set[str] | None = None,
    passed_commands: set[str] | None = None,
) -> str:
    """把测试结果压缩成可执行的问题清单，不重复发送完整 traceback。"""

    addressed = addressed_paths or set()
    passed = passed_commands or set()
    lines: list[str] = []
    for item in tests:
        if item.get("ok"):
            continue
        cmd = str(item.get("cmd") or "检查")
        if cmd in passed:
            continue
        issues = item.get("issues")
        if isinstance(issues, list) and issues:
            for issue in issues[:12]:
                issue_text = str(issue)
                if addressed and any(
                    path in issue_text or Path(path).name in issue_text
                    for path in addressed
                ):
                    continue
                lines.append(f"- [{cmd}] {_middle_trim(issue_text, 700)}")
            continue
        output = str(item.get("out") or item.get("err") or "失败但没有输出")
        lines.append(f"- [{cmd}] {_middle_trim(output, 1_400)}")
    if not lines and (addressed or passed):
        return ""
    return _middle_trim("\n".join(lines) or "- 暂无失败项", 8_000)


def _repair_paths(tests: list[dict], changes: list[str], current_files: list[str]) -> list[str]:
    """按验证反馈提取修复文件，避免重试时无差别加载全部变更。"""

    current = set(current_files)
    found: list[str] = []
    feedback_text = json.dumps(tests, ensure_ascii=False)
    for match in _CODE_FILE_PATTERN.findall(feedback_text):
        rel = match.lstrip("./")
        if rel in current and rel not in found:
            found.append(rel)
    for rel in changes:
        if rel in current and rel not in found:
            found.append(rel)
    return found[:10]


def _task_brief(state: AgentState) -> str:
    """构造 Coding 专用任务摘要，排除与当前实现无关的完整会话历史。"""

    ctx = state.get("context")
    if not ctx:
        text = str(state.get("text", ""))
        guidance = list(state.get("guidance") or [])
        if guidance:
            text += (
                "\n\n运行中用户追加指导（优先于原任务中的冲突要求）：\n"
                + "\n".join(f"- {item}" for item in guidance)
            )
        return _middle_trim(text, 18_000)
    constraints = "\n".join(f"- {item}" for item in ctx.constraints)
    skill_sections = "\n\n".join(
        f"### Skill: {item.name}\n{item.content}"
        for item in ctx.skills
    )
    text = f"任务目标：\n{ctx.goal}\n\n约束：\n{constraints}"
    guidance = list(state.get("guidance") or getattr(ctx, "guidance", []))
    if guidance:
        text += (
            "\n\n运行中用户追加指导（优先于原任务中的冲突要求，越靠后优先级越高）：\n"
            + "\n".join(f"- {item}" for item in guidance)
        )
    if skill_sections:
        text += "\n\n必须遵循的用户选定 Skills：\n" + skill_sections
    return _middle_trim(text, 100_000)


def _repo_brief(repo: dict, *, include_snippets: bool) -> str:
    """压缩仓库摘要；修复轮次只需要当前清单，不再携带初始旧代码。"""

    files = list(repo.get("files") or [])
    stack = "、".join(repo.get("stack") or []) or "未知"
    lines = [
        "技术栈：" + stack,
        "文件清单：" + "、".join(files[:120]) + (f"（共 {len(files)} 个）" if len(files) > 120 else ""),
    ]
    if include_snippets:
        fresh_snippets: list[str] = []
        budget = 12_000
        for path, content in (repo.get("snippets") or {}).items():
            if budget <= 0:
                break
            snippet = _middle_trim(str(content), min(4_000, budget))
            fresh_snippets.append(f"[{path}]\n{snippet}")
            budget -= len(snippet)
        if fresh_snippets:
            lines.append("关键片段：\n" + "\n\n".join(fresh_snippets))
    return "\n".join(lines)


def _is_discovery_command(command: str) -> bool:
    """识别只用于重复枚举仓库的 shell 命令。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    program = argv[0].rsplit("/", 1)[-1].lower()
    return program in {"find", "ls", "pwd", "tree"}


def _is_inspection_command(command: str) -> bool:
    """识别用 shell 绕过文件快照工具的只读检查命令。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    program = Path(argv[0]).name.lower()
    return program in {"awk", "cat", "grep", "head", "rg", "sed", "tail", "wc"}


def _guidance_forbids_reads(state: AgentState) -> bool:
    """用户明确要求停止读取或搜索后，由权限层阻止模型继续空转。"""

    sources = [str(state.get("original_text") or ""), *(str(item) for item in (state.get("guidance") or []))]
    for latest in reversed(sources):
        if any(marker in latest for marker in (
            "不要再搜索或读取",
            "停止继续读取",
            "禁止继续读取",
            "不要再读取",
            "不要再搜索",
        )):
            return True
        if any(marker in latest for marker in ("允许读取", "恢复读取", "可以继续读取", "允许搜索")):
            return False
    return False


def _is_dependency_mutation_command(command: str) -> bool:
    """识别会改变依赖或锁文件、因此需要允许重跑旧验证命令的命令。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    program = Path(argv[0]).name.lower()
    if program == "npm":
        return "install" in argv[1:] or "ci" in argv[1:]
    if program == "uv":
        return any(action in argv[1:] for action in {"add", "remove", "sync", "lock"})
    return False


def _is_verification_command(command: str) -> bool:
    """识别能够作为实现完成证据的测试、构建、语法检查或烟测命令。"""

    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if any(arg in {"-c", "-e"} for arg in argv):
        return False
    names = [Path(arg).name.lower() for arg in argv]
    if any(Path(arg).suffix in {".py", ".js", ".mjs", ".cjs", ".ts"}
           and any(part.lower() in {"tests", "test", "e2e"} for part in Path(arg).parts[:-1])
           for arg in argv):
        return True
    if any(name in {"pytest", "ruff", "mypy", "pyright", "compileall", "py_compile", "unittest"} for name in names):
        return True
    if "node" in names and "--check" in argv:
        return True
    if "-m" in argv:
        index = argv.index("-m")
        if index + 1 < len(argv):
            parts = argv[index + 1].split(".")
            if parts[-1] in {"build", "test", "check", "smoke", "lint", "typecheck"} or any(part in {"tests", "test", "e2e"} for part in parts[:-1]):
                return True
    return any(re.search(r"(?:^|[-_])(?:test|typecheck|build|smoke|lint|check)(?:[-_]|$)", Path(name).stem) for name in names)


def _native_coder_specs(mcp_tools: list[dict], reads_forbidden: bool, repo_ready: bool = False) -> list[dict]:
    """构造原生工具清单；用户停止读取后不再向模型暴露只读文件工具。"""

    specs = tool_specs()
    if repo_ready:
        specs = [item for item in specs if item["name"] != "list_files"]
    if reads_forbidden:
        blocked = {"list_files", "read_file", "search_files"}
        specs = [item for item in specs if str(item.get("name")) not in blocked]
    specs.extend(
        {
            "name": str(item["action"]),
            "description": str(item.get("description") or "调用已授权 MCP 工具"),
            "parameters": item.get("input_schema") or {"type": "object", "properties": {}},
            "risk": "low" if item.get("shared") else "medium",
            "permission": "allow",
            "timeout": 60,
            "concurrency": "parallel" if item.get("shared") else "serial",
            "output_limit": 12_000,
            "mode": "read" if item.get("shared") else "write",
        }
        for item in mcp_tools
    )
    return specs


def _observation_brief(action: dict, observation: dict) -> str:
    """把工具返回值转换为下一轮模型可直接判断的权威结果。"""

    status = "成功" if observation.get("ok") else "失败"
    tool = str(action.get("tool") or "工具")
    target = str(action.get("path") or action.get("cmd") or "").strip()
    prefix = f"工具结果 [{status}] {tool}" + (f"：{target}" if target else "")
    result = observation.get("result")
    if isinstance(result, dict):
        code = result.get("code")
        output = str(result.get("out") or result.get("err") or "").strip()
        detail = f"退出码={code}；" + (_middle_trim(output, 1_200) if output else "命令无输出，已正常结束")
        return f"{prefix}。{detail}"
    text = str(observation.get("text") or "").strip()
    return f"{prefix}。{_middle_trim(text, 1_400) if text else '工具未返回附加文本'}"


def coder(graph: AgentGraph, state: AgentState) -> AgentState:
    """Coding 智能体：通过 ReAct 循环调用工具，真实修改项目文件。"""

    retry = int(state.get("retry", 0))
    graph._emit(state, "coder", "start", f"Coding 智能体开始 ReAct 执行，第 {retry + 1} 轮")
    fs = FsTool(state["workdir"])
    shell = graph._shell(state)
    git = GitTool(state["workdir"])
    client = graph._client("coder", state)
    current_model_id = str(getattr(getattr(client, "cfg", None), "id", ""))
    fallback_configs = (
        graph._fallback_models("coder", state, current_model_id) if current_model_id else []
    )
    failed_model_ids: set[str] = {current_model_id} if current_model_id else set()
    mcp_tools = graph._ensure_mcp_tools(state)
    mcp_prompt = graph._mcp_prompt(state)
    model_name = str(getattr(getattr(client, "cfg", None), "model", "")).lower()

    def switch_coder_model(reason: str) -> bool:
        """切到下一个备用模型；成功返回 True。"""

        nonlocal client, model_name, parse_failures, empty_steps
        next_cfg = next((cfg for cfg in fallback_configs if cfg.id not in failed_model_ids), None)
        if next_cfg is None:
            return False
        failed_model_ids.add(next_cfg.id)
        client = LlmClient(next_cfg)
        client.default_effort = graph.model_store.effort_for_agent("coder", state.get("effort"), model=next_cfg)
        model_name = str(next_cfg.model).lower()
        parse_failures = 0
        empty_steps = 0
        observations.append(
            f"Coding 模型因{reason}已切换到 {next_cfg.id}；继续基于当前失败清单和磁盘快照推进，禁止重新开始。"
        )
        graph._emit(state, "coder", "model", f"Coding 模型因{reason}自动切换到 {next_cfg.id}")
        return True
    observations: list[str] = []
    file_observations: dict[str, str] = {}
    interface_summaries: dict[str, str] = {}
    seen_search_hits: set[str] = set()
    search_cache: dict[str, dict] = {}
    read_cache: dict[str, dict] = {}
    version_cache: dict[str, dict] = {}
    seen_actions: set[str] = set()
    progress_items: list[str] = []
    read_paths: set[str] = set()
    unchecked_rewrites: set[str] = set()
    mutation_count = 0
    file_revision = 0
    preflight_revision: tuple[int, str] | None = None
    preflight_issues: list[str] = []
    preflight_kind = ""
    pending_native_command = ""
    completed_checks: list[dict] = []
    failed_checks: dict[str, dict] = {}
    action_protocol_failures = 0
    edit_protocol_failures = 0
    assertion_protocol_failures = 0
    idle_after_mutation = 0
    repo = state.get("repo") or {}
    repo_files = list(repo.get("files") or [])
    current_files = repo_files
    if state.get("repo") is not None:
        preview = "、".join(repo_files[:20]) or "空项目"
        observations.append(f"仓库读取节点已完成文件扫描：{preview}。不要调用 list_files，直接读取目标文件或开始写入。")
    if retry > 0:
        current_files = fs.list(max_files=300)
        observations.append(
            "这是验证失败后的修复轮次。项目文件已经存在，只修复失败项，禁止从头重建、重复创建或扫描目录。"
        )
        repair_paths = _repair_paths(state.get("tests", []), state.get("changes", []), current_files)
        for rel in reversed(repair_paths):
            try:
                file_observations[rel] = fs.read(rel, FILE_CONTEXT_LIMIT)
                read_paths.add(rel)
            except (OSError, ValueError):
                continue
    changes = list(state.get("changes", []))
    commands = list(state.get("commands", []))
    verification_commands = list(state.get("verification_commands", []))
    failed_tests = ""
    addressed_failure_paths: set[str] = set()
    passed_failure_commands: set[str] = set()
    successful_verification_commands: set[str] = set()
    if state.get("tests") and not state.get("tests_ok", True):
        failed_tests = _test_brief(state["tests"])
        if all(item.get("cmd") == "Coding 完成状态" for item in state["tests"] if not item.get("ok")):
            observations.append("上一轮只有 Coding 完成状态失败：当前文件若已满足需求，应直接返回 done=true，不要重新扫描或重建。")

    task_brief = _task_brief(state)
    repo_context = dict(repo)
    repo_context["snippets"] = {
        path: content
        for path, content in (repo.get("snippets") or {}).items()
        if path not in file_observations
    }
    repo_brief = _repo_brief(repo_context, include_snippets=True)

    def prune_snapshots_after_timeout() -> None:
        """模型超时后只保留任务明确点名的文件，缩小下一轮输入并允许其他文件按需重读。"""

        named_paths = [path for path in file_observations if path in task_brief]
        keep_paths = set(named_paths[-2:] or list(file_observations)[-1:])
        for observed_path in list(file_observations):
            if observed_path in keep_paths:
                continue
            file_observations.pop(observed_path, None)
            read_paths.discard(observed_path)
            stale_signatures = {
                signature
                for signature in seen_actions
                if '"tool": "read_file"' in signature and f'"path": "{observed_path}"' in signature
            }
            seen_actions.difference_update(stale_signatures)
        observations.append(
            "模型超时后已收缩文件上下文，仅保留：" + ("、".join(sorted(keep_paths)) if keep_paths else "无")
        )

    coding_ok = False
    coding_summary = ""
    parse_failures = 0
    transient_failures = 0
    empty_steps = 0
    duplicate_read_failures = 0
    max_steps = CODER_MAX_STEPS
    loop_guard = LoopGuard()
    native_fallback = False
    coder_output_limit = CODER_OUTPUT_TOKENS
    native_trace = ToolTrace()
    for step in range(1, max_steps + 1):
        native_trace.complete(observations)
        native_mode = isinstance(client, LlmClient) and "tools" in getattr(client.cfg, "capabilities", []) and empty_steps < 2 and not native_fallback
        state = graph._apply_guidance(state, checkpoint=f"Coding 第 {step} 步规划前")
        task_brief = _task_brief(state)
        planned_guidance_revision = int(state.get("guidance_revision", 0))
        reads_forbidden = _guidance_forbids_reads(state)
        recent_observations = [_middle_trim(item, 1_200) for item in observations[-6:]]
        current_snapshots: list[str] = []
        visible_complete_paths: set[str] = set()
        visible_cached_paths: set[str] = set()
        snapshot_budget = SNAPSHOT_BUDGET
        for path, content in reversed(list(file_observations.items())[-8:]):
            if snapshot_budget <= 0:
                break
            snapshot = _middle_trim(content, min(FILE_CONTEXT_LIMIT, snapshot_budget))
            metadata = fs.last_reads.get(path, {})
            segment = ""
            if not metadata.get("complete", True) or int(metadata.get("total_chars", 0)) > len(content):
                segment = f"；仅包含字符 {metadata.get('start', 0)}～{metadata.get('end', len(content))}，文件共 {metadata.get('total_chars', '未知')} 字符，禁止据此整体覆盖"
            if metadata.get("complete") and snapshot == content and int(metadata.get("total_chars", 0)) == len(content):
                visible_complete_paths.add(path)
            if snapshot == content:
                visible_cached_paths.add(path)
            current_snapshots.append(f"[{path} 当前内容{segment}]\n{snapshot}")
            snapshot_budget -= len(snapshot)
        progress = "、".join(progress_items[-30:]) or "尚未执行成功工具"
        manifest = "、".join(current_files[:120]) or "空项目"
        readable = "、".join(sorted(read_paths)) or "无"
        finish_instruction = ""
        if step >= max_steps - 3 and (mutation_count or changes):
            finish_instruction = (
                "\n已进入收尾阶段：禁止仅为确认而读取文件。若实现已完成，必须返回 actions=[]、done=true；"
                "若仍有缺口，只执行直接修复缺口的写入或测试命令。\n"
            )
            if native_mode:
                finish_instruction = finish_instruction.replace("返回 actions=[]、done=true", "调用 finish")
        stable_text = (
            task_brief
            + "\n\n仓库摘要：\n"
            + repo_brief
        )
        dynamic_text = (
            f"\n\n当前是第 {step}/{max_steps} 步。当前模式：{'失败修复' if retry else '首次实现'}。\n"
            + f"当前磁盘文件：{manifest}\n"
            + f"已完成进度：{progress}\n"
            + f"已读取且当前可直接修改的文件：{readable}\n"
            + "已生成模块的接口签名（调用点参数必须与之逐项一致）：\n"
            + _middle_trim("\n".join(f"{name}: {brief}" for name, brief in interface_summaries.items() if brief), 8_000)
            + "\n"
            + "必须根据失败清单和已有观察产生新进展；禁止重复扫描、重复创建已有文件或重新开始任务。\n"
            + "最近工具观察是已经发生的权威事实；标记为成功的命令无需再次确认，不得声称没有看到结果。\n"
            + ("当前是失败修复轮次：验证失败清单的优先级高于原始任务中的整体重写等实现方式，只精确修复清单中的问题。\n" if retry else "")
            + (
                "修复轮必须先创建失败清单中明确缺失的文件、测试、文档或配置；"
                "在这些交付物落盘前，禁止读取无关文件或运行全量验证。\n"
                if retry and failed_tests
                else ""
            )
            + "若目标文件出现在“已读取且当前可直接修改”列表，禁止再次 read_file；短改动直接 replace_file，长函数直接 replace_block。\n"
            + f"输入工作区：{fs.root}；新增文件和项目命令默认交付根目录：{delivery_root(fs, state)}。\n"
            + "项目命令直接使用相对交付目录的路径；明确使用工作区交付前缀或 --project 前缀的命令会在父工作区执行。禁止 cd 或复合 shell 语法。\n"
            + "每轮最多返回 3 个紧密相关动作；修复轮次优先修改失败清单中的文件并运行对应测试。\n\n"
            + (
                "用户已禁止继续读取和搜索：本轮原生工具清单已移除 "
                "list_files/read_file/search_files。只能直接写入、精确修改或运行验证命令。\n"
                if reads_forbidden
                else ""
            )
            + finish_instruction
            + ("\n\n必须解决的验证失败：\n" + failed_tests if failed_tests else "")
            + ("\n\n本轮仍未通过的真实检查（修改文件不会自动消除失败；旧版本诊断不能证明当前版本仍有同一缺陷，需重新执行裁定）：\n"
               + "\n\n".join(f"命令：{command}；失败时文件修订：{item.get('revision', '未知')}；当前修订：{file_revision}\n{item['output']}" for command, item in list(failed_checks.items())[-3:])
               if failed_checks else "")
            + ("\n\n最近读取的当前文件：\n" + "\n\n".join(current_snapshots) if current_snapshots else "")
            + "\n\n最近工具观察：\n"
            + "\n".join(recent_observations)
        )
        messages = [
            {
                "role": "system",
                "content": (NATIVE_CODER_PROMPT if native_mode else CODER_PROMPT)
                + (LONGCAT_CODER_SUFFIX if "longcat" in model_name else "")
                + (f"\n\n{mcp_prompt}" if mcp_prompt else ""),
            },
            {
                "role": "user",
                "content": graph._multimodal_content(stable_text, state, dynamic_text),
            },
        ]
        protocol = getattr(getattr(client, "cfg", None), "protocol", "auto")
        chat_history = protocol == "chat_completions" or (protocol == "auto" and "deepseek" not in model_name and "qwen3.8" not in model_name)
        if native_mode and chat_history:
            messages = [messages[0], {"role": "user", "content": graph._multimodal_content(stable_text, state)},
                        *native_trace.messages(), {"role": "user", "content": dynamic_text}]
        messages = graph.model_store.redact(messages)
        usage_before = getattr(client, "last_usage", None)
        try:
            output_limit = graph._check_model_budget(state, coder_output_limit)
            if native_mode:
                native_specs = _native_coder_specs(mcp_tools, reads_forbidden, state.get("repo") is not None)
                data = client.chat_tools(
                    messages,
                    native_specs,
                    max_tokens=output_limit,
                    require_call=True,
                )
            elif "longcat" in model_name or empty_steps >= 2 or native_fallback:
                if isinstance(client, LlmClient):
                    data = client.chat_json(messages, plain_text=True, max_tokens=output_limit)
                else:
                    data = client.chat_json(messages, plain_text=True)
            else:
                if isinstance(client, LlmClient):
                    data = client.chat_json(messages, max_tokens=output_limit)
                else:
                    data = client.chat_json(messages)
            graph._add_tokens(state, client.last_usage, agent="coder", model_id=str(getattr(getattr(client, "cfg", None), "id", "")), effort=str(getattr(client, "default_effort", "")))
        except LlmError as exc:
            failed_usage = getattr(client, "last_usage", None)
            if failed_usage is not usage_before:
                graph._add_tokens(state, failed_usage, agent="coder", model_id=str(getattr(getattr(client, "cfg", None), "id", "")), effort=str(getattr(client, "default_effort", "")), failed=True)
            error_text = str(exc)[:240]
            if unavailable_error(exc):
                graph.unavailable_models.add(str(getattr(getattr(client, "cfg", None), "id", "")))
            if unavailable_error(exc) and switch_coder_model("余额或额度不足" if isinstance(exc, LlmBalanceError) else "服务不可用"):
                transient_failures = 0
                continue
            if isinstance(exc, LlmBalanceError):
                raise
            if isinstance(exc, LlmOutputLimitError) and isinstance(client, LlmClient):
                native_fallback = True
                coder_output_limit = min(coder_output_limit + 2_000, client.cfg.max_output_tokens, 16_000)
                observations.append(
                    "供应商输出预算触顶，已切到结构化文本兼容模式。下一轮只能返回一个"
                    "正文不超过 4000 字符的小动作；长文件分块 write_file/append_file，"
                    "不要再次输出整份长文件。当前配置仍使用用户指定的同一模型。"
                )
                graph._emit(state, "coder", "model", "原生工具输出触顶，改用结构化文本与小块写入恢复", data={"reason": "native_output_limit", "max_tokens": coder_output_limit})
            if (
                ("原生工具调用参数不是合法 JSON" in error_text or isinstance(exc, LlmOutputLimitError))
                and isinstance(client, LlmClient)
                and client.default_effort != "none"
                and not getattr(client.cfg, "effort_budgets", {})
            ):
                supported = set(getattr(client.cfg, "efforts", []))
                lowered = (
                    "none"
                    if "none" in supported
                    else "minimal"
                    if client.default_effort == "low" and "minimal" in supported
                    else "low"
                    if "low" in supported
                    else "minimal"
                    if "minimal" in supported
                    else ""
                )
                if lowered:
                    previous_effort = client.default_effort
                    client.default_effort = lowered
                    graph._emit(
                        state,
                        "coder",
                        "model",
                        (
                            f"原生工具参数未完整输出，思考等级已从 {previous_effort} "
                            f"降为 {lowered} 后重试"
                        ),
                        data={"reason": "invalid_tool_json", "effort": lowered},
                    )
            protocol = "LongCat 原生工具标签" if "longcat" in model_name else "约定 JSON"
            transient_error = _is_transient_llm_error(error_text)
            if transient_error:
                transient_failures += 1
                parse_failures = 0
            else:
                parse_failures += 1
                transient_failures = 0
            if "响应超时" in error_text or "超过总时长" in error_text:
                prune_snapshots_after_timeout()
                observations.append(
                    f"第 {step} 轮模型响应超时：{error_text}。下一轮只返回一个小动作，"
                    f"短改动使用 replace_file，长函数使用 replace_block，并严格使用{protocol}。"
                )
            elif transient_error:
                observations.append(
                    f"第 {step} 轮模型服务暂时不可用：{error_text}。退避后重试相同任务。"
                )
            else:
                if "原生工具调用参数不是合法 JSON" in error_text:
                    observations.append(
                        f"第 {step} 轮原生工具参数在输出上限处截断。下一轮只能返回一个写入动作，"
                        "正文不得超过 4000 字符；长文件先 write_file 写首块，后续回合再用 "
                        "append_file 分块追加。不要在同一响应中生成多个文件。"
                    )
                else:
                    observations.append(
                        f"第 {step} 轮输出格式无效：{error_text}。请严格使用{protocol}。"
                    )
            if transient_error:
                retry_delay = CODER_TRANSIENT_RETRY_DELAYS[
                    min(transient_failures - 1, len(CODER_TRANSIENT_RETRY_DELAYS) - 1)
                ]
                graph._emit(
                    state,
                    "coder",
                    "error" if transient_failures >= CODER_MAX_TRANSIENT_FAILURES else "model",
                    f"模型服务暂时不可用，{retry_delay:g} 秒后重试"
                    f"（{transient_failures}/{CODER_MAX_TRANSIENT_FAILURES}）：{error_text}",
                )
                if transient_failures >= CODER_MAX_TRANSIENT_FAILURES:
                    coding_summary = (
                        f"Coding 模型服务连续 {CODER_MAX_TRANSIENT_FAILURES} 次不可用。"
                    )
                    break
                time.sleep(retry_delay)
                continue
            graph._emit(
                state,
                "coder",
                "error" if parse_failures >= 3 else "model",
                f"模型输出格式无效，正在重试（{parse_failures}/3）：{error_text}",
            )
            if parse_failures >= 2 and switch_coder_model("连续输出格式无效"):
                continue
            if parse_failures >= 3:
                coding_summary = "Coding 模型连续三次没有返回有效执行指令。"
                break
            continue
        state = graph._apply_guidance(state, checkpoint=f"Coding 第 {step} 步模型返回后")
        if int(state.get("guidance_revision", 0)) != planned_guidance_revision:
            observations.append("模型生成动作期间收到新指导，已丢弃尚未执行的旧动作并重新规划。")
            continue
        parse_failures = 0
        transient_failures = 0
        native_trace.start(graph.model_store.redact(data.get("native_message")))
        thought = data.get("thought", "")
        if thought:
            graph._emit(state, "coder", "thought", thought)
        actions = graph._normalize_actions(data.get("actions"), mcp_tools)
        invalid_actions = [str(action["_error"]) for action in actions if action.get("_error")]
        if invalid_actions:
            action_protocol_failures += 1
            observations.append("本批工具尚未执行，必须修正参数：" + "；".join(invalid_actions[:3]))
            terminal = action_protocol_failures >= 3
            graph._emit(state, "coder", "error" if terminal else "model",
                        f"工具参数协议需修订（{action_protocol_failures}/3），整批未执行：" + "；".join(invalid_actions[:3]),
                        data={"reason": "invalid_action_protocol", "issues": invalid_actions[:3]})
            if terminal:
                coding_summary = "工具参数协议连续三次未修正，已停止且未执行无效动作。"
                break
            continue
        action_protocol_failures = 0
        if len(actions) > 3:
            observations.append(f"模型本轮返回 {len(actions)} 个动作，只执行前 3 个；其余动作请下一轮继续。")
            actions = actions[:3]
            data = {**data, "done": False}
        unresolved_missing_delivery = retry > 0 and failed_tests and any(
            marker in failed_tests
            for marker in ("缺失", "不存在", "尚未创建", "没有交付")
        )
        if not actions and data.get("done") is True and not unresolved_missing_delivery and not preflight_issues:
            empty_steps = 0
            coding_ok = True
            coding_summary = graph._clean_summary(data.get("summary"), "Coding 阶段已完成。")
            break
        if not actions:
            if unresolved_missing_delivery:
                observations.append(
                    "验证清单仍明确指出交付物缺失，不能只返回完成标记；"
                    "下一步必须直接创建缺失文件或测试。"
                )
            completion = int(getattr(getattr(client, "last_usage", None), "completion", 0) or 0)
            if completion >= int(CODER_OUTPUT_TOKENS * 0.95) and switch_coder_model(
                "输出预算耗尽但没有工具动作"
            ):
                observations.append(
                    f"上一模型消耗 {completion} 输出 token 却没有工具动作；"
                    "已立即切换模型，下一步只执行一个小型写入或验证动作。"
                )
                continue
            empty_steps += 1
            observations.append("本轮既没有工具动作也没有完成标记，请读取、修改或验证后再继续。")
            graph._emit(
                state,
                "coder",
                "error" if empty_steps >= 5 else "model",
                f"模型未给出可执行动作，正在重试（{empty_steps}/5）"
                + ("；当前使用普通文本兼容模式" if "longcat" in model_name or empty_steps >= 2 else ""),
            )
            if empty_steps >= 5:
                coding_summary = "Coding 模型连续五次没有给出工具动作或完成标记。"
                break
            continue
        empty_steps = 0
        action_failed = False
        step_signatures: list[str] = []
        batch_progress = False
        diagnostic_progress = False
        switched_this_step = False
        mutated_paths: set[str] = set()
        dependency_mutated = False
        guidance_replan = False
        for action_index, action in enumerate(actions):
            state = graph._apply_guidance(state, checkpoint=f"Coding 第 {step} 步工具动作前")
            if int(state.get("guidance_revision", 0)) != planned_guidance_revision:
                guidance_replan = True
                break
            tool = str(action.get("tool") or "")
            if pending_native_command and tool == pending_native_command:
                preflight_issues, preflight_kind, pending_native_command = [], "", ""
            if tool in {"assert_text_absent", "assert_text_present"}:
                assertion_paths = [fs.safe(tool_path(fs, state, str(path), write=False)) for path in action.get("paths", [])]
                advice = ["文字断言尚未执行：私密文件不能通过断言读取正文或核对值。模型配置格式使用受控工具提供的字段合同，安全元信息可用 read_file；真实应用在运行时读取配置，不要反复探查私密正文。"] if any(fs.is_sensitive(path) for path in assertion_paths) else assertion_issues(state, action)
                if advice:
                    assertion_protocol_failures += 1
                    observations.extend(advice)
                    graph._emit(state, "coder", "error" if assertion_protocol_failures >= 3 else "review", advice[0])
                    action_failed = True
                    diagnostic_progress = True
                    if assertion_protocol_failures >= 3:
                        coding_summary = "文字断言连续三次未改为需求或行为验证，已停止。"
                    break
                assertion_protocol_failures = 0
            if tool == "run_command" and not reads_forbidden:
                adapted = inspection_read(fs, state, str(action.get("cmd", "")))
                if adapted:
                    action, tool = adapted, "read_file"
            if tool == "run_command":
                policy_issues = command_issues(fs, state, str(action.get("cmd", "")))
                if policy_issues:
                    diagnostic_progress = diagnostic_progress or policy_issues != preflight_issues
                    preflight_issues, preflight_kind = policy_issues, "command"
                    pending_native_command = native_command_name(str(action.get("cmd", "")))
                    observations.extend(policy_issues)
                    graph._emit(state, "coder", "review", policy_issues[0])
                    break
                if preflight_kind == "command":
                    preflight_issues, preflight_kind = [], ""
            if action.get("path") and tool in {"read_file", "write_file", "append_file", "replace_file", "replace_block"}:
                action = {**action, "path": tool_path(fs, state, str(action["path"]), write=tool != "read_file")}
            edit_path = str(action.get("path") or "")
            if tool in {"write_file", "append_file", "replace_file", "replace_block"} and edit_path and fs.safe(edit_path).is_file() and not fs.is_sensitive(fs.safe(edit_path)) and (
                edit_path not in read_paths or (tool == "write_file" and edit_path not in visible_complete_paths)
            ):
                edit_protocol_failures += 1
                refreshed = graph.model_store.redact_text(fs.read(edit_path, FILE_CONTEXT_LIMIT))
                file_observations.pop(edit_path, None)
                file_observations[edit_path] = refreshed
                interface_summaries[edit_path] = _interface_brief(edit_path, refreshed)
                read_paths.add(edit_path)
                message = f"编辑尚未执行：拒绝覆盖现有文件 {edit_path}，本次模型上下文没有足够的当前内容。已刷新真实快照，下一轮基于它精确修改；不要盲目覆盖自动生成或已有文件。"
                observations.append(message)
                graph._emit(state, "coder", "error" if edit_protocol_failures >= 3 else "review", message,
                            data={"reason": "edit_visibility", "path": edit_path})
                action_failed = True
                if edit_protocol_failures >= 3:
                    coding_summary = "编辑连续三次未遵守当前快照边界，已停止且未覆盖文件。"
                break
            if tool == "write_file" and edit_path in unchecked_rewrites:
                edit_protocol_failures += 1
                message = f"编辑尚未执行：拒绝重复整体重写 {edit_path}，该文件已成功写入但尚未验证。请基于当前源码精确修改，或先运行对应检查再根据具体失败修复；不要再次生成整份文件。"
                observations.append(message)
                graph._emit(state, "coder", "error" if edit_protocol_failures >= 3 else "review", message,
                            data={"reason": "unchecked_rewrite", "path": edit_path})
                action_failed = True
                diagnostic_progress = True
                if edit_protocol_failures >= 3:
                    coding_summary = "整体重写连续三次未改为精确修改或验证，已停止且保留原文件。"
                break
            if tool in {"replace_file", "replace_block"} and edit_path in read_paths and not fs.is_sensitive(fs.safe(edit_path)):
                try:
                    if tool == "replace_file":
                        fs.preview_replace(edit_path, str(action["old"]), str(action["new"]), int(action.get("expected", 1)))
                    else:
                        fs.preview_block(edit_path, str(action["start_marker"]), str(action["end_marker"]), str(action["content"]))
                except (OSError, ValueError) as exc:
                    edit_protocol_failures += 1
                    try:
                        refreshed = graph.model_store.redact_text(fs.read(edit_path, FILE_CONTEXT_LIMIT))
                        file_observations.pop(edit_path, None)
                        file_observations[edit_path] = refreshed
                        interface_summaries[edit_path] = _interface_brief(edit_path, refreshed)
                    except (OSError, ValueError):
                        pass
                    message = f"编辑尚未执行：{exc}。已刷新真实源码；使用现有锚点，替换至文件末尾请传空 end_marker，不要猜测不存在的边界。"
                    observations.append(message)
                    graph._emit(state, "coder", "error" if edit_protocol_failures >= 3 else "review", message,
                                data={"reason": "edit_precondition", "path": edit_path})
                    action_failed = True
                    if edit_protocol_failures >= 3:
                        coding_summary = "编辑定位连续三次未修正，未写入无效替换。"
                    break
                edit_protocol_failures = 0
            signature = json.dumps(action, ensure_ascii=False, sort_keys=True)
            step_signatures.append(signature)
            path = str(action.get("path") or (action.get("target") if tool == "model_copy" else "") or "")
            inspection_command = tool == "run_command" and _is_inspection_command(
                str(action.get("cmd", ""))
            )
            secret_write = tool != "model_copy" and tool in {
                "append_file",
                "replace_block",
                "replace_file",
                "run_command",
                "write_file",
            } and graph.model_store.contains_secret(action)
            if action.get("_error"):
                observation = {"ok": False, "text": str(action["_error"])}
            elif secret_write:
                observation = {
                    "ok": False,
                    "credential_facts": graph.model_store.secret_facts(action),
                    "text": (
                        "拒绝执行：工具参数包含真实或疑似凭据。密钥只能通过 model_copy "
                        "写入被忽略的私密配置；源码、脚本、命令和测试必须使用占位符或从配置读取。"
                    ),
                }
            elif reads_forbidden and (tool in {"list_files", "read_file", "search_files"} or inspection_command):
                duplicate_read_failures += 1
                idle_after_mutation = 0
                observation = {
                    "ok": False,
                    "text": (
                        "用户已明确要求停止读取和搜索。当前上下文足以执行；"
                        "下一步只能写入、精确修改或运行测试与构建。"
                    ),
                }
            elif signature in seen_actions and tool == "read_file" and path in read_paths and path in visible_cached_paths:
                duplicate_read_failures += 1
                saved_read = read_cache.get(signature)
                if saved_read and (saved_read.get("read") or {}).get("version") == fs.version(path):
                    observation = {**saved_read, "cached": True}
                    fs.last_reads[path] = dict(observation["read"])
                else:
                    observation = graph._do_action(action, fs, shell, git, state)
            elif tool == "search_files" and signature in search_cache:
                observation = {**search_cache[signature], "cached": True,
                               "text": "复用未变更源码的已有搜索结果：\n" + search_cache[signature]["text"]}
            elif tool == "run_command" and _version_query(str(action.get("cmd", ""))) and signature in version_cache:
                observation = {**version_cache[signature], "cached": True}
            elif signature in seen_actions and tool not in {"git_diff", "git_status", "read_file", "model_copy"} and not (tool == "run_command" and _version_query(str(action.get("cmd", "")))):
                observation = {
                    "ok": False,
                    "text": f"拒绝重复动作：{action.get('tool')} 已执行或拒绝过。下一步必须改用其他动作推进任务。",
                }
            elif tool == "list_files" and state.get("repo") is not None:
                observation = {
                    "ok": True,
                    "cached": True,
                    "text": "使用现有仓库文本索引（未重新扫描）：" + "、".join(current_files) + "。请直接推进实现或验证。",
                }
            elif tool == "run_command" and state.get("repo") is not None and _is_discovery_command(str(action.get("cmd", ""))):
                observation = {
                    "ok": True,
                    "cached": True,
                    "text": f"使用现有仓库文本索引（未执行 shell 命令）；工作区：{fs.root}；默认命令交付目录：{delivery_root(fs, state)}；"
                    + "、".join(current_files) + "。二进制资源不在文本索引中；请按数据中的资源路径验证，直接推进实现。",
                }
            elif inspection_command:
                duplicate_read_failures += 1
                observation = {
                    "ok": False,
                    "text": (
                        "拒绝使用 shell 的 cat/grep/sed 等命令绕过文件上下文管理。"
                        "需要新内容时使用 read_file/search_files；当前已知文件应直接修改。"
                    ),
                }
            elif (
                tool == "write_file"
                and path
                and fs.safe(path).exists()
                and path not in read_paths
            ):
                observation = {
                    "ok": False,
                    "text": (
                        f"拒绝覆盖现有文件：{path}。请先 read_file 获取当前完整内容；"
                        "小改动使用 replace_file，确实需要整体整理时才使用 write_file。"
                    ),
                }
            elif (
                tool == "write_file"
                and path
                and fs.safe(path).exists()
                and path not in visible_complete_paths
            ):
                observation = {
                    "ok": False,
                    "text": f"拒绝用局部快照整体覆盖文件：{path}。请精确替换已读取片段，或读取完整文件后重写。",
                }
            elif tool in {"append_file", "replace_block", "replace_file"} and path and fs.safe(path).exists() and path not in read_paths:
                observation = {
                    "ok": False,
                    "text": f"修改前必须先读取当前文件：{path}。请先调用 read_file。",
                }
            else:
                if tool == "run_command" and isinstance(client, LlmClient) and state.get("task_type") in {"code_gen", "code_mod"}:
                    code_issues = inline_issues(str(action.get("cmd", "")))
                    if code_issues:
                        diagnostic_progress = diagnostic_progress or code_issues != preflight_issues
                        preflight_issues, preflight_kind = code_issues, "inline"
                        observations.extend(code_issues)
                        graph._emit(state, "coder", "review", code_issues[0])
                        break
                    if preflight_kind == "inline":
                        preflight_issues, preflight_kind = [], ""
                    service_issues = server_issues(str(action.get("cmd", "")))
                    if service_issues:
                        diagnostic_progress = diagnostic_progress or service_issues != preflight_issues
                        preflight_issues, preflight_kind = service_issues, "server"
                        observations.extend(service_issues)
                        graph._emit(state, "coder", "review", service_issues[0])
                        break
                    if preflight_kind == "server":
                        preflight_issues, preflight_kind = [], ""
                if tool == "run_command" and changes and isinstance(client, LlmClient) and state.get("task_type") in {"code_gen", "code_mod"}:
                    metadata_issues = manifest_issues(fs, state, str(action.get("cmd", "")))
                    if metadata_issues:
                        diagnostic_progress = diagnostic_progress or metadata_issues != preflight_issues
                        preflight_issues = metadata_issues
                        preflight_kind = "manifest"
                        observations.append("构建前必须补齐元数据引用：" + "；".join(metadata_issues))
                        graph._emit(state, "coder", "review", "构建元数据待完善：" + "；".join(metadata_issues))
                        break
                    if preflight_kind == "manifest":
                        preflight_issues = []
                        preflight_kind = ""
                if (tool == "run_command" and _is_verification_command(str(action.get("cmd", "")))
                        and changes and isinstance(client, LlmClient)
                        and state.get("task_type") in {"code_gen", "code_mod"}):
                    undefined = name_issues(fs, state, current_files)
                    if undefined:
                        preflight_issues, preflight_kind = undefined, "names"
                        observations.append("测试前静态名称检查：" + "；".join(undefined))
                        graph._emit(state, "coder", "review", "测试前静态名称检查：" + "；".join(undefined))
                        diagnostic_progress = True
                        break
                    if preflight_kind == "names":
                        preflight_issues, preflight_kind = [], ""
                    structural_issues = (imported_inputs(fs, state, current_files, str(action["cmd"]))
                                         + import_issues(fs, state, current_files) + router_issues(fs, state, current_files))
                    if structural_issues:
                        diagnostic_progress = diagnostic_progress or structural_issues != preflight_issues
                        preflight_issues = structural_issues
                        preflight_kind = "imports"
                        observations.append("执行前确定的本地导入合同问题：" + "；".join(structural_issues))
                        graph._emit(state, "coder", "review", "本地导入合同待完善：" + "；".join(structural_issues))
                        break
                    if preflight_kind == "imports":
                        preflight_issues = []
                        preflight_kind = ""
                    review_key = (file_revision, str(action["cmd"]))
                    fresh_review = preflight_revision != review_key
                    if fresh_review:
                        preflight_issues = middleware_issues(fs, state, current_files) + review_before_verification(graph, {**state, "completed_checks": completed_checks, "failed_checks": list(failed_checks.values())}, fs, current_files, str(action["cmd"]))
                        preflight_kind = "review" if preflight_issues else ""
                        preflight_revision = review_key
                        graph._emit(state, "coder", "review", "执行前合同审查：" + ("；".join(preflight_issues) if preflight_issues else "未发现阻碍，继续实际验证"))
                    if preflight_issues and fresh_review:
                        observations.append(
                            "执行前审查候选（模型意见，不是实际执行结果）：" + "；".join(preflight_issues)
                            + "。证据充分时精确修复；若候选引用旧版本错误、与当前源码或标准接口不符，"
                            "保持文件不变，再次选择完全相同的命令即可直接实际执行，以本版本结果裁定，不能只为迎合候选反复改写。"
                        )
                        diagnostic_progress = True
                        break
                    preflight_issues = []
                    preflight_kind = ""
                observation = graph._do_action(action, fs, shell, git, state)
            observation = graph.model_store.redact(observation)
            if tool == "run_command" and not observation.get("cached") and observation.get("result"):
                command = str(action.get("cmd") or "")
                if observation.get("ok"):
                    failed_checks.pop(command, None)
                elif (observation.get("result") or {}).get("code") != 126:
                    failed_checks[command] = {
                        "command": command, "output": _middle_trim(str(observation["text"]), 12_000),
                        "revision": file_revision, "versions": source_versions(fs, state, current_files),
                    }
            if tool == "run_command" and observation.get("ok") and not observation.get("cached") and _is_dependency_mutation_command(str(action.get("cmd", ""))):
                version_cache.clear()
            if tool == "run_command" and observation.get("ok") and not observation.get("cached") and _version_query(str(action.get("cmd", ""))):
                version_cache[signature] = dict(observation)
            if action.get("_source_command"):
                observation = {**observation, "adapted_from": graph.model_store.redact_text(action["_source_command"])}
            if observation.get("ok") and observation.get("content_available") is not False and not observation.get("cached") and observation.get("exists") is not False and tool == "read_file" and path not in visible_complete_paths:
                diagnostic_progress = True
            if observation.get("ok") and tool == "search_files":
                hits = [json.dumps(hit, ensure_ascii=False, sort_keys=True) for hit in observation.get("hits", [])]
                diagnostic_progress = diagnostic_progress or any(hit not in seen_search_hits for hit in hits)
                seen_search_hits.update(hits)
                if not observation.get("cached"):
                    search_cache[signature] = observation
            if not observation.get("cached"):
                seen_actions.add(signature)
            if observation.get("ok") and observation.get("content_available") is not False and tool == "read_file" and observation.get("exists") is not False:
                read_cache[signature] = {**observation, "read": dict(observation.get("read") or {})}
                if len(read_cache) > 24:
                    read_cache.pop(next(iter(read_cache)))
            if tool == "read_file" and observation.get("ok") and observation.get("content_available") is not False and observation.get("exists") is not False and path:
                marker = "\n"
                raw_text = str(observation["text"])
                file_observations.pop(path, None)
                file_observations[path] = _middle_trim(
                    graph.model_store.redact_text(raw_text.split(marker, 1)[-1]),
                    FILE_CONTEXT_LIMIT,
                )
                if (observation.get("read") or {}).get("complete"):
                    interface_summaries[path] = _interface_brief(path, file_observations[path])
                observations.append(f"已读取当前文件：{path}")
            else:
                observations.append(_observation_brief(action, observation))
                if (
                    not observation.get("ok")
                    and tool in {"append_file", "replace_block", "replace_file", "write_file"}
                    and path
                    and fs.safe(path).exists()
                ):
                    try:
                        refreshed = graph.model_store.redact_text(
                            fs.read(path, FILE_CONTEXT_LIMIT)
                        )
                        file_observations.pop(path, None)
                        file_observations[path] = refreshed
                        read_paths.add(path)
                        observations.append(
                            f"{path} 修改定位失败，已重新加载当前磁盘内容；"
                            "下一步必须基于下方当前快照中的真实文本修改，禁止继续猜测不存在的导入或锚点。"
                        )
                    except (OSError, ValueError):
                        pass
            action_failed = action_failed or not bool(observation.get("ok"))
            if observation.get("ok"):
                if tool in {"append_file", "model_copy", "replace_block", "replace_file", "write_file"} or observation.get("files"):
                    file_revision += 1
                    search_cache.clear()
                    read_cache.clear()
                    seen_actions = {
                        item for item in seen_actions
                        if '"tool": "search_files"' not in item and '"tool": "run_command"' not in item
                    }
                detail = path or str(action.get("cmd") or "")
                progress_items.append(f"{tool}:{detail}" if detail else tool)
                if not observation.get("cached") and (tool in {"append_file", "assert_text_absent", "model_copy", "replace_block", "replace_file", "run_command", "write_file"} or tool.startswith("mcp__")):
                    batch_progress = True
                if tool == "write_file" and path:
                    unchecked_rewrites.add(path)
                elif tool == "run_command" and not observation.get("cached"):
                    unchecked_rewrites.clear()
                    command = str(action.get("cmd") or "")
                    dependency_mutated = dependency_mutated or _is_dependency_mutation_command(command)
                    passed_failure_commands.add(command)
                    if _is_verification_command(command):
                        successful_verification_commands.add(command)
                        completed_checks.append({"command": command,
                                                 "output": graph._trim(str((observation.get("result") or {}).get("out") or observation["text"]), 1_600),
                                                 "versions": source_versions(fs, state, current_files)})
                        completed_checks = completed_checks[-8:]
                    failed_tests = _test_brief(
                        state.get("tests", []),
                        addressed_failure_paths,
                        passed_failure_commands,
                    )
                if tool == "read_file" and observation.get("content_available") is not False and observation.get("exists") is not False and path:
                    read_paths.add(path)
                elif tool in {"append_file", "model_copy", "replace_block", "replace_file", "write_file"} and path and not observation.get("cached"):
                    successful_verification_commands.clear()
                    passed_failure_commands.clear()
                    mutated_paths.add(path)
                    if tool == "write_file":
                        read_paths.add(path)
                    if path not in current_files:
                        current_files.append(path)
                if not observation.get("cached") and (tool in {"append_file", "model_copy", "replace_block", "replace_file", "write_file", "run_command"} or tool.startswith("mcp__")):
                    mutation_count += 1
            if observation.get("file"):
                changes.append(str(observation["file"]))
            for related in observation.get("files", []):
                rel = str(related)
                if rel not in changes:
                    changes.append(rel)
                if rel not in current_files:
                    current_files.append(rel)
                mutated_paths.add(rel)
            if observation.get("cmd"):
                commands.append(str(observation["cmd"]))
                command = str(observation["cmd"])
                if _is_verification_command(command) and (observation.get("result") or {}).get("code") != 126:
                    verification_commands.append(command)
            native_trace.record(action_index, graph.model_store.redact(observation))
            graph._emit(state, "coder", "tool", observation["text"], data={
                **observation, "tool": tool,
                "arguments": {"说明": "参数被凭据守卫拒绝"} if secret_write else action,
                "path": path,
                "command": "[参数被凭据守卫拒绝]" if secret_write else str(action.get("cmd") or ""),
            })
            if (
                (not observation.get("ok") or observation.get("cached"))
                and (tool == "read_file" or inspection_command)
                and duplicate_read_failures >= 2
                and switch_coder_model("重复读取已知文件")
            ):
                switched_this_step = True
                break
            if not observation.get("ok"):
                observations.append("本批工具出现失败，已停止剩余动作；下一轮必须先处理该失败再继续。")
                break
        if switched_this_step:
            continue
        for mutated_path in mutated_paths:
            try:
                file_observations.pop(mutated_path, None)
                file_observations[mutated_path] = graph.model_store.redact_text(
                    fs.read(mutated_path, FILE_CONTEXT_LIMIT)
                )
                interface_summaries[mutated_path] = _interface_brief(mutated_path, file_observations[mutated_path])
                read_paths.add(mutated_path)
                observations.append(f"已自动刷新修改后的当前文件：{mutated_path}；下一轮禁止重复读取，可直接继续修改。")
            except (OSError, ValueError):
                read_paths.discard(mutated_path)
                file_observations.pop(mutated_path, None)
        if mutated_paths:
            addressed_failure_paths.update(mutated_paths)
            failed_tests = _test_brief(
                state.get("tests", []),
                addressed_failure_paths,
                passed_failure_commands,
            )
            observations.append(
                "以下文件已成功修改，其旧验证描述已转为等待 Verifier 复验，不再作为本轮当前事实："
                + "、".join(sorted(mutated_paths))
                + "。请只处理仍显示的失败项；若清单为空，完成必要测试后立即结束本轮。"
            )
        if mutated_paths or dependency_mutated:
            if dependency_mutated:
                version_cache.clear()
                successful_verification_commands.clear()
                passed_failure_commands.clear()
            seen_actions = {item for item in seen_actions if '"tool": "run_command"' not in item}
        if batch_progress or diagnostic_progress:
            idle_after_mutation = 0
            duplicate_read_failures = 0
        elif mutation_count:
            idle_after_mutation += 1
        if edit_protocol_failures >= 3 or assertion_protocol_failures >= 3:
            break
        if guidance_replan:
            loop_guard.history.clear()
            observations.append("工具批次执行期间收到新指导，已保留完成的原子动作并取消剩余旧动作。")
            continue
        if loop_guard.record("\n".join(step_signatures), progress=batch_progress or diagnostic_progress):
            coding_summary = "检测到连续重复或交替工具空转，保留当前文件并交由验证阶段提供具体修复反馈。"
            graph._emit(state, "coder", "error", coding_summary)
            break
        if data.get("done") is True:
            if action_failed or preflight_issues:
                observations.append("本轮仍有失败动作或待修合同，不能标记完成；请修正后运行实际验证。")
                continue
            coding_ok = True
            coding_summary = graph._clean_summary(data.get("summary"), "Coding 阶段已完成。")
            graph._emit(state, "coder", "done", coding_summary)
            break
        if idle_after_mutation >= 3:
            if len(successful_verification_commands) >= 2 and not failed_tests:
                coding_ok = True
                coding_summary = (
                    "Coding 已完成实现并通过多项测试、构建或烟测；"
                    "模型随后连续只读确认，现交由独立 Verifier 做最终需求审查。"
                )
                graph._emit(state, "coder", "done", coding_summary)
            else:
                coding_summary = "Coding 已产生文件变更，但连续三轮只有失败、读取或状态查询，没有新的实现进展，也没有明确完成任务。"
            break
    if not coding_ok and not coding_summary:
        coding_summary = "Coding 在最大 ReAct 步数内没有明确完成任务。"
    unique_changes = sorted(dict.fromkeys(changes))
    safe_commands = graph.model_store.redact(commands)
    safe_summary = graph.model_store.redact_text(coding_summary)
    graph.memory.append(
        state["workdir"],
        state["session_id"],
        "coder",
        "react",
        "summary",
        f"变更文件：{unique_changes}",
        {"commands": safe_commands, "ok": coding_ok, "summary": safe_summary},
    )
    return {
        **state,
        "changes": unique_changes,
        "commands": safe_commands,
        "verification_commands": list(dict.fromkeys(verification_commands)),
        "retry": retry + 1,
        "coding_ok": coding_ok,
        "coding_summary": safe_summary,
        "coded_guidance_revision": int(state.get("guidance_revision", 0)),
        "error": "" if coding_ok else safe_summary,
    }
