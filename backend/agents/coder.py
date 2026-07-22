from __future__ import annotations

import json
import re
import shlex
from typing import TYPE_CHECKING

from backend.agents.prompts import CODER_PROMPT, LONGCAT_CODER_SUFFIX
from backend.agents.types import AgentState
from backend.tools import FsTool, GitTool, ShellTool
from llm import LlmClient, LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


# 这些后缀用于从验证反馈中识别真正需要修复的代码和配置文件。
_CODE_FILE_PATTERN = re.compile(
    r"(?<![\w.-])([\w./-]+\.(?:css|html|js|json|md|py|sh|toml|ts|tsx|yaml|yml))(?![\w.-])"
)

# FILE_CONTEXT_LIMIT 是单个当前文件快照的最大字符数。常见源码应完整保留，避免首尾压缩恰好丢掉中部待修改代码。
FILE_CONTEXT_LIMIT = 24_000
# SNAPSHOT_BUDGET 控制所有当前文件快照的总量；超大文件仍由 read_file 的 start/max_chars 分段读取。
SNAPSHOT_BUDGET = 48_000
# CODER_OUTPUT_TOKENS 限制单轮 ReAct 工具参数。提示词已要求单次写入不超过 12000 字符，
# 6000 token 足以容纳该动作和协议标签，同时可阻止模型失控生成超长完整文件。
CODER_OUTPUT_TOKENS = 6_000


def _middle_trim(text: str, limit: int) -> str:
    """保留长文本首尾，避免错误根因或文件末尾代码被截掉。"""

    if len(text) <= limit:
        return text
    # head_len 给文件开头更多预算，tail_len 保留测试错误和收尾代码。
    head_len = max(1, int(limit * 0.7))
    tail_len = max(1, limit - head_len)
    return text[:head_len] + "\n... 中间内容已省略 ...\n" + text[-tail_len:]


def _test_brief(tests: list[dict]) -> str:
    """把测试结果压缩成可执行的问题清单，不重复发送完整 traceback。"""

    lines: list[str] = []
    for item in tests:
        if item.get("ok"):
            continue
        # cmd 标明问题来自编译、测试、语义审查还是 Coding 状态。
        cmd = str(item.get("cmd") or "检查")
        issues = item.get("issues")
        if isinstance(issues, list) and issues:
            for issue in issues[:12]:
                lines.append(f"- [{cmd}] {_middle_trim(str(issue), 700)}")
            continue
        # 普通命令失败只保留首尾，冗长调用栈不会反复污染每一步上下文。
        output = str(item.get("out") or item.get("err") or "失败但没有输出")
        lines.append(f"- [{cmd}] {_middle_trim(output, 1_400)}")
    return _middle_trim("\n".join(lines) or "- 暂无失败项", 8_000)


def _repair_paths(tests: list[dict], changes: list[str], current_files: list[str]) -> list[str]:
    """按验证反馈提取修复文件，避免重试时无差别加载全部变更。"""

    current = set(current_files)
    found: list[str] = []
    # feedback_text 只用于提取文件路径，不直接进入模型上下文。
    feedback_text = json.dumps(tests, ensure_ascii=False)
    for match in _CODE_FILE_PATTERN.findall(feedback_text):
        rel = match.lstrip("./")
        if rel in current and rel not in found:
            found.append(rel)
    # 若错误文本没有明确文件名，再从本任务变更中补少量核心文件。
    for rel in changes:
        if rel in current and rel not in found:
            found.append(rel)
    return found[:6]


def _task_brief(state: AgentState) -> str:
    """构造 Coding 专用任务摘要，排除与当前实现无关的完整会话历史。"""

    ctx = state.get("context")
    if not ctx:
        return _middle_trim(str(state.get("text", "")), 18_000)
    constraints = "\n".join(f"- {item}" for item in ctx.constraints)
    return _middle_trim(f"任务目标：\n{ctx.goal}\n\n约束：\n{constraints}", 20_000)


def _repo_brief(repo: dict, *, include_snippets: bool) -> str:
    """压缩仓库摘要；修复轮次只需要当前清单，不再携带初始旧代码。"""

    files = list(repo.get("files") or [])
    stack = "、".join(repo.get("stack") or []) or "未知"
    lines = [
        "技术栈：" + stack,
        "文件清单：" + "、".join(files[:120]) + (f"（共 {len(files)} 个）" if len(files) > 120 else ""),
    ]
    if include_snippets:
        # fresh_snippets 只保留仓库读取智能体选出的少量入口片段，总量受硬预算限制。
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
    # 仓库读取智能体已经提供稳定清单，Coding 不应再用这些命令绕过 list_files 限制。
    program = argv[0].rsplit("/", 1)[-1].lower()
    return program in {"find", "ls", "pwd", "tree"}


def _observation_brief(action: dict, observation: dict) -> str:
    """把工具返回值转换为下一轮模型可直接判断的权威结果。"""

    # status 明确写出成功或失败，避免“命令没有 stdout”被模型误认为尚未执行。
    status = "成功" if observation.get("ok") else "失败"
    tool = str(action.get("tool") or "工具")
    target = str(action.get("path") or action.get("cmd") or "").strip()
    prefix = f"工具结果 [{status}] {tool}" + (f"：{target}" if target else "")
    result = observation.get("result")
    if isinstance(result, dict):
        # code/out/err 是命令验证最关键的三个字段；空输出也要显式说明。
        code = result.get("code")
        output = str(result.get("out") or result.get("err") or "").strip()
        detail = f"退出码={code}；" + (_middle_trim(output, 1_200) if output else "命令无输出，已正常结束")
        return f"{prefix}。{detail}"
    text = str(observation.get("text") or "").strip()
    return f"{prefix}。{_middle_trim(text, 1_400) if text else '工具未返回附加文本'}"


def coder(graph: AgentGraph, state: AgentState) -> AgentState:
    """Coding 智能体：通过 ReAct 循环调用工具，真实修改项目文件。"""

    # retry 是当前 Coding 轮次，验证失败回到 coder 时会递增。
    retry = int(state.get("retry", 0))
    graph._emit(state, "coder", "start", f"Coding 智能体开始 ReAct 执行，第 {retry + 1} 轮")
    # fs/shell/git 是 Coding 智能体可调用的工具集合。
    fs = FsTool(state["workdir"])
    shell = ShellTool(state["workdir"])
    git = GitTool(state["workdir"])
    # client 是 coder 智能体对应的模型客户端。
    client = graph._client("coder", state)
    # model_name 用于选择供应商稳定支持的输出协议，不改变用户配置的模型映射。
    model_name = str(getattr(getattr(client, "cfg", None), "model", "")).lower()
    # observations 只保存精简后的工具结果，避免把完整文件和 traceback 每一步重复发送。
    observations: list[str] = []
    # file_observations 保存最近一次读取或写入后自动刷新的当前文件快照。
    file_observations: dict[str, str] = {}
    # seen_actions 保存已经执行或明确拒绝的动作签名，防止模型无视观察反复消耗相同工具。
    seen_actions: set[str] = set()
    # progress_items 记录工具、路径或命令，模型每一步都能看到稳定的任务进度账本。
    progress_items: list[str] = []
    # read_paths 记录本轮已经读取的文件，修改现有文件前必须命中这里。
    read_paths: set[str] = set()
    # unchecked_rewrites 记录已整体重写但尚未经过命令验证的文件，防止模型反复生成同一完整文件。
    unchecked_rewrites: set[str] = set()
    # mutation_count 记录真实写入或修改次数，用于识别“已经产出代码却持续只读”的失控循环。
    mutation_count = 0
    # idle_after_mutation 统计最近连续多少轮没有新的写入或命令进展。
    idle_after_mutation = 0
    # repo_files 来自前置仓库读取节点，已有索引时 Coder 无需再次 list_files。
    repo = state.get("repo") or {}
    repo_files = list(repo.get("files") or [])
    # current_files 是本轮开始时的磁盘事实；修复轮次依靠它避免把已有项目误判为空项目。
    current_files = repo_files
    if state.get("repo") is not None:
        preview = "、".join(repo_files[:20]) or "空项目"
        observations.append(f"仓库读取节点已完成文件扫描：{preview}。不要调用 list_files，直接读取目标文件或开始写入。")
    if retry > 0:
        # 验证失败后的修复轮次必须以当前磁盘为准，不能沿用最初仓库快照从头覆盖项目。
        current_files = fs.list(max_files=300)
        observations.append(
            "这是验证失败后的修复轮次。项目文件已经存在，只修复失败项，禁止从头重建、重复创建或扫描目录。"
        )
        # repair_paths 只预读验证反馈明确涉及的文件，最多 6 个；无关变更不会占用模型上下文。
        repair_paths = _repair_paths(state.get("tests", []), state.get("changes", []), current_files)
        for rel in repair_paths:
            try:
                file_observations[rel] = fs.read(rel, FILE_CONTEXT_LIMIT)
                read_paths.add(rel)
            except (OSError, ValueError):
                continue
    # changes 继承已有变更列表，验证失败重试时不会丢掉前一轮变更记录。
    changes = list(state.get("changes", []))
    # commands 继承已有命令列表。
    commands = list(state.get("commands", []))
    # failed_tests 是每一步都保留的稳定修复清单，不会被后续工具观察挤出上下文。
    failed_tests = ""
    if state.get("tests") and not state.get("tests_ok", True):
        # 如果上一轮验证失败，把测试结果作为观察反馈给模型修复。
        failed_tests = _test_brief(state["tests"])
        if all(item.get("cmd") == "Coding 完成状态" for item in state["tests"] if not item.get("ok")):
            observations.append("上一轮只有 Coding 完成状态失败：当前文件若已满足需求，应直接返回 done=true，不要重新扫描或重建。")

    # task_brief/repo_brief 在循环外只构造一次，保证每一步目标一致且输入体积稳定。
    task_brief = _task_brief(state)
    repo_brief = _repo_brief(repo, include_snippets=retry == 0)

    def prune_snapshots_after_timeout() -> None:
        """模型超时后只保留任务明确点名的文件，缩小下一轮输入并允许其他文件按需重读。"""

        # named_paths 按当前快照顺序收集任务文字中出现的相对路径；通常就是正在修复的目标文件。
        named_paths = [path for path in file_observations if path in task_brief]
        # 如果任务没有写出文件名，保留最后读取的一份，避免把全部代码再次发送给模型。
        keep_paths = set(named_paths[-2:] or list(file_observations)[-1:])
        for observed_path in list(file_observations):
            if observed_path in keep_paths:
                continue
            file_observations.pop(observed_path, None)
            read_paths.discard(observed_path)
            # 被裁掉的快照后续可以按需重新读取，旧 read_file 签名不能阻止恢复动作。
            stale_signatures = {
                signature
                for signature in seen_actions
                if '"tool": "read_file"' in signature and f'"path": "{observed_path}"' in signature
            }
            seen_actions.difference_update(stale_signatures)
        observations.append(
            "模型超时后已收缩文件上下文，仅保留：" + ("、".join(sorted(keep_paths)) if keep_paths else "无")
        )

    # coding_ok 只有模型明确 done 且最后一批动作都成功时才会变为 True。
    coding_ok = False
    # coding_summary 保存模型明确给出的完成摘要或循环失败原因。
    coding_summary = ""
    # parse_failures 统计连续结构化输出失败次数，偶发格式错误不会立刻终止任务。
    parse_failures = 0
    # empty_steps 统计模型返回合法结构但没有动作和完成标记的次数，用于触发 LongCat 普通文本降级。
    empty_steps = 0
    # 单轮 Coding 最多执行 16 次模型-工具循环；长项目需要足够的分批写入空间，同时仍有硬上限防失控。
    max_steps = 16
    for step in range(1, max_steps + 1):
        # recent_observations 只保留最近 6 条精简事件；文件内容由 file_observations 单独管理。
        recent_observations = [_middle_trim(item, 1_200) for item in observations[-6:]]
        # current_snapshots 只携带仍然有效的最近文件快照，总字符数设硬上限。
        current_snapshots: list[str] = []
        snapshot_budget = SNAPSHOT_BUDGET
        for path, content in list(file_observations.items())[-4:]:
            if snapshot_budget <= 0:
                break
            snapshot = _middle_trim(content, min(FILE_CONTEXT_LIMIT, snapshot_budget))
            current_snapshots.append(f"[{path} 当前内容]\n{snapshot}")
            snapshot_budget -= len(snapshot)
        # messages 是本轮发给 Coding 模型的上下文。
        progress = "、".join(progress_items[-30:]) or "尚未执行成功工具"
        manifest = "、".join(current_files[:120]) or "空项目"
        readable = "、".join(sorted(read_paths)) or "无"
        finish_instruction = ""
        if step >= max_steps - 3 and (mutation_count or changes):
            finish_instruction = (
                "\n已进入收尾阶段：禁止仅为确认而读取文件。若实现已完成，必须返回 actions=[]、done=true；"
                "若仍有缺口，只执行直接修复缺口的写入或测试命令。\n"
            )
        messages = [
            {
                "role": "system",
                "content": CODER_PROMPT + (LONGCAT_CODER_SUFFIX if "longcat" in model_name else ""),
            },
            {
                "role": "user",
                "content": f"当前是第 {step}/{max_steps} 步。当前模式：{'失败修复' if retry else '首次实现'}。\n"
                + f"当前磁盘文件：{manifest}\n"
                + f"已完成进度：{progress}\n"
                + f"已读取且当前可直接修改的文件：{readable}\n"
                + "必须根据失败清单和已有观察产生新进展；禁止重复扫描、重复创建已有文件或重新开始任务。\n"
                + "最近工具观察是已经发生的权威事实；标记为成功的命令无需再次确认，不得声称没有看到结果。\n"
                + ("当前是失败修复轮次：验证失败清单的优先级高于原始任务中的整体重写等实现方式，只精确修复清单中的问题。\n" if retry else "")
                + "若目标文件出现在“已读取且当前可直接修改”列表，禁止再次 read_file；短改动直接 replace_file，长函数直接 replace_block。\n"
                + "run_command 已经在项目工作目录执行，命令中禁止使用 cd，也不要添加工作目录前缀。\n"
                + "每轮最多返回 3 个紧密相关动作；修复轮次优先修改失败清单中的文件并运行对应测试。\n\n"
                + finish_instruction
                + task_brief
                + "\n\n仓库摘要：\n"
                + repo_brief
                + ("\n\n必须解决的验证失败：\n" + failed_tests if failed_tests else "")
                + ("\n\n最近读取的当前文件：\n" + "\n\n".join(current_snapshots) if current_snapshots else "")
                + "\n\n最近工具观察：\n"
                + "\n".join(recent_observations),
            },
        ]
        try:
            # data 是模型返回的 ReAct JSON。
            if "longcat" in model_name or empty_steps >= 2:
                # LongCat 的普通文本模式会稳定返回原生工具标签；直接使用该协议，避免先用
                # response_format 连续得到空对象。其他模型只有连续空响应后才进入兼容模式。
                if isinstance(client, LlmClient):
                    data = client.chat_json(messages, plain_text=True, max_tokens=CODER_OUTPUT_TOKENS)
                else:
                    data = client.chat_json(messages, plain_text=True)
            else:
                if isinstance(client, LlmClient):
                    data = client.chat_json(messages, max_tokens=CODER_OUTPUT_TOKENS)
                else:
                    data = client.chat_json(messages)
            graph._add_tokens(state, client.last_usage.total)
        except LlmError as exc:
            parse_failures += 1
            # error_text 只保留错误前 240 字符，完整模型残片不会进入前端和记忆。
            error_text = graph._trim(str(exc), 240)
            protocol = "LongCat 原生工具标签" if "longcat" in model_name else "约定 JSON"
            if "响应超时" in error_text or "超过总时长" in error_text:
                # 超时通常来自一次生成过多完整文件内容。下一轮明确缩小动作，而不是原样重放。
                prune_snapshots_after_timeout()
                observations.append(
                    f"第 {step} 轮模型响应超时：{error_text}。下一轮只返回一个小动作，"
                    f"短改动使用 replace_file，长函数使用 replace_block，并严格使用{protocol}。"
                )
            else:
                observations.append(f"第 {step} 轮输出格式无效：{error_text}。请严格使用{protocol}。")
            graph._emit(
                state,
                "coder",
                "error",
                f"模型输出格式无效，正在重试（{parse_failures}/3）：{error_text}",
            )
            if parse_failures >= 3:
                coding_summary = "Coding 模型连续三次没有返回有效执行指令。"
                break
            continue
        parse_failures = 0
        # thought 是模型本轮判断，会进入事件流方便用户观察。
        thought = data.get("thought", "")
        if thought:
            graph._emit(state, "coder", "thought", thought)
        # actions 是模型请求执行的工具动作列表。
        actions = graph._normalize_actions(data.get("actions"))
        if len(actions) > 3:
            # 只执行前 3 个动作，剩余动作留到下一轮，避免一个错误批次破坏多个文件。
            observations.append(f"模型本轮返回 {len(actions)} 个动作，只执行前 3 个；其余动作请下一轮继续。")
            actions = actions[:3]
        if not actions and data.get("done"):
            empty_steps = 0
            coding_ok = True
            coding_summary = graph._clean_summary(data.get("summary"), "Coding 阶段已完成。")
            break
        if not actions:
            empty_steps += 1
            observations.append("本轮既没有工具动作也没有完成标记，请读取、修改或验证后再继续。")
            graph._emit(
                state,
                "coder",
                "error",
                f"模型未给出可执行动作，正在重试（{empty_steps}/5）"
                + ("；当前使用普通文本兼容模式" if "longcat" in model_name or empty_steps >= 2 else ""),
            )
            if empty_steps >= 5:
                coding_summary = "Coding 模型连续五次没有给出工具动作或完成标记。"
                break
            continue
        empty_steps = 0
        # action_failed 表示本批动作至少有一个失败，模型必须观察并修复后才能 done。
        action_failed = False
        # batch_progress 表示本批至少有一个成功且不是纯 Git 状态查询的动作。
        batch_progress = False
        # mutated_paths 收集本批已经改动的文件。一个批次内的多个精确替换共享批次起始快照；
        # 批次结束后由编排器直接刷新磁盘快照，避免再消耗一次模型调用请求 read_file。
        mutated_paths: set[str] = set()
        for action in actions:
            # signature 用稳定 JSON 表示本次动作，相同参数和内容的动作不会重复执行。
            signature = json.dumps(action, ensure_ascii=False, sort_keys=True)
            tool = str(action.get("tool") or "")
            path = str(action.get("path") or "")
            if signature in seen_actions and tool not in {"git_diff", "git_status"}:
                observation = {
                    "ok": False,
                    "text": f"拒绝重复动作：{action.get('tool')} 已执行或拒绝过。下一步必须改用其他动作推进任务。",
                }
            elif tool == "write_file" and path in unchecked_rewrites:
                observation = {
                    "ok": False,
                    "text": (
                        f"拒绝重复整体重写：{path} 已成功写入但尚未验证。"
                        "请先运行该文件对应的语法检查、测试或构建；只有验证暴露具体问题后才能继续修改。"
                    ),
                }
            elif tool == "list_files" and state.get("repo") is not None:
                observation = {
                    "ok": False,
                    "text": "拒绝重复调用 list_files：仓库读取节点已经提供文件索引。请改用 read_file、write_file 或其他实际推进任务的工具。",
                }
            elif tool == "run_command" and state.get("repo") is not None and _is_discovery_command(str(action.get("cmd", ""))):
                observation = {
                    "ok": False,
                    "text": "拒绝使用 shell 重复枚举仓库：当前文件清单已经在提示中。请直接读取目标文件、写入实现或运行验证命令。",
                }
            elif (
                tool == "write_file"
                and path
                and fs.safe(path).exists()
                # 单用户本地场景允许在完整读取后整体重写。未读取时仍拒绝，避免模型盲目覆盖用户文件。
                and path not in read_paths
            ):
                observation = {
                    "ok": False,
                    "text": (
                        f"拒绝覆盖现有文件：{path}。请先 read_file 获取当前完整内容；"
                        "小改动使用 replace_file，确实需要整体整理时才使用 write_file。"
                    ),
                }
            elif tool in {"append_file", "replace_block", "replace_file"} and path and fs.safe(path).exists() and path not in read_paths:
                observation = {
                    "ok": False,
                    "text": f"修改前必须先读取当前文件：{path}。请先调用 read_file。",
                }
            else:
                # observation 是工具执行后的结构化观察。
                observation = graph._do_action(action, fs, shell, git)
            seen_actions.add(signature)
            # read_file 的完整内容只进入文件快照，普通观察只保留简短结果。
            if tool == "read_file" and observation.get("ok") and path:
                marker = "\n"
                raw_text = str(observation["text"])
                file_observations[path] = _middle_trim(raw_text.split(marker, 1)[-1], FILE_CONTEXT_LIMIT)
                observations.append(f"已读取当前文件：{path}")
            else:
                observations.append(_observation_brief(action, observation))
            action_failed = action_failed or not bool(observation.get("ok"))
            if observation.get("ok"):
                detail = path or str(action.get("cmd") or "")
                progress_items.append(f"{tool}:{detail}" if detail else tool)
                if tool in {"append_file", "assert_text_absent", "replace_block", "replace_file", "run_command", "write_file"}:
                    batch_progress = True
                if tool == "write_file" and path:
                    unchecked_rewrites.add(path)
                elif tool == "run_command":
                    # 一次成功验证后允许根据新的客观结果继续整体修复。
                    unchecked_rewrites.clear()
                if tool == "read_file" and path:
                    read_paths.add(path)
                elif tool in {"append_file", "replace_block", "replace_file", "write_file"} and path:
                    # 同一批的后续精确修改仍可基于本批起始快照；批次结束后再统一失效。
                    mutated_paths.add(path)
                    if path not in current_files:
                        current_files.append(path)
                if tool in {"append_file", "replace_block", "replace_file", "write_file", "run_command"}:
                    mutation_count += 1
            if observation.get("file"):
                changes.append(str(observation["file"]))
            if observation.get("cmd"):
                commands.append(str(observation["cmd"]))
            graph._emit(state, "coder", "tool", observation["text"], data=observation)
        for mutated_path in mutated_paths:
            try:
                # 工具成功写入后，以磁盘为唯一事实源重新加载文件。下一轮模型可直接基于这个
                # 最新快照继续 replace_file，不需要经过“修改 -> 再读 -> 再修改”的空转循环。
                file_observations[mutated_path] = fs.read(mutated_path, FILE_CONTEXT_LIMIT)
                read_paths.add(mutated_path)
                observations.append(f"已自动刷新修改后的当前文件：{mutated_path}；下一轮禁止重复读取，可直接继续修改。")
            except (OSError, ValueError):
                # 极少数情况下文件在工具返回后被外部删除或变得不可读，此时撤销可修改标记，
                # 后续动作仍会被修改前读取守卫拦截，避免模型基于过期内容继续覆盖。
                read_paths.discard(mutated_path)
                file_observations.pop(mutated_path, None)
        if mutated_paths:
            # 文件状态变化后，之前运行过的同一测试已经不再是重复动作，必须允许重新验证修复结果。
            seen_actions = {item for item in seen_actions if '"tool": "run_command"' not in item}
        # 同一批中只要已有真实写入或验证命令成功，就属于有效推进；
        # 另一个并行动作失败会反馈给下一轮修复，但不能把已落盘进展误算为空转。
        if batch_progress:
            idle_after_mutation = 0
        elif mutation_count or changes:
            idle_after_mutation += 1
        if data.get("done"):
            if action_failed:
                observations.append("本轮存在失败动作，不能标记完成；请根据观察修复。")
                continue
            coding_ok = True
            coding_summary = graph._clean_summary(data.get("summary"), "Coding 阶段已完成。")
            graph._emit(state, "coder", "done", coding_summary)
            break
        if idle_after_mutation >= 6:
            coding_summary = "Coding 已产生文件变更，但连续六轮只有失败、读取或状态查询，没有新的实现进展，也没有明确完成任务。"
            break
    if not coding_ok and not coding_summary:
        coding_summary = "Coding 在最大 ReAct 步数内没有明确完成任务。"
    # unique_changes 去重并排序，保证最终结果稳定。
    unique_changes = sorted(dict.fromkeys(changes))
    graph.memory.append(
        state["workdir"],
        state["session_id"],
        "coder",
        "react",
        "summary",
        f"变更文件：{unique_changes}",
        {"commands": commands, "ok": coding_ok, "summary": coding_summary},
    )
    return {
        **state,
        "changes": unique_changes,
        "commands": commands,
        "retry": retry + 1,
        "coding_ok": coding_ok,
        "coding_summary": coding_summary,
        "error": "" if coding_ok else coding_summary,
    }
