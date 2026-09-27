from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

from backend.agents.prompts import PLANNER_PROMPT
from backend.agents.types import AgentState
from llm import LlmBalanceError, LlmClient, LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


# Planner 与 Coding 一样会遇到限流、网关 5xx 和短暂连接故障；这些错误不应立刻降级成通用计划。
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "连接", "Connect", "响应超时", "ReadTimeout")
_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0)
_MAX_PLAN_REPAIRS = 3


def _is_transient(error: str) -> bool:
    """判断 Plan 模型错误是否适合退避重试。"""

    return any(marker in error for marker in _TRANSIENT_MARKERS)


def _schema_fields(repo: dict) -> list[str]:
    """从 Repo 摘要中的 JSON Schema 提取真实对象字段。"""

    snippets = repo.get("snippets") if isinstance(repo, dict) else {}
    if not isinstance(snippets, dict):
        return []
    for path, content in snippets.items():
        if not str(path).endswith(".schema.json"):
            continue
        try:
            schema = json.loads(str(content))
        except json.JSONDecodeError:
            continue
        definitions = schema.get("$defs") if isinstance(schema, dict) else {}
        exercise = definitions.get("exercise") if isinstance(definitions, dict) else {}
        properties = exercise.get("properties") if isinstance(exercise, dict) else {}
        if isinstance(properties, dict):
            return [str(field) for field in properties]
    return []


def _mentions_field(text: str, field: str) -> bool:
    """识别 Markdown 中的完整字段名及 `field.zh` 这类真实子字段写法。"""

    escaped = re.escape(field)
    return bool(
        re.search(
            rf"(?:`{escaped}(?:`|\.)|\"{escaped}\"|\|\s*{escaped}\s*\||^\s*[-*]\s*{escaped}\s*[:：])",
            text,
            re.MULTILINE,
        )
    )


def _normalize_chat_contract(markdown: str, goal: str) -> str:
    """把模型反复产生的聊天合同矛盾归一化为服务端可信 system 方案。"""

    if "第一次提问" not in goal and not ("首次" in goal and "历史对话" in goal):
        return markdown
    normalized: list[str] = []
    canonical_added = False
    for line in markdown.splitlines():
        lower = line.lower()
        frontend_system_creation = (
            "前端" in line
            and "system" in lower
            and any(word in line for word in ("初始化", "构造", "创建"))
            and not any(word in line for word in ("不初始化", "不构造", "不创建", "禁止"))
        )
        renders_system = (
            "system" in lower
            and any(word in line for word in ("渲染", "显示"))
            and not any(word in line for word in ("不渲染", "不显示", "禁止渲染", "禁止显示"))
        )
        filters_system = any(
            marker in lower
            for marker in (
                "不含 system",
                "排除 system",
                "withoutsystem",
                "role !== 'system'",
                'role !== "system"',
            )
        )
        appends_current_user = bool(
            re.search(
                r"(?i)(?:messages|history|currentmessages)\s*\.(?:push|append)\s*\(\s*\{[^}]*role[^}]*user",
                line,
            )
        ) or bool(
            re.search(
                r"(?i)(?:追加|添加).{0,10}(?:新|当前).{0,8}user.{0,24}(?:messages|历史|回传)",
                line,
            )
        )
        if frontend_system_creation or renders_system or filters_system or appends_current_user:
            if not canonical_added:
                indent = line[: len(line) - len(line.lstrip())]
                normalized.append(
                    indent
                    + "- 前端不创建 system、不初始化 system、不过滤 system，也不显示 system；首轮仅发送 "
                    "`exercise_id`、独立问题字段和空 `messages`；客户端保存服务端返回的完整 messages"
                    "（包含 system/user/assistant），后续原样回传，界面只渲染 user/assistant。"
                )
                canonical_added = True
            continue
        normalized.append(line)
    return "\n".join(normalized)


def _plan_quality_issues(graph: AgentGraph, state: AgentState, markdown: str) -> list[str]:
    """按用户硬约束和仓库事实检查计划初稿，返回需要模型修订的问题。"""

    goal = graph._plan_goal(state)
    goal_lower = goal.lower()
    plan_lower = markdown.lower()
    issues: list[str] = []

    # 用户明确给出的配置字段名属于 HTTP/配置合同，不允许模型自行改名。
    for field in ("api_key", "base_url", "model_name"):
        if field in goal_lower and field not in plan_lower:
            issues.append(f"配置计划遗漏或改名了用户指定字段 `{field}`")
    if "apikey" in goal_lower and "api_key" not in plan_lower and "apikey" not in plan_lower:
        issues.append("配置计划遗漏了用户指定的 API Key 字段")
    if re.search(r"(?i)\b(?:sk-|ak_|agt_)[A-Za-z0-9._-]{12,}\b", markdown):
        issues.append("计划初稿复制了真实凭据，必须改成不含真实值的占位说明")

    # Repo 已读到 schema 时，字段清单必须完整使用真实名称。
    fields = _schema_fields(state.get("repo") or {})
    missing_fields = [field for field in fields if not _mentions_field(markdown, field)]
    if missing_fields:
        issues.append("数据字段说明遗漏真实 schema 字段：" + "、".join(missing_fields))

    data_site = "数据集" in goal and ("网站" in goal or "web" in goal_lower)
    if data_site:
        # 前端不能只写“读取某 JSON”而不说明浏览器如何访问工作区外的数据。
        if "/api/exercises" not in markdown:
            issues.append("缺少可执行的数据访问合同：应规划 `/api/exercises` 等服务端接口")
        if "pytest" not in plan_lower:
            issues.append("缺少后端核心接口和数据逻辑的 pytest 自动化测试")

    # 一旦计划选择 React + TypeScript，就必须规划独立类型检查和生产构建。
    if "react" in plan_lower and "typescript" in plan_lower:
        if "typecheck" not in plan_lower and "tsc --noemit" not in plan_lower and "tsc --noemit" not in plan_lower.replace(" ", ""):
            issues.append("React + TypeScript 计划缺少独立 `typecheck`/`tsc --noEmit` 检查")
        if "npm run build" not in plan_lower:
            issues.append("前端计划缺少 `npm run build` 生产构建验收")

    # 用户要求完整详情时，计划必须明确逐项渲染原始对象，而非只列基础信息。
    complete_detail_claim = any(phrase in markdown for phrase in ("所有字段", "全部原始字段", "逐字段")) or bool(
        re.search(r"(?:所有|完整)\s*\d+\s*个?原始字段", markdown)
    )
    if "所有详细信息" in goal and not complete_detail_claim:
        issues.append("详情弹窗未明确展示动作 JSON 的所有原始字段")
    if "所有详细信息" in goal and fields and "左侧" in markdown:
        detail_start = markdown.find("左侧")
        detail_section = markdown[detail_start : detail_start + 2_500]
        dynamic_complete = bool(
            re.search(r"(?:动态遍历|逐项遍历|递归渲染).{0,40}(?:所有|全部|完整).{0,20}字段", detail_section)
            or re.search(r"左侧.{0,80}(?:所有|全部|完整).{0,20}(?:原始)?字段", detail_section)
        )
        missing_detail_fields = [] if dynamic_complete else [
            field for field in fields if not _mentions_field(detail_section, field)
        ]
        if missing_detail_fields:
            issues.append(
                "详情左侧的明确字段清单仍有遗漏（媒体路径字段也应作为文本展示）："
                + "、".join(missing_detail_fields)
            )
    if fields and "/api/exercises/{" in markdown:
        detail_api_start = markdown.find("/api/exercises/{")
        next_heading = markdown.find("\n### ", detail_api_start + 1)
        detail_api_section = markdown[
            detail_api_start : next_heading if next_heading > detail_api_start else detail_api_start + 3_500
        ]
        complete_api_claims = (
            "全部原始字段",
            "所有原始字段",
            "原始对象完整返回",
            "完整对象",
            "完整详情",
            "完整记录",
            "返回完整动作详情",
            "返回完整动作对象",
            "返回完整记录",
            "包含全部字段",
            "包含所有字段",
        )
        complete_api_claim = any(phrase in detail_api_section for phrase in complete_api_claims) or bool(
            re.search(r"(?:返回|包含|提供).{0,30}(?:所有|全部|完整).{0,20}(?:字段|对象|记录|详情)", detail_api_section)
        )
        if not complete_api_claim:
            missing_api_fields = [field for field in fields if not _mentions_field(detail_api_section, field)]
            if missing_api_fields:
                issues.append("动作详情 API 合同遗漏原始字段：" + "、".join(missing_api_fields))

    # 无状态多轮对话必须把首轮 system 连同完整历史交还客户端保存。
    if "第一次提问" in goal or ("首次" in goal and "历史对话" in goal):
        complete_history = "完整 messages" in plan_lower or bool(
            re.search(r"完整.{0,12}(?:messages|消息历史)", plan_lower)
        )
        if not complete_history or not all(role in plan_lower for role in ("system", "user", "assistant")):
            issues.append("未明确首轮响应返回包含 system/user/assistant 的完整 messages")
        for phrase, issue in (
            ("客户端保存", "未明确客户端保存但不渲染 system 历史"),
            ("服务端", "未明确由服务端根据动作数据构造和验证 system 上下文"),
        ):
            if phrase.lower() not in plan_lower:
                issues.append(issue)
        if any(
            "system" in line.lower()
            and "前端" in line
            and any(word in line for word in ("初始化", "构造", "创建"))
            and not any(word in line for word in ("不初始化", "不构造", "不创建", "禁止初始化", "禁止构造", "禁止创建"))
            for line in markdown.splitlines()
        ):
            issues.append("计划又让前端初始化 system，与服务端可信注入方案冲突")
        contract_fields_present = all(f'"{field}"' in markdown for field in ("exercise_id", "question", "messages"))
        if not contract_fields_present:
            issues.append(
                "聊天接口缺少一份前后端统一请求体合同；必须固定 exercise_id、问题字段和完整 messages 字段名"
            )
        empty_first_messages = bool(re.search(r'"messages"\s*:\s*\[\s*\]', markdown)) or bool(
            re.search(r"messages.{0,16}(?:为空数组|为空|空数组)", markdown, re.IGNORECASE)
        )
        if not empty_first_messages:
            issues.append("首轮请求示例必须把当前问题只放在 question，并明确 messages 为空数组")
        if any(
            marker in plan_lower
            for marker in (
                "不含 system",
                "排除 system",
                "withoutsystem",
                "role !== 'system'",
                'role !== "system"',
            )
        ):
            issues.append("客户端不得过滤 system；后续必须原样回传服务端返回的完整 messages")
        if any(
            "system" in line.lower()
            and any(word in line for word in ("渲染", "显示"))
            and not any(word in line for word in ("不渲染", "不显示", "禁止渲染", "禁止显示"))
            for line in markdown.splitlines()
        ):
            issues.append("system 消息只能由客户端保存用于回传，界面不得渲染或显示")
        if re.search(
            r"(?i)(?:messages|history|currentmessages)\s*\.(?:push|append)\s*\(\s*\{[^}]*role[^}]*user",
            markdown,
        ) or re.search(
            r"(?i)(?:追加|添加).{0,10}(?:新|当前).{0,8}user.{0,24}(?:messages|历史|回传)",
            markdown,
        ):
            issues.append("使用独立 question 字段时，客户端不得先把当前 user 再追加到 messages，避免服务端重复添加")
        if re.search(
            r"(?:服务端|后端).{0,20}(?:保存|维护|存储).{0,30}(?:messages|消息|历史|对话)",
            markdown,
            re.IGNORECASE,
        ) or "exercise_contexts" in markdown:
            issues.append("聊天接口必须无状态：服务端不得保存 messages 历史，应只验证并使用客户端回传的完整 messages")

    if "videos" in goal_lower and "images-exercises" in goal_lower:
        excludes_images = bool(re.search(r"(?:不使用|不引用|禁止(?:使用|引用)?).{0,24}(?:`?/?images-exercises)", plan_lower))
        if "videos" not in plan_lower or not excludes_images:
            issues.append("未把仅使用 videos GIF、禁止 images-exercises 静图写成明确媒体合同")

    selected_skills = {str(item.id).lower() for item in state.get("skills") or []}
    if "web-design-guidelines" in selected_skills:
        required_guidelines = ("aria-label", "focus-visible", "prefers-reduced-motion")
        missing_guidelines = [item for item in required_guidelines if item not in plan_lower]
        if missing_guidelines:
            issues.append("所选 web-design-guidelines 关键验收遗漏：" + "、".join(missing_guidelines))

    attachments = list(state.get("attachments") or [])
    if attachments:
        missing_names = [
            str(item.get("name") or item.get("id") or "未命名附件")
            for item in attachments
            if str(item.get("name") or "") not in markdown
        ]
        if missing_names:
            issues.append("计划未逐张关联图片附件：" + "、".join(missing_names))
        if not re.search(r"(?i)(?:参考编号|reference\s*(?:id|code)|FIT[-_][A-Z0-9_-]+)", markdown):
            issues.append("图片附件可能含可见参考编号；必须重新查看原图并逐张准确抄录，不能只写文件名")

    implementation_fences = sum(
        markdown.lower().count(f"```{language}") for language in ("python", "javascript", "typescript", "html", "css")
    )
    if implementation_fences > 2 or len(markdown) > 24_000:
        issues.append("计划内嵌了过多近似成品源码；应保留合同、步骤和短示例，把完整实现交给 Coding 智能体")

    return issues


def _should_auto_answer_plan_questions(graph: AgentGraph, state: AgentState) -> bool:
    """信息足够的建站任务不应让 Plan 模式额外追问推荐选项。"""

    goal = graph._plan_goal(state)
    goal_lower = goal.lower()
    data_site = "数据集" in goal and ("网站" in goal or "web" in goal_lower)
    has_model_config = ("apikey" in goal_lower or "api_key" in goal_lower) and "base_url" in goal_lower and "model_name" in goal_lower
    asks_one_shot = any(phrase in goal for phrase in ("一次性生成", "一次性", "直接生成", "不需要确认"))
    return bool(data_site and (has_model_config or asks_one_shot))


def _recommended_plan_answers(questions: list[dict]) -> str:
    """把模型澄清题转换成推荐答案摘要，用于同轮继续生成计划。"""

    lines: list[str] = []
    for index, item in enumerate(questions, 1):
        question = str(item.get("question") or "").strip()
        recommended = str(item.get("recommended") or "").strip()
        if question and recommended:
            lines.append(f"{index}. {question} -> {recommended}")
    return "\n".join(lines) or "全部按推荐选项执行。"


def planner(graph: AgentGraph, state: AgentState) -> AgentState:
    """Plan 智能体：生成澄清问题或可执行 Markdown 计划。"""

    graph._emit(state, "planner", "start", "Plan 智能体正在生成问题或可执行计划")
    # client 是 planner 智能体对应的模型客户端。
    client = graph._client("planner", state)
    # fallback_configs 只包含 enabled=True 的其他模型；显式 model_id 覆盖时尊重用户选择，不自动切换。
    fallback_configs = (
        graph.model_store.fallbacks(getattr(getattr(client, "cfg", None), "id", ""))
        if getattr(getattr(client, "cfg", None), "id", "") and not state.get("model_id")
        else []
    )
    failed_models: set[str] = set()
    transient_counts: dict[str, int] = {}
    # messages 是发给模型的系统提示词和上下文包。
    # context_text 除结构化任务上下文外，还包含前置 Repo 节点筛选出的真实文件片段。
    # Planner 必须先看 schema/清单等仓库事实，避免按常见项目印象臆造字段和目录。
    context_text = graph._ctx_text(state)
    if state.get("repo"):
        repo_text = json.dumps(state["repo"], ensure_ascii=False)
        context_text += "\n\n仓库事实摘要（字段、路径和 schema 以此为准）：\n" + graph._trim(repo_text, 24_000)
    user_content: list[dict[str, str]] = [{"type": "input_text", "text": context_text}]
    user_content.extend(
        {
            "type": "input_image",
            "path": str(item["path"]),
            "mime": str(item["mime"]),
        }
        for item in state.get("attachments") or []
        if item.get("path") and item.get("mime")
    )
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": user_content},
    ]
    data = None
    last_error: LlmError | None = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            # data 应该是模型返回的 JSON，status 为 questions 或 plan。
            data = graph._model_json(state, "planner", client, messages)
            break
        except LlmBalanceError:
            raise
        except LlmError as exc:
            last_error = exc
            error_text = str(exc)
            if not _is_transient(error_text) or attempt >= len(_RETRY_DELAYS):
                break
            current_id = str(getattr(getattr(client, "cfg", None), "id", ""))
            transient_counts[current_id] = transient_counts.get(current_id, 0) + 1
            # 读取超时表示本次长生成已经占满整个 timeout，立即故障转移；普通 5xx 连续两次再切换。
            should_failover = "响应超时" in error_text or transient_counts[current_id] >= 2
            if should_failover and fallback_configs:
                failed_models.add(current_id)
                next_cfg = next((cfg for cfg in fallback_configs if cfg.id not in failed_models), None)
                if next_cfg is not None:
                    graph._emit(
                        state,
                        "planner",
                        "model",
                        f"Plan 模型 {current_id} 连续不可用，自动切换到 {next_cfg.id}",
                    )
                    client = LlmClient(next_cfg)
                    continue
            delay = _RETRY_DELAYS[attempt]
            graph._emit(
                state,
                "planner",
                "error",
                f"Plan 模型服务暂时不可用，{delay:g} 秒后重试"
                f"（{attempt + 1}/{len(_RETRY_DELAYS) + 1}）：{error_text[:120]}",
            )
            time.sleep(delay)
    if data is None:
        # 模型返回非 JSON 时使用兜底问题或兜底计划，避免 Plan 流程卡死。
        graph._emit(
            state,
            "planner",
            "error",
            f"Plan 模型返回异常，已使用默认澄清问题：{str(last_error or '未知错误')[:120]}",
        )
        if state.get("pending_plan"):
            # 如果已经在回答上一轮问题，模型异常时直接生成默认计划。
            data = {"status": "plan", "title": "默认执行计划", "markdown": graph._fallback_plan_markdown(state)}
        else:
            # 第一轮 Plan 模型异常时，返回一个默认选择题让用户继续。
            data = {
                "status": "questions",
                "questions": [
                    {
                        "question": "是否直接按默认工程架构执行？",
                        "options": ["直接执行", "继续细化计划"],
                        "recommended": "直接执行",
                        "reason": "模型返回格式异常，先用默认问题继续推进，避免阻塞后续执行。",
                        "allow_custom": True,
                    }
                ],
            }
    if data.get("status") == "questions":
        # 清洗模型输出的问题，防止 JSON 错误信息或异常长文本展示给用户。
        data = {**data, "questions": graph._normalize_plan_questions(data.get("questions"))}
        if _should_auto_answer_plan_questions(graph, state):
            graph._emit(state, "planner", "review", "任务信息已足够，自动采用推荐选项继续生成计划")
            current_id = str(getattr(getattr(client, "cfg", None), "id", ""))
            next_cfg = next((cfg for cfg in fallback_configs if cfg.id != current_id), None)
            if next_cfg is not None:
                graph._emit(state, "planner", "model", f"Plan 问题由 {current_id} 生成，切换到 {next_cfg.id} 直接产出计划")
                client = LlmClient(next_cfg)
            try:
                data = graph._model_json(
                    state,
                    "planner",
                    client,
                    [
                        {"role": "system", "content": PLANNER_PROMPT},
                        {
                            "role": "user",
                            "content": graph._multimodal_content(
                                context_text
                                + "\n\n模型刚才生成了澄清问题，但当前用户需求已经足够执行。"
                                "请按以下推荐答案直接生成完整可执行计划，不要再返回 questions：\n"
                                + _recommended_plan_answers(data["questions"]),
                                state,
                            ),
                        },
                    ]
                )
            except LlmBalanceError:
                raise
            except LlmError as exc:
                graph._emit(state, "planner", "error", f"自动采用推荐选项生成计划失败：{str(exc)[:120]}")
    if data.get("status") == "plan":
        # Plan 初稿必须先通过确定性合同审查；不合格时按上限自动修订，用户只看到最终计划。
        normalized_markdown = _normalize_chat_contract(
            str(data.get("markdown") or ""),
            graph._plan_goal(state),
        )
        best_data = {**data, "markdown": normalized_markdown}
        best_issues = _plan_quality_issues(graph, state, normalized_markdown)
        repair_models = {str(getattr(getattr(client, "cfg", None), "id", ""))}
        repair_counts: dict[str, int] = {}
        for repair_index in range(_MAX_PLAN_REPAIRS):
            if not best_issues:
                break
            graph._emit(
                state,
                "planner",
                "review",
                f"计划初稿存在 {len(best_issues)} 项执行级问题，"
                f"正在自动修订（{repair_index + 1}/{_MAX_PLAN_REPAIRS}）",
            )
            repair_messages = [
                {"role": "system", "content": PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": graph._multimodal_content(
                        context_text
                        + "\n\n待修订计划初稿：\n"
                        + str(best_data.get("markdown") or "")
                        + "\n\n必须逐项修复以下问题，并返回一份自洽、可直接执行的完整计划：\n- "
                        + "\n- ".join(best_issues)
                        + "\n不得只追加勘误；要消除初稿中的冲突和错误字段。",
                        state,
                    ),
                },
            ]
            try:
                repaired = graph._model_json(
                    state,
                    "planner",
                    client,
                    repair_messages,
                )
            except LlmBalanceError:
                raise
            except LlmError as exc:
                next_cfg = next((cfg for cfg in fallback_configs if cfg.id not in repair_models), None)
                if next_cfg is None:
                    graph._emit(state, "planner", "error", f"计划自动修订失败，保留当前最佳版本：{str(exc)[:120]}")
                    break
                repair_models.add(next_cfg.id)
                graph._emit(
                    state,
                    "planner",
                    "model",
                    f"Plan 修订模型不可用，自动切换到 {next_cfg.id}",
                )
                client = LlmClient(next_cfg)
                continue
            if repaired.get("status") != "plan" or not repaired.get("markdown"):
                break
            repaired = {
                **repaired,
                "markdown": _normalize_chat_contract(
                    str(repaired["markdown"]),
                    graph._plan_goal(state),
                ),
            }
            current_repair_model = str(getattr(getattr(client, "cfg", None), "id", ""))
            repair_counts[current_repair_model] = repair_counts.get(current_repair_model, 0) + 1
            repaired_issues = _plan_quality_issues(graph, state, str(repaired["markdown"]))
            improved = len(repaired_issues) < len(best_issues)
            if improved:
                best_data = repaired
                best_issues = repaired_issues
            should_cross_review = repair_counts[current_repair_model] >= 2 and bool(best_issues)
            if repair_index + 1 < _MAX_PLAN_REPAIRS and (
                not improved or should_cross_review
            ):
                next_cfg = next((cfg for cfg in fallback_configs if cfg.id not in repair_models), None)
                if next_cfg is not None:
                    repair_models.add(next_cfg.id)
                    graph._emit(
                        state,
                        "planner",
                        "model",
                        (
                            f"Plan 修订由同一模型连续执行，切换到 {next_cfg.id} 交叉审查"
                            if should_cross_review
                            else f"Plan 修订质量未改善，自动切换到 {next_cfg.id}"
                        ),
                    )
                    client = LlmClient(next_cfg)
        data = best_data
        if best_issues:
            error = "计划质量审查未通过：" + "；".join(best_issues[:4])
            graph._emit(state, "planner", "error", error)
            graph.memory.append(
                state["workdir"],
                state["session_id"],
                "planner",
                "state",
                "plan_invalid",
                "计划质量审查未通过",
                {"issues": best_issues},
            )
            return {
                **state,
                "plan": {"status": "invalid", "issues": best_issues},
                "route": "final",
                "tests_ok": False,
                "coding_ok": False,
                "error": error,
                "final": "",
            }
    if data.get("status") == "plan" or state.get("execute_plan"):
        # md 是最终计划正文；缺失时使用兜底计划。
        md = graph._redact_plan_secrets(data.get("markdown") or graph._fallback_plan_markdown(state))
        # Plan 模式是只读权限层；计划正文只进入会话记忆，绝不写入用户工作目录。
        graph._emit(state, "planner", "plan", "计划已保存到当前会话记忆")
        graph.memory.append(
            state["workdir"],
            state["session_id"],
            "planner",
            "state",
            "plan_done",
            "计划已生成",
            {"status": "plan", "path": None, "markdown": md, "goal": graph._plan_goal(state)},
        )
        # execute_plan=True 时直接进入 repo，否则停在 final 等用户确认。
        route = "repo" if state.get("execute_plan") else "final"
        # final 是返回给用户的确认提示；直接执行时保持空字符串。
        final = (
            ""
            if route == "repo"
            else md.strip() + "\n\n计划已保存到当前会话记忆。确认后回复“执行计划”。"
        )
        return {
            **state,
            "plan": {"status": "plan", "path": None, "markdown": md},
            "route": route,
            "after_repo": "coder",
            "after_verify": "doc",
            "need_doc": True,
            "final": final,
        }
    # summary 是格式化后的选择题文本，直接展示给用户。
    summary = graph._format_plan_questions(data.get("questions", []))
    # 保存 pending_plan，下一轮用户回复选项时 manager 可以找回上下文。
    graph.memory.append(
        state["workdir"],
        state["session_id"],
        "planner",
        "state",
        "pending_plan",
        "Plan 问题等待用户回答",
        {
            "status": "questions",
            "goal": graph._plan_goal(state),
            "questions": data.get("questions", []),
            "answers": state.get("plan_answers", []),
        },
    )
    graph._emit(state, "planner", "questions", "已生成澄清问题，等待用户选择")
    return {**state, "plan": data, "route": "final", "final": summary}
