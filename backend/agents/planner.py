from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

from backend.agents.prompts import PLANNER_PROMPT
from backend.agents.types import AgentState
from backend.tools.specs import tool_specs
from llm import LlmBalanceError, LlmClient, LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


# Planner 与 Coding 一样会遇到限流、网关 5xx 和短暂连接故障；这些错误不应立刻降级成通用计划。
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "连接", "Connect", "响应超时", "ReadTimeout")
_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0)
_MAX_PLAN_REPAIRS = 3


def _semantic_plan_issues(graph: AgentGraph, state: AgentState, context: str, markdown: str) -> list[str]:
    """独立检查执行合同的规则冲突，接受满足需求的不同实现架构。"""

    client = graph._client("verifier", state)
    result = graph._model_json(
        state, "verifier", client,
        [
            {"role": "system", "content": (
                "你是计划一致性审查员，只审查计划是否可执行，不编写代码。对照原始需求、仓库事实和工具合同，"
                "检查必须交付的内容、依赖顺序及规则之间是否有真实矛盾。跨步骤或跨请求的必要状态应能在下一步取得；"
                "检查删除某数据后又依赖它、输出与读取格式不一致、路径或工具参数违反执行合同等问题。"
                "只报告具体阻碍用户需求的问题，并引用冲突规则；不强制字段名、消息角色、技术栈、客户端或服务端状态架构，"
                "不要把风格偏好和无关增强当成失败。只返回 JSON："
                '{"ok":true,"issues":[],"summary":"中文结论"}'
            )},
            {"role": "user", "content": graph._multimodal_content(
                context, state, "\n\n待审查的完整执行合同：\n" + markdown,
            )},
        ], temperature=0, plain_text=True,
    )
    issues = result.get("issues")
    if not isinstance(issues, list):
        return ["一致性审查未返回有效问题列表"]
    normalized = [str(issue).strip() for issue in issues if str(issue).strip()][:8]
    if result.get("ok") is not True and not normalized:
        return ["一致性审查未确认计划可执行：" + str(result.get("summary") or "缺少结论")]
    return normalized


def _is_transient(error: str) -> bool:
    """判断 Plan 模型错误是否适合退避重试。"""

    return any(marker in error for marker in _TRANSIENT_MARKERS)


def _schema_fields(repo: dict) -> list[str]:
    """从 Repo 摘要中的 JSON Schema 提取真实对象字段。"""

    snippets = repo.get("snippets") if isinstance(repo, dict) else {}
    if not isinstance(snippets, dict):
        return []
    fields: list[str] = []
    for path, content in snippets.items():
        if not str(path).endswith((".schema.json", "/schema.json")) and str(path) != "schema.json":
            continue
        try:
            schema = json.loads(str(content))
        except json.JSONDecodeError:
            continue
        if not isinstance(schema, dict):
            continue
        record = schema
        visited: set[str] = set()
        while isinstance(record, dict):
            ref = record.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/") and ref not in visited:
                visited.add(ref)
                resolved = schema
                for part in ref[2:].split("/"):
                    resolved = resolved.get(part.replace("~1", "/").replace("~0", "~"), {}) if isinstance(resolved, dict) else {}
                record = resolved
                continue
            if isinstance(record.get("items"), dict):
                record = record["items"]
                continue
            break
        properties = record.get("properties", {}) if isinstance(record, dict) else {}
        if not properties:
            definitions = schema.get("$defs") or schema.get("definitions") or {}
            candidates = [item.get("properties") for item in definitions.values() if isinstance(item, dict) and isinstance(item.get("properties"), dict)] if isinstance(definitions, dict) else []
            if len(candidates) == 1:
                properties = candidates[0]
        if isinstance(properties, dict):
            fields.extend(str(field) for field in properties)
    return list(dict.fromkeys(fields))


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
        if not re.search(r"/api/[\w/{.-]+", markdown) and not any(word in markdown for word in ("静态资源", "公开数据 URL", "静态 JSON")):
            issues.append("缺少可执行的数据访问合同：应规划明确的数据接口或公开静态资源地址")
        if not any(word in plan_lower for word in ("pytest", "vitest", "jest", "go test", "cargo test", "自动化测试")):
            issues.append("缺少核心接口和数据逻辑的自动化测试")

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
        issues.append("详情未明确展示数据记录的所有原始字段")
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
    detail_endpoint = re.search(r"/api/[\w/.-]+/\{", markdown)
    if fields and detail_endpoint:
        detail_api_start = detail_endpoint.start()
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
            "返回完整实体详情",
            "返回完整业务对象",
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
                issues.append("详情 API 合同遗漏原始字段：" + "、".join(missing_api_fields))

    attachments = list(state.get("attachments") or [])
    if attachments:
        missing_names = [
            str(item.get("name") or item.get("id") or "未命名附件")
            for item in attachments
            if str(item.get("name") or "") not in markdown
        ]
        if missing_names:
            issues.append("计划未逐张关联图片附件：" + "、".join(missing_names))
        if not re.search(r"(?:布局|配色|视觉|间距|参考编号|reference\s*(?:id|code))", markdown, re.IGNORECASE):
            issues.append("图片附件缺少可验证的布局、配色或视觉要素描述，不能只写文件名")

    implementation_fences = sum(
        markdown.lower().count(f"```{language}") for language in ("python", "javascript", "typescript", "html", "css")
    )
    if implementation_fences > 2 or len(markdown) > 24_000:
        issues.append("计划内嵌了过多近似成品源码；应保留合同、步骤和短示例，把完整实现交给 Coding 智能体")

    return issues


def _should_auto_answer_plan_questions(graph: AgentGraph, state: AgentState) -> bool:
    """用户要求直接交付时，同轮采用可选工程决策的推荐项。"""

    goal = graph._plan_goal(state)
    # 自动选择只服从用户的一次性交付指令，不依赖项目领域或配置字段。
    return any(phrase in goal for phrase in ("一次性生成", "一次性", "直接生成", "不需要确认", "按推荐选项"))


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
    # 规划阶段不执行工具，但必须依据执行层同源合同选择路径、格式及参数。
    context_text += "\n\n执行层工具合同（当前阶段只读，供规划后续操作）：\n" + json.dumps(
        [{"name": item["name"], "description": item["description"], "parameters": item["parameters"]} for item in tool_specs()],
        ensure_ascii=False,
    )
    context_text += "\n\n可复用模型编号（不包含凭据）：\n" + json.dumps(
        [{"id": item.id, "model": item.model} for item in graph.model_store.public() if item.enabled],
        ensure_ascii=False,
    )
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
                                context_text,
                                state,
                                "\n\n模型刚才生成了澄清问题，但当前用户需求已经足够执行。"
                                "请按以下推荐答案直接生成完整可执行计划，不要再返回 questions：\n"
                                + _recommended_plan_answers(data["questions"]),
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
        normalized_markdown = str(data.get("markdown") or "")
        best_data = {**data, "markdown": normalized_markdown}
        best_issues = _plan_quality_issues(graph, state, normalized_markdown)
        if not best_issues:
            best_issues = _semantic_plan_issues(graph, state, context_text, normalized_markdown)
        repair_models = {str(getattr(getattr(client, "cfg", None), "id", ""))}
        repair_counts: dict[str, int] = {}
        repair_transient_counts: dict[str, int] = {}
        for repair_index in range(_MAX_PLAN_REPAIRS):
            if not best_issues:
                break
            graph._emit(
                state,
                "planner",
                "review",
                f"计划初稿存在 {len(best_issues)} 项执行级问题，"
                f"正在自动修订（{repair_index + 1}/{_MAX_PLAN_REPAIRS}）",
                data={"issues": best_issues},
            )
            repair_messages = [
                {"role": "system", "content": PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": graph._multimodal_content(
                        context_text,
                        state,
                        "\n\n待修订计划初稿：\n"
                        + str(best_data.get("markdown") or "")
                        + "\n\n必须逐项修复以下问题，并返回一份自洽、可直接执行的完整计划：\n- "
                        + "\n- ".join(best_issues)
                        + "\n不得只追加勘误；要消除初稿中的冲突和错误字段。",
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
                current_id = str(getattr(getattr(client, "cfg", None), "id", ""))
                if _is_transient(str(exc)):
                    repair_transient_counts[current_id] = repair_transient_counts.get(current_id, 0) + 1
                    if repair_transient_counts[current_id] == 1 and repair_index + 1 < _MAX_PLAN_REPAIRS:
                        graph._emit(
                            state,
                            "planner",
                            "error",
                            f"Plan 修订模型 {current_id} 瞬时不可用，将使用同一计划输入重试一次：{str(exc)[:120]}",
                        )
                        continue
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
                "markdown": str(repaired["markdown"]),
            }
            current_repair_model = str(getattr(getattr(client, "cfg", None), "id", ""))
            repair_counts[current_repair_model] = repair_counts.get(current_repair_model, 0) + 1
            repaired_issues = _plan_quality_issues(graph, state, str(repaired["markdown"]))
            if not repaired_issues:
                repaired_issues = _semantic_plan_issues(graph, state, context_text, str(repaired["markdown"]))
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
