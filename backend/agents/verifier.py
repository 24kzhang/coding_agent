from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.agents.prompts import VERIFIER_PROMPT
from backend.agents.types import AgentState
from backend.tools import FsTool
from backend.tools.scope import delivery_root
from llm import LlmError

if TYPE_CHECKING:
    from backend.agents.graph import AgentGraph


# 这些警告说明异步代码或测试替身实际没有执行完成，不能和普通弃用提示一样放行。
_CRITICAL_WARNING_MARKERS = (
    "RuntimeWarning: coroutine",
    "was never awaited",
    "Task was destroyed but it is pending",
)


def _private_model_config_facts(fs: FsTool, files: list[str]) -> list[str]:
    """只向审查模型提供私密配置的可验证事实，绝不暴露凭据正文。"""

    facts: list[str] = []
    for rel in files:
        if not rel.endswith(".local/model.json"):
            continue
        try:
            data = json.loads(fs.safe(rel).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            facts.append(f"{rel}：文件存在，但不是合法 JSON。")
            continue
        if not isinstance(data, dict):
            facts.append(f"{rel}：文件存在，但顶层不是对象。")
            continue
        api_key = data.get("api_key")
        placeholder_values = {
            "",
            "[密钥已脱敏]",
            "[已脱敏]",
            "[已脱敏，执行时使用原始需求中的用户凭据]",
        }
        facts.append(
            f"{rel}：文件存在且为合法 JSON；model={data.get('model')!r}；"
            f"base_url_nonempty={bool(str(data.get('base_url') or '').strip())}；"
            f"api_key_nonempty={bool(str(api_key or '').strip())}；"
            f"api_key_is_redaction_placeholder={str(api_key or '').strip() in placeholder_values}。"
        )
    return facts


def _reject_critical_warnings(result: dict[str, Any]) -> dict[str, Any]:
    """把会掩盖真实异步失败的运行时警告转换为验证失败。"""

    output = str(result.get("out") or "") + "\n" + str(result.get("err") or "")
    matched = [marker for marker in _CRITICAL_WARNING_MARKERS if marker in output]
    if not matched:
        return result
    return {
        **result,
        "ok": False,
        "issues": [f"检测到严重运行时警告：{marker}" for marker in matched],
    }


def _reject_empty_tests(result: dict[str, Any]) -> dict[str, Any]:
    """测试运行器零退出码但没有执行测试时，不得宣称测试通过。"""

    cmd = str(result.get("cmd") or "")
    output = str(result.get("out") or "") + "\n" + str(result.get("err") or "")
    is_test = "pytest" in cmd or bool(re.search(r"\brun\s+test\b", cmd))
    if is_test and result.get("ok") and re.search(r"\b(?:no tests? (?:ran|found)|0 tests?|tests?\s+0|0 passing)\b", output, re.IGNORECASE):
        return {**result, "ok": False, "issues": ["测试命令未执行任何测试，缺少有效验收证据"]}
    return result


def _typescript_project_checks(fs: FsTool, files: list[str]) -> list[dict[str, Any]]:
    """确保 TypeScript 前端具备真实类型检查，而不是只依赖 Vite 转译。"""

    checks: list[dict[str, Any]] = []
    for manifest in sorted(rel for rel in files if Path(rel).name == "package.json"):
        project_dir = Path(manifest).parent
        ts_files = [
            rel
            for rel in files
            if rel.endswith((".ts", ".tsx"))
            and (project_dir == Path(".") or project_dir in Path(rel).parents)
        ]
        if not ts_files:
            continue
        configs = [
            rel
            for rel in files
            if Path(rel).name.startswith("tsconfig")
            and Path(rel).suffix == ".json"
            and (project_dir == Path(".") or project_dir in Path(rel).parents)
        ]
        try:
            package = json.loads(fs.read(manifest, 80_000))
            scripts = package.get("scripts") if isinstance(package, dict) else {}
        except (json.JSONDecodeError, OSError, ValueError):
            scripts = {}
        issues: list[str] = []
        if not configs:
            issues.append(f"{project_dir.as_posix()}：存在 TypeScript 源码，但缺少 tsconfig.json")
        if not isinstance(scripts, dict) or "typecheck" not in scripts:
            issues.append(f"{manifest}：缺少可执行的 typecheck 脚本")
        checks.append(
            {
                "cmd": f"TypeScript 项目配置（{project_dir.as_posix()}）",
                "ok": not issues,
                "out": "通过" if not issues else "；".join(issues),
                "issues": issues,
            }
        )
    return checks


def verifier(graph: AgentGraph, state: AgentState) -> AgentState:
    """验证智能体：根据项目文件选择测试命令并执行。"""

    graph._emit(state, "verifier", "start", "验证智能体正在选择并运行测试")
    if graph._guidance_requires_coding(state):
        graph._emit(state, "verifier", "guidance", "验证开始前收到新指导，返回 Coding 重新规划")
        return state
    # fs 用于列出文件并做静态 Web 检查。
    fs = FsTool(state["workdir"])
    # shell 用于执行测试命令。
    shell = graph._shell(state)
    # files 是当前项目文件列表。
    files = fs.list()
    output_root = delivery_root(fs, state)
    if output_root != fs.root:
        prefix = output_root.relative_to(fs.root).as_posix() + "/"
        files = [rel for rel in files if rel.startswith(prefix)]
    # commands 根据项目真实配置生成，不把某个测试框架硬编码为所有项目默认值。
    commands = list(dict.fromkeys(graph._verification_commands(fs, files) + list(state.get("verification_commands", []))))
    # tests 保存每条测试或静态检查的结构化结果。
    tests: list[dict[str, Any]] = []
    # ok 是整体验证状态，任意关键检查失败都会变为 False。
    ok = bool(state.get("coding_ok", True))

    if not state.get("coding_ok", True):
        tests.append(
            {
                "cmd": "Coding 完成状态",
                "ok": False,
                "out": state.get("coding_summary") or "Coding 智能体没有明确完成任务。",
            }
        )

    # static_check 是静态 Web 项目的轻量接线检查结果。
    static_check = graph._static_web_check(fs, files)
    if static_check:
        tests.append(static_check)
        ok = ok and bool(static_check.get("ok"))

    type_checks = _typescript_project_checks(fs, files)
    for check in type_checks:
        tests.append(check)
        ok = ok and bool(check.get("ok"))
        graph._emit(
            state,
            "verifier",
            "test",
            f"{check['cmd']} -> {'通过' if check.get('ok') else '失败'}",
            data=check,
        )

    if not commands and not static_check and not type_checks:
        required = state.get("task_type") in {"code_gen", "code_mod"} or (
            state.get("task_type") == "verify" and any(Path(rel).suffix in {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java"} for rel in files)
        )
        tests.append({"cmd": "验收入口检查", "ok": not required, "skipped": not required, "out": "没有识别到可运行验收入口，不能证明实现正确。"})
        ok = ok and not required
    # 已发现的验收入口全部执行，不能把后半部分子项目默默丢弃。
    for command in commands:
        # result 是命令执行结果。
        result = shell.run(command, timeout=240)
        result = _reject_critical_warnings(result)
        result = _reject_empty_tests(result)
        tests.append(result)
        ok = ok and bool(result.get("ok"))
        graph._emit(state, "verifier", "test", f"{command} -> {'通过' if result.get('ok') else '失败'}", data=result)
        state = graph._apply_guidance(state, checkpoint=f"验证命令 {command} 完成后")
        if graph._guidance_requires_coding(state):
            graph._emit(state, "verifier", "guidance", "验证期间收到新指导，停止剩余旧检查")
            break

    # 代码变更和显式只读验收都追加需求对照审查，弥补编译和静态正则无法理解业务语义的问题。
    task_type = state.get("task_type")
    needs_review = bool(state.get("changes")) and task_type in {"code_gen", "code_mod"}
    needs_review = needs_review or task_type == "verify"
    if needs_review and not graph._guidance_requires_coding(state):
        review_files: list[str] = []
        budget = 60_000
        # verify 没有本轮 changes，改用仓库读取智能体选择的关键文件作为只读审查依据。
        review_targets = list(state.get("changes", [])) or list((state.get("repo") or {}).get("selected") or [])
        private_config_facts = _private_model_config_facts(fs, review_targets)
        # 界面源码常排在文件清单末尾，不能因前面的测试/文档占满名额而完全漏审。
        review_targets.sort(key=lambda rel: ({".css": 0, ".html": 1, ".js": 2, ".ts": 2, ".tsx": 2, ".py": 3}.get(Path(rel).suffix, 4), rel))
        for rel in review_targets:
            if budget <= 0:
                break
            try:
                # 普通源码优先完整读取；过去固定截到 12K 会让模型把“上下文结尾”误判成
                # “磁盘文件被截断”。多文件总预算仍限制为 60K，单文件最多占 48K。
                read_limit = min(48_000, budget)
                observed = fs.read(rel, read_limit + 1)
            except (OSError, ValueError, UnicodeError):
                continue
            truncated = len(observed) > read_limit
            content = observed[:read_limit]
            if truncated:
                content += "\n[审查上下文在此截断，磁盘文件仍有后续内容；不得据此断言源码不完整。]"
            review_files.append(f"## {rel}\n{graph.model_store.redact_text(content)}")
            budget -= len(content)
        try:
            client = graph._client("verifier", state)
            review = graph._model_json(
                state,
                "verifier",
                client,
                [
                    {"role": "system", "content": VERIFIER_PROMPT},
                    {
                        "role": "user",
                        "content": graph._ctx_text(state)
                        + "\n\n仓库事实摘要（字段和路径以这里及变更文件为准，优先级高于计划假设）：\n"
                        + graph.model_store.redact_text(
                            str((state.get("repo") or {}).get("snippets") or {})[:24_000]
                        )
                        + "\n\n变更文件：\n"
                        + "\n\n".join(review_files)
                        + "\n\n磁盘文件清单（有些文件未进入正文预算，不能据此说它们不存在）：\n"
                        + "\n".join(files[:250])
                        + (
                            "\n\n私密配置安全核验（以下事实由程序直接读取磁盘后生成，优先级高于"
                            "变更文件中的脱敏占位文本；不得把脱敏展示误判为磁盘占位符）：\n"
                            + "\n".join(private_config_facts)
                            if private_config_facts
                            else ""
                        )
                        + "\n\n已运行检查：\n"
                        + graph.model_store.redact_text(tests),
                    },
                ],
                temperature=0,
                plain_text=True,
            )
            raw_issues = review.get("issues", [])
            issues = [str(item).strip() for item in raw_issues if str(item).strip()][:12] if isinstance(raw_issues, list) else ["审查模型返回的 issues 不是列表"]
            review_ok = review.get("ok") is True and not issues
            review_result = {
                "cmd": "需求实现审查",
                "ok": review_ok,
                "out": str(review.get("summary") or ("通过" if review_ok else "发现需求实现问题")),
                "issues": issues,
            }
            tests.append(review_result)
            ok = ok and review_ok
            graph._emit(state, "verifier", "test", f"需求实现审查 -> {'通过' if review_ok else '失败'}", data=review_result)
        except (LlmError, KeyError) as exc:
            # 语义审查是代码任务的必要验收项；不可用时不能伪装为通过，也不能让 Coding 修复基础设施。
            tests.append(
                {
                    "cmd": "需求实现审查",
                    "ok": False,
                    "infra": True,
                    "out": f"审查模型不可用：{graph._trim(str(exc), 160)}",
                }
            )
            ok = False
    state = graph._apply_guidance(state, checkpoint="验证阶段结束前")
    tests = graph.model_store.redact(tests)
    graph.memory.append(
        state["workdir"],
        state["session_id"],
        "verifier",
        "test",
        "summary",
        f"测试结果：{ok}",
        {"tests": tests},
    )
    return {**state, "tests": tests, "tests_ok": ok}
