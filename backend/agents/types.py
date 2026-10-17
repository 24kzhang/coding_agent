# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

from __future__ import annotations

from typing import Any, TypedDict

from api.schema import ContextPackage, SkillContext


class AgentState(TypedDict, total=False):
    """LangGraph 在各个智能体节点之间传递的状态。"""

    session_id: str
    workdir: str
    text: str
    original_text: str
    original_goal: str
    guidance: list[str]
    guidance_revision: int
    coded_guidance_revision: int
    plan_mode: bool
    execute_plan: bool
    executing_plan: bool
    model_id: str | None
    effort: str | None
    attachments: list[dict[str, Any]]
    resuming: bool
    skills: list[SkillContext]
    mcp_servers: list[dict[str, Any]]
    required_mcps: list[str]
    mcp_tools: list[dict[str, Any]]
    mcp_errors: list[str]
    mcp_discovered_servers: list[str]
    container_shell: bool
    started_at: float
    task_type: str
    route: str
    after_repo: str
    after_verify: str
    need_doc: bool
    context: ContextPackage
    repo: dict[str, Any]
    plan: dict[str, Any]
    pending_plan: dict[str, Any]
    plan_answers: list[dict[str, Any]]
    changes: list[str]
    commands: list[str]
    verification_commands: list[str]
    tests: list[dict[str, Any]]
    tests_ok: bool
    retry: int
    coding_ok: bool
    coding_summary: str
    error: str
    final: str
    result: dict[str, Any]
    tokens: int
    model_calls: int
    model_requests: int
    cached_input_tokens: int
    uncached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
