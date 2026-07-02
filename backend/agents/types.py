from __future__ import annotations

from typing import Any, TypedDict

from api.schema import ContextPackage


class AgentState(TypedDict, total=False):
    """LangGraph 在各个智能体节点之间传递的状态。"""

    session_id: str
    workdir: str
    text: str
    plan_mode: bool
    execute_plan: bool
    model_id: str | None
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
    tests: list[dict[str, Any]]
    tests_ok: bool
    retry: int
    final: str
    result: dict[str, Any]
    tokens: int
