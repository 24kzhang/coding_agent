from __future__ import annotations

from typing import Any, TypedDict

from api.schema import ContextPackage


class AgentState(TypedDict, total=False):
    """LangGraph 在各个智能体节点之间传递的状态。"""

    # 当前会话 id，用于写入和读取 memory jsonl。
    session_id: str
    # 当前项目工作目录，文件工具和命令工具都以它作为安全边界。
    workdir: str
    # 当前节点要处理的任务文本；Plan 回复时可能被管理者改写为“原始需求 + 用户答案”。
    text: str
    # 前端 Plan 模式开关状态。
    plan_mode: bool
    # 是否强制执行已保存计划。
    execute_plan: bool
    # 临时覆盖模型 id；为空时按每个智能体自己的模型映射选择。
    model_id: str | None
    # 管理者分类出的任务类型，例如 direct、code_gen、doc_gen。
    task_type: str
    # 管理者之后的下一跳节点名。
    route: str
    # 仓库读取节点执行后要去的节点名。
    after_repo: str
    # 验证节点执行后要去的节点名。
    after_verify: str
    # 当前任务是否需要文档智能体参与。
    need_doc: bool
    # 管理者构造的结构化上下文包，下游智能体主要读取它而不是完整历史。
    context: ContextPackage
    # 仓库读取智能体产出的文件列表、代码片段和技术栈摘要。
    repo: dict[str, Any]
    # Plan 智能体产出的待选问题或已保存计划信息。
    plan: dict[str, Any]
    # 会话记忆里最近一条尚未完成的 Plan 问题状态。
    pending_plan: dict[str, Any]
    # 用户对 Plan 选择题的结构化回答。
    plan_answers: list[dict[str, Any]]
    # Coding 或文档智能体修改过的文件列表。
    changes: list[str]
    # Coding 智能体执行过的命令列表。
    commands: list[str]
    # 验证智能体产生的测试结果。
    tests: list[dict[str, Any]]
    # 验证结果是否整体通过。
    tests_ok: bool
    # Coding/验证失败后的重试轮次计数。
    retry: int
    # 当前准备给用户看的最终文本摘要。
    final: str
    # final 节点整理出的最终 TaskResult 字典。
    result: dict[str, Any]
    # 本次任务累计 token 估算。
    tokens: int
