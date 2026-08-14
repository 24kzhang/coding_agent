from __future__ import annotations

from typing import Any, TypedDict

from api.schema import ContextPackage, SkillContext


class AgentState(TypedDict, total=False):
    """LangGraph 在各个智能体节点之间传递的状态。"""

    # 当前会话 id，用于写入和读取 memory jsonl。
    session_id: str
    # 当前项目工作目录，文件工具和命令工具都以它作为安全边界。
    workdir: str
    # 当前节点要处理的任务文本；Plan 回复时可能被管理者改写为“原始需求 + 用户答案”。
    text: str
    # original_text 保存本轮最初输入，便于早到指导参与 manager 分类时稳定重建有效任务。
    original_text: str
    # guidance 保存本轮运行中已经应用的用户指导，后发送的指导优先级更高。
    guidance: list[str]
    # guidance_revision 每应用一批指导递增，Coder 用它丢弃基于旧要求生成的未执行动作。
    guidance_revision: int
    # coded_guidance_revision 记录最近一次 Coding 已处理到哪一版指导。
    coded_guidance_revision: int
    # 前端 Plan 模式开关状态。
    plan_mode: bool
    # 是否强制执行已保存计划。
    execute_plan: bool
    # executing_plan 表示本轮正在执行已保存计划，成功后 final 才结束计划生命周期。
    executing_plan: bool
    # 临时覆盖模型 id；为空时按每个智能体自己的模型映射选择。
    model_id: str | None
    # resuming 表示开始本轮前检测到上一轮异常中断。
    resuming: bool
    # skills 是用户本轮手动选择并由后端安全读取的 Skill 正文。
    skills: list[SkillContext]
    # mcp_servers 是用户本轮手动选择的 MCP 服务配置，不向前端或模型暴露密钥字段。
    mcp_servers: list[dict[str, Any]]
    # mcp_tools 是连接所选 MCP 后通过 tools/list 发现的动态工具定义。
    mcp_tools: list[dict[str, Any]]
    # mcp_errors 保存 MCP 服务初始化失败摘要，答疑和 Coding 模型都可据此返回真实状态。
    mcp_errors: list[str]
    # mcp_discovered_servers 记录本轮已经 tools/list 或连接失败的服务，支持分阶段发现共享与专有 MCP。
    mcp_discovered_servers: list[str]
    # container_shell 决定 Coder 与 Verifier 使用 Docker 容器还是原生 Shell。
    container_shell: bool
    # started_at 保存单调时钟起点，用于计算本轮执行耗时。
    started_at: float
    # 管理者分类出的任务类型，例如 direct、code_gen、doc_gen、verify。
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
    # coding_ok 表示 Coding ReAct 是否明确完成，而不只是循环退出。
    coding_ok: bool
    # coding_summary 保存 Coding 模型对本轮修改的结构化摘要。
    coding_summary: str
    # error 保存执行失败时的用户可读原因。
    error: str
    # 当前准备给用户看的最终文本摘要。
    final: str
    # final 节点整理出的最终 TaskResult 字典。
    result: dict[str, Any]
    # 本次任务累计 token 估算。
    tokens: int
