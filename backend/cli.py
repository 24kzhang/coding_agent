from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.agents.graph import AgentGraph, new_session_id
from backend.capabilities import CapabilityStore, McpClient
from backend.memory import MemoryStore
from llm import ModelStore


def main() -> None:
    """命令行入口，用于不打开前端时直接运行一次 agent 任务。"""

    # parser 负责解析命令行参数。
    parser = argparse.ArgumentParser(description="命令行运行多智能体任务")
    # --workdir 是 agent 操作的项目目录。
    parser.add_argument("--workdir", required=True, help="项目工作目录")
    # task_group 允许直接传短任务，也允许从 UTF-8 文件读取长任务，二者必须且只能选一个。
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="用户任务")
    task_group.add_argument("--task-file", help="从 UTF-8 文本文件读取用户任务")
    # --plan-mode 控制是否进入 Plan 模式。
    parser.add_argument("--plan-mode", action="store_true", help="是否启用 Plan 模式")
    # --execute-plan 控制是否执行已有计划。
    parser.add_argument("--execute-plan", action="store_true", help="是否执行已有计划")
    # --session-id 允许 Plan 提问、回答和执行计划复用同一会话；不传时保持自动创建。
    parser.add_argument("--session-id", default=None, help="复用指定会话 id；首次调用可自行指定")
    # --model-id 用于临时覆盖所有智能体模型选择。
    parser.add_argument("--model-id", default=None, help="覆盖使用的模型 id")
    # --skill 可重复传入，启用项目根目录 skills 中对应的 Skill。
    parser.add_argument("--skill", action="append", default=[], help="启用一个 Skill id，可重复使用")
    # --mcp 可重复传入，启用 MCP/servers.json 中对应的服务。
    parser.add_argument("--mcp", action="append", default=[], help="启用一个 MCP 服务 id，可重复使用")
    # args 是解析后的命令行参数对象。
    args = parser.parse_args()
    if args.task_file:
        try:
            task_text = Path(args.task_file).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            parser.error(f"任务文件读取失败：{exc}")
    else:
        task_text = str(args.task or "")
    task_text = task_text.strip()
    if not task_text:
        parser.error("用户任务不能为空")
    if len(task_text) > 200_000:
        parser.error("用户任务超过 200000 字符限制")

    # workdir 统一解析为绝对路径，避免相对路径在 memory 中产生多个项目 id。
    workdir = str(Path(args.workdir).expanduser().resolve())
    # 如果目标目录不存在，自动创建，便于从空项目开始。
    Path(workdir).mkdir(parents=True, exist_ok=True)
    # memory 是记忆仓库实例。
    memory = MemoryStore()
    # session_id 默认自动生成；显式传入后可跨多次 CLI 调用延续同一 Plan 会话。
    session_id = str(args.session_id or new_session_id())
    try:
        # existing_records 用于区分首次创建与继续已有会话，同时触发统一的安全 id 校验。
        existing_records = memory.read_session(workdir, session_id)
    except ValueError as exc:
        parser.error(str(exc))
    # resuming 必须在写入本轮 run/start 前计算，语义与 Web 入口保持一致。
    resuming = memory.interrupted(workdir, session_id) if existing_records else False
    # capabilities 负责校验 CLI 选择的 Skill 和 MCP，不允许命令行注入任意服务配置。
    capabilities = CapabilityStore()
    # skills 是受预算限制的 SKILL.md 正文。
    skills = capabilities.load_skills(args.skill)
    # mcp_servers 是 MCP/servers.json 中已启用且被用户明确选择的服务。
    mcp_servers = capabilities.select_mcps(args.mcp)
    # CLI 模式下只在首次调用写入会话开始记录；复用时不会伪造一个新会话。
    if not existing_records:
        memory.append(workdir, session_id, "manager", "cli", "start", "命令行会话开始")
    memory.append(
        workdir,
        session_id,
        "manager",
        "capability",
        "selection",
        f"本轮启用 {len(skills)} 个 Skill、{len(mcp_servers)} 个 MCP",
        {"skills": [item.id for item in skills], "mcps": [item.id for item in mcp_servers]},
    )
    # CLI 与 Web 入口记录相同的用户输入和运行边界，使 Plan 上下文、历史恢复与中断判断一致。
    memory.append(workdir, session_id, "user", "input", "message", task_text)
    memory.append(workdir, session_id, "manager", "run", "start", "本轮任务开始")

    def emit(event: object) -> None:
        """把 AgentEvent 直接打印成 JSON 行。"""

        print(json.dumps(event.model_dump(), ensure_ascii=False), flush=True)

    # agent 是本次 CLI 任务的 LangGraph 编排实例。
    graph = AgentGraph(ModelStore(), memory, emit=emit, mcp_client=McpClient())
    # result 是最终任务结果。
    try:
        result = graph.run(
            session_id=session_id,
            workdir=workdir,
            text=task_text,
            plan_mode=args.plan_mode,
            execute_plan=args.execute_plan,
            model_id=args.model_id,
            resuming=resuming,
            skills=skills,
            mcp_servers=[item.model_dump() for item in mcp_servers],
        )
    except Exception as exc:
        memory.append(workdir, session_id, "manager", "run", "error", f"本轮任务失败：{exc}")
        raise
    memory.append(workdir, session_id, "manager", "run", "done", "本轮任务正常结束")
    # 最后一行打印 session_id 和 result，方便脚本调用方解析。
    print(json.dumps({"session_id": session_id, "result": result.model_dump()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
