# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

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

    parser = argparse.ArgumentParser(description="命令行运行多智能体任务")
    parser.add_argument("--workdir", required=True, help="项目工作目录")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="用户任务")
    task_group.add_argument("--task-file", help="从 UTF-8 文本文件读取用户任务")
    parser.add_argument("--plan-mode", action="store_true", help="是否启用 Plan 模式")
    parser.add_argument("--execute-plan", action="store_true", help="是否执行已有计划")
    parser.add_argument("--session-id", default=None, help="复用指定会话 id；首次调用可自行指定")
    parser.add_argument("--model-id", default=None, help="覆盖使用的模型 id")
    parser.add_argument("--skill", action="append", default=[], help="启用一个 Skill id，可重复使用")
    parser.add_argument("--mcp", action="append", default=[], help="启用一个 MCP 服务 id，可重复使用")
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

    workdir = str(Path(args.workdir).expanduser().resolve())
    Path(workdir).mkdir(parents=True, exist_ok=True)
    memory = MemoryStore()
    session_id = str(args.session_id or new_session_id())
    try:
        existing_records = memory.read_session(workdir, session_id)
    except ValueError as exc:
        parser.error(str(exc))
    resuming = memory.interrupted(workdir, session_id) if existing_records else False
    capabilities = CapabilityStore()
    skills = capabilities.load_skills(args.skill)
    mcp_servers = capabilities.select_mcps(args.mcp)
    if not existing_records:
        memory.append(workdir, session_id, "manager", "cli", "start", "命令行会话开始")
    memory.append(
        workdir,
        session_id,
        "manager",
        "capability",
        "selection",
        f"本轮启用 {len(skills)} 个 Skill、{len(mcp_servers)} 个 MCP",
        {
            "skills": [item.id for item in skills],
            "mcps": [item.id for item in mcp_servers],
            "container_shell": False,
        },
    )
    memory.append(workdir, session_id, "user", "input", "message", task_text)
    memory.append(workdir, session_id, "manager", "run", "start", "本轮任务开始")

    def emit(event: object) -> None:
        """把 AgentEvent 直接打印成 JSON 行。"""

        print(json.dumps(event.model_dump(), ensure_ascii=False), flush=True)

    graph = AgentGraph(ModelStore(), memory, emit=emit, mcp_client=McpClient())
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
            required_mcps=args.mcp,
        )
    except Exception as exc:
        memory.append(workdir, session_id, "manager", "run", "error", f"本轮任务失败：{exc}")
        raise
    memory.append(workdir, session_id, "manager", "run", "done", "本轮任务正常结束")
    print(json.dumps({"session_id": session_id, "result": result.model_dump()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
