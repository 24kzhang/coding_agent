from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.agents.graph import AgentGraph, new_session_id
from backend.memory import MemoryStore
from llm import ModelStore


def main() -> None:
    parser = argparse.ArgumentParser(description="命令行运行多智能体任务")
    parser.add_argument("--workdir", required=True, help="项目工作目录")
    parser.add_argument("--task", required=True, help="用户任务")
    parser.add_argument("--plan-mode", action="store_true", help="是否启用 Plan 模式")
    parser.add_argument("--execute-plan", action="store_true", help="是否执行已有计划")
    parser.add_argument("--model-id", default=None, help="覆盖使用的模型 id")
    args = parser.parse_args()

    workdir = str(Path(args.workdir).expanduser().resolve())
    Path(workdir).mkdir(parents=True, exist_ok=True)
    session_id = new_session_id()
    memory = MemoryStore()
    memory.append(workdir, session_id, "manager", "cli", "start", "命令行会话开始")

    def emit(event: object) -> None:
        print(json.dumps(event.model_dump(), ensure_ascii=False), flush=True)

    graph = AgentGraph(ModelStore(), memory, emit=emit)
    result = graph.run(
        session_id=session_id,
        workdir=workdir,
        text=args.task,
        plan_mode=args.plan_mode,
        execute_plan=args.execute_plan,
        model_id=args.model_id,
    )
    print(json.dumps({"session_id": session_id, "result": result.model_dump()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
