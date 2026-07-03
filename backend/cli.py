from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.agents.graph import AgentGraph, new_session_id
from backend.memory import MemoryStore
from llm import ModelStore


def main() -> None:
    """命令行入口，用于不打开前端时直接运行一次 agent 任务。"""

    # parser 负责解析命令行参数。
    parser = argparse.ArgumentParser(description="命令行运行多智能体任务")
    # --workdir 是 agent 操作的项目目录。
    parser.add_argument("--workdir", required=True, help="项目工作目录")
    # --task 是用户任务文本。
    parser.add_argument("--task", required=True, help="用户任务")
    # --plan-mode 控制是否进入 Plan 模式。
    parser.add_argument("--plan-mode", action="store_true", help="是否启用 Plan 模式")
    # --execute-plan 控制是否执行已有计划。
    parser.add_argument("--execute-plan", action="store_true", help="是否执行已有计划")
    # --model-id 用于临时覆盖所有智能体模型选择。
    parser.add_argument("--model-id", default=None, help="覆盖使用的模型 id")
    # args 是解析后的命令行参数对象。
    args = parser.parse_args()

    # workdir 统一解析为绝对路径，避免相对路径在 memory 中产生多个项目 id。
    workdir = str(Path(args.workdir).expanduser().resolve())
    # 如果目标目录不存在，自动创建，便于从空项目开始。
    Path(workdir).mkdir(parents=True, exist_ok=True)
    # session_id 是本次 CLI 运行的独立会话 id。
    session_id = new_session_id()
    # memory 是记忆仓库实例。
    memory = MemoryStore()
    # CLI 模式下手动写入会话开始记录，方便历史会话能看到这次运行。
    memory.append(workdir, session_id, "manager", "cli", "start", "命令行会话开始")

    def emit(event: object) -> None:
        """把 AgentEvent 直接打印成 JSON 行。"""

        print(json.dumps(event.model_dump(), ensure_ascii=False), flush=True)

    # graph 是本次 CLI 任务的 LangGraph 编排实例。
    graph = AgentGraph(ModelStore(), memory, emit=emit)
    # result 是最终任务结果。
    result = graph.run(
        session_id=session_id,
        workdir=workdir,
        text=args.task,
        plan_mode=args.plan_mode,
        execute_plan=args.execute_plan,
        model_id=args.model_id,
    )
    # 最后一行打印 session_id 和 result，方便脚本调用方解析。
    print(json.dumps({"session_id": session_id, "result": result.model_dump()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
