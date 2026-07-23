# 基于 LangGraph 的本地多智能体编程系统

**个人项目｜2026.07 - 至今**  
**技术栈：** Python、FastAPI、LangGraph、Pydantic、React、TypeScript、pytest、uv

面向本地软件开发场景，独立设计并实现可读取代码仓库、修改文件、执行命令、运行测试和恢复历史会话的多智能体 Coding Agent。

- 基于 **LangGraph StateGraph** 编排 Manager、Planner、Repo Reader、Coder、Verifier、Doc 等智能体，由 Manager 结合确定性规则与 LLM 完成任务分类和动态路由，避免普通问候、代码解释等简单请求触发完整 Agent 流水线。
- 实现 **Plan-Execute + ReAct** 执行架构：支持多轮选择题澄清和计划持久化；Coder 调用受工作目录约束的文件、Shell、Git 工具真实修改磁盘，Verifier 自动运行测试，并将失败结果反馈给 Coder 完成限次修复闭环。
- 设计 **Context Package 与分层记忆机制**，按任务筛选目标、约束、最近对话和相关代码；使用 JSONL/Markdown 保存会话、项目及全局记忆，支持历史恢复、中断检测和上下文窗口阈值压缩。
- 封装 **OpenAI-compatible 模型层**，支持每个 Agent 独立模型映射、API 连通测试、token 统计及 LongCat SSE/原生工具协议；通过动作去重、磁盘快照刷新、输出预算、锚点代码块替换和路径安全校验提升长任务稳定性。

**项目成果：** 主项目 90 个 Python 测试全部通过，并通过 Ruff、TypeScript 检查和 Vite 构建；使用该 Agent 完整生成一个基于 1324 条健身动作数据的 FastAPI + HTML 网站，生成项目 24 个测试全部通过。

**尚未落地的生产能力：** LangGraph checkpoint 节点级恢复、跨进程工具幂等、容器级 Shell 沙箱、Human-in-the-loop 高风险审批、代码 RAG/符号索引、标准化 Agent 评测集、自动模型路由以及多用户任务队列。
