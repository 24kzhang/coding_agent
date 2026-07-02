# 多智能体编程系统

这是一个基于 `uv`、FastAPI、LangGraph 和 React 的本地 vibe coding agent 系统。系统包含上下文管理、Plan 生成、代码仓库读取、Coding、验证测试和文档生成等多个智能体，并提供浏览器调试台查看模型、任务、工具调用、token 估算、文件变更和测试结果。

## 快速启动

Mac：

```bash
./start.command
```

Windows：

```bat
start.bat
```

也可以手动启动：

```bash
uv sync
uv run python scripts/dev.py
```

详细学习文档见 [docs/guide.md](docs/guide.md)。
