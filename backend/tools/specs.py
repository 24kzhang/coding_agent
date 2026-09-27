from __future__ import annotations

from typing import Any


def tool_specs() -> list[dict[str, Any]]:
    """返回 Coding 原生函数工具及其本地执行策略。"""

    string = {"type": "string"}
    integer = {"type": "integer"}
    array = {"type": "array", "items": string}
    definitions = [
        ("read_file", "读取工作区文本文件的一段内容", {"path": string, "start": integer, "max_chars": integer}, ["path"], "allow", "read"),
        ("list_files", "列出工作区内受支持的文本文件", {}, [], "allow", "read"),
        ("search_files", "在工作区文本文件中搜索内容", {"query": string, "max_results": integer}, ["query"], "allow", "read"),
        ("write_file", "写入完整文本文件", {"path": string, "content": string}, ["path", "content"], "allow", "write"),
        ("append_file", "向文本文件末尾追加内容", {"path": string, "content": string}, ["path", "content"], "allow", "write"),
        ("replace_file", "精确替换文件中的文本", {"path": string, "old": string, "new": string, "expected": integer}, ["path", "old", "new"], "allow", "write"),
        ("replace_block", "按开始和结束锚点替换代码块", {"path": string, "start_marker": string, "end_marker": string, "content": string}, ["path", "start_marker", "end_marker", "content"], "allow", "write"),
        ("assert_text_absent", "断言指定文件中不存在禁用文本", {"paths": array, "texts": array}, ["paths", "texts"], "allow", "read"),
        ("run_command", "在工作区运行一个受限命令", {"cmd": string}, ["cmd"], "ask", "write"),
        ("git_status", "读取 Git 工作区状态", {}, [], "allow", "read"),
        ("git_diff", "读取 Git 差异", {}, [], "allow", "read"),
        ("model_copy", "把指定模型的私密配置安全复制到工作区", {"model_id": string, "target": string}, ["model_id"], "allow", "write"),
        ("finish", "所有实现与检查完成后提交中文摘要", {"summary": string}, ["summary"], "allow", "finish"),
    ]
    return [
        {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "risk": "high" if permission == "ask" else ("medium" if mode == "write" else "low"),
            "permission": permission,
            "timeout": 180 if name == "run_command" else 30,
            "concurrency": "parallel" if mode == "read" else "serial",
            "output_limit": 12_000,
            "mode": mode,
        }
        for name, description, properties, required, permission, mode in definitions
    ]
