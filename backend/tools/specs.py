from __future__ import annotations

from typing import Any


def tool_specs() -> list[dict[str, Any]]:
    """返回 Coding 原生函数工具及其本地执行策略。"""

    string = {"type": "string"}
    public_path = {"type": "string", "description": "普通文本路径；有交付目录时新增文件可用项目相对路径，也可明确带交付前缀。输入文件按实际工作区路径读取。若用户明确要求参数前缀必须遵守。不可读写 .local/、.env、.ssh/、.aws/，真实配置通过 model_copy。"}
    # 原生函数参数同样计入模型输出。限制单次正文大小，避免模型把多个大文件塞进
    # 一次响应后在输出上限处截断，造成整批 JSON 无法解析和重复计费。
    short_text = {"type": "string", "maxLength": 6_000}
    match_text = {"type": "string", "maxLength": 6_000}
    integer = {"type": "integer"}
    array = {"type": "array", "items": string}
    definitions = [
        ("read_file", "读取工作区文本文件的一段内容", {"path": public_path, "start": integer, "max_chars": integer}, ["path"], "allow", "read"),
        ("list_files", "列出工作区内受支持的文本文件", {}, [], "allow", "read"),
        ("search_files", "在工作区文本文件中搜索内容", {"query": string, "max_results": integer}, ["query"], "allow", "read"),
        (
            "write_file",
            "写入一个完整文本文件；正文不得超过 10000 字符，更长文件先写首块再用 append_file 分块追加",
            {"path": public_path, "content": short_text},
            ["path", "content"],
            "allow",
            "write",
        ),
        (
            "append_file",
            "向一个文本文件末尾追加不超过 10000 字符的一块内容",
            {"path": public_path, "content": short_text},
            ["path", "content"],
            "allow",
            "write",
        ),
        (
            "replace_file",
            "精确替换文件中的一小段文本",
            {"path": public_path, "old": match_text, "new": short_text, "expected": integer},
            ["path", "old", "new"],
            "allow",
            "write",
        ),
        (
            "replace_block",
            "按开始和结束锚点替换不超过 10000 字符的代码块",
            {
                "path": public_path,
                "start_marker": match_text,
                "end_marker": match_text,
                "content": short_text,
            },
            ["path", "start_marker", "end_marker", "content"],
            "allow",
            "write",
        ),
        ("assert_text_absent", "断言指定文件中不存在禁用文本", {"paths": array, "texts": array}, ["paths", "texts"], "allow", "read"),
        ("run_command", "默认在用户声明的交付根目录运行一个独立命令；明确带工作区交付前缀的命令在父工作区运行。uv --project 不切目录：子项目模块调用应使用 uv run --directory <交付前缀> 和项目相对参数，或提供项目包导入配置。不支持管道、重定向、分号、&& 等复合 shell 语法，也不支持 python -c/node -e 等解释器内联代码。应写入脚本后运行。只读 cat/grep/sed 须改用文件工具；已有仓库索引时目录确认返回缓存；Git 仅在执行目录有独立 .git 时可用", {"cmd": string}, ["cmd"], "ask", "write"),
        ("git_status", "读取 Git 工作区状态", {}, [], "allow", "read"),
        ("git_diff", "读取 Git 差异", {}, [], "allow", "read"),
        ("model_copy", "把已配置模型安全复制为 JSON 配置；目标必须位于交付项目的 .local/ 私密目录，默认 .local/model.json。自动写入忽略规则并限制权限；字段包含 id、base_url、model、api_key、protocol、timeout、default_effort，不能选择其他目录或把格式当成 TOML", {"model_id": string, "target": string}, ["model_id"], "allow", "write"),
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
