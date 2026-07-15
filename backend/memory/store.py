from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MemoryStore:
    """管理会话记忆、项目长期记忆和全局长期记忆。

    会话记忆使用 jsonl，便于持续追加和故障恢复；长期记忆使用 md，便于用户直接阅读。
    """

    def __init__(self, root: Path | None = None):
        # project_root 是本 agent 项目根目录，用于定位默认 memory/data。
        project_root = Path(__file__).resolve().parents[2]
        # root 是所有记忆文件的根目录；测试时可传临时目录隔离真实 memory。
        self.root = root or project_root / "memory" / "data"
        # 确保记忆根目录存在。
        self.root.mkdir(parents=True, exist_ok=True)
        # global_path 是跨项目长期记忆文件。
        self.global_path = self.root / "global.md"
        # 首次启动时创建全局长期记忆文件，便于用户直接打开阅读。
        if not self.global_path.exists():
            self.global_path.write_text("# 全局长期记忆\n\n暂无。\n", encoding="utf-8")

    def project_id(self, workdir: str) -> str:
        """根据项目绝对路径生成稳定项目 id。"""

        # digest 使用 sha1 前 12 位，避免把中文路径或特殊字符直接作为目录名。
        digest = hashlib.sha1(str(Path(workdir).resolve()).encode("utf-8")).hexdigest()[:12]
        return digest

    def project_path(self, workdir: str) -> Path:
        """返回项目 memory 目录路径，但不主动创建目录。"""

        return self.root / "projects" / self.project_id(workdir)

    def project_dir(self, workdir: str) -> Path:
        """返回项目 memory 目录，并确保 project.md 和 sessions 目录存在。"""

        # path 是该项目在 memory/data/projects 下的独立目录。
        path = self.project_path(workdir)
        # sessions 保存该项目下所有会话 jsonl。
        (path / "sessions").mkdir(parents=True, exist_ok=True)
        # project_md 保存当前项目长期记忆。
        project_md = path / "project.md"
        # 首次见到项目时写入项目路径，后续 list_history 会从这里反查真实 workdir。
        if not project_md.exists():
            project_md.write_text(
                f"# 项目长期记忆\n\n项目路径：`{Path(workdir).resolve()}`\n\n",
                encoding="utf-8",
            )
        return path

    def session_path(self, workdir: str, session_id: str) -> Path:
        """返回指定会话的 jsonl 文件路径，并确保项目 memory 目录存在。"""

        return self.project_dir(workdir) / "sessions" / f"{session_id}.jsonl"

    def delete_session(self, workdir: str, session_id: str) -> bool:
        """删除某个会话的 memory 文件，不删除真实项目文件。"""

        # path 指向待删除的会话 jsonl；这里不调用 project_dir，避免删除不存在会话时创建目录。
        path = self.project_path(workdir) / "sessions" / f"{session_id}.jsonl"
        # 删除前确认路径一定在 memory/data/projects 内。
        self._ensure_under_projects(path)
        if not path.exists():
            return False
        path.unlink()
        return True

    def delete_project(self, workdir: str) -> bool:
        """删除某个项目的全部 memory 目录，不删除用户真实项目目录。"""

        # path 是项目在 memory/data/projects 下的目录。
        path = self.project_path(workdir)
        # 删除前确认路径一定在 memory/data/projects 内。
        self._ensure_under_projects(path)
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True

    def rename_session(self, workdir: str, session_id: str, title: str) -> dict[str, Any]:
        """写入一条会话重命名记录，历史列表会优先读取最新 rename。"""

        # clean_title 去掉多余空白并限制长度，避免前端状态卡片被长文本撑乱。
        clean_title = " ".join(title.strip().split())[:60]
        if not clean_title:
            raise ValueError("会话名称不能为空")
        return self.append(
            workdir,
            session_id,
            "manager",
            "session",
            "rename",
            f"会话重命名：{clean_title}",
            {"title": clean_title},
        )

    def session_title(self, workdir: str, session_id: str) -> str:
        """读取会话展示名称；没有自定义名称时返回 session_id。"""

        # path 是会话 jsonl 文件路径；不存在时返回空字符串，让调用方自行处理。
        path = self.project_path(workdir) / "sessions" / f"{session_id}.jsonl"
        if not path.exists():
            return ""
        return self._session_title(self._read_session_file(path)) or session_id

    def append(
        self,
        workdir: str,
        session_id: str,
        agent: str,
        tool: str,
        kind: str,
        out: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """追加一条短字段记忆，字段名短但可辨识。"""
        # path 是本次要追加的会话 jsonl 文件路径。
        path = self.session_path(workdir, session_id)
        # last_id 保存当前会话最后一条记录 id，新记录会在它基础上 +1。
        last_id = 0
        if path.exists():
            # 只读取最后一条记录即可得到当前最大 id。
            for rec in self.read_session(workdir, session_id)[-1:]:
                last_id = int(rec.get("id", 0))
        # rec 是最终写入 jsonl 的一条记忆记录，字段名短是为了减少上下文占用。
        rec = {
            "id": last_id + 1,
            "ts": datetime.now(UTC).isoformat(),
            "ag": agent,
            "tl": tool,
            "k": kind,
            "out": out,
            "m": meta or {},
        }
        # 追加写入一行 JSON，jsonl 格式便于任务中断后保留已有记录。
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def read_session(self, workdir: str, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """读取会话 jsonl；limit 为空读全部，否则只返回最后 limit 条。"""

        # path 是会话 jsonl 文件路径；调用 session_path 会确保目录存在。
        path = self.session_path(workdir, session_id)
        if not path.exists():
            return []
        # records 保存解析后的所有记忆记录。
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                # line 是 jsonl 的一行，空行直接跳过。
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records[-limit:] if limit else records

    def list_history(self) -> list[dict[str, Any]]:
        """按项目聚合历史会话，供前端恢复对话使用。"""
        # projects 是最终返回给前端的项目列表。
        projects: list[dict[str, Any]] = []
        # project_dir 是 memory/data/projects 下的每个项目记忆目录。
        for project_dir in sorted((self.root / "projects").glob("*")):
            if not project_dir.is_dir():
                continue
            # workdir 从 project.md 中读取，表示用户真实项目路径。
            workdir = self._read_project_workdir(project_dir)
            if not workdir:
                continue
            # sessions 是当前项目下所有会话摘要，None 表示空文件或无法恢复的会话。
            sessions = [self._session_summary(workdir, path) for path in sorted((project_dir / "sessions").glob("*.jsonl"))]
            sessions = [session for session in sessions if session is not None]
            # 会话按最后更新时间倒序排列，前端默认先看到最近会话。
            sessions.sort(key=lambda item: item["updated_at"], reverse=True)
            projects.append(
                {
                    # id 是项目 memory 目录名，也就是 project_id(workdir)。
                    "id": project_dir.name,
                    # name 是前端展示用项目名称，默认取项目目录名。
                    "name": Path(workdir).name or workdir,
                    # workdir 是用户真实项目路径，恢复会话时必须带回前端。
                    "workdir": workdir,
                    # sessions 是该项目下的历史会话列表。
                    "sessions": sessions,
                }
            )
        # 项目按最近会话更新时间倒序排列；没有会话的项目排在后面。
        projects.sort(key=lambda item: item["sessions"][0]["updated_at"] if item["sessions"] else "", reverse=True)
        return projects

    def find_session_workdir(self, session_id: str) -> str:
        """从记忆目录中反查会话所属项目路径。"""
        # project 是 memory/data/projects 下的项目记忆目录。
        for project in (self.root / "projects").glob("*"):
            # session_file 是当前项目下可能匹配的会话 jsonl。
            session_file = project / "sessions" / f"{session_id}.jsonl"
            if session_file.exists():
                return self._read_project_workdir(project)
        return ""

    def interrupted(self, workdir: str, session_id: str) -> bool:
        """判断会话最后一次运行是否可能异常中断。"""

        # records 是当前会话全部记忆记录。
        records = self.read_session(workdir, session_id)
        if not records:
            return False
        # last 是最后一条记录；正常结束时应为 manager/final/result。
        last = records[-1]
        return not (last.get("ag") == "manager" and last.get("k") == "result")

    def project_memory(self, workdir: str) -> str:
        """读取当前项目长期记忆 Markdown。"""

        return (self.project_dir(workdir) / "project.md").read_text(encoding="utf-8")

    def global_memory(self) -> str:
        """读取全局长期记忆 Markdown。"""

        return self.global_path.read_text(encoding="utf-8")

    def update_project_memory(self, workdir: str, text: str) -> None:
        """覆盖写入当前项目长期记忆。"""

        # path 是当前项目长期记忆文件。
        path = self.project_dir(workdir) / "project.md"
        path.write_text(text.strip() + "\n", encoding="utf-8")

    def update_global_memory(self, text: str) -> None:
        """覆盖写入全局长期记忆。"""

        self.global_path.write_text(text.strip() + "\n", encoding="utf-8")

    def maybe_compress(self, workdir: str, session_id: str, ctx: int) -> None:
        """当会话记忆超过上下文窗口约 85% 时进行粗粒度压缩。

        这里按 4 字符约等于 1 token 做近似估算。压缩保留最早、最新和阶段摘要，
        且重复工具输出只保留更晚的记录。
        """
        # records 是压缩前的完整会话记录。
        records = self.read_session(workdir, session_id)
        # raw 是把所有记录重新序列化后的字符串，用于粗略估算 token。
        raw = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
        # 如果估算 token 未超过上下文窗口 85%，直接不压缩。
        if len(raw) / 4 <= ctx * 0.85:
            return
        # latest_by_tool 保存每个 agent/tool 组合的最新记录，去掉重复工具噪音。
        latest_by_tool: dict[tuple[str, str], dict[str, Any]] = {}
        for rec in records:
            latest_by_tool[(rec.get("ag", ""), rec.get("tl", ""))] = rec
        # first 保留最开始几条记录，方便知道会话如何开始。
        first = records[:3]
        # latest 保留最后 20 条记录，方便恢复最近阶段。
        latest = records[-20:]
        # summary 是压缩后插入的说明记录。
        summary = {
            "id": 1,
            "ts": datetime.now(UTC).isoformat(),
            "ag": "manager",
            "tl": "memory",
            "k": "summary",
            "out": "会话记忆已压缩：保留开头、最新阶段和每个工具最近一次输出。",
            "m": {"old_count": len(records)},
        }
        # merged 是候选保留记录集合。
        merged = [summary] + first + list(latest_by_tool.values()) + latest
        # seen 用原记录 id 去重，避免同一条记录因为多种策略被保留多次。
        seen: set[int] = set()
        # compact 是最终写回文件的压缩后记录。
        compact: list[dict[str, Any]] = []
        for rec in merged:
            # rid 是原记录 id；summary 的 id 固定为 1。
            rid = int(rec.get("id", 0))
            if rid not in seen:
                seen.add(rid)
                compact.append(rec)
        # path 是原会话 jsonl，压缩会直接覆盖写回。
        path = self.session_path(workdir, session_id)
        with path.open("w", encoding="utf-8") as fh:
            # idx 是压缩后重新分配的连续 id。
            for idx, rec in enumerate(compact, start=1):
                rec["id"] = idx
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _read_project_workdir(self, project_dir: Path) -> str:
        """从项目 project.md 中读取真实项目路径。"""

        # project_md 是项目长期记忆文件，里面第一段保存了项目路径。
        project_md = project_dir / "project.md"
        if not project_md.exists():
            return ""
        # text 是 project.md 原文。
        text = project_md.read_text(encoding="utf-8")
        # marker 是项目路径在 md 中的固定前缀。
        marker = "项目路径：`"
        if marker not in text:
            return ""
        # start/end 定位反引号包裹的真实项目路径。
        start = text.find(marker) + len(marker)
        end = text.find("`", start)
        return text[start:end] if end != -1 else ""

    def _session_summary(self, workdir: str, path: Path) -> dict[str, Any] | None:
        """把一个会话 jsonl 文件转换成历史弹窗需要的摘要。"""

        # records 是会话内全部记忆记录。
        records = self._read_session_file(path)
        if not records:
            return None
        # messages 是可恢复到对话窗口的用户消息和最终回复。
        messages = self._history_messages(records)
        # title 是会话展示名；没有自定义名时用文件名也就是 session_id。
        title = self._session_title(records) or path.stem
        # last 是最后一条记录，用于更新时间和中断判断。
        last = records[-1]
        return {
            "id": path.stem,
            "title": title,
            "workdir": workdir,
            "updated_at": str(last.get("ts", "")),
            "interrupted": not (last.get("ag") == "manager" and last.get("k") == "result"),
            "messages": messages,
        }

    def _read_session_file(self, path: Path) -> list[dict[str, Any]]:
        """直接读取指定 jsonl 文件，不根据 workdir/session_id 重新计算路径。"""

        # records 保存解析后的记忆记录。
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                # line 是 jsonl 的一行，空行忽略。
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _history_messages(self, records: list[dict[str, Any]]) -> list[dict[str, str]]:
        """从完整会话记录中提取前端对话窗口需要恢复的消息。"""

        # messages 只包含用户输入和 agent 最终结果，不包含中间事件流。
        messages: list[dict[str, str]] = []
        for rec in records:
            # 用户输入记录由 backend/main.py 在 chat_stream 开始时写入。
            if rec.get("ag") == "user" and rec.get("tl") == "input" and rec.get("k") == "message":
                messages.append({"id": str(rec.get("id", "")), "role": "user", "content": str(rec.get("out", ""))})
            # agent 最终回复记录由 AgentGraph.final() 写入。
            if rec.get("ag") == "manager" and rec.get("tl") == "final" and rec.get("k") == "result":
                messages.append(
                    {
                        "id": str(rec.get("id", "")),
                        "role": "agent",
                        "content": self._format_result(rec),
                    }
                )
        return messages

    def _session_title(self, records: list[dict[str, Any]]) -> str:
        """从会话记录中计算展示名称，最新 rename 优先。"""

        # 倒序查找 rename，保证最近一次重命名生效。
        for rec in reversed(records):
            if rec.get("ag") == "manager" and rec.get("tl") == "session" and rec.get("k") == "rename":
                # title 来自 rename 记录的元数据。
                title = str((rec.get("m") or {}).get("title") or "").strip()
                if title:
                    return title[:60]
        # 如果没有 rename，再看创建会话时是否传入过自定义名称。
        for rec in records:
            if rec.get("ag") == "manager" and rec.get("tl") == "session" and rec.get("k") == "start":
                # title 是创建会话时的初始标题。
                title = str((rec.get("m") or {}).get("title") or "").strip()
                # custom=True 表示用户真的输入过标题；否则默认标题不展示，改用 session_id。
                if title and (rec.get("m") or {}).get("custom"):
                    return title[:60]
                break
        return ""

    def _format_result(self, rec: dict[str, Any]) -> str:
        """把 manager/final/result 记录转换成前端历史对话中的 Agent 文本。"""

        # result 是 final 节点写入到 m 字段的结构化 TaskResult。
        result = rec.get("m") or {}
        # lines 是最终拼接成多行文本的片段列表。
        lines = [str(result.get("summary") or rec.get("out") or "")]
        # files 是本轮任务修改的文件列表。
        files = result.get("files") or []
        # tests 是本轮验证结果列表。
        tests = result.get("tests") or []
        if files:
            lines.append(f"文件变更：{'、'.join(str(item) for item in files)}")
        if tests:
            # checks 保存每条测试命令的中文通过/失败摘要。
            checks = []
            for item in tests:
                checks.append(f"{item.get('cmd', '检查')}：{'通过' if item.get('ok') else '失败'}")
            lines.append(f"验证结果：{'；'.join(checks)}")
        if result.get("doc_path"):
            lines.append(f"文档：{result['doc_path']}")
        if result.get("plan_path"):
            lines.append(f"计划：{result['plan_path']}")
        return "\n".join(line for line in lines if line)

    def _ensure_under_projects(self, path: Path) -> None:
        """确认待删除路径位于 memory/data/projects 内，防止误删真实项目。"""

        # projects_root 是允许删除的 memory 项目根目录。
        projects_root = (self.root / "projects").resolve()
        # resolved 是待检查路径的真实绝对路径。
        resolved = path.resolve()
        if projects_root != resolved and projects_root not in resolved.parents:
            raise ValueError("非法记忆路径")
