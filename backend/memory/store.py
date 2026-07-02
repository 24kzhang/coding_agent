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
        project_root = Path(__file__).resolve().parents[2]
        self.root = root or project_root / "memory" / "data"
        self.root.mkdir(parents=True, exist_ok=True)
        self.global_path = self.root / "global.md"
        if not self.global_path.exists():
            self.global_path.write_text("# 全局长期记忆\n\n暂无。\n", encoding="utf-8")

    def project_id(self, workdir: str) -> str:
        digest = hashlib.sha1(str(Path(workdir).resolve()).encode("utf-8")).hexdigest()[:12]
        return digest

    def project_path(self, workdir: str) -> Path:
        return self.root / "projects" / self.project_id(workdir)

    def project_dir(self, workdir: str) -> Path:
        path = self.project_path(workdir)
        (path / "sessions").mkdir(parents=True, exist_ok=True)
        project_md = path / "project.md"
        if not project_md.exists():
            project_md.write_text(
                f"# 项目长期记忆\n\n项目路径：`{Path(workdir).resolve()}`\n\n暂无。\n",
                encoding="utf-8",
            )
        return path

    def session_path(self, workdir: str, session_id: str) -> Path:
        return self.project_dir(workdir) / "sessions" / f"{session_id}.jsonl"

    def delete_session(self, workdir: str, session_id: str) -> bool:
        path = self.project_path(workdir) / "sessions" / f"{session_id}.jsonl"
        self._ensure_under_projects(path)
        if not path.exists():
            return False
        path.unlink()
        return True

    def delete_project(self, workdir: str) -> bool:
        path = self.project_path(workdir)
        self._ensure_under_projects(path)
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True

    def rename_session(self, workdir: str, session_id: str, title: str) -> dict[str, Any]:
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
        path = self.session_path(workdir, session_id)
        last_id = 0
        if path.exists():
            for rec in self.read_session(workdir, session_id)[-1:]:
                last_id = int(rec.get("id", 0))
        rec = {
            "id": last_id + 1,
            "ts": datetime.now(UTC).isoformat(),
            "ag": agent,
            "tl": tool,
            "k": kind,
            "out": out,
            "m": meta or {},
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def read_session(self, workdir: str, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        path = self.session_path(workdir, session_id)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records[-limit:] if limit else records

    def list_history(self) -> list[dict[str, Any]]:
        """按项目聚合历史会话，供前端恢复对话使用。"""
        projects: list[dict[str, Any]] = []
        for project_dir in sorted((self.root / "projects").glob("*")):
            if not project_dir.is_dir():
                continue
            workdir = self._read_project_workdir(project_dir)
            if not workdir:
                continue
            sessions = [self._session_summary(workdir, path) for path in sorted((project_dir / "sessions").glob("*.jsonl"))]
            sessions = [session for session in sessions if session is not None]
            sessions.sort(key=lambda item: item["updated_at"], reverse=True)
            projects.append(
                {
                    "id": project_dir.name,
                    "name": Path(workdir).name or workdir,
                    "workdir": workdir,
                    "sessions": sessions,
                }
            )
        projects.sort(key=lambda item: item["sessions"][0]["updated_at"] if item["sessions"] else "", reverse=True)
        return projects

    def find_session_workdir(self, session_id: str) -> str:
        """从记忆目录中反查会话所属项目路径。"""
        for project in (self.root / "projects").glob("*"):
            session_file = project / "sessions" / f"{session_id}.jsonl"
            if session_file.exists():
                return self._read_project_workdir(project)
        return ""

    def interrupted(self, workdir: str, session_id: str) -> bool:
        records = self.read_session(workdir, session_id)
        if not records:
            return False
        last = records[-1]
        return not (last.get("ag") == "manager" and last.get("k") == "result")

    def project_memory(self, workdir: str) -> str:
        return (self.project_dir(workdir) / "project.md").read_text(encoding="utf-8")

    def global_memory(self) -> str:
        return self.global_path.read_text(encoding="utf-8")

    def update_project_memory(self, workdir: str, text: str) -> None:
        path = self.project_dir(workdir) / "project.md"
        path.write_text(text.strip() + "\n", encoding="utf-8")

    def update_global_memory(self, text: str) -> None:
        self.global_path.write_text(text.strip() + "\n", encoding="utf-8")

    def maybe_compress(self, workdir: str, session_id: str, ctx: int) -> None:
        """当会话记忆超过上下文窗口约 85% 时进行粗粒度压缩。

        这里按 4 字符约等于 1 token 做近似估算。压缩保留最早、最新和阶段摘要，
        且重复工具输出只保留更晚的记录。
        """
        records = self.read_session(workdir, session_id)
        raw = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
        if len(raw) / 4 <= ctx * 0.85:
            return
        latest_by_tool: dict[tuple[str, str], dict[str, Any]] = {}
        for rec in records:
            latest_by_tool[(rec.get("ag", ""), rec.get("tl", ""))] = rec
        first = records[:3]
        latest = records[-20:]
        summary = {
            "id": 1,
            "ts": datetime.now(UTC).isoformat(),
            "ag": "manager",
            "tl": "memory",
            "k": "summary",
            "out": "会话记忆已压缩：保留开头、最新阶段和每个工具最近一次输出。",
            "m": {"old_count": len(records)},
        }
        merged = [summary] + first + list(latest_by_tool.values()) + latest
        seen: set[int] = set()
        compact: list[dict[str, Any]] = []
        for rec in merged:
            rid = int(rec.get("id", 0))
            if rid not in seen:
                seen.add(rid)
                compact.append(rec)
        path = self.session_path(workdir, session_id)
        with path.open("w", encoding="utf-8") as fh:
            for idx, rec in enumerate(compact, start=1):
                rec["id"] = idx
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _read_project_workdir(self, project_dir: Path) -> str:
        project_md = project_dir / "project.md"
        if not project_md.exists():
            return ""
        text = project_md.read_text(encoding="utf-8")
        marker = "项目路径：`"
        if marker not in text:
            return ""
        start = text.find(marker) + len(marker)
        end = text.find("`", start)
        return text[start:end] if end != -1 else ""

    def _session_summary(self, workdir: str, path: Path) -> dict[str, Any] | None:
        records = self._read_session_file(path)
        if not records:
            return None
        messages = self._history_messages(records)
        title = self._session_title(records) or path.stem
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
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _history_messages(self, records: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for rec in records:
            if rec.get("ag") == "user" and rec.get("tl") == "input" and rec.get("k") == "message":
                messages.append({"id": str(rec.get("id", "")), "role": "user", "content": str(rec.get("out", ""))})
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
        for rec in reversed(records):
            if rec.get("ag") == "manager" and rec.get("tl") == "session" and rec.get("k") == "rename":
                title = str((rec.get("m") or {}).get("title") or "").strip()
                if title:
                    return title[:60]
        for rec in records:
            if rec.get("ag") == "manager" and rec.get("tl") == "session" and rec.get("k") == "start":
                title = str((rec.get("m") or {}).get("title") or "").strip()
                if title and (rec.get("m") or {}).get("custom"):
                    return title[:60]
                break
        return ""

    def _format_result(self, rec: dict[str, Any]) -> str:
        result = rec.get("m") or {}
        lines = [str(result.get("summary") or rec.get("out") or "")]
        files = result.get("files") or []
        tests = result.get("tests") or []
        if files:
            lines.append(f"文件变更：{'、'.join(str(item) for item in files)}")
        if tests:
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
        projects_root = (self.root / "projects").resolve()
        resolved = path.resolve()
        if projects_root != resolved and projects_root not in resolved.parents:
            raise ValueError("非法记忆路径")
