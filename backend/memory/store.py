from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from backend.clock import utc_stamp


class MemoryStore:
    """管理会话记忆、项目长期记忆和全局长期记忆。

    会话记忆使用 jsonl，便于持续追加和故障恢复；长期记忆使用 md，便于用户直接阅读。
    """

    def __init__(self, root: Path | None = None):
        # project_root 是本 agent 项目根目录，用于定位默认 memory/data。
        project_root = Path(__file__).resolve().parents[2]
        # root 是所有记忆文件的根目录；测试时可传临时目录隔离真实 memory。
        self.root = root or project_root / "memory" / "data"
        # lock 串行化当前进程中的 JSONL 追加、压缩和删除，避免并发任务写坏文件。
        self.lock = threading.RLock()
        # last_ids 缓存每个会话最后一个递增 id，首次访问时才从磁盘计算。
        self.last_ids: dict[Path, int] = {}
        # session_index 缓存 session_id 到 workdir 的映射，避免每次请求扫描全部项目。
        self.session_index: dict[str, str] = {}
        # 确保记忆根目录存在。
        self.root.mkdir(parents=True, exist_ok=True)
        # 旧版把所有 Agent 写进同一个 sessions/<id>.jsonl；新版首次启动时只归档，不迁移。
        self._archive_legacy_sessions()
        # global_path 是跨项目长期记忆文件。
        self.global_path = self.root / "global.md"
        # 首次启动时创建全局长期记忆文件，便于用户直接打开阅读。
        if not self.global_path.exists():
            self.global_path.write_text("# 全局长期记忆\n\n暂无。\n", encoding="utf-8")
        # 启动时只扫描一次已有会话，后续创建和删除会增量维护索引。
        self._rebuild_session_index()
        # 旧版 isoformat 时间带微秒和 +00:00；启动时一次性迁移为秒级 Z 格式。
        self._normalize_existing_timestamps()

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

        with self.lock:
            # path 是该项目在 memory/data/projects 下的独立目录。
            path = self.project_path(workdir)
            # sessions 保存该项目下所有会话 jsonl。
            (path / "sessions").mkdir(parents=True, exist_ok=True)
            # meta_path 只保存系统定位项目所需的元数据，不混入长期记忆正文。
            meta_path = path / "meta.json"
            if not meta_path.exists():
                self._atomic_write(
                    meta_path,
                    json.dumps({"workdir": str(Path(workdir).resolve())}, ensure_ascii=False, indent=2) + "\n",
                )
            # project_md 只保存项目级长期偏好和事实，不再把目录路径当成记忆内容。
            project_md = path / "project.md"
            if not project_md.exists():
                self._atomic_write(project_md, "# 项目长期记忆\n\n暂无。\n")
            return path

    def session_dir(self, workdir: str, session_id: str) -> Path:
        """返回新版会话目录，并创建各类独立记忆容器。"""

        safe_id = self._safe_session_id(session_id)
        path = self.project_dir(workdir) / "sessions" / safe_id
        (path / "agents").mkdir(parents=True, exist_ok=True)
        (path / "files").mkdir(parents=True, exist_ok=True)
        meta = path / "meta.json"
        if not meta.exists():
            self._atomic_write(
                meta,
                json.dumps(
                    {"version": 2, "id": safe_id, "workdir": str(Path(workdir).resolve())},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        return path

    def session_path(self, workdir: str, session_id: str) -> Path:
        """返回会话的用户对话日志路径，保留旧调用方使用的接口名。"""

        return self.session_dir(workdir, session_id) / "chat.jsonl"

    def delete_session(self, workdir: str, session_id: str) -> bool:
        """删除某个会话的 memory 文件，不删除真实项目文件。"""

        # path 指向待删除的会话 jsonl；这里不调用 project_dir，避免删除不存在会话时创建目录。
        # safe_id 是通过白名单校验后的会话 id。
        safe_id = self._safe_session_id(session_id)
        path = self.project_path(workdir) / "sessions" / safe_id
        # 删除前确认路径一定在 memory/data/projects 内。
        self._ensure_under_projects(path)
        with self.lock:
            if not path.exists():
                return False
            shutil.rmtree(path)
            self.last_ids = {
                item: value
                for item, value in self.last_ids.items()
                if item != path and path not in item.parents
            }
            self.session_index.pop(safe_id, None)
            return True

    def delete_project(self, workdir: str) -> bool:
        """删除某个项目的全部 memory 目录，不删除用户真实项目目录。"""

        # path 是项目在 memory/data/projects 下的目录。
        path = self.project_path(workdir)
        # 删除前确认路径一定在 memory/data/projects 内。
        self._ensure_under_projects(path)
        with self.lock:
            if not path.exists():
                return False
            # session_ids 是删除前需要从内存索引移除的全部会话。
            session_ids = [session_path.name for session_path in (path / "sessions").glob("*") if session_path.is_dir()]
            shutil.rmtree(path)
            for session_id in session_ids:
                self.session_index.pop(session_id, None)
            # 删除项目后丢弃所有指向该目录下文件的 last_ids 缓存。
            self.last_ids = {item: value for item, value in self.last_ids.items() if path not in item.parents}
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
        # safe_id 保证读取路径无法由外部参数构造到 sessions 目录之外。
        safe_id = self._safe_session_id(session_id)
        path = self.project_path(workdir) / "sessions" / safe_id
        if not path.exists():
            return ""
        return self._session_title(self._read_all_records(path)) or session_id

    def session_capabilities(self, workdir: str, session_id: str) -> dict[str, Any]:
        """读取会话最近一次任务选择的 Skills、MCP 服务和 Shell 模式。"""

        # safe_id 保证外部 session_id 不能构造越界路径。
        safe_id = self._safe_session_id(session_id)
        # path 直接定位已有会话，不因读取不存在会话而创建 memory 目录。
        path = self.project_path(workdir) / "sessions" / safe_id
        if not path.exists():
            return {"skills": [], "mcps": [], "container_shell": True}
        return self._session_capabilities(self._read_all_records(path))

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
        with self.lock:
            session_dir = self.session_dir(workdir, session_id)
            # 用户对话和生命周期进入 chat；交接进入 tasks；其余过程进入对应 Agent。
            if tool == "task" or kind in {"handoff", "outcome", "needs_clarification"}:
                path = session_dir / "tasks.jsonl"
            elif agent == "user" or tool in {"session", "input", "guidance", "capability", "run", "final"}:
                path = session_dir / "chat.jsonl"
            else:
                safe_agent = re.sub(r"[^A-Za-z0-9._-]+", "-", str(agent)).strip("-")[:40] or "unknown"
                path = session_dir / "agents" / safe_agent / "log.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
            # last_id 使用会话目录作为键，保证分文件写入仍使用全局递增序号。
            last_id = self.last_ids.get(session_dir)
            if last_id is None:
                records = self._read_all_records(session_dir)
                last_id = max((int(rec.get("id", 0)) for rec in records), default=0)
            # rec 是最终写入 jsonl 的一条记忆记录，字段名短是为了减少上下文占用。
            rec = {
                "id": last_id + 1,
                "ts": utc_stamp(),
                "ag": str(agent)[:40],
                "tl": str(tool)[:40],
                "k": str(kind)[:40],
                "out": str(out),
                "m": meta or {},
            }
            # 追加写入一行 JSON；flush 保证线程返回前数据已经交给操作系统。
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                fh.flush()
            self.last_ids[session_dir] = rec["id"]
            self.session_index[self._safe_session_id(session_id)] = str(Path(workdir).resolve())
            return rec

    def read_session(self, workdir: str, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """读取会话 jsonl；limit 为空读全部，否则只返回最后 limit 条。"""

        # path 是会话 jsonl 文件路径；调用 session_path 会确保目录存在。
        path = self.session_dir(workdir, session_id)
        with self.lock:
            records = self._active_records(path)
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
            sessions = [
                self._session_summary(workdir, path)
                for path in sorted((project_dir / "sessions").glob("*"))
                if path.is_dir()
            ]
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

        # safe_id 先校验接口传入值，避免路径穿越，也与索引键格式保持一致。
        safe_id = self._safe_session_id(session_id)
        with self.lock:
            # indexed 是启动扫描或后续 append 记录下来的项目路径。
            indexed = self.session_index.get(safe_id, "")
            if indexed:
                return indexed
            # 缓存没有命中时做一次兼容扫描，用于发现进程外新写入的旧会话。
            for project in (self.root / "projects").glob("*"):
                # session_file 是当前项目下可能匹配的会话 jsonl。
                session_file = project / "sessions" / safe_id
                if session_file.is_dir():
                    workdir = self._read_project_workdir(project)
                    if workdir:
                        self.session_index[safe_id] = workdir
                    return workdir
            return ""

    def interrupted(self, workdir: str, session_id: str) -> bool:
        """判断会话最后一次运行是否可能异常中断。"""

        # records 是当前会话全部记忆记录。
        records = self.read_session(workdir, session_id)
        return self._records_interrupted(records)

    def conversation_context(
        self,
        workdir: str,
        session_id: str,
        limit: int = 12,
        *,
        exclude_latest_user: bool = False,
        max_chars: int = 24_000,
    ) -> list[str]:
        """提取最近的用户与 Agent 对话，过滤事件流和工具日志。

        管理者把这个结果放进 Context Package，使“继续修改”“按刚才的方案”等
        追问能够看到真正的对话上下文，而不会被大量工具事件挤掉。
        """

        # records 是会话原始记录；history_messages 只保留用户输入和最终回复。
        records = self.read_session(workdir, session_id)
        history_messages = self._history_messages(records)
        if exclude_latest_user and history_messages and history_messages[-1]["role"] == "user":
            # 当前用户输入已经单独放在 goal/current 中，recent 不再重复携带一份。
            history_messages = history_messages[:-1]
        # recent_messages 先按条数取最近部分，再按字符预算从后往前装入。
        recent_messages = history_messages[-max(limit, 1) :]
        # reversed_result 临时按“最新到最旧”顺序积累，结束时再反转回正常时间顺序。
        reversed_result: list[str] = []
        remaining = max(int(max_chars), 1000)
        for message in reversed(recent_messages):
            role = "用户" if message["role"] == "user" else "Agent"
            # content 单条最多保留 6000 字符，超长回复优先保留结尾的结果信息。
            content = str(message["content"])
            content = content if len(content) <= 6000 else "…" + content[-5999:]
            item = f"{role}：{content}"
            if len(item) > remaining:
                if not reversed_result:
                    reversed_result.append(item[-remaining:])
                break
            reversed_result.append(item)
            remaining -= len(item)
        return list(reversed(reversed_result))

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
        with self.lock:
            self._atomic_write(path, text.strip() + "\n")

    def update_global_memory(self, text: str) -> None:
        """覆盖写入全局长期记忆。"""

        with self.lock:
            self._atomic_write(self.global_path, text.strip() + "\n")

    def remember(self, workdir: str, text: str, *, global_scope: bool = False) -> str:
        """按用户明确指令追加一条长期记忆，不根据模型推断自动写入。"""

        # clean 合并空白并限制单条偏好长度，防止把整段任务误写入长期记忆。
        clean = re.sub(r"\s+", " ", text).strip()[:500]
        if not clean:
            raise ValueError("长期记忆内容不能为空")
        # path 根据用户是否明确要求全局生效，选择全局或当前项目长期记忆。
        path = self.global_path if global_scope else self.project_dir(workdir) / "project.md"
        with self.lock:
            # current 是写入前的 Markdown；默认占位“暂无”会在第一次记忆时移除。
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            current = current.replace("\n暂无。\n", "\n")
            # entry 带日期但不带微秒，便于用户追踪偏好何时写入。
            entry = f"- {utc_stamp()[:10]}：{clean}"
            # remembered_contents 忽略日期比较正文，防止同一偏好在不同日期重复累积。
            remembered_contents = {
                line.split("：", 1)[1].strip()
                for line in current.splitlines()
                if line.startswith("- ") and "：" in line
            }
            if clean not in remembered_contents:
                current = current.rstrip() + "\n\n" + entry + "\n"
                self._atomic_write(path, current)
        return "全局长期记忆" if global_scope else "项目长期记忆"

    def maybe_compress(self, workdir: str, session_id: str, ctx: int) -> None:
        """超过上下文 65% 时追加压缩视图，原始 Agent 日志保持不变。"""

        with self.lock:
            session_dir = self.session_dir(workdir, session_id)
            records = self._read_all_records(session_dir, include_summaries=False)
            max_chars = max(int(ctx * 4 * 0.65), 1000)
            if self._records_size(records) <= max_chars:
                return
            runtime_counts: dict[str, int] = {}
            runtime_latest: dict[str, str] = {}
            for rec in records:
                if self._critical(rec):
                    continue
                key = f"{rec.get('ag', '')}/{rec.get('tl', '')}/{rec.get('k', '')}"
                runtime_counts[key] = runtime_counts.get(key, 0) + 1
                runtime_latest[key] = str(rec.get("out", ""))[-240:]
            detail = [
                {"key": key, "count": runtime_counts[key], "latest": runtime_latest[key]}
                for key in sorted(runtime_counts)
            ]
            latest_cutoff = max((int(rec.get("id", 0)) for rec in records), default=0)
            previous = [
                rec
                for rec in self._read_all_records(session_dir)
                if rec.get("tl") == "memory" and rec.get("k") == "summary"
            ]
            if previous and int((previous[-1].get("m") or {}).get("cutoff", 0)) >= latest_cutoff:
                return
            summary = {
                "id": latest_cutoff + 1,
                "ts": utc_stamp(),
                "ag": "manager",
                "tl": "memory",
                "k": "summary",
                "out": "旧运行过程已折叠；原始记录仍保存在各 Agent 日志中。",
                "m": {"old_count": len(records), "cutoff": latest_cutoff, "runtime": detail},
            }
            path = session_dir / "agents" / "manager" / "sum.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.last_ids[session_dir] = summary["id"]

    def _read_project_workdir(self, project_dir: Path) -> str:
        """从项目 project.md 中读取真实项目路径。"""

        # meta_path 是新格式的项目定位元数据，长期记忆正文不再承担索引职责。
        meta_path = project_dir / "meta.json"
        if meta_path.exists():
            try:
                # data 是 meta.json 的结构化内容。
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                workdir = str(data.get("workdir") or "").strip()
                if workdir:
                    return workdir
            except (OSError, json.JSONDecodeError, TypeError):
                # 元数据损坏时继续尝试旧版 project.md，保证历史会话仍可恢复。
                pass

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

        records = self._read_all_records(path, include_summaries=False)
        if not records:
            return None
        # messages 是可恢复到对话窗口的用户消息和最终回复。
        messages = self._history_messages(records)
        # events 是整个会话产生的事件摘要，恢复会话时重新显示在右侧事件流。
        events = self._history_events(records)
        # title 是会话展示名；没有自定义名时用文件名也就是 session_id。
        title = self._session_title(records) or path.name
        # last 是最后一条记录，用于更新时间和中断判断。
        last = records[-1]
        return {
            "id": path.name,
            "title": title,
            "workdir": workdir,
            "updated_at": str(last.get("ts", "")),
            "interrupted": self._records_interrupted(records),
            "messages": messages,
            "events": events,
            **self._session_capabilities(records),
        }

    def _read_session_file(self, path: Path) -> list[dict[str, Any]]:
        """直接读取指定 jsonl 文件，不根据 workdir/session_id 重新计算路径。"""

        # records 保存解析后的记忆记录。
        records: list[dict[str, Any]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                # line 是 jsonl 的一行，空行忽略。
                line = line.strip()
                if line:
                    try:
                        # record 是当前行解析出的 JSON 对象；非对象记录没有可恢复价值。
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # 异常中断最常见的是最后一行只写了一半，跳过坏行以恢复其余记录。
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        return records

    def _read_all_records(
        self,
        session_dir: Path,
        *,
        include_summaries: bool = True,
    ) -> list[dict[str, Any]]:
        """合并会话内各独立日志，仅在读取视图时按全局 id 排序。"""

        if not session_dir.is_dir():
            return []
        paths = [session_dir / "chat.jsonl", session_dir / "tasks.jsonl"]
        paths.extend(session_dir.glob("agents/*/log.jsonl"))
        if include_summaries:
            paths.extend(session_dir.glob("agents/*/sum.jsonl"))
        records: list[dict[str, Any]] = []
        for path in paths:
            records.extend(self._read_session_file(path))
        records.sort(key=lambda rec: (int(rec.get("id", 0)), str(rec.get("ts", ""))))
        return records

    def _active_records(self, session_dir: Path) -> list[dict[str, Any]]:
        """返回压缩后的逻辑视图，原始文件不会被删除或重写。"""

        records = self._read_all_records(session_dir)
        summaries = [
            rec for rec in records if rec.get("tl") == "memory" and rec.get("k") == "summary"
        ]
        if not summaries:
            return records
        summary = summaries[-1]
        cutoff = int((summary.get("m") or {}).get("cutoff", 0))
        raw = [rec for rec in records if rec.get("k") != "summary"]
        critical = [rec for rec in raw if int(rec.get("id", 0)) <= cutoff and self._critical(rec)]
        recent = [rec for rec in raw if int(rec.get("id", 0)) > cutoff]
        if len(recent) < 20:
            recent_ids = {int(rec.get("id", 0)) for rec in raw[-(20 - len(recent)) :]}
            recent = [
                self._compact_record(rec)
                for rec in raw
                if int(rec.get("id", 0)) in recent_ids and rec not in critical
            ] + recent
        active = [summary] + critical[-16:] + recent
        unique: dict[int, dict[str, Any]] = {}
        for rec in active:
            unique[int(rec.get("id", 0))] = rec
        return [summary] + [
            rec for _, rec in sorted(unique.items()) if rec is not summary
        ]

    @staticmethod
    def _critical(rec: dict[str, Any]) -> bool:
        return bool(
            rec.get("tl") in {"session", "input", "final", "state", "run", "capability", "task"}
            or rec.get("k")
            in {"message", "result", "pending_plan", "plan_done", "handoff", "outcome", "needs_clarification"}
        )

    def append_task(
        self,
        workdir: str,
        session_id: str,
        agent: str,
        kind: str,
        text: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """写入只包含任务或结果的 Agent 交接记录。"""

        if kind not in {"handoff", "outcome", "needs_clarification"}:
            raise ValueError("非法任务交接类型")
        return self.append(workdir, session_id, agent, "task", kind, text, payload)

    def save_file(
        self,
        workdir: str,
        session_id: str,
        name: str,
        mime: str,
        content: bytes,
    ) -> dict[str, Any]:
        """保存图片附件，返回不含二进制内容的元数据。"""

        if not content or len(content) > 20 * 1024 * 1024:
            raise ValueError("单张图片必须在 1 字节到 20 MiB 之间")
        allowed = {
            "image/png": b"\x89PNG\r\n\x1a\n",
            "image/jpeg": b"\xff\xd8\xff",
            "image/gif": b"GIF8",
            "image/webp": b"RIFF",
        }
        if mime not in allowed or not content.startswith(allowed[mime]):
            raise ValueError("仅支持真实的 PNG、JPEG、WebP 或 GIF 图片")
        if mime == "image/webp" and content[8:12] != b"WEBP":
            raise ValueError("WebP 文件头无效")
        try:
            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                actual_mime = Image.MIME.get(image.format or "", "")
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise ValueError("图片内容损坏或无法解码") from exc
        if actual_mime != mime or width <= 0 or height <= 0:
            raise ValueError("图片真实格式或尺寸无效")
        if width * height > 100_000_000:
            raise ValueError("图片像素尺寸过大")
        digest = hashlib.sha256(content).hexdigest()
        ext = mimetypes.guess_extension(mime) or ".bin"
        if ext == ".jpe":
            ext = ".jpg"
        folder = self.session_dir(workdir, session_id) / "files"
        path = folder / f"{digest}{ext}"
        if not path.exists():
            path.write_bytes(content)
        index_path = folder / "index.json"
        index = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                index = {}
        info = {
            "id": digest,
            "name": Path(name).name[:160] or f"image{ext}",
            "mime": mime,
            "size": len(content),
            "width": width,
            "height": height,
            "file": path.name,
        }
        index[digest] = info
        self._atomic_write(index_path, json.dumps(index, ensure_ascii=False, indent=2) + "\n")
        return info

    def file_info(self, workdir: str, session_id: str, file_id: str) -> dict[str, Any]:
        """读取附件元数据并返回真实路径。"""

        if not re.fullmatch(r"[0-9a-f]{64}", file_id):
            raise ValueError("非法附件 id")
        folder = self.session_dir(workdir, session_id) / "files"
        index_path = folder / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(file_id)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        info = dict(index.get(file_id) or {})
        if not info:
            raise FileNotFoundError(file_id)
        path = (folder / str(info["file"])).resolve()
        if folder.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(file_id)
        info["path"] = path
        return info

    def delete_file(self, workdir: str, session_id: str, file_id: str) -> bool:
        """删除会话附件及索引项。"""

        try:
            info = self.file_info(workdir, session_id, file_id)
        except FileNotFoundError:
            return False
        folder = self.session_dir(workdir, session_id) / "files"
        Path(info["path"]).unlink(missing_ok=True)
        index_path = folder / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index.pop(file_id, None)
        self._atomic_write(index_path, json.dumps(index, ensure_ascii=False, indent=2) + "\n")
        return True

    def _history_messages(self, records: list[dict[str, Any]]) -> list[dict[str, str]]:
        """从完整会话记录中提取前端对话窗口需要恢复的消息。"""

        # messages 只包含用户输入和 agent 最终结果，不包含中间事件流。
        messages: list[dict[str, str]] = []
        for rec in records:
            # 初始输入和运行中指导都恢复为用户消息；指导不是新的 run/start。
            if (
                rec.get("ag") == "user"
                and rec.get("tl") in {"input", "guidance"}
                and rec.get("k") == "message"
            ):
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

    def _history_events(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从会话记录中恢复全部事件流摘要。"""

        # event_records 保留整个会话中 AgentGraph._emit 写入的事件，排除用户消息、最终回复和工具原始状态。
        event_records = [rec for rec in records if rec.get("tl") == "event"]
        events: list[dict[str, Any]] = []
        for event_id, rec in enumerate(event_records, start=1):
            # 旧记录兼容 usage 元数据，新记录恢复完整脱敏数据。
            meta = rec.get("m") if isinstance(rec.get("m"), dict) else {}
            events.append(
                {
                    "id": event_id,
                    "ts": str(meta.get("event_ts") or rec.get("ts", "")),
                    "agent": str(rec.get("ag", "manager")),
                    "kind": str(rec.get("k", "event")),
                    "msg": str(rec.get("out", "")),
                    "tokens": int(meta.get("tokens", 0) or 0),
                    "data": meta.get("data") if isinstance(meta.get("data"), dict) else (meta.get("usage") if isinstance(meta.get("usage"), dict) else {}),
                }
            )
        return events

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

    def _session_capabilities(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """从会话记录倒序提取最近一次能力选择。"""

        for rec in reversed(records):
            if rec.get("ag") != "manager" or rec.get("tl") != "capability" or rec.get("k") != "selection":
                continue
            # meta 只接受字典，损坏或旧格式记录按空选择处理。
            meta = rec.get("m") if isinstance(rec.get("m"), dict) else {}
            # skills/mcps 逐项转字符串并去重，避免旧数据把非字符串传给前端。
            skills_raw = meta.get("skills") if isinstance(meta.get("skills"), list) else []
            mcps_raw = meta.get("mcps") if isinstance(meta.get("mcps"), list) else []
            return {
                "skills": list(dict.fromkeys(str(item) for item in skills_raw if str(item))),
                "mcps": list(dict.fromkeys(str(item) for item in mcps_raw if str(item))),
                "container_shell": bool(meta.get("container_shell", True)),
            }
        return {"skills": [], "mcps": [], "container_shell": True}

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

    def _safe_session_id(self, session_id: str) -> str:
        """校验会话 id 只包含可安全用于文件名的短字符。"""

        # clean 是外部传入的原始会话 id 去除首尾空白后的值。
        clean = str(session_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", clean) or ".." in clean:
            raise ValueError("非法会话 id")
        return clean

    def _archive_legacy_sessions(self) -> None:
        """把旧版单文件会话移动到独立归档，绝不导入新版活动记忆。"""

        legacy_files = sorted((self.root / "projects").glob("*/sessions/*.jsonl"))
        if not legacy_files:
            return
        archive_root = self.root.parent / "legacy"
        archive_id = utc_stamp().replace(":", "").replace("-", "")
        destination_root = archive_root / archive_id
        manifest: list[dict[str, Any]] = []
        for source in legacy_files:
            relative = source.relative_to(self.root)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_bytes()
            manifest.append(
                {
                    "source": relative.as_posix(),
                    "target": destination.relative_to(archive_root).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                }
            )
            source.replace(destination)
        self._atomic_write(
            destination_root / "index.json",
            json.dumps(
                {"version": 1, "archived_at": utc_stamp(), "files": manifest},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    def _rebuild_session_index(self) -> None:
        """启动时扫描已有项目，建立会话到工作目录的内存索引。"""

        with self.lock:
            # projects_root 是所有项目记忆目录的父目录。
            projects_root = self.root / "projects"
            projects_root.mkdir(parents=True, exist_ok=True)
            self.session_index.clear()
            for project_dir in projects_root.glob("*"):
                if not project_dir.is_dir():
                    continue
                # workdir 来自 meta.json，旧数据则回退到 project.md。
                workdir = self._read_project_workdir(project_dir)
                if not workdir:
                    continue
                for session_path in (project_dir / "sessions").glob("*"):
                    if session_path.is_dir():
                        self.session_index[session_path.name] = workdir

    def _normalize_existing_timestamps(self) -> None:
        """把已有 JSONL 中的长 ISO 时间迁移为紧凑 UTC 时间。"""

        with self.lock:
            for path in (self.root / "projects").glob("*/sessions/**/*.jsonl"):
                if path.name == "index.json":
                    continue
                # records 是容错读取后的旧会话内容。
                records = self._read_session_file(path)
                changed = False
                for rec in records:
                    # raw 是旧 ts 字段；空值或已经以秒级 Z 结尾时无需处理。
                    raw = str(rec.get("ts") or "")
                    if not raw or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", raw):
                        continue
                    try:
                        # parsed 兼容 +00:00、微秒和 Z，并统一转换到 UTC。
                        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=UTC)
                        rec["ts"] = parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                        changed = True
                    except ValueError:
                        # 非标准用户自定义时间保持原样，避免迁移破坏信息。
                        continue
                if changed:
                    # lines 使用紧凑 JSON，进一步减少会话文件体积。
                    lines = [json.dumps(rec, ensure_ascii=False, separators=(",", ":")) for rec in records]
                    self._atomic_write(path, "\n".join(lines) + "\n")

    def _records_interrupted(self, records: list[dict[str, Any]]) -> bool:
        """根据已读取记录判断最近一轮运行是否没有正常收尾。"""

        if not records:
            return False
        # 新格式优先查找最近生命周期记录；会话重命名和普通事件不影响结果。
        for rec in reversed(records):
            if rec.get("ag") == "manager" and rec.get("tl") == "run":
                return rec.get("k") == "start"
        # 旧格式通过最后用户输入是否晚于最后最终回复来判断。
        last_user = -1
        last_result = -1
        for index, rec in enumerate(records):
            if rec.get("ag") == "user" and rec.get("tl") == "input" and rec.get("k") == "message":
                last_user = index
            if rec.get("ag") == "manager" and rec.get("tl") == "final" and rec.get("k") == "result":
                last_result = index
        if last_user >= 0:
            return last_user > last_result
        # 极早期旧版记录可能没有 user/input，只要存在非会话元数据且没有最终结果就视为中断。
        has_runtime_record = any(rec.get("tl") != "session" for rec in records)
        return has_runtime_record and last_result < 0

    def _compact_record(self, rec: dict[str, Any]) -> dict[str, Any]:
        """截断一条运行记录中的大文本和工具数据。"""

        # compact 是浅拷贝，避免压缩过程中修改调用方持有的原对象。
        compact = dict(rec)
        compact["out"] = str(compact.get("out", ""))[-500:]
        # meta 只保留 token 和简短状态，不重复保存完整命令输出。
        meta = compact.get("m") if isinstance(compact.get("m"), dict) else {}
        compact["m"] = {key: value for key, value in meta.items() if key in {"tokens", "status", "title"}}
        return compact

    def _records_size(self, records: list[dict[str, Any]]) -> int:
        """返回记录序列化后的字符数，用于上下文窗口近似计算。"""

        return sum(len(json.dumps(rec, ensure_ascii=False, separators=(",", ":"))) + 1 for rec in records)

    def _atomic_write(self, path: Path, content: str) -> None:
        """在目标文件同目录写临时文件，再原子替换正式文件。"""

        # parent 必须先存在，临时文件和目标文件位于同一文件系统才能原子替换。
        path.parent.mkdir(parents=True, exist_ok=True)
        # temp_path 带线程 id，避免同一进程的不同文件写入互相覆盖临时内容。
        temp_path = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
