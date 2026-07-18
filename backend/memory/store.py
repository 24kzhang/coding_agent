from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

    def session_path(self, workdir: str, session_id: str) -> Path:
        """返回指定会话的 jsonl 文件路径，并确保项目 memory 目录存在。"""

        # safe_id 拒绝路径分隔符和上级目录片段，防止接口参数逃出 sessions 目录。
        safe_id = self._safe_session_id(session_id)
        return self.project_dir(workdir) / "sessions" / f"{safe_id}.jsonl"

    def delete_session(self, workdir: str, session_id: str) -> bool:
        """删除某个会话的 memory 文件，不删除真实项目文件。"""

        # path 指向待删除的会话 jsonl；这里不调用 project_dir，避免删除不存在会话时创建目录。
        # safe_id 是通过白名单校验后的会话 id。
        safe_id = self._safe_session_id(session_id)
        path = self.project_path(workdir) / "sessions" / f"{safe_id}.jsonl"
        # 删除前确认路径一定在 memory/data/projects 内。
        self._ensure_under_projects(path)
        with self.lock:
            if not path.exists():
                return False
            path.unlink()
            self.last_ids.pop(path, None)
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
            session_ids = [session_path.stem for session_path in (path / "sessions").glob("*.jsonl")]
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
        path = self.project_path(workdir) / "sessions" / f"{safe_id}.jsonl"
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
        with self.lock:
            # path 是本次要追加的会话 jsonl 文件路径。
            path = self.session_path(workdir, session_id)
            # last_id 从缓存读取；第一次写已有会话时才扫描一次文件。
            last_id = self.last_ids.get(path)
            if last_id is None:
                # records 是当前文件中仍可解析的记录，损坏尾行不会阻断后续恢复。
                records = self._read_session_file(path)
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
            self.last_ids[path] = rec["id"]
            self.session_index[self._safe_session_id(session_id)] = str(Path(workdir).resolve())
            return rec

    def read_session(self, workdir: str, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """读取会话 jsonl；limit 为空读全部，否则只返回最后 limit 条。"""

        # path 是会话 jsonl 文件路径；调用 session_path 会确保目录存在。
        path = self.session_path(workdir, session_id)
        with self.lock:
            # records 由统一容错读取器解析，异常中断留下的半行不会破坏整个会话。
            records = self._read_session_file(path)
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
                session_file = project / "sessions" / f"{safe_id}.jsonl"
                if session_file.exists():
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
        """当会话记忆超过模型上下文 85% 时压缩运行日志。

        用户消息、最终回复、Plan 状态和会话元数据属于恢复所需的关键记录；工具
        事件属于可压缩运行日志。第一次压缩先汇总运行日志，仍超限时再成组丢弃最早
        的非最近对话，始终保留最近阶段。
        """

        with self.lock:
            # records 是压缩前的完整会话记录。
            records = self.read_session(workdir, session_id)
            # max_chars 按 4 字符约 1 token 估算，并使用配置窗口的 85%。
            max_chars = max(int(ctx * 4 * 0.85), 1000)
            if self._records_size(records) <= max_chars:
                return

            # critical 保存恢复对话、Plan 和会话生命周期所需的关键记录。
            critical: list[dict[str, Any]] = []
            # runtime_counts 统计被折叠的 agent/tool/kind 组合。
            runtime_counts: dict[str, int] = {}
            # runtime_latest 保存每个组合最近一次输出的极简摘要。
            runtime_latest: dict[str, str] = {}
            for rec in records:
                # is_critical 表示这条记录不能在第一阶段被丢弃。
                is_critical = (
                    rec.get("tl") in {"session", "input", "final", "state", "run"}
                    or rec.get("k") in {"message", "result", "pending_plan", "plan_done"}
                )
                if is_critical:
                    critical.append(dict(rec))
                    continue
                # key 用短字符串标识一类重复运行记录。
                key = f"{rec.get('ag', '')}/{rec.get('tl', '')}/{rec.get('k', '')}"
                runtime_counts[key] = runtime_counts.get(key, 0) + 1
                runtime_latest[key] = str(rec.get("out", ""))[-240:]

            # detail 把重复工具调用压缩为“次数 + 最新摘要”，不保留大段命令输出。
            detail = [
                {"key": key, "count": runtime_counts[key], "latest": runtime_latest[key]}
                for key in sorted(runtime_counts)
            ]
            # summary 是压缩阶段说明记录，本身也使用短时间戳。
            summary: dict[str, Any] = {
                "id": 0,
                "ts": utc_stamp(),
                "ag": "manager",
                "tl": "memory",
                "k": "summary",
                "out": "会话运行日志已压缩，保留对话、计划、生命周期和最近阶段。",
                "m": {"old_count": len(records), "runtime": detail},
            }
            # recent_ids 确保最后 20 条记录中的关键现场仍被保留。
            recent_ids = {int(rec.get("id", 0)) for rec in records[-20:]}
            # recent_runtime 保存最近阶段的非关键记录，并截断大字段。
            recent_runtime = [self._compact_record(rec) for rec in records if int(rec.get("id", 0)) in recent_ids and rec not in critical]
            compact = [summary] + critical + recent_runtime
            compact.sort(key=lambda rec: (int(rec.get("id", 0)) != 0, int(rec.get("id", 0))))

            # 仍超限时优先丢弃最早的普通对话，只保留最近 12 条关键记录和会话起点。
            if self._records_size(compact) > max_chars and len(critical) > 12:
                # session_meta 保存创建与重命名记录，避免压缩后会话名丢失。
                session_meta = [rec for rec in critical if rec.get("tl") == "session"][-2:]
                # newest_critical 是最近的任务对话与计划状态。
                newest_critical = critical[-12:]
                compact = [summary] + session_meta + newest_critical + recent_runtime[-8:]

            # 如果单条用户长文本仍导致超限，只截断最旧记录；最近 6 条保持完整。
            while self._records_size(compact) > max_chars and len(compact) > 8:
                # drop_index 从 summary 后开始删除，summary 和最近记录始终保留。
                drop_index = 1
                compact.pop(drop_index)

            # path 是原会话 JSONL，使用同目录临时文件原子替换，避免中途断电留下半文件。
            path = self.session_path(workdir, session_id)
            lines: list[str] = []
            for index, rec in enumerate(compact, start=1):
                # normalized 是副本，避免修改调用方仍持有的原始记录。
                normalized = dict(rec)
                normalized["id"] = index
                lines.append(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))
            self._atomic_write(path, "\n".join(lines) + "\n")
            self.last_ids[path] = len(lines)

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
            "interrupted": self._records_interrupted(records),
            "messages": messages,
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

    def _safe_session_id(self, session_id: str) -> str:
        """校验会话 id 只包含可安全用于文件名的短字符。"""

        # clean 是外部传入的原始会话 id 去除首尾空白后的值。
        clean = str(session_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", clean) or ".." in clean:
            raise ValueError("非法会话 id")
        return clean

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
                for session_path in (project_dir / "sessions").glob("*.jsonl"):
                    self.session_index[session_path.stem] = workdir

    def _normalize_existing_timestamps(self) -> None:
        """把已有 JSONL 中的长 ISO 时间迁移为紧凑 UTC 时间。"""

        with self.lock:
            for path in (self.root / "projects").glob("*/sessions/*.jsonl"):
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
