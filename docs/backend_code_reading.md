# Agent 后端代码逐步精读

这份文档只讲后端和 agent 编排代码，不讲前端实现。

它的目标不是告诉你“哪个文件实现了什么功能”，而是带你按代码实际执行顺序理解这个项目。你读完后应该能自己判断：

- 一个用户请求从哪里进入后端。
- 后端如何找到会话和项目目录。
- LangGraph 如何决定调用哪些智能体。
- 管理者为什么不会把所有任务都走流水线。
- Plan 模式为什么能接住上一轮选择题的回答。
- Coding 智能体如何通过 ReAct 真正写文件。
- 验证智能体如何选择测试命令。
- memory 为什么能恢复历史会话和识别中断。
- 以后要加功能时应该改哪一段代码。

建议你打开代码和本文档并排阅读。每读一小节，就去对应文件里找到原函数。

---

> 当前版本边界：这是单用户本地 Coding Agent，不实现多人队列或分布式并发。代码把复杂度集中在按需编排、上下文筛选、运行中断恢复、受控文件/命令工具和失败后验证修复。

## 0. 先建立后端代码地图

后端相关代码可以分成 7 块：

```text
api/
  schema.py              # 后端接口、LangGraph 结果、上下文包的数据结构

backend/
  main.py                # FastAPI 入口，接收请求，返回流式事件
  cli.py                 # 命令行入口，不走前端也能执行 agent
  agents/
    graph.py             # 核心：LangGraph 多智能体编排
    types.py             # LangGraph 共享状态 AgentState
    prompts.py           # 各智能体系统提示词
  memory/
    store.py             # 会话记忆、项目记忆、全局记忆
  tools/
    fs.py                # 受 workdir 限制的文件读写
    shell.py             # 受控命令执行
    git.py               # 非破坏性 Git 操作

llm/
  store.py               # 模型配置读写
  client.py              # OpenAI-compatible 模型调用

scripts/
  dev.py                 # 本地开发启动脚本
```

你先记住一个原则：

> `backend/main.py` 负责“请求怎么进来”，`backend/agents/graph.py` 负责“任务怎么被编排”，`backend/memory/store.py` 负责“状态怎么留下来”。

后面所有代码都围绕这三件事展开。

## 1. 从数据结构开始读：`api/schema.py`

先看数据结构，是因为后端、前端、LangGraph、memory 最终都要围绕这些对象传递信息。

### 1.1 `ModelConfig`：一个模型需要哪些字段

```python
class ModelConfig(BaseModel):
    id: str
    name: str
    base_url: str
    api_key: str
    model: str
    ctx: int = Field(default=128000, description="模型上下文窗口，单位为 token")
    enabled: bool = True
    timeout: int = 120
```

逐个字段理解：

| 字段 | 作用 |
| --- | --- |
| `id` | 本地唯一标识，其他地方不直接引用模型名，而是引用这个 id |
| `name` | 给用户看的名称 |
| `base_url` | 模型服务地址，后面 `LlmClient` 会自动拼接 chat completions 路径 |
| `api_key` | 密钥，当前项目按需求明文存储 |
| `model` | 供应商要求的模型名 |
| `ctx` | 上下文窗口，memory 压缩时会用它判断是否超过 85% |
| `enabled` | 没有指定模型时优先选择启用模型 |
| `timeout` | HTTP 请求超时时间 |

你现在要建立一个意识：模型配置不是散落在代码里，而是统一由 `ModelConfig -> ModelStore -> LlmClient` 这条链处理。

### 1.2 `AgentModelMap`：为什么每个智能体能切换不同模型

```python
class AgentModelMap(BaseModel):
    manager: str = "longcat"
    planner: str = "longcat"
    repo: str = "longcat"
    coder: str = "longcat"
    verifier: str = "longcat"
    doc: str = "longcat"
```

这里保存的是“智能体名 -> 模型 id”。

例如：

```text
manager -> longcat
coder   -> qwen-coder
doc     -> deepseek
```

真正使用时在 `AgentGraph._client()`：

```python
def _client(self, agent: str, state: AgentState) -> LlmClient:
    return LlmClient(self.model_store.for_agent(agent, state.get("model_id")))
```

也就是说：

1. 节点知道自己是谁，例如 `coder`。
2. `_client("coder", state)` 去 `ModelStore` 查 coder 绑定哪个模型。
3. 查到模型配置后创建 `LlmClient`。

### 1.3 `ChatRequest`：一次任务请求长什么样

```python
class ChatRequest(BaseModel):
    session_id: str
    text: str
    plan_mode: bool = False
    execute_plan: bool = False
    model_id: str | None = None
```

这是 `/api/chat/stream` 的请求体。它没有 `workdir` 字段，这一点很重要。

为什么不让前端每次都传 `workdir`？

因为后端会通过 `session_id` 反查会话所属项目目录：

```python
workdir = _session_workdir(req.session_id)
```

这样可以避免前端随便传一个路径让后端执行任务。会话创建时已经绑定了项目目录，后续任务只需要传会话 id。

### 1.4 `ContextPackage`：管理者给下游智能体的上下文包

```python
class ContextPackage(BaseModel):
    goal: str
    task_type: str
    workdir: str
    plan_mode: bool
    relevant_files: list[str] = Field(default_factory=list)
    project_memory: str = ""
    global_memory: str = ""
    constraints: list[str] = Field(default_factory=list)
    recent: list[str] = Field(default_factory=list)
```

这比“把所有历史对话丢给模型”要稳定。

字段含义：

| 字段 | 说明 |
| --- | --- |
| `goal` | 当前任务目标，可能是用户原始输入，也可能是合并 Plan 回答后的文本 |
| `task_type` | 管理者分类结果 |
| `workdir` | 当前项目目录 |
| `plan_mode` | 是否处于 Plan 流程 |
| `relevant_files` | 仓库读取后填入的相关文件 |
| `project_memory` | 当前项目长期记忆 |
| `global_memory` | 全局长期记忆 |
| `constraints` | 本轮任务必须遵守的约束 |
| `recent` | 最近几轮用户输入与 Agent 最终回复 |

你后面读 `manager()` 时会看到它怎么构造这个对象。

### 1.5 `TaskResult`：任务最终怎么返回给用户

```python
class TaskResult(BaseModel):
    ok: bool
    summary: str
    files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    plan_path: str | None = None
    doc_path: str | None = None
    tokens: int = 0
    duration_ms: int = 0
```

这就是最终结果。

所有任务最后都会在 `AgentGraph.final()` 里整理成这个结构：

```python
result = {
    "ok": ok,
    "summary": summary,
    "files": state.get("changes", []),
    "commands": state.get("commands", []),
    "tests": state.get("tests", []),
    "plan_path": (state.get("plan") or {}).get("path"),
    "doc_path": (state.get("repo") or {}).get("doc_path"),
    "tokens": int(state.get("tokens", 0)),
    "duration_ms": duration_ms,
}
```

`tokens` 用于展示本轮模型消耗，`duration_ms` 使用单调时钟计算任务耗时。如果以后还要增加结果字段，必须同步修改：

1. `api/schema.py` 的 `TaskResult`
2. `AgentGraph.final()`
3. 前端或 CLI 的展示逻辑
4. 对应测试

## 2. 模型配置怎么读写：`llm/store.py`

读模型层时先看 `ModelStore`，它只负责配置，不负责调用模型。

### 2.1 初始化时发生什么

```python
def __init__(self, path: Path | None = None):
    self.path = path or Path("memory/config/models.json")
    self.path.parent.mkdir(parents=True, exist_ok=True)
    if not self.path.exists():
        self._write({"models": [], "agent_models": AgentModelMap().model_dump()})
```

逐行理解：

1. 如果测试传入 `path`，就用测试路径。
2. 否则默认使用 `memory/config/models.json`。
3. 确保父目录存在。
4. 如果配置文件不存在，写入默认结构。

默认结构长这样：

```json
{
  "models": [],
  "agent_models": {
    "manager": "longcat",
    "planner": "longcat",
    "repo": "longcat",
    "coder": "longcat",
    "verifier": "longcat",
    "doc": "longcat"
  }
}
```

注意：这里不保证 `longcat` 模型一定存在。它只是默认映射。真正使用时，如果没有任何模型，`get()` 会抛出错误。

### 2.2 `all()`：读取模型列表

```python
def all(self) -> list[ModelConfig]:
    data = self._read()
    return [ModelConfig(**item) for item in data.get("models", [])]
```

这段代码做了两个事情：

1. 从 JSON 文件读出原始 dict。
2. 把每个 dict 转成 `ModelConfig`。

为什么要转成 `ModelConfig`？

因为 Pydantic 会帮你补默认值和校验字段。比如某个旧配置没有 `timeout`，转成 `ModelConfig` 后会默认变成 `120`。

### 2.3 `get()`：模型选择规则

```python
def get(self, model_id: str | None) -> ModelConfig:
    models = self.all()
    if not models:
        raise KeyError("还没有配置任何模型")
    if model_id:
        for model in models:
            if model.id == model_id:
                return model
        raise KeyError(f"模型不存在：{model_id}")
    for model in models:
        if model.enabled:
            return model
    return models[0]
```

按优先级理解：

1. 没配置任何模型，直接报错。
2. 如果传了 `model_id`，必须精确找到。
3. 如果没传 `model_id`，找第一个 `enabled=True` 的模型。
4. 如果都禁用，仍返回第一个，避免系统完全不可用。

### 2.4 `for_agent()`：每个智能体怎么找到自己的模型

```python
def for_agent(self, agent: str, override: str | None = None) -> ModelConfig:
    if override:
        return self.get(override)
    model_map = self.model_map()
    return self.get(getattr(model_map, agent, None))
```

这里有两个层级：

| 优先级 | 说明 |
| --- | --- |
| `override` | 本次请求临时指定模型，优先级最高 |
| `agent_models` | 每个智能体自己的模型映射 |

如果用户本次请求传了 `model_id`，所有智能体都会用这个模型。否则每个智能体按 `AgentModelMap` 找自己的模型。

## 3. 模型怎么被调用：`llm/client.py`

`ModelStore` 只管理配置。真正调用模型的是 `LlmClient`。

### 3.1 为什么 `_urls()` 要生成多个候选地址

用户可能填不同形式的 `base_url`：

```text
https://api.example.com/openai
https://api.example.com/v1
https://api.example.com/v1/chat/completions
```

所以 `_urls()` 会生成候选地址：

```python
if base.endswith("/chat/completions"):
    urls.append(base)
if base.endswith("/v1"):
    urls.append(f"{base}/chat/completions")
else:
    urls.append(f"{base}/v1/chat/completions")
    urls.append(f"{base}/chat/completions")
```

这样就不用把某个供应商的 URL 写死。

### 3.2 `chat()` 的执行过程

普通兼容模型走标准 JSON 响应；模型名包含 `longcat` 时走 SSE 流式响应。简化后的分支如下：

```python
use_stream = "longcat" in self.cfg.model.lower()
if use_stream:
    payload["stream"] = True

with httpx.Client(timeout=self.cfg.timeout) as client:
    for url in self._urls():
        if use_stream:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                return self._stream_text(resp, messages)
        resp = client.post(url, headers=headers, json=payload)
        data = resp.json()
        return self._content_text(data["choices"][0]["message"]["content"])
```

按执行顺序：

1. 构造 OpenAI-compatible 请求体。
2. 构造 Authorization header。
3. 依次尝试候选 URL。
4. 如果是 404 或 405，说明可能 URL 拼错，继续试下一个。
5. 如果是其他错误码，认为模型接口失败。
6. 普通响应解析 `choices[0].message.content`。
7. LongCat 响应逐行解析 `data:` 分片，跳过心跳、usage 和空 `choices`，拼接 `delta.content`。
8. 记录接口返回或估算的 token 用量。
9. 对流式总时长和正文长度设置硬上限后返回文本。

这里的设计重点是“兼容性”：不同供应商 URL 规则不同，代码尽量自动适配。

### 3.3 `chat_json()` 为什么很重要

很多智能体要求模型返回 JSON：

- 管理者分类要 JSON。
- Plan 问题要 JSON。
- Coding ReAct action 要 JSON。
- 文档生成要 JSON。

`chat_json()` 做了容错：

```python
text = self.chat(...).strip()
if text.startswith("```"):
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    text = "\n".join(lines).strip()
start = text.find("{")
end = text.rfind("}")
if start >= 0 and end >= start:
    text = text[start : end + 1]
return json.loads(text)
```

上面的片段只展示 JSON 提取主线。当前实现还会调用 `_parse_longcat_tool_calls()`：当 LongCat 返回 `<longcat_tool_call>` 标签时，把工具名和参数转换成统一 `actions`；`finish` 或 `done` 标签转换成统一完成状态。Coder 因此可以使用供应商稳定支持的原生协议，下游执行层不需要出现 LongCat 专用分支。

它能处理这类模型输出：

```text
下面是 JSON：

    {"task_type": "direct"}
```

但它不是万能的。如果模型输出的 JSON 本身语法错了，就会抛 `LlmError`。后面你会看到 `planner()`、`coder()`、`doc()` 对 `LlmError` 的不同处理。

## 4. 会话和记忆怎么保存：`backend/memory/store.py`

这个文件很关键。它决定了：

- 历史会话能不能恢复。
- Plan 下一轮能不能接住上一轮问题。
- 会话中断后能不能识别。
- 长上下文何时压缩。

### 4.1 memory 目录结构

```text
memory/data/
  global.md
  projects/
    <项目路径哈希>/
      meta.json
      project.md
      sessions/
        <会话id>.jsonl
```

每个项目目录不是用真实路径命名，而是用哈希：

```python
digest = hashlib.sha1(str(Path(workdir).resolve()).encode("utf-8")).hexdigest()[:12]
```

为什么？

因为真实路径可能有中文、空格、斜杠，直接做目录名不合适。用 hash 更稳。

### 4.2 `project_dir()`：第一次选择项目时发生什么

```python
def project_dir(self, workdir: str) -> Path:
    path = self.project_path(workdir)
    (path / "sessions").mkdir(parents=True, exist_ok=True)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        self._atomic_write(meta_path, json.dumps({"workdir": str(Path(workdir).resolve())}, ensure_ascii=False))
    project_md = path / "project.md"
    if not project_md.exists():
        self._atomic_write(project_md, "# 项目长期记忆\n\n暂无。\n")
    return path
```

逐步解释：

1. 根据 `workdir` 算出项目 memory 路径。
2. 创建 `sessions/` 目录。
3. 用原子写入创建 `meta.json`，其中只保存真实项目路径。
4. 如果没有 `project.md`，创建纯文本项目长期记忆。

为什么路径和长期记忆要分开？

因为项目路径属于索引元数据，不是用户偏好或长期知识。`list_history()` 扫描 memory 时从 `meta.json` 反查真实目录；旧项目仍兼容从 `project.md` 读取这一行：

```text
项目路径：`/Users/zhang/learning/coding_agent/kefu`
```

### 4.3 `append()`：所有状态都是一行一行追加进去的

```python
def append(...):
    with self.lock:
        path = self.session_path(workdir, session_id)
        last_id = self.last_ids.get(path)
        if last_id is None:
            records = self._read_session_file(path)
            last_id = max((int(rec.get("id", 0)) for rec in records), default=0)
        rec = {
            "id": last_id + 1,
            "ts": utc_stamp(),
            "ag": agent,
            "tl": tool,
            "k": kind,
            "out": out,
            "m": meta or {},
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh.flush()
        self.last_ids[path] = rec["id"]
    return rec
```

一条记录长这样：

```json
{
  "id": 12,
  "ts": "2026-07-03T10:00:00Z",
  "ag": "coder",
  "tl": "react",
  "k": "summary",
  "out": "变更文件：['index.html']",
  "m": {"commands": ["python -m py_compile app.py"]}
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `id` | 会话内递增编号 |
| `ts` | 精简到秒的 UTC 时间 |
| `ag` | 哪个智能体 |
| `tl` | 哪个工具或阶段 |
| `k` | 记录类型 |
| `out` | 文本输出 |
| `m` | 结构化元数据 |

为什么用 jsonl？

因为任务可能中途失败。jsonl 是追加式的，已经写进去的状态不会因为最后失败而丢掉。启动时会把旧版微秒级时间迁移为秒级 `Z` 格式，损坏的单行会被跳过，不阻断整个会话恢复。

### 4.4 历史会话为什么能恢复

`list_history()` 做了这件事：

1. 遍历 `memory/data/projects/*`。
2. 读取每个项目的真实 `workdir`。
3. 遍历该项目的 `sessions/*.jsonl`。
4. 把每个会话变成摘要。

对话窗口由 `_history_messages()` 恢复：

```python
if rec.get("ag") == "user" and rec.get("tl") == "input" and rec.get("k") == "message":
    messages.append({"role": "user", "content": rec["out"]})

if rec.get("ag") == "manager" and rec.get("tl") == "final" and rec.get("k") == "result":
    messages.append({"role": "agent", "content": self._format_result(rec)})
```

对话窗口只恢复两类消息：

- 用户输入
- agent 最终回复

中间事件不会混进对话框。右侧事件流由 `_history_events()` 单独恢复：它提取整个会话中全部 `tl=event` 的记录，保持 JSONL 原始顺序并重新生成连续事件编号。因此多轮任务恢复后仍能查看完整执行过程。

### 4.5 中断判断为什么这么写

```python
def interrupted(self, workdir: str, session_id: str) -> bool:
    records = self.read_session(workdir, session_id)
    return self._records_interrupted(records)

def _records_interrupted(self, records):
    for rec in reversed(records):
        if rec.get("ag") == "manager" and rec.get("tl") == "run":
            return rec.get("k") == "start"
    # 旧格式再比较最后用户输入与最后最终回复的位置。
    ...
```

HTTP 入口会在每次任务前写 `manager/run/start`，结束时写 `manager/run/done`，异常时写 `manager/run/error`。因此会话重命名、压缩摘要等后续管理记录不会误触发中断。

旧会话没有生命周期记录时，才回退检查 `AgentGraph.final()` 写入的：

```python
self.memory.append(..., "manager", "final", "result", summary, result)
```

关键顺序是：`backend/main.py` 必须在追加当前用户输入前调用 `interrupted()`，否则当前输入本身会被误认为上一轮残留。

### 4.6 memory 压缩怎么做

`maybe_compress()` 的核心逻辑：

```python
raw = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
if len(raw) / 4 <= ctx * 0.85:
    return
```

它用 `4 字符 ≈ 1 token` 粗估。如果超过模型上下文 85%，就压缩。

压缩分两阶段：先把连续工具事件极简化为阶段摘要、重复命令只保留最新结果，同时保留用户消息、最终回复、Plan 状态和运行生命周期；仍超过 85% 时，再按阶段成组丢弃最早记录。最终用临时文件原子替换原 jsonl。

`conversation_context()` 与压缩是两件事：前者每次只把用户消息和 Agent 最终回复组装给管理者，事件与工具日志不会成为聊天上下文；后者控制磁盘会话长期增长。

## 5. 后端请求入口：`backend/main.py`

现在开始看请求怎么进入后端。

### 5.1 后端启动时创建两个全局对象

```python
model_store = ModelStore()
memory_store = MemoryStore()
```

这意味着：

- 模型配置仓库在进程启动时创建。
- memory 仓库也在进程启动时创建。
- 所有请求共享这两个对象。

本地单用户工具这样可以接受。如果以后要做多用户服务，需要考虑隔离。

### 5.2 创建会话：`POST /api/sessions`

```python
def create_session(req: SessionCreate) -> SessionInfo:
    workdir = str(Path(req.workdir).expanduser().resolve())
    Path(workdir).mkdir(parents=True, exist_ok=True)
    session_id = new_session_id()
    title = " ".join((req.title or "").strip().split())[:60] or session_id
    memory_store.append(workdir, session_id, "manager", "session", "start", ...)
    return SessionInfo(...)
```

按执行顺序：

1. 把用户选择的目录转成绝对路径。
2. 如果目录不存在就创建。
3. 生成短会话 id。
4. 计算会话标题。
5. 写入一条 `manager/session/start` 记忆。
6. 返回 `SessionInfo`。

这一步之后，memory 中会出现：

```text
memory/data/projects/<项目哈希>/sessions/<会话id>.jsonl
```

### 5.3 核心入口：`POST /api/chat/stream`

这是 agent 任务真正开始的地方。

```python
def chat_stream(req: ChatRequest) -> StreamingResponse:
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
    workdir = _session_workdir(req.session_id)
    memory_store.append(workdir, req.session_id, "user", "input", "message", req.text)
```

前三行做了三件事：

1. 创建队列。
2. 通过会话 id 找到项目目录。
3. 先把用户输入写入 memory。

为什么先写用户输入？

因为即使后面模型报错、工具报错、进程中断，历史会话里也能看到用户发过什么。

### 5.4 为什么要用 `event_queue`

后端希望边执行边给前端返回事件。执行任务的代码在后台线程里跑：

```python
def worker() -> None:
    try:
        graph = AgentGraph(model_store, memory_store, emit=emit)
        result = graph.run(...)
        event_queue.put({"type": "result", "data": result.model_dump()})
    except Exception as exc:
        event_queue.put({"type": "error", "data": {"msg": str(exc)}})
    finally:
        event_queue.put(None)
```

而 HTTP 响应用生成器持续读取队列：

```python
def stream() -> Any:
    while True:
        item = event_queue.get()
        if item is None:
            break
        yield json.dumps(item, ensure_ascii=False) + "\n"
```

所以完整关系是：

```text
AgentGraph 执行线程 -> event_queue -> StreamingResponse -> 前端事件流
```

### 5.5 `emit()` 是怎么把事件送出来的

在 `chat_stream()` 里：

```python
def emit(event: Any) -> None:
    event_queue.put({"type": "event", "data": event.model_dump()})
```

然后创建图时传进去：

```python
graph = AgentGraph(model_store, memory_store, emit=emit)
```

后面 `AgentGraph._emit()` 会调用这个回调：

```python
if self.emit_cb:
    self.emit_cb(event)
```

这样 LangGraph 内部节点就不需要知道 HTTP 流式响应怎么实现，它只负责发事件。

## 6. 命令行入口：`backend/cli.py`

CLI 是另一条入口，不经过前端和 FastAPI。

核心逻辑：

```python
workdir = str(Path(args.workdir).expanduser().resolve())
Path(workdir).mkdir(parents=True, exist_ok=True)
session_id = new_session_id()
memory = MemoryStore()
memory.append(workdir, session_id, "manager", "cli", "start", "命令行会话开始")

graph = AgentGraph(ModelStore(), memory, emit=emit)
result = graph.run(...)
```

你会发现它和 `chat_stream()` 做的事情很像：

| Web 入口 | CLI 入口 |
| --- | --- |
| 从请求里拿 `session_id` | 自己生成 `session_id` |
| 通过 session 反查 `workdir` | 从命令行参数拿 `workdir` |
| 把事件放入队列 | 把事件直接 print |
| 返回 NDJSON | 最后 print JSON |

这说明项目核心不是 FastAPI，而是 `AgentGraph`。FastAPI 和 CLI 都只是壳。

## 7. LangGraph 状态：`backend/agents/types.py`

`AgentState` 是 LangGraph 节点之间传递的共享字典。

你可以把它理解成“任务执行期间的背包”。每个节点从背包里拿东西，也往背包里放东西。

关键字段：

| 字段 | 谁写入 | 谁读取 | 作用 |
| --- | --- | --- | --- |
| `session_id` | `run()` | 所有节点 | 定位 memory |
| `workdir` | `run()` | repo/coder/verifier/doc/tools | 项目目录 |
| `text` | `run()` / manager | 所有模型节点 | 当前任务文本 |
| `route` | manager / planner | 路由函数 | 决定下一跳 |
| `after_repo` | manager / planner | `route_after_repo()` | repo 后去哪 |
| `after_verify` | manager / planner | `route_after_verifier()` | verifier 后去哪 |
| `context` | manager | planner/answer/coder/doc | Context Package |
| `repo` | repo | answer/coder/doc/final | 仓库摘要 |
| `changes` | coder/doc | final | 变更文件 |
| `tests` | verifier | coder/doc/final | 测试结果 |
| `tests_ok` | verifier | route_after_verifier/final | 是否通过 |
| `retry` | coder | verifier 路由 | 修复次数 |
| `result` | final | `run()` | 最终结果 |

读 `graph.py` 时，如果你看到某个节点返回：

```python
return {**state, "repo": repo, "context": ctx}
```

意思是：复制原状态，再覆盖 `repo` 和 `context` 两个字段，传给下一个节点。

## 8. 编排核心第一步：`AgentGraph.__init__()` 和 `run()`

### 8.1 初始化

```python
def __init__(self, model_store, memory, emit=None):
    self.model_store = model_store
    self.memory = memory
    self.emit_cb = emit
    self.event_id = 0
    self.graph = self._build()
```

每次任务都会创建一个新的 `AgentGraph` 实例。

为什么每次任务都新建？

因为这些状态不应该串到别的任务：

- `event_id`
- `emit_cb`
- LangGraph 执行状态
- token 统计

### 8.2 `run()` 做了什么

`run()` 是执行入口：

```python
state: AgentState = {
    "session_id": session_id,
    "workdir": workdir,
    "text": text,
    ...
}
final = self.graph.invoke(state)
result = final.get("result") or {...}
return TaskResult(**result)
```

它只做三件事：

1. 初始化 `AgentState`。
2. 调用编译后的 LangGraph。
3. 从最终状态里拿 `result`。

真正的智能体逻辑不在 `run()`，而在各个节点函数中。

## 9. 图是怎么搭起来的：`AgentGraph._build()`

先看节点：

```python
graph.add_node("manager", self.manager)
graph.add_node("planner", self.planner)
graph.add_node("repo", self.repo)
graph.add_node("answer", self.answer)
graph.add_node("coder", self.coder)
graph.add_node("verifier", self.verifier)
graph.add_node("doc", self.doc)
graph.add_node("final", self.final)
```

每个节点就是一个 Python 函数。函数输入是 `state`，输出也是 `state`。

再看边：

```python
graph.add_edge(START, "manager")
```

所有任务都从 `manager` 开始。

接着是条件边：

```python
graph.add_conditional_edges(
    "manager",
    self.route_after_manager,
    {"planner": "planner", "repo": "repo", "answer": "answer", "final": "final"},
)
```

意思是：

1. manager 执行完。
2. 调用 `route_after_manager(state)`。
3. 如果返回 `"repo"`，就去 repo。
4. 如果返回 `"final"`，就结束。

这就是“按需调用智能体”的关键。

完整路由可以记成：

```text
START
  -> manager
      -> planner -> final 或 repo
      -> repo -> coder/doc/answer/final
      -> answer -> final
      -> final
  coder -> verifier
  verifier -> coder/doc/final
  doc -> final
```

注意：并不是所有任务都会经过所有节点。

## 10. 管理者智能体：`manager()`

`manager()` 是最重要的函数之一。它决定后续到底要不要读仓库、要不要写代码、要不要生成文档。

### 10.1 先取出会话和 Plan 状态

```python
workdir = state["workdir"]
session_id = state["session_id"]
interrupted = self.memory.interrupted(workdir, session_id)

pending_plan = self._latest_pending_plan(workdir, session_id)
saved_plan = self._latest_saved_plan(workdir, session_id)
```

这里读取三种状态：

| 变量 | 含义 |
| --- | --- |
| `interrupted` | 上次会话是否可能中断 |
| `pending_plan` | 上一轮 Plan 是否提出了问题但用户还没回答完 |
| `saved_plan` | 是否已有生成好的计划等待执行 |

### 10.2 第一种特殊情况：用户取消 Plan

```python
if pending_plan and self._is_plan_cancel(state["text"]):
    self.memory.append(..., "plan_cancelled", ...)
    classification = {
        "task_type": "direct",
        ...
        "direct_reply": "已取消上一轮 Plan 流程。你可以重新描述新的需求。",
    }
```

如果当前有未完成 Plan，并且用户说“取消计划”，系统会：

1. 写入 `plan_cancelled`。
2. 本轮分类为 `direct`。
3. 直接回复用户。

写 `plan_cancelled` 的目的很重要：下一轮 `_latest_pending_plan()` 倒序查找时，看到 `plan_cancelled` 就不会再拿旧问题当 pending。

### 10.3 第二种特殊情况：用户在回答 Plan 问题

```python
elif pending_plan and self._should_treat_as_plan_reply(...):
    reply = self._build_plan_reply(state["text"], pending_plan)
    state = {
        **state,
        "text": self._compose_pending_plan_text(pending_plan, reply),
        "plan_mode": True,
        "pending_plan": pending_plan,
        "plan_answers": reply["answers"],
    }
    classification = {"task_type": "plan_gen", ...}
```

这是解决 “1A, 2A, 3A 没上下文” 的关键。

用户本轮输入可能只是：

```text
1A, 2A, 3A
```

如果直接把这句话交给模型，模型当然不知道原始需求是什么。

所以代码会构造一个新的 `text`：

```text
原始需求：
创建一个本地多智能体编程系统

上一轮 Plan 问题与用户回答：
1. 问题：前端技术栈选择？
   用户回答：React + TypeScript + Vite
2. 问题：后端框架选择？
   用户回答：FastAPI

用户原始回复：1A, 2A

请基于原始需求和用户回答继续 Plan 流程；信息足够时生成可执行计划。
```

这样 Plan 模型就有上下文了。

### 10.4 第三种特殊情况：用户确认执行已保存计划

```python
elif saved_plan and (state.get("execute_plan") or self._is_execute_plan_text(state["text"])):
    state = {
        **state,
        "text": self._compose_saved_plan_text(saved_plan, state["text"]),
        "plan": saved_plan,
    }
    classification = {
        "task_type": "code_gen",
        "need_repo": True,
        "need_code": True,
        "need_doc": True,
    }
```

如果用户输入：

```text
执行计划
```

系统不会把这当普通聊天。它会找到最近的 `plan_done`，把计划正文塞回 `state["text"]`，然后分类为代码生成任务。

### 10.5 普通情况：调用 `_classify()`

如果没有 Plan 特殊状态，就走：

```python
classification = self._classify(state)
```

后面会单独讲 `_classify()`。

### 10.6 构造 Context Package

分类完成后：

```python
ctx = ContextPackage(
    goal=state["text"],
    task_type=task_type,
    workdir=workdir,
    plan_mode=bool(state.get("plan_mode")),
    project_memory=self._trim(self.memory.project_memory(workdir), 5000),
    global_memory=self._trim(self.memory.global_memory(), 3000),
    constraints=[...],
    recent=self.memory.conversation_context(
        workdir,
        session_id,
        limit=12,
        exclude_latest_user=True,
    ),
)
```

这里你要看懂三个设计点：

1. 下游智能体拿到的是 `ContextPackage`，不是完整 jsonl。
2. 项目记忆和全局记忆都会截断，避免太长。
3. 最近上下文只保留用户消息和 Agent 最终回复，不把工具事件当成聊天内容。
4. 当前输入已经单独放在 `goal`，所以 `exclude_latest_user=True` 防止重复发送。

如果会话可能中断：

```python
if state.get("resuming"):
    ctx.recent.append("检测到上次任务可能异常中断，请先以当前磁盘和 Git 状态为准。")
```

### 10.7 manager 返回的新 state

```python
return {
    **state,
    "task_type": task_type,
    "route": route,
    "after_repo": after_repo,
    "after_verify": after_verify,
    "need_doc": after_verify == "doc" or after_repo == "doc",
    "context": ctx,
    "final": classification.get("direct_reply") or classification.get("reason", ""),
}
```

这里写入的字段会影响后续路由。

尤其是：

- `route`：manager 后去哪。
- `after_repo`：repo 后去哪。
- `after_verify`：verifier 后去哪。

## 11. 分类逻辑：`_classify()`

这个函数决定“输入是什么任务”。

### 11.1 Plan 模式优先

```python
if state.get("plan_mode"):
    return {"task_type": "plan_gen", ...}
```

如果前端开了 Plan 模式，直接进入计划生成，不靠自然语言猜。

### 11.2 普通对话直接回复

```python
direct_reply = self._direct_reply(text)
if direct_reply:
    return {
        "task_type": "direct",
        "need_repo": False,
        "need_code": False,
        "need_doc": False,
        "direct_reply": direct_reply,
    }
```

`_direct_reply()` 处理：

- 你好
- 在吗
- 谢谢
- 你是谁
- 你能做什么

这样普通聊天不会读仓库，也不会写 README。

### 11.3 代码生成任务

```python
code_intent = any(word in lower for word in ["创建", "实现", ...])
product_intent = any(word in lower for word in ["系统", "应用", ...])
doc_intent = any(word in lower for word in ["文档", "readme", ...])

if code_intent and product_intent:
    return {
        "task_type": "code_gen",
        "need_repo": True,
        "need_code": True,
        "need_doc": doc_intent,
    }
```

这里用两个条件组合：

- 有“创建/实现/开发”意图。
- 目标是“系统/应用/页面/接口/功能”等交付物。

这样“创建一个手机销售客服系统”会被识别为 `code_gen`。

### 11.4 文档任务、解释任务、修改任务

```python
if any(word in lower for word in ["解释", "为什么", "报错", ...]):
    return {"task_type": "code_explain", ...}

if doc_intent:
    return {"task_type": "doc_gen", ...}

if any(word in lower for word in ["修改", "修复", "bug", "重构", "适配"]):
    return {"task_type": "code_mod", ...}
```

顺序很重要。

当前顺序是：

1. Plan
2. direct
3. code_gen
4. code_explain
5. doc_gen
6. code_mod
7. 模型兜底

如果以后分类不准，先看顺序是否导致某类任务提前命中。

### 11.5 模型兜底

规则没覆盖到时：

```python
client = self._client("manager", state)
data = client.chat_json([...])
return data
```

如果模型失败：

```python
return {"task_type": "general_answer", ...}
```

注意这个兜底不会读仓库、不会写代码。这是保守设计。

## 12. 路由转换：`_flow_for()`

分类结果只是“任务是什么”。真正决定图怎么走的是 `_flow_for()`。

```python
if task_type == "direct":
    return "final", "final", "final"
if task_type == "general_answer":
    return "answer", "final", "final"
if task_type == "doc_gen":
    return "repo", "doc", "final"
if task_type == "code_explain":
    return ("repo", "answer", "final")
if task_type in {"code_gen", "code_mod"}:
    after_verify = "doc" if need_doc else "final"
    return "repo", "coder", after_verify
```

三个返回值分别是：

| 返回值 | 存到 state | 含义 |
| --- | --- | --- |
| 第 1 个 | `route` | manager 后去哪 |
| 第 2 个 | `after_repo` | repo 后去哪 |
| 第 3 个 | `after_verify` | verifier 后去哪 |

举例：

```python
return "repo", "coder", "doc"
```

表示：

```text
manager -> repo -> coder -> verifier -> doc -> final
```

这就是避免流水线的关键。不同任务返回不同路径。

## 13. Plan 智能体：`planner()`

Plan 节点只做两件事：

1. 信息不足时生成选择题。
2. 信息足够时生成计划 Markdown。

### 13.1 调用模型

```python
client = self._client("planner", state)
messages = [
    {"role": "system", "content": PLANNER_PROMPT},
    {"role": "user", "content": self._ctx_text(state)},
]
data = client.chat_json(messages)
```

模型必须返回 JSON：

```json
{"status": "questions", "questions": [...]}
```

或：

```json
{"status": "plan", "markdown": "..."}
```

### 13.2 模型异常时怎么处理

```python
except LlmError as exc:
    if state.get("pending_plan"):
        data = {"status": "plan", "markdown": self._fallback_plan_markdown(state)}
    else:
        data = {"status": "questions", "questions": [...]}
```

如果第一轮 Plan 模型异常，就给一个默认问题。

如果已经在回答上一轮 Plan 问题，就生成默认计划。

这样 Plan 流程不会因为模型 JSON 格式错而彻底卡住。

### 13.3 status = questions

```python
data = {**data, "questions": self._normalize_plan_questions(data.get("questions"))}
summary = self._format_plan_questions(data.get("questions", []))
self.memory.append(..., "pending_plan", ..., {"questions": ...})
return {**state, "plan": data, "route": "final", "final": summary}
```

流程：

1. 清洗问题。
2. 格式化成用户能看的文本。
3. 保存 `pending_plan`。
4. 本轮进入 final，等待用户回答。

保存 `pending_plan` 是下一轮能识别 `1A, 2A` 的关键。

### 13.4 status = plan

```python
md = data.get("markdown") or self._fallback_plan_markdown(state)
plan_dir = Path(state["workdir"]) / "docs" / "plans"
path = plan_dir / f"{state['session_id']}.md"
path.write_text(md.strip() + "\n", encoding="utf-8")
self.memory.append(..., "plan_done", ..., {"path": str(path), "markdown": md})
```

计划文件写到用户项目内：

```text
<项目目录>/docs/plans/<会话id>.md
```

然后：

```python
route = "repo" if state.get("execute_plan") else "final"
```

通常不会立刻执行，而是等用户确认。

## 14. 仓库读取智能体：`repo()`

`repo()` 是下游智能体理解项目的入口。

### 14.1 列出文件

```python
fs = FsTool(state["workdir"])
files = fs.list()
```

`FsTool.list()` 会跳过：

- `.git`
- `__pycache__`
- `node_modules`
- `.venv`
- `dist`、`build`、覆盖率目录和常见缓存

并最多返回 1200 个文件。列表是代码索引，不会直接全部送给模型。

### 14.2 计算任务相关文件

```python
candidates = self._repo_candidates(fs, files, state["text"])
```

`_repo_candidates()` 综合以下信号排序：

- `README.md`、`pyproject.toml`、`package.json` 等项目入口。
- 用户需求中出现的文件名、目录名和英文/中文关键词。
- 源码扩展名与常见入口名称。
- 敏感配置拦截结果。

### 14.3 在总预算内读取片段

```python
budget = 60_000
for rel in candidates:
    if budget <= 0:
        break
    chunk = fs.read(rel, min(7000, budget))
    snippets[rel] = chunk
    budget -= len(chunk)
```

这里按相关性读取，而不是按目录顺序碰运气。

原因：

- 避免上下文爆炸。
- 给 Coding/Answer/Doc 一个足够的项目摘要。
- 真正需要更多文件时，Coding 可以用 `search_files` 和分段 `read_file` 继续读。

### 14.4 识别技术栈

```python
stack = self._detect_stack(files)
repo = {"files": files, "snippets": snippets, "stack": stack, "empty": len(files) == 0}
```

`repo` 会被放入 state，后续节点可以读取。

## 15. 答疑智能体：`answer()`

`answer()` 用于解释和答疑，不写文件。

```python
messages = [
    {"role": "system", "content": ANSWER_PROMPT},
    {
        "role": "user",
        "content": self._ctx_text(state)
        + "\n\n仓库摘要：\n"
        + json.dumps(state.get("repo", {}), ensure_ascii=False)[:16000],
    },
]
text = client.chat(messages, temperature=0.2).strip()
```

这里输入包含：

- Context Package。
- 仓库摘要。

它不会调用工具，不会写磁盘。最后：

```python
return {**state, "final": text}
```

然后进入 `final()`。

## 16. Coding 智能体：`coder()`

这是写代码的核心。

### 16.1 初始化工具和状态

```python
fs = FsTool(state["workdir"])
shell = ShellTool(state["workdir"])
git = GitTool(state["workdir"])
client = self._client("coder", state)
observations: list[str] = []
changes = list(state.get("changes", []))
commands = list(state.get("commands", []))
```

含义：

| 变量 | 作用 |
| --- | --- |
| `fs` | 读写项目内文件 |
| `shell` | 在项目目录内执行命令 |
| `git` | 查看 Git 状态 |
| `client` | coder 模型 |
| `observations` | 最近的精简工具结果，反馈给下一轮模型 |
| `file_observations` | 最近读取且仍有效的当前文件快照 |
| `seen_actions` | 已执行或已拒绝的动作签名，用于阻止原地重复 |
| `progress_items` | 本轮已经产生的有效进展账本 |
| `read_paths` | 当前已经读取、允许精确修改的文件 |
| `unchecked_rewrites` | 已整体重写但尚未通过命令验证的文件 |
| `changes` | 已修改文件 |
| `commands` | 已执行命令 |

如果上一轮测试失败，`_test_brief()` 会把失败压缩成可执行清单，`_repair_paths()` 从错误和变更中选择最多六个相关文件重新读取。这样 Coder 根据当前磁盘修复，不会携带完整 traceback，也不会用第一次读取的旧代码覆盖已经正确的修改。

### 16.2 ReAct 循环

```python
for step in range(1, 17):
    messages = [...]
    data = client.chat_json(messages, plain_text=is_longcat)
    actions = self._normalize_actions(data.get("actions"))
    for action in actions:
        obs = self._do_action(action, fs, shell, git)
        observations.append(obs["text"])
```

最多 16 轮。每轮做：

1. 把上下文、仓库摘要、已有观察发给模型。
2. 普通模型返回 JSON；LongCat 返回原生工具标签，由模型封装层转换成同一结构。
3. 读取 `thought` 和 `actions`。
4. 执行每个 action。
5. 把工具结果追加到 observations。
6. 如果工具动作失败，把失败写回 observations，不能接受本轮 `done`。
7. 只有模型明确完成且本批动作全部成功，才设置 `coding_ok=True`。

连续三次拿不到可执行结构，或达到最大步数仍未完成时，`coding_ok=False`。每轮最多执行三个紧密相关动作；重复动作会被拒绝，写入后必须运行命令验证，临近上限时会强制收尾。这个字段会进入 verifier 和 final，避免“模型没完成但空测试把任务判成功”。

### 16.3 模型必须返回什么

由 `CODER_PROMPT` 约束：

```json
{
  "thought": "本轮判断，中文",
  "actions": [
    {"tool": "search_files", "query": "目标符号"},
    {"tool": "read_file", "path": "相对路径", "start": 0, "max_chars": 24000},
    {"tool": "replace_file", "path": "相对路径", "old": "旧文本", "new": "新文本", "expected": 1},
    {"tool": "replace_block", "path": "相对路径", "start_marker": "唯一开始标记", "end_marker": "结束边界或空字符串", "content": "新代码块"},
    {"tool": "run_command", "cmd": "命令"}
  ],
  "done": false,
  "summary": "完成时的中文摘要"
}
```

这就是 ReAct：

- 模型先思考。
- 再决定工具动作。
- 工具返回观察。
- 再进入下一轮。

单个当前文件快照最多 24,000 字符，全部快照最多 48,000 字符，单轮模型输出最多 6,000 token。工具 observation 只保留成功状态、目标、退出码和首尾关键输出；写入成功后由编排器从磁盘自动刷新快照，模型不需要再读一次。超时后还会丢弃任务未点名的快照，因此 16 步并不等于把 16 轮完整日志全部重复塞给模型。

### 16.4 `_do_action()`：工具动作怎么落地

```python
if tool == "write_file":
    rel = fs.write(...)
    return {"ok": True, "text": f"写入文件：{rel}", "file": rel}
```

支持的工具：

| 工具名 | 实际调用 |
| --- | --- |
| `write_file` | `FsTool.write()` |
| `append_file` | `FsTool.append()` |
| `replace_file` | `FsTool.replace()`，要求匹配次数符合 `expected` |
| `replace_block` | `FsTool.replace_block()`，要求开始和结束边界唯一；空结束边界表示到 EOF |
| `read_file` | `FsTool.read()`，支持 `start` 和 `max_chars` |
| `search_files` | `FsTool.search()`，返回文件、行号和片段 |
| `list_files` | `FsTool.list()` |
| `run_command` | `ShellTool.run()` |
| `git_status` | `GitTool.status()` |
| `git_diff` | `GitTool.diff()` |

如果工具执行成功并返回 `file`，coder 会把它加入 `changes`：

```python
if obs.get("file"):
    changes.append(str(obs["file"]))
```

如果返回 `cmd`，加入 `commands`：

```python
if obs.get("cmd"):
    commands.append(str(obs["cmd"]))
```

最后去重：

```python
unique_changes = sorted(dict.fromkeys(changes))
```

## 17. 工具层：`backend/tools/`

### 17.1 `FsTool`：文件工具的安全边界

核心是 `safe()`：

```python
path = (self.root / rel).resolve()
if self.root != path and self.root not in path.parents:
    raise ValueError(...)
return path
```

这阻止模型写出项目目录。

例如当前 `workdir=/tmp/proj`：

| 输入路径 | 结果 |
| --- | --- |
| `index.html` | 允许 |
| `src/app.py` | 允许 |
| `../secret.txt` | 拒绝 |
| `/etc/passwd` | 拒绝 |

`FsTool` 还承担三类工程边界：

- `list()` 剪枝依赖、缓存和构建目录，避免扫描成本失控。
- `safe()` 额外拒绝 `.env`、密钥、凭据和模型配置等敏感文件。
- `write()`、`replace()` 使用同目录临时文件原子替换；`replace()` 会校验旧文本命中次数，防止模型改错位置。

### 17.2 `ShellTool`：命令执行的安全边界

危险程序和危险 Git 子命令分别维护：

```python
dangerous_programs = {
    "rm",
    "rmdir",
    "del",
    "erase",
    "format",
    "mkfs",
    "shutdown",
    "reboot",
}
dangerous_git = {"checkout", "clean", "reset", "restore", "rm"}
```

执行前先用 `shlex.split()` 得到参数列表，再以 `shell=False` 启动进程：

```python
argv = shlex.split(clean, posix=os.name != "nt")
proc = subprocess.run(argv, cwd=self.root, shell=False, ...)
```

当前版本会直接拒绝复合 shell、重定向、命令替换、项目目录外参数、直接或经 `uv run` 转交的解释器内联代码。它不是完整容器隔离，但比字符串前缀黑名单更难绕过。

还有三个容易忽略的项目边界：`uv init` 自动补 `--no-workspace`，外部 `VIRTUAL_ENV` 不会传入用户项目；所选目录没有独立 `.git` 时，除 `git init` 外的 Git 命令都会被拒绝，避免 Git 向上读取父仓库。

### 17.3 `GitTool`

只做非破坏性操作：

- `init()`
- `status()`
- `diff()`

当前 Coding prompt 暴露 `git_status` 和只读 `git_diff`；破坏性 Git 操作仍不开放。

## 18. 验证智能体：`verifier()`

验证智能体根据项目文件选择命令。

### 18.1 初始化

```python
files = fs.list()
commands: list[str] = []
tests: list[dict[str, Any]] = []
ok = bool(state.get("coding_ok", True))
```

| 变量 | 含义 |
| --- | --- |
| `files` | 当前项目文件列表 |
| `commands` | 待执行测试命令 |
| `tests` | 测试结果 |
| `ok` | 整体验证状态，初始值继承 Coding 是否真正完成 |

### 18.2 静态 Web 检查

```python
static_check = self._static_web_check(fs, files)
if static_check:
    tests.append(static_check)
    ok = ok and bool(static_check.get("ok"))
```

`_static_web_check()` 会检查：

- JS 里 `getElementById("xxx")` 的 id 是否存在。
- HTML 内联事件 `onclick="foo()"` 调用的函数是否在 JS 中定义。
- HTML/JS 使用的业务 CSS class 是否能在样式表中找到；状态类和动态类按白名单处理。

这不是浏览器测试，但能抓住很多静态页面常见错误。

无 `package.json` 的静态 JavaScript 项目还会自动选择一个常见入口执行 `node --check`。语义审查在 60,000 字符总预算内让单个普通源码最多占 48,000 字符；只有真的超出预算时才追加“审查上下文截断”标记，模型不能把这个标记误判成磁盘文件损坏。`verify` 只读任务无论成功或失败都直接进入 Final，不会回流 Coding。

### 18.3 Python 项目

```python
if python_files:
    commands.append("uv run python -m compileall -q <源码目录>")
if has_pytest:
    commands.append("uv run pytest -q")
```

Python 项目先对识别出的源码根目录做 `compileall`。只有仓库确实存在 `tests/` 或 `test_*.py` 时才跑 pytest，避免 pytest 的“未收集到测试”被误判成代码失败。

### 18.4 Node 项目和静态 Web

```python
package_data = json.loads(fs.read("package.json", 80_000))
scripts = package_data.get("scripts", {})
for script in ["lint", "test", "typecheck", "build"]:
    if script in scripts:
        commands.append(f"npm run {script}")
```

Node 验证不会猜测 Vitest、Jest 或其他框架参数，而是结构化读取项目已有 scripts。静态 Web 仍执行 DOM id 和内联函数接线检查，但不会为了“看起来验证过”而伪造一个服务器命令。

### 18.5 失败后怎么回到 coder

路由在这里：

```python
def route_after_verifier(self, state):
    if not state.get("tests_ok", True) and int(state.get("retry", 0)) < 2:
        return "coder"
    if state.get("tests_ok", True) and state.get("after_verify") == "doc":
        return "doc"
    return "final"
```

也就是说：

- 测试失败且重试次数不足：回 coder。
- 测试成功且需要文档：去 doc。
- 其他情况：final。

注意 `retry` 是完整 Coding 尝试次数，不是内部 16 步 ReAct 的 step。最多进行 3 次 Coding 尝试；标记为 `infra` 的模型或网络故障不会错误地交给 Coder 修改用户代码。

## 19. 文档智能体：`doc()`

文档节点只在需要文档时执行。

### 19.1 调用模型

```python
data = client.chat_json([
    {"role": "system", "content": DOC_PROMPT},
    {"role": "user", "content": self._ctx_text(state) + 仓库 + 变更 + 测试}
])
```

模型应该返回：

```json
{
  "path": "README.md",
  "content": "完整 Markdown 内容",
  "summary": "文档变更摘要"
}
```

### 19.2 防止写出项目目录

```python
path = data.get("path") or "README.md"
if Path(path).is_absolute():
    path = "README.md"
written = fs.write(path, ...)
```

如果模型返回绝对路径，强制改成 `README.md`。

真正的路径安全仍然由 `FsTool.safe()` 保证。

### 19.3 文档路径怎么进入最终结果

```python
repo = dict(state.get("repo", {}))
repo["doc_path"] = str((Path(state["workdir"]) / written).resolve())
changes = sorted(dict.fromkeys(list(state.get("changes", [])) + [written]))
return {**state, "repo": repo, "changes": changes}
```

这里把文档路径写进 `repo["doc_path"]`，后面 `final()` 会取出来：

```python
"doc_path": (state.get("repo") or {}).get("doc_path")
```

## 20. 最终节点：`final()`

所有路径最后都会到 `final()`。

```python
ok = bool(state.get("tests_ok", True)) and bool(state.get("coding_ok", True))
summary = state.get("final") or ("任务完成" if ok else "任务完成，但验证存在失败")
```

如果前面节点已经设置了 `final`，就用它。否则根据验证结果给默认摘要。

然后整理结果：

```python
result = {
    "ok": ok,
    "summary": summary,
    "files": state.get("changes", []),
    "commands": state.get("commands", []),
    "tests": state.get("tests", []),
    "plan_path": (state.get("plan") or {}).get("path"),
    "doc_path": (state.get("repo") or {}).get("doc_path"),
    "tokens": int(state.get("tokens", 0)),
    "duration_ms": duration_ms,
}
```

写 memory：

```python
self.memory.append(state["workdir"], state["session_id"], "manager", "final", "result", summary, result)
```

这条记录非常关键：

- 历史会话恢复靠它。
- 旧格式中断判断会参考它；新格式优先看 `run/start|done|error`。
- 前端最终回复也是它格式化出来的。

最后尝试压缩 memory：

```python
cfg = self.model_store.for_agent("manager", state.get("model_id"))
self.memory.maybe_compress(..., cfg.ctx)
```

## 21. 事件是怎么产生的：`_emit()` 和 `_add_tokens()`

每个节点都会调用 `_emit()`：

```python
self._emit(state, "manager", "start", "管理者正在分类任务并构造上下文包")
```

`_emit()` 做两件事：

1. 写入 memory。
2. 调用外部 `emit_cb`。

```python
self.memory.append(..., agent, "event", kind, self._trim(msg, 3000), {"tokens": tokens})
if self.emit_cb:
    self.emit_cb(event)
```

所以事件既能出现在前端事件流，也会保存在会话 jsonl。memory 只记录精简消息和 token 等必要元数据，不再重复保存完整命令输出；实时 NDJSON 流包含完整结构化事件，历史恢复则从精简记录重建整个会话的事件，`data` 字段为空对象。

`_add_tokens()` 是特殊事件：

```python
state["tokens"] = int(state.get("tokens", 0)) + tokens
self._emit(state, "llm", "usage", f"本次模型调用约消耗 {tokens} token", tokens=tokens)
```

它把模型调用 token 累加到 state，并发出 usage 事件。

## 22. Prompt 文件怎么读：`backend/agents/prompts.py`

这里不是普通文本，而是每个智能体的行为边界。

### 22.1 `MANAGER_PROMPT`

要求管理者只返回分类 JSON：

```json
{
  "task_type": "direct|general_answer|code_gen|code_mod|code_explain|doc_gen|plan_gen",
  "need_repo": true,
  "need_code": false,
  "need_doc": false,
  "need_clarify": false,
  "reason": "一句中文理由"
}
```

但注意：代码里已经用规则覆盖了大部分高频情况。Prompt 只是兜底。

### 22.2 `PLANNER_PROMPT`

Plan 智能体只能返回：

- questions
- plan

不能自由发挥长篇解释。

它还明确要求：

```text
如果上下文中包含“上一轮 Plan 问题与用户回答”，必须把原始需求和用户选择合并理解
```

这是配合 `_compose_pending_plan_text()` 使用的。

### 22.3 `CODER_PROMPT`

最关键的约束：

```text
负责真正写入磁盘，不允许只输出代码片段。
每次只返回 JSON。
写文件时给出完整内容，不要给 diff。
```

这个 prompt 和 `_do_action()` 是一对：

- prompt 告诉模型有哪些工具。
- `_do_action()` 真正执行工具。

如果你新增工具，一定要同时改两处。

## 23. 启动脚本：`scripts/dev.py`

这个脚本不是 agent 核心，但它决定项目怎么启动。

### 23.1 端口选择

```python
backend_port = pick_port(8710, 8730)
frontend_port = pick_port(5173, 5199)
```

不固定端口，避免占用。

### 23.2 注入前端 API 地址

```python
env["VITE_API_URL"] = f"http://127.0.0.1:{backend_port}"
```

前端通过这个环境变量知道后端端口。

### 23.3 启动两个进程

```python
backend = run([sys.executable, "-m", "uvicorn", ...])
frontend = run([npm, "run", "dev", ...])
```

然后主循环监控两个进程。任意一个退出，就关闭另一个。

启动器会先等待后端和 Vite 端口真正可连接，再自动打开浏览器。服务器或自动化环境可设置 `AGENT_NO_BROWSER=1` 禁用该行为。

## 24. 测试怎么保护这些行为

重点看 `tests/test_agent.py`、`tests/test_memory.py`、`tests/test_tools.py` 和 `tests/test_llm.py`。测试夹具只存在于测试目录，不会写入 Agent 运行时代码或前端默认数据。

### 24.1 普通问候不能读仓库

```python
def test_direct_greeting_does_not_read_repo_or_write_files(tmp_path):
    result = graph.run(..., text="你好")
    assert result.files == []
    assert result.commands == []
    assert result.tests == []
    assert not (workdir / "README.md").exists()
```

这个测试保护了你之前强调的问题：输入“你好”不能触发全流水线。

### 24.2 文档任务不能走 Coding

```python
def test_doc_task_routes_to_repo_then_doc_without_coder(tmp_path):
    next_state = graph.manager(state)
    assert next_state["route"] == "repo"
    assert next_state["after_repo"] == "doc"
```

它保证纯文档任务只走 repo 和 doc。

### 24.3 Plan 选项回复必须带上下文

```python
def test_manager_continues_pending_plan_from_option_reply(tmp_path):
    memory.append(..., "pending_plan", ..., {"goal": "创建一个本地多智能体编程系统", "questions": [...]})
    next_state = graph.manager({"text": "1A, 2A", ...})
    assert "创建一个本地多智能体编程系统" in next_state["context"].goal
    assert "React + TypeScript + Vite" in next_state["context"].goal
```

这个测试保护 `_latest_pending_plan()`、`_build_plan_reply()`、`_compose_pending_plan_text()` 这一整条链。

### 24.4 memory 历史恢复

```python
def test_memory_lists_history_messages(tmp_path):
    store.append(..., "user", "input", "message", "你好")
    store.append(..., "manager", "final", "result", "你好，我在。", ...)
    history = store.list_history()
    assert history[0]["sessions"][0]["messages"] == [...]
```

它保证对话窗口只恢复用户输入和最终回复。`test_memory_history_restores_all_session_events()` 另外保证右侧事件流按顺序恢复整个会话的全部事件。

## 25. 如果你要新增功能，按这个思路改

### 25.1 新增一个任务类型

比如新增 `code_review`：

1. `api/schema.py` 不一定要改，除非要把任务类型枚举化。
2. `AgentGraph._classify()` 增加识别规则。
3. `AgentGraph._flow_for()` 增加路由。
4. 如果需要新节点，`_build()` 注册节点。
5. 增加 prompt。
6. 增加测试。

### 25.2 新增一个智能体

比如新增 `reviewer`：

1. `api/schema.py` 的 `AgentName` 加 reviewer。
2. `AgentModelMap` 加 reviewer。
3. `prompts.py` 加 `REVIEWER_PROMPT`。
4. `graph.py` 加 `reviewer()` 节点函数。
5. `_build()` 注册节点和边。
6. `_flow_for()` 或路由函数把任务导向 reviewer。
7. tests 加路由测试。

### 25.3 新增一个 Coding 工具

比如新增 `replace_file`：

1. `backend/tools/fs.py` 增加方法。
2. `AgentGraph._do_action()` 增加分支。
3. `CODER_PROMPT` 里告诉模型工具格式。
4. 测试路径不能越界。

### 25.4 修改验证策略

改 `verifier()`。

比如想加入 Ruff：

```python
if "pyproject.toml" in files:
    commands.append("uv run pytest")
    commands.append("uv run ruff check .")
```

然后改测试。

## 26. 最后用一条完整路径串起来

假设用户输入：

```text
创建一个手机销售店铺智能客服系统，包含 README 文档
```

后端执行路径是：

```text
1. POST /api/chat/stream
2. chat_stream() 通过 session_id 找到 workdir
3. memory 写入 user/input/message
4. 创建 AgentGraph
5. AgentGraph.run() 初始化 state
6. manager()
   - 没有 pending_plan
   - 没有 saved_plan 执行意图
   - _classify() 命中 code_gen
   - _flow_for() 返回 repo/coder/doc
   - 构造 ContextPackage
7. repo()
   - 列文件
   - 读 README/package/pyproject/index 等关键文件
   - 识别技术栈
8. coder()
   - 调用 coder 模型
   - 模型返回 actions
   - _do_action() 调 FsTool/ShellTool 写文件或跑命令
9. verifier()
   - 根据文件选择测试命令
   - 如果失败并且 retry < 2，回 coder
10. doc()
   - 生成或更新 README/docs 文档
11. final()
   - 整理 TaskResult
   - 写 manager/final/result
   - 压缩 memory
12. chat_stream() 把 result 作为 NDJSON 返回
```

再看普通问候：

```text
你好
```

路径是：

```text
1. POST /api/chat/stream
2. manager()
3. _direct_reply() 命中
4. _flow_for() 返回 final/final/final
5. final()
```

不会 repo，不会 coder，不会 verifier，不会 doc。

这就是这个项目最核心的设计。

## 27. 你现在应该怎么练习

按下面顺序练习，每一步都去代码里找对应函数。

1. 找到 `ChatRequest`，说出每个字段的作用。
2. 找到 `chat_stream()`，画出 event_queue 如何连接 worker 和 stream。
3. 找到 `AgentGraph.run()`，说出初始化 state 里每个字段后面谁会用。
4. 找到 `_build()`，用文字写出所有可能路径。
5. 找到 `manager()`，解释 pending_plan、saved_plan、classification 三者关系。
6. 找到 `_classify()`，新增一个测试用例判断某句话会走什么任务类型。
7. 找到 `planner()`，说明 questions 和 plan 两种返回分别怎么处理。
8. 找到 `coder()`，解释 observations 为什么存在。
9. 找到 `_do_action()`，说出每个 tool 最终调用哪个工具类。
10. 找到 `verifier()`，解释一个静态 HTML 项目会得到什么测试结果。
11. 找到 `final()`，解释历史会话为什么能恢复最终回复。

完成这些练习后，你就能比较稳地修改这个 agent 项目的后端功能。
