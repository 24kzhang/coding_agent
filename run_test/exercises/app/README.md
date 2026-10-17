# 力场 · 健身动作库与动作咨询

基于 `exercises-dataset` 中文数据构建的健身动作库：首页覆盖全部 1324 个动作，支持分类 / 目标肌肉 / 器械组合筛选与名称搜索；点击卡片打开双栏弹窗，左侧展示该动作 JSON 的全部字段，右侧为接入真实模型（LongCat-2.0）的动作咨询聊天，服务端按会话保存历史并流式回复。

## 启动

```bash
uv sync --directory app
uv run --directory app uvicorn main:app --reload --port 8000
```

浏览器打开 <http://127.0.0.1:8000/>。

## 测试与验证

```bash
# 单元测试（数据层 + 聊天上下文，替换上游传输边界）
uv run --directory app python -m pytest tests/test_data.py tests/test_chat.py -q

# 真实启动烟测（uvicorn + HTTP 接口）
uv run --directory app python tests/smoke.py

# 真实模型连续两轮流式对话（不打印任何凭据）
uv run --directory app python tests/live_chat.py

# 浏览器交互验收（Playwright 无头 Chromium，含真实模型两轮对话）
uv run --directory app python -m playwright install chromium
uv run --directory app python -m pytest tests/test_e2e.py -q
```

## 验证结果

| 项 | 命令 | 结果 |
|---|---|---|
| 数据层与接口单元测试 | `pytest tests/test_data.py` | 通过（10 项） |
| 聊天上下文单元测试 | `pytest tests/test_chat.py` | 通过（10 项） |
| 真实启动烟测 | `python tests/smoke.py` | 通过：列表 1324 条、组合筛选 89 条、详情、GIF 媒体、404 |
| 真实模型两轮流式对话 | `python tests/live_chat.py` | 通过：两轮均收到 `delta` 与 `done`，第二轮回复非空 |
| 浏览器交互验收 | `pytest tests/test_e2e.py` | 通过（13 项，111.7s）：GIF 来源、无静图请求、筛选与清空、空态、完整详情（含 image / gif_url 原始路径）、真实模型流式两轮与会话载荷、会话隔离、同动作重开与刷新恢复历史、键盘 Esc/Tab 与停止中断、挂起-放行自检、晚到历史不覆盖新消息、旧流结束不污染新面板、详情失败清除旧信息 |

## 目录说明

```
app/
├── pyproject.toml      # uv 项目与依赖（无 build-system，按服务方式运行）
├── main.py             # FastAPI 入口：页面、列表、详情、媒体、聊天 SSE
├── data.py             # 数据加载、派生字段（gif）、筛选与 facets
├── chat.py             # 模型配置加载、会话存储、上游流式请求与 SSE 转发
├── tpl/index.html      # 页面骨架（Jinja2）
├── static/app.css      # 设计令牌与全部样式
├── static/app.js       # 筛选、弹窗、聊天流式渲染、键盘交互
├── tests/              # 单元测试、烟测、真实模型验证、浏览器验收
└── .local/model.json   # 私密模型配置，已忽略提交
```

## 数据与媒体

- 数据来源：`exercises-dataset/data/exercises_zh.json`（只读，1324 条），字段含义见 `exercises.schema.json`。
- 页面图片全部来自 `exercises-dataset/videos/*.gif`，经 `/media/<文件名>` 提供；不使用数据集中的静图目录，也不引用附件设计图。
- 媒体版权：© [Gym visual](https://gymvisual.com/)，详见 `exercises-dataset/NOTICE.md`。

## 聊天设计要点

- 前端为每个动作生成独立 `session_id`（`crypto.randomUUID()`），保存在 `sessionStorage` 的 `chat:<动作 id>` 键下；切换动作即切换会话。
- 服务端内存字典保存会话；首轮提问时注入包含动作名称与描述的系统消息，此后只追加用户消息并发送完整历史，不重复插入动作信息。
- 会话保存在进程内存中，服务重启即清空；前端保留的 `session_id` 在服务端无历史时会自动重建首轮上下文。
- **刷新后历史仅在服务端进程存活期间可恢复**：`sessionStorage` 只保存 `session_id`，聊天记录本身存在服务端内存；服务端重启（或进程被回收）后，同一动作重新打开或刷新页面只能看到欢迎语，随后提问会以该动作信息重建首轮上下文，旧对话不可找回。这是当前内存会话方案的已知边界，如需跨重启保留需改为持久化存储。
- 可见历史与模型历史一致：`GET /api/chat/history?session_id=&exercise_id=` 返回该会话的 user/assistant 消息（**不含** system 动作上下文），同一动作关闭弹窗再打开或刷新页面后据此恢复聊天记录；不同动作的会话互不串扰。
- 上游失败或用户点击“停止”中断时，本轮未被回答的用户消息会从服务端历史回滚，避免出现“有提问无回复”的错位记录；前端保留气泡并给出“已中断”或“重试”反馈。
- 旧流回调隔离：切换动作或关闭弹窗时先 `AbortController.abort()`，并用会话令牌校验，过期流的增量不会写入新动作的面板。
- 模型配置位于 `app/.local/model.json`，由 `model_copy` 写入，已加入 `.gitignore`；密钥只在内存中使用，不写日志、不回显、不进入任何响应。配置缺失时页面其余功能仍可用，聊天返回中文错误提示。

## 外部资料核对（agent-reach / Exa）

实现与验收前用 Exa 检索了官方文档，并把结论落到代码与测试中：

| 来源 | 采用的要点 | 对应实现 |
|---|---|---|
| [MDN：Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) | 事件流为 UTF-8 文本，消息之间以一对换行分隔；`event` 指定事件类型，连续多行 `data:` 需拼接 | `static/app.js` 的 `parseFrames` 按空行分帧、拼接 `data:` 行；`chat.sse` 输出 `event:` + `data:` + 空行 |
| [HTML Standard：Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) | 行结束符可为 CRLF、LF 或 CR，空行才触发事件派发 | `parseFrames` 使用 `/\r?\n\r?\n/` 与 `/\r?\n/`，兼容 CRLF 与 LF 两种换行 |
| [Playwright Python：Network](https://playwright.dev/python/docs/network) | `page.on("request")` 订阅请求事件；`page.route()` + `route.abort()` 可中断请求 | `tests/test_e2e.py` 用 `page.on("request")` 捕获 `/api/chat` 请求体，并检查页面无静图请求 |
| [Playwright Python：Request](https://playwright.dev/python/docs/api/class-request) | `request.post_data_json` 返回解析后的 JSON 请求体 | 验收中用它深复制两轮聊天载荷，断言第二轮只含新消息 |
| [Playwright Python：Mock APIs](https://playwright.dev/python/docs/mock) | `page.route()` 可拦截请求并延迟放行，用于模拟慢网络 | `tests/test_e2e.py` 延迟 `/api/chat/history` 与 `/api/chat` 响应，验证晚到历史不覆盖新消息、旧流结束不污染新面板 |
| [MDN：AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController) | `abort()` 使 `fetch` 以 `AbortError` 拒绝，可中断流式读取 | 停止按钮与关闭弹窗调用 `controller.abort()`，中断后气泡标记“已中断”，服务端不写入该轮回复 |
| [MDN：Dialog element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/dialog) | `showModal()` 打开模态框，`Esc` 触发 `cancel` 事件，`close` 事件在关闭后触发 | 弹窗用原生 `<dialog>`，`cancel` 中先中止流再关闭，`close` 中恢复 body 滚动与焦点 |

## 键盘与可访问性

- 卡片为 `<button>`，整卡可 Tab 聚焦；`:focus-visible` 使用柠檬黄外环。
- 弹窗使用原生 `<dialog>`，`Esc` 关闭（流式中先中止请求），`Tab` 在弹窗内循环，关闭后焦点回到触发卡片。
- 聊天输入框 `Enter` 发送、`Shift+Enter` 换行；发送中显示“停止”按钮可中断，气泡标记“已中断”。
- 异步状态（计数、错误、聊天记录）使用 `aria-live`；图片均有 `alt` 与显式宽高；动画尊重 `prefers-reduced-motion`。
