# MCP 配置

本目录保存 Coding Agent 的 MCP 服务配置。主配置文件是 `servers.json`，支持本地
`stdio` 和远程 `streamable_http` 两种传输方式。修改配置后，在前端重新打开
“能力”弹窗即可刷新列表。

## 已安装服务

项目已固定安装两个官方 MCP 服务。

### Filesystem

官方 Filesystem MCP `2026.7.10`：

- 配置 id：`filesystem`
- 前端名称：`项目文件系统`
- 允许目录：当前会话选择的项目目录
- 可用工具：文本读取、目录列表、文件搜索、元数据读取，以及文件写入和编辑等

Node 依赖声明在 `MCP/package.json`，锁定结果在 `MCP/package-lock.json`。首次双击
启动项目时，`scripts/dev.py` 会在 `MCP/node_modules` 不存在时自动运行
`npm install`。npm 缓存使用项目内 `.cache/npm`，不依赖用户目录的全局缓存权限。

在前端先选择项目，再点击“能力”，勾选“项目文件系统”。右侧测试按钮只执行 MCP
初始化和 `tools/list`，不会写入文件。真正执行任务时，Coder 只能调用当前会话已勾选、
并且由该服务实际发现的工具。

### GitHub

官方 GitHub MCP Server `1.7.0`：

- 配置 id：`github`
- 前端名称：`GitHub（只读）`
- 启动方式：`scripts/github_mcp.py` 按系统下载、校验并缓存官方原生二进制
- 支持系统：Apple Silicon macOS、Windows x86_64、Windows ARM64
- 工具范围：仓库内容、提交、分支、Issue、Pull Request、Release、用户等只读查询
- 授权方式：GitHub 官方设备授权，适用于浏览器无法访问本机 OAuth 回调的情况

先选择项目，再打开“能力”弹窗。GitHub 行的钥匙按钮用于授权，信号按钮只执行
`initialize` 和 `tools/list`。授权页要求输入一次性设备码；成功后后端调用
`get_me` 验证身份。前端授权弹窗会持续展示设备码，并提供复制和打开 GitHub
授权页两个独立操作，切换到 GitHub 页面后仍可回到 Agent 查看设备码。

身份验证通过后，访问令牌明文保存在本机 `MCP/auth.json`，并通过
`auth_token_env` 声明的环境变量传给 GitHub MCP 子进程。该文件已加入 `.gitignore`，
macOS 上写入权限为 `0600`，不会进入 `servers.json`、memory、日志、事件流、模型
上下文或前端响应。项目重启后会加载已保存 token，并在用户再次打开授权窗口时调用
`get_me` 验证；token 失效时会删除旧值并自动生成新的设备码。

`MCP/auth.example.json` 只展示配置结构，不包含真实凭据。`MCP/auth.json` 属于本机
敏感文件，不应提交、分享或放入项目截图；删除该文件即可清除所有 MCP 持久授权。

GitHub 服务使用 `--read-only` 启动，并把 OAuth 范围限制为
`repo read:org read:user user:email`。其中经典 `repo` scope 是读取私有仓库所需的
GitHub 粗粒度权限，因此 GitHub 授权页仍会提示仓库访问；服务自身的只读模式会继续
禁用写工具。项目尚未实现高风险 MCP 工具审批前，不应移除 `--read-only`。

`servers.json` 中与持久授权有关的字段：

```json
{
  "persistent": true,
  "auth_tool": "get_me",
  "auth_mode": "github_device",
  "auth_token_env": "GITHUB_PERSONAL_ACCESS_TOKEN"
}
```

## stdio 配置

```json
{
  "servers": [
    {
      "id": "service-id",
      "name": "服务显示名",
      "description": "这个服务提供什么能力",
      "enabled": true,
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "package-name", "${workspace}"],
      "env": {},
      "timeout": 60,
      "persistent": false
    }
  ]
}
```

`${workspace}` 会替换为前端当前选择的用户项目目录，`${agent_root}` 会替换为本
Coding Agent 的项目根目录。`cwd` 未配置时默认使用 `${workspace}`。需要把认证
状态保存在服务进程内的 MCP 应配置 `persistent: true`；普通服务保持默认短连接。

## Streamable HTTP 配置

```json
{
  "servers": [
    {
      "id": "remote-service",
      "name": "远程服务",
      "description": "通过 Streamable HTTP 提供工具",
      "enabled": true,
      "transport": "streamable_http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer your-token"
      },
      "timeout": 60
    }
  ]
}
```

也兼容常见的 `mcpServers` 对象格式。`env` 和 `headers` 只保存在本机文件中，后端
能力列表接口不会把它们返回给浏览器。前端“测试连接”只执行 MCP 初始化和工具发现，
不会调用任何有副作用的工具。
