# Skills

每个 Skill 使用独立英文目录，入口文件固定命名为 `SKILL.md`：

```text
skills/
└── your-skill/
    ├── SKILL.md
    ├── AGENTS.md
    ├── scripts/
    └── references/
```

`SKILL.md` 推荐使用 YAML front matter：

```markdown
---
name: your-skill
description: 说明这个 Skill 适合解决什么任务
---

# Skill 指令

这里编写需要 Agent 遵循的流程、约束和检查项。
```

前端只显示格式有效的 Skill。只有用户在“能力”弹窗中手动选中的 Skill 才会进入
本轮结构化上下文。加载时始终先读取 `SKILL.md`，然后在单包 40,000 字符、单轮总计
100,000 字符的预算内，依次附加 `references/**/*.md` 和 `AGENTS.md`。因此第三方
Skill 的规则文件可以真正进入模型上下文，同时不会一次加载整个 `skills/` 目录。

## 已安装的前端 Skills

- `frontend-design`：Anthropic 的视觉设计与前端实现指导。
- `web-design-guidelines`：Vercel 的界面、可访问性和 UX 审查规则，包含离线规则快照。
- `vercel-react-best-practices`：Vercel 的 React/Next.js 性能规则。
- `vercel-composition-patterns`：Vercel 的 React 组件组合与状态设计模式。

`web-design-guidelines` 的本地快照来源于 Vercel 官方
`web-interface-guidelines/command.md`。更新第三方 Skill 时需要重新检查文件内容，
不要在未审查的情况下覆盖本项目已经补充的离线兼容说明。
