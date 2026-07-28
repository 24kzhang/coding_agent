---
name: web-design-guidelines
description: Review UI code for Web Interface Guidelines compliance. Use when asked to "review my UI", "check accessibility", "audit design", "review UX", or "check my site against best practices".
metadata:
  author: vercel
  version: "1.0.0"
  argument-hint: <file-or-pattern>
---

# Web Interface Guidelines

Review files for compliance with Web Interface Guidelines.

## How It Works

1. Use the bundled `references/guidelines.md` snapshot injected with this skill
2. Read the specified files (or prompt user for files/pattern)
3. Check against all bundled rules
4. Output findings in the terse `file:line` format

## Guidelines Source and Offline Fallback

The bundled snapshot comes from:

```
https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md
```

If a web-fetch tool is available, it may refresh the rules from that URL. Network access is optional:
never block a review when the URL is unavailable, and use the bundled snapshot as the authoritative
fallback.

## Usage

When a user provides a file or pattern argument:
1. Read the bundled guidelines included below the main skill instructions
2. Read the specified files
3. Apply all rules from the fetched guidelines
4. Output findings using the format specified in the guidelines

If no files specified, ask the user which files to review.
