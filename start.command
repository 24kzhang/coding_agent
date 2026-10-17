#!/usr/bin/env bash
# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

set -e
cd "$(dirname "$0")"
uv sync
uv run python scripts/dev.py
