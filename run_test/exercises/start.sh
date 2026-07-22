#!/usr/bin/env bash
# macOS / Linux 启动脚本

set -e

# 检查 uv 是否已安装
if ! command -v uv &> /dev/null; then
    echo "错误：未找到 uv，请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# 检查 LONGCAT_API_KEY 是否已设置
if [ -z "$LONGCAT_API_KEY" ]; then
    echo "错误：未设置 LONGCAT_API_KEY 环境变量，请先设置后再启动。"
    exit 1
fi

# 安装/同步依赖
echo "正在同步项目依赖..."
uv sync

# 启动服务
echo "正在启动健身动作网站..."
uv run uvicorn main:app --host 0.0.0.0 --port 8000
