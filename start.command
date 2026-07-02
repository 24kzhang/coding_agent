#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv sync
uv run python scripts/dev.py
