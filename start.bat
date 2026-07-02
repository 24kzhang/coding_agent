@echo off
cd /d %~dp0
uv sync
uv run python scripts\dev.py
