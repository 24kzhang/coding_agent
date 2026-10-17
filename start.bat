@REM 作者：张又升
@REM 邮箱：1041586605@qq.com
@REM 单位：西北大学 电子信息学院(人工智能学院)

@echo off
cd /d %~dp0
uv sync
uv run python scripts\dev.py
