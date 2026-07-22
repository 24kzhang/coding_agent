@echo off
REM Windows 启动脚本
chcp 65001 >nul 2>&1

REM 检查 uv 是否已安装
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到 uv，请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

REM 检查 LONGCAT_API_KEY 是否已设置
if "%LONGCAT_API_KEY%"=="" (
    echo 错误：未设置 LONGCAT_API_KEY 环境变量，请先设置后再启动。
    exit /b 1
)

REM 安装/同步依赖
echo 正在同步项目依赖...
uv sync
if %errorlevel% neq 0 (
    echo 错误：依赖同步失败，请检查网络连接或 pyproject.toml 配置。
    exit /b 1
)

REM 启动服务
echo 正在启动健身动作网站...
uv run uvicorn main:app --host 0.0.0.0 --port 8000
