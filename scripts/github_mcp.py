from __future__ import annotations

import hashlib
import os
import platform
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# ROOT 是 Coding Agent 项目根目录，下载内容只保存在项目自身的忽略目录中。
ROOT = Path(__file__).resolve().parents[1]
# VERSION 固定经过验证的 GitHub MCP 官方发行版本，升级时必须同步更新校验值。
VERSION = "1.7.0"
# RELEASE_BASE 是 GitHub 官方仓库的版本资产下载根地址。
RELEASE_BASE = f"https://github.com/github/github-mcp-server/releases/download/v{VERSION}"
# TARGETS 把运行平台映射到官方资产名和发布页提供的 SHA-256。
TARGETS = {
    ("Darwin", "arm64"): (
        "github-mcp-server_Darwin_arm64.tar.gz",
        "dbe22c73b3eb9491cc3e27dba45352b980b1c43e4c857645689a0251c5e06a21",
    ),
    ("Windows", "AMD64"): (
        "github-mcp-server_Windows_x86_64.zip",
        "14882ca059cd2eccc037388d586b552b60a891b126050f8cefff8beab3c9157f",
    ),
    ("Windows", "ARM64"): (
        "github-mcp-server_Windows_arm64.zip",
        "68a67fb249f846058c58cab88e1e73981650fca29e1c843be523959af00a0070",
    ),
}


def target() -> tuple[str, str]:
    """返回当前系统对应的官方资产与校验值。"""

    # system 和 machine 使用 Python 标准平台名称，避免依赖 bash/uname。
    system = platform.system()
    machine = platform.machine()
    # aliases 统一 Windows 和 macOS 常见架构别名。
    aliases = {
        "aarch64": "arm64" if system == "Darwin" else "ARM64",
        "arm64": "arm64" if system == "Darwin" else "ARM64",
        "x86_64": "AMD64",
        "amd64": "AMD64",
    }
    key = (system, aliases.get(machine.lower(), machine))
    if key not in TARGETS:
        raise RuntimeError(f"GitHub MCP 暂不支持当前平台：{system} {machine}")
    return TARGETS[key]


def ensure_binary() -> Path:
    """下载、校验并解压当前平台的 GitHub MCP 官方二进制。"""

    # asset 和 expected_digest 来自固定版本官方 checksums 文件。
    asset, expected_digest = target()
    # cache_dir 按版本隔离，升级失败不会覆盖此前可用二进制。
    cache_dir = ROOT / ".cache" / "mcp" / "github" / VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    # binary_name 在 Windows 带 exe 后缀，其他平台使用无后缀可执行文件。
    binary_name = "github-mcp-server.exe" if os.name == "nt" else "github-mcp-server"
    binary = cache_dir / binary_name
    if binary.is_file():
        return binary

    # archive 是固定版本压缩包；part 避免中断下载被误认为完整文件。
    archive = cache_dir / asset
    part = archive.with_suffix(archive.suffix + ".part")
    request = urllib.request.Request(
        f"{RELEASE_BASE}/{asset}",
        headers={"User-Agent": "coding-agent-github-mcp-bootstrap"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != expected_digest:
        part.unlink(missing_ok=True)
        raise RuntimeError("GitHub MCP 下载文件校验失败，已拒绝执行")
    part.replace(archive)

    # 只提取名称精确匹配的服务器文件，不使用 extractall，避免压缩包路径穿越。
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            member = next((name for name in bundle.namelist() if Path(name).name == binary_name), None)
            if member is None:
                raise RuntimeError("GitHub MCP 压缩包中没有服务器可执行文件")
            binary.write_bytes(bundle.read(member))
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            member = next((item for item in bundle.getmembers() if item.isfile() and Path(item.name).name == binary_name), None)
            if member is None:
                raise RuntimeError("GitHub MCP 压缩包中没有服务器可执行文件")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError("GitHub MCP 可执行文件读取失败")
            binary.write_bytes(source.read())

    # macOS 需要显式增加当前用户执行位；Windows 忽略该位。
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def main() -> None:
    """启动只读 GitHub MCP stdio 服务，首次运行由官方二进制发起 OAuth。"""

    # args 默认只读，后续若实现高风险工具确认再考虑开放写操作。
    binary = ensure_binary()
    args = [str(binary), "stdio", "--read-only", *sys.argv[1:]]
    os.execv(str(binary), args)


if __name__ == "__main__":
    main()
