# 作者：张又升
# 邮箱：1041586605@qq.com
# 单位：西北大学 电子信息学院(人工智能学院)

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

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.7.0"
RELEASE_BASE = f"https://github.com/github/github-mcp-server/releases/download/v{VERSION}"
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

    system = platform.system()
    machine = platform.machine()
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

    asset, expected_digest = target()
    cache_dir = ROOT / ".cache" / "mcp" / "github" / VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary_name = "github-mcp-server.exe" if os.name == "nt" else "github-mcp-server"
    binary = cache_dir / binary_name
    if binary.is_file():
        return binary

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

    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def main() -> None:
    """启动只读 GitHub MCP stdio 服务，首次运行由官方二进制发起 OAuth。"""

    binary = ensure_binary()
    args = [str(binary), "stdio", "--read-only", *sys.argv[1:]]
    os.execv(str(binary), args)


if __name__ == "__main__":
    main()
