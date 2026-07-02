from __future__ import annotations

import socket

from scripts.dev import pick_port


def test_pick_port_skips_used_port() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    used = sock.getsockname()[1]
    try:
        assert pick_port(used, used + 1) == used + 1
    finally:
        sock.close()
