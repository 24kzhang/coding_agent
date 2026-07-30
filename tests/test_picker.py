from __future__ import annotations

import subprocess

import pytest
from fastapi import HTTPException

from backend import main


def test_pick_dir_returns_resolved_system_selection(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(main, "_pick_dir_by_system_dialog", lambda: str(project))

    assert main.pick_dir() == {"workdir": str(project.resolve())}


def test_pick_dir_failure_is_recoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_pick_dir_by_system_dialog", lambda: "")

    with pytest.raises(HTTPException) as caught:
        main.pick_dir()

    assert caught.value.status_code == 400
    assert "系统目录选择器不可用" in caught.value.detail


def test_macos_cancel_does_not_open_tkinter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "用户取消"),
    )

    assert main._pick_dir_by_system_dialog() == ""
