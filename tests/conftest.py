"""测试隔离：trace 与记忆默认写到临时目录，不污染用户主目录。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_local_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIAGENT_LOG_DIR", str(tmp_path / "trace-logs"))
    monkeypatch.setenv("MINIAGENT_MEMORY_DIR", str(tmp_path / "memory"))
    # 测试不应受开发者本地 soul.md 影响
    monkeypatch.chdir(tmp_path)
