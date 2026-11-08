"""测试隔离：trace 日志默认写到临时目录，不污染用户主目录。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_trace_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIAGENT_LOG_DIR", str(tmp_path / "trace-logs"))
