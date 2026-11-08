"""CLI 与配置：Mock 单次执行、缺密钥报错、会话文件落盘。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miniagent.cli import main
from miniagent.errors import ConfigError
from miniagent.llm import OpenAICompatProvider


def test_mock_single_shot(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    exit_code = main(
        [
            "--mock",
            "-p",
            "你好",
            "--config",
            "no-such-config.toml",
            "--session-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[mock] 收到：你好" in captured.out
    assert "session:" in captured.out
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


def test_single_shot_writes_complete_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--mock", "-p", "hi", "--config", "no-such-config.toml", "--session-dir", str(tmp_path)])
    capsys.readouterr()
    session_file = next(tmp_path.glob("*.jsonl"))

    lines = session_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3  # system + user + assistant
    assert '"role": "user"' in lines[1]


def test_resume_unknown_session_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--mock", "-p", "hi", "--resume", "ghost", "--session-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "会话不存在" in captured.err


def test_missing_api_key_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="未配置模型 API 密钥"):
        OpenAICompatProvider(api_key="", model="any", base_url="http://localhost:1")


def test_mock_ignores_missing_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MINIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = main(
        ["--mock", "-p", "hi", "--config", "no-such-config.toml", "--session-dir", str(tmp_path)]
    )

    assert exit_code == 0
