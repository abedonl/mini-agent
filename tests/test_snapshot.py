"""高危操作的 git 快照：触发范围、仓库零污染、降级、回滚与开关。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from miniagent.checkpoint import Checkpoint, read_records
from miniagent.config import load_config
from miniagent.errors import ConfigError
from miniagent.llm import Message, ToolCall
from miniagent.logging import TraceLogger
from miniagent.security import PathPolicy
from miniagent.session import Session
from miniagent.snapshot import (
    GitSnapshotter,
    SnapshotError,
    default_checkpoint_path,
    list_snapshots,
    rollback_snapshots,
)
from miniagent.tools.builtin import ReadFileTool, WriteFileTool
from miniagent.tools.registry import ToolRegistry

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "tester")
    (repo / "a.txt").write_text("v1", encoding="utf-8")
    (repo / "b.txt").write_text("b1", encoding="utf-8")
    git(repo, "add", "a.txt", "b.txt")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def make_session(tmp_path: Path) -> Session:
    session = Session.create("s1", tmp_path / "sessions")
    session.append(Message(role="system", content="sys"))
    return session


def make_registry(repo: Path | None) -> ToolRegistry:
    registry = ToolRegistry()
    policy = PathPolicy([repo] if repo else [Path.cwd()], workdir=str(repo) if repo else None)
    registry.register(WriteFileTool(policy))
    registry.register(ReadFileTool(policy))
    return registry


def write_call(call_id: str, path: Path, content: str) -> ToolCall:
    return ToolCall(
        call_id,
        "write_file",
        json.dumps({"path": str(path), "content": content}, ensure_ascii=False),
    )


class FailingSnapshotter:
    def capture(self) -> Any:
        raise SnapshotError("模拟 git 故障")


# ---- 触发范围 -------------------------------------------------------------


def test_high_risk_write_triggers_snapshot_without_polluting_repo(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    (repo / "b.txt").write_text("b2", encoding="utf-8")  # 保持未提交改动
    head_before = git(repo, "rev-parse", "HEAD").stdout.strip()
    staged_before = git(repo, "diff", "--cached", "--name-only").stdout

    session = make_session(tmp_path)
    registry = make_registry(repo)
    registry.checkpoint = Checkpoint.for_session(session)
    registry.snapshotter = GitSnapshotter(repo)

    result = registry.execute(write_call("c1", repo / "a.txt", "v2"), 1)
    assert not result.is_error

    start = read_records(default_checkpoint_path(session.path))[0]
    assert start["snapshot_kind"] == "stash"
    assert start["snapshot"].startswith(
        git(repo, "rev-parse", "--verify", start["snapshot"]).stdout.strip()[:7]
    )
    assert start["snapshot_workdir"] == str(repo.resolve())

    # 快照不污染仓库：HEAD、暂存区都不变
    assert git(repo, "rev-parse", "HEAD").stdout.strip() == head_before
    assert git(repo, "diff", "--cached", "--name-only").stdout == staged_before
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v2"


def test_read_only_tool_never_snapshots(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    session = make_session(tmp_path)
    tracer = TraceLogger("s1", tmp_path / "logs")
    registry = make_registry(repo)
    registry.checkpoint = Checkpoint.for_session(session, tracer=tracer)
    registry.snapshotter = GitSnapshotter(repo, tracer=tracer)

    result = registry.execute(
        ToolCall("c1", "read_file", json.dumps({"path": str(repo / "a.txt")})), 1
    )
    assert not result.is_error

    start = read_records(default_checkpoint_path(session.path))[0]
    assert start["snapshot"] == ""
    assert start["snapshot_kind"] == ""
    events = [record.event for record in TraceLogger.load("s1", tmp_path / "logs")]
    assert "snapshot" not in events
    assert "checkpoint" in events


def test_non_git_dir_degrades_to_metadata_only(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    notices: list[str] = []
    registry = make_registry(tmp_path)
    registry.checkpoint = Checkpoint.for_session(session)
    registry.snapshotter = GitSnapshotter(tmp_path, notice=notices.append)

    result = registry.execute(write_call("c1", tmp_path / "a.txt", "x"), 1)

    assert not result.is_error
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"
    start = read_records(default_checkpoint_path(session.path))[0]
    assert start["snapshot_kind"] == "unavailable"
    assert start["snapshot"] == ""
    assert notices and "不是 git 仓库" in notices[0]


def test_snapshot_failure_refuses_high_risk_tool(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    registry = make_registry(tmp_path)
    registry.checkpoint = Checkpoint.for_session(session)
    registry.snapshotter = FailingSnapshotter()

    result = registry.execute(write_call("c1", tmp_path / "a.txt", "x"), 1)

    assert result.is_error
    assert "快照失败" in result.content
    assert not (tmp_path / "a.txt").exists()
    # 没有执行，也就不该留下 start 记录
    assert read_records(default_checkpoint_path(session.path)) == []


# ---- 回滚 -----------------------------------------------------------------


def test_rollback_restores_latest_or_specific_snapshot(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    session = make_session(tmp_path)
    registry = make_registry(repo)
    registry.checkpoint = Checkpoint.for_session(session)
    registry.snapshotter = GitSnapshotter(repo)

    registry.execute(write_call("c1", repo / "a.txt", "v2"), 1)
    registry.execute(write_call("c2", repo / "a.txt", "v3"), 2)

    refs = list_snapshots(default_checkpoint_path(session.path))
    assert len(refs) == 2

    # 默认回到最后一个快照（v2 之前的状态）
    latest = rollback_snapshots("s1", session_dir=tmp_path / "sessions", force=True)
    assert latest.seq == refs[-1].seq
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v2"

    # --to 回到第一个快照（v1 之前）
    rollback_snapshots(
        "s1", session_dir=tmp_path / "sessions", to_seq=refs[0].seq, force=True
    )
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v1"


def test_rollback_protects_uncommitted_changes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    session = make_session(tmp_path)
    registry = make_registry(repo)
    registry.checkpoint = Checkpoint.for_session(session)
    registry.snapshotter = GitSnapshotter(repo)
    registry.execute(write_call("c1", repo / "a.txt", "v2"), 1)

    with pytest.raises(ConfigError, match="未提交改动"):
        rollback_snapshots("s1", session_dir=tmp_path / "sessions")
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v2"


def test_untracked_files_are_not_restored_or_removed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "loose.txt").write_text("untracked", encoding="utf-8")
    (repo / "b.txt").write_text("b2", encoding="utf-8")  # 让快照走 stash 分支
    session = make_session(tmp_path)
    registry = make_registry(repo)
    registry.checkpoint = Checkpoint.for_session(session)
    registry.snapshotter = GitSnapshotter(repo)
    registry.execute(write_call("c1", repo / "a.txt", "v2"), 1)

    (repo / "later.txt").write_text("created-after", encoding="utf-8")
    ref = rollback_snapshots("s1", session_dir=tmp_path / "sessions", force=True)

    assert ref.kind == "stash"
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v1"
    # 未跟踪文件既不进快照，也不会被回滚删除
    assert (repo / "loose.txt").read_text(encoding="utf-8") == "untracked"
    assert (repo / "later.txt").read_text(encoding="utf-8") == "created-after"


# ---- 开关 -----------------------------------------------------------------


def test_snapshot_off_is_zero_behavior(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    session = make_session(tmp_path)
    registry = make_registry(repo)
    registry.checkpoint = Checkpoint.for_session(session)
    # 关闭：不挂 snapshotter
    registry.snapshotter = None

    registry.execute(write_call("c1", repo / "a.txt", "v2"), 1)

    start = read_records(default_checkpoint_path(session.path))[0]
    assert start["snapshot"] == ""
    assert start["snapshot_kind"] == ""
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v2"


def test_config_reads_snapshot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIAGENT_SNAPSHOT", "off")
    assert load_config(tmp_path / "none.toml").snapshot == "off"

    monkeypatch.setenv("MINIAGENT_SNAPSHOT", "sometimes")
    with pytest.raises(ConfigError, match="snapshot"):
        load_config(tmp_path / "none.toml")


def test_rollback_cli_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from miniagent.cli import main

    repo = make_repo(tmp_path)
    session = make_session(tmp_path)
    registry = make_registry(repo)
    registry.checkpoint = Checkpoint.for_session(session)
    registry.snapshotter = GitSnapshotter(repo)
    registry.execute(write_call("c1", repo / "a.txt", "v2"), 1)

    code = main(
        [
            "rollback",
            "s1",
            "--session-dir",
            str(tmp_path / "sessions"),
            "--force",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "已恢复" in out and "未跟踪文件" in out
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v1"
