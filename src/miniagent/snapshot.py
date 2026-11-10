"""高危工具执行前的 git 快照：只创建对象，不改工作区、索引与 HEAD。

机制：``git stash create`` 会生成一个游离的 commit 对象并把 sha 打印出来，
但不会把它存进任何 ref、不会改暂存区、也不会改工作区；工作区干净时它不输出
sha，这时退化为记录当前 HEAD。整个模块不执行 ``git add`` / ``git commit`` /
``git checkout <branch>``，因此不会污染用户仓库的状态。

已知限制（不做隐藏）：
- 未跟踪文件、``.gitignore`` 忽略的文件不在快照里；
- 快照是"工作区里已跟踪文件"的状态，回滚只恢复工作区，不动暂存区；
- ``run_shell`` 的副作用可能发生在仓库目录之外，快照管不到。
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miniagent.checkpoint import default_checkpoint_path, read_records
from miniagent.errors import ConfigError

SNAPSHOT_MODES = ("on", "off")
SNAPSHOT_TIMEOUT = 30.0

_KIND_STASH = "stash"
_KIND_CLEAN = "clean"
_KIND_UNAVAILABLE = "unavailable"


class SnapshotError(RuntimeError):
    """git 快照本身失败（超时、git 报错），调用方应拒绝执行高危操作。"""


@dataclass(frozen=True)
class Snapshot:
    kind: str  # stash | clean | unavailable
    sha: str = ""
    reason: str = ""
    workdir: str = ""
    duration_ms: int = 0

    @property
    def usable(self) -> bool:
        return self.kind in {_KIND_STASH, _KIND_CLEAN} and bool(self.sha)


class GitSnapshotter:
    """在指定工作目录创建快照；同一进程内对同一原因只提示一次。"""

    def __init__(
        self,
        workdir: str | Path,
        *,
        tracer: Any | None = None,
        notice: Any | None = None,
        timeout: float = SNAPSHOT_TIMEOUT,
    ) -> None:
        self.workdir = str(Path(workdir).expanduser().resolve())
        self.tracer = tracer
        self.notice = notice
        self.timeout = timeout
        self._notified: set[str] = set()

    def capture(self) -> Snapshot:
        started = time.perf_counter()
        inside = self._git("rev-parse", "--is-inside-work-tree")
        if inside is None or inside.returncode != 0 or inside.stdout.strip() != "true":
            return self._unavailable("not_a_repo", started)

        stash = self._git("stash", "create")
        if stash is None:
            self._trace("failed", reason="timeout", started=started)
            raise SnapshotError(
                f"git stash create 超时（>{self.timeout:.0f}s）。"
                "为避免无保护执行，本次高危操作已拒绝；可用 --snapshot off 显式关闭快照。"
            )
        if stash.returncode != 0:
            message = (stash.stderr or stash.stdout).strip()
            if "initial commit" in message or "does not have any commits" in message:
                return self._unavailable("no_initial_commit", started)
            self._trace("failed", reason="git_error", started=started)
            raise SnapshotError(
                f"git stash create 失败：{message or '未知错误'}。"
                "为避免无保护执行，本次高危操作已拒绝；"
                "请检查 git 配置（如 user.name / user.email），或用 --snapshot off 显式关闭快照。"
            )

        sha = stash.stdout.strip()
        if sha:
            snapshot = Snapshot(
                _KIND_STASH, sha, workdir=self.workdir, duration_ms=_elapsed_ms(started)
            )
        else:
            head = self._git("rev-parse", "HEAD")
            if head is None or head.returncode != 0:
                return self._unavailable("no_head", started)
            snapshot = Snapshot(
                _KIND_CLEAN,
                head.stdout.strip(),
                workdir=self.workdir,
                duration_ms=_elapsed_ms(started),
            )

        self._trace(
            "capture",
            reason=snapshot.kind,
            started=started,
            sha=snapshot.sha,
        )
        return snapshot

    # ---- 内部 ---------------------------------------------------------------

    def _git(self, *args: str) -> subprocess.CompletedProcess[str] | None:
        """返回 None 表示超时；其余错误留给调用方判断 returncode。"""
        env = dict(os.environ)
        env.setdefault("GIT_OPTIONAL_LOCKS", "0")
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    def _unavailable(self, reason: str, started: float) -> Snapshot:
        snapshot = Snapshot(
            _KIND_UNAVAILABLE,
            reason=reason,
            workdir=self.workdir,
            duration_ms=_elapsed_ms(started),
        )
        self._trace("unavailable", reason=reason, started=started)
        if reason not in self._notified:
            self._notified.add(reason)
            text = {
                "not_a_repo": "当前目录不是 git 仓库，无法为高危操作生成回滚快照；"
                "本次只记录 checkpoint 元信息。",
                "no_initial_commit": "git 仓库还没有任何提交，无法生成回滚快照；"
                "本次只记录 checkpoint 元信息。",
                "no_head": "无法读取 git HEAD，无法生成回滚快照；本次只记录 checkpoint 元信息。",
            }.get(reason, f"无法生成 git 快照（{reason}）；本次只记录 checkpoint 元信息。")
            self._notice(text)
        return snapshot

    def _notice(self, text: str) -> None:
        if self.notice is not None:
            self.notice(f"[snapshot] {text}")

    def _trace(
        self, action: str, *, reason: str, started: float, sha: str = ""
    ) -> None:
        if self.tracer is None:
            return
        detail = (
            f"action={action} kind={reason} duration_ms={_elapsed_ms(started)} "
            f"workdir={self.workdir}"
        )
        if sha:
            detail += f" sha={sha}"
        self.tracer.log("snapshot", detail=detail)


@dataclass(frozen=True)
class SnapshotRef:
    seq: int
    tool: str
    sha: str
    kind: str
    workdir: str
    ts: float


def list_snapshots(checkpoint_path: Path | str) -> list[SnapshotRef]:
    """从 checkpoint 里挑出可回滚的快照（按 seq 升序）。"""
    refs: list[SnapshotRef] = []
    for record in read_records(checkpoint_path):
        if record.get("type") != "start":
            continue
        sha = str(record.get("snapshot", ""))
        kind = str(record.get("snapshot_kind", ""))
        if kind not in {_KIND_STASH, _KIND_CLEAN} or not sha:
            continue
        refs.append(
            SnapshotRef(
                seq=int(record.get("seq", 0)),
                tool=str(record.get("tool", "")),
                sha=sha,
                kind=kind,
                workdir=str(record.get("snapshot_workdir", "")),
                ts=float(record.get("ts", 0.0)),
            )
        )
    refs.sort(key=lambda ref: ref.seq)
    return refs


def rollback_snapshots(
    session_id: str,
    *,
    session_dir: Path | str | None,
    to_seq: int | None = None,
    force: bool = False,
) -> SnapshotRef:
    """把工作区恢复到指定快照；默认取该会话最后一个快照。"""
    from miniagent.session import Session, default_session_dir

    base = Path(session_dir).expanduser() if session_dir else default_session_dir()
    session = Session.load(session_id, base)
    checkpoint_path = default_checkpoint_path(session.path)
    refs = list_snapshots(checkpoint_path)
    if not refs:
        raise ConfigError(
            f"会话 {session_id} 没有可回滚的 git 快照"
            f"（checkpoint：{checkpoint_path}）；未跟踪文件也不会出现在快照里。"
        )
    if to_seq is not None:
        matches = [ref for ref in refs if ref.seq == to_seq]
        if not matches:
            raise ConfigError(
                f"会话 {session_id} 的 checkpoint 中没有 seq={to_seq} 的快照；"
                f"可用 seq：{', '.join(str(ref.seq) for ref in refs)}"
            )
        target = matches[-1]
    else:
        target = refs[-1]
    restore_snapshot(target, force=force)
    return target


def restore_snapshot(ref: SnapshotRef, *, force: bool = False) -> None:
    workdir = Path(ref.workdir).expanduser() if ref.workdir else Path.cwd()
    if not workdir.is_dir():
        raise ConfigError(f"快照记录的工作目录不存在：{workdir}")

    inside = _git_at(workdir, "rev-parse", "--is-inside-work-tree")
    if inside is None or inside.returncode != 0 or inside.stdout.strip() != "true":
        raise ConfigError(f"工作目录不是 git 仓库，无法回滚：{workdir}")

    if not force:
        status = _git_at(workdir, "status", "--porcelain")
        if status is None:
            raise ConfigError("git status 超时，无法确认工作区是否干净；确认后可用 --force")
        changed = [line for line in status.stdout.splitlines() if line.strip()]
        if changed:
            raise ConfigError(
                f"工作区有 {len(changed)} 处未提交改动（含未跟踪文件），"
                "回滚会覆盖或遗漏它们；确认要丢弃后加 --force。"
            )

    restore = _git_at(workdir, "restore", f"--source={ref.sha}", "--worktree", "--", ".")
    if restore is None:
        raise ConfigError("git restore 超时，回滚未完成")
    if restore.returncode != 0:
        raise ConfigError(
            f"git restore 失败：{(restore.stderr or restore.stdout).strip() or '未知错误'}"
        )


def _git_at(workdir: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    env = dict(os.environ)
    env.setdefault("GIT_OPTIONAL_LOCKS", "0")
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=SNAPSHOT_TIMEOUT,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
