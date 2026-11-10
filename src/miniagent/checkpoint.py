"""工具调用 checkpoint：进程被杀后能接着跑，而不是回到上一轮。

只记工具调用的元信息（轮次、调用 ID、工具名、参数指纹、状态、耗时、结果摘要），
不重复存对话正文；正文的唯一来源仍是会话 JSONL。恢复时需要的工具参数也从
会话里的 assistant.tool_calls 取，因此 checkpoint 本身不保存参数原文。

格式：一行一个 JSON 对象，每条记录都带版本号；追加写入，不重写历史。

    {"v":1,"type":"start","seq":1,"ts":...,"turn":1,"call_id":"c1",
     "tool":"read_file","args_fp":"sha256:...","read_only":true,
     "replay":"safe","idem_key":""}
    {"v":1,"type":"end","seq":1,"ts":...,"turn":1,"call_id":"c1",
     "tool":"read_file","status":"ok","duration_ms":3,"error_code":"",
     "summary":"...","recovered":false,"action":""}

读文件时任何坏行、未知版本或未知记录类型都会报错并指出行号，不静默跳过：
半截状态必须让人看见，否则恢复语义就不可信。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miniagent.errors import ConfigError

CHECKPOINT_VERSION = 1

# 恢复策略：safe = 只读可重跑；idempotent = 有幂等键，可判定是否已生效；
# never = 无法保证幂等（如 run_shell），只能标记状态未知。
REPLAY_SAFE = "safe"
REPLAY_IDEMPOTENT = "idempotent"
REPLAY_NEVER = "never"
REPLAY_POLICIES = (REPLAY_SAFE, REPLAY_IDEMPOTENT, REPLAY_NEVER)

SUMMARY_MAX_CHARS = 300


def default_checkpoint_path(session_path: Path) -> Path:
    """与会话文件同目录的 checkpoints 子目录；不放进 sessions 根目录。

    ``Session.list`` 只扫 ``sessions/*.jsonl``，放子目录可以避免把 checkpoint
    误当成会话列出来。
    """
    session_path = Path(session_path)
    return session_path.parent / "checkpoints" / f"{session_path.stem}.jsonl"


def fingerprint(args: dict[str, Any]) -> str:
    """参数指纹：排序后的紧凑 JSON 再取 SHA-256，便于人工核对同一次调用。"""
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def summarize(text: str, limit: int = SUMMARY_MAX_CHARS) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + f"…(共 {len(flat)} 字符)"


@dataclass(frozen=True)
class PendingCall:
    """一条只有 start、没有 end 的调用：上次进程在执行期间中断。"""

    seq: int
    turn: int
    call_id: str
    tool: str
    args_fp: str
    read_only: bool
    replay: str
    idem_key: str


@dataclass(frozen=True)
class FinishedCall:
    """一条 end 记录；summary 只够回填消息，不保存完整输出。"""

    seq: int
    turn: int
    call_id: str
    tool: str
    status: str
    duration_ms: int
    error_code: str
    summary: str
    recovered: bool = False
    action: str = ""


@dataclass(frozen=True)
class RecoveryResult:
    """一次恢复判定，供 CLI 展示与 trace 记录。"""

    call_id: str
    tool: str
    action: str  # rerun | completed | idempotent | unknown | message_present
    reason: str = ""


def read_records(path: Path | str) -> list[dict[str, Any]]:
    """严格读取 checkpoint；坏行 / 未知版本 / 未知类型都带行号报错。"""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return []

    records: list[dict[str, Any]] = []
    with checkpoint_path.open(encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"checkpoint 文件 {checkpoint_path.name} 第 {lineno} 行损坏：{exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ConfigError(
                    f"checkpoint 文件 {checkpoint_path.name} 第 {lineno} 行不是 JSON 对象"
                )
            version = data.get("v")
            if version != CHECKPOINT_VERSION:
                raise ConfigError(
                    f"checkpoint 文件 {checkpoint_path.name} 第 {lineno} 行版本不支持："
                    f"v={version!r}（本程序支持 v={CHECKPOINT_VERSION}）"
                )
            kind = data.get("type")
            if kind not in {"start", "end"}:
                raise ConfigError(
                    f"checkpoint 文件 {checkpoint_path.name} 第 {lineno} 行记录类型未知："
                    f"{kind!r}（可选 start / end）"
                )
            records.append(data)
    return records


class Checkpoint:
    """线程安全的追加写入器；只读工具会并发调用 begin/finish。"""

    def __init__(self, path: Path | str, tracer: Any | None = None) -> None:
        self.path = Path(path)
        self.tracer = tracer
        self._lock = threading.Lock()
        self._pending: dict[int, PendingCall] = {}
        self._finished: dict[int, FinishedCall] = {}
        self._next_seq = 1
        self._load()

    @classmethod
    def for_session(cls, session: Any, tracer: Any | None = None) -> Checkpoint:
        return cls(default_checkpoint_path(session.path), tracer=tracer)

    # ---- 正常执行路径 -------------------------------------------------------

    def begin(
        self,
        *,
        turn: int,
        call_id: str,
        tool: str,
        args: dict[str, Any],
        read_only: bool,
        replay: str,
        idem_key: str = "",
        snapshot: str = "",
        snapshot_kind: str = "",
        snapshot_workdir: str = "",
    ) -> int:
        """在执行前落一条 start；返回 seq，供 finish 引用。

        snapshot 是高危操作执行前的 git 快照 sha（只创建对象，不改仓库状态）；
        只读操作与关闭快照时为空。
        """
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            record = {
                "v": CHECKPOINT_VERSION,
                "type": "start",
                "seq": seq,
                "ts": time.time(),
                "turn": turn,
                "call_id": call_id,
                "tool": tool,
                "args_fp": fingerprint(args),
                "read_only": read_only,
                "replay": replay,
                "idem_key": idem_key,
                "snapshot": snapshot,
                "snapshot_kind": snapshot_kind,
                "snapshot_workdir": snapshot_workdir,
            }
            self._append(record)
            self._pending[seq] = PendingCall(
                seq=seq,
                turn=turn,
                call_id=call_id,
                tool=tool,
                args_fp=record["args_fp"],
                read_only=read_only,
                replay=replay,
                idem_key=idem_key,
            )
        self._trace(
            turn=turn,
            tool=tool,
            detail=(
                f"action=begin seq={seq} call={call_id} tool={tool} "
                f"replay={replay} read_only={int(read_only)}"
            ),
        )
        return seq

    def finish(
        self,
        seq: int,
        *,
        turn: int,
        call_id: str,
        tool: str,
        content: str,
        is_error: bool,
        duration_ms: int,
    ) -> None:
        end = self._end_record(
            seq=seq,
            turn=turn,
            call_id=call_id,
            tool=tool,
            status="error" if is_error else "ok",
            duration_ms=duration_ms,
            error_code="tool_error" if is_error else "",
            summary=summarize(content),
        )
        with self._lock:
            self._append(end)
            self._pending.pop(seq, None)
            self._finished[seq] = _finished_from(end)
        self._trace(
            turn=turn,
            tool=tool,
            detail=(
                f"action=end seq={seq} call={call_id} status={end['status']} "
                f"duration_ms={duration_ms}"
            ),
        )

    # ---- 恢复路径 -----------------------------------------------------------

    def pending(self) -> list[PendingCall]:
        with self._lock:
            return [self._pending[seq] for seq in sorted(self._pending)]

    def finished(self) -> list[FinishedCall]:
        with self._lock:
            return [self._finished[seq] for seq in sorted(self._finished)]

    def resolve(
        self,
        pending: PendingCall,
        *,
        action: str,
        status: str = "unknown",
        summary: str = "",
        duration_ms: int = 0,
        error_code: str = "",
    ) -> None:
        """给悬空 start 补一条 end，标记恢复时的判定，避免下次重复处理。"""
        end = self._end_record(
            seq=pending.seq,
            turn=pending.turn,
            call_id=pending.call_id,
            tool=pending.tool,
            status=status,
            duration_ms=duration_ms,
            error_code=error_code,
            summary=summary,
            recovered=True,
            action=action,
        )
        with self._lock:
            self._append(end)
            self._pending.pop(pending.seq, None)
            self._finished[pending.seq] = _finished_from(end)
        self._trace(
            turn=pending.turn,
            tool=pending.tool,
            detail=(
                f"action=resolve seq={pending.seq} call={pending.call_id} "
                f"result={action} status={status}"
            ),
        )

    # ---- 内部 ---------------------------------------------------------------

    def _load(self) -> None:
        starts: dict[int, dict[str, Any]] = {}
        for record in read_records(self.path):
            seq = int(record.get("seq", 0))
            if seq < 1:
                raise ConfigError(
                    f"checkpoint 文件 {self.path.name} 记录缺少合法 seq：{record!r}"
                )
            self._next_seq = max(self._next_seq, seq + 1)
            if record["type"] == "start":
                starts[seq] = record
                self._pending[seq] = PendingCall(
                    seq=seq,
                    turn=int(record.get("turn", 0)),
                    call_id=str(record.get("call_id", "")),
                    tool=str(record.get("tool", "")),
                    args_fp=str(record.get("args_fp", "")),
                    read_only=bool(record.get("read_only", False)),
                    replay=str(record.get("replay", REPLAY_NEVER)),
                    idem_key=str(record.get("idem_key", "")),
                )
            else:
                if seq not in starts:
                    raise ConfigError(
                        f"checkpoint 文件 {self.path.name} 中 seq={seq} 的 end 记录"
                        "找不到对应的 start"
                    )
                self._pending.pop(seq, None)
                self._finished[seq] = _finished_from(record)

    def _end_record(
        self,
        *,
        seq: int,
        turn: int,
        call_id: str,
        tool: str,
        status: str,
        duration_ms: int,
        error_code: str,
        summary: str,
        recovered: bool = False,
        action: str = "",
    ) -> dict[str, Any]:
        return {
            "v": CHECKPOINT_VERSION,
            "type": "end",
            "seq": seq,
            "ts": time.time(),
            "turn": turn,
            "call_id": call_id,
            "tool": tool,
            "status": status,
            "duration_ms": duration_ms,
            "error_code": error_code,
            "summary": summary,
            "recovered": recovered,
            "action": action,
        }

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _trace(self, *, turn: int, tool: str, detail: str) -> None:
        if self.tracer is not None:
            self.tracer.log("checkpoint", turn=turn, tool=tool, detail=detail)


def _finished_from(record: dict[str, Any]) -> FinishedCall:
    return FinishedCall(
        seq=int(record.get("seq", 0)),
        turn=int(record.get("turn", 0)),
        call_id=str(record.get("call_id", "")),
        tool=str(record.get("tool", "")),
        status=str(record.get("status", "unknown")),
        duration_ms=int(record.get("duration_ms", 0)),
        error_code=str(record.get("error_code", "")),
        summary=str(record.get("summary", "")),
        recovered=bool(record.get("recovered", False)),
        action=str(record.get("action", "")),
    )
