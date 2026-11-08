"""可观测性：固定字段 trace 日志与会话回放。

只记录元数据（轮次、事件、工具、耗时、token 估算、错误码），
不记录对话正文、工具参数/输出与密钥，避免隐私泄露。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miniagent.errors import ConfigError

TRACE_FIELDS = (
    "ts",
    "session_id",
    "turn",
    "event",
    "tool",
    "duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "ok",
    "error_code",
)


def default_log_dir() -> Path:
    env = os.environ.get("MINIAGENT_LOG_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".miniagent" / "logs"


@dataclass
class TraceRecord:
    ts: float
    session_id: str
    turn: int
    event: str  # run_start | request | response | tool | trim | error | run_end
    tool: str = ""
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True
    error_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in TRACE_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceRecord:
        return cls(
            ts=float(data.get("ts", 0.0)),
            session_id=str(data.get("session_id", "")),
            turn=int(data.get("turn", 0)),
            event=str(data.get("event", "")),
            tool=str(data.get("tool", "")),
            duration_ms=int(data.get("duration_ms", 0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            ok=bool(data.get("ok", True)),
            error_code=str(data.get("error_code", "")),
        )


class TraceLogger:
    """所有会话追加写入同一个 trace.jsonl，回放时按 session_id 过滤。"""

    def __init__(self, session_id: str, directory: Path | str | None = None) -> None:
        self.session_id = session_id
        self.directory = Path(directory).expanduser() if directory else default_log_dir()
        self.path = self.directory / "trace.jsonl"

    def log(
        self,
        event: str,
        *,
        turn: int = 0,
        tool: str = "",
        duration_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ok: bool = True,
        error_code: str = "",
    ) -> TraceRecord:
        record = TraceRecord(
            ts=time.time(),
            session_id=self.session_id,
            turn=turn,
            event=event,
            tool=tool,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ok=ok,
            error_code=error_code,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    @classmethod
    def load(cls, session_id: str, directory: Path | str | None = None) -> list[TraceRecord]:
        base = Path(directory).expanduser() if directory else default_log_dir()
        path = base / "trace.jsonl"
        if not path.is_file():
            raise ConfigError(f"没有找到 trace 文件：{path}")

        records: list[TraceRecord] = []
        with path.open(encoding="utf-8") as fp:
            for lineno, line in enumerate(fp, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"trace 文件 {path.name} 第 {lineno} 行损坏：{exc}") from exc
                if data.get("session_id") == session_id:
                    records.append(TraceRecord.from_dict(data))

        if not records:
            raise ConfigError(f"会话 {session_id} 没有 trace 记录（{path}）")
        return records


def format_record(record: TraceRecord) -> str:
    """把一条 trace 压成终端里一行，便于逐轮排查。"""
    head = f"[turn {record.turn}] {record.event:<8}"
    if record.event == "tool":
        status = "ok" if record.ok else "error"
        tail = f"{record.tool} {status} {record.duration_ms}ms"
        if record.error_code:
            tail += f" code={record.error_code}"
        return f"{head} {tail}"
    if record.event in {"request", "trim", "run_start"}:
        return f"{head} prompt≈{record.prompt_tokens} tokens"
    if record.event == "response":
        return f"{head} {record.duration_ms}ms completion≈{record.completion_tokens} tokens"
    if record.event == "error":
        return f"{head} {record.error_code or 'error'} {record.duration_ms}ms"
    if record.event == "run_end":
        status = "ok" if record.ok else "failed"
        tail = f"{status} {record.duration_ms}ms"
        if record.error_code:
            tail += f" code={record.error_code}"
        return f"{head} {tail}"
    return f"{head} {record.tool} {record.ok}"
