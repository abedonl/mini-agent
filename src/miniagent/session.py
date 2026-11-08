"""会话持久化：一条消息一行 JSONL，追加写入，可跨进程恢复。"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from miniagent.errors import ConfigError
from miniagent.llm import Message

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def default_session_dir() -> Path:
    env = os.environ.get("MINIAGENT_SESSION_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".miniagent" / "sessions"


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    path: Path
    message_count: int
    updated_at: float


class Session:
    """一个会话对应一个 JSONL 文件；每次 append 立即落盘，进程中断不丢历史。"""

    def __init__(self, session_id: str, path: Path, messages: list[Message]) -> None:
        self.session_id = session_id
        self.path = path
        self.messages = messages

    @classmethod
    def create(cls, session_id: str | None = None, directory: Path | str | None = None) -> Session:
        base = Path(directory).expanduser() if directory else default_session_dir()
        sid = session_id or new_session_id()
        if not _SESSION_ID.fullmatch(sid):
            raise ConfigError(f"非法的会话 ID：{sid!r}（只允许字母、数字、下划线、短横线）")
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{sid}.jsonl"
        path.touch(exist_ok=True)
        return cls(sid, path, [])

    @classmethod
    def load(cls, session_id: str, directory: Path | str | None = None) -> Session:
        base = Path(directory).expanduser() if directory else default_session_dir()
        if not _SESSION_ID.fullmatch(session_id):
            raise ConfigError(f"非法的会话 ID：{session_id!r}")
        path = base / f"{session_id}.jsonl"
        if not path.is_file():
            raise ConfigError(f"会话不存在：{session_id}（目录：{base}）")

        messages: list[Message] = []
        with path.open(encoding="utf-8") as fp:
            for lineno, line in enumerate(fp, start=1):
                if not line.strip():
                    continue
                try:
                    messages.append(Message.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ConfigError(f"会话文件 {path.name} 第 {lineno} 行损坏：{exc}") from exc
        if not messages:
            raise ConfigError(f"会话文件为空：{path}")
        return cls(session_id, path, messages)

    @classmethod
    def list(cls, directory: Path | str | None = None, limit: int = 10) -> list[SessionInfo]:
        base = Path(directory).expanduser() if directory else default_session_dir()
        if not base.is_dir():
            return []
        infos = []
        for path in base.glob("*.jsonl"):
            stat = path.stat()
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            infos.append(SessionInfo(path.stem, path, count, stat.st_mtime))
        infos.sort(key=lambda info: info.updated_at, reverse=True)
        return infos[:limit]

    def append(self, message: Message) -> None:
        line = json.dumps(message.to_dict(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
        self.messages.append(message)
