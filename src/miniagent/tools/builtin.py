"""内置工具：读文件、写文件、执行命令。

安全边界由 security.py 提供：文件工具可挂路径白名单，
命令工具执行前过一遍危险命令黑名单。
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from miniagent.errors import ToolError
from miniagent.security import PathPolicy, check_command
from miniagent.tools.base import (
    MAX_OUTPUT_CHARS,
    REPLAY_IDEMPOTENT,
    REPLAY_NEVER,
    truncate_output,
)

__all__ = [
    "MAX_OUTPUT_CHARS",
    "ReadFileTool",
    "RunShellTool",
    "WriteFileTool",
    "builtin_tools",
]

MAX_WRITE_BYTES = 1024 * 1024  # 单次写入上限 1 MiB，避免误写巨大文件


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    return truncate_output(text, limit)


class ReadFileTool:
    name = "read_file"
    description = "读取文本文件内容。查看代码、配置、日志时使用；路径支持绝对路径与 ~。"
    read_only = True
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "max_chars": {
                "type": "integer",
                "description": "最多返回的字符数，默认 20000",
            },
        },
        "required": ["path"],
    }

    def __init__(self, policy: PathPolicy | None = None) -> None:
        self._policy = policy

    def run(self, args: dict[str, Any]) -> str:
        path = self._resolve(args["path"])
        if not path.is_file():
            raise ToolError(f"文件不存在或不是普通文件：{path}")
        limit = int(args.get("max_chars", 20000))
        text = path.read_text(encoding="utf-8", errors="replace")
        return _truncate(text, limit)

    def precheck(self, args: dict[str, Any]) -> None:
        """执行前先过白名单，避免先让用户确认、再被权限边界拒绝。"""
        if isinstance(args.get("path"), str):
            self._resolve(args["path"])

    def _resolve(self, raw: str) -> Path:
        if self._policy is not None:
            return self._policy.resolve(raw)
        return Path(raw).expanduser()


class WriteFileTool:
    name = "write_file"
    description = "写入文本文件，覆盖原内容；父目录不存在会自动创建。"
    read_only = False
    # 内容摘要 + 路径构成幂等键：恢复时可判定"上次是否已经写成功"，不必盲重试。
    replay = REPLAY_IDEMPOTENT
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的完整内容"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, policy: PathPolicy | None = None) -> None:
        self._policy = policy

    def run(self, args: dict[str, Any]) -> str:
        path = self._write_path(args["path"])
        content = str(args["content"])
        size = len(content.encode("utf-8"))
        if size > MAX_WRITE_BYTES:
            raise ToolError(f"写入内容过大：{size} 字节，超过上限 {MAX_WRITE_BYTES} 字节")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"已写入 {path}（{size} 字节）"

    def idempotency_key(self, args: dict[str, Any]) -> str:
        path = self._write_path(args["path"])
        digest = hashlib.sha256(str(args["content"]).encode("utf-8")).hexdigest()
        return "sha256:" + hashlib.sha256(f"{path}\0{digest}".encode()).hexdigest()

    def effect_applied(self, args: dict[str, Any]) -> bool:
        """目标文件内容与本次写入完全一致才算已生效；不存在或不一致都不算。"""
        path = self._write_path(args["path"])
        if not path.is_file():
            return False
        return path.read_text(encoding="utf-8", errors="replace") == str(args["content"])

    def precheck(self, args: dict[str, Any]) -> None:
        if isinstance(args.get("path"), str):
            self._write_path(args["path"])

    def _write_path(self, raw: str) -> Path:
        return (
            self._policy.resolve(raw, for_write=True)
            if self._policy is not None
            else Path(raw).expanduser()
        )


class RunShellTool:
    name = "run_shell"
    description = "在用户机器上执行一条 shell 命令并返回退出码与输出。执行类操作，谨慎使用。"
    read_only = False
    # 命令没有可靠的幂等键：中断后可能已经执行过，也可能没有，只能标记状态未知。
    replay = REPLAY_NEVER
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令"},
            "timeout_seconds": {
                "type": "integer",
                "description": "超时秒数，默认 30，最大 120",
            },
        },
        "required": ["command"],
    }

    def __init__(self, policy: PathPolicy | None = None) -> None:
        # 命令在会话工作目录里执行，保证界面显示的目录与实际执行目录一致
        self._cwd = policy.workdir if policy is not None else None

    def run(self, args: dict[str, Any]) -> str:
        command = str(args["command"])
        check_command(command)
        timeout = max(1, min(int(args.get("timeout_seconds", 30)), 120))
        argv = self._shell_argv(command)
        try:
            completed = subprocess.run(  # noqa: S603 - 这是 agent 的通用执行工具
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                cwd=self._cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"命令超时（{timeout}s）：{command}") from exc

        parts = [f"exit_code: {completed.returncode}"]
        if completed.stdout.strip():
            parts.append("stdout:\n" + completed.stdout.rstrip())
        if completed.stderr.strip():
            parts.append("stderr:\n" + completed.stderr.rstrip())
        return _truncate("\n".join(parts))

    def precheck(self, args: dict[str, Any]) -> None:
        if isinstance(args.get("command"), str):
            check_command(args["command"])

    @staticmethod
    def _shell_argv(command: str) -> list[str]:
        if sys.platform == "win32":
            return [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ]
        shell = os.environ.get("SHELL", "/bin/bash")
        return [shell, "-c", command]


def builtin_tools(policy: PathPolicy | None = None) -> list[Any]:
    return [ReadFileTool(policy), WriteFileTool(policy), RunShellTool(policy)]
