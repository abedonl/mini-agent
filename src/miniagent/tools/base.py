"""工具契约与结果结构。

工具只做一件事：拿到已校验的参数字典，返回字符串。
不依赖 Agent，因此可以独立测试与替换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 子集
    read_only: bool  # 为后续"只读并发、写串行"预留

    def run(self, args: dict[str, Any]) -> str: ...


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False
    duration_ms: int = 0
