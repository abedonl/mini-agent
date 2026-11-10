"""工具契约与结果结构。

工具只做一件事：拿到已校验的参数字典，返回字符串；需要给 trace 补充
元数据（如 HTTP 状态码、字节数）时返回 ToolOutput。
不依赖 Agent，因此可以独立测试与替换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from miniagent.hitl import HitlInfo

MAX_OUTPUT_CHARS = 8000

# 断点恢复时的重放策略：只读可重跑；有幂等键的先判定再决定；其余一律不重放。
REPLAY_SAFE = "safe"
REPLAY_IDEMPOTENT = "idempotent"
REPLAY_NEVER = "never"


def truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """统一的工具输出截断口径：保留开头并注明总长度。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[已截断，共 {len(text)} 字符]"


@dataclass(frozen=True)
class ToolOutput:
    """带 trace 元数据的工具输出；trace_detail 只放元信息，不放正文。"""

    text: str
    detail: str = ""
    fail: bool = False


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 子集
    read_only: bool  # 为后续"只读并发、写串行"预留

    def run(self, args: dict[str, Any]) -> str | ToolOutput: ...


def replay_policy(tool: Tool) -> str:
    """工具没声明时按保守口径推导：只读 => safe，其他 => never。"""
    policy = getattr(tool, "replay", "")
    if policy in {REPLAY_SAFE, REPLAY_IDEMPOTENT, REPLAY_NEVER}:
        return policy
    return REPLAY_SAFE if getattr(tool, "read_only", False) else REPLAY_NEVER


def idempotency_key(tool: Tool, args: dict[str, Any]) -> str:
    """写文件类工具的幂等键（内容摘要 + 路径）；没有实现则返回空串。"""
    key = getattr(tool, "idempotency_key", None)
    if not callable(key):
        return ""
    try:
        return str(key(args) or "")
    except Exception:  # noqa: BLE001 - 取不到键就按"无法判定"处理
        return ""


def effect_applied(tool: Tool, args: dict[str, Any]) -> bool | None:
    """幂等探针：True 已生效 / False 未生效 / None 无法判定。"""
    probe = getattr(tool, "effect_applied", None)
    if not callable(probe):
        return None
    try:
        return bool(probe(args))
    except Exception:  # noqa: BLE001 - 探测失败不能当成"已生效"
        return None


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False
    duration_ms: int = 0
    # 只有 HITL 真正介入（确认 / 补充参数 / 权限拦截）时才有值
    hitl: HitlInfo | None = None
    # 工具自报的元信息（查询词、状态码、字节数），写入 tool trace 的 detail
    trace_detail: str = field(default="")
