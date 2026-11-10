"""Human-in-the-loop：工具调用前的统一拦截层。

只挂在 ``ToolRegistry.execute`` 上：ReAct 与 Plan-and-Execute 都经过这唯一入口，
所以确认、补充参数、权限提示三种情况只实现一份。

默认关闭（``hitl=off``）时不注册拦截器，行为与引入本模块前完全一致；
开启后按规则表决定哪些工具调用需要用户确认，规则可配置而不是写死在代码里。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

HITL_MODES = ("off", "on")

# 内置确认规则：名字用于配置，顺序即优先级（先命中先报）。
RULE_NAMES = ("overwrite", "mutate", "delete")
DEFAULT_CONFIRM_RULES = ("overwrite", "mutate", "delete")
RULE_DESCRIPTIONS = {
    "overwrite": "覆盖已有文件",
    "mutate": "执行非只读工具（含新写文件、执行命令）",
    "delete": "删除类命令",
}

# 删除类命令只做粗粒度识别，命中后仍交给用户判断；这里不替代 security 的黑名单。
_DELETE_RE = re.compile(
    r"(?:^|[;&|()\s])(?:rm|del|erase|rd|rmdir|remove-item|unlink)\b",
    re.I,
)


def _path_exists(raw: str) -> bool:
    try:
        return Path(raw).expanduser().exists()
    except OSError:
        return False


def classify(
    tool: str,
    args: dict[str, Any],
    read_only: bool,
    enabled: tuple[str, ...] | list[str] | set[str],
) -> tuple[str, str] | None:
    """返回命中的 (规则名, 人类可读原因)；没有需要确认的规则时返回 None。

    规则按 overwrite -> delete -> mutate 的顺序判断，命中第一个已启用的规则。
    """
    enabled_set = set(enabled)
    if "overwrite" in enabled_set and tool == "write_file":
        path = args.get("path")
        if isinstance(path, str) and _path_exists(path):
            return "overwrite", RULE_DESCRIPTIONS["overwrite"]
    if "delete" in enabled_set and tool == "run_shell":
        command = args.get("command")
        if isinstance(command, str) and _DELETE_RE.search(command):
            return "delete", RULE_DESCRIPTIONS["delete"]
    if "mutate" in enabled_set and not read_only:
        return "mutate", f"非只读工具 {tool}"
    return None


def summarize_args(args: dict[str, Any], limit: int = 120) -> str:
    """把参数压成一行提示，长内容只显示长度，避免终端被整段文件内容刷屏。"""
    parts: list[str] = []
    for key, value in args.items():
        text = str(value)
        if len(text) > limit:
            text = f"{text[:limit]}…(共 {len(text)} 字符)"
        parts.append(f"{key}={text!r}")
    return " ".join(parts) or "（无参数）"


@dataclass(frozen=True)
class ToolRequest:
    """一次需要人处理的情况：确认、补参数，或权限提示。"""

    tool: str
    args: dict[str, Any]
    read_only: bool
    phase: str = "confirm"  # confirm | missing
    risk: str = ""
    reason: str = ""
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class InteractionAnswer:
    """交互端返回的结果；text 在"信息不足"时作为用户补充的取值。"""

    decision: str  # allow | deny
    text: str = ""
    reason: str = ""


@dataclass(frozen=True)
class HitlInfo:
    """拦截结论；由注册表放进 ToolResult，Agent 统一写 trace 与事件。"""

    decision: str  # allow | deny | blocked
    reason: str = ""
    risk: str = ""
    wait_ms: int = 0
    mode: str = ""
    phase: str = "confirm"
    text: str = ""

    @property
    def blocked(self) -> bool:
        return self.decision in {"deny", "blocked"}


class InteractionHandler(Protocol):
    """CLI / 网页各自实现；交互式场景可以一直等，不设超时。"""

    name: str

    def ask(self, request: ToolRequest) -> InteractionAnswer: ...


class AutoApproveHandler:
    """自动放行：用于测试或明确知道无需人工介入的批处理。"""

    name = "auto"

    def ask(self, request: ToolRequest) -> InteractionAnswer:
        return InteractionAnswer("allow")


class AutoDenyHandler:
    """无法交互时一律拒绝：非交互模式与"没有处理器"的兜底。"""

    name = "noninteractive"

    def __init__(self, reason: str = "该操作需要确认，请在交互模式下运行") -> None:
        self.reason = reason

    def ask(self, request: ToolRequest) -> InteractionAnswer:
        return InteractionAnswer("deny", reason=self.reason)


class HitlController:
    """规则判定 + 交互调度的中间层。

    同一个控制器可能被并发工具线程调用（只读工具会并发），
    因此用锁把"提问"串行化，保证终端 / 网页同一时刻只弹一个确认。
    """

    def __init__(
        self,
        *,
        rules: tuple[str, ...] = DEFAULT_CONFIRM_RULES,
        handler: InteractionHandler,
        enabled: bool = True,
    ) -> None:
        self.rules = tuple(rules)
        self.handler = handler
        self.enabled = enabled
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return getattr(self.handler, "name", "unknown")

    def review(self, request: ToolRequest) -> HitlInfo:
        """返回 skipped 表示不需要人参与（调用方不应产生 hitl 记录）。"""
        if not self.enabled:
            return HitlInfo("skipped")
        if request.phase == "confirm":
            hit = classify(request.tool, request.args, request.read_only, self.rules)
            if hit is None:
                return HitlInfo("skipped")
            risk, reason = hit
        else:
            risk, reason = "", "缺少必需参数"

        prepared = ToolRequest(
            tool=request.tool,
            args=request.args,
            read_only=request.read_only,
            phase=request.phase,
            risk=risk,
            reason=reason,
            missing=request.missing,
        )
        started = time.perf_counter()
        try:
            with self._lock:
                answer = self.handler.ask(prepared)
        except Exception as exc:  # noqa: BLE001 - 交互端故障不能放行危险操作
            return HitlInfo(
                "deny",
                reason=f"handler_error:{type(exc).__name__}",
                risk=risk,
                wait_ms=_elapsed_ms(started),
                mode=self.mode,
                phase=prepared.phase,
            )
        wait_ms = _elapsed_ms(started)
        decision = answer.decision if answer.decision in {"allow", "deny"} else "deny"
        if decision == "allow" and prepared.phase == "missing" and not answer.text.strip():
            decision = "deny"
            answer = InteractionAnswer("deny", reason="empty_input")
        reason_code = answer.reason or decision
        if prepared.phase == "missing" and decision == "allow":
            reason_code = answer.reason or "user_supplied"
        return HitlInfo(
            decision=decision,
            reason=reason_code,
            risk=risk,
            wait_ms=wait_ms,
            mode=self.mode,
            phase=prepared.phase,
            text=answer.text,
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
