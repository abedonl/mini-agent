"""工具调度：同一轮内按路径依赖决定并发还是串行。

规则（成本很低，只做静态判断）：
- 读-读之间永远可以并发。
- 读与写路径重叠时降级串行；重叠包含同路径与父子目录。
- 写与降级读都按原调用顺序执行，因此“同轮先写后读”不会被并发打乱。
- run_shell 或无 path 参数的写无法静态判断影响范围，视为全路径屏障：整轮串行。

路径比较前统一按当前工作目录解析为绝对路径，并做真实路径与大小写归一化，
避免同一个文件因相对/绝对、大小写或符号链接被漏判。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from miniagent.llm import ToolCall

BARRIER_TOOLS = {"run_shell"}


@dataclass
class CallPlan:
    concurrent: list[int] = field(default_factory=list)
    serial: list[int] = field(default_factory=list)
    downgraded: int = 0  # 本可并发、因路径重叠或屏障改为串行的只读调用数
    reasons: tuple[str, ...] = ()

    @property
    def is_barrier(self) -> bool:
        """run_shell / 无 path 写：路径规则已判不了，上层应整轮串行并跳过模型判定。"""
        return any(reason in {"shell_barrier", "write_barrier"} for reason in self.reasons)

    @property
    def detail(self) -> str:
        reasons = "+".join(self.reasons) or "-"
        return (
            f"concurrent={len(self.concurrent)} serial={len(self.serial)} "
            f"downgraded={self.downgraded} reason={reasons}"
        )


def plan_calls(calls: list[ToolCall], is_read_only: Callable[[str], bool]) -> CallPlan:
    """返回并发集合、串行集合与降级数量；索引始终对应原调用位置。"""
    total = len(calls)
    read_set = {index for index, call in enumerate(calls) if is_read_only(call.name)}
    reads = sorted(read_set)
    writes = [index for index in range(total) if index not in read_set]

    if not writes:
        return CallPlan(concurrent=list(range(total)))

    barrier = ""
    scopes: list[str] = []
    for index in writes:
        call = calls[index]
        path = _path_arg(call)
        if call.name in BARRIER_TOOLS:
            barrier = "shell_barrier"
            break
        if not path:
            barrier = "write_barrier"
            break
        scopes.append(normalize_path(path))

    if barrier:
        # 无法界定影响范围时整轮串行，避免读与不可预测的副作用竞争
        return CallPlan(
            serial=list(range(total)),
            downgraded=len(reads),
            reasons=(barrier,),
        )

    concurrent: list[int] = []
    serial_reads: list[int] = []
    reasons: set[str] = set()
    for index in reads:
        raw = _path_arg(calls[index])
        if not raw:
            serial_reads.append(index)
            reasons.add("unknown_read_scope")
            continue
        path = normalize_path(raw)
        if any(paths_overlap(path, scope) for scope in scopes):
            serial_reads.append(index)
            reasons.add("path_overlap")
        else:
            concurrent.append(index)

    return CallPlan(
        concurrent=concurrent,
        serial=sorted([*writes, *serial_reads]),
        downgraded=len(serial_reads),
        reasons=tuple(sorted(reasons)),
    )


def normalize_path(raw: str) -> str:
    """相对路径按 cwd 绝对化，解析符号链接，Windows 下统一大小写与分隔符。"""
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path.absolute()
    text = os.path.normcase(str(resolved))
    # Windows 解析符号链接后可能带扩展路径前缀，去掉后才能与普通绝对路径相等
    if text.startswith("\\\\?\\unc\\"):
        text = "\\\\" + text[len("\\\\?\\unc\\") :]
    elif text.startswith(("\\\\?\\", "\\\\.\\")):
        text = text[4:]
    return text.rstrip("\\/") or text


def paths_overlap(left: str, right: str) -> bool:
    """同路径或互为父子目录都算重叠；入参须已 normalize。"""
    if not left or not right:
        return False
    return _covers(left, right) or _covers(right, left)


def _covers(parent: str, child: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("\\/") + os.sep)


def _path_arg(call: ToolCall) -> str:
    try:
        data = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return ""
    value = data.get("path", "") if isinstance(data, dict) else ""
    return str(value) if value else ""
