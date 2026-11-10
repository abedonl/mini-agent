"""工具注册表：登记、导出 Schema、校验参数、统一执行与错误兜底。"""

from __future__ import annotations

import json
import time
from typing import Any

from miniagent.errors import PermissionDenied, ToolError
from miniagent.hitl import HitlController, HitlInfo, ToolRequest
from miniagent.llm import ToolCall
from miniagent.snapshot import SnapshotError
from miniagent.tools.base import (
    Tool,
    ToolOutput,
    ToolResult,
    idempotency_key,
    replay_policy,
)

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """轻量 JSON Schema 校验：required / type / enum / 未知字段。

    只覆盖工具参数真正用得到的部分，不引第三方校验库。
    """
    errors: list[str] = []
    if not isinstance(args, dict):
        return ["参数必须是 JSON 对象"]

    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"缺少必需参数：{key}")

    for key, value in args.items():
        spec = properties.get(key)
        if spec is None:
            errors.append(f"未知参数：{key}")
            continue
        expected = spec.get("type")
        if expected in _TYPE_MAP:
            python_type = _TYPE_MAP[expected]
            if isinstance(value, bool) and expected in {"integer", "number"}:
                errors.append(f"参数 {key} 类型应为 {expected}，实际为 boolean")
            elif not isinstance(value, python_type):
                errors.append(f"参数 {key} 类型应为 {expected}，实际为 {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"参数 {key} 取值必须是 {spec['enum']} 之一")
    return errors


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # HITL 的唯一挂载点：默认 None，行为与引入拦截前一致
        self.approver: HitlController | None = None
        # 断点记录的唯一挂载点：默认 None，库调用 / 无会话时零额外写入
        self.checkpoint: object | None = None
        # 高危操作前的 git 快照：默认 None，只读工具永远不会走到它
        self.snapshotter: object | None = None

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ToolError(f"工具名重复注册：{tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def is_read_only(self, name: str) -> bool:
        """未知工具按“有副作用”处理，宁可串行也不并发。"""
        tool = self._tools.get(name)
        return bool(tool is not None and getattr(tool, "read_only", False))

    def specs(self) -> list[dict[str, Any]]:
        """导出为 OpenAI function calling 需要的工具描述。"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def execute(self, call: ToolCall, turn: int = 0) -> ToolResult:
        """执行一次工具调用。任何失败都转成错误文本，交给模型自行修正。

        HITL 就挂在这里：确认、补充参数、权限提示都只实现一份，
        ReAct 与 Plan-and-Execute 都经过同一个入口。checkpoint 也挂在这里，
        但写在确认之后、执行之前——确认前不会留下"已开始"的记录。
        """
        start = time.perf_counter()
        info: HitlInfo | None = None
        detail = ""
        failed = False
        try:
            content, info, detail, failed = self._execute(call, turn)
            is_error = failed or bool(info is not None and info.blocked)
        except PermissionDenied as exc:
            content = _permission_text(exc)
            is_error = True
            info = self._blocked_info()
        except Exception as exc:  # noqa: BLE001 - 工具错误必须回传而不是终止进程
            content = f"工具执行失败：{type(exc).__name__}: {exc}"
            is_error = True
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            call.id, content, is_error, duration_ms, hitl=info, trace_detail=detail
        )

    def _blocked_info(self) -> HitlInfo | None:
        """命中硬边界时也记一条 hitl；关闭 HITL 时保持原有 trace 形态。"""
        if self.approver is None:
            return None
        return HitlInfo(
            "blocked",
            reason="permission",
            risk="permission",
            mode=self.approver.mode,
            phase="permission",
        )

    def _execute(self, call: ToolCall, turn: int) -> tuple[str, HitlInfo | None, str, bool]:
        tool = self.get(call.name)
        if tool is None:
            raise ToolError(f"未知工具：{call.name}，可用工具：{', '.join(self.names())}")

        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolError(f"参数不是合法 JSON：{exc}") from exc

        errors = validate_args(tool.parameters, args)
        info: HitlInfo | None = None
        if errors:
            missing = _missing_keys(errors)
            if self.approver is not None and missing:
                info = self.approver.review(
                    ToolRequest(
                        call.name,
                        args,
                        bool(getattr(tool, "read_only", False)),
                        phase="missing",
                        missing=tuple(missing),
                    )
                )
                if info.decision == "skipped":
                    raise ToolError("参数校验失败：" + "；".join(errors))
                if info.blocked:
                    return _missing_text(call.name, missing, info), info, "", False
                # 只补字符串参数：其余类型（整数/布尔）不适合直接用一行文本顶替
                properties = tool.parameters.get("properties", {})
                for key in missing:
                    if properties.get(key, {}).get("type", "string") == "string":
                        args[key] = info.text.strip()
                errors = validate_args(tool.parameters, args)
            if errors:
                raise ToolError("参数校验失败：" + "；".join(errors))

        # 先过安全边界再问用户，避免确认了一个注定会被拒绝的操作
        precheck = getattr(tool, "precheck", None)
        if callable(precheck):
            precheck(args)

        if self.approver is not None:
            confirm = self.approver.review(
                ToolRequest(
                    call.name,
                    args,
                    bool(getattr(tool, "read_only", False)),
                )
            )
            if confirm.blocked:
                return _denied_text(call.name, confirm), confirm, "", False
            if confirm.decision != "skipped":
                # 只有确认阶段真正命中规则时才覆盖；否则保留"补充参数"那条记录
                info = confirm

        # 确认之后、执行之前：先拍 git 快照，再落 start。
        # 快照失败就拒绝执行，不带着"无保护状态"继续跑高危操作。
        snapshot = self._capture_snapshot(call, bool(getattr(tool, "read_only", False)))
        seq = self._checkpoint_begin(call, args, tool, turn, snapshot)
        started = time.perf_counter()
        try:
            output = tool.run(args)
        except Exception as exc:  # noqa: BLE001 - 先落完成记录，再原样抛出给上层兜底
            self._checkpoint_end(
                seq,
                turn=turn,
                call_id=call.id,
                tool=call.name,
                content=f"工具执行失败：{type(exc).__name__}: {exc}",
                is_error=True,
                duration_ms=_elapsed_ms(started),
            )
            raise
        duration_ms = _elapsed_ms(started)
        if isinstance(output, ToolOutput):
            self._checkpoint_end(
                seq,
                turn=turn,
                call_id=call.id,
                tool=call.name,
                content=output.text,
                is_error=output.fail,
                duration_ms=duration_ms,
            )
            return output.text, info, output.detail, output.fail
        self._checkpoint_end(
            seq,
            turn=turn,
            call_id=call.id,
            tool=call.name,
            content=output,
            is_error=False,
            duration_ms=duration_ms,
        )
        return output, info, "", False

    def _capture_snapshot(self, call: ToolCall, read_only: bool) -> object | None:
        snapshotter = self.snapshotter
        if read_only or snapshotter is None:
            return None
        try:
            return snapshotter.capture()  # type: ignore[attr-defined]
        except SnapshotError as exc:
            raise ToolError(
                f"快照失败，已拒绝执行高危工具 {call.name}：{exc}"
                "（不希望快照时可显式使用 --snapshot off）"
            ) from exc

    def _checkpoint_begin(
        self,
        call: ToolCall,
        args: dict,
        tool: Tool,
        turn: int,
        snapshot: object | None,
    ) -> int | None:
        checkpoint = self.checkpoint
        if checkpoint is None:
            return None
        sha = ""
        kind = ""
        workdir = ""
        if snapshot is not None:
            sha_value = getattr(snapshot, "sha", "")
            kind = getattr(snapshot, "kind", "")
            workdir = getattr(snapshot, "workdir", "")
            if kind in {"stash", "clean"} and sha_value:
                sha = str(sha_value)
        return checkpoint.begin(  # type: ignore[attr-defined]
            turn=turn,
            call_id=call.id,
            tool=call.name,
            args=args,
            read_only=bool(getattr(tool, "read_only", False)),
            replay=replay_policy(tool),
            idem_key=idempotency_key(tool, args),
            snapshot=sha,
            snapshot_kind=kind,
            snapshot_workdir=workdir,
        )

    def _checkpoint_end(
        self,
        seq: int | None,
        *,
        turn: int,
        call_id: str,
        tool: str,
        content: str,
        is_error: bool,
        duration_ms: int,
    ) -> None:
        if seq is None or self.checkpoint is None:
            return
        self.checkpoint.finish(  # type: ignore[attr-defined]
            seq,
            turn=turn,
            call_id=call_id,
            tool=tool,
            content=content,
            is_error=is_error,
            duration_ms=duration_ms,
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _missing_keys(errors: list[str]) -> list[str]:
    prefix = "缺少必需参数："
    return [error[len(prefix) :] for error in errors if error.startswith(prefix)]


def _denied_text(tool: str, info: HitlInfo) -> str:
    return (
        f"HITL 拦截：用户拒绝执行（工具 {tool}，风险 {info.risk or 'none'}，"
        f"原因 {info.reason or '-'}）。"
        "该操作没有执行，不要原样重试；请换方案、改用只读操作，或向用户说明需要授权。"
    )


def _missing_text(tool: str, missing: list[str], info: HitlInfo) -> str:
    names = "、".join(missing)
    return (
        f"HITL 拦截：用户未提供必要信息（工具 {tool} 缺少参数：{names}，"
        f"原因 {info.reason or '-'}）。"
        "不要猜测参数；请改用已知信息，或向用户说明需要补充什么。"
    )


def _permission_text(exc: PermissionDenied) -> str:
    hint = f"；需要放行：{exc.hint}" if exc.hint else ""
    return (
        f"权限边界拦截：{exc}{hint}。"
        "该边界不能由模型绕过，请改用允许范围内的方案，或向用户说明需要调整什么。"
    )
