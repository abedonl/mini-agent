"""工具注册表：登记、导出 Schema、校验参数、统一执行与错误兜底。"""

from __future__ import annotations

import json
import time
from typing import Any

from miniagent.errors import ToolError
from miniagent.llm import ToolCall
from miniagent.tools.base import Tool, ToolResult

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

    def execute(self, call: ToolCall) -> ToolResult:
        """执行一次工具调用。任何失败都转成错误文本，交给模型自行修正。"""
        start = time.perf_counter()
        try:
            content = self._execute(call)
            is_error = False
        except Exception as exc:  # noqa: BLE001 - 工具错误必须回传而不是终止进程
            content = f"工具执行失败：{type(exc).__name__}: {exc}"
            is_error = True
        duration_ms = int((time.perf_counter() - start) * 1000)
        return ToolResult(call.id, content, is_error, duration_ms)

    def _execute(self, call: ToolCall) -> str:
        tool = self.get(call.name)
        if tool is None:
            raise ToolError(f"未知工具：{call.name}，可用工具：{', '.join(self.names())}")

        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolError(f"参数不是合法 JSON：{exc}") from exc

        errors = validate_args(tool.parameters, args)
        if errors:
            raise ToolError("参数校验失败：" + "；".join(errors))

        return tool.run(args)
