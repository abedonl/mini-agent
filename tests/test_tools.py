"""工具层：注册、校验、执行与内置工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from miniagent.llm import ToolCall
from miniagent.tools import ToolRegistry, validate_args
from miniagent.tools.builtin import ReadFileTool, RunShellTool, WriteFileTool


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "mode": {"type": "string", "enum": ["fast", "safe"]},
            "count": {"type": "integer"},
        },
        "required": ["path"],
    }


def test_validate_ok() -> None:
    assert validate_args(schema(), {"path": "a", "mode": "fast", "count": 2}) == []


def test_validate_reports_all_problems() -> None:
    errors = validate_args(schema(), {"mode": "wild", "count": True, "extra": 1})
    joined = "；".join(errors)
    assert "缺少必需参数：path" in joined
    assert "参数 mode 取值必须是" in joined
    assert "参数 count 类型应为 integer" in joined
    assert "未知参数：extra" in joined


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    with pytest.raises(Exception, match="重复注册"):
        registry.register(ReadFileTool())


def test_registry_specs_are_openai_shaped() -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    spec = registry.specs()[0]

    assert spec["name"] == "read_file"
    assert spec["parameters"]["required"] == ["path"]


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    registry.register(ReadFileTool())
    target = tmp_path / "sub" / "note.txt"

    write_result = registry.execute(
        ToolCall(
            id="1",
            name="write_file",
            arguments=f'{{"path": "{target.as_posix()}", "content": "第一行\\n第二行"}}',
        )
    )
    read_result = registry.execute(
        ToolCall(id="2", name="read_file", arguments=f'{{"path": "{target.as_posix()}"}}')
    )

    assert write_result.is_error is False
    assert "已写入" in write_result.content
    assert read_result.content == "第一行\n第二行"


def test_read_missing_file_is_error_result(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    result = registry.execute(
        ToolCall(
            id="1",
            name="read_file",
            arguments=f'{{"path": "{(tmp_path / "nope.txt").as_posix()}"}}',
        )
    )

    assert result.is_error is True
    assert "文件不存在" in result.content


def test_read_truncates() -> None:
    tool = ReadFileTool()
    assert len(tool.run({"path": __file__, "max_chars": 20})) <= 60


def test_run_shell_returns_exit_code_and_output() -> None:
    registry = ToolRegistry()
    registry.register(RunShellTool())

    result = registry.execute(
        ToolCall(id="1", name="run_shell", arguments='{"command": "echo miniagent-ok"}')
    )

    assert result.is_error is False
    assert "exit_code: 0" in result.content
    assert "miniagent-ok" in result.content
