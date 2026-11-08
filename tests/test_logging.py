"""M4 可观测性：固定字段 trace、脱敏、失败记录与 CLI 回放。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from miniagent.agent import Agent
from miniagent.cli import main
from miniagent.errors import ProviderError
from miniagent.llm import LLMResponse, Message, MockProvider, ToolCall
from miniagent.logging import TRACE_FIELDS, TraceLogger
from miniagent.tools import ToolRegistry


class EchoTool:
    name = "echo"
    description = "回显文本"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, args: dict[str, Any]) -> str:
        return args["text"]


class FailingProvider:
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        on_text: Any = None,
    ) -> LLMResponse:
        raise ProviderError("upstream 503")


def make_agent(tracer: TraceLogger, script: list[LLMResponse], max_turns: int = 5) -> Agent:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return Agent(
        provider=MockProvider(script),
        registry=registry,
        max_turns=max_turns,
        tracer=tracer,
    )


def test_records_have_fixed_fields_and_no_content(tmp_path: Path) -> None:
    tracer = TraceLogger("s-secret", tmp_path)
    agent = make_agent(tracer, [LLMResponse(content="机密回答")])

    agent.run("我的密钥是 sk-test-secret")

    raw = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    first = json.loads(raw.splitlines()[0])
    assert tuple(first) == TRACE_FIELDS
    assert "sk-test-secret" not in raw
    assert "机密回答" not in raw


def test_agent_trace_covers_tool_and_final_answer(tmp_path: Path) -> None:
    tracer = TraceLogger("s-tool", tmp_path)
    agent = make_agent(
        tracer,
        [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')]),
            LLMResponse(content="完成"),
        ],
    )

    result = agent.run("测试")

    assert result.ok is True
    records = TraceLogger.load("s-tool", tmp_path)
    assert [record.event for record in records] == [
        "run_start",
        "request",
        "response",
        "tool",
        "request",
        "response",
        "run_end",
    ]
    tool = next(record for record in records if record.event == "tool")
    assert tool.tool == "echo"
    assert tool.ok is True
    assert tool.error_code == ""
    assert records[-1].ok is True


def test_provider_error_is_traced_and_reraised(tmp_path: Path) -> None:
    tracer = TraceLogger("s-fail", tmp_path)
    agent = Agent(provider=FailingProvider(), registry=ToolRegistry(), tracer=tracer)

    with pytest.raises(ProviderError):
        agent.run("hi")

    records = TraceLogger.load("s-fail", tmp_path)
    assert [record.event for record in records] == ["run_start", "request", "error"]
    assert records[-1].ok is False
    assert records[-1].error_code == "provider_error"


def test_tool_error_is_traced(tmp_path: Path) -> None:
    tracer = TraceLogger("s-tool-fail", tmp_path)
    agent = make_agent(
        tracer,
        [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="missing", arguments="{}")]),
            LLMResponse(content="换一个工具"),
        ],
    )

    agent.run("测试")

    records = TraceLogger.load("s-tool-fail", tmp_path)
    tool = next(record for record in records if record.event == "tool")
    assert tool.ok is False
    assert tool.error_code == "tool_error"


def test_max_turns_is_traced(tmp_path: Path) -> None:
    tracer = TraceLogger("s-max", tmp_path)
    looping = LLMResponse(
        tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "again"}')]
    )
    agent = make_agent(tracer, [looping, looping, looping], max_turns=2)

    result = agent.run("测试")

    assert result.ok is False
    records = TraceLogger.load("s-max", tmp_path)
    assert records[-1].event == "run_end"
    assert records[-1].ok is False
    assert records[-1].error_code == "max_turns"


def test_load_filters_by_session(tmp_path: Path) -> None:
    TraceLogger("a", tmp_path).log("run_start")
    TraceLogger("b", tmp_path).log("run_start")
    TraceLogger("a", tmp_path).log("run_end")

    assert [record.event for record in TraceLogger.load("a", tmp_path)] == [
        "run_start",
        "run_end",
    ]


def test_trace_cli_replays_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tracer = TraceLogger("s-cli", tmp_path)
    tracer.log("request", turn=1, prompt_tokens=12)
    tracer.log("tool", turn=1, tool="read_file", duration_ms=3, ok=False, error_code="tool_error")

    exit_code = main(["trace", "s-cli", "--log-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "session: s-cli（2 条记录）" in captured.out
    assert "read_file error 3ms code=tool_error" in captured.out


def test_trace_cli_reports_missing_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["trace", "ghost", "--log-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "没有找到 trace 文件" in captured.err
