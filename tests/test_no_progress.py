"""无进展循环检测：指纹、提示/终止、结果变化、不同工具、开关关闭。"""

from __future__ import annotations

from typing import Any

from miniagent.agent import Agent
from miniagent.cli import build_parser
from miniagent.config import load_config
from miniagent.llm import LLMResponse, MockProvider, ToolCall
from miniagent.tools import ToolRegistry


class ConstTool:
    """可配置名字与输出的只读工具，用来构造"参数相同、结果相同/不同"的场景。"""

    read_only = True
    description = "返回固定或递增内容，测试用"

    def __init__(self, name: str = "echo", *, dynamic: bool = False) -> None:
        self.name = name
        self.dynamic = dynamic
        self.calls = 0
        self.parameters = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def run(self, args: dict[str, Any]) -> str:
        self.calls += 1
        if self.dynamic:
            return f"输出序号 {self.calls}"
        return "固定输出"


class RecordingTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))


def tool_turn(turn: int, name: str = "echo") -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(id=f"c{turn}", name=name, arguments='{"text": "same"}')
        ]
    )


def build_agent(*, tools: list[ConstTool] | None = None, **kwargs: Any) -> Agent:
    registry = ToolRegistry()
    for tool in tools or [ConstTool()]:
        registry.register(tool)
    return Agent(
        provider=MockProvider(script=[]),
        registry=registry,
        max_turns=kwargs.pop("max_turns", 6),
        **kwargs,
    )


def test_repeat_hint_then_stop() -> None:
    tracer = RecordingTracer()
    agent = build_agent(no_progress_repeat=3, tracer=tracer)
    agent.provider = MockProvider(
        script=[tool_turn(turn) for turn in range(1, 5)]
        + [LLMResponse(content="结束")]
    )

    result = agent.run("重复调用")

    assert result.ok is False
    assert "无进展" in result.text
    assert result.turns == 4
    hints = [
        message
        for message in agent.messages
        if message.role == "user" and "无进展检测" in (message.content or "")
    ]
    assert len(hints) == 1

    actions = [
        fields.get("detail", "")
        for event, fields in tracer.events
        if event == "no_progress"
    ]
    assert any("action=hint" in detail for detail in actions)
    assert any("action=stop" in detail for detail in actions)


def test_changed_result_is_not_repeat() -> None:
    tool = ConstTool(dynamic=True)
    agent = build_agent(tools=[tool])
    agent.provider = MockProvider(
        script=[tool_turn(turn) for turn in range(1, 5)]
        + [LLMResponse(content="结束")]
    )

    result = agent.run("每次结果不同")

    assert result.ok is True
    assert tool.calls == 4
    assert not any(
        message.role == "user" and "无进展检测" in (message.content or "")
        for message in agent.messages
    )


def test_different_tools_with_same_args_are_not_repeat() -> None:
    first = ConstTool("alpha")
    second = ConstTool("beta")
    agent = build_agent(tools=[first, second])
    agent.provider = MockProvider(
        script=[tool_turn(1, "alpha"), tool_turn(2, "beta"), tool_turn(3, "alpha")]
        + [LLMResponse(content="结束")]
    )

    result = agent.run("交替调用")

    assert result.ok is True
    assert not any(
        message.role == "user" and "无进展检测" in (message.content or "")
        for message in agent.messages
    )


def test_disabled_switch_keeps_old_behaviour() -> None:
    agent = build_agent(no_progress="off", max_turns=3)
    agent.provider = MockProvider(
        script=[tool_turn(turn) for turn in range(1, 4)]
        + [LLMResponse(content="结束")]
    )

    result = agent.run("关闭检测")

    assert result.ok is False
    assert "最大轮次" in result.text
    assert not any(
        message.role == "user" and "无进展检测" in (message.content or "")
        for message in agent.messages
    )


def test_repeat_threshold_is_configurable() -> None:
    agent = build_agent(no_progress_repeat=2)
    agent.provider = MockProvider(
        script=[tool_turn(turn) for turn in range(1, 4)]
        + [LLMResponse(content="结束")]
    )

    result = agent.run("阈值 2")

    assert result.ok is False
    assert result.turns == 3
    assert "连续 3 次" in result.text


def test_config_and_cli_flags(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("MINIAGENT_NO_PROGRESS", "off")
    monkeypatch.setenv("MINIAGENT_NO_PROGRESS_REPEAT", "4")
    config = load_config(tmp_path / "missing.toml")
    assert config.no_progress == "off"
    assert config.no_progress_repeat == 4

    args = build_parser().parse_args(
        ["--no-progress", "on", "--no-progress-repeat", "5"]
    )
    assert args.no_progress == "on"
    assert args.no_progress_repeat == 5
