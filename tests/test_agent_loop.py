"""核心循环的四条异常路径 + 正常路径。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from miniagent.agent import Agent
from miniagent.llm import LLMResponse, MockProvider, ToolCall
from miniagent.tools import ToolRegistry, builtin_tools


class EchoTool:
    name = "echo"
    description = "回显文本"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        return args["text"]


class ExplodingTool:
    name = "boom"
    description = "总是失败"
    read_only = False
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def run(self, args: dict[str, Any]) -> str:
        raise RuntimeError("炸了")


def build(script: list[LLMResponse], max_turns: int = 5):
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(ExplodingTool())
    events = []
    agent = Agent(
        provider=MockProvider(script),
        registry=registry,
        max_turns=max_turns,
        on_event=events.append,
    )
    return agent, events


def tool_response(name: str, arguments: str, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


def last_tool_message(agent: Agent) -> str:
    return next(m.content or "" for m in reversed(agent.messages) if m.role == "tool")


def test_tool_call_then_answer() -> None:
    agent, events = build(
        [
            tool_response("echo", '{"text": "你好"}'),
            LLMResponse(content="完成"),
        ]
    )

    result = agent.run("测试")

    assert result.ok is True
    assert result.turns == 2
    assert result.text == "完成"
    assert last_tool_message(agent) == "你好"
    kinds = [e.kind for e in events]
    assert kinds[:-2] == ["tool_start", "tool_end"]
    assert kinds[-2:] == ["text_delta", "text_end"]
    assert "".join(e.text for e in events if e.kind == "text_delta") == "完成"
    assert events[1].ok is True

    # 第二次请求必须带回工具结果，模型才有上下文
    second_call = agent.provider.calls[1]
    assert second_call[-1].role == "tool"
    assert second_call[-1].tool_call_id == "c1"


def test_bad_arguments_are_fed_back() -> None:
    agent, _ = build(
        [
            tool_response("echo", '{"wrong": "x"}'),
            LLMResponse(content="已修正"),
        ]
    )

    result = agent.run("测试")

    assert result.ok is True
    tool_message = last_tool_message(agent)
    assert "参数校验失败" in tool_message
    assert "缺少必需参数：text" in tool_message
    assert "未知参数：wrong" in tool_message


def test_unknown_tool_is_fed_back() -> None:
    agent, _ = build(
        [
            tool_response("nope", "{}"),
            LLMResponse(content="换一个工具"),
        ]
    )

    result = agent.run("测试")

    assert result.ok is True
    assert "未知工具：nope" in last_tool_message(agent)


def test_malformed_json_is_fed_back() -> None:
    agent, _ = build(
        [
            tool_response("echo", '{"text": '),
            LLMResponse(content="重试"),
        ]
    )

    result = agent.run("测试")

    assert result.ok is True
    assert "参数不是合法 JSON" in last_tool_message(agent)


def test_tool_exception_does_not_break_loop() -> None:
    agent, events = build(
        [
            tool_response("boom", "{}"),
            LLMResponse(content="工具挂了，我换个办法"),
        ]
    )

    result = agent.run("测试")

    assert result.ok is True
    assert "工具执行失败" in last_tool_message(agent)
    end_event = next(e for e in events if e.kind == "tool_end")
    assert end_event.ok is False


def test_max_turns_stops_loop() -> None:
    looping = LLMResponse(tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "再次"}')])
    agent, events = build([looping, looping, looping], max_turns=2)

    result = agent.run("测试")

    assert result.ok is False
    assert result.turns == 2
    assert "最大轮次" in result.text
    assert events[-1].kind == "error"


def test_multiple_tool_calls_in_one_turn_keep_order() -> None:
    agent, events = build(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments='{"text": "一"}'),
                    ToolCall(id="c2", name="echo", arguments='{"text": "二"}'),
                    ToolCall(id="c3", name="echo", arguments='{"text": "三"}'),
                ]
            ),
            LLMResponse(content="三轮工具都执行完了"),
        ]
    )

    result = agent.run("一次调用三个工具")

    assert result.ok is True
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert [m.content for m in tool_messages] == ["一", "二", "三"]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2", "c3"]
    assert [e.tool for e in events if e.kind == "tool_end"] == ["echo", "echo", "echo"]


def test_multiple_tool_turns_accumulate_history() -> None:
    agent, _ = build(
        [
            tool_response("echo", '{"text": "第一轮"}', call_id="c1"),
            tool_response("echo", '{"text": "第二轮"}', call_id="c2"),
            LLMResponse(content="完成"),
        ]
    )

    result = agent.run("连续两轮工具")

    assert result.ok is True
    assert result.turns == 3
    assert [m.role for m in agent.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_end_to_end_with_builtin_tools(tmp_path: Path) -> None:
    """离线跑通真实工具：写文件 -> 读回 -> 给出结论。"""
    registry = ToolRegistry()
    for tool in builtin_tools():
        registry.register(tool)
    target = tmp_path / "note.txt"
    script = [
        tool_response(
            "write_file",
            json.dumps({"path": str(target), "content": "hello miniagent"}),
        ),
        tool_response("read_file", json.dumps({"path": str(target)}), call_id="c2"),
        LLMResponse(content="文件已写好，读回内容一致"),
    ]
    events = []
    agent = Agent(provider=MockProvider(script), registry=registry, on_event=events.append)

    result = agent.run("写一个文件再读回来")

    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "hello miniagent"
    tool_events = [e for e in events if e.kind == "tool_end"]
    assert [e.ok for e in tool_events] == [True, True]
    assert "hello miniagent" in tool_events[1].text
