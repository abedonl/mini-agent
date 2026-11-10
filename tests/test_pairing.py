"""注入前轨迹体检：补配对、清空 tool_calls、幂等、正常轨迹不受影响。"""

from __future__ import annotations

from miniagent.agent import Agent
from miniagent.context import repair_pairing
from miniagent.llm import Message, MockProvider, ToolCall
from miniagent.tools import ToolRegistry


def assistant(
    call_id: str,
    name: str = "read_file",
    arguments: str = '{"path": "a.txt"}',
) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def test_fills_missing_tool_result() -> None:
    messages = [Message(role="user", content="读文件"), assistant("c1")]

    report = repair_pairing(messages)

    assert report.filled == ["c1"]
    assert report.messages[1].tool_calls[0].id == "c1"
    assert report.messages[2].role == "tool"
    assert report.messages[2].tool_call_id == "c1"
    assert report.messages[2].name == "read_file"
    assert "轨迹修复" in (report.messages[2].content or "")
    # 只返回新列表，原轨迹不受影响
    assert len(messages) == 2


def test_cleans_empty_tool_calls() -> None:
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="直接回答",
            tool_calls=[ToolCall(id="x", name="", arguments="{}")],
        ),
        Message(role="tool", content="孤儿结果", tool_call_id="x", name=""),
    ]

    report = repair_pairing(messages)

    assert report.dropped_calls == 1
    assert report.dropped_results == 1
    assert report.messages[1].tool_calls == []
    assert report.messages[1].content == "直接回答"
    assert [m.role for m in report.messages] == ["user", "assistant"]


def test_repair_is_idempotent() -> None:
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="q"),
        assistant("c1"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c2", name="", arguments="{}")],
        ),
        Message(role="tool", content="孤儿", tool_call_id="ghost", name="x"),
    ]

    first = repair_pairing(messages)
    second = repair_pairing(first.messages)

    assert first.changed is True
    assert second.changed is False
    assert [m.to_dict() for m in first.messages] == [m.to_dict() for m in second.messages]


def test_normal_trajectory_untouched() -> None:
    messages = [
        Message(role="user", content="q"),
        assistant("c1"),
        Message(role="tool", content="ok", tool_call_id="c1", name="read_file"),
    ]

    report = repair_pairing(messages)

    assert report.changed is False
    assert [m.to_dict() for m in report.messages] == [m.to_dict() for m in messages]


def test_build_context_repairs_injection_copy_only() -> None:
    agent = Agent(provider=MockProvider(), registry=ToolRegistry())
    agent.messages.append(Message(role="user", content="q"))
    agent.messages.append(assistant("c1"))
    before = [m.to_dict() for m in agent.messages]

    context = agent._build_context("", 1)

    # 会话里的原轨迹没有被补消息
    assert [m.to_dict() for m in agent.messages] == before
    assert any(
        m.role == "tool"
        and m.tool_call_id == "c1"
        and "轨迹修复" in (m.content or "")
        for m in context
    )
