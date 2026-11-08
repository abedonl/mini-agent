"""上下文裁剪：保留 system 与最近轮次，只改请求视图，不动原始历史。"""

from __future__ import annotations

from pathlib import Path

from miniagent.agent import Agent
from miniagent.context import estimate_messages, estimate_tokens, trim_messages
from miniagent.llm import LLMResponse, Message, MockProvider, ToolCall
from miniagent.session import Session
from miniagent.tools import ToolRegistry


class EchoTool:
    name = "echo"
    description = "回显"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, args: dict) -> str:
        return args["text"]


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def turn(index: int, size: int = 300) -> list[Message]:
    return [
        Message(role="user", content=f"问题 {index}"),
        Message(role="assistant", content="字" * size),
    ]


def test_estimate_tokens_counts_cjk_higher() -> None:
    assert estimate_tokens("字" * 100) > estimate_tokens("a" * 100)
    assert estimate_tokens("") == 0


def test_no_trim_when_under_budget() -> None:
    messages = [Message(role="system", content="sys"), *turn(1), *turn(2)]

    result = trim_messages(messages, max_tokens=10_000)

    assert result == messages
    assert len(result) == 5


def test_trim_keeps_system_and_recent_turns() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(6):
        messages.extend(turn(index))

    result = trim_messages(messages, max_tokens=1200, keep_turns=2)

    assert result[0].role == "system"
    user_contents = [m.content for m in result if m.role == "user"]
    assert user_contents == ["问题 4", "问题 5"]
    assert estimate_messages(result) <= 1200


def test_old_tool_output_truncated_without_mutating_original() -> None:
    long_output = "字" * 3000
    messages = [
        Message(role="system", content="sys"),
        Message(
            role="assistant",
            tool_calls=[ToolCall("c1", "echo", '{"text": "x"}')],
        ),
        Message(role="tool", content=long_output, tool_call_id="c1", name="echo"),
        Message(role="assistant", content="第一轮完成"),
        Message(role="user", content="继续"),
        Message(role="assistant", content="第二轮完成"),
        Message(role="user", content="再继续"),
        Message(role="assistant", content="第三轮完成"),
    ]

    result = trim_messages(messages, max_tokens=1000, keep_turns=6)

    tool_messages = [m for m in result if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "截断" in (tool_messages[0].content or "")
    assert len(tool_messages[0].content or "") < 1000
    # 原始消息不被改写，会话文件里的历史仍然完整
    assert messages[2].content == long_output


def test_trim_never_drops_latest_turn() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(4):
        messages.extend(turn(index, size=2000))

    result = trim_messages(messages, max_tokens=500, keep_turns=6)

    assert any(m.content == "问题 3" for m in result)


def test_agent_trims_request_view_but_session_keeps_full_history(tmp_path: Path) -> None:
    session = Session.create("long", tmp_path)
    session.append(Message(role="system", content="sys"))
    for index in range(8):
        for message in turn(index):
            session.append(message)

    provider = MockProvider([LLMResponse(content="ok")])
    events = []
    agent = Agent(
        provider=provider,
        registry=make_registry(),
        session=session,
        max_context_tokens=1200,
        on_event=events.append,
    )

    assert agent.run("新问题").ok is True

    sent = provider.calls[0]
    assert len(sent) < len(agent.messages)
    assert sent[0].role == "system"
    assert any(event.kind == "context_trimmed" for event in events)
    # 落盘文件 === Agent 完整历史（system + 8 轮 + 本轮 user/assistant）
    loaded = Session.load("long", tmp_path)
    assert len(loaded.messages) == len(agent.messages)
    assert len(loaded.messages) == 19
