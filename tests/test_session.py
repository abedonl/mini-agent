"""会话持久化：落盘、恢复、列表与非法 ID。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miniagent.agent import Agent
from miniagent.errors import ConfigError
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


def test_message_roundtrip_keeps_tool_calls() -> None:
    message = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')],
    )

    restored = Message.from_dict(message.to_dict())

    assert restored.role == "assistant"
    assert restored.tool_calls == message.tool_calls


def test_create_append_and_load(tmp_path: Path) -> None:
    session = Session.create("s1", tmp_path)
    session.append(Message(role="user", content="你好"))
    session.append(Message(role="assistant", content="在"))

    loaded = Session.load("s1", tmp_path)

    assert [m.content for m in loaded.messages] == ["你好", "在"]
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert len(loaded.path.read_text(encoding="utf-8").splitlines()) == 2


def test_illegal_session_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="非法"):
        Session.create("../escape", tmp_path)


def test_missing_session_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="会话不存在"):
        Session.load("nope", tmp_path)


def test_corrupt_line_has_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"role": "user", "content": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="第 2 行"):
        Session.load("bad", tmp_path)


def test_list_sessions_recent_first(tmp_path: Path) -> None:
    first = Session.create("older", tmp_path)
    first.append(Message(role="user", content="a"))
    second = Session.create("newer", tmp_path)
    second.append(Message(role="user", content="b"))
    second.append(Message(role="assistant", content="c"))

    infos = Session.list(tmp_path)

    assert {info.session_id for info in infos} == {"older", "newer"}
    counts = {info.session_id: info.message_count for info in infos}
    assert counts == {"older": 1, "newer": 2}


def test_resume_restores_history_and_continues(tmp_path: Path) -> None:
    session = Session.create("chat", tmp_path)
    first_provider = MockProvider(
        [
            LLMResponse(tool_calls=[ToolCall("c1", "echo", '{"text": "第一轮"}')]),
            LLMResponse(content="第一轮完成"),
        ]
    )
    first = Agent(provider=first_provider, registry=make_registry(), session=session)
    assert first.run("开始").ok is True

    resumed_session = Session.load("chat", tmp_path)
    assert [m.role for m in resumed_session.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    second_provider = MockProvider([LLMResponse(content="记得")])
    second = Agent(provider=second_provider, registry=make_registry(), session=resumed_session)
    assert second.run("我刚才让你做了什么？").ok is True

    # 第二次请求里必须带着恢复出来的历史，模型才有上下文
    history_roles = [m.role for m in second_provider.calls[0]]
    assert history_roles[:5] == ["system", "user", "assistant", "tool", "assistant"]
    assert len(Session.load("chat", tmp_path).messages) == 7
