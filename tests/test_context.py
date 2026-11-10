"""上下文裁剪：保留 system 与最近轮次，只改请求视图，不动原始历史。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miniagent.agent import Agent
from miniagent.cli import main
from miniagent.config import load_config
from miniagent.context import (
    DEFAULT_SHORT_TERM_TOKENS,
    apply_short_term_window,
    estimate_messages,
    estimate_tokens,
    trim_messages,
)
from miniagent.errors import ConfigError
from miniagent.llm import LLMResponse, Message, MockProvider, ToolCall
from miniagent.logging import TraceLogger
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


# ---- 短期窗口 ---------------------------------------------------------------


def tool_turn(index: int, size: int = 300) -> list[Message]:
    call = ToolCall(f"c{index}", "echo", f'{{"text": "{index}"}}')
    return [
        Message(role="user", content=f"问题 {index}"),
        Message(role="assistant", tool_calls=[call]),
        Message(role="tool", content="字" * size, tool_call_id=f"c{index}", name="echo"),
        Message(role="assistant", content="完成"),
    ]


def test_short_term_defaults_to_2000() -> None:
    assert DEFAULT_SHORT_TERM_TOKENS == 2000


def test_short_term_under_budget_keeps_everything() -> None:
    messages = [Message(role="system", content="sys"), *turn(1, 100), *turn(2, 100)]

    result = apply_short_term_window(messages, budget=2000)

    assert result.stats.trimmed is False
    assert result.stats.kept_messages == result.stats.total_messages == 5
    assert result.messages == messages


def test_short_term_trims_to_budget_and_keeps_recent_turns() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(6):
        messages.extend(turn(index, size=400))

    result = apply_short_term_window(messages, budget=2000)

    assert result.stats.trimmed is True
    assert result.stats.to_tokens <= 2000
    assert result.stats.from_tokens > 2000
    assert result.messages[0].role == "system"
    users = [m.content for m in result.messages if m.role == "user"]
    assert users[-1] == "问题 5"
    assert "问题 0" not in users
    # 只改请求视图，原始消息对象不被移动或改写
    assert len(messages) == 13


def test_short_term_never_splits_tool_calls_from_results() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(4):
        messages.extend(tool_turn(index))
    budget = estimate_messages([*tool_turn(2), *tool_turn(3)])

    result = apply_short_term_window(messages, budget=budget)

    assert result.stats.kept_turns == 2
    kept = result.messages
    call_ids = {
        call.id
        for message in kept
        for call in (message.tool_calls or [])
    }
    tool_ids = {m.tool_call_id for m in kept if m.role == "tool"}
    assert call_ids == tool_ids == {"c2", "c3"}


def test_short_term_keeps_latest_turn_even_over_budget() -> None:
    messages = [Message(role="system", content="sys"), *turn(1, 100), *turn(2, 3000)]

    result = apply_short_term_window(messages, budget=50)

    assert result.stats.trimmed is True
    assert result.stats.to_tokens > 50
    assert [m.content for m in result.messages if m.role == "user"] == ["问题 2"]
    assert any(m.content == "字" * 3000 for m in result.messages)


def test_memory_note_is_not_counted_in_short_term_budget() -> None:
    agent = Agent(
        provider=MockProvider([]),
        registry=make_registry(),
        short_term_tokens=500,
    )
    agent.messages = [Message(role="system", content="sys")]
    for index in range(6):
        agent.messages.extend(turn(index, size=200))
    note = "长期记忆" * 5000

    context = agent._build_context(note, turn=1)

    assert context[0].role == "system" and context[0].content == note
    history = context[1:]
    assert estimate_messages(history) <= 500
    assert "budget=500" in agent._window_detail
    assert "trimmed=1" in agent._window_detail


def test_short_term_trace_fields_and_first_turn_snapshot(tmp_path: Path) -> None:
    tracer = TraceLogger("s-window", tmp_path)
    provider = MockProvider([LLMResponse(content="ok")])
    agent = Agent(
        provider=provider,
        registry=make_registry(),
        max_context_tokens=100_000,
        short_term_tokens=200,
        tracer=tracer,
    )
    for index in range(5):
        agent.messages.extend(turn(index, size=200))

    assert agent.run("新问题").ok is True

    records = TraceLogger.load("s-window", tmp_path)
    trims = [record for record in records if record.event == "trim"]
    assert len(trims) == 1
    detail = trims[0].detail
    assert "stage=short_term" in detail
    assert "budget=200" in detail
    assert "from=" in detail and "to=" in detail
    assert "messages=" in detail and "turns=" in detail
    assert "trimmed=1" in detail
    assert trims[0].prompt_tokens == estimate_messages(provider.calls[0])
    # 请求记录也带同样的窗口字段，首轮没触发裁剪时 trimmed=0
    request = next(record for record in records if record.event == "request")
    assert request.detail == trims[0].detail


def test_short_term_first_turn_snapshot_when_not_trimmed(tmp_path: Path) -> None:
    tracer = TraceLogger("s-window-ok", tmp_path)
    agent = Agent(
        provider=MockProvider([LLMResponse(content="ok")]),
        registry=make_registry(),
        short_term_tokens=2000,
        tracer=tracer,
    )

    assert agent.run("短问题").ok is True

    records = TraceLogger.load("s-window-ok", tmp_path)
    assert not [record for record in records if record.event == "trim"]
    request = next(record for record in records if record.event == "request")
    assert "stage=short_term" in request.detail
    assert "trimmed=0" in request.detail


def test_short_term_from_config_env_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("short_term_tokens = 1234\n", encoding="utf-8")
    assert load_config(config_path).short_term_tokens == 1234

    monkeypatch.setenv("MINIAGENT_SHORT_TERM_TOKENS", "1500")
    assert load_config(config_path).short_term_tokens == 1500

    # 非法值在配置层明确报错，而不是等到请求时才炸
    monkeypatch.setenv("MINIAGENT_SHORT_TERM_TOKENS", "abc")
    with pytest.raises(ConfigError, match="short_term_tokens"):
        load_config(config_path)
    monkeypatch.setenv("MINIAGENT_SHORT_TERM_TOKENS", "0")
    with pytest.raises(ConfigError, match="大于 0"):
        load_config(config_path)


def test_short_term_invalid_agent_and_cli_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="short_term_tokens"):
        Agent(provider=MockProvider([]), registry=make_registry(), short_term_tokens=0)

    exit_code = main(
        [
            "--mock",
            "--config",
            str(tmp_path / "missing.toml"),
            "--short-term-tokens",
            "0",
            "-p",
            "hi",
        ]
    )
    assert exit_code == 2
    assert "--short-term-tokens" in capsys.readouterr().err
