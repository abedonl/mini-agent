"""三级压缩流水线：短对话不动、逐级升级、摘要失败回退、原始历史不被改写。"""

from __future__ import annotations

import json

from miniagent.agent import Agent
from miniagent.compress import (
    SUMMARY_PROMPT,
    compress_messages,
    prune,
    structure,
)
from miniagent.context import estimate_messages
from miniagent.errors import ProviderError
from miniagent.llm import LLMResponse, Message, MockProvider, ToolCall
from miniagent.logging import TraceLogger
from miniagent.tools import ToolRegistry


class NoProvider:
    """被调用即失败：用来证明前两级够用时不会走到摘要。"""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools, *, on_text=None):  # noqa: ANN001
        self.calls.append(list(messages))
        raise AssertionError("不应触发第三级摘要")


class ExplodingProvider:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools, *, on_text=None):  # noqa: ANN001
        self.calls.append(list(messages))
        raise ProviderError("摘要模型不可用")


class NoopTool:
    name = "noop"
    description = "什么都不做"
    read_only = True
    parameters = {"type": "object", "properties": {}}

    def run(self, args: dict) -> str:
        return "ok"


def text_turn(index: int, size: int = 300) -> list[Message]:
    return [
        Message(role="user", content=f"问题 {index}"),
        Message(role="assistant", content="字" * size),
    ]


def tool_turn(
    index: int,
    tool: str,
    args: dict,
    output: str,
    call_id: str,
) -> list[Message]:
    return [
        Message(role="user", content=f"任务 {index}"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(call_id, tool, json.dumps(args))],
        ),
        Message(role="tool", content=output, tool_call_id=call_id, name=tool),
        Message(role="assistant", content=f"第 {index} 步完成"),
    ]


def test_short_conversation_not_compressed() -> None:
    provider = NoProvider()
    messages = [Message(role="system", content="sys"), *text_turn(1), *text_turn(2)]

    result = compress_messages(messages, max_tokens=10_000, provider=provider)

    assert result.messages == messages
    assert result.stats.stages == []
    assert result.stats.changed is False
    assert provider.calls == []


def test_stage_one_prune_is_enough() -> None:
    messages = [
        Message(role="system", content="sys"),
        *tool_turn(1, "read_file", {"path": "/a.txt"}, "字" * 5000, "c1"),
        *text_turn(2, size=50),
        *text_turn(3, size=50),
    ]
    provider = NoProvider()

    result = compress_messages(messages, max_tokens=2000, provider=provider)

    assert result.stats.stages == ["prune"]
    assert result.stats.saved["prune"] > 0
    assert result.stats.to_tokens <= 2000
    assert provider.calls == []
    # 原始历史不被改写
    assert messages[3].content == "字" * 5000


def test_stage_two_structure_folds_tool_calls() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(1, 4):
        messages.extend(
            tool_turn(
                index,
                "read_file",
                {"path": f"/f{index}.py"},
                "字" * 2500 + f"#{index}",
                f"c{index}",
            )
        )
    messages.extend(text_turn(4, size=40))
    messages.extend(text_turn(5, size=40))
    provider = NoProvider()

    result = compress_messages(messages, max_tokens=1500, provider=provider)

    assert result.stats.stages[:2] == ["prune", "structure"]
    assert "summary" not in result.stats.stages
    assert result.stats.to_tokens <= 1500
    assert provider.calls == []
    folded = [m.content or "" for m in result.messages if m.role == "assistant"]
    assert any("[工具调用] read_file" in content for content in folded)
    # 关键结论留槽：目标写进 system，不随折叠丢失
    system = result.messages[0].content or ""
    assert "关键结论" in system and "任务 1" in system


def test_stage_three_summarizes_when_rules_cannot_help() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(1, 7):
        messages.extend(text_turn(index, size=200))
    provider = MockProvider([LLMResponse(content="摘要：读取 /a.py，运行 pytest，待办是补测试")])

    result = compress_messages(messages, max_tokens=500, provider=provider)

    assert result.stats.stages == ["summary"]
    assert result.stats.to_tokens <= 500
    assert len(provider.calls) == 1
    prompt = provider.calls[0]
    assert prompt[0].content == SUMMARY_PROMPT
    summary = result.messages[0].content or ""
    assert "历史摘要" in summary
    assert "/a.py" in summary


def test_summary_failure_falls_back_to_hard_trim() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(1, 7):
        messages.extend(text_turn(index, size=200))

    result = compress_messages(messages, max_tokens=500, provider=ExplodingProvider())

    assert "hard_trim" in result.stats.fallback
    assert result.stats.to_tokens <= 500
    assert estimate_messages(result.messages) <= 500


def test_empty_summary_is_treated_as_failure() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(1, 7):
        messages.extend(text_turn(index, size=200))

    result = compress_messages(
        messages, max_tokens=500, provider=MockProvider([LLMResponse(content="   ")])
    )

    assert "hard_trim" in result.stats.fallback
    assert estimate_messages(result.messages) <= 500


def test_verbose_summary_is_discarded_and_hard_trimmed() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(1, 7):
        messages.extend(text_turn(index, size=200))
    provider = MockProvider([LLMResponse(content="摘要" * 3000)])

    result = compress_messages(messages, max_tokens=500, provider=provider)

    assert "hard_trim" in result.stats.fallback
    assert "summary" not in result.stats.stages
    assert estimate_messages(result.messages) <= 500


def test_stale_read_is_replaced_after_file_is_overwritten() -> None:
    messages = [
        Message(role="system", content="sys"),
        *tool_turn(1, "read_file", {"path": "/tmp/a.txt"}, "旧内容" * 200, "c1"),
        *tool_turn(2, "write_file", {"path": "/tmp/a.txt", "content": "新"}, "已写入", "c2"),
        *text_turn(3),
        *text_turn(4),
    ]

    result = prune(messages)

    reads = [m for m in result if m.role == "tool" and m.name == "read_file"]
    assert "已失效" in (reads[0].content or "")
    assert "旧内容" not in (reads[0].content or "")
    # 原始消息仍是完整内容
    reads_in_original = [m for m in messages if m.role == "tool" and m.name == "read_file"]
    assert reads_in_original[0].content == "旧内容" * 200


def test_duplicate_tool_results_are_dropped() -> None:
    same = "相同的输出" * 40
    messages = [
        Message(role="system", content="sys"),
        *tool_turn(1, "read_file", {"path": "/a"}, same, "c1"),
        *tool_turn(2, "read_file", {"path": "/b"}, same, "c2"),
        *text_turn(3),
        *text_turn(4),
    ]

    result = prune(messages)

    tool_contents = [m.content or "" for m in result if m.role == "tool"]
    assert "重复工具结果已省略" in tool_contents[1]
    assert tool_contents[0] == same


def test_old_tool_output_keeps_head_tail_and_error_lines() -> None:
    output = "开始\n" + "日志" * 1000 + "\nexit_code: 3\n错误：文件不存在\n" + "结尾" * 100
    messages = [
        Message(role="system", content="sys"),
        *tool_turn(1, "run_shell", {"command": "pytest"}, output, "c1"),
        *text_turn(2),
        *text_turn(3),
    ]

    result = prune(messages)

    clipped = next(m.content or "" for m in result if m.role == "tool")
    assert len(clipped) < len(output)
    assert clipped.startswith("开始")
    assert "已省略" in clipped
    assert "exit_code: 3" in clipped
    assert "错误：文件不存在" in clipped
    assert clipped.endswith("结尾" * 100)


def test_ack_messages_are_dropped_but_recent_turns_kept() -> None:
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="好的"),
        Message(role="assistant", content="收到"),
        *text_turn(2, size=400),
        *text_turn(3, size=40),
        # 最近两轮里的确认语不动
        Message(role="user", content="好的"),
        Message(role="assistant", content="收到"),
    ]

    result = prune(messages)

    contents = [m.content for m in result]
    assert "好的" in contents  # 只保留了最近轮次那条
    assert contents.count("好的") == 1
    assert "问题 2" in contents


def test_structure_adds_key_slot_only_when_tool_calls_folded() -> None:
    messages = [Message(role="system", content="sys"), *text_turn(1), *text_turn(2)]

    assert structure(messages) == messages


def make_agent(messages: list[Message], script: list[LLMResponse], tracer: TraceLogger) -> Agent:
    registry = ToolRegistry()
    registry.register(NoopTool())
    return Agent(
        provider=MockProvider(script),
        registry=registry,
        max_context_tokens=500,
        tracer=tracer,
        messages=messages,
    )


def test_agent_trace_records_stages_and_token_delta() -> None:
    messages = [Message(role="system", content="sys")]
    for index in range(1, 7):
        messages.extend(text_turn(index, size=200))
    agent = make_agent(
        messages,
        [LLMResponse(content="摘要：目标与待办"), LLMResponse(content="最终回答")],
        TraceLogger("compress-trace"),
    )

    assert agent.run("最后问题").ok is True

    trims = [r for r in TraceLogger.load("compress-trace") if r.event == "trim"]
    assert len(trims) == 1
    assert "stages=summary" in trims[0].detail
    assert "from=" in trims[0].detail and "to=" in trims[0].detail
    assert trims[0].prompt_tokens <= 500


def test_agent_short_conversation_writes_no_trim_trace() -> None:
    agent = make_agent(
        [Message(role="system", content="sys")],
        [LLMResponse(content="直接回答")],
        TraceLogger("compress-short"),
    )

    assert agent.run("你好").ok is True

    assert [r for r in TraceLogger.load("compress-short") if r.event == "trim"] == []
