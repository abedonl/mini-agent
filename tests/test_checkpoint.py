"""工具调用 checkpoint：半截状态恢复、幂等判定、版本与坏行校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from miniagent.agent import Agent
from miniagent.checkpoint import (
    REPLAY_IDEMPOTENT,
    REPLAY_NEVER,
    REPLAY_SAFE,
    Checkpoint,
    default_checkpoint_path,
    read_records,
)
from miniagent.errors import ConfigError
from miniagent.hitl import AutoApproveHandler, HitlController, InteractionAnswer
from miniagent.llm import LLMResponse, Message, MockProvider, ToolCall
from miniagent.logging import TraceLogger
from miniagent.security import PathPolicy
from miniagent.session import Session
from miniagent.tools.base import replay_policy
from miniagent.tools.builtin import ReadFileTool, RunShellTool, WriteFileTool
from miniagent.tools.registry import ToolRegistry


class CountingTool:
    """只读工具：记录真正执行的次数，用于证明"已完成的不再重跑"。"""

    name = "probe"
    description = "只读探针"
    read_only = True
    replay = REPLAY_SAFE
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self) -> None:
        self.calls = 0

    def run(self, args: dict[str, Any]) -> str:
        self.calls += 1
        return "fresh-output"


class DenyHandler:
    name = "deny"

    def ask(self, request):
        return InteractionAnswer("deny", reason="user_rejected")


def make_session(tmp_path: Path, *messages: Message) -> Session:
    session = Session.create("s1", tmp_path)
    for message in messages:
        session.append(message)
    return session


def assistant_call(call_id: str, tool: str, args: dict[str, Any]) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(call_id, tool, json.dumps(args, ensure_ascii=False))],
    )


def checkpoint_path(session: Session) -> Path:
    return default_checkpoint_path(session.path)


def write_start(
    session: Session,
    *,
    call_id: str,
    tool: str,
    args: dict[str, Any],
    read_only: bool,
    replay: str,
    idem_key: str = "",
) -> int:
    """模拟上次进程"执行前落完开始记录"就被杀。"""
    checkpoint = Checkpoint.for_session(session)
    return checkpoint.begin(
        turn=1,
        call_id=call_id,
        tool=tool,
        args=args,
        read_only=read_only,
        replay=replay,
        idem_key=idem_key,
    )


def agent_for(
    session: Session,
    registry: ToolRegistry,
    checkpoint: Checkpoint,
    tracer=None,
) -> Agent:
    return Agent(
        provider=MockProvider(),
        registry=registry,
        session=session,
        checkpoint=checkpoint,
        tracer=tracer,
    )


# ---- 正常路径 -------------------------------------------------------------


def test_normal_path_writes_start_and_end(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    registry = ToolRegistry()
    registry.register(CountingTool())
    checkpoint = Checkpoint.for_session(session)
    provider = MockProvider(
        [
            LLMResponse(tool_calls=[ToolCall("c1", "probe", '{"path": "a"}')]),
            LLMResponse(content="完成"),
        ]
    )

    agent = Agent(
        provider=provider,
        registry=registry,
        session=session,
        checkpoint=checkpoint,
    )
    assert agent.run("试一下").ok

    records = read_records(checkpoint_path(session))
    assert [record["type"] for record in records] == ["start", "end"]
    assert all(record["v"] == 1 for record in records)
    assert records[0]["call_id"] == "c1"
    assert records[0]["replay"] == REPLAY_SAFE
    assert records[1]["status"] == "ok"
    assert records[1]["recovered"] is False
    # 没有半截状态：恢复阶段不产生任何额外工具调用
    assert len(provider.calls) == 2
    assert agent.recovery == []


def test_no_session_means_no_checkpoint_write() -> None:
    registry = ToolRegistry()
    registry.register(CountingTool())
    agent = Agent(provider=MockProvider(), registry=registry)
    assert agent.checkpoint is None
    assert not list(Path.cwd().glob("checkpoints/*.jsonl"))


# ---- 情况 1：只读未完成 ---------------------------------------------------


def test_unfinished_read_only_is_rerun(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    args = {"path": str(target)}
    session = make_session(
        tmp_path, Message(role="system", content="sys"), assistant_call("c1", "read_file", args)
    )
    write_start(
        session,
        call_id="c1",
        tool="read_file",
        args=args,
        read_only=True,
        replay=REPLAY_SAFE,
    )

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    tracer = TraceLogger("s1", tmp_path / "logs")
    agent = agent_for(
        session, registry, Checkpoint.for_session(session, tracer=tracer), tracer=tracer
    )

    assert [item.action for item in agent.recovery] == ["rerun"]
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "已安全重跑" in tool_messages[0].content
    assert "hello" in tool_messages[0].content
    assert tool_messages[0].tool_call_id == "c1"

    records = read_records(checkpoint_path(session))
    resolved = [r for r in records if r["type"] == "end" and r["recovered"]]
    assert resolved and resolved[0]["action"] == "replayed"
    # 写入与恢复判定都进现有 trace
    events = TraceLogger.load("s1", tmp_path / "logs")
    assert any(
        r.event == "checkpoint" and "action=begin" in r.detail for r in events
    )
    assert any(
        r.event == "checkpoint_recover" and "action=rerun" in r.detail for r in events
    )
    # 已经修复过：再恢复一次不会再重跑
    again = agent_for(session, registry, Checkpoint.for_session(session))
    assert again.recovery == []


# ---- 情况 2：写工具未完成 -------------------------------------------------


def test_unfinished_write_is_marked_unknown_without_retry(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    args = {"path": str(target), "content": "new"}
    session = make_session(tmp_path, assistant_call("c1", "write_file", args))
    write_start(
        session,
        call_id="c1",
        tool="write_file",
        args=args,
        read_only=False,
        replay=REPLAY_IDEMPOTENT,
    )

    registry = ToolRegistry()
    registry.register(WriteFileTool())
    agent = agent_for(session, registry, Checkpoint.for_session(session))

    assert [item.action for item in agent.recovery] == ["unknown"]
    assert not target.exists()
    tool_message = [m for m in agent.messages if m.role == "tool"][0]
    assert "状态未知" in tool_message.content
    assert "不要原样重试" in tool_message.content

    end = [r for r in read_records(checkpoint_path(session)) if r["type"] == "end"][0]
    assert end["status"] == "unknown"
    assert end["action"] == "unknown"


def test_write_already_applied_is_treated_as_done(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("new", encoding="utf-8")
    args = {"path": str(target), "content": "new"}
    session = make_session(tmp_path, assistant_call("c1", "write_file", args))
    write_start(
        session,
        call_id="c1",
        tool="write_file",
        args=args,
        read_only=False,
        replay=REPLAY_IDEMPOTENT,
    )

    registry = ToolRegistry()
    registry.register(WriteFileTool())
    agent = agent_for(session, registry, Checkpoint.for_session(session))

    assert [item.action for item in agent.recovery] == ["idempotent"]
    tool_message = [m for m in agent.messages if m.role == "tool"][0]
    assert "写入已完成" in tool_message.content
    assert target.read_text(encoding="utf-8") == "new"
    end = [r for r in read_records(checkpoint_path(session)) if r["type"] == "end"][0]
    assert end["action"] == "idempotent"


def test_shell_is_explicitly_not_replayable(tmp_path: Path) -> None:
    assert replay_policy(RunShellTool()) == REPLAY_NEVER
    args = {"command": "echo hi"}
    session = make_session(tmp_path, assistant_call("c1", "run_shell", args))
    write_start(
        session,
        call_id="c1",
        tool="run_shell",
        args=args,
        read_only=False,
        replay=REPLAY_NEVER,
    )

    registry = ToolRegistry()
    registry.register(RunShellTool())
    agent = agent_for(session, registry, Checkpoint.for_session(session))

    assert [item.action for item in agent.recovery] == ["unknown"]
    assert "状态未知" in [m for m in agent.messages if m.role == "tool"][0].content


# ---- 情况 3：完成记录存在但消息缺失 ---------------------------------------


def test_finished_but_missing_message_is_filled_without_execution(tmp_path: Path) -> None:
    tool = CountingTool()
    args = {"path": "whatever"}
    session = make_session(
        tmp_path, Message(role="system", content="sys"), assistant_call("c1", "probe", args)
    )
    checkpoint = Checkpoint.for_session(session)
    seq = checkpoint.begin(
        turn=1,
        call_id="c1",
        tool="probe",
        args=args,
        read_only=True,
        replay=REPLAY_SAFE,
    )
    checkpoint.finish(
        seq,
        turn=1,
        call_id="c1",
        tool="probe",
        content="summary-value",
        is_error=False,
        duration_ms=7,
    )

    registry = ToolRegistry()
    registry.register(tool)
    agent = agent_for(session, registry, Checkpoint.for_session(session))

    assert [item.action for item in agent.recovery] == ["completed"]
    assert tool.calls == 0  # 绝不重复执行
    tool_message = [m for m in agent.messages if m.role == "tool"][0]
    assert "该调用已完成" in tool_message.content
    assert "summary-value" in tool_message.content


def test_missing_tool_message_without_checkpoint_is_filled(tmp_path: Path) -> None:
    session = make_session(
        tmp_path, assistant_call("c1", "read_file", {"path": "x"})
    )
    registry = ToolRegistry()
    registry.register(CountingTool())
    agent = agent_for(session, registry, Checkpoint.for_session(session))

    assert [item.action for item in agent.recovery] == ["unknown"]
    assert agent.recovery[0].reason == "no_checkpoint"
    tool_message = [m for m in agent.messages if m.role == "tool"][0]
    assert "未执行" in tool_message.content
    assert read_records(checkpoint_path(session)) == []


# ---- 格式与兼容 -----------------------------------------------------------


def test_unknown_version_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    path.write_text(
        json.dumps({"v": 99, "type": "start", "seq": 1}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="第 1 行.*版本"):
        Checkpoint(path)


def test_corrupt_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"v": 1, "type": "start", "seq": 1}) + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="第 2 行"):
        Checkpoint(path)


# ---- 幂等键与 HITL 对齐 ---------------------------------------------------


def test_write_idempotency_key_depends_on_path_and_content(tmp_path: Path) -> None:
    tool = WriteFileTool()
    base = {"path": str(tmp_path / "a.txt"), "content": "x"}

    assert tool.idempotency_key(base) != tool.idempotency_key(
        {"path": str(tmp_path / "b.txt"), "content": "x"}
    )
    assert tool.idempotency_key(base) != tool.idempotency_key(
        {"path": str(tmp_path / "a.txt"), "content": "y"}
    )
    assert tool.effect_applied(base) is False
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert tool.effect_applied(base) is True


def test_hitl_denial_leaves_no_start_record(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    checkpoint = Checkpoint.for_session(session)
    registry = ToolRegistry()
    registry.register(WriteFileTool(PathPolicy([tmp_path])))
    registry.approver = HitlController(handler=DenyHandler())
    registry.checkpoint = checkpoint

    result = registry.execute(
        ToolCall(
            "c1",
            "write_file",
            json.dumps({"path": str(tmp_path / "a.txt"), "content": "x"}),
        ),
        1,
    )

    assert result.is_error
    assert not (tmp_path / "a.txt").exists()
    assert read_records(checkpoint_path(session)) == []


def test_hitl_approval_starts_after_confirmation(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    checkpoint = Checkpoint.for_session(session)
    registry = ToolRegistry()
    registry.register(WriteFileTool(PathPolicy([tmp_path])))
    registry.approver = HitlController(handler=AutoApproveHandler())
    registry.checkpoint = checkpoint

    result = registry.execute(
        ToolCall(
            "c1",
            "write_file",
            json.dumps({"path": str(tmp_path / "a.txt"), "content": "x"}),
        ),
        1,
    )

    assert not result.is_error
    records = read_records(checkpoint_path(session))
    assert [record["type"] for record in records] == ["start", "end"]
    assert records[0]["read_only"] is False
    assert records[0]["idem_key"].startswith("sha256:")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"


def test_tool_exception_still_writes_error_end(tmp_path: Path) -> None:
    class BoomTool:
        name = "boom"
        description = "必定失败"
        read_only = True
        parameters = {"type": "object", "properties": {}}

        def run(self, args: dict[str, Any]) -> str:
            raise RuntimeError("炸了")

    session = make_session(tmp_path)
    checkpoint = Checkpoint.for_session(session)
    registry = ToolRegistry()
    registry.register(BoomTool())
    registry.checkpoint = checkpoint

    result = registry.execute(ToolCall("c1", "boom", "{}"), 2)

    assert result.is_error
    end = read_records(checkpoint_path(session))[-1]
    assert end["type"] == "end"
    assert end["status"] == "error"
    assert "炸了" in end["summary"]
