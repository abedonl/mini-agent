"""Human-in-the-loop：统一工具拦截、确认 / 补充参数 / 权限提示与可观测。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from miniagent.agent import Agent
from miniagent.cli import make_agent
from miniagent.config import Config, load_config
from miniagent.errors import ConfigError
from miniagent.hitl import (
    DEFAULT_CONFIRM_RULES,
    AutoApproveHandler,
    AutoDenyHandler,
    HitlController,
    InteractionAnswer,
    ToolRequest,
)
from miniagent.llm import LLMResponse, MockProvider, ToolCall
from miniagent.logging import TraceLogger
from miniagent.security import PathPolicy
from miniagent.tools.builtin import ReadFileTool, RunShellTool, WriteFileTool
from miniagent.tools.registry import ToolRegistry


class ScriptedHandler:
    """按顺序返回预置答复，并记录每次请求。"""

    name = "scripted"

    def __init__(self, *answers: InteractionAnswer, delay: float = 0.0) -> None:
        self.answers = list(answers)
        self.delay = delay
        self.requests: list[ToolRequest] = []

    def ask(self, request: ToolRequest) -> InteractionAnswer:
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        return self.answers.pop(0)


class RecordTool:
    """非只读工具：记录是否真的执行。"""

    name = "write_file"
    description = "记录调用的写工具"
    read_only = False
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, args: dict) -> str:
        self.calls.append(dict(args))
        return "已写入"


def registry_with(tool, handler, *, rules=DEFAULT_CONFIRM_RULES) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    registry.approver = HitlController(rules=tuple(rules), handler=handler)
    return registry


def write_call(path: Path, content: str = "new") -> ToolCall:
    return ToolCall(
        id="c1",
        name="write_file",
        arguments=json.dumps({"path": str(path), "content": content}, ensure_ascii=False),
    )


# ---- 默认行为 -------------------------------------------------------------


def test_default_off_never_interacts(tmp_path: Path) -> None:
    tool = RecordTool()
    registry = ToolRegistry()
    registry.register(tool)
    assert registry.approver is None

    result = registry.execute(write_call(tmp_path / "a.txt"))

    assert not result.is_error
    assert result.hitl is None
    assert len(tool.calls) == 1


def test_agent_default_off_keeps_registry_untouched() -> None:
    tool = RecordTool()
    registry = ToolRegistry()
    registry.register(tool)
    Agent(provider=MockProvider(), registry=registry)
    assert registry.approver is None


# ---- 确认 -----------------------------------------------------------------


def test_confirm_allow_executes(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    handler = ScriptedHandler(InteractionAnswer("allow"))
    registry = registry_with(WriteFileTool(), handler, rules=("overwrite",))

    result = registry.execute(write_call(target))

    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "new"
    assert result.hitl is not None
    assert (result.hitl.decision, result.hitl.risk) == ("allow", "overwrite")
    assert handler.requests[0].phase == "confirm"
    assert handler.requests[0].reason == "覆盖已有文件"


def test_confirm_deny_keeps_file_and_tells_model(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    handler = ScriptedHandler(InteractionAnswer("deny", reason="user_rejected"))
    registry = registry_with(WriteFileTool(), handler, rules=("overwrite",))

    result = registry.execute(write_call(target))

    assert result.is_error
    assert target.read_text(encoding="utf-8") == "old"
    assert "用户拒绝执行" in result.content
    assert "不要原样重试" in result.content
    assert result.hitl is not None and result.hitl.decision == "deny"
    assert result.hitl.reason == "user_rejected"


def test_interactive_handler_waits_without_timeout(tmp_path: Path) -> None:
    gate = threading.Event()

    class WaitHandler:
        name = "wait"

        def ask(self, request: ToolRequest) -> InteractionAnswer:
            gate.wait(5)
            return InteractionAnswer("allow")

    registry = registry_with(RecordTool(), WaitHandler())
    results: list[Any] = []
    worker = threading.Thread(
        target=lambda: results.append(registry.execute(write_call(tmp_path / "a.txt")))
    )
    worker.start()
    time.sleep(0.15)
    assert worker.is_alive(), "交互式场景不应自动超时"

    gate.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results[0].hitl is not None
    assert results[0].hitl.decision == "allow"


def test_wait_ms_is_measured(tmp_path: Path) -> None:
    handler = ScriptedHandler(InteractionAnswer("allow"), delay=0.05)
    registry = registry_with(RecordTool(), handler)

    result = registry.execute(write_call(tmp_path / "a.txt"))

    assert result.hitl is not None
    assert result.hitl.wait_ms >= 30


def test_non_interactive_denies_and_model_continues(tmp_path: Path) -> None:
    tool = RecordTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = MockProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[write_call(tmp_path / "a.txt")],
            ),
            LLMResponse(content="改用只读方案。"),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        tool_deps="off",
        hitl="on",
        hitl_handler=AutoDenyHandler("该操作需要确认，请在交互模式下运行"),
    )

    result = agent.run("写文件")

    assert result.ok
    assert result.text == "改用只读方案。"
    assert tool.calls == []
    second_context = provider.calls[1]
    tool_message = next(message for message in second_context if message.role == "tool")
    assert "HITL 拦截" in (tool_message.content or "")
    assert "该操作需要确认，请在交互模式下运行" in (tool_message.content or "")
    assert registry.approver is not None
    assert registry.approver.mode == "noninteractive"


# ---- 信息不足 -------------------------------------------------------------


def test_missing_argument_is_supplied_then_runs(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    handler = ScriptedHandler(InteractionAnswer("allow", text=str(target)))
    registry = registry_with(ReadFileTool(), handler)

    result = registry.execute(ToolCall(id="c1", name="read_file", arguments="{}"))

    assert not result.is_error
    assert result.content == "hello"
    assert result.hitl is not None
    assert result.hitl.phase == "missing"
    assert result.hitl.reason == "user_supplied"
    assert handler.requests[0].missing == ("path",)


def test_missing_argument_denied_stops_that_call(tmp_path: Path) -> None:
    handler = ScriptedHandler(InteractionAnswer("deny", reason="empty_input"))
    registry = registry_with(ReadFileTool(), handler)

    result = registry.execute(ToolCall(id="c1", name="read_file", arguments="{}"))

    assert result.is_error
    assert "缺少参数" in result.content
    assert "用户未提供必要信息" in result.content
    assert result.hitl is not None and result.hitl.decision == "deny"


# ---- 权限硬边界 -----------------------------------------------------------


def test_path_whitelist_block_is_explained(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    registry = registry_with(WriteFileTool(PathPolicy([allowed])), AutoApproveHandler())

    result = registry.execute(write_call(tmp_path / "outside.txt"))

    assert result.is_error
    assert "权限边界拦截" in result.content
    assert "白名单" in result.content
    assert "需要放行" in result.content
    assert result.hitl is not None
    assert (result.hitl.decision, result.hitl.risk) == ("blocked", "permission")
    assert not (tmp_path / "outside.txt").exists()


def test_dangerous_command_block_is_explained(tmp_path: Path) -> None:
    registry = registry_with(RunShellTool(), AutoApproveHandler())

    result = registry.execute(
        ToolCall(id="c1", name="run_shell", arguments=json.dumps({"command": "rm -rf /"}))
    )

    assert result.is_error
    assert "权限边界拦截" in result.content
    assert "危险命令" in result.content
    assert "需要放行" in result.content
    assert result.hitl is not None and result.hitl.decision == "blocked"


def test_permission_block_has_no_hitl_record_when_off(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(WriteFileTool(PathPolicy([tmp_path / "allowed"])))

    result = registry.execute(write_call(tmp_path / "outside.txt"))

    assert result.is_error
    assert "权限边界拦截" in result.content
    assert result.hitl is None


# ---- 与计划模式叠加 + trace ------------------------------------------------


class PlanTool(RecordTool):
    """计划模式下的非只读工具，参数只需要 path。"""

    name = "write_file"
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


def plan_agent(
    tmp_path: Path, handler, tracer: TraceLogger
) -> tuple[Agent, MockProvider, PlanTool]:
    tool = PlanTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = MockProvider(
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "steps": [
                            {
                                "intent": "写文件",
                                "tool": "write_file",
                                "args": {"path": "a.txt"},
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            ),
            LLMResponse(content=json.dumps({"action": "finish", "answer": "完成"})),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        tool_deps="off",
        plan_mode="on",
        hitl="on",
        hitl_handler=handler,
        tracer=tracer,
    )
    return agent, provider, tool


def test_plan_mode_uses_same_gate_and_records_trace(tmp_path: Path) -> None:
    handler = ScriptedHandler(InteractionAnswer("allow"))
    tracer = TraceLogger("hitl-plan", tmp_path / "logs")
    agent, _, tool = plan_agent(tmp_path, handler, tracer)

    result = agent.run("写一个文件")

    assert result.ok
    assert tool.calls == [{"path": "a.txt"}]
    assert handler.requests[0].tool == "write_file"
    records = [
        record
        for record in TraceLogger.load("hitl-plan", tmp_path / "logs")
        if record.event == "hitl"
    ]
    assert len(records) == 1
    detail = records[0].detail
    assert "mode=hitl phase=confirm tool=write_file" in detail
    assert "risk=mutate" in detail
    assert "decision=allow" in detail
    assert "wait_ms=" in detail
    assert "handler=scripted" in detail


def test_plan_mode_rejection_reaches_review(tmp_path: Path) -> None:
    handler = ScriptedHandler(InteractionAnswer("deny", reason="user_rejected"))
    tracer = TraceLogger("hitl-plan-deny", tmp_path / "logs")
    agent, provider, tool = plan_agent(tmp_path, handler, tracer)

    result = agent.run("写一个文件")

    assert result.ok
    assert tool.calls == []
    review_context = provider.calls[1]
    assert any(
        "HITL 拦截" in (message.content or "")
        for message in review_context
    )
    records = [
        record
        for record in TraceLogger.load("hitl-plan-deny", tmp_path / "logs")
        if record.event == "hitl"
    ]
    assert records and records[0].ok is False
    assert "decision=deny" in records[0].detail


# ---- 配置与装配 -----------------------------------------------------------


def test_config_reads_hitl_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIAGENT_HITL", "on")
    monkeypatch.setenv("MINIAGENT_HITL_CONFIRM", "overwrite, delete")

    config = load_config(tmp_path / "missing.toml")

    assert config.hitl == "on"
    assert config.hitl_confirm == ("overwrite", "delete")


def test_config_rejects_unknown_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MINIAGENT_HITL_CONFIRM", "nope")
    with pytest.raises(ConfigError, match="未知的 hitl_confirm"):
        load_config(tmp_path / "missing.toml")


def test_make_agent_uses_noninteractive_handler() -> None:
    agent = make_agent(
        Config(hitl="on", memory_enabled=False),
        mock=True,
        non_interactive=True,
    )
    assert agent.registry.approver is not None
    assert agent.registry.approver.mode == "noninteractive"
    answer = agent.registry.approver.handler.ask(
        ToolRequest(tool="write_file", args={}, read_only=False)
    )
    assert answer.decision == "deny"
    assert answer.reason == "该操作需要确认，请在交互模式下运行"


def test_hitl_controller_rejects_unknown_rule_through_agent() -> None:
    tool = RecordTool()
    registry = ToolRegistry()
    registry.register(tool)
    with pytest.raises(ValueError, match="hitl_confirm"):
        Agent(provider=MockProvider(), registry=registry, hitl="on", hitl_confirm=("nope",))


def test_auto_deny_handler_returns_reason() -> None:
    answer = AutoDenyHandler("非交互模式").ask(
        ToolRequest(tool="x", args={}, read_only=False)
    )
    assert answer.decision == "deny"
    assert answer.reason == "非交互模式"
