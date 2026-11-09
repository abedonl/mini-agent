"""同轮工具依赖：路径硬边 + 模型补充边 + 拓扑分层，以及所有降级路径。"""

from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import pytest

from miniagent.agent import Agent
from miniagent.config import load_config
from miniagent.dependencies import build_layers, merge_edges, parse_edges
from miniagent.errors import ConfigError
from miniagent.llm import LLMResponse, MockProvider, ToolCall
from miniagent.logging import TraceLogger
from miniagent.tools import ToolRegistry


class Tracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def __enter__(self) -> None:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)

    def __exit__(self, *exc_info: object) -> None:
        with self._lock:
            self._active -= 1


class StepTool:
    """记录开始/结束顺序与并发度，用来验证分层是否真的生效。"""

    def __init__(
        self,
        name: str,
        *,
        read_only: bool = True,
        with_path: bool = False,
        delay: float = 0.0,
        tracker: Tracker | None = None,
        events: list[tuple[str, str]] | None = None,
    ) -> None:
        self.name = name
        self.description = name
        self.read_only = read_only
        self.parameters = (
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }
            if with_path
            else {"type": "object", "properties": {}, "required": []}
        )
        self._delay = delay
        self._tracker = tracker
        self._events = events

    def run(self, args: dict) -> str:
        if self._events is not None:
            self._events.append(("start", self.name))
        with self._tracker if self._tracker is not None else nullcontext():
            time.sleep(self._delay)
        if self._events is not None:
            self._events.append(("end", self.name))
        return f"{self.name} ok"


def tool_calls(*calls: tuple[str, dict]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(id=f"c{index}", name=name, arguments=json.dumps(args))
            for index, (name, args) in enumerate(calls)
        ]
    )


def build_agent(
    calls: tuple[tuple[str, dict], ...],
    tools: list,
    *,
    script: list[LLMResponse],
    tracer: TraceLogger | None = None,
    workers: int = 4,
    tool_deps: str = "model",
    deps_timeout: float = 2.0,
):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    provider = MockProvider([tool_calls(*calls), *script])
    agent = Agent(
        provider=provider,
        registry=registry,
        max_tool_workers=workers,
        tracer=tracer,
        tool_deps=tool_deps,
        deps_timeout=deps_timeout,
    )
    return agent, provider


def read_pair(
    tmp_path: Path,
    *,
    delay: float = 0.15,
    events: list[tuple[str, str]] | None = None,
):
    tracker = Tracker()
    tools = [
        StepTool("read_a", with_path=True, delay=delay, tracker=tracker, events=events),
        StepTool("read_b", with_path=True, delay=delay, tracker=tracker, events=events),
    ]
    calls = (
        ("read_a", {"path": str(tmp_path / "a.txt")}),
        ("read_b", {"path": str(tmp_path / "b.txt")}),
    )
    return tools, calls, tracker


def schedule_detail(tracer: TraceLogger, session: str, log_dir: Path) -> str:
    records = [r for r in TraceLogger.load(session, log_dir) if r.event == "schedule"]
    assert records
    return records[0].detail


def test_no_dependencies_run_in_one_layer(tmp_path: Path) -> None:
    tools, calls, tracker = read_pair(tmp_path)
    tracer = TraceLogger("deps-parallel", tmp_path / "logs")
    agent, _ = build_agent(
        calls,
        tools,
        script=[LLMResponse(content='{"edges": []}'), LLMResponse(content="done")],
        tracer=tracer,
    )

    started = time.perf_counter()
    assert agent.run("go").ok is True
    elapsed = time.perf_counter() - started

    assert tracker.max_active >= 2
    assert elapsed < 0.28  # 两个 0.15s 真并发
    detail = schedule_detail(tracer, "deps-parallel", tmp_path / "logs")
    assert "layers=1 sizes=2" in detail
    assert "edges=0" in detail


def test_model_edges_create_layers_and_keep_order(tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    tools, calls, tracker = read_pair(tmp_path, events=events)
    tracer = TraceLogger("deps-layers", tmp_path / "logs")
    agent, _ = build_agent(
        calls,
        tools,
        script=[
            LLMResponse(content='{"edges": [{"from": 0, "to": 1}]}'),
            LLMResponse(content="done"),
        ],
        tracer=tracer,
    )

    assert agent.run("go").ok is True

    assert tracker.max_active == 1
    assert events == [
        ("start", "read_a"),
        ("end", "read_a"),
        ("start", "read_b"),
        ("end", "read_b"),
    ]
    detail = schedule_detail(tracer, "deps-layers", tmp_path / "logs")
    assert "layers=2 sizes=1+1" in detail
    assert "model_edges=1" in detail
    assert "reason=-" in detail


def test_cycle_degrades_whole_turn_to_serial(tmp_path: Path) -> None:
    tools, calls, tracker = read_pair(tmp_path)
    tracer = TraceLogger("deps-cycle", tmp_path / "logs")
    agent, _ = build_agent(
        calls,
        tools,
        script=[
            LLMResponse(
                content='{"edges": [{"from": 0, "to": 1}, {"from": 1, "to": 0}]}'
            ),
            LLMResponse(content="done"),
        ],
        tracer=tracer,
    )

    assert agent.run("go").ok is True

    assert tracker.max_active == 1
    detail = schedule_detail(tracer, "deps-cycle", tmp_path / "logs")
    assert "layers=2 sizes=1+1" in detail
    assert "reason=cycle" in detail


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("这不是 JSON", "judge_parse_error"),
        ("", "judge_empty"),
        ('{"edges": [{"from": 0, "to": 9}]}', "judge_out_of_range"),
    ],
)
def test_bad_judgement_degrades_to_serial(
    tmp_path: Path, reply: str, reason: str
) -> None:
    tools, calls, tracker = read_pair(tmp_path)
    session = f"deps-bad-{reason}"
    tracer = TraceLogger(session, tmp_path / "logs")
    agent, _ = build_agent(
        calls,
        tools,
        script=[LLMResponse(content=reply), LLMResponse(content="done")],
        tracer=tracer,
    )

    assert agent.run("go").ok is True

    assert tracker.max_active == 1
    detail = schedule_detail(tracer, session, tmp_path / "logs")
    assert "layers=2 sizes=1+1" in detail
    assert f"reason={reason}" in detail


class SlowJudgeProvider(MockProvider):
    """判定请求（tools 为空）故意变慢，验证超时不阻塞主流程。"""

    def __init__(self, script: list[LLMResponse], *, delay: float = 0.3) -> None:
        super().__init__(script)
        self._delay = delay

    def chat(self, messages, tools, *, on_text=None):  # type: ignore[override]
        if not tools:
            time.sleep(self._delay)
            return LLMResponse(content='{"edges": []}')
        return super().chat(messages, tools, on_text=on_text)


def test_judge_timeout_degrades_to_serial(tmp_path: Path) -> None:
    tracker = Tracker()
    tools = [
        StepTool("read_a", with_path=True, delay=0.05, tracker=tracker),
        StepTool("read_b", with_path=True, delay=0.05, tracker=tracker),
    ]
    calls = (
        ("read_a", {"path": str(tmp_path / "a.txt")}),
        ("read_b", {"path": str(tmp_path / "b.txt")}),
    )
    tracer = TraceLogger("deps-timeout", tmp_path / "logs")
    provider = SlowJudgeProvider(
        [tool_calls(*calls), LLMResponse(content="done")], delay=0.3
    )
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    agent = Agent(
        provider=provider,
        registry=registry,
        max_tool_workers=4,
        tracer=tracer,
        tool_deps="model",
        deps_timeout=0.05,
    )

    assert agent.run("go").ok is True

    assert tracker.max_active == 1
    detail = schedule_detail(tracer, "deps-timeout", tmp_path / "logs")
    assert "reason=judge_timeout" in detail


def test_path_edges_win_over_model_conflict(tmp_path: Path) -> None:
    path = str(tmp_path / "a.txt")
    events: list[tuple[str, str]] = []
    tools = [
        StepTool("write_file", read_only=False, with_path=True, events=events),
        StepTool("read_file", with_path=True, events=events),
    ]
    tracer = TraceLogger("deps-conflict", tmp_path / "logs")
    agent, _ = build_agent(
        (("write_file", {"path": path}), ("read_file", {"path": path})),
        tools,
        script=[
            LLMResponse(content='{"edges": [{"from": 1, "to": 0}]}'),
            LLMResponse(content="done"),
        ],
        tracer=tracer,
    )

    assert agent.run("go").ok is True

    # 路径硬边 0 -> 1 保留；模型给的反向边被丢弃，不能变成环
    assert events == [
        ("start", "write_file"),
        ("end", "write_file"),
        ("start", "read_file"),
        ("end", "read_file"),
    ]
    detail = schedule_detail(tracer, "deps-conflict", tmp_path / "logs")
    assert "path_edges=1" in detail
    assert "dropped=1" in detail
    assert "reason=-" in detail


def test_off_mode_skips_judge_and_uses_path_plan(tmp_path: Path) -> None:
    tools, calls, tracker = read_pair(tmp_path)
    tracer = TraceLogger("deps-off", tmp_path / "logs")
    agent, provider = build_agent(
        calls,
        tools,
        script=[LLMResponse(content="done")],
        tracer=tracer,
        tool_deps="off",
    )

    assert agent.run("go").ok is True

    # 只有两次主请求，没有判定请求
    assert len(provider.calls) == 2
    assert tracker.max_active >= 2
    records = [r for r in TraceLogger.load("deps-off", tmp_path / "logs") if r.event == "schedule"]
    assert records == []  # 无降级时 off 档不写调度 trace


def test_identical_calls_reuse_previous_judgement(tmp_path: Path) -> None:
    tools, calls, _tracker = read_pair(tmp_path)
    tracer = TraceLogger("deps-cache", tmp_path / "logs")
    first = [
        LLMResponse(content='{"edges": []}'),
        tool_calls(*calls),
        LLMResponse(content="done"),
    ]
    agent, provider = build_agent(calls, tools, script=first, tracer=tracer)

    assert agent.run("go").ok is True

    # 主请求 -> 判定 -> 主请求 -> 收尾；第二轮判定命中缓存，没有额外请求
    assert len(provider.calls) == 4
    details = [
        r.detail for r in TraceLogger.load("deps-cache", tmp_path / "logs") if r.event == "schedule"
    ]
    assert "cache=miss" in details[0]
    assert "cache=hit" in details[1]


def test_results_are_backfilled_in_original_order(tmp_path: Path) -> None:
    delay = 0.05
    tracker = Tracker()
    tools = [
        StepTool("read_a", with_path=True, delay=delay, tracker=tracker),
        StepTool("read_b", with_path=True, delay=delay, tracker=tracker),
        StepTool("read_c", with_path=True, delay=delay, tracker=tracker),
    ]
    tracer = TraceLogger("deps-order", tmp_path / "logs")
    agent, _ = build_agent(
        (
            ("read_a", {"path": str(tmp_path / "a.txt")}),
            ("read_b", {"path": str(tmp_path / "b.txt")}),
            ("read_c", {"path": str(tmp_path / "c.txt")}),
        ),
        tools,
        script=[
            LLMResponse(content='{"edges": [{"from": 0, "to": 2}]}'),
            LLMResponse(content="done"),
        ],
        tracer=tracer,
    )

    assert agent.run("go").ok is True

    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert [m.name for m in tool_messages] == ["read_a", "read_b", "read_c"]
    assert [m.content for m in tool_messages] == ["read_a ok", "read_b ok", "read_c ok"]
    detail = schedule_detail(tracer, "deps-order", tmp_path / "logs")
    assert "layers=2 sizes=2+1" in detail


def test_parse_edges_rejects_out_of_range_and_non_int() -> None:
    assert parse_edges('{"edges": [{"from": 0, "to": 2}]}', 2) == ([], "out_of_range")
    assert parse_edges('{"edges": [{"from": true, "to": 1}]}', 2) == ([], "parse_error")
    assert parse_edges("", 2) == ([], "empty")


def test_merge_edges_drops_reverse_hard_edge_and_detects_cycle() -> None:
    merged, dropped, cycle = merge_edges([(0, 1)], [(1, 0)], 3)
    assert merged == [(0, 1)]
    assert dropped == 1
    assert cycle is False

    merged, dropped, cycle = merge_edges([], [(0, 1), (1, 2), (2, 0)], 3)
    assert len(merged) == 3
    assert dropped == 0
    assert cycle is True


def test_build_layers_puts_independent_calls_together() -> None:
    assert build_layers(3, [(0, 2)]) == [[0, 1], [2]]
    assert build_layers(2, []) == [[0, 1]]


def test_config_reads_and_validates_tool_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIAGENT_TOOL_DEPS", "off")
    assert load_config("missing.toml").tool_deps == "off"

    monkeypatch.setenv("MINIAGENT_TOOL_DEPS", "on")
    with pytest.raises(ConfigError, match="tool_deps"):
        load_config("missing.toml")
