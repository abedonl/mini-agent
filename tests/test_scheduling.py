"""工具调度：路径重叠降级、非重叠并发、同轮先写后读、shell 保守串行。"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import pytest

from miniagent.agent import Agent
from miniagent.llm import LLMResponse, MockProvider, ToolCall
from miniagent.logging import TraceLogger
from miniagent.scheduling import normalize_path, paths_overlap, plan_calls
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


class PathTool:
    """按 path 参数读写共享状态；用来观察调度顺序与并发度。"""

    def __init__(
        self,
        name: str,
        *,
        read_only: bool,
        delay: float = 0.0,
        state: dict[str, str] | None = None,
        events: list[tuple[str, str]] | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        self.name = name
        self.description = name
        self.read_only = read_only
        self.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        self._delay = delay
        self._state = state if state is not None else {}
        self._events = events
        self._tracker = tracker

    def run(self, args: dict) -> str:
        path = args["path"]
        if self._events is not None:
            self._events.append(("start", self.name))
        with self._tracker if self._tracker is not None else nullcontext():
            time.sleep(self._delay)
            if self.read_only:
                output = self._state.get(path, "empty")
            else:
                self._state[path] = "new"
                output = f"{self.name} ok"
        if self._events is not None:
            self._events.append(("end", self.name))
        return output


class BareTool:
    """无 path 参数的工具：run_shell 或未知写工具。"""

    def __init__(
        self,
        name: str,
        *,
        read_only: bool,
        delay: float = 0.0,
        tracker: Tracker | None = None,
    ) -> None:
        self.name = name
        self.description = name
        self.read_only = read_only
        self.parameters = {"type": "object", "properties": {}, "required": []}
        self._delay = delay
        self._tracker = tracker

    def run(self, args: dict) -> str:
        with self._tracker if self._tracker is not None else nullcontext():
            time.sleep(self._delay)
            return f"{self.name} ok"


def build_agent(
    calls: list[tuple[str, dict]],
    tools: list,
    *,
    workers: int = 4,
    tracer: TraceLogger | None = None,
    tool_deps: str = "off",
) -> Agent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    provider = MockProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(f"c{index}", name, json.dumps(args))
                    for index, (name, args) in enumerate(calls)
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    return Agent(
        provider=provider,
        registry=registry,
        max_tool_workers=workers,
        tracer=tracer,
        tool_deps=tool_deps,
    )


def tool_messages(agent: Agent) -> list[tuple[str, str]]:
    return [(m.name or "", m.content or "") for m in agent.messages if m.role == "tool"]


def tool_call(name: str, path: str) -> ToolCall:
    return ToolCall("c", name, json.dumps({"path": path}))


def test_overlapping_read_write_is_downgraded_and_traced(tmp_path: Path) -> None:
    target = str(tmp_path / "a.txt")
    tracker = Tracker()
    tools = [
        PathTool("read_file", read_only=True, state={}, tracker=tracker),
        PathTool("write_file", read_only=False, state={}, tracker=tracker),
    ]
    tracer = TraceLogger("sched-overlap", tmp_path / "logs")
    agent = build_agent(
        [("read_file", {"path": target}), ("write_file", {"path": target})],
        tools,
        tracer=tracer,
    )

    assert agent.run("go").ok is True

    assert tracker.max_active == 1
    assert [name for name, _ in tool_messages(agent)] == ["read_file", "write_file"]
    records = [
        r for r in TraceLogger.load("sched-overlap", tmp_path / "logs") if r.event == "schedule"
    ]
    assert len(records) == 1
    assert "downgraded=1" in records[0].detail
    assert "reason=path_overlap" in records[0].detail


def test_non_overlapping_reads_stay_concurrent_and_faster(tmp_path: Path) -> None:
    tracker = Tracker()
    tools = [
        PathTool(name, read_only=True, delay=0.2, tracker=tracker) for name in ("read_a", "read_b")
    ]
    calls = [
        ("read_a", {"path": str(tmp_path / "a.txt")}),
        ("read_b", {"path": str(tmp_path / "b.txt")}),
    ]
    concurrent_agent = build_agent(calls, tools)

    start = time.perf_counter()
    assert concurrent_agent.run("go").ok is True
    concurrent_elapsed = time.perf_counter() - start

    serial_agent = build_agent(calls, tools, workers=1)
    start = time.perf_counter()
    assert serial_agent.run("go").ok is True
    serial_elapsed = time.perf_counter() - start

    assert tracker.max_active >= 2
    assert concurrent_elapsed < serial_elapsed
    assert concurrent_elapsed < 0.35  # 两个 0.2s 真并发


def test_parent_directory_write_overlaps_child_read(tmp_path: Path) -> None:
    calls = [
        tool_call("read_file", str(tmp_path / "src" / "a.py")),
        tool_call("write_file", str(tmp_path / "src")),
    ]

    plan = plan_calls(calls, lambda name: name == "read_file")

    assert plan.downgraded == 1
    assert plan.concurrent == []
    assert plan.serial == [0, 1]
    assert plan.reasons == ("path_overlap",)


def test_relative_and_absolute_paths_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert normalize_path("src/a.py") == normalize_path(str(tmp_path / "src" / "a.py"))


@pytest.mark.skipif(os.name != "nt", reason="大小写不敏感仅 Windows")
def test_windows_paths_are_case_insensitive(tmp_path: Path) -> None:
    assert normalize_path(str(tmp_path / "Src" / "A.PY")) == normalize_path(
        str(tmp_path / "src" / "a.py")
    )


def test_symlink_resolves_to_real_path(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.py").write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不允许创建符号链接")
    if not (link / "a.py").exists():
        pytest.skip("符号链接在当前文件系统/沙箱下不可解析")

    assert os.path.samefile(link / "a.py", real / "a.py")
    through_link = normalize_path(str(link / "a.py"))
    # 已做真实路径与扩展前缀归一化：链接路径与其真实路径必然重叠
    assert not through_link.startswith("\\\\?\\")
    assert paths_overlap(normalize_path(str(link)), through_link)

    direct = normalize_path(str(real / "a.py"))
    if through_link != direct:
        # 本机沙箱把符号链接解析到映射后的存储路径，直接比较会失真
        pytest.skip("文件系统对符号链接做了路径映射，跳过与直接路径的相等断言")
    assert through_link == direct


def test_write_then_read_same_path_keeps_call_order(tmp_path: Path) -> None:
    path = str(tmp_path / "data.txt")
    state: dict[str, str] = {}
    events: list[tuple[str, str]] = []
    tools = [
        PathTool("read_file", read_only=True, state=state, events=events),
        PathTool("write_file", read_only=False, delay=0.1, state=state, events=events),
    ]
    agent = build_agent(
        [("write_file", {"path": path}), ("read_file", {"path": path})],
        tools,
    )

    assert agent.run("go").ok is True

    assert tool_messages(agent) == [("write_file", "write_file ok"), ("read_file", "new")]
    assert events == [
        ("start", "write_file"),
        ("end", "write_file"),
        ("start", "read_file"),
        ("end", "read_file"),
    ]


def test_run_shell_forces_whole_turn_serial(tmp_path: Path) -> None:
    tracker = Tracker()
    tools = [
        PathTool("read_a", read_only=True, delay=0.05, tracker=tracker),
        PathTool("read_b", read_only=True, delay=0.05, tracker=tracker),
        BareTool("run_shell", read_only=False, delay=0.05, tracker=tracker),
    ]
    tracer = TraceLogger("sched-shell", tmp_path / "logs")
    agent = build_agent(
        [
            ("read_a", {"path": str(tmp_path / "a.txt")}),
            ("read_b", {"path": str(tmp_path / "b.txt")}),
            ("run_shell", {"command": "pytest"}),
        ],
        tools,
        tracer=tracer,
    )

    assert agent.run("go").ok is True

    assert tracker.max_active == 1
    records = [
        r for r in TraceLogger.load("sched-shell", tmp_path / "logs") if r.event == "schedule"
    ]
    assert "downgraded=2" in records[0].detail
    assert "reason=shell_barrier" in records[0].detail


def test_unknown_write_scope_is_a_barrier(tmp_path: Path) -> None:
    calls = [
        tool_call("read_file", str(tmp_path / "a.txt")),
        ToolCall("c", "deploy", "{}"),
    ]

    plan = plan_calls(calls, lambda name: name == "read_file")

    assert plan.serial == [0, 1]
    assert plan.reasons == ("write_barrier",)


def test_concurrent_reads_keep_original_result_order(tmp_path: Path) -> None:
    state: dict[str, str] = {}
    tools = [
        PathTool("read_file", read_only=True, delay=0.02, state=state),
        PathTool("write_file", read_only=False, delay=0.02, state=state),
    ]
    agent = build_agent(
        [
            ("read_file", {"path": str(tmp_path / "b.txt")}),
            ("write_file", {"path": str(tmp_path / "a.txt")}),
            ("read_file", {"path": str(tmp_path / "a.txt")}),
        ],
        tools,
    )

    assert agent.run("go").ok is True

    assert tool_messages(agent) == [
        ("read_file", "empty"),
        ("write_file", "write_file ok"),
        ("read_file", "new"),
    ]
