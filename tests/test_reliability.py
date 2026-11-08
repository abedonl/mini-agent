"""M3 可靠性：错误分类、令牌桶限流、退避重试、工具并发与串行。"""

from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext

import httpx
import pytest

from miniagent.agent import Agent
from miniagent.errors import ProviderError
from miniagent.llm import LLMResponse, Message, MockProvider, OpenAICompatProvider, ToolCall
from miniagent.reliability import RetryPolicy, TokenBucket, is_retryable_status
from miniagent.tools import ToolRegistry


class FakeClock:
    """测试里把“睡”变成推进虚拟时间，避免真的等待。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_provider(handler, **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="test",
        model="test-model",
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
        rate_limit_rpm=0,
        **kwargs,
    )


def make_policy(sleeps: list[float], attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=attempts,
        base_delay=0.5,
        sleep=sleeps.append,
        rand=lambda: 0.0,
    )


def sse_body(*chunks: dict) -> str:
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def test_error_classification() -> None:
    assert is_retryable_status(429)
    assert is_retryable_status(408)
    assert is_retryable_status(500)
    assert is_retryable_status(503)
    assert not is_retryable_status(400)
    assert not is_retryable_status(401)
    assert not is_retryable_status(404)


def test_token_bucket_waits_for_rate() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=60, capacity=1, clock=clock, sleep=clock.sleep)

    bucket.acquire()
    assert clock.sleeps == []
    bucket.acquire()
    assert clock.sleeps == [pytest.approx(1.0)]


def test_token_bucket_times_out_instead_of_hanging() -> None:
    clock = FakeClock()
    bucket = TokenBucket(
        rate_per_minute=6,
        capacity=1,
        max_wait=5,
        clock=clock,
        sleep=clock.sleep,
    )

    bucket.acquire()
    with pytest.raises(ProviderError, match="限流"):
        bucket.acquire()


def test_token_bucket_disabled_when_rate_is_zero() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=0, clock=clock, sleep=clock.sleep)

    bucket.acquire()
    bucket.acquire()
    assert bucket.enabled is False
    assert clock.sleeps == []


def test_retry_policy_exponential_backoff_and_retry_after() -> None:
    policy = RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=8, rand=lambda: 0.0)

    assert policy.delay_for(1) == pytest.approx(0.5)
    assert policy.delay_for(2) == pytest.approx(1.0)
    assert policy.delay_for(3) == pytest.approx(2.0)
    assert policy.delay_for(1, retry_after=7) == pytest.approx(7.0)
    assert policy.delay_for(1, retry_after=99) == pytest.approx(8.0)


def test_429_retry_after_is_honored() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    sleeps: list[float] = []
    provider = make_provider(handler, retry_policy=make_policy(sleeps))

    response = provider.chat([Message(role="user", content="hi")], [])

    assert response.content == "ok"
    assert len(requests) == 2
    assert sleeps == [pytest.approx(2.0)]


def test_client_error_is_not_retried() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(400, text="bad request")

    sleeps: list[float] = []
    provider = make_provider(handler, retry_policy=make_policy(sleeps))

    with pytest.raises(ProviderError, match="400"):
        provider.chat([Message(role="user", content="hi")], [])

    assert len(requests) == 1
    assert sleeps == []


def test_server_error_stops_after_max_attempts() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(503, text="busy")

    sleeps: list[float] = []
    provider = make_provider(handler, retry_policy=make_policy(sleeps))

    with pytest.raises(ProviderError, match="503"):
        provider.chat([Message(role="user", content="hi")], [])

    assert len(requests) == 3
    assert sleeps == [pytest.approx(0.5), pytest.approx(1.0)]


def test_stream_retries_before_first_text() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(503, text="busy")
        body = sse_body({"choices": [{"delta": {"content": "好"}}]})
        return httpx.Response(200, text=body)

    sleeps: list[float] = []
    provider = make_provider(handler, retry_policy=make_policy(sleeps))
    deltas: list[str] = []

    response = provider.chat([Message(role="user", content="hi")], [], on_text=deltas.append)

    assert response.content == "好"
    assert deltas == ["好"]
    assert len(requests) == 2
    assert sleeps == [pytest.approx(0.5)]


def test_partial_stream_never_retries() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = 'data: {"choices": [{"delta": {"content": "半"}}]}\n\ndata: not-json\n\n'
        return httpx.Response(200, text=body)

    sleeps: list[float] = []
    provider = make_provider(handler, retry_policy=make_policy(sleeps))
    deltas: list[str] = []

    with pytest.raises(ProviderError, match="流式输出中断"):
        provider.chat([Message(role="user", content="hi")], [], on_text=deltas.append)

    assert deltas == ["半"]
    assert len(requests) == 1
    assert sleeps == []


class Tracker:
    """记录同时执行的最大并发数，用来区分并发与串行。"""

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


class SlowTool:
    def __init__(
        self,
        name: str,
        *,
        delay: float = 0.05,
        read_only: bool = True,
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


def make_agent(
    tools: list[SlowTool],
    calls: list[str],
    *,
    workers: int = 4,
) -> tuple[Agent, list]:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    provider = MockProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id=f"c{index}", name=name, arguments="{}")
                    for index, name in enumerate(calls)
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    events: list = []
    agent = Agent(
        provider=provider,
        registry=registry,
        max_tool_workers=workers,
        on_event=events.append,
    )
    return agent, events


def tool_outputs(agent: Agent) -> list[str]:
    return [m.content or "" for m in agent.messages if m.role == "tool"]


def test_read_only_tools_run_concurrently() -> None:
    tracker = Tracker()
    tools = [
        SlowTool(name, delay=0.2, read_only=True, tracker=tracker)
        for name in ("read_a", "read_b", "read_c")
    ]
    agent, events = make_agent(tools, ["read_a", "read_b", "read_c"])

    start = time.perf_counter()
    result = agent.run("go")
    elapsed = time.perf_counter() - start

    assert result.ok is True
    assert tracker.max_active >= 2
    assert elapsed < 0.5  # 串行需要 0.6s
    assert tool_outputs(agent) == ["read_a ok", "read_b ok", "read_c ok"]
    assert [e.tool for e in events if e.kind == "tool_end"] == [
        "read_a",
        "read_b",
        "read_c",
    ]


def test_write_tools_stay_serial() -> None:
    tracker = Tracker()
    tools = [
        SlowTool(name, delay=0.05, read_only=False, tracker=tracker)
        for name in ("write_a", "write_b", "write_c")
    ]
    agent, _events = make_agent(tools, ["write_a", "write_b", "write_c"])

    start = time.perf_counter()
    result = agent.run("go")
    elapsed = time.perf_counter() - start

    assert result.ok is True
    assert tracker.max_active == 1
    assert elapsed >= 0.14  # 3 次串行，每次 0.05s
    assert tool_outputs(agent) == ["write_a ok", "write_b ok", "write_c ok"]


def test_mixed_calls_keep_original_order() -> None:
    tracker = Tracker()
    tools = [
        SlowTool("read_a", read_only=True, tracker=tracker),
        SlowTool("write_a", read_only=False, tracker=tracker),
        SlowTool("read_b", read_only=True, tracker=tracker),
        SlowTool("write_b", read_only=False, tracker=tracker),
    ]
    calls = ["read_a", "write_a", "read_b", "write_b"]
    agent, events = make_agent(tools, calls)

    result = agent.run("go")

    assert result.ok is True
    assert tool_outputs(agent) == ["read_a ok", "write_a ok", "read_b ok", "write_b ok"]
    assert [e.tool for e in events if e.kind == "tool_end"] == calls


def test_unknown_tool_is_treated_as_write() -> None:
    registry = ToolRegistry()
    assert registry.is_read_only("read_file") is False
    registry.register(SlowTool("read_a", read_only=True))
    assert registry.is_read_only("read_a") is True
