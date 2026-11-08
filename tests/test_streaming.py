"""流式输出：Mock 增量、SSE 解析、降级重试与中断安全。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from miniagent.agent import Agent
from miniagent.errors import ProviderError
from miniagent.llm import LLMResponse, Message, MockProvider, OpenAICompatProvider, ToolCall
from miniagent.reliability import RetryPolicy
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


def make_provider(handler, **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="test",
        model="test-model",
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
        rate_limit_rpm=0,
        **kwargs,
    )


def sse_body(*chunks: dict) -> str:
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def test_mock_streams_deltas_through_agent() -> None:
    events = []
    agent = Agent(
        provider=MockProvider([LLMResponse(content="你好，世界")]),
        registry=make_registry(),
        on_event=events.append,
    )

    result = agent.run("hi")

    assert result.ok is True
    assert "".join(e.text for e in events if e.kind == "text_delta") == "你好，世界"
    assert [e.kind for e in events] == ["text_delta", "text_end"]
    # 流式路径不再重复发整段 text 事件
    assert not any(e.kind == "text" for e in events)


def test_openai_stream_parses_text_and_tool_calls() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = sse_body(
            {"choices": [{"delta": {"content": "你"}}]},
            {"choices": [{"delta": {"content": "好"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "echo", "arguments": '{"te'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'xt": "hi"}'}}]
                        }
                    }
                ]
            },
        )
        return httpx.Response(200, text=body)

    provider = make_provider(handler)
    deltas: list[str] = []

    response = provider.chat([Message(role="user", content="hi")], [], on_text=deltas.append)

    assert deltas == ["你", "好"]
    assert response.content == "你好"
    assert response.tool_calls == [ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')]
    assert requests[0]["stream"] is True


def test_stream_failure_falls_back_to_non_stream_once() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("stream"):
            return httpx.Response(503, text="boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": "降级成功"}}]})

    provider = make_provider(
        handler,
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0, sleep=lambda _: None),
    )
    deltas: list[str] = []

    response = provider.chat([Message(role="user", content="hi")], [], on_text=deltas.append)

    assert response.content == "降级成功"
    assert deltas == []
    # 流式先重试到上限（2 次），再降级为非流式 1 次
    assert len(requests) == 3
    assert requests[0]["stream"] is True
    assert requests[1]["stream"] is True
    assert "stream" not in requests[2]


def test_partial_stream_failure_does_not_retry() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = 'data: {"choices": [{"delta": {"content": "半"}}]}\n\ndata: not-json\n\n'
        return httpx.Response(200, text=body)

    provider = make_provider(handler)
    deltas: list[str] = []

    with pytest.raises(ProviderError, match="流式输出中断"):
        provider.chat([Message(role="user", content="hi")], [], on_text=deltas.append)

    assert deltas == ["半"]
    assert len(requests) == 1


class InterruptingProvider:
    def chat(self, messages, tools, *, on_text=None):
        if on_text is not None:
            on_text("已经输出一半")
        raise KeyboardInterrupt


def test_interrupt_does_not_corrupt_session(tmp_path: Path) -> None:
    session = Session.create("interrupt", tmp_path)
    agent = Agent(
        provider=InterruptingProvider(),
        registry=make_registry(),
        session=session,
    )

    with pytest.raises(KeyboardInterrupt):
        agent.run("做一件很久的事")

    loaded = Session.load("interrupt", tmp_path)
    assert [m.role for m in loaded.messages] == ["system", "user"]
    # 半截 assistant 文本不落盘，恢复后还能继续
    assert loaded.messages[-1].content == "做一件很久的事"
