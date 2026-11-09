"""M5 记忆与人格：中文检索、攒批蒸馏、记忆门两条分支、soul 优先级。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from miniagent.agent import Agent
from miniagent.errors import ProviderError
from miniagent.llm import LLMResponse, Message, MockProvider
from miniagent.logging import TraceLogger, format_record
from miniagent.memory import MemoryStore
from miniagent.soul import DEFAULT_SOUL, load_soul
from miniagent.tools import ToolRegistry


class FailingProvider:
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        on_text: Any = None,
    ) -> LLMResponse:
        raise ProviderError("gate 不可用")


def test_chinese_fact_is_found_by_chinese_query(tmp_path: Path) -> None:
    """先红后绿：中文事实必须能被中文提问检索到，不能因词元切分退化成空查询。"""
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")

    hits = store.search("我平时用什么管理依赖？")

    assert hits, "中文查询必须能命中中文事实"
    assert hits[0].text == "用户偏好用 uv 管理 Python 项目依赖"


def test_unrelated_query_returns_nothing(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")

    assert store.search("今天天气怎么样") == []


def test_add_writes_human_readable_view(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "项目使用 Python 3.12")
    store.add("event", "完成了 M4 的 trace 回放")

    view = (tmp_path / "memory.md").read_text(encoding="utf-8")

    assert "# 记忆" in view
    assert "共 2 条：事实 1 条，事件 1 条" in view
    assert "项目使用 Python 3.12" in view
    assert "完成了 M4 的 trace 回放" in view


def test_distill_only_runs_when_batch_is_full(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, batch_size=2)
    provider = MockProvider(
        [
            LLMResponse(
                content=(
                    "```json\n"
                    '[{"kind": "fact", "text": "用户喜欢深色主题"},'
                    ' {"kind": "event", "text": "今天完成了 M5 记忆"}]\n'
                    "```"
                )
            )
        ]
    )

    store.record_turn("我喜欢深色主题", "好的")
    assert store.maybe_distill(provider) == "skipped"
    assert provider.calls == []

    store.record_turn("记一下今天", "已记录")
    assert store.maybe_distill(provider) == "ok"

    assert len(provider.calls) == 1
    entries = store.entries()
    assert [entry.kind for entry in entries] == ["fact", "event"]
    assert entries[0].text == "用户喜欢深色主题"
    assert store.pending() == []
    assert "用户喜欢深色主题" in (tmp_path / "memory.md").read_text(encoding="utf-8")


def test_distill_failure_keeps_pending(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, batch_size=1)
    store.record_turn("这条不能丢", "好")

    assert store.maybe_distill(MockProvider([LLMResponse(content="不是 JSON")])) == "failed"
    assert store.maybe_distill(FailingProvider()) == "failed"

    assert len(store.pending()) == 1
    assert store.entries() == []
    assert not (tmp_path / "memory.md").exists()


def test_gate_skip_does_not_inject_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    provider = MockProvider([LLMResponse(content="NO"), LLMResponse(content="好笑的笑话")])
    tracer = TraceLogger("s-gate-skip", tmp_path / "logs")
    agent = Agent(provider=provider, registry=ToolRegistry(), memory=store, tracer=tracer)

    result = agent.run("讲个笑话")

    assert result.ok is True
    assert len(provider.calls) == 2  # 门 + 主对话
    main_context = "\n".join(message.content or "" for message in provider.calls[1])
    assert "uv 管理 Python 项目依赖" not in main_context

    gate = next(
        record
        for record in TraceLogger.load("s-gate-skip", tmp_path / "logs")
        if record.event == "memory_gate"
    )
    assert gate.detail == "skip:model"
    assert gate.ok is False
    assert "memory_gate" in format_record(gate)
    assert "skip:model" in format_record(gate)


def test_gate_search_injects_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    provider = MockProvider([LLMResponse(content="YES"), LLMResponse(content="你常用 uv")])
    tracer = TraceLogger("s-gate-search", tmp_path / "logs")
    agent = Agent(provider=provider, registry=ToolRegistry(), memory=store, tracer=tracer)

    agent.run("我平时用什么管理依赖？")

    main_context = "\n".join(message.content or "" for message in provider.calls[1])
    assert "uv 管理 Python 项目依赖" in main_context

    records = TraceLogger.load("s-gate-search", tmp_path / "logs")
    gate = next(record for record in records if record.event == "memory_gate")
    assert gate.detail == "search:model"
    assert gate.ok is True
    hit = next(record for record in records if record.event == "memory_hit")
    assert hit.detail == "hits=1"


def test_gate_parse_failure_defaults_to_search(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    provider = MockProvider([LLMResponse(content="让我想想"), LLMResponse(content="答")])
    tracer = TraceLogger("s-gate-parse", tmp_path / "logs")
    agent = Agent(provider=provider, registry=ToolRegistry(), memory=store, tracer=tracer)

    agent.run("我平时用什么管理依赖？")

    gate = next(
        record
        for record in TraceLogger.load("s-gate-parse", tmp_path / "logs")
        if record.event == "memory_gate"
    )
    assert gate.detail == "search:parse_failed"
    assert "uv 管理 Python 项目依赖" in "\n".join(
        message.content or "" for message in provider.calls[1]
    )


def test_gate_provider_error_defaults_to_search(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    provider = MockProvider([LLMResponse(content="答")])
    tracer = TraceLogger("s-gate-error", tmp_path / "logs")
    agent = Agent(
        provider=provider,
        registry=ToolRegistry(),
        memory=store,
        gate_provider=FailingProvider(),
        tracer=tracer,
    )

    agent.run("我平时用什么管理依赖？")

    gate = next(
        record
        for record in TraceLogger.load("s-gate-error", tmp_path / "logs")
        if record.event == "memory_gate"
    )
    assert gate.detail == "search:provider_error"
    assert "uv 管理 Python 项目依赖" in "\n".join(
        message.content or "" for message in provider.calls[0]
    )


def test_soul_priority_and_missing_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "soul.md").write_text("项目人格：只回答要点。", encoding="utf-8")

    text, source = load_soul(cwd=project)
    assert (text, source) == ("项目人格：只回答要点。", "project")

    explicit = tmp_path / "custom.md"
    explicit.write_text("显式人格", encoding="utf-8")
    assert load_soul(explicit, cwd=project) == ("显式人格", "explicit")

    missing = tmp_path / "missing.md"
    assert load_soul(missing, cwd=project) == (DEFAULT_SOUL, "default")


def test_soul_becomes_system_prompt_base() -> None:
    agent = Agent(
        provider=MockProvider(),
        registry=ToolRegistry(),
        soul="人格：只说一个字。",
    )

    system = agent.messages[0]
    assert system.role == "system"
    assert system.content is not None
    assert system.content.startswith("人格：只说一个字。")
    assert "可用工具" in system.content


def test_disabled_memory_adds_no_gate_call() -> None:
    provider = MockProvider([LLMResponse(content="答")])
    agent = Agent(provider=provider, registry=ToolRegistry(), memory=None)

    agent.run("你好")

    assert len(provider.calls) == 1


@pytest.mark.parametrize("answer,expected", [("YES", True), ("no", False), ("可能需要", True)])
def test_gate_answer_parsing(answer: str, expected: bool) -> None:
    from miniagent.memory import parse_gate_answer

    assert parse_gate_answer(answer).search is expected
