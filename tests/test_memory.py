"""M5 记忆与人格：向量检索、攒批蒸馏、记忆门两条分支、soul 优先级。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from miniagent.agent import Agent
from miniagent.embeddings import EmbeddingClient
from miniagent.errors import ProviderError
from miniagent.llm import LLMResponse, Message, MockProvider
from miniagent.logging import TraceLogger, format_record
from miniagent.memory import MemoryStore
from miniagent.soul import DEFAULT_SOUL, load_soul
from miniagent.tools import ToolRegistry


def memory_store_with_vectors(
    tmp_path: Path, tracer: TraceLogger | None = None
) -> MemoryStore:
    """固定返回同一向量的 MockTransport，供"记忆门 + 注入"链路测试使用。"""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        data = [
            {"index": index, "embedding": [1.0, 0.0, 0.0]}
            for index in range(len(payload["input"]))
        ]
        return httpx.Response(200, json={"data": data})

    embedder = EmbeddingClient(
        api_key="test-key",
        model="test-model",
        base_url="https://embed.test/v1",
        transport=httpx.MockTransport(handler),
        retry_delay=0.0,
    )
    return MemoryStore(tmp_path, embedder=embedder, tracer=tracer)


class FailingProvider:
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        on_text: Any = None,
    ) -> LLMResponse:
        raise ProviderError("gate 不可用")


def test_add_writes_human_readable_view(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "项目使用 Python 3.12")
    store.add("event", "完成了 M4 的 trace 回放")

    view = (tmp_path / "memory.md").read_text(encoding="utf-8")

    assert "# 记忆" in view
    assert "共 2 条：偏好 0 条，项目事实 1 条，待办 0 条，事件 1 条" in view
    assert "## 项目事实" in view
    assert "## 事件" in view
    assert "项目使用 Python 3.12" in view
    assert "完成了 M4 的 trace 回放" in view
    # 每条带可追溯 id（memory.jsonl 条目 id 的前 8 位）
    assert "- [" in view


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
    # 记忆门只控制"本轮检索注入"；长期记忆快照是 system prompt 的另一层，
    # 按设计固定带上，不受这里影响（见 tests/test_context_layers.py）。
    assert "与本轮相关的历史记忆" not in main_context

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
    tracer = TraceLogger("s-gate-search", tmp_path / "logs")
    store = memory_store_with_vectors(tmp_path, tracer)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    provider = MockProvider([LLMResponse(content="YES"), LLMResponse(content="你常用 uv")])
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
    tracer = TraceLogger("s-gate-parse", tmp_path / "logs")
    store = memory_store_with_vectors(tmp_path, tracer)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    provider = MockProvider([LLMResponse(content="让我想想"), LLMResponse(content="答")])
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
    tracer = TraceLogger("s-gate-error", tmp_path / "logs")
    store = memory_store_with_vectors(tmp_path, tracer)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    provider = MockProvider([LLMResponse(content="答")])
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


def _distill_once(store: MemoryStore, provider: MockProvider, user: str) -> str:
    store.record_turn(user, "好")
    return store.maybe_distill(provider)


def test_second_session_sees_previous_deposits(tmp_path: Path) -> None:
    """两个独立 Agent 实例模拟两次会话：第二次启动时能看到第一次的沉淀。"""
    first_store = MemoryStore(tmp_path, batch_size=1, session_id="s-first")
    first_provider = MockProvider(
        [LLMResponse(content='[{"kind": "fact", "text": "项目默认分支是 main"}]')]
    )
    first_agent = Agent(
        provider=first_provider,
        registry=ToolRegistry(),
        memory=first_store,
    )
    assert first_agent.messages  # 第一个会话正常建立
    assert _distill_once(first_store, first_provider, "记一下：项目默认分支是 main") == "ok"

    second_store = MemoryStore(tmp_path, session_id="s-second")
    second_agent = Agent(
        provider=MockProvider(),
        registry=ToolRegistry(),
        memory=second_store,
    )

    system = second_agent.messages[0].content or ""
    assert "项目默认分支是 main" in system
    assert "## 长期记忆快照" in system


def test_same_fact_distilled_twice_is_not_duplicated(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, batch_size=1)
    response = '[{"kind": "fact", "text": "用户喜欢深色主题"}]'
    provider = MockProvider([LLMResponse(content=response), LLMResponse(content=response)])

    assert _distill_once(store, provider, "我喜欢深色主题") == "ok"
    assert _distill_once(store, provider, "再说一次：我喜欢深色主题") == "ok"

    view = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert view.count("用户喜欢深色主题") == 1
    assert "共 1 条" in view
    # memory.jsonl 仍是只追加，去重只发生在沉淀视图
    assert len(store.entries()) == 2


def test_conflicting_fact_keeps_newest_value(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, batch_size=1)
    provider = MockProvider(
        [
            LLMResponse(content='[{"kind": "fact", "text": "项目默认分支是 main"}]'),
            LLMResponse(content='[{"kind": "fact", "text": "项目默认分支是 develop"}]'),
        ]
    )

    assert _distill_once(store, provider, "项目默认分支是 main") == "ok"
    assert _distill_once(store, provider, "后来改成了 develop") == "ok"

    view = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert "项目默认分支是 develop" in view
    assert "项目默认分支是 main" not in view
    assert "共 1 条" in view


def test_snapshot_truncates_by_section_priority(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    for index in range(20):
        store.add("event", f"事件{index:02d} " + "历史" * 20)
        store.add("fact", f"偏好{index:02d} 用户喜欢方案{index:02d} " + "说明" * 20)

    full = store.snapshot(10000)
    limited = store.snapshot(600)

    assert len(limited) <= 600
    assert len(full) > 600
    assert "已截断，共省略" in limited
    assert "## 偏好" in limited
    assert "用户喜欢方案19" in limited  # 高优先级整条保留
    assert "## 事件" not in limited  # 低优先级先被省略
    assert limited.count("- [") < full.count("- [")


def test_view_write_failure_degrades_and_retries(tmp_path: Path, monkeypatch) -> None:
    import miniagent.memory as memory_module

    tracer = TraceLogger("s-write-fail", tmp_path / "logs")
    store = MemoryStore(tmp_path, tracer=tracer)
    real_replace = memory_module.os.replace

    def fail_replace(src: str, dst: str) -> None:
        if Path(dst).name == "memory.md":
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(memory_module.os, "replace", fail_replace)
    store.add("fact", "项目使用 Python 3.12")  # 写视图失败不影响条目落盘

    assert len(store.entries()) == 1
    assert not (tmp_path / "memory.md").exists()
    assert not [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")]

    event = next(
        record
        for record in TraceLogger.load("s-write-fail", tmp_path / "logs")
        if record.event == "memory_write"
    )
    assert event.ok is False
    assert event.error_code == "memory_view_write_failed"

    monkeypatch.setattr(memory_module.os, "replace", real_replace)
    store.add("fact", "项目使用 Python 3.13")  # 下次写入自动重试

    view = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert "Python 3.12" in view
    assert "Python 3.13" in view
    assert not [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")]
