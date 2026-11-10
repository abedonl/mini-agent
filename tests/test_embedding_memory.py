"""记忆向量通道：全部用 MockTransport，不发起真实网络请求。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import miniagent.cli as cli
from miniagent.config import Config, load_config
from miniagent.embeddings import EmbeddingClient, EmbeddingIndex
from miniagent.llm import LLMResponse, MockProvider
from miniagent.logging import TraceLogger
from miniagent.memory import MemoryStore

MODEL = "test-model"


def make_embedder(
    vectors: dict[str, list[float]],
    *,
    calls: list[dict[str, Any]] | None = None,
    status: int = 200,
    exc: Exception | None = None,
    dim: int = 3,
) -> EmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if calls is not None:
            calls.append(payload)
        if exc is not None:
            raise exc
        if status >= 400:
            return httpx.Response(status, text="bad request")
        data = [
            {"index": index, "embedding": vectors.get(text, [0.0] * dim)}
            for index, text in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": data, "model": payload["model"]})

    return EmbeddingClient(
        api_key="test-key",
        model=MODEL,
        base_url="https://embed.test/v1",
        timeout=1.0,
        transport=httpx.MockTransport(handler),
        retry_delay=0.0,
    )


def attach(store: MemoryStore, embedder: EmbeddingClient) -> EmbeddingClient:
    store.embedder = embedder
    store.index = EmbeddingIndex(store.embeddings_path, embedder.model)
    return embedder


def test_without_embedder_returns_empty_and_traces_reason(tmp_path: Path) -> None:
    """没有向量能力时不退回关键词，直接返回空并记录原因。"""
    store = MemoryStore(
        tmp_path, embedding_disabled_reason="no_api_key", tracer=TraceLogger(
            "s-no-key", tmp_path / "logs"
        )
    )
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    store.add("event", "今天天气不错")

    assert store.search("我平时用什么管理依赖？", top_k=1) == []
    assert not (tmp_path / "embeddings.jsonl").exists()

    event = next(
        record
        for record in TraceLogger.load("s-no-key", tmp_path / "logs")
        if record.event == "memory_embedding"
    )
    assert event.ok is False
    assert "degraded=1" in event.detail
    assert "reason=no_api_key" in event.detail


def test_embedding_channel_recalls_semantic_match(tmp_path: Path) -> None:
    """关键词无交集、语义相近的条目，只能靠向量通道召回。"""
    store = MemoryStore(tmp_path)
    store.add("fact", "用户喜欢喝手冲咖啡")
    store.add("fact", "项目使用 Python 3.12")
    query = "我平时爱喝什么"

    attach(
        store,
        make_embedder(
            {query: [1.0, 0.0, 0.0], "用户喜欢喝手冲咖啡": [1.0, 0.0, 0.0]}
        ),
    )

    hits = store.search(query, top_k=1)

    assert [entry.text for entry in hits] == ["用户喜欢喝手冲咖啡"]


def test_embedding_order_is_pure_cosine(tmp_path: Path) -> None:
    """有 embedder 时只看余弦：关键词最相关的条目也不能插队。"""
    store = MemoryStore(tmp_path)
    store.add("fact", "alpha keyword project")
    store.add("fact", "beta other thing")
    store.add("fact", "alpha keyword decision")

    query = "alpha keyword"
    attach(
        store,
        make_embedder(
            {
                query: [1.0, 0.0, 0.0],
                # 关键词会排第一，但余弦为 0，不参与排序
                "alpha keyword project": [0.0, 1.0, 0.0],
                # 这条与查询余弦最高
                "beta other thing": [1.0, 0.0, 0.0],
                "alpha keyword decision": [0.8, 0.6, 0.0],
            }
        ),
    )

    hits = store.search(query, top_k=2)

    assert [entry.text for entry in hits] == [
        "beta other thing",
        "alpha keyword decision",
    ]


def test_index_records_model_and_dimension(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "条目一")
    attach(
        store,
        make_embedder({"查询": [1.0, 0.0, 0.0], "条目一": [1.0, 0.0, 0.0]}),
    )

    store.search("查询")

    records = [
        json.loads(line)
        for line in store.embeddings_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["model"] == MODEL
    assert records[0]["dim"] == 3
    assert len(records[0]["vector"]) == 3


def test_index_is_reused_and_only_query_is_embedded_again(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "条目一")
    store.add("fact", "条目二")
    calls: list[dict[str, Any]] = []
    attach(
        store,
        make_embedder(
            {
                "查询": [1.0, 0.0, 0.0],
                "条目一": [1.0, 0.0, 0.0],
                "条目二": [0.5, 0.5, 0.0],
            },
            calls=calls,
        ),
    )

    store.search("查询")
    first_round = len(calls)
    store.search("查询")

    assert first_round == 2  # 一次补条目，一次查查询
    assert len(calls) == first_round + 1  # 第二次只嵌入查询
    assert calls[-1]["input"] == ["查询"]


def test_model_mismatch_rebuilds_whole_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    entry = store.add("fact", "条目一")
    store.embeddings_path.write_text(
        json.dumps(
            {
                "v": 1,
                "id": entry.key(),
                "model": "old-model",
                "dim": 3,
                "vector": [1.0, 0.0, 0.0],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    attach(
        store,
        make_embedder(
            {"查询": [1.0, 0.0, 0.0], "条目一": [1.0, 0.0, 0.0]}, calls=calls
        ),
    )
    tracer = TraceLogger("s-rebuild-model", tmp_path / "logs")
    store.tracer = tracer

    store.search("查询")

    records = [
        json.loads(line)
        for line in store.embeddings_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {record["model"] for record in records} == {MODEL}
    assert len(calls) == 2  # 重建条目 + 查询
    event = next(
        record
        for record in TraceLogger.load("s-rebuild-model", tmp_path / "logs")
        if record.event == "memory_embedding"
    )
    assert "rebuilt=1" in event.detail
    assert "degraded=0" in event.detail


def test_dimension_mismatch_rebuilds_whole_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    entry = store.add("fact", "条目一")
    store.embeddings_path.write_text(
        json.dumps(
            {
                "v": 1,
                "id": entry.key(),
                "model": MODEL,
                "dim": 2,
                "vector": [1.0, 0.0],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    attach(
        store,
        make_embedder(
            {"查询": [1.0, 0.0, 0.0], "条目一": [1.0, 0.0, 0.0]},
            calls=calls,
            dim=3,
        ),
    )
    tracer = TraceLogger("s-rebuild-dim", tmp_path / "logs")
    store.tracer = tracer

    store.search("查询")

    records = [
        json.loads(line)
        for line in store.embeddings_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["dim"] for record in records] == [3]
    assert len(calls) == 2  # 查询 + 强制重建条目
    event = next(
        record
        for record in TraceLogger.load("s-rebuild-dim", tmp_path / "logs")
        if record.event == "memory_embedding"
    )
    assert "rebuilt=1" in event.detail


def test_timeout_returns_empty_and_traces(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    calls: list[dict[str, Any]] = []
    attach(store, make_embedder({}, calls=calls, exc=httpx.ConnectTimeout("boom")))
    tracer = TraceLogger("s-degrade-timeout", tmp_path / "logs")
    store.tracer = tracer

    assert store.search("管理依赖") == []
    assert len(calls) == 2  # 超时只重试一次

    event = next(
        record
        for record in TraceLogger.load("s-degrade-timeout", tmp_path / "logs")
        if record.event == "memory_embedding"
    )
    assert event.ok is False
    assert "degraded=1" in event.detail
    assert "reason=embedding_request_failed" in event.detail


def test_http_400_is_not_retried_and_returns_empty(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "用户偏好用 uv 管理 Python 项目依赖")
    calls: list[dict[str, Any]] = []
    attach(store, make_embedder({}, calls=calls, status=400))

    assert store.search("管理依赖") == []
    assert len(calls) == 1  # 4xx 不重试


def test_distill_embeds_new_entries_immediately(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    store = MemoryStore(
        tmp_path,
        batch_size=1,
        embedder=make_embedder(
            {
                "用户喜欢深色主题": [1.0, 0.0, 0.0],
                "用户喜欢什么": [1.0, 0.0, 0.0],
            },
            calls=calls,
        ),
    )
    provider = MockProvider(
        [LLMResponse(content='[{"kind": "fact", "text": "用户喜欢深色主题"}]')]
    )
    store.record_turn("我喜欢深色主题", "好的")

    assert store.maybe_distill(provider) == "ok"
    assert len(calls) == 1  # 写入时一次批量向量化
    assert store.embeddings_path.is_file()

    store.search("用户喜欢什么")
    assert len(calls) == 2  # 检索只补查询这一条
    assert calls[-1]["input"] == ["用户喜欢什么"]


def test_build_embedder_without_key_notifies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_embedding_notice_shown", False)
    messages: list[str] = []
    config = Config(embedding_api_key="")

    assert cli.build_embedder(config, mock=False, notice=messages.append) is None
    assert cli.build_embedder(config, mock=False, notice=messages.append) is None

    assert len(messages) == 1
    assert "SILICONFLOW_API_KEY" in messages[0]


def test_mock_mode_never_enables_embedding() -> None:
    config = Config(embedding_api_key="sk-test")

    assert cli.build_embedder(config, mock=True) is None


def test_config_reads_independent_embedding_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("MINIAGENT_EMBEDDING_MODEL", raising=False)
    path = tmp_path / "config.toml"
    path.write_text(
        'embedding_base_url = "https://embed.example/v1/"\n'
        'embedding_model = "BAAI/bge-m3"\n'
        'embedding_api_key = "file-key"\n',
        encoding="utf-8",
    )

    config = load_config(path)
    assert config.embedding_base_url == "https://embed.example/v1"
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.embedding_api_key == "file-key"

    monkeypatch.setenv("SILICONFLOW_API_KEY", "env-key")
    assert load_config(path).embedding_api_key == "env-key"


def test_cosine_matches_known_values() -> None:
    from miniagent.embeddings import cosine

    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 2.0], [1.0]) == 0.0  # 维度不一致按 0 处理


def test_entry_key_is_stable_and_distinct() -> None:
    from miniagent.embeddings import entry_key

    first = entry_key("fact", "同一句话", "2026-01-01 00:00:00")
    assert first == entry_key("fact", "同一句话", "2026-01-01 00:00:00")
    assert first != entry_key("event", "同一句话", "2026-01-01 00:00:00")
    assert first != entry_key("fact", "同一句话", "2026-01-02 00:00:00")
