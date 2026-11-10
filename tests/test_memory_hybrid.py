"""混合记忆：subject / rel 卡片字段、过滤检索、补足与 trace、视图分区。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from miniagent.embeddings import EmbeddingClient, EmbeddingIndex
from miniagent.llm import OpenAICompatProvider
from miniagent.logging import TraceLogger
from miniagent.memory import MemoryEntry, MemoryStore, subject_filter

MODEL = "test-model"


def make_embedder(
    vectors: dict[str, list[float]],
    *,
    calls: list[dict[str, Any]] | None = None,
    dim: int = 3,
) -> EmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if calls is not None:
            calls.append(payload)
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


def attach(store: MemoryStore, embedder: EmbeddingClient) -> None:
    store.embedder = embedder
    store.index = EmbeddingIndex(store.embeddings_path, embedder.model)


def card(kind: str, text: str, subject: str = "", rel: str = "") -> MemoryEntry:
    return MemoryEntry(
        kind=kind,
        text=text,
        created_at="2026-01-01 00:00:00",
        subject=subject,
        rel=rel,
    )


# ---------- subject_filter ----------


def test_filter_pins_same_name_by_relation() -> None:
    mine = card("fact", "张医生在城东口腔诊所出诊", "张医生", "本人")
    dads = card("fact", "张医生在人民医院心内科出诊", "张医生", "家人")
    colleague = card("fact", "张医生负责接口评审", "张医生", "同事")

    picked, label = subject_filter([mine, dads, colleague], "我爸看的那个张医生复诊约在哪天？")

    assert picked == [dads]
    assert label == "subject=张医生,rel=家人"


def test_filter_ignores_role_subjects_and_keeps_relation() -> None:
    daughter = card("fact", "女儿周三下午有兴趣班", "女儿", "家人")
    teacher = card("fact", "李老师教语文，家长会排在月底", "李老师", "家人")

    picked, label = subject_filter([daughter, teacher], "我女儿班主任那边的会安排在什么时候？")

    # subject=女儿 是角色词，不能把班主任条目排除掉
    assert picked == [daughter, teacher]
    assert label == "rel=家人"


def test_filter_returns_all_when_nothing_matches() -> None:
    entries = [
        card("fact", "项目默认分支是 main"),
        card("fact", "用户喜欢手冲咖啡", "用户", "本人"),
    ]

    picked, label = subject_filter(entries, "项目默认分支是什么？")

    assert picked == entries
    assert label == ""


def test_filter_falls_back_when_relation_misses_everything() -> None:
    entries = [card("fact", "项目默认分支是 main")]

    picked, label = subject_filter(entries, "我爸的体检套餐")

    assert picked == entries
    assert label == ""


# ---------- 检索 ----------


def test_search_prefers_filtered_entry_then_fills_top_k(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    dads = store.add("fact", "张医生是我爸的心内科医生", subject="张医生", rel="家人")
    mine = store.add("fact", "张医生是我本人的牙医", subject="张医生", rel="本人")
    other = store.add("fact", "项目默认分支是 main")
    query = "我爸看的那个张医生复诊约在哪天？"
    tracer = TraceLogger("s-hybrid-order", tmp_path / "logs")
    store.tracer = tracer
    attach(
        store,
        make_embedder(
            {
                query: [1.0, 0.0, 0.0],
                dads.text: [0.6, 0.8, 0.0],  # 余弦低于未过滤的条目
                mine.text: [1.0, 0.0, 0.0],
                other.text: [0.9, 0.1, 0.0],
            }
        ),
    )

    hits = store.search(query, top_k=3)

    # 过滤命中的条目优先，其余按余弦补足；不允许直接返回空
    assert [entry.text for entry in hits] == [dads.text, mine.text, other.text]
    event = next(
        record
        for record in TraceLogger.load("s-hybrid-order", tmp_path / "logs")
        if record.event == "memory_embedding"
    )
    assert "filtered=1" in event.detail
    assert "filter=subject=张医生,rel=家人" in event.detail
    assert "filled=2" in event.detail


def test_search_never_returns_empty_after_filter(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "张医生是我爸的心内科医生", subject="张医生", rel="家人")
    fallback = store.add("fact", "项目默认分支是 main")
    query = "我爸看的那个张医生"
    tracer = TraceLogger("s-hybrid-fill", tmp_path / "logs")
    store.tracer = tracer
    attach(
        store,
        make_embedder(
            {
                query: [1.0, 0.0, 0.0],
                # 过滤命中的条目与查询正交：过滤一轮会得到 0 条
                "张医生是我爸的心内科医生": [0.0, 1.0, 0.0],
                fallback.text: [0.9, 0.1, 0.0],
            }
        ),
    )

    hits = store.search(query, top_k=1)

    assert [entry.text for entry in hits] == [fallback.text]
    event = next(
        record
        for record in TraceLogger.load("s-hybrid-fill", tmp_path / "logs")
        if record.event == "memory_embedding"
    )
    assert "filtered=1" in event.detail
    assert "filled=1" in event.detail


def test_search_without_card_fields_keeps_pure_cosine(tmp_path: Path) -> None:
    """旧条目没有 subject/rel 时过滤不生效，排序与改造前一致。"""
    store = MemoryStore(tmp_path)
    store.add("fact", "alpha keyword project")
    store.add("fact", "beta other thing")
    query = "alpha keyword"
    tracer = TraceLogger("s-hybrid-legacy", tmp_path / "logs")
    store.tracer = tracer
    attach(
        store,
        make_embedder(
            {
                query: [1.0, 0.0, 0.0],
                "alpha keyword project": [0.0, 1.0, 0.0],
                "beta other thing": [1.0, 0.0, 0.0],
            }
        ),
    )

    hits = store.search(query, top_k=2)

    assert [entry.text for entry in hits] == ["beta other thing"]
    event = next(
        record
        for record in TraceLogger.load("s-hybrid-legacy", tmp_path / "logs")
        if record.event == "memory_embedding"
    )
    assert "filtered=0" in event.detail


# ---------- 存储与视图 ----------


def test_card_fields_survive_roundtrip_and_missing_fields_are_legal(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "张医生是我爸的心内科医生", subject="张医生", rel="家人", backstory="陪诊")
    store.add("event", "上周完成了迁移")

    entries = store.entries()

    assert (entries[0].subject, entries[0].rel, entries[0].backstory) == ("张医生", "家人", "陪诊")
    assert (entries[1].subject, entries[1].rel, entries[1].backstory) == ("", "", "")


def test_card_fields_do_not_change_entry_key(tmp_path: Path) -> None:
    """embeddings.jsonl 只跟 text 有关：补字段不会触发重建。"""
    store = MemoryStore(tmp_path)
    entry = store.add("fact", "张医生在城东口腔诊所出诊")
    original_key = entry.key()

    record = json.loads(store.entries_path.read_text(encoding="utf-8").strip())
    record["subject"] = "张医生"
    record["rel"] = "本人"
    store.entries_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    reloaded = store.entries()[0]

    assert reloaded.key() == original_key
    assert reloaded.subject == "张医生"


def test_view_marks_cards_and_leaves_stream_unchanged(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add("fact", "张医生是我爸的心内科医生", subject="张医生", rel="家人")
    store.add("event", "上周完成了迁移")

    lines = (tmp_path / "memory.md").read_text(encoding="utf-8").splitlines()
    card_line = next(line for line in lines if "张医生是我爸" in line)
    stream_line = next(line for line in lines if "上周完成了迁移" in line)

    assert "张医生｜家人｜" in card_line
    assert "｜" not in stream_line


# ---------- 蒸馏 ----------


def test_distill_prompt_asks_for_card_fields_and_parses_them(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = json.dumps(
            [
                {
                    "kind": "fact",
                    "text": "张医生是我爸的心内科医生",
                    "subject": "张医生",
                    "rel": "家人",
                    "backstory": "陪父亲复诊时记下",
                },
                {
                    "kind": "event",
                    "text": "上周完成了迁移",
                    "subject": None,
                    "backstory": "",
                },
            ],
            ensure_ascii=False,
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = OpenAICompatProvider(
        api_key="test-key",
        model="test-model",
        base_url="http://test.local/v1",
        transport=httpx.MockTransport(handler),
        rate_limit_rpm=0,
    )
    store = MemoryStore(tmp_path, batch_size=1)
    store.record_turn("记一下我爸的心内科医生叫张医生", "好")

    assert store.maybe_distill(provider) == "ok"

    system_prompt = requests[0]["messages"][0]["content"]
    for field in ("subject", "rel", "backstory"):
        assert field in system_prompt
    assert "不许编造" in system_prompt

    entries = store.entries()
    assert entries[0].subject == "张医生"
    assert entries[0].rel == "家人"
    assert entries[0].backstory == "陪父亲复诊时记下"
    # 模型留空 / 返回 null 都按空字符串落库，不报错
    assert (entries[1].subject, entries[1].rel, entries[1].backstory) == ("", "", "")
