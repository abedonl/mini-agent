"""记忆层：事实/事件存储、向量检索、攒批蒸馏、记忆门。

取舍：
- 存储是两个人类可读文件：memory.jsonl（结构化、只追加）与 memory.md（跨会话沉淀层）。
- memory.md 按主题分节合并：同文去重、同主体冲突以新条目为准，每条带可追溯 id。
- 检索只走 embedding 余弦排序；没有向量能力时返回空结果，不退回关键词。
- 写入不逐轮落库：先追加 pending.jsonl，攒够批次再蒸馏；失败保留 pending，数据不丢。
- 记忆门只做一次便宜的模型调用，走同一个 LLMProvider；判定失败倾向放行。
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miniagent.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    EmbeddingIndex,
    cosine,
    entry_key,
)
from miniagent.errors import ProviderError
from miniagent.guard import PromptGuard
from miniagent.llm import LLMProvider, Message

MEMORY_KINDS = ("fact", "event")
DISTILL_SKIPPED = "skipped"
DISTILL_OK = "ok"
DISTILL_FAILED = "failed"

# 长期记忆快照进入 system prompt 的字符上限：快照是"固化摘要"，
# 不是检索结果，必须限长，避免随记忆条数线性抬高每轮成本。
SNAPSHOT_MAX_CHARS = 2000

# memory.md 的分节与截断优先级：偏好/事实长期有效，待办会过期，事件历史价值最低。
VIEW_SECTIONS = ("偏好", "项目事实", "待办", "事件")
# 关系词表：查询里命中这些词时，先按条目的 rel 收窄候选，再算余弦。
# 只做本地字符串匹配，不额外调模型；命中不了就走原来的全量余弦。
REL_GROUPS: dict[str, tuple[str, ...]] = {
    "家人": (
        "家人",
        "家属",
        "我爸",
        "爸爸",
        "父亲",
        "老爸",
        "我妈",
        "妈妈",
        "母亲",
        "老妈",
        "女儿",
        "儿子",
        "孩子",
        "老婆",
        "妻子",
        "老公",
        "丈夫",
    ),
    "同事": ("同事", "组里"),
    "朋友": ("朋友", "好友", "邻居"),
    "本人": ("我的", "我家", "我自己", "我", "公司", "团队"),
}
REL_NAMES = tuple(REL_GROUPS)
# 带"我的/我家"这类所有格的查询是强信号：即使点名了人物，也继续按 rel 收窄。
REL_POSSESSIVE = ("我的", "我家", "我自己")
# 这些 subject 是关系/角色词而不是人名，不能靠"出现在查询里"来收窄候选：
# 否则"我女儿班主任"会因为 subject=女儿 把班主任条目排除掉。
GENERIC_SUBJECTS = frozenset(
    {"用户", "本人", "家人", "父亲", "母亲", "女儿", "儿子", "同事", "朋友", "孩子"}
)

_SECTION_ALIASES = {"事实": "项目事实"}
_PREFERENCE_HINTS = (
    "偏好",
    "喜欢",
    "不喜欢",
    "习惯",
    "倾向",
    "讨厌",
    "更愿意",
    "更喜欢",
)
_TODO_HINTS = ("待办", "下一步", "接下来", "计划", "尚未", "还没", "TODO", "todo")
# 只认 "X是Y" / "X为Y" 这类明确结构来判定"同一主体"，避免过度合并普通句子。
_CONFLICT_PATTERN = re.compile(r"^(.{1,40}?)(?:是|为)(.+)$")
_TRUNCATE_NOTICE = (
    "\n...[已截断，共省略 {omitted} 条；按 偏好 > 项目事实 > 待办 > 事件 优先级保留]"
)

_YES = re.compile(r"\bYES\b")
_NO = re.compile(r"\bNO\b")

DISTILL_PROMPT = (
    "你是记忆蒸馏器。从对话片段中抽取值得长期保存的信息，只输出 JSON 数组，不要解释。\n"
    '每条格式：{"kind": "fact" 或 "event", "text": "一句话", '
    '"subject": "这条说的是谁", "rel": "与用户的关系", "backstory": "一句话来源"}\n'
    "fact：长期为真的事实（用户偏好、项目约定、环境信息）。\n"
    "event：某个时间点发生的事（完成的任务、遇到的问题、做出的决定）。\n"
    "subject / rel / backstory：偏好与人物关系类必须尽量填，rel 从 本人 / 家人 / 同事 / 朋友 中选；"
    "事件类可以留空。\n"
    "填不出就留空字符串，不许编造。\n"
    "没有值得保存的内容就输出 []。"
)

GATE_PROMPT = (
    "你是记忆检索判定器。判断用户这一轮输入是否需要查阅历史记忆才能答好。\n"
    "需要回忆用户偏好、历史决定、之前发生的事就回答 YES；"
    "闲聊、寒暄、纯当前上下文就能回答的问题回答 NO。\n"
    "只回答 YES 或 NO。"
)


def default_memory_dir() -> Path:
    env = os.environ.get("MINIAGENT_MEMORY_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".miniagent" / "memory"


@dataclass(frozen=True)
class MemoryEntry:
    kind: str  # fact | event
    text: str
    created_at: str
    session_id: str = ""
    # 卡片字段：可选，旧条目没有也合法（字段缺失按空字符串处理）。
    subject: str = ""  # 这条说的是谁
    rel: str = ""  # 与用户的关系：本人 / 家人 / 同事 / 朋友
    backstory: str = ""  # 一句话的来源背景

    def to_dict(self) -> dict[str, str]:
        record = {
            "kind": self.kind,
            "text": self.text,
            "created_at": self.created_at,
            "session_id": self.session_id,
        }
        # 空字段不落盘：旧文件不迁移，新条目也只有填了才多出键。
        for field in ("subject", "rel", "backstory"):
            value = getattr(self, field)
            if value:
                record[field] = value
        return record

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            kind=str(data.get("kind", "fact")),
            text=str(data.get("text", "")),
            created_at=str(data.get("created_at", "")),
            session_id=str(data.get("session_id", "")),
            subject=str(data.get("subject", "")),
            rel=str(data.get("rel", "")),
            backstory=str(data.get("backstory", "")),
        )

    def key(self) -> str:
        """稳定 ID：kind + text + created_at，重建索引时用来对齐向量。

        只跟 text 有关的三元组：新增 subject / rel / backstory 不改变 key，
        因此既有的 embeddings.jsonl 不会因为字段变更被整份重建。
        """
        return entry_key(self.kind, self.text, self.created_at)


def subject_filter(
    entries: Sequence[MemoryEntry], query: str
) -> tuple[list[MemoryEntry], str]:
    """按查询里的主体 / 关系词收窄候选，返回 (候选, 过滤标签)。

    只做关键词命中，不调模型：
    - 命中关系词（我爸 / 家人 / 同事 …）时按 rel 过滤；
    - 查询里出现条目 subject 全名时再按 subject 收窄；
    - 一个都没命中、或过滤后为空时原样返回（交给调用方走全量余弦）。
    """
    rels = [
        rel
        for rel in REL_NAMES
        if rel != "本人" and any(word in query for word in REL_GROUPS[rel])
    ]
    if not rels and any(word in query for word in REL_GROUPS["本人"]):
        rels = ["本人"]
    subjects = {
        entry.subject
        for entry in entries
        if entry.subject
        and entry.subject not in GENERIC_SUBJECTS
        and entry.subject in query
    }
    if subjects and rels == ["本人"] and not any(word in query for word in REL_POSSESSIVE):
        # 点名了具体人物时，"我/公司"这类兜底关系词只会把候选人挡在外面
        rels = []
    if not rels and not subjects:
        return list(entries), ""

    picked = [
        entry
        for entry in entries
        if (not subjects or entry.subject in subjects) and (not rels or entry.rel in rels)
    ]
    if not picked:
        return list(entries), ""
    parts = []
    if subjects:
        parts.append("subject=" + "|".join(sorted(subjects)))
    if rels:
        parts.append("rel=" + "|".join(rels))
    return picked, ",".join(parts)


def embedding_rank(
    entries: Sequence[MemoryEntry],
    vectors: dict[str, list[float]],
    query_vector: Sequence[float],
    limit: int,
) -> list[MemoryEntry]:
    """向量通道：纯 Python 余弦，只保留相似度为正的条目。"""
    scored: list[tuple[float, MemoryEntry]] = []
    for entry in entries:
        vector = vectors.get(entry.key())
        if vector is None:
            continue
        score = cosine(query_vector, vector)
        if score > 0.0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
    return [entry for _, entry in scored[:limit]]


@dataclass(frozen=True)
class GateDecision:
    search: bool
    reason: str  # model | parse_failed | provider_error


def parse_gate_answer(text: str | None) -> GateDecision:
    cleaned = (text or "").strip().upper()
    if not cleaned:
        return GateDecision(True, "parse_failed")
    if _NO.search(cleaned) or "不需要" in cleaned or "无需" in cleaned:
        return GateDecision(False, "model")
    if _YES.search(cleaned) or "需要" in cleaned:
        return GateDecision(True, "model")
    return GateDecision(True, "parse_failed")


class MemoryGate:
    """一次便宜的模型调用决定"这轮要不要查记忆"。

    判定失败（超时、非 YES/NO、空响应）一律放行：多查的代价小于漏查。
    """

    def review(self, provider: LLMProvider, user_input: str) -> GateDecision:
        messages = [
            Message(role="system", content=GATE_PROMPT),
            Message(role="user", content=user_input),
        ]
        try:
            response = provider.chat(messages, [])
        except ProviderError:
            return GateDecision(True, "provider_error")
        return parse_gate_answer(response.content)


class MemoryStore:
    """记忆目录：memory.jsonl、memory.md、pending.jsonl，以及可选的 embeddings.jsonl。"""

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        batch_size: int = 6,
        session_id: str = "",
        embedder: EmbeddingClient | None = None,
        tracer: Any = None,
        embedding_disabled_reason: str = "no_embedder",
    ) -> None:
        self.directory = Path(directory).expanduser() if directory else default_memory_dir()
        self.entries_path = self.directory / "memory.jsonl"
        self.pending_path = self.directory / "pending.jsonl"
        self.view_path = self.directory / "memory.md"
        self.embeddings_path = self.directory / "embeddings.jsonl"
        self.batch_size = max(1, batch_size)
        self.session_id = session_id
        self.embedder = embedder
        self.tracer = tracer
        # 没有向量能力时用于 trace：no_api_key / mock_mode / no_embedder
        self.embedding_disabled_reason = embedding_disabled_reason
        self.index = (
            EmbeddingIndex(self.embeddings_path, embedder.model) if embedder else None
        )
        self.last_dropped: list[str] = []  # 最近一次蒸馏里被丢弃条目的命中规则

    # ---------- 读写 ----------

    def entries(self) -> list[MemoryEntry]:
        return [MemoryEntry.from_dict(data) for data in _read_jsonl(self.entries_path)]

    def pending(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.pending_path)

    def add(
        self,
        kind: str,
        text: str,
        *,
        subject: str = "",
        rel: str = "",
        backstory: str = "",
    ) -> MemoryEntry:
        if kind not in MEMORY_KINDS:
            raise ValueError(f"未知的记忆类型：{kind}（可选 fact / event）")
        entry = MemoryEntry(
            kind=kind,
            text=text.strip(),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=self.session_id,
            subject=subject.strip(),
            rel=rel.strip(),
            backstory=backstory.strip(),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.entries_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._render_view()
        return entry

    def record_turn(self, user_input: str, assistant_text: str) -> None:
        """只进 pending，不落 memory；攒批后由 maybe_distill 统一蒸馏。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
            "user": user_input[:2000],
            "assistant": assistant_text[:2000],
        }
        with self.pending_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- 检索 ----------

    def search(self, query: str, *, top_k: int = 3) -> list[MemoryEntry]:
        """混合检索：先按查询里的主体/关系过滤，再算余弦；不够 top_k 再放开补足。

        降级不抛异常，也不退回关键词：没有 key、请求失败、索引重建失败时
        记忆检索就是"这轮没有命中"，写入与蒸馏不受影响。
        """
        entries = self.entries()
        if not entries or not query.strip():
            return []
        if self.embedder is None or self.index is None:
            self._trace_search(
                hits=0,
                ms=0,
                degraded=True,
                reason=self.embedding_disabled_reason or "no_embedder",
            )
            return []

        started = time.perf_counter()
        rebuilt = False
        added = 0
        filter_label = ""
        filled = 0
        try:
            # 正常路径下历史条目已在蒸馏时向量化，这里只会请求查询这一条；
            # 首次启用 / 模型维度变更时才会补历史（见 EmbeddingIndex.sync）。
            pairs = [(entry.key(), entry.text) for entry in entries]
            sync = self.index.sync(pairs, self.embedder)
            rebuilt = sync.rebuilt
            added = sync.added
            if sync.degraded:
                self._trace_search(
                    hits=0,
                    ms=int((time.perf_counter() - started) * 1000),
                    degraded=True,
                    rebuilt=rebuilt,
                    added=added,
                    reason="embedding_request_failed",
                    error_code="embedding_index_failed",
                )
                return []
            if not sync.vectors:
                self._trace_search(
                    hits=0,
                    ms=int((time.perf_counter() - started) * 1000),
                    degraded=True,
                    rebuilt=rebuilt,
                    added=added,
                    reason="index_empty",
                    error_code="embedding_index_empty",
                )
                return []

            query_vector = self.embedder.embed_one(query)
            if sync.dim is not None and len(query_vector) != sync.dim:
                # 查询向量维度与索引不一致：整份索引重建后再算
                sync = self.index.sync(pairs, self.embedder, force=True)
                rebuilt = True
                added = sync.added
                if sync.degraded or not sync.vectors:
                    self._trace_search(
                        hits=0,
                        ms=int((time.perf_counter() - started) * 1000),
                        degraded=True,
                        rebuilt=rebuilt,
                        added=added,
                        reason="embedding_request_failed",
                        error_code="embedding_index_failed",
                    )
                    return []
            candidates, filter_label = subject_filter(entries, query)
            hits = embedding_rank(candidates, sync.vectors, query_vector, top_k)
            if filter_label and len(hits) < top_k:
                # 过滤后不足 top_k：放开过滤补足，避免"过滤一轮直接返回空"
                chosen = {entry.key() for entry in hits}
                rest = [entry for entry in entries if entry.key() not in chosen]
                extra = embedding_rank(rest, sync.vectors, query_vector, top_k - len(hits))
                hits.extend(extra)
                filled = len(extra)
        except EmbeddingError:
            self._trace_search(
                hits=0,
                ms=int((time.perf_counter() - started) * 1000),
                degraded=True,
                rebuilt=rebuilt,
                added=added,
                reason="embedding_request_failed",
                error_code="embedding_failed",
            )
            return []

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._trace_search(
            hits=len(hits),
            ms=elapsed_ms,
            degraded=False,
            rebuilt=rebuilt,
            added=added,
            filter_label=filter_label,
            filled=filled,
        )
        return hits

    def _trace_search(
        self,
        *,
        hits: int,
        ms: int,
        degraded: bool,
        rebuilt: bool = False,
        added: int = 0,
        reason: str = "",
        error_code: str = "",
        filter_label: str = "",
        filled: int = 0,
    ) -> None:
        model = self.embedder.model if self.embedder is not None else ""
        self._trace_detail(
            "memory_embedding",
            ok=not degraded,
            duration_ms=ms,
            error_code=error_code,
            detail=(
                f"hits={hits} ms={ms} degraded={int(degraded)} "
                f"rebuilt={int(rebuilt)} added={added} model={model} reason={reason} "
                f"filtered={int(bool(filter_label))} filter={filter_label} filled={filled}"
            ),
        )

    def _trace_detail(self, event: str, **fields: Any) -> None:
        if self.tracer is not None:
            self.tracer.log(event, **fields)

    # ---------- 蒸馏 ----------

    def maybe_distill(
        self, provider: LLMProvider, *, guard: PromptGuard | None = None
    ) -> str:
        """返回 skipped（未到批次）/ ok / failed（pending 原样保留）。

        记忆会跨会话回流，所以命中注入规则的条目在**写入时**就丢弃；
        注入路径上还会再扫一次，兜住旧数据与手工编辑。
        """
        self.last_dropped = []
        batch = self.pending()
        if len(batch) < self.batch_size:
            return DISTILL_SKIPPED

        prompt = "\n\n".join(
            f"用户：{item.get('user', '')}\n助手：{item.get('assistant', '')}" for item in batch
        )
        messages = [
            Message(role="system", content=DISTILL_PROMPT),
            Message(role="user", content=prompt),
        ]
        try:
            response = provider.chat(messages, [])
            items = _parse_distilled(response.content)
        except (ProviderError, ValueError, TypeError, json.JSONDecodeError):
            return DISTILL_FAILED

        created: list[MemoryEntry] = []
        for item in items:
            if guard is not None and guard.mode != "off":
                scan = guard.scan(item["text"])
                if scan.flagged:
                    self.last_dropped.extend(scan.hits)
                    continue
            created.append(
                self.add(
                    item["kind"],
                    item["text"],
                    subject=item.get("subject", ""),
                    rel=item.get("rel", ""),
                    backstory=item.get("backstory", ""),
                )
            )
        self.pending_path.write_text("", encoding="utf-8")
        if created:
            self._sync_embeddings()
        return DISTILL_OK

    def _sync_embeddings(self) -> None:
        """蒸馏产出新条目后立即生成向量，避免每次检索重算历史。"""
        if self.embedder is None or self.index is None:
            return
        entries = self.entries()
        if not entries:
            return
        started = time.perf_counter()
        try:
            sync = self.index.sync(
                [(entry.key(), entry.text) for entry in entries], self.embedder
            )
        except EmbeddingError:
            self._trace_detail(
                "memory_embedding",
                ok=False,
                error_code="embedding_write_failed",
                duration_ms=int((time.perf_counter() - started) * 1000),
                detail="phase=write degraded=1",
            )
            return
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._trace_detail(
            "memory_embedding",
            ok=not sync.degraded,
            error_code="embedding_write_failed" if sync.degraded else "",
            duration_ms=elapsed_ms,
            detail=(
                f"phase=write added={sync.added} rebuilt={int(sync.rebuilt)} "
                f"degraded={int(sync.degraded)} ms={elapsed_ms} "
                f"model={self.embedder.model}"
            ),
        )

    # ---------- 视图 ----------

    def snapshot(self, limit: int = SNAPSHOT_MAX_CHARS) -> str:
        """长期记忆快照：读取 memory.md 的沉淀内容，供 system prompt 固化注入。

        与每轮的检索注入（_memory_note）是两条路径：这里给的是跨会话沉淀的
        长期记忆摘要，不依赖本轮问题，也不占 short_term_tokens 的额度。
        超预算时不整段丢弃，按 偏好 > 项目事实 > 待办 > 事件 的优先级逐条保留，
        低优先级条目先被省略；文件不存在时返回空字符串。
        """
        if not self.view_path.is_file():
            return ""
        text = self.view_path.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return _truncate_view(text, limit)

    def _render_view(self) -> bool:
        """把全部条目合并成 memory.md：按主题分节、去重、冲突取新、原子替换。

        每次渲染都基于 memory.jsonl 全量条目，因此之前会话沉淀的内容不会丢；
        同文本只保留一条，同一主体（"X是Y"）的冲突以最新条目为准。
        失败只记 trace，不抛异常：条目已落盘，下次写入会再试。
        """
        groups: dict[str, list[MemoryEntry]] = {title: [] for title in VIEW_SECTIONS}
        for entry in _merge_entries(self.entries()):
            groups[_section_for(entry)].append(entry)
        for group in groups.values():
            group.sort(key=lambda item: (item.created_at, item.text), reverse=True)

        total = sum(len(group) for group in groups.values())
        counts = "，".join(f"{title} {len(groups[title])} 条" for title in VIEW_SECTIONS)
        lines = [
            "# 记忆",
            "",
            "> 由 miniagent 自动沉淀，请勿手改；结构化数据在同目录 memory.jsonl。",
            "",
            f"共 {total} 条：{counts}。",
            "",
        ]
        for title in VIEW_SECTIONS:
            lines.append(f"## {title}")
            lines.append("")
            if not groups[title]:
                lines.append("- （暂无）")
            for entry in groups[title]:
                lines.append(_view_line(entry))
            lines.append("")
        return self._write_view("\n".join(lines))

    def _write_view(self, text: str) -> bool:
        """临时文件 + os.replace 原子替换，避免并发/中断留下半截 memory.md。"""
        temp_path = self.view_path.with_name(
            f".{self.view_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8", newline="\n") as fp:
                fp.write(text)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(temp_path, self.view_path)
        except OSError as exc:
            self._trace_detail(
                "memory_write",
                ok=False,
                error_code="memory_view_write_failed",
                detail=f"target=memory.md error={type(exc).__name__}",
            )
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _view_line(entry: MemoryEntry) -> str:
    """卡片条目带 subject｜rel 前缀；没有卡片字段的流水条目保持原样。"""
    prefix = "｜".join(part for part in (entry.subject, entry.rel) if part)
    if prefix:
        return f"- [{entry.key()[:8]}] {prefix}｜{entry.text}"
    return f"- [{entry.key()[:8]}] {entry.text}"


def _section_for(entry: MemoryEntry) -> str:
    """按主题归类：偏好 > 待办 > 项目事实 / 事件。

    关键词只是启发式：蒸馏模型不产出 topic，本地也不额外调模型判别。
    """
    if any(word in entry.text for word in _PREFERENCE_HINTS):
        return "偏好"
    if any(word in entry.text for word in _TODO_HINTS):
        return "待办"
    if entry.kind == "fact":
        return "项目事实"
    return "事件"


def _conflict_subject(text: str) -> str:
    """抽取"同一主体"的冲突键；没有明确结构时返回空串（只按文本去重）。"""
    match = _CONFLICT_PATTERN.match(_normalize(text))
    if match is None:
        return ""
    subject = match.group(1).strip("，,。.、:： ")
    return subject if len(subject) >= 2 else ""


def _merge_entries(entries: Sequence[MemoryEntry]) -> list[MemoryEntry]:
    """按时间从旧到新合并：同文去重，同主体冲突保留最新一条。"""
    merged: dict[tuple[str, str], MemoryEntry] = {}
    for entry in entries:
        section = _section_for(entry)
        identity = _conflict_subject(entry.text) or _normalize(entry.text)
        merged[(section, identity)] = entry
    return list(merged.values())


def _split_view(text: str) -> tuple[str, dict[str, list[str]]]:
    """拆出头部与各分节条目；兼容旧版"事实/事件"分节。"""
    header_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            current = _SECTION_ALIASES.get(title, title)
            sections.setdefault(current, [])
        elif current is None:
            header_lines.append(line)
        elif line.startswith("- ") and line.strip() != "- （暂无）":
            sections[current].append(line)
    return "\n".join(header_lines).strip() + "\n\n", sections


def _truncate_view(text: str, limit: int) -> str:
    """按分节优先级保留整条记忆，不切断单条文本。"""
    limit = max(0, limit)
    header, sections = _split_view(text)
    notice_reserve = len(_TRUNCATE_NOTICE.format(omitted=999999))
    while header and len(header) + notice_reserve > limit:
        # 先让掉说明行，再硬裁头部，保证至少给 notice 留出位置
        lines = header.rstrip("\n").splitlines(keepends=True)
        index = next((i for i, line in enumerate(lines) if line.startswith(">")), -1)
        if index >= 0:
            del lines[index]
            header = "".join(lines).strip() + "\n\n"
        else:
            keep = max(0, limit - notice_reserve - 2)
            header = header[:keep].rstrip() + "\n\n"
            break

    budget = max(0, limit - len(header) - notice_reserve)
    kept: list[str] = []
    omitted = 0
    for title in VIEW_SECTIONS:
        bullets = sections.get(title) or []
        if not bullets:
            continue
        block = f"## {title}\n\n"
        if budget < len(block) + len(bullets[0]) + 1:
            omitted += len(bullets)
            continue
        kept.append(block)
        budget -= len(block)
        for index, line in enumerate(bullets):
            cost = len(line) + 1
            if cost <= budget:
                kept.append(line + "\n")
                budget -= cost
            else:
                omitted += len(bullets) - index
                break
    if omitted == 0:
        # 头部已经占满但条目在别处被省略（极少见）：兜底硬裁，保证不超预算
        body = header + "".join(kept)
        return body[: max(0, limit - 1)] + "…"
    return header + "".join(kept) + _TRUNCATE_NOTICE.format(omitted=omitted)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # 记忆是派生数据，单行损坏不应拖垮 agent
            if isinstance(data, dict):
                records.append(data)
    return records


def _parse_distilled(text: str | None) -> list[dict[str, str]]:
    raw = (text or "").strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("蒸馏结果不是 JSON 数组")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("蒸馏结果不是 JSON 数组")

    items: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        text_value = str(item.get("text", "")).strip()
        if kind in MEMORY_KINDS and text_value:
            items.append(
                {
                    "kind": kind,
                    "text": text_value,
                    # 模型没填或填了 null 都按空字符串处理：字段缺失是合法状态
                    "subject": str(item.get("subject") or "").strip(),
                    "rel": str(item.get("rel") or "").strip(),
                    "backstory": str(item.get("backstory") or "").strip(),
                }
            )
    return items
