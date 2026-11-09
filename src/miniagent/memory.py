"""记忆层：事实/事件存储、中文友好检索、攒批蒸馏、记忆门。

取舍：
- 存储是两个人类可读文件：memory.jsonl（结构化）与 memory.md（视图，程序生成）。
- 检索用重叠二字窗口做包含匹配，不引分词依赖，也不整串匹配。
- 写入不逐轮落库：先追加 pending.jsonl，攒够批次再蒸馏；失败保留 pending，数据不丢。
- 记忆门只做一次便宜的模型调用，走同一个 LLMProvider；判定失败倾向放行。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miniagent.errors import ProviderError
from miniagent.guard import PromptGuard
from miniagent.llm import LLMProvider, Message

MEMORY_KINDS = ("fact", "event")
DISTILL_SKIPPED = "skipped"
DISTILL_OK = "ok"
DISTILL_FAILED = "failed"

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_YES = re.compile(r"\bYES\b")
_NO = re.compile(r"\bNO\b")

DISTILL_PROMPT = (
    "你是记忆蒸馏器。从对话片段中抽取值得长期保存的信息，只输出 JSON 数组，不要解释。\n"
    '每条格式：{"kind": "fact" 或 "event", "text": "一句话"}\n'
    "fact：长期为真的事实（用户偏好、项目约定、环境信息）。\n"
    "event：某个时间点发生的事（完成的任务、遇到的问题、做出的决定）。\n"
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


def memory_tokens(text: str) -> list[str]:
    """中文取重叠二字窗口，英文/数字取整词；全小写，去重交给调用方。

    例："管理依赖" -> ["管理", "理依", "依赖"]；"用 uv 管理" 额外得到 "uv"。
    这是"包含匹配"而不是分词，避免把中文查询切成空词元。
    """
    raw = text or ""
    tokens: list[str] = []
    for run in _CJK_RUN.findall(raw):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    tokens.extend(word.lower() for word in _WORD.findall(raw))
    return tokens


@dataclass(frozen=True)
class MemoryEntry:
    kind: str  # fact | event
    text: str
    created_at: str
    session_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "text": self.text,
            "created_at": self.created_at,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            kind=str(data.get("kind", "fact")),
            text=str(data.get("text", "")),
            created_at=str(data.get("created_at", "")),
            session_id=str(data.get("session_id", "")),
        )


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
    """记忆目录下三个文件：memory.jsonl、memory.md、pending.jsonl。"""

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        batch_size: int = 6,
        session_id: str = "",
    ) -> None:
        self.directory = Path(directory).expanduser() if directory else default_memory_dir()
        self.entries_path = self.directory / "memory.jsonl"
        self.pending_path = self.directory / "pending.jsonl"
        self.view_path = self.directory / "memory.md"
        self.batch_size = max(1, batch_size)
        self.session_id = session_id
        self.last_dropped: list[str] = []  # 最近一次蒸馏里被丢弃条目的命中规则

    # ---------- 读写 ----------

    def entries(self) -> list[MemoryEntry]:
        return [MemoryEntry.from_dict(data) for data in _read_jsonl(self.entries_path)]

    def pending(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.pending_path)

    def add(self, kind: str, text: str) -> MemoryEntry:
        if kind not in MEMORY_KINDS:
            raise ValueError(f"未知的记忆类型：{kind}（可选 fact / event）")
        entry = MemoryEntry(
            kind=kind,
            text=text.strip(),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=self.session_id,
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

    def search(self, query: str, *, top_k: int = 3, min_score: float = 0.15) -> list[MemoryEntry]:
        query_tokens = set(memory_tokens(query))
        if not query_tokens:
            return []
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self.entries():
            entry_tokens = set(memory_tokens(entry.text))
            score = len(query_tokens & entry_tokens) / len(query_tokens)
            if score >= min_score:
                scored.append((score, entry))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [entry for _, entry in scored[:top_k]]

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

        for item in items:
            if guard is not None and guard.mode != "off":
                scan = guard.scan(item["text"])
                if scan.flagged:
                    self.last_dropped.extend(scan.hits)
                    continue
            self.add(item["kind"], item["text"])
        self.pending_path.write_text("", encoding="utf-8")
        return DISTILL_OK

    # ---------- 视图 ----------

    def _render_view(self) -> None:
        entries = self.entries()
        facts = [entry for entry in entries if entry.kind == "fact"]
        events = [entry for entry in entries if entry.kind == "event"]
        lines = [
            "# 记忆",
            "",
            "> 由 miniagent 自动生成，请勿手改；结构化数据在同目录 memory.jsonl。",
            "",
            f"共 {len(entries)} 条：事实 {len(facts)} 条，事件 {len(events)} 条。",
            "",
        ]
        for title, group in (("事实", facts), ("事件", events)):
            lines.append(f"## {title}")
            lines.append("")
            if not group:
                lines.append("- （暂无）")
            for entry in reversed(group):
                lines.append(f"- {entry.created_at} {entry.text}")
            lines.append("")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.view_path.write_text("\n".join(lines), encoding="utf-8")


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
            items.append({"kind": kind, "text": text_value})
    return items
