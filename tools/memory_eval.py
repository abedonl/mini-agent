"""记忆消歧评测：一份评测集、同一个向量后端，对照纯余弦基线与混合检索。

用法（不需要任何 API key，数字可逐位复现）：

    python tools/memory_eval.py
    python tools/memory_eval.py --check tests/data/memory_disambiguation_baseline.json

要跑真实的 bge-m3（同一份评测集）：

    SILICONFLOW_API_KEY=... python tools/memory_eval.py --embedder siliconflow

说明：
- baseline 把 `subject_filter` 换成恒等函数，等价于改造前的纯余弦排序；
  hybrid 走当前 `MemoryStore.search` 的完整实现。
- 默认的 offline 后端只是确定性的字符 n-gram 哈希向量，用于无网络、无 key
  时也能复现同一组数字；它不是 bge-m3，跨后端数字不可直接比较。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import miniagent.memory as memory
from miniagent.embeddings import (
    EMBEDDING_DEFAULT_BASE_URL,
    EMBEDDING_DEFAULT_MODEL,
    EmbeddingClient,
    entry_key,
)
from miniagent.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "data" / "memory_disambiguation.json"
DEFAULT_BASELINE = ROOT / "tests" / "data" / "memory_disambiguation_baseline.json"


# ---------- 离线向量后端 ----------


def _features(text: str) -> dict[str, float]:
    """CJK 一元组 / 二元组 + ASCII 词，权重手工固定，保证跨机器一致。

    只在连续的汉字段内取二元组，避免把标点两侧的字拼成假词。
    """
    normalized = re.sub(r"\s+", " ", text.casefold())
    weights: dict[str, float] = {}
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        for char in run:
            key = f"c:{char}"
            weights[key] = weights.get(key, 0.0) + 0.3
        for left, right in zip(run, run[1:], strict=False):
            key = f"b:{left}{right}"
            weights[key] = weights.get(key, 0.0) + 1.0
    for word in re.findall(r"[a-z0-9][a-z0-9_.+-]*", normalized):
        key = f"w:{word}"
        weights[key] = weights.get(key, 0.0) + 1.0
    return weights


class LocalHashEmbedder:
    """确定性的本地向量，只用于评测复现；不是语义模型，也不参与线上检索。"""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.model = f"offline-char-bigram-{dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, weight in _features(text).items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            # 一律无符号累加：带符号哈希在特征很少时会把真实重合抵消成负数
            vector[index] += weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            return vector
        return [value / norm for value in vector]


class CachedEmbedder:
    """按文本缓存向量：同一份评测里 baseline / hybrid 共用，避免重复请求。

    同时记录真实调用：请求次数、文本条数、每次耗时与维度，供报告展示。
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.model = inner.model
        self.cache: dict[str, list[float]] = {}
        self.dim: int | None = getattr(inner, "dim", None)
        self.calls: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            started = time.perf_counter()
            try:
                vectors = self.inner.embed(missing)
            except Exception as exc:  # noqa: BLE001 - 记录真实失败原因后原样抛出
                self.errors.append(
                    {
                        "kind": "batch",
                        "texts": len(missing),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
            elapsed = time.perf_counter() - started
            self.calls.append({"kind": "batch", "texts": len(missing), "seconds": elapsed})
            for text, vector in zip(missing, vectors, strict=True):
                self.cache[text] = vector
                self.dim = len(vector)
        return [self.cache[text] for text in texts]

    def embed_one(self, text: str) -> list[float]:
        if text not in self.cache:
            started = time.perf_counter()
            try:
                vector = self.inner.embed_one(text)
            except Exception as exc:  # noqa: BLE001 - 记录真实失败原因后原样抛出
                self.errors.append(
                    {
                        "kind": "one",
                        "texts": 1,
                        "query": text,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
            elapsed = time.perf_counter() - started
            self.calls.append({"kind": "one", "texts": 1, "seconds": elapsed})
            self.cache[text] = vector
            self.dim = len(vector)
        return self.cache[text]


# ---------- 评测 ----------


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _materialize(store: MemoryStore, entries: Sequence[dict[str, Any]]) -> dict[str, str]:
    """按评测集的固定 created_at 直写 memory.jsonl，返回 条目 key -> 评测 id。"""
    store.directory.mkdir(parents=True, exist_ok=True)
    keys: dict[str, str] = {}
    lines: list[str] = []
    for item in entries:
        record = {
            "kind": item["kind"],
            "text": item["text"],
            "created_at": item["created_at"],
            "session_id": "eval-seed",
        }
        for field in ("subject", "rel", "backstory"):
            if item.get(field):
                record[field] = item[field]
        lines.append(json.dumps(record, ensure_ascii=False))
        keys[entry_key(item["kind"], item["text"], item["created_at"])] = item["id"]
    store.entries_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return keys


@contextlib.contextmanager
def _baseline_patch() -> Iterator[None]:
    """把过滤函数换成恒等函数：等价于实现混合检索之前的纯余弦排序。"""
    if not hasattr(memory, "subject_filter"):
        yield
        return
    original = memory.subject_filter
    memory.subject_filter = lambda entries, query: (list(entries), "")
    try:
        yield
    finally:
        memory.subject_filter = original


def _evaluate_mode(
    dataset: dict[str, Any], embedder: Any, *, baseline: bool, top_k: int
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="memory-eval-") as directory:
        store = MemoryStore(Path(directory), embedder=embedder)
        keys = _materialize(store, dataset["entries"])

        per_query: list[dict[str, Any]] = []
        context = _baseline_patch() if baseline else contextlib.nullcontext()
        with context:
            for item in dataset["queries"]:
                hits = store.search(item["query"], top_k=top_k)
                ids = [keys.get(hit.key(), "") for hit in hits]
                expected = list(dict.fromkeys(item["expected"]))
                per_query.append(
                    {
                        "id": item["id"],
                        "category": item["category"],
                        "expected": expected,
                        "ids": ids,
                        "recall@3": _recall(expected, ids[:3]),
                        "recall@5": _recall(expected, ids[:5]),
                        "top1": bool(ids) and ids[0] in expected,
                    }
                )
        index_bytes = (
            store.embeddings_path.stat().st_size if store.embeddings_path.is_file() else 0
        )

    single = [row for row in per_query if len(row["expected"]) == 1]
    by_category: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        grouped[row["category"]].append(row)
    for category, rows in grouped.items():
        single_rows = [row for row in rows if len(row["expected"]) == 1]
        by_category[category] = {
            "queries": len(rows),
            "recall@3": _mean(row["recall@3"] for row in rows),
            "recall@5": _mean(row["recall@5"] for row in rows),
            "disambiguation": _mean(1.0 if row["top1"] else 0.0 for row in single_rows)
            if single_rows
            else None,
        }

    return {
        "recall@3": _mean(row["recall@3"] for row in per_query),
        "recall@5": _mean(row["recall@5"] for row in per_query),
        "disambiguation": _mean(1.0 if row["top1"] else 0.0 for row in single),
        "by_category": by_category,
        "per_query": per_query,
        "index_bytes": index_bytes,
    }


def _recall(expected: Sequence[str], top: Sequence[str]) -> float:
    if not expected:
        return 0.0
    return len(set(expected) & set(top)) / len(expected)


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def run_evaluation(
    dataset: dict[str, Any],
    embedder: Any,
    *,
    top_k: int = 5,
    modes: Sequence[str] = ("baseline", "hybrid"),
) -> dict[str, Any]:
    cached = CachedEmbedder(embedder)
    results: dict[str, Any] = {}
    for mode in modes:
        results[mode] = _evaluate_mode(dataset, cached, baseline=mode == "baseline", top_k=top_k)
    return {
        "dataset": {
            "entries": len(dataset["entries"]),
            "queries": len(dataset["queries"]),
        },
        "embedder": {
            "kind": type(embedder).__name__,
            "model": getattr(embedder, "model", ""),
            "dim": cached.dim,
        },
        "embedding": _embedding_metrics(cached, results),
        "modes": results,
    }


def _embedding_metrics(cached: CachedEmbedder, results: dict[str, Any]) -> dict[str, Any]:
    """汇总真实 embedding 调用：次数、条数、总耗时、分类耗时与索引体积。"""
    query_seconds = [call["seconds"] for call in cached.calls if call["kind"] == "one"]
    batch_seconds = [call["seconds"] for call in cached.calls if call["kind"] == "batch"]
    return {
        "requests": len(cached.calls),
        "texts": sum(call["texts"] for call in cached.calls),
        "total_seconds": sum(call["seconds"] for call in cached.calls),
        "query_count": len(query_seconds),
        "query_seconds": query_seconds,
        "query_total_seconds": sum(query_seconds),
        "query_mean_ms": _mean(query_seconds) * 1000 if query_seconds else 0.0,
        "query_max_ms": max(query_seconds, default=0.0) * 1000,
        "batch_count": len(batch_seconds),
        "batch_texts": sum(call["texts"] for call in cached.calls if call["kind"] == "batch"),
        "batch_total_seconds": sum(batch_seconds),
        "index_bytes": results.get("hybrid", {}).get("index_bytes", 0),
        "failures": cached.errors,
    }


# ---------- 输出 ----------


def format_report(result: dict[str, Any], dataset_path: Path) -> str:
    lines = [
        "== 记忆消歧评测 ==",
        f"数据集：{dataset_path}（{result['dataset']['entries']} 条记忆，"
        f"{result['dataset']['queries']} 条查询）",
        f"向量后端：{result['embedder']['model']}"
        f"（dim={result['embedder']['dim'] or '未知'}，{result['embedder']['kind']}）",
        "",
    ]
    modes = list(result["modes"])
    header = f"{'指标':<14}" + "".join(f"{mode:>14}" for mode in modes)
    if len(modes) == 2:
        header += f"{'Δ(hybrid-baseline)':>20}"
    lines.append(header)
    for key, label in (
        ("recall@3", "recall@3"),
        ("recall@5", "recall@5"),
        ("disambiguation", "消歧正确率"),
    ):
        row = f"{label:<14}" + "".join(f"{result['modes'][mode][key]:>14.4f}" for mode in modes)
        if len(modes) == 2:
            delta = result["modes"]["hybrid"][key] - result["modes"]["baseline"][key]
            row += f"{delta:>+20.4f}"
        lines.append(row)

    metrics = result.get("embedding") or {}
    if metrics:
        lines.append("")
        lines.append(
            "嵌入调用："
            f"模型 {result['embedder']['model']} · dim={result['embedder']['dim'] or '未知'} · "
            f"请求 {metrics['requests']} 次 / 文本 {metrics['texts']} 条 / "
            f"嵌入总耗时 {metrics['total_seconds']:.3f}s"
            f"（记忆 {metrics['batch_texts']} 条批量 {metrics['batch_total_seconds']:.3f}s；"
            f"查询 {metrics['query_count']} 条合计 {metrics['query_total_seconds']:.3f}s，"
            f"平均 {metrics['query_mean_ms']:.0f}ms、最大 {metrics['query_max_ms']:.0f}ms）；"
            f"embeddings.jsonl {metrics['index_bytes']:,} 字节"
        )
        failures = metrics.get("failures") or []
        for failure in failures:
            target = failure.get("query") or f"{failure.get('texts', 0)} 条批量"
            lines.append(f"调用失败：{target} → {failure.get('error', '')}")

    categories = sorted(
        {category for mode in modes for category in result["modes"][mode]["by_category"]}
    )
    lines.append("")
    lines.append("分类型（recall@3 / recall@5 / 消歧正确率）：")
    for category in categories:
        cells = []
        for mode in modes:
            row = result["modes"][mode]["by_category"].get(category)
            if row is None:
                cells.append("-")
                continue
            accuracy = "n/a" if row["disambiguation"] is None else f"{row['disambiguation']:.4f}"
            cells.append(f"{row['recall@3']:.3f}/{row['recall@5']:.3f}/{accuracy}")
        lines.append(f"  {category:<10}" + "".join(f"{cell:>26}" for cell in cells))
    return "\n".join(lines)


def compare_with_frozen(result: dict[str, Any], frozen: dict[str, Any]) -> list[str]:
    """返回不一致项；空列表表示复现成功。"""
    problems: list[str] = []
    for mode, metrics in frozen.items():
        if mode not in result["modes"]:
            problems.append(f"缺少模式 {mode}")
            continue
        for key, expected in metrics.items():
            actual = result["modes"][mode].get(key)
            if actual is None or abs(float(actual) - float(expected)) > 1e-6:
                problems.append(f"{mode}.{key}: 期望 {expected}，实际 {actual}")
    return problems


def select_frozen(document: dict[str, Any], model: str) -> dict[str, Any] | None:
    """按向量后端挑选固化数字：一个文件里可以同时存离线与 bge-m3 两套。"""
    runs = document.get("runs")
    if isinstance(runs, dict):
        run = runs.get(model)
        if not isinstance(run, dict):
            return None
        frozen = run.get("frozen", run)
        return frozen if isinstance(frozen, dict) else None
    frozen = document.get("frozen", document)
    return frozen if isinstance(frozen, dict) else None


def build_embedder(kind: str) -> Any:
    if kind == "auto":
        key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get(
            "MINIAGENT_EMBEDDING_API_KEY", ""
        )
        kind = "siliconflow" if key else "offline"
    if kind == "offline":
        return LocalHashEmbedder()
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get(
        "MINIAGENT_EMBEDDING_API_KEY", ""
    )
    if not api_key:
        raise SystemExit(
            "siliconflow 后端需要 SILICONFLOW_API_KEY；"
            "只用本地可复现评测请加 --embedder offline"
        )
    return EmbeddingClient(
        api_key=api_key,
        model=os.environ.get("MINIAGENT_EMBEDDING_MODEL", EMBEDDING_DEFAULT_MODEL),
        base_url=os.environ.get("MINIAGENT_EMBEDDING_BASE_URL", EMBEDDING_DEFAULT_BASE_URL),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="记忆检索消歧评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--embedder",
        choices=("offline", "siliconflow", "auto"),
        default="offline",
        help="offline 为无网络的确定性向量；siliconflow 为真实 bge-m3（需要 key）",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="与固化的基线数字比对，不一致时返回非零退出码",
    )
    parser.add_argument("--json", type=Path, default=None, help="把完整结果写到指定文件")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)
    result = run_evaluation(dataset, build_embedder(args.embedder))
    print(format_report(result, args.dataset))

    if args.json is not None:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整结果已写入 {args.json}")

    if args.check is not None:
        document = json.loads(args.check.read_text(encoding="utf-8"))
        frozen = select_frozen(document, result["embedder"]["model"])
        if frozen is None:
            print(
                f"\n复现失败：{args.check} 里没有 {result['embedder']['model']} 的固化数字；"
                "请确认 --embedder 与文件中的 runs 对应。"
            )
            return 1
        problems = compare_with_frozen(result, frozen)
        if problems:
            print("\n复现失败：")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("\n复现成功：与固化数字一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
