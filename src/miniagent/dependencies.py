"""同轮工具依赖：路径硬约束 + 小模型边判定 + 拓扑分层。

路径规则先给出不可推翻的串行边；模型只负责补充语义依赖。任何判定失败
（超时、不可解析、索引越界、空回答）都退回整轮串行——把依赖误判成并行
会造成静默的数据错误，比慢更严重。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field

from miniagent.llm import LLMProvider, Message, ToolCall
from miniagent.scheduling import CallPlan

DEPS_MODES = ("off", "model")
DEFAULT_JUDGE_TIMEOUT = 2.0
SUMMARY_KEYS = ("path", "file", "target", "target_file")
JUDGE_PROMPT = (
    "你是工具依赖判定器。给定同一轮的工具调用列表，判断哪些调用之间存在先后依赖。"
    '只输出 JSON：{"edges": [{"from": 0, "to": 1}]}，表示 to 依赖 from、from 必须先执行。'
    '没有依赖时输出 {"edges": []}。不要输出解释，不要使用列表之外的索引。'
)


@dataclass
class JudgeResult:
    edges: list[tuple[int, int]] = field(default_factory=list)
    ok: bool = True
    cached: bool = False
    latency_ms: int = 0
    reason: str = ""  # timeout | provider_error | parse_error | out_of_range | empty


@dataclass
class LayerPlan:
    layers: list[list[int]]
    edges: list[tuple[int, int]] = field(default_factory=list)
    path_edges: int = 0
    model_edges: int = 0
    dropped: int = 0
    mode: str = "model"
    judge_ms: int = 0
    cached: bool = False
    reason: str = ""

    def detail(self) -> str:
        sizes = "+".join(str(len(layer)) for layer in self.layers) or "-"
        cache = "hit" if self.cached else "miss"
        return (
            f"mode={self.mode} layers={len(self.layers)} sizes={sizes} "
            f"edges={len(self.edges)} path_edges={self.path_edges} "
            f"model_edges={self.model_edges} dropped={self.dropped} "
            f"judge_ms={self.judge_ms} cache={cache} reason={self.reason or '-'}"
        )


class DependencyJudge:
    """一次只缓存上一轮判定：参数摘要完全相同就直接复用，避免重复付费。"""

    def __init__(self, provider: LLMProvider, *, timeout: float = DEFAULT_JUDGE_TIMEOUT) -> None:
        self.provider = provider
        self.timeout = timeout
        self._cache_key = ""
        self._cache_edges: list[tuple[int, int]] = []

    def judge(self, calls: list[ToolCall]) -> JudgeResult:
        summaries = [summarize_call(call) for call in calls]
        key = "\n".join(summaries)
        if key == self._cache_key:
            return JudgeResult(edges=list(self._cache_edges), ok=True, cached=True)

        started = time.perf_counter()
        messages = [
            Message(role="system", content=JUDGE_PROMPT),
            Message(
                role="user",
                content="\n".join(f"{index}. {line}" for index, line in enumerate(summaries)),
            ),
        ]
        try:
            response = self._chat(messages)
        except TimeoutError:
            return JudgeResult(ok=False, latency_ms=_elapsed_ms(started), reason="timeout")
        except Exception:  # noqa: BLE001 - 判定失败一律降级，不能影响主请求
            return JudgeResult(ok=False, latency_ms=_elapsed_ms(started), reason="provider_error")

        edges, error = parse_edges(response.content or "", len(calls))
        elapsed = _elapsed_ms(started)
        if error:
            return JudgeResult(ok=False, latency_ms=elapsed, reason=error)

        self._cache_key = key
        self._cache_edges = edges
        return JudgeResult(edges=edges, cached=False, latency_ms=elapsed)

    def _chat(self, messages: list[Message]):
        if self.timeout <= 0:
            return self.provider.chat(messages, [])
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.provider.chat, messages, [])
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError("judge timeout") from exc
        finally:
            # 卡住的请求无法真正取消，只在等待侧超时，主流程不陪它等
            pool.shutdown(wait=False)


def summarize_call(call: ToolCall) -> str:
    """只给工具名与参数摘要，绝不塞工具输出或文件全文。"""
    args = _args(call)
    parts: list[str] = []
    for key in SUMMARY_KEYS:
        if key in args:
            parts.append(f"{key}={_clip(str(args[key]), 80)}")
    command = args.get("command")
    if command:
        parts.append(f"command={_clip(str(command).splitlines()[0], 100)}")
    if not parts:
        keys = ",".join(sorted(args)[:6])
        parts.append(f"args={keys or '-'}")
    return f"{call.name} {' '.join(parts)}"


def parse_edges(content: str, count: int) -> tuple[list[tuple[int, int]], str]:
    """严格解析 edges；任何可疑输出都返回错误原因，由上层整轮串行。"""
    text = content.strip()
    if not text:
        return [], "empty"
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()

    data = _load_json(text)
    if data is None:
        return [], "parse_error"
    if not isinstance(data, dict) or not isinstance(data.get("edges"), list):
        return [], "parse_error"

    edges: list[tuple[int, int]] = []
    for item in data["edges"]:
        if not isinstance(item, dict):
            return [], "parse_error"
        source, target = item.get("from"), item.get("to")
        if not _is_index(source) or not _is_index(target):
            return [], "parse_error"
        # 自环留给无环检测：它同样会触发整轮串行，但原因是 cycle 而不是越界
        if not (0 <= source < count and 0 <= target < count):
            return [], "out_of_range"
        edges.append((source, target))
    return list(dict.fromkeys(edges)), ""


def merge_edges(
    path_edges: list[tuple[int, int]],
    model_edges: list[tuple[int, int]],
    count: int,
) -> tuple[list[tuple[int, int]], int, bool]:
    """路径边优先：反向覆盖路径边的模型边直接丢弃；剩余图有环则返回 cycle。"""
    merged = list(dict.fromkeys(path_edges))
    hard = set(merged)
    dropped = 0
    for edge in model_edges:
        if edge in hard or edge in merged:
            continue
        if (edge[1], edge[0]) in hard:
            dropped += 1
            continue
        merged.append(edge)
    return merged, dropped, _has_cycle(count, merged)


def build_layers(count: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """最长路径分层：同层互不依赖，层间串行；孤立节点在第 0 层。"""
    adjacency: list[list[int]] = [[] for _ in range(count)]
    indegree = [0] * count
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1

    level = [0] * count
    queue = [index for index in range(count) if indegree[index] == 0]
    while queue:
        node = queue.pop(0)
        for target in adjacency[node]:
            level[target] = max(level[target], level[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    layers: list[list[int]] = []
    for index in range(count):
        while len(layers) <= level[index]:
            layers.append([])
        layers[level[index]].append(index)
    return layers


def plan_layers(calls: list[ToolCall], plan: CallPlan, judge: DependencyJudge) -> LayerPlan:
    count = len(calls)
    hard = path_edges(plan)
    result = judge.judge(calls)
    common = {"mode": "model", "path_edges": len(hard), "judge_ms": result.latency_ms}
    if not result.ok:
        return LayerPlan(
            layers=_serial_layers(count),
            edges=hard,
            reason=f"judge_{result.reason}",
            **common,
        )

    merged, dropped, cycle = merge_edges(hard, result.edges, count)
    if cycle:
        return LayerPlan(
            layers=_serial_layers(count),
            edges=merged,
            model_edges=len(result.edges),
            dropped=dropped,
            reason="cycle",
            cached=result.cached,
            **common,
        )
    return LayerPlan(
        layers=build_layers(count, merged),
        edges=merged,
        model_edges=len(result.edges),
        dropped=dropped,
        cached=result.cached,
        **common,
    )


def path_edges(plan: CallPlan) -> list[tuple[int, int]]:
    """路径规则已判定必须串行的相邻调用，连成不可推翻的链。"""
    if plan.is_barrier:
        return []
    ordered = plan.serial
    return list(zip(ordered, ordered[1:], strict=False))


def _serial_layers(count: int) -> list[list[int]]:
    """降级表示：每个调用单独一层，执行器会逐层串行，不会被并成一波并发。"""
    return [[index] for index in range(count)]


def _load_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _is_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _has_cycle(count: int, edges: list[tuple[int, int]]) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(count)]
    indegree = [0] * count
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    queue = [index for index in range(count) if indegree[index] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited != count


def _args(call: ToolCall) -> dict[str, object]:
    try:
        data = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
