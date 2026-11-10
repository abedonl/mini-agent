"""记忆检索的向量通道：SiliconFlow embedding 客户端、纯 Python 余弦与向量索引。

设计取舍：
- 只依赖 httpx，不引入 numpy/faiss：几百条记忆暴力算余弦足够快。
- 向量与记忆条目分文件存（embeddings.jsonl），每条记录带模型名与维度；
  模型或维度不匹配时整份索引重建，重建失败则本次检索返回空结果。
- 失败不抛出：检索是"尽力而为"的增强项，一次网络抖动不该影响写入与蒸馏。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from miniagent.reliability import is_retryable_status

EMBEDDING_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_DEFAULT_MODEL = "BAAI/bge-m3"
EMBEDDING_TIMEOUT = 5.0
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_RETRY_DELAY = 0.3


class EmbeddingError(Exception):
    """embedding 请求失败（超时、网络、非 2xx、响应格式不对）。"""


_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def _redact(text: str) -> str:
    """错误信息里若回显了 key，统一遮盖，避免被日志或终端带出去。"""
    return _SECRET.sub("sk-***", text)


def entry_key(kind: str, text: str, created_at: str) -> str:
    """记忆条目的稳定 ID：同一条目的 kind/text/created_at 不变。

    不直接用列表下标：重建索引或并发写入时下标会漂移，向量就挂错了条目。
    """
    raw = f"{kind}\0{text}\0{created_at}".encode()
    return hashlib.sha256(raw).hexdigest()


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """纯 Python 余弦相似度；维度不一致时返回 0，避免索引脏数据拖垮检索。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=False):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / ((left_norm**0.5) * (right_norm**0.5))


@dataclass
class EmbeddingClient:
    """OpenAI 兼容的 /embeddings 客户端；只重试一次，避免拖慢检索。"""

    api_key: str
    model: str = EMBEDDING_DEFAULT_MODEL
    base_url: str = EMBEDDING_DEFAULT_BASE_URL
    timeout: float = EMBEDDING_TIMEOUT
    transport: httpx.BaseTransport | None = None
    retry_delay: float = EMBEDDING_RETRY_DELAY

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.timeout = max(0.1, float(self.timeout))

    @property
    def url(self) -> str:
        return f"{self.base_url}/embeddings"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量取向量；失败时抛 EmbeddingError，由调用方决定降级。"""
        if not texts:
            return []
        payload = {"model": self.model, "input": list(texts)}
        last_error = ""
        for attempt in (1, 2):
            try:
                return self._post(payload)
            except _RetryableEmbedding as exc:
                last_error = str(exc)
                if attempt == 1:
                    time.sleep(self.retry_delay)
        raise EmbeddingError(last_error or "embedding 请求失败")

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed([text])
        if not vectors:
            raise EmbeddingError("embedding 响应为空")
        return vectors[0]

    def _post(self, payload: dict[str, Any]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(self.url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise _RetryableEmbedding(f"embedding 请求超时：{exc}") from exc
        except httpx.TransportError as exc:
            raise _RetryableEmbedding(
                f"embedding 网络错误：{type(exc).__name__}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding 请求失败：{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            message = (
                f"embedding 请求失败 HTTP {response.status_code}："
                f"{_redact(response.text[:300])}"
            )
            if is_retryable_status(response.status_code):
                raise _RetryableEmbedding(message)
            raise EmbeddingError(message)

        try:
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            vectors = [list(map(float, item["embedding"])) for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(
                f"embedding 响应格式不符合预期：{_redact(response.text[:300])}"
            ) from exc
        if len(vectors) != len(payload["input"]):
            raise EmbeddingError(
                f"embedding 返回数量不一致：期望 {len(payload['input'])}，实际 {len(vectors)}"
            )
        return vectors


class _RetryableEmbedding(EmbeddingError):
    """内部信号：只用于标记值得重试的错误。"""


@dataclass
class IndexSync:
    vectors: dict[str, list[float]]
    rebuilt: bool = False
    added: int = 0
    degraded: bool = False
    dim: int | None = None


class EmbeddingIndex:
    """embeddings.jsonl：一行一条向量，附模型名与维度，便于自检与重建。"""

    VERSION = 1

    def __init__(self, path: Path, model: str) -> None:
        self.path = path
        self.model = model

    def load(self) -> tuple[dict[str, list[float]], bool]:
        """返回 (向量表, 是否需要重建)。

        模型名不一致、维度不一致、记录缺字段，都按"整份不信"处理：
        向量是派生数据，重建比修修补补更可靠。
        """
        if not self.path.is_file():
            return {}, False
        vectors: dict[str, list[float]] = {}
        expected_dim: int | None = None
        stale = False
        try:
            fp = self.path.open(encoding="utf-8", errors="replace")
        except OSError:
            return {}, True
        with fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stale = True
                    continue
                if not isinstance(record, dict):
                    stale = True
                    continue
                if record.get("v") != self.VERSION or record.get("model") != self.model:
                    stale = True
                    continue
                key = str(record.get("id", ""))
                dim = record.get("dim")
                vector = record.get("vector")
                if (
                    not key
                    or not isinstance(dim, int)
                    or not isinstance(vector, list)
                    or len(vector) != dim
                ):
                    stale = True
                    continue
                try:
                    values = [float(value) for value in vector]
                except (TypeError, ValueError):
                    stale = True
                    continue
                if expected_dim is None:
                    expected_dim = dim
                elif expected_dim != dim:
                    stale = True
                vectors[key] = values
        if stale:
            return {}, True
        return vectors, False

    def sync(
        self,
        entries: Iterable[tuple[str, str]],
        embedder: EmbeddingClient,
        *,
        force: bool = False,
    ) -> IndexSync:
        """补齐缺失向量；需要重建时把已有向量全部作废。

        entries 是 (条目 ID, 文本) 序列。只请求缺失的条目，
        因此正常检索每次只有"查询一条"的网络调用。
        force=True 用于查询向量维度与索引不一致时强制整份重建。
        """
        vectors, rebuilt = self.load()
        if force:
            rebuilt = True
        if rebuilt:
            vectors = {}
        wanted = [(key, text) for key, text in entries if key not in vectors]
        # 同一条目在文件里出现多次时只保留第一次，避免重复计费
        seen: set[str] = set()
        missing: list[tuple[str, str]] = []
        for key, text in wanted:
            if key in seen:
                continue
            seen.add(key)
            missing.append((key, text))

        added = 0
        degraded = False
        dim = len(next(iter(vectors.values()))) if vectors else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
            batch = missing[start : start + EMBEDDING_BATCH_SIZE]
            try:
                batch_vectors = embedder.embed([text for _, text in batch])
            except EmbeddingError:
                degraded = True
                break
            for (key, _), vector in zip(batch, batch_vectors, strict=False):
                vectors[key] = vector
                added += 1
                dim = len(vector)
            try:
                self._save(vectors)
            except OSError:
                degraded = True
                break
        if rebuilt and not missing:
            try:
                self._save(vectors)
            except OSError:
                degraded = True
        return IndexSync(
            vectors=vectors,
            rebuilt=rebuilt,
            added=added,
            degraded=degraded,
            dim=dim,
        )

    def _save(self, vectors: dict[str, list[float]]) -> None:
        """先写临时文件再原子替换，避免进程被杀留下半截索引。"""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            for key, vector in vectors.items():
                record = {
                    "v": self.VERSION,
                    "id": key,
                    "model": self.model,
                    "dim": len(vector),
                    "vector": vector,
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

