"""可靠性：错误分类、令牌桶限流、指数退避重试。

单进程内实现，不依赖 Redis 或消息队列：
- 限流：每次模型请求前取一个令牌，超出速率就阻塞等待。
- 重试：只作用于模型请求；工具调用永不自动重试（副作用不可重复）。
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from miniagent.errors import ProviderError

RETRYABLE_STATUS = frozenset({408, 429})


def is_retryable_status(status_code: int) -> bool:
    """408/429/5xx 是可重试的服务端或限流问题；其他 4xx 是请求本身有错，立即失败。"""
    return status_code in RETRYABLE_STATUS or status_code >= 500


@dataclass
class TokenBucket:
    """线程安全的令牌桶，按每分钟请求数匀速补充。

    rate_per_minute <= 0 表示关闭限流（Mock 默认不启用）。
    """

    rate_per_minute: float = 60.0
    capacity: int = 1
    max_wait: float = 30.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _tokens: float = field(init=False)
    _last: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.rate_per_minute = max(0.0, float(self.rate_per_minute))
        self.capacity = max(1, int(self.capacity))
        self._tokens = float(self.capacity)
        self._last = self.clock()

    @property
    def enabled(self) -> bool:
        return self.rate_per_minute > 0

    def acquire(self) -> None:
        """阻塞到拿到令牌；预计等待超过 max_wait 就报错，避免进程无期限挂住。"""
        if not self.enabled:
            return
        per_second = self.rate_per_minute / 60.0
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                elapsed = max(0.0, now - self._last)
                self._last = now
                self._tokens = min(float(self.capacity), self._tokens + elapsed * per_second)
                if self._tokens >= 1.0 - 1e-9:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / per_second
            if waited + wait > self.max_wait:
                raise ProviderError(
                    f"模型请求被限流：还需等待 {wait:.1f}s，超过上限 {self.max_wait:.0f}s"
                )
            self.sleep(wait)
            waited += wait


@dataclass(frozen=True)
class RetryPolicy:
    """指数退避 + 抖动；服务端给了 Retry-After 时优先听它的。"""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    sleep: Callable[[float], None] = time.sleep
    rand: Callable[[], float] = random.random

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """attempt 从 1 开始（第几次失败）；返回下一次尝试前的等待秒数。"""
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.max_delay)
        base = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return min(base + base * 0.5 * self.rand(), self.max_delay)
