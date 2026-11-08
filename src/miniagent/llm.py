"""模型层：消息模型、LLMProvider 契约、Mock 与 OpenAI 兼容实现。

流式与非流式共用 chat() 一个入口；重试与限流只发生在真实 Provider 的请求层。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from miniagent.errors import ConfigError, ProviderError
from miniagent.reliability import RetryPolicy, TokenBucket, is_retryable_status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，交给工具层校验后再解析


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict[str, Any]:
        """转成 OpenAI 兼容协议的消息结构。"""
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None or self.role != "assistant":
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload

    def to_dict(self) -> dict[str, Any]:
        """会话文件里的一行；字段与 API 结构保持接近，便于排查。"""
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        tool_calls = [
            ToolCall(
                id=call.get("id", f"call_{index}"),
                name=call.get("name", ""),
                arguments=call.get("arguments") or "{}",
            )
            for index, call in enumerate(data.get("tool_calls") or [])
        ]
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


TextCallback = Callable[[str], None]


class LLMProvider(Protocol):
    """模型提供方的最小契约：一次请求换一次完整响应。

    传入 on_text 时，实现应边接收边回调文本增量；返回值仍是完整响应，
    因此非流式与流式共用同一条 Agent 循环。
    """

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        on_text: TextCallback | None = None,
    ) -> LLMResponse: ...


class MockProvider:
    """离线实现：默认回显，也可按脚本依次返回预置响应（测试用）。"""

    def __init__(self, script: list[LLMResponse] | None = None) -> None:
        self._script = list(script or [])
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._script:
            response = self._script.pop(0)
        else:
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            names = "、".join(t["name"] for t in tools)
            response = LLMResponse(
                content=(
                    f"[mock] 收到：{last_user}\n"
                    f"[mock] 离线回显，未调用真实模型；本次上下文 {len(messages)} 条消息。"
                    f"可用工具：{names}。"
                    "设置 MINIAGENT_PROVIDER=openai 并配置 API 后走真实链路。"
                )
            )
        if on_text is not None and response.content:
            for chunk in _chunk_text(response.content):
                on_text(chunk)
        return response


def _chunk_text(text: str, size: int = 16) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


class _RetryableError(Exception):
    """内部信号：这次失败值得重试，不直接暴露给上层。"""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """只解析秒数形式；HTTP 日期形式忽略，走默认指数退避。"""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _raise_http_error(status_code: int, body: str, retry_after: float | None) -> None:
    message = f"模型请求失败 HTTP {status_code}：{body[:300]}"
    if is_retryable_status(status_code):
        raise _RetryableError(message, retry_after=retry_after)
    raise ProviderError(message)


class OpenAICompatProvider:
    """任何 OpenAI 兼容端点：只改 base_url / api_key / model。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
        rate_limit_rpm: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        limiter: TokenBucket | None = None,
    ) -> None:
        if not api_key:
            raise ConfigError(
                "未配置模型 API 密钥：请设置 MINIAGENT_API_KEY 或 OPENAI_API_KEY；"
                "只想离线体验可加 --mock"
            )
        self._api_key = api_key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout = timeout
        self._transport = transport
        self._retry = retry_policy or RetryPolicy(max_attempts=max(1, int(max_retries)))
        self._limiter = limiter or TokenBucket(rate_per_minute=rate_limit_rpm)

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_api() for m in messages],
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        if on_text is None:
            return self._with_retry(lambda: self._chat_once(payload))

        emitted = False

        def emit(chunk: str) -> None:
            nonlocal emitted
            emitted = True
            on_text(chunk)

        try:
            return self._with_retry(
                lambda: self._chat_stream(payload, emit),
                can_retry=lambda: not emitted,
            )
        except ProviderError as exc:
            if emitted:
                raise ProviderError(f"流式输出中断，已保留部分内容：{exc}") from exc
            # 一个字都没收到：沿用降级策略，再试一次非流式
            return self._with_retry(lambda: self._chat_once(payload))

    def _with_retry(
        self,
        call: Callable[[], LLMResponse],
        *,
        can_retry: Callable[[], bool] | None = None,
    ) -> LLMResponse:
        """每次尝试前先过令牌桶；只在可重试错误上退避重来。"""
        attempts = max(1, self._retry.max_attempts)
        last: _RetryableError | None = None
        for attempt in range(1, attempts + 1):
            self._limiter.acquire()
            try:
                return call()
            except _RetryableError as exc:
                last = exc
                if attempt >= attempts or (can_retry is not None and not can_retry()):
                    break
                self._retry.sleep(self._retry.delay_for(attempt, exc.retry_after))
        assert last is not None
        raise ProviderError(str(last)) from last

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _chat_once(self, payload: dict[str, Any]) -> LLMResponse:
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(self._url, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise _RetryableError(f"模型请求超时：{exc}") from exc
        except httpx.TransportError as exc:
            raise _RetryableError(f"模型网络错误：{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"模型请求失败：{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            _raise_http_error(
                response.status_code, response.text, _retry_after_seconds(response)
            )

        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"模型响应格式不符合预期：{response.text[:300]}") from exc

        raw_calls = message.get("tool_calls") or []
        tool_calls = []
        for index, raw in enumerate(raw_calls):
            function = raw.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{index}",
                    name=function.get("name", ""),
                    arguments=function.get("arguments") or "{}",
                )
            )
        return LLMResponse(content=message.get("content"), tool_calls=tool_calls)

    def _chat_stream(self, payload: dict[str, Any], on_text: TextCallback) -> LLMResponse:
        payload = {**payload, "stream": True}
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                with client.stream(
                    "POST", self._url, headers=self._headers(), json=payload
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", "replace")
                        _raise_http_error(
                            response.status_code, body, _retry_after_seconds(response)
                        )
                    return self._consume_sse(response.iter_lines(), on_text)
        except httpx.TimeoutException as exc:
            raise _RetryableError(f"模型请求超时：{exc}") from exc
        except httpx.TransportError as exc:
            raise _RetryableError(f"流式读取中断：{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"模型请求失败：{type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _consume_sse(lines: Any, on_text: TextCallback) -> LLMResponse:
        """解析 OpenAI 兼容的 SSE：文本增量直接回调，tool_calls 按 index 拼接。"""
        parts: list[str] = []
        pending: dict[int, dict[str, str]] = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"SSE 数据不是合法 JSON：{data[:120]}") from exc

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                parts.append(text)
                on_text(text)

            for raw in delta.get("tool_calls") or []:
                index = int(raw.get("index", 0))
                slot = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if raw.get("id"):
                    slot["id"] = raw["id"]
                function = raw.get("function") or {}
                if function.get("name"):
                    slot["name"] += function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]

        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )
            for index, slot in sorted(pending.items())
        ]
        return LLMResponse(content="".join(parts) or None, tool_calls=tool_calls)


def dumps_args(arguments: str, limit: int = 160) -> str:
    """把工具参数压成一行摘要，用于终端显示。"""
    try:
        compact = json.dumps(json.loads(arguments), ensure_ascii=False)
    except ValueError:
        compact = arguments
    return compact if len(compact) <= limit else compact[:limit] + "..."
