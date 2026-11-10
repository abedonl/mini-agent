"""恢复摘要：ProviderError 字段、CLI 输出、SSE error 事件字段。"""

from __future__ import annotations

import http.client
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from miniagent.cli import format_error, main
from miniagent.config import Config
from miniagent.errors import ProviderError, RecoverySummary
from miniagent.llm import OpenAICompatProvider
from miniagent.reliability import RetryPolicy
from miniagent.web import create_server, error_payload


def make_provider(handler: Any, attempts: int = 3) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="test-key",
        model="test-model",
        base_url="http://test.local/v1",
        transport=httpx.MockTransport(handler),
        rate_limit_rpm=0,
        retry_policy=RetryPolicy(
            max_attempts=attempts,
            sleep=lambda _: None,
            rand=lambda: 0.0,
        ),
    )


def test_retry_exhaustion_carries_attempts_and_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    provider = make_provider(handler)

    with pytest.raises(ProviderError) as info:
        provider.chat([], [])

    summary = info.value.recovery
    assert summary is not None
    assert summary.attempts == 3
    assert summary.last_status == 503
    assert "重试 2 次" in info.value.recovery_text
    assert "HTTP 503" in info.value.recovery_text


def test_timeout_summary_has_error_type_without_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    provider = make_provider(handler, attempts=2)

    with pytest.raises(ProviderError) as info:
        provider.chat([], [])

    summary = info.value.recovery
    assert summary is not None
    assert summary.attempts == 2
    assert summary.last_status is None
    assert "超时" in info.value.recovery_text


def test_format_error_appends_recovery_section() -> None:
    exc = ProviderError(
        "模型请求失败 HTTP 503",
        recovery=RecoverySummary(
            attempts=3,
            last_status=503,
            last_error="模型请求失败 HTTP 503：overloaded",
            degraded=["记忆门降级放行（provider_error）", "上下文压缩降级为硬裁（summary_failed）"],
        ),
    )

    text = format_error(exc)

    assert text.startswith("模型请求失败 HTTP 503")
    assert "[recovery]" in text
    assert "重试 2 次" in text
    assert "记忆门降级放行" in text
    assert "硬裁" in text


def test_format_error_without_recovery_stays_one_line() -> None:
    assert format_error(ProviderError("boom")) == "boom"


def test_cli_prints_recovery_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path
) -> None:
    class FailingAgent:
        session = None

        def run(self, text: str) -> None:
            raise ProviderError(
                "模型请求失败 HTTP 503",
                recovery=RecoverySummary(attempts=3, last_status=503),
            )

    monkeypatch.setattr("miniagent.cli.make_agent", lambda *args, **kwargs: FailingAgent())

    code = main(
        [
            "-p",
            "hi",
            "--mock",
            "--config",
            str(tmp_path / "missing.toml"),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--workdir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert code == 3
    assert "[recovery]" in captured.err
    assert "HTTP 503" in captured.err


@pytest.fixture()
def web(tmp_path: Path) -> Any:
    server = create_server(
        replace(Config(), memory_enabled=False),
        port=0,
        session_dir=tmp_path / "sessions",
        log_dir=str(tmp_path / "logs"),
        mock=True,
        allow_dirs=[str(tmp_path)],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(server: Any, method: str, path: str, body: Any = None) -> tuple[int, bytes]:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    payload = None
    headers: dict[str, str] = {}
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def _parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if name and data_lines:
            events.append((name, json.loads("\n".join(data_lines))))
    return events


def test_sse_error_event_carries_recovery_fields(web: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingAgent:
        def __init__(self) -> None:
            self.on_event = None

        def run(self, text: str) -> None:
            raise ProviderError(
                "模型请求失败 HTTP 503",
                recovery=RecoverySummary(
                    attempts=3,
                    last_status=503,
                    last_error="HTTP 503：overloaded",
                    degraded=["工具依赖判定降级为整轮串行"],
                ),
            )

    monkeypatch.setattr("miniagent.web.make_agent", lambda *args, **kwargs: FailingAgent())

    status, body = _request(web, "POST", "/api/sessions")
    assert status == 200
    session_id = json.loads(body)["id"]

    status, body = _request(
        web,
        "POST",
        f"/api/sessions/{session_id}/messages",
        {"message": "hi"},
    )
    assert status == 200
    events = dict(_parse_sse(body.decode("utf-8")))

    assert "error" in events
    payload = events["error"]
    assert payload["attempts"] == 3
    assert payload["last_status"] == 503
    assert payload["degraded"] == ["工具依赖判定降级为整轮串行"]
    assert "重试 2 次" in payload["recovery"]


def test_error_payload_redacts_secret() -> None:
    exc = ProviderError("bad key sk-abc123", recovery=RecoverySummary(attempts=1))

    payload = error_payload(exc)

    assert "sk-abc123" not in payload["message"]
    assert payload["attempts"] == 1
