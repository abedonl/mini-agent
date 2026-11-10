"""本地网页界面：会话 API、SSE 流式输出、trace 白名单与输入校验。"""

from __future__ import annotations

import http.client
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from miniagent.config import Config
from miniagent.hitl import ToolRequest
from miniagent.llm import LLMResponse, Message, MockProvider, ToolCall
from miniagent.logging import TraceLogger
from miniagent.session import Session
from miniagent.web import TRACE_PUBLIC_FIELDS, WebInteraction, create_server


@pytest.fixture()
def web(tmp_path: Path) -> Any:
    session_dir = tmp_path / "sessions"
    log_dir = tmp_path / "logs"
    server = create_server(
        replace(Config(), memory_enabled=False),
        port=0,
        session_dir=session_dir,
        log_dir=str(log_dir),
        mock=True,
        allow_dirs=[str(tmp_path)],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, session_dir, log_dir
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(
    server: Any,
    method: str,
    path: str,
    body: Any = None,
    raw: bytes | None = None,
) -> tuple[int, str, bytes]:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    headers: dict[str, str] = {}
    payload = raw
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    content_type = response.getheader("Content-Type") or ""
    conn.close()
    return response.status, content_type, data


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
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


def create_session(server: Any) -> str:
    status, _, body = request(server, "POST", "/api/sessions")
    assert status == 200
    session_id = json.loads(body)["id"]
    assert isinstance(session_id, str)
    return session_id


def test_index_html_has_key_elements_and_no_secret(web: Any) -> None:
    server, _, _ = web
    status, content_type, body = request(server, "GET", "/")
    text = body.decode("utf-8")

    assert status == 200
    assert "text/html" in content_type
    assert "charset=utf-8" in content_type
    assert 'id="session-list"' in text
    assert 'id="messages"' in text
    assert 'id="composer"' in text
    assert 'id="set-plan-mode"' in text
    assert 'id="set-plan-verify"' in text
    assert 'id="set-hitl"' in text
    assert 'id="set-workdir"' in text
    assert 'id="set-workdir-apply"' in text
    assert 'id="workdir-roots"' in text
    assert "发送" in text
    assert "sk-" not in text
    assert "api_key" not in text


def test_create_and_list_sessions(web: Any) -> None:
    server, _, _ = web
    status, content_type, body = request(server, "GET", "/api/sessions")
    assert status == 200
    assert "charset=utf-8" in content_type
    assert json.loads(body) == []

    session_id = create_session(server)

    status, _, body = request(server, "GET", "/api/sessions")
    items = json.loads(body)
    assert status == 200
    assert [item["id"] for item in items] == [session_id]
    assert items[0]["title"] == ""
    assert items[0]["message_count"] == 0
    assert isinstance(items[0]["updated_at"], float)


def test_session_messages_restore_tool_calls(web: Any) -> None:
    server, session_dir, _ = web
    session_id = create_session(server)

    session = Session.create(session_id=session_id, directory=session_dir)
    session.append(Message(role="system", content="系统提示"))
    session.append(Message(role="user", content="看看 a.txt"))
    session.append(
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments='{"path": "a.txt"}')],
        )
    )
    session.append(
        Message(role="tool", content="文件内容", tool_call_id="c1", name="read_file")
    )

    status, _, body = request(server, "GET", f"/api/sessions/{session_id}")
    data = json.loads(body)

    assert status == 200
    assert data["id"] == session_id
    messages = data["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert messages[1]["content"] == "看看 a.txt"
    assert messages[2]["tool_calls"] == [
        {"id": "c1", "name": "read_file", "arguments": '{"path": "a.txt"}'}
    ]
    assert messages[3]["name"] == "read_file"
    assert messages[3]["tool_call_id"] == "c1"
    assert messages[3]["content"] == "文件内容"


def test_send_message_streams_sse(web: Any) -> None:
    server, _, _ = web
    session_id = create_session(server)

    status, content_type, body = request(
        server, "POST", f"/api/sessions/{session_id}/messages", {"message": "你好"}
    )
    text = body.decode("utf-8")

    assert status == 200
    assert "text/event-stream" in content_type
    assert "charset=utf-8" in content_type
    events = parse_sse(text)
    names = [name for name, _ in events]
    assert "meta" in names
    assert "text" in names
    assert "done" in names
    assert events[0] == ("meta", {"session_id": session_id})

    done = next(data for name, data in events if name == "done")
    assert done["ok"] is True
    assert done["turns"] == 1
    assert "你好" in done["text"]
    assert "sk-" not in text

    status, _, body = request(server, "GET", "/api/sessions")
    items = json.loads(body)
    assert items[0]["title"] == "你好"
    assert items[0]["message_count"] == 3  # system + user + assistant


def test_bad_requests_are_rejected(web: Any) -> None:
    server, _, _ = web
    session_id = create_session(server)

    status, _, body = request(server, "GET", "/api/sessions/bad.id")
    assert status == 400
    assert "error" in json.loads(body)

    status, _, _ = request(
        server, "POST", f"/api/sessions/{session_id}/messages", {"message": "   "}
    )
    assert status == 400

    status, _, _ = request(
        server,
        "POST",
        f"/api/sessions/{session_id}/messages",
        {"message": "x" * 10001},
    )
    assert status == 400

    status, _, _ = request(
        server,
        "POST",
        f"/api/sessions/{session_id}/messages",
        raw=b"not json",
    )
    assert status == 400

    status, _, _ = request(server, "GET", "/api/unknown")
    assert status == 404

    status, _, _ = request(server, "GET", "/api/sessions/ghost")
    assert status == 404


def test_post_body_is_consumed_on_keep_alive(web: Any) -> None:
    """POST 带 body 后复用同一连接，请求体不能残留到下一个请求行。"""
    server, _, _ = web
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request(
            "POST",
            "/api/sessions",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        assert response.status == 200

        conn.request("GET", "/api/sessions")
        response = conn.getresponse()
        body = response.read()
        assert response.status == 200
        assert len(json.loads(body)) == 1
    finally:
        conn.close()


def test_trace_returns_whitelist_and_redacts_detail(web: Any) -> None:
    server, _, log_dir = web
    session_id = create_session(server)
    request(server, "POST", f"/api/sessions/{session_id}/messages", {"message": "hi"})
    TraceLogger(session_id, log_dir).log(
        "request", turn=1, detail="api_key=sk-abcdef123456"
    )

    status, _, body = request(server, "GET", f"/api/sessions/{session_id}/trace")
    records = json.loads(body)
    raw = body.decode("utf-8")

    assert status == 200
    assert records
    for record in records:
        assert tuple(record) == TRACE_PUBLIC_FIELDS
        assert "session_id" not in record
    assert "sk-abcdef123456" not in raw
    marked = next(record for record in records if "sk-***" in record["detail"])
    assert marked["event"] == "request"

    status, _, body = request(
        server, "GET", f"/api/sessions/{session_id}/trace?turn=1"
    )
    assert status == 200
    assert all(record["turn"] == 1 for record in json.loads(body))

    status, _, _ = request(
        server, "GET", f"/api/sessions/{session_id}/trace?turn=abc"
    )
    assert status == 400


# ---- 每会话开关 -----------------------------------------------------------


def test_settings_defaults_and_persist(web: Any) -> None:
    server, session_dir, _ = web
    session_id = create_session(server)
    settings_path = session_dir / f"{session_id}.settings.json"
    roots = [str(root) for root in server.authorized_roots]
    default_workdir = str(Path.cwd().resolve())
    assert str(server.default_workdir) == default_workdir

    status, content_type, body = request(
        server, "GET", f"/api/sessions/{session_id}/settings"
    )
    assert status == 200
    assert "charset=utf-8" in content_type
    assert json.loads(body) == {
        "plan_mode": "off",
        "plan_verify": "off",
        "hitl": "off",
        "workdir": default_workdir,
        "roots": roots,
    }
    assert not settings_path.exists(), "只读默认值不应落盘"

    status, _, body = request(
        server,
        "POST",
        f"/api/sessions/{session_id}/settings",
        {"plan_mode": "on", "hitl": "on"},
    )
    assert status == 200
    assert json.loads(body) == {
        "plan_mode": "on",
        "plan_verify": "off",
        "hitl": "on",
        "workdir": default_workdir,
        "roots": roots,
    }
    assert settings_path.is_file()

    # 换一个服务进程（同会话目录）仍读得到，等价于刷新页面
    other = create_server(
        replace(Config(), memory_enabled=False),
        port=0,
        session_dir=session_dir,
        log_dir=None,
        mock=True,
        allow_dirs=[str(session_dir.parent)],
    )
    try:
        settings = other.session_settings(session_id)
        assert settings["plan_mode"] == "on"
        assert settings["plan_verify"] == "off"
        assert settings["hitl"] == "on"
        assert settings["workdir"] == str(other.default_workdir)
        effective = other.effective_config(session_id)
        assert effective.plan_mode == "on"
        assert effective.plan_verify == "off"
        assert effective.hitl == "on"
        assert effective.workdir == str(other.default_workdir)
        # 基础配置不因会话开关被改写
        assert other.config.plan_mode == "off"
        assert other.config.hitl == "off"
    finally:
        other.server_close()


def test_workdir_switch_validates_persists_and_affects_config(web: Any) -> None:
    server, session_dir, _ = web
    session_id = create_session(server)
    url = f"/api/sessions/{session_id}/settings"
    target = session_dir.parent / "project"
    target.mkdir()
    outside = session_dir.parent.parent / "outside-workdir"
    outside.mkdir()

    status, _, body = request(server, "POST", url, {"workdir": str(target)})
    assert status == 200
    data = json.loads(body)
    assert data["workdir"] == str(target.resolve())
    assert json.loads((session_dir / f"{session_id}.settings.json").read_text(
        encoding="utf-8"))["workdir"] == str(target.resolve())
    assert server.effective_config(session_id).workdir == str(target.resolve())

    # 目录不存在 / 是文件 / 不在授权根内：400，且说明如何授权
    status, _, body = request(server, "POST", url, {"workdir": str(session_dir.parent / "nope")})
    assert status == 400
    assert "不存在" in json.loads(body)["error"]

    note = session_dir.parent / "note.txt"
    note.write_text("x", encoding="utf-8")
    status, _, body = request(server, "POST", url, {"workdir": str(note)})
    assert status == 400
    assert "不是目录" in json.loads(body)["error"]

    status, _, body = request(server, "POST", url, {"workdir": str(outside)})
    assert status == 400
    error = json.loads(body)["error"]
    assert "授权根" in error and "--allow-dir" in error

    status, _, body = request(server, "POST", url, {"workdir": "   "})
    assert status == 400

    # 非法尝试不覆盖已保存的值
    assert server.session_settings(session_id)["workdir"] == str(target.resolve())

    # 同目录新进程仍记得会话级工作目录
    other = create_server(
        replace(Config(), memory_enabled=False),
        port=0,
        session_dir=session_dir,
        log_dir=None,
        mock=True,
        allow_dirs=[str(session_dir.parent)],
    )
    try:
        assert other.session_settings(session_id)["workdir"] == str(target.resolve())
    finally:
        other.server_close()


def test_stored_workdir_falls_back_when_missing(web: Any) -> None:
    server, session_dir, _ = web
    session_id = create_session(server)
    target = session_dir.parent / "gone"
    target.mkdir()
    (session_dir / f"{session_id}.settings.json").write_text(
        json.dumps({"workdir": str(target)}), encoding="utf-8"
    )
    assert server.session_settings(session_id)["workdir"] == str(target.resolve())

    target.rmdir()
    assert server.session_settings(session_id)["workdir"] == str(server.default_workdir)


def test_settings_reject_unknown_fields_and_bad_values(web: Any) -> None:
    server, session_dir, _ = web
    session_id = create_session(server)
    url = f"/api/sessions/{session_id}/settings"

    # 安全边界：路径白名单、确认规则等不允许由浏览器改写
    for payload in (
        {"allow_dirs": ["/"]},
        {"hitl_confirm": ["overwrite"]},
        {"plan_mode": "maybe"},
        {"plan_verify": True},
        {"hitl": "on", "unknown": "x"},
        {},
    ):
        status, _, body = request(server, "POST", url, payload)
        assert status == 400, payload
        assert "error" in json.loads(body)

    status, _, _ = request(server, "GET", "/api/sessions/ghost/settings")
    assert status == 404
    status, _, _ = request(server, "POST", "/api/sessions/ghost/settings", {"hitl": "on"})
    assert status == 404

    assert not (session_dir / f"{session_id}.settings.json").exists()


def test_settings_take_effect_on_next_message(web: Any) -> None:
    server, _, log_dir = web
    session_id = create_session(server)

    request(server, "POST", f"/api/sessions/{session_id}/messages", {"message": "hi"})
    before = [record.event for record in TraceLogger.load(session_id, log_dir)]
    assert "plan" not in before

    status, _, _ = request(
        server, "POST", f"/api/sessions/{session_id}/settings", {"plan_mode": "on"}
    )
    assert status == 200

    status, _, body = request(
        server, "POST", f"/api/sessions/{session_id}/messages", {"message": "再来一次"}
    )
    assert status == 200
    after = [record.event for record in TraceLogger.load(session_id, log_dir)]
    assert "plan" in after
    # Mock 返回的不是 JSON，计划校验失败后退回 ReAct，本轮仍然完成
    assert "plan_fallback" in after
    assert any(name == "done" and data["ok"] for name, data in parse_sse(body.decode("utf-8")))


# ---- 人工确认闭环 ---------------------------------------------------------


@pytest.fixture()
def web_hitl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    provider = MockProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments=json.dumps(
                            {"path": str(target), "content": "new"}, ensure_ascii=False
                        ),
                    )
                ],
            ),
            LLMResponse(content="已处理"),
        ]
    )
    monkeypatch.setattr("miniagent.cli.build_provider", lambda config, mock: provider)
    session_dir = tmp_path / "sessions"
    log_dir = tmp_path / "logs"
    server = create_server(
        replace(
            Config(),
            memory_enabled=False,
            hitl="on",
            hitl_confirm=("overwrite",),
        ),
        port=0,
        session_dir=session_dir,
        log_dir=str(log_dir),
        mock=True,
        allow_dirs=[str(tmp_path)],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, provider, target, log_dir
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def open_stream(server: Any, session_id: str, message: str) -> tuple[Any, Any]:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    body = json.dumps({"message": message}, ensure_ascii=False).encode("utf-8")
    conn.request(
        "POST",
        f"/api/sessions/{session_id}/messages",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    return conn, conn.getresponse()


def read_event(response: Any) -> tuple[str, dict[str, Any]] | None:
    name = ""
    data: list[str] = []
    while True:
        line = response.readline()
        if not line:
            return None
        text = line.decode("utf-8").rstrip("\r\n")
        if text == "":
            if name and data:
                return name, json.loads("\n".join(data))
            name = ""
            data = []
            continue
        if text.startswith("event:"):
            name = text[len("event:") :].strip()
        elif text.startswith("data:"):
            data.append(text[len("data:") :].strip())


def wait_approval(response: Any) -> dict[str, Any]:
    while True:
        event = read_event(response)
        assert event is not None, "SSE 在确认事件前结束"
        name, data = event
        if name == "approval":
            return data
        assert name not in {"done", "error"}, f"未收到确认请求：{name} {data}"


def wait_done(response: Any) -> tuple[dict[str, Any], list[str]]:
    names: list[str] = []
    while True:
        event = read_event(response)
        assert event is not None, "SSE 在 done 前结束"
        name, data = event
        names.append(name)
        if name == "done":
            return data, names
        assert name != "error", f"执行出错：{data}"


def test_web_approval_allow_resumes_same_call(web_hitl: Any) -> None:
    server, _, target, log_dir = web_hitl
    session_id = create_session(server)
    conn, response = open_stream(server, session_id, "覆盖它")
    try:
        approval = wait_approval(response)
        assert approval["tool"] == "write_file"
        assert approval["risk"] == "overwrite"
        assert approval["session_id"] == session_id

        status, _, body = request(
            server,
            "POST",
            f"/api/sessions/{session_id}/approvals/{approval['id']}",
            {"decision": "allow"},
        )
        assert status == 200

        done, names = wait_done(response)
        assert done["ok"] is True
        assert "tool_end" in names
        assert target.read_text(encoding="utf-8") == "new"
        records = [
            record
            for record in TraceLogger.load(session_id, log_dir)
            if record.event == "hitl"
        ]
        assert records and "decision=allow" in records[0].detail
    finally:
        conn.close()


def test_web_approval_deny_returns_to_model(web_hitl: Any) -> None:
    server, provider, target, log_dir = web_hitl
    session_id = create_session(server)
    conn, response = open_stream(server, session_id, "覆盖它")
    try:
        approval = wait_approval(response)
        status, _, _ = request(
            server,
            "POST",
            f"/api/sessions/{session_id}/approvals/{approval['id']}",
            {"decision": "deny"},
        )
        assert status == 200

        done, _ = wait_done(response)
        assert done["ok"] is True
        assert target.read_text(encoding="utf-8") == "old"
        second_context = provider.calls[1]
        assert any(
            "HITL 拦截" in (message.content or "")
            for message in second_context
            if message.role == "tool"
        )
        records = [
            record
            for record in TraceLogger.load(session_id, log_dir)
            if record.event == "hitl"
        ]
        assert records and "decision=deny" in records[0].detail
    finally:
        conn.close()


def test_web_unknown_approval_is_rejected(web_hitl: Any) -> None:
    server, _, _, _ = web_hitl
    session_id = create_session(server)

    status, _, _ = request(
        server,
        "POST",
        f"/api/sessions/{session_id}/approvals/not-exist",
        {"decision": "allow"},
    )
    assert status == 409

    status, _, _ = request(
        server,
        "POST",
        f"/api/sessions/{session_id}/approvals/not-exist",
        {"decision": "maybe"},
    )
    assert status == 400


def test_web_interaction_waits_and_cancel_on_page_close(tmp_path: Path) -> None:
    server = create_server(
        replace(Config(), memory_enabled=False, hitl="on"),
        port=0,
        session_dir=tmp_path / "sessions",
        log_dir=str(tmp_path / "logs"),
        mock=True,
    )
    try:
        payloads: list[Any] = []
        interaction = WebInteraction(server, "s1", payloads.append)

        results: list[Any] = []
        waiting = threading.Thread(
            target=lambda: results.append(
                interaction.ask(
                    ToolRequest(
                        tool="write_file",
                        args={"path": "b.txt"},
                        read_only=False,
                        risk="overwrite",
                    )
                )
            )
        )
        waiting.start()
        for _ in range(100):
            if payloads:
                break
            time.sleep(0.01)
        assert len(payloads) == 1
        time.sleep(0.2)
        assert waiting.is_alive(), "网页端未确认时应一直等待，不自动超时"

        server.cancel_approvals("s1", "页面连接已关闭")
        waiting.join(timeout=3)
        assert not waiting.is_alive()
        assert results[0].decision == "deny"
        assert results[0].reason == "页面连接已关闭"
        assert not server.approvals
    finally:
        server.server_close()
