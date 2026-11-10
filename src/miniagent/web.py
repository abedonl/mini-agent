"""本地网页界面：标准库 HTTP 服务 + SSE 流式输出。

只监听 127.0.0.1；会话读写、trace 回放与 Agent 复用 CLI 的既有实现，
不引入第三方依赖，也不把密钥或 trace 内部字段暴露给浏览器。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from miniagent.agent import AgentEvent
from miniagent.cli import make_agent
from miniagent.config import Config, load_config
from miniagent.errors import ConfigError, MiniAgentError
from miniagent.hitl import (
    HITL_MODES,
    RULE_NAMES,
    InteractionAnswer,
    ToolRequest,
    summarize_args,
)
from miniagent.llm import Message
from miniagent.logging import TraceLogger
from miniagent.planner import PLAN_MODES, VERIFY_MODES
from miniagent.security import check_workdir
from miniagent.session import Session, default_session_dir

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{3,}")
MAX_MESSAGE_CHARS = 10000
# 浏览器可改的开关白名单：只换 Agent 行为，不触碰路径白名单、命令黑名单等安全边界
SETTINGS_FIELDS: dict[str, tuple[str, ...]] = {
    "plan_mode": PLAN_MODES,
    "plan_verify": VERIFY_MODES,
    "hitl": HITL_MODES,
}
MAX_BODY_BYTES = 1_000_000
TRACE_PUBLIC_FIELDS = (
    "ts",
    "turn",
    "event",
    "tool",
    "duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "ok",
    "error_code",
    "detail",
)


def redact(text: str) -> str:
    """把形如 sk-xxx 的密钥片段替换掉，避免进入响应。"""
    return SECRET_RE.sub("sk-***", text)


def error_payload(exc: BaseException) -> dict[str, Any]:
    """SSE error 事件载荷：消息 + 恢复摘要（重试次数/状态码/降级动作）。"""
    payload: dict[str, Any] = {"message": redact(str(exc)) or "内部错误"}
    summary = getattr(exc, "recovery", None)
    if summary is not None:
        payload["attempts"] = summary.attempts
        payload["last_status"] = summary.last_status
        payload["last_error"] = redact(summary.last_error)
        payload["degraded"] = [redact(item) for item in summary.degraded]
        payload["recovery"] = redact(summary.describe())
    return payload


def _validate_session_id(session_id: str) -> None:
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ConfigError(f"非法的会话 ID：{session_id!r}")


def _session_path(base: Path, session_id: str) -> Path:
    """严格校验 ID 并确认拼出的路径仍在会话目录内。"""
    _validate_session_id(session_id)
    path = (base / f"{session_id}.jsonl").resolve()
    if path.parent != base:
        raise ConfigError("非法的会话路径")
    return path


def _settings_path(base: Path, session_id: str) -> Path:
    """每个会话一个开关文件；沿用与消息文件相同的路径校验。"""
    _validate_session_id(session_id)
    path = (base / f"{session_id}.settings.json").resolve()
    if path.parent != base:
        raise ConfigError("非法的会话路径")
    return path


def _load_settings(path: Path) -> dict[str, str]:
    """读已保存的开关；文件缺失、损坏或值非法时回落到启动配置。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    settings: dict[str, str] = {}
    for key, allowed in SETTINGS_FIELDS.items():
        value = data.get(key)
        if isinstance(value, str) and value in allowed:
            settings[key] = value
    workdir = data.get("workdir")
    if isinstance(workdir, str) and workdir.strip():
        # 只回填字符串，能否使用交由 session_settings 结合授权根重新校验
        settings["workdir"] = workdir.strip()
    return settings


def _write_settings(path: Path, settings: dict[str, str]) -> None:
    """先写临时文件再原子替换，避免并发读写看到半份 JSON。"""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        body = json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _load_messages(path: Path) -> list[Message]:
    messages: list[Message] = []
    with path.open(encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                messages.append(Message.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ConfigError(f"会话文件 {path.name} 第 {lineno} 行损坏：{exc}") from exc
    return messages


def _open_session(base: Path, session_id: str) -> Session:
    """发送消息用：空文件视作新会话，其余交给 Session.load 校验。"""
    path = _session_path(base, session_id)
    if not path.is_file():
        raise ConfigError(f"会话不存在：{session_id}")
    try:
        return Session.load(session_id, base)
    except ConfigError:
        if path.stat().st_size == 0:
            return Session.create(session_id=session_id, directory=base)
        raise


def _session_title(path: Path) -> str:
    """取首条 user 消息压成一行，最多 40 字。"""
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("role") == "user" and data.get("content"):
                    return " ".join(str(data["content"]).split())[:40]
    except OSError:
        return ""
    return ""


def _list_sessions(base: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": info.session_id,
            "title": _session_title(info.path),
            "message_count": info.message_count,
            "updated_at": info.updated_at,
        }
        for info in Session.list(base, limit=200)
    ]


def _message_dict(message: Message) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
    }
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    return data


@dataclass
class PendingApproval:
    """一次等待浏览器确认的工具调用；由 SSE 线程发起、POST 端点落地。"""

    approval_id: str
    session_id: str
    event: threading.Event
    answer: InteractionAnswer | None = None


class WebServer(ThreadingHTTPServer):
    """带配置的线程 HTTP 服务；只由 create_server 绑定 127.0.0.1。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        config: Config,
        mock: bool,
        session_dir: Path | str | None,
        log_dir: Path | str | None,
        allow_dirs: list[str] | None,
    ) -> None:
        super().__init__(address, WebHandler)
        self.config = config
        self.mock = mock
        self.session_base = (
            Path(session_dir).expanduser() if session_dir else default_session_dir().expanduser()
        ).resolve()
        self.log_dir = Path(log_dir).expanduser() if log_dir else None
        self.allow_dirs = list(allow_dirs or [])
        # 授权根固定为启动 cwd + --allow-dir；工作目录只能在其中选择
        self.authorized_roots = tuple(
            Path(root).expanduser().resolve()
            for root in [Path.cwd(), *(allow_dirs or [])]
        )
        self.default_workdir = check_workdir(
            config.workdir or Path.cwd(), self.authorized_roots
        )
        self.approvals: dict[str, PendingApproval] = {}
        self.approvals_lock = threading.Lock()
        self.settings_lock = threading.Lock()

    # ---- 每会话开关 -------------------------------------------------------

    def session_settings(self, session_id: str) -> dict[str, str]:
        """启动配置打底，再叠加该会话已保存的开关。

        已保存的工作目录会重新校验：目录被删、变成文件或不再被授权时
        回落到默认值，避免界面显示一个实际用不了的路径。
        """
        settings = {
            "plan_mode": self.config.plan_mode,
            "plan_verify": self.config.plan_verify,
            "hitl": self.config.hitl,
            "workdir": str(self.default_workdir),
        }
        stored = _load_settings(_settings_path(self.session_base, session_id))
        for key in ("plan_mode", "plan_verify", "hitl"):
            if key in stored:
                settings[key] = stored[key]
        if "workdir" in stored:
            try:
                settings["workdir"] = str(
                    check_workdir(stored["workdir"], self.authorized_roots)
                )
            except ConfigError:
                pass
        return settings

    def update_settings(self, session_id: str, changes: dict[str, str]) -> dict[str, str]:
        """只合并白名单字段；调用方已校验取值。"""
        with self.settings_lock:
            settings = self.session_settings(session_id)
            settings.update(changes)
            _write_settings(_settings_path(self.session_base, session_id), settings)
        return settings

    def effective_config(self, session_id: str) -> Config:
        """本轮真正使用的配置：基础配置 + 该会话的开关与工作目录。

        其余字段（allow_dirs、hitl_confirm、危险命令策略等）不接受浏览器覆盖，
        所以网页上的开关无法放大权限或绕过服务端校验。
        """
        return replace(self.config, **self.session_settings(session_id))

    # ---- 人工确认的挂起 / 落地 -------------------------------------------

    def resolve_approval(
        self, session_id: str, approval_id: str, answer: InteractionAnswer
    ) -> bool:
        """把浏览器决策交给等待中的工具调用；已结束或不属于该会话时返回 False。"""
        with self.approvals_lock:
            pending = self.approvals.get(approval_id)
            if pending is None or pending.session_id != session_id:
                return False
            if pending.event.is_set():
                return False
            pending.answer = answer
        pending.event.set()
        return True

    def cancel_approvals(self, session_id: str, reason: str = "页面连接已关闭") -> None:
        """页面断开或本轮结束时，把仍在等待的确认一律按拒绝放行等待线程。"""
        with self.approvals_lock:
            pending_list = [
                item for item in self.approvals.values() if item.session_id == session_id
            ]
        for pending in pending_list:
            self.resolve_approval(
                session_id,
                pending.approval_id,
                InteractionAnswer("deny", reason=reason),
            )


class WebInteraction:
    """网页端交互：把确认请求推给 SSE，阻塞当前工具调用等待浏览器 POST。"""

    name = "web"

    def __init__(self, server: WebServer, session_id: str, emit: Any) -> None:
        self.server = server
        self.session_id = session_id
        self.emit = emit  # 投递到 SSE 队列，形如 ("approval", payload)

    def ask(self, request: ToolRequest) -> InteractionAnswer:
        approval_id = uuid.uuid4().hex
        pending = PendingApproval(approval_id, self.session_id, threading.Event())
        with self.server.approvals_lock:
            self.server.approvals[approval_id] = pending
        self.emit(
            (
                "approval",
                {
                    "id": approval_id,
                    "session_id": self.session_id,
                    "phase": request.phase,
                    "tool": request.tool,
                    "risk": request.risk,
                    "reason": request.reason,
                    "missing": list(request.missing),
                    "args": summarize_args(request.args, limit=300),
                },
            )
        )
        try:
            # 交互式场景一直等用户点：不设超时，也不自动放行 / 判拒
            triggered = pending.event.wait()
        finally:
            with self.server.approvals_lock:
                self.server.approvals.pop(approval_id, None)
        if not triggered:
            return InteractionAnswer("deny", reason="no_answer")
        return pending.answer or InteractionAnswer("deny", reason="no_answer")


class WebHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: WebServer

    # ---- 路由 -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        parts = self._parts()
        if not parts or parts == ["index.html"]:
            self._serve_index()
            return
        if parts[:2] != ["api", "sessions"]:
            self._send_json(404, {"error": "未知路径"})
            return
        rest = parts[2:]
        if not rest:
            self._send_json(200, _list_sessions(self.server.session_base))
            return
        session_id = self._session_id_or_error(rest[0])
        if session_id is None:
            return
        if len(rest) == 1:
            self._get_session(session_id)
            return
        if len(rest) == 2 and rest[1] == "trace":
            self._get_trace(session_id)
            return
        if len(rest) == 2 and rest[1] == "settings":
            self._get_settings(session_id)
            return
        self._send_json(404, {"error": "未知路径"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._body = self._read_body()
        if self._body is None:
            return
        parts = self._parts()
        if parts == ["api", "sessions"]:
            self._create_session()
            return
        if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "messages":
            session_id = self._session_id_or_error(parts[2])
            if session_id is not None:
                self._send_message(session_id)
            return
        if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "settings":
            session_id = self._session_id_or_error(parts[2])
            if session_id is not None:
                self._update_settings(session_id)
            return
        if len(parts) == 5 and parts[:2] == ["api", "sessions"] and parts[3] == "approvals":
            session_id = self._session_id_or_error(parts[2])
            if session_id is not None:
                self._resolve_approval(session_id, parts[4])
            return
        self._send_json(404, {"error": "未知路径"})

    # ---- 会话 API ---------------------------------------------------------

    def _create_session(self) -> None:
        try:
            session = Session.create(directory=self.server.session_base)
        except (ConfigError, OSError) as exc:
            self._send_json(400, {"error": redact(str(exc))})
            return
        self._send_json(200, {"id": session.session_id})

    def _get_session(self, session_id: str) -> None:
        base = self.server.session_base
        try:
            path = _session_path(base, session_id)
        except ConfigError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        if not path.is_file():
            self._send_json(404, {"error": "会话不存在"})
            return
        try:
            messages = _load_messages(path)
        except ConfigError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(
            200,
            {"id": session_id, "messages": [_message_dict(message) for message in messages]},
        )

    def _session_exists(self, session_id: str) -> bool:
        try:
            return _session_path(self.server.session_base, session_id).is_file()
        except ConfigError:
            return False

    def _get_settings(self, session_id: str) -> None:
        if not self._session_exists(session_id):
            self._send_json(404, {"error": "会话不存在"})
            return
        self._send_json(200, self._settings_payload(self.server.session_settings(session_id)))

    def _settings_payload(self, settings: dict[str, str]) -> dict[str, Any]:
        """开关 + 工作目录 + 授权根；roots 只用于界面提示，不能反过来当授权依据。"""
        payload: dict[str, Any] = dict(settings)
        payload["roots"] = [str(root) for root in self.server.authorized_roots]
        return payload

    def _update_settings(self, session_id: str) -> None:
        if not self._session_exists(session_id):
            self._send_json(404, {"error": "会话不存在"})
            return
        data = self._read_json()
        if data is None:
            return
        if not data:
            self._send_json(400, {"error": "至少提供一个开关"})
            return
        changes: dict[str, str] = {}
        for key, value in data.items():
            if key == "workdir":
                if not isinstance(value, str) or not value.strip():
                    self._send_json(400, {"error": "workdir 必须是非空字符串"})
                    return
                try:
                    # 服务端校验：存在、是目录、在授权根内；与前端显示无关
                    changes["workdir"] = str(
                        check_workdir(value, self.server.authorized_roots)
                    )
                except ConfigError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                continue
            allowed = SETTINGS_FIELDS.get(key)
            if allowed is None:
                self._send_json(
                    400,
                    {
                        "error": (
                            f"不支持修改的字段：{key}"
                            "（只允许 plan_mode / plan_verify / hitl / workdir）"
                        )
                    },
                )
                return
            text = value.strip().lower() if isinstance(value, str) else ""
            if text not in allowed:
                self._send_json(
                    400, {"error": f"{key} 只能是 {' / '.join(allowed)}"}
                )
                return
            changes[key] = text
        try:
            settings = self.server.update_settings(session_id, changes)
        except OSError as exc:
            self._send_json(500, {"error": f"开关保存失败：{exc}"})
            return
        self._send_json(200, self._settings_payload(settings))

    def _send_message(self, session_id: str) -> None:
        data = self._read_json()
        if data is None:
            return
        message = data.get("message")
        if not isinstance(message, str):
            self._send_json(400, {"error": "message 必须是字符串"})
            return
        text = message.strip()
        if not text:
            self._send_json(400, {"error": "消息不能为空"})
            return
        if len(message) > MAX_MESSAGE_CHARS:
            self._send_json(400, {"error": f"消息过长（最多 {MAX_MESSAGE_CHARS} 字符）"})
            return

        base = self.server.session_base
        try:
            path = _session_path(base, session_id)
        except ConfigError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        if not path.is_file():
            self._send_json(404, {"error": "会话不存在"})
            return
        try:
            session = _open_session(base, session_id)
        except ConfigError as exc:
            self._send_json(400, {"error": redact(str(exc))})
            return

        queue: Queue[Any] = Queue()
        server = self.server
        # 本轮配置在请求线程里定格：执行中切换开关只影响下一条消息
        config = server.effective_config(session_id)

        def work() -> None:
            try:
                agent = make_agent(
                    config,
                    mock=server.mock,
                    session=session,
                    allow_dirs=server.allow_dirs,
                    workdir=config.workdir,
                    log_dir=str(server.log_dir) if server.log_dir else None,
                    interaction=WebInteraction(server, session_id, queue.put),
                )
                agent.on_event = queue.put
                queue.put(("done", agent.run(text)))
            except Exception as exc:  # noqa: BLE001 - 统一转成 SSE error 事件
                traceback.print_exc()
                queue.put(("error", exc))
            finally:
                queue.put(("eof", None))

        self._start_sse()
        self._sse("meta", {"session_id": session_id})
        threading.Thread(target=work, name=f"agent-{session_id}", daemon=True).start()

        collected: list[str] = []
        try:
            while True:
                try:
                    item = queue.get(timeout=1.0)
                except Empty:
                    # 心跳：页面关掉后这里写失败，finally 会取消挂起的确认
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    break
                if isinstance(item, AgentEvent):
                    self._emit_agent_event(item, collected)
                    continue
                kind, value = item
                if kind == "approval":
                    self._sse("approval", value)
                    continue
                if kind == "done":
                    self._sse(
                        "done",
                        {
                            "ok": value.ok,
                            "text": value.text or "".join(collected),
                            "turns": value.turns,
                        },
                    )
                    break
                if kind == "error":
                    self._sse("error", error_payload(value))
                    self._sse("done", {"ok": False, "text": "".join(collected), "turns": 0})
                    break
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            # 页面关掉或本轮结束时，不能让等待确认的工具线程继续挂着
            self.server.cancel_approvals(session_id)

    def _resolve_approval(self, session_id: str, approval_id: str) -> None:
        data = self._read_json()
        if data is None:
            return
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in {"allow", "deny"}:
            self._send_json(400, {"error": "decision 必须是 allow 或 deny"})
            return
        answer = InteractionAnswer(
            decision,
            text=str(data.get("text", "")),
            reason=f"user_{decision}",
        )
        if not self.server.resolve_approval(session_id, approval_id, answer):
            self._send_json(409, {"error": "确认请求不存在、已处理或不属于该会话"})
            return
        self._send_json(200, {"ok": True, "decision": decision})

    def _emit_agent_event(self, event: AgentEvent, collected: list[str]) -> None:
        if event.kind in {"text_delta", "text"}:
            if event.text:
                collected.append(event.text)
                self._sse("text", {"delta": event.text})
        elif event.kind == "tool_start":
            self._sse("tool_start", {"tool": event.tool, "args": event.args})
        elif event.kind == "tool_end":
            self._sse(
                "tool_end",
                {
                    "tool": event.tool,
                    "ok": event.ok,
                    "text": event.text,
                    "duration_ms": event.duration_ms,
                },
            )
        elif event.kind == "hitl":
            self._sse(
                "hitl",
                {
                    "tool": event.tool,
                    "ok": event.ok,
                    "text": event.text,
                    "wait_ms": event.duration_ms,
                },
            )
        elif event.kind == "error":
            self._sse("error", {"message": redact(event.text)})

    # ---- trace API --------------------------------------------------------

    def _get_trace(self, session_id: str) -> None:
        query = parse_qs(urlsplit(self.path).query)
        turn: int | None = None
        if "turn" in query:
            try:
                turn = int(query["turn"][0])
            except ValueError:
                self._send_json(400, {"error": "turn 必须是整数"})
                return
            if turn < 0:
                self._send_json(400, {"error": "turn 必须是非负整数"})
                return

        try:
            records = TraceLogger.load(session_id, self.server.log_dir)
        except ConfigError:
            records = []

        payload = []
        for record in records:
            if turn is not None and record.turn != turn:
                continue
            row = {field: getattr(record, field) for field in TRACE_PUBLIC_FIELDS}
            row["detail"] = redact(str(row["detail"]))
            payload.append(row)
        self._send_json(200, payload)

    # ---- 响应工具 ---------------------------------------------------------

    def _parts(self) -> list[str]:
        path = urlsplit(self.path).path
        return [unquote(part) for part in path.split("/") if part]

    def _session_id_or_error(self, raw: str) -> str | None:
        try:
            _validate_session_id(raw)
        except ConfigError as exc:
            self._send_json(400, {"error": str(exc)})
            return None
        return raw

    def _read_body(self) -> bytes | None:
        """POST 体只读一次：路由即使不关心内容也要消费掉，避免 keep-alive 串包。"""
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            self._send_json(400, {"error": "Content-Length 非法"})
            self.close_connection = True
            return None
        if length < 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {"error": "请求体过大"})
            self.close_connection = True
            return None
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any] | None:
        raw = self._body
        if not raw:
            self._send_json(400, {"error": "请求体不能为空"})
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "请求体必须是 JSON"})
            return None
        if not isinstance(data, dict):
            self._send_json(400, {"error": "请求体必须是 JSON 对象"})
            return None
        return data

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self) -> None:
        path = Path(__file__).with_name("web_index.html")
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_json(500, {"error": f"界面文件不可用：{exc}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

    def _sse(self, event: str, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
        self.wfile.flush()


def create_server(
    config: Config,
    *,
    port: int = 8770,
    session_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    mock: bool = False,
    allow_dirs: list[str] | None = None,
) -> WebServer:
    """创建但未启动服务；固定只绑定 127.0.0.1。"""
    return WebServer(
        ("127.0.0.1", port),
        config=config,
        mock=mock,
        session_dir=session_dir,
        log_dir=log_dir,
        allow_dirs=allow_dirs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniagent.web",
        description="miniagent 本地网页界面（只监听 127.0.0.1）",
    )
    parser.add_argument("--port", type=int, default=8770, help="监听端口，默认 8770")
    parser.add_argument("--mock", action="store_true", help="使用离线 Mock 模型，不请求网络")
    parser.add_argument("--session-dir", help="会话文件目录，默认 ~/.miniagent/sessions")
    parser.add_argument("--log-dir", help="trace 日志目录，默认 ~/.miniagent/logs")
    parser.add_argument("--no-memory", action="store_true", help="关闭记忆检索与写入")
    parser.add_argument(
        "--allow-dir",
        action="append",
        metavar="DIR",
        help="允许读写的目录，可重复；默认只允许当前工作目录",
    )
    parser.add_argument(
        "--workdir",
        metavar="DIR",
        help="默认工作目录，必须在启动 cwd 或 --allow-dir 授权范围内",
    )
    parser.add_argument(
        "--hitl",
        choices=HITL_MODES,
        help="工具调用人工确认：off 关闭（默认）/ on 浏览器弹窗确认",
    )
    parser.add_argument(
        "--hitl-confirm",
        metavar="RULES",
        help="需要确认的规则，逗号分隔：overwrite,mutate,delete（默认全部）",
    )
    parser.add_argument("--config", default="config.toml", help="配置文件路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if not 0 <= args.port <= 65535:
        print("[error] --port 必须在 0-65535 之间", file=sys.stderr)
        return 2

    try:
        config = load_config(Path(args.config))
        if args.no_memory:
            config = replace(config, memory_enabled=False)
        if args.hitl:
            config = replace(config, hitl=args.hitl)
        if args.hitl_confirm is not None:
            rules = tuple(
                part.strip().lower() for part in args.hitl_confirm.split(",") if part.strip()
            )
            unknown = [rule for rule in rules if rule not in RULE_NAMES]
            if unknown:
                raise ConfigError(
                    f"未知的 --hitl-confirm 规则：{'、'.join(unknown)}"
                    f"（可选 {' / '.join(RULE_NAMES)}）"
                )
            config = replace(config, hitl_confirm=rules)
        if args.workdir:
            config = replace(config, workdir=args.workdir)
        server = create_server(
            config,
            port=args.port,
            session_dir=args.session_dir,
            log_dir=args.log_dir,
            mock=args.mock,
            allow_dirs=args.allow_dir,
        )
    except MiniAgentError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"[error] 无法监听 127.0.0.1:{args.port}：{exc}", file=sys.stderr)
        return 1

    host, port = server.server_address[:2]
    mode = "mock" if args.mock or config.provider == "mock" else "openai"
    print(f"miniagent web: http://{host}:{port}/")
    print(f"会话目录：{server.session_base} | 模型：{mode}")
    print(f"默认工作目录：{server.default_workdir}")
    print(f"授权根：{'、'.join(str(root) for root in server.authorized_roots)}")
    print("只监听 127.0.0.1，按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
