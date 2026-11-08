"""CLI 层：参数解析、REPL、事件渲染、退出码。"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from miniagent.agent import Agent, AgentEvent
from miniagent.config import Config, load_config
from miniagent.errors import ConfigError, MiniAgentError, ProviderError
from miniagent.llm import MockProvider, OpenAICompatProvider, dumps_args
from miniagent.logging import TraceLogger, format_record
from miniagent.security import PathPolicy
from miniagent.session import Session
from miniagent.tools import ToolRegistry, builtin_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniagent",
        description="极简命令行 AI agent：模型调用 + 工具循环",
    )
    parser.add_argument("-p", "--prompt", help="单次执行一条指令后退出")
    parser.add_argument("--mock", action="store_true", help="使用离线 Mock 模型，不请求网络")
    parser.add_argument("--resume", metavar="ID", help="恢复指定会话后继续")
    parser.add_argument("--session-dir", help="会话文件目录，默认 ~/.miniagent/sessions")
    parser.add_argument("--model", help="覆盖模型名")
    parser.add_argument("--base-url", help="覆盖 OpenAI 兼容端点地址")
    parser.add_argument("--max-turns", type=int, help="单次任务的最大轮次，默认 10")
    parser.add_argument(
        "--allow-dir",
        action="append",
        metavar="DIR",
        help="允许读写的目录，可重复；默认只允许当前工作目录",
    )
    parser.add_argument("--log-dir", help="trace 日志目录，默认 ~/.miniagent/logs")
    parser.add_argument("--config", default="config.toml", help="配置文件路径")
    return parser


def build_trace_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniagent trace",
        description="回放指定会话的结构化 trace：轮次、工具、耗时、错误码",
    )
    parser.add_argument("session_id", help="要回放的会话 ID")
    parser.add_argument("--log-dir", help="trace 日志目录，默认 ~/.miniagent/logs")
    return parser


def build_registry(allow_dirs: list[str] | None = None) -> ToolRegistry:
    policy = PathPolicy([Path.cwd(), *(allow_dirs or [])])
    registry = ToolRegistry()
    for tool in builtin_tools(policy):
        registry.register(tool)
    return registry


def build_provider(config: Config, mock: bool):
    if mock or config.provider == "mock":
        return MockProvider()
    return OpenAICompatProvider(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        rate_limit_rpm=config.rate_limit_rpm,
    )


def render_event(event: AgentEvent) -> None:
    if event.kind == "text_delta":
        print(event.text, end="", flush=True)
    elif event.kind == "text_end":
        print()
    elif event.kind == "text":
        print(event.text)
    elif event.kind == "context_trimmed":
        print(f"[context] {event.text}")
    elif event.kind == "tool_start":
        print(f"  -> {event.tool} {dumps_args(event.args)}")
    elif event.kind == "tool_end":
        status = "ok" if event.ok else "error"
        preview = " ".join(event.text.split())
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"  <- {event.tool} {status} {event.duration_ms}ms | {preview}")
    elif event.kind == "error":
        print(f"[error] {event.text}", file=sys.stderr)


def make_agent(
    config: Config,
    *,
    mock: bool,
    session: Session | None = None,
    allow_dirs: list[str] | None = None,
    log_dir: str | None = None,
) -> Agent:
    tracer = (
        TraceLogger(session.session_id, Path(log_dir) if log_dir else None)
        if session is not None
        else None
    )
    return Agent(
        provider=build_provider(config, mock),
        registry=build_registry(allow_dirs),
        max_turns=config.max_turns,
        max_context_tokens=config.max_context_tokens,
        session=session,
        tracer=tracer,
        on_event=render_event,
    )


def run_single(agent: Agent, prompt: str) -> int:
    result = agent.run(prompt)
    if agent.session is not None:
        print(f"session: {agent.session.session_id}（--resume {agent.session.session_id} 可继续）")
    return 0 if result.ok else 1


def print_sessions(directory: Path | None) -> None:
    sessions = Session.list(directory)
    if not sessions:
        print("没有历史会话")
        return
    for info in sessions:
        updated = time.strftime("%m-%d %H:%M", time.localtime(info.updated_at))
        print(f"- {info.session_id}  {info.message_count:3d} 条消息  {updated}")


def run_repl(
    config: Config,
    agent: Agent,
    *,
    mock: bool,
    session_dir: Path | None,
    allow_dirs: list[str] | None = None,
    log_dir: str | None = None,
) -> int:
    tools = "、".join(agent.registry.names())
    session_id = agent.session.session_id if agent.session else "-"
    print(f"session: {session_id} | 工具：{tools}")
    print("命令：/tools /sessions /resume <id> /clear /exit")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        text = line.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return 0
        if text == "/clear":
            agent = make_agent(
                config,
                mock=mock,
                session=Session.create(directory=session_dir),
                allow_dirs=allow_dirs,
                log_dir=log_dir,
            )
            print(f"已开启新会话：{agent.session.session_id}")
            continue
        if text == "/tools":
            for spec in agent.registry.specs():
                print(f"- {spec['name']}：{spec['description']}")
            continue
        if text == "/sessions" or text.startswith("/resume"):
            _, _, session_id = text.partition(" ")
            if not session_id.strip():
                print_sessions(session_dir)
                continue
            try:
                session = Session.load(session_id.strip(), session_dir)
            except ConfigError as exc:
                print(f"[error] {exc}", file=sys.stderr)
                continue
            agent = make_agent(
                config,
                mock=mock,
                session=session,
                allow_dirs=allow_dirs,
                log_dir=log_dir,
            )
            print(f"已恢复会话：{session.session_id}（{len(session.messages)} 条消息）")
            continue

        try:
            agent.run(text)
        except ProviderError as exc:
            print(f"[error] {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n已中断当前任务")


def trace_main(argv: list[str]) -> int:
    args = build_trace_parser().parse_args(argv)
    try:
        records = TraceLogger.load(args.session_id, args.log_dir)
    except MiniAgentError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return exc.exit_code
    print(f"session: {args.session_id}（{len(records)} 条记录）")
    for record in records:
        print(format_record(record))
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "trace":
        return trace_main(args_list[1:])
    args = build_parser().parse_args(args_list)
    try:
        config = load_config(Path(args.config))
        if args.model:
            config = replace(config, model=args.model)
        if args.base_url:
            config = replace(config, base_url=args.base_url.rstrip("/"))
        if args.max_turns is not None:
            if args.max_turns < 1:
                raise ConfigError("--max-turns 必须大于 0")
            config = replace(config, max_turns=args.max_turns)

        session_dir = Path(args.session_dir).expanduser() if args.session_dir else None
        session = (
            Session.load(args.resume, session_dir)
            if args.resume
            else Session.create(directory=session_dir)
        )
        agent = make_agent(
            config,
            mock=args.mock,
            session=session,
            allow_dirs=args.allow_dir,
            log_dir=args.log_dir,
        )
        if args.prompt:
            return run_single(agent, args.prompt)
        return run_repl(
            config,
            agent,
            mock=args.mock,
            session_dir=session_dir,
            allow_dirs=args.allow_dir,
            log_dir=args.log_dir,
        )
    except KeyboardInterrupt:
        print("\n已取消", file=sys.stderr)
        return 130
    except MiniAgentError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
