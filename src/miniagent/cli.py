"""CLI 层：参数解析、REPL、事件渲染、退出码。"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from miniagent.agent import NO_PROGRESS_MODES, Agent, AgentEvent
from miniagent.checkpoint import Checkpoint
from miniagent.config import Config, load_config
from miniagent.dependencies import DEPS_MODES
from miniagent.embeddings import EmbeddingClient
from miniagent.errors import ConfigError, MiniAgentError, ProviderError
from miniagent.guard import MODES as GUARD_MODES
from miniagent.guard import PromptGuard
from miniagent.hitl import (
    HITL_MODES,
    RULE_NAMES,
    AutoDenyHandler,
    InteractionAnswer,
    InteractionHandler,
    ToolRequest,
    summarize_args,
)
from miniagent.llm import MockProvider, OpenAICompatProvider, dumps_args
from miniagent.logging import TraceLogger, format_record
from miniagent.memory import MemoryStore
from miniagent.planner import PLAN_MODES, VERIFY_MODES
from miniagent.security import PathPolicy
from miniagent.session import Session
from miniagent.snapshot import SNAPSHOT_MODES, GitSnapshotter, rollback_snapshots
from miniagent.soul import load_soul, load_user_profile
from miniagent.tools import ToolRegistry, builtin_tools, web_tools


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
        "--short-term-tokens",
        type=int,
        metavar="N",
        help="每轮请求携带的最近对话预算（token），默认 2000；"
        "不含 system 与长期记忆注入，也不替代 max_context_tokens 硬上限",
    )
    parser.add_argument(
        "--allow-dir",
        action="append",
        metavar="DIR",
        help="允许读写的目录，可重复；默认只允许当前工作目录",
    )
    parser.add_argument(
        "--workdir",
        metavar="DIR",
        help="工作目录：相对路径与 run_shell 的基准，默认进程启动时的 cwd；"
        "必须在 --allow-dir 授权范围内",
    )
    parser.add_argument("--log-dir", help="trace 日志目录，默认 ~/.miniagent/logs")
    parser.add_argument("--soul", help="人格文件路径，默认 ./soul.md 或 ~/.miniagent/soul.md")
    parser.add_argument(
        "--user",
        help="用户画像文件路径，默认 ./user.md 或 ~/.miniagent/user.md；与人格、记忆不是一回事",
    )
    parser.add_argument("--memory-dir", help="记忆目录，默认 ~/.miniagent/memory")
    parser.add_argument("--no-memory", action="store_true", help="关闭记忆检索与写入")
    parser.add_argument("--gate-model", help="记忆门使用的小模型名，默认与主模型相同")
    parser.add_argument(
        "--prompt-guard",
        choices=GUARD_MODES,
        help="提示词注入防护：off 关闭 / annotate 仅标注（默认）/ block 命中即拦截",
    )
    parser.add_argument(
        "--tool-deps",
        choices=DEPS_MODES,
        help="同轮工具依赖判定：off 仅路径调度 / model 叠加小模型判定（默认）",
    )
    parser.add_argument(
        "--plan-mode",
        choices=PLAN_MODES,
        help="计划模式：off 直接 ReAct（默认）/ on 先出计划、校验后逐步执行",
    )
    parser.add_argument(
        "--plan-verify",
        choices=VERIFY_MODES,
        help="收尾校验：off 关闭（默认）/ on 计划执行完后核对原始意图与实际结果",
    )
    parser.add_argument(
        "--hitl",
        choices=HITL_MODES,
        help="工具调用人工确认：off 关闭（默认）/ on 危险操作先问用户",
    )
    parser.add_argument(
        "--hitl-confirm",
        metavar="RULES",
        help="需要确认的规则，逗号分隔：overwrite,mutate,delete（默认全部）",
    )
    parser.add_argument(
        "--snapshot",
        choices=SNAPSHOT_MODES,
        help="高危操作前的 git 快照：on 默认开启（只对非只读工具）/ off 完全关闭",
    )
    parser.add_argument(
        "--no-progress",
        choices=NO_PROGRESS_MODES,
        help="无进展循环检测：on 默认开启（相同 工具+参数+结果 反复出现时先提示再终止）/ off 关闭",
    )
    parser.add_argument(
        "--no-progress-repeat",
        type=int,
        metavar="N",
        help="连续 N 次相同指纹判定为无进展，默认 3（最小 2）",
    )
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


def build_rollback_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniagent rollback",
        description="把工作区恢复到某个高危操作执行前的 git 快照（只改工作区，不动索引与 HEAD）",
    )
    parser.add_argument("session_id", help="目标会话 ID")
    parser.add_argument(
        "--to",
        type=int,
        metavar="SEQ",
        help="恢复到 checkpoint 中指定 seq 的快照；默认最后一个可用快照",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="工作区有未提交改动时仍强制恢复（会覆盖这些改动）",
    )
    parser.add_argument(
        "--session-dir", help="会话文件目录，默认 ~/.miniagent/sessions"
    )
    return parser


def build_registry(
    allow_dirs: list[str] | None = None,
    *,
    api_key: str = "",
    workdir: str | None = None,
) -> ToolRegistry:
    policy = PathPolicy([Path.cwd(), *(allow_dirs or [])], workdir=workdir)
    registry = ToolRegistry()
    for tool in builtin_tools(policy):
        registry.register(tool)
    for tool in web_tools(api_key):
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


def build_gate_provider(config: Config, provider, mock: bool):
    """记忆门与主对话共用 LLMProvider 契约；真实模式下可换更便宜的模型。"""
    if mock or config.provider == "mock" or not config.gate_model:
        return provider
    return OpenAICompatProvider(
        api_key=config.api_key,
        model=config.gate_model,
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
    elif event.kind == "hitl":
        print(f"[hitl] {event.text}")
    elif event.kind == "error":
        print(f"[error] {event.text}", file=sys.stderr)


class CliInteraction:
    """终端交互：危险操作确认、缺少参数时补充。

    交互式场景一直等用户回答，不设超时、不自动判拒；
    只有 stdin 不是 TTY（没有交互通道）时才明确拒绝。
    """

    name = "cli"

    def ask(self, request: ToolRequest) -> InteractionAnswer:
        if not sys.stdin.isatty():
            return InteractionAnswer("deny", reason="该操作需要确认，请在交互模式下运行")
        if request.phase == "missing":
            print(f"[HITL] 工具 {request.tool} 缺少必需参数：{'、'.join(request.missing)}")
            print(f"  当前参数：{summarize_args(request.args)}")
            answer = input("  请补充取值（直接回车表示放弃）: ")
            if not answer.strip():
                return InteractionAnswer("deny", reason="empty_input")
            return InteractionAnswer("allow", text=answer.strip())

        print(f"[HITL] 工具 {request.tool} 需要确认（{request.reason or request.risk}）")
        print(f"  参数：{summarize_args(request.args)}")
        answer = input("  允许执行？[y/N]: ")
        if answer.strip().lower() in {"y", "yes", "是"}:
            return InteractionAnswer("allow")
        return InteractionAnswer("deny", reason="user_rejected")


_embedding_notice_shown = False


def build_embedder(
    config: Config, *, mock: bool, notice: Any = print
) -> EmbeddingClient | None:
    """按独立配置构建 embedding 客户端；缺 key 自动关闭，提示只在进程内出现一次。

    mock 模式承诺"不请求网络"，因此即使配了 key 也不启用向量通道。
    """
    global _embedding_notice_shown
    if mock or not config.memory_enabled:
        return None
    if not config.embedding_api_key:
        if not _embedding_notice_shown:
            notice(
                "[embedding] 未配置 SILICONFLOW_API_KEY（或 config 的 embedding_api_key），"
                "向量检索已关闭，记忆检索将返回空结果（写入与蒸馏不受影响）"
            )
            _embedding_notice_shown = True
        return None
    return EmbeddingClient(
        api_key=config.embedding_api_key,
        model=config.embedding_model,
        base_url=config.embedding_base_url,
        timeout=config.embedding_timeout,
    )


def embedding_disabled_reason(config: Config, *, mock: bool) -> str:
    """给 trace 用的降级原因；能启用向量通道时返回空串。"""
    if mock:
        return "mock_mode"
    if not config.embedding_api_key:
        return "no_api_key"
    return ""


def make_agent(
    config: Config,
    *,
    mock: bool,
    session: Session | None = None,
    allow_dirs: list[str] | None = None,
    workdir: str | None = None,
    log_dir: str | None = None,
    interaction: InteractionHandler | None = None,
    non_interactive: bool = False,
) -> Agent:
    tracer = (
        TraceLogger(session.session_id, Path(log_dir) if log_dir else None)
        if session is not None
        else None
    )
    checkpoint = Checkpoint.for_session(session, tracer=tracer) if session else None
    registry = build_registry(
        allow_dirs, api_key=config.tavily_api_key, workdir=workdir
    )
    if config.snapshot == "on" and checkpoint is not None:
        base = Path(workdir).expanduser().resolve() if workdir else Path.cwd().resolve()
        # 默认开启但只作用于非只读工具；非 git 目录只提示一次并降级为只记元信息
        registry.snapshotter = GitSnapshotter(base, tracer=tracer, notice=print)
    provider = build_provider(config, mock)
    soul_text, _ = load_soul(config.soul_path or None)
    user_text, _ = load_user_profile(config.user_path or None)
    memory = (
        MemoryStore(
            Path(config.memory_dir) if config.memory_dir else None,
            batch_size=config.memory_batch_size,
            session_id=session.session_id if session else "",
            embedder=build_embedder(config, mock=mock),
            tracer=tracer,
            embedding_disabled_reason=embedding_disabled_reason(config, mock=mock),
        )
        if config.memory_enabled
        else None
    )
    handler: InteractionHandler | None = interaction
    if config.hitl == "on" and handler is None:
        handler = (
            AutoDenyHandler("该操作需要确认，请在交互模式下运行")
            if non_interactive
            else CliInteraction()
        )
    agent = Agent(
        provider=provider,
        registry=registry,
        max_turns=config.max_turns,
        max_context_tokens=config.max_context_tokens,
        short_term_tokens=config.short_term_tokens,
        soul=soul_text,
        user_profile=user_text,
        memory=memory,
        gate_provider=build_gate_provider(config, provider, mock),
        memory_top_k=config.memory_top_k,
        session=session,
        checkpoint=checkpoint,
        tracer=tracer,
        guard=PromptGuard(config.prompt_guard),
        tool_deps=config.tool_deps,
        plan_mode=config.plan_mode,
        plan_verify=config.plan_verify,
        plan_verify_rounds=config.plan_verify_rounds,
        no_progress=config.no_progress,
        no_progress_repeat=config.no_progress_repeat,
        hitl=config.hitl,
        hitl_confirm=config.hitl_confirm,
        hitl_handler=handler,
        workdir=workdir,
        on_event=render_event,
    )
    for item in agent.recovery:
        print(f"[checkpoint] {item.tool} {item.action}（{item.reason}）")
    return agent


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
    workdir: str | None = None,
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
                workdir=workdir,
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
                workdir=workdir,
                log_dir=log_dir,
            )
            print(f"已恢复会话：{session.session_id}（{len(session.messages)} 条消息）")
            continue

        try:
            agent.run(text)
        except ProviderError as exc:
            print(f"[error] {format_error(exc)}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n已中断当前任务")


def format_error(exc: BaseException) -> str:
    """错误消息 + 恢复摘要；没有恢复信息时与旧输出完全一致。"""
    summary = getattr(exc, "recovery_text", "")
    return f"{exc}\n[recovery] {summary}" if summary else str(exc)


def trace_main(argv: list[str]) -> int:
    args = build_trace_parser().parse_args(argv)
    try:
        records = TraceLogger.load(args.session_id, args.log_dir)
    except MiniAgentError as exc:
        print(f"[error] {format_error(exc)}", file=sys.stderr)
        return exc.exit_code
    print(f"session: {args.session_id}（{len(records)} 条记录）")
    for record in records:
        print(format_record(record))
    return 0


def rollback_main(argv: list[str]) -> int:
    args = build_rollback_parser().parse_args(argv)
    try:
        ref = rollback_snapshots(
            args.session_id,
            session_dir=args.session_dir,
            to_seq=args.to,
            force=args.force,
        )
    except MiniAgentError as exc:
        print(f"[error] {format_error(exc)}", file=sys.stderr)
        return exc.exit_code
    print(
        f"已恢复到 seq={ref.seq}（工具 {ref.tool}，快照 {ref.sha[:12]}，"
        f"工作区 {ref.workdir}）"
    )
    print("提示：只恢复了已跟踪文件的工作区内容；未跟踪文件与暂存区未改动。")
    return 0


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "trace":
        return trace_main(args_list[1:])
    if args_list and args_list[0] == "rollback":
        return rollback_main(args_list[1:])
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
        if args.short_term_tokens is not None:
            if args.short_term_tokens < 1:
                raise ConfigError("--short-term-tokens 必须大于 0")
            config = replace(
                config, short_term_tokens=args.short_term_tokens
            )
        if args.soul:
            config = replace(config, soul_path=args.soul)
        if args.user:
            config = replace(config, user_path=args.user)
        if args.memory_dir:
            config = replace(config, memory_dir=args.memory_dir)
        if args.no_memory:
            config = replace(config, memory_enabled=False)
        if args.gate_model:
            config = replace(config, gate_model=args.gate_model)
        if args.prompt_guard:
            config = replace(config, prompt_guard=args.prompt_guard)
        if args.tool_deps:
            config = replace(config, tool_deps=args.tool_deps)
        if args.plan_mode:
            config = replace(config, plan_mode=args.plan_mode)
        if args.plan_verify:
            config = replace(config, plan_verify=args.plan_verify)
        if args.hitl:
            config = replace(config, hitl=args.hitl)
        if args.snapshot:
            config = replace(config, snapshot=args.snapshot)
        if args.no_progress:
            config = replace(config, no_progress=args.no_progress)
        if args.no_progress_repeat is not None:
            if args.no_progress_repeat < 2:
                raise ConfigError("--no-progress-repeat 不能小于 2")
            config = replace(config, no_progress_repeat=args.no_progress_repeat)
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

        # 默认工作目录 = 启动进程时的 cwd；命令行优先于 config
        workdir = args.workdir or config.workdir or None

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
            workdir=workdir,
            log_dir=args.log_dir,
            non_interactive=bool(args.prompt),
        )
        if args.prompt:
            return run_single(agent, args.prompt)
        return run_repl(
            config,
            agent,
            mock=args.mock,
            session_dir=session_dir,
            allow_dirs=args.allow_dir,
            workdir=workdir,
            log_dir=args.log_dir,
        )
    except KeyboardInterrupt:
        print("\n已取消", file=sys.stderr)
        return 130
    except MiniAgentError as exc:
        print(f"[error] {format_error(exc)}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
