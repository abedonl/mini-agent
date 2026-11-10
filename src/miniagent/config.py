"""配置：环境变量 + 可选的 config.toml，环境变量优先。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from miniagent.agent import DEFAULT_NO_PROGRESS_REPEAT, NO_PROGRESS_MODES
from miniagent.dependencies import DEPS_MODES
from miniagent.embeddings import (
    EMBEDDING_DEFAULT_BASE_URL,
    EMBEDDING_DEFAULT_MODEL,
    EMBEDDING_TIMEOUT,
)
from miniagent.errors import ConfigError
from miniagent.guard import MODES as GUARD_MODES
from miniagent.hitl import DEFAULT_CONFIRM_RULES, HITL_MODES, RULE_NAMES
from miniagent.planner import DEFAULT_VERIFY_ROUNDS, PLAN_MODES, VERIFY_MODES
from miniagent.snapshot import SNAPSHOT_MODES

DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_MODEL = "agnes-3.0-flash"


@dataclass(frozen=True)
class Config:
    provider: str = "mock"  # mock | openai
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    max_turns: int = 10
    max_context_tokens: int = 12000
    short_term_tokens: int = 2000  # 每轮带多少最近对话；不含 system 与记忆注入
    request_timeout: float = 60.0
    max_retries: int = 3  # 含首次请求的最大尝试次数
    rate_limit_rpm: float = 60.0  # 每分钟最多模型请求数，0 表示不限流
    soul_path: str = ""  # 人格文件；空 = 自动查找 ./soul.md 与 ~/.miniagent/soul.md
    user_path: str = ""  # 用户画像文件；空 = 自动查找 ./user.md 与 ~/.miniagent/user.md
    memory_enabled: bool = True
    memory_dir: str = ""  # 空 = ~/.miniagent/memory
    memory_batch_size: int = 6  # 攒够多少轮再蒸馏
    memory_top_k: int = 3
    gate_model: str = ""  # 记忆门用的模型名，空 = 与主模型相同
    prompt_guard: str = "annotate"  # off | annotate | block
    tool_deps: str = "model"  # off | model
    plan_mode: str = "off"  # off | on（Plan-and-Execute）
    plan_verify: str = "off"  # off | on；计划执行完的收尾校验，只在 plan_mode=on 时生效
    plan_verify_rounds: int = DEFAULT_VERIFY_ROUNDS  # 不通过时允许的补充执行轮次，0 = 不补
    no_progress: str = "on"  # off | on；相同 工具+参数+结果 反复出现时提前止损
    no_progress_repeat: int = DEFAULT_NO_PROGRESS_REPEAT  # 连续多少次相同指纹算无进展
    hitl: str = "off"  # off | on；默认关闭，不改变脚本 / 测试 / 自动运行的既有行为
    hitl_confirm: tuple[str, ...] = DEFAULT_CONFIRM_RULES  # 哪些操作需要用户确认
    tavily_api_key: str = ""  # 空 = web_search 调用时给出明确提示；不影响启动
    workdir: str = ""  # 空 = 启动进程时的 cwd；相对路径与 run_shell 都以它为基准
    snapshot: str = "on"  # on | off；高危操作执行前的 git 快照，默认开启
    # embedding 通道与主对话 provider 完全独立；没有 key 时自动关闭，检索返回空结果
    embedding_base_url: str = EMBEDDING_DEFAULT_BASE_URL
    embedding_model: str = EMBEDDING_DEFAULT_MODEL
    embedding_api_key: str = ""
    embedding_timeout: float = EMBEDDING_TIMEOUT


def load_config(path: str | Path = "config.toml") -> Config:
    """读取配置。文件不存在时全部走默认值 + 环境变量。"""
    data: dict[str, object] = {}
    config_path = Path(path)
    if config_path.is_file():
        try:
            with config_path.open("rb") as fp:
                data = tomllib.load(fp)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"配置文件 {config_path} 格式错误：{exc}") from exc

    def pick(env_name: str, key: str, default: str) -> str:
        return os.environ.get(env_name) or str(data.get(key, default))

    provider = pick("MINIAGENT_PROVIDER", "provider", "mock").lower()
    if provider not in {"mock", "openai"}:
        raise ConfigError(f"未知的 provider：{provider}（可选 mock / openai）")

    base_url = pick("MINIAGENT_BASE_URL", "base_url", DEFAULT_BASE_URL).rstrip("/")
    api_key = os.environ.get("MINIAGENT_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    api_key = api_key or str(data.get("api_key", ""))
    model = pick("MINIAGENT_MODEL", "model", DEFAULT_MODEL)

    raw_turns = os.environ.get("MINIAGENT_MAX_TURNS") or data.get("max_turns", 10)
    try:
        max_turns = int(raw_turns)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"max_turns 必须是整数，当前为 {raw_turns!r}") from exc
    if max_turns < 1:
        raise ConfigError("max_turns 必须大于 0")

    raw_context = os.environ.get("MINIAGENT_MAX_CONTEXT_TOKENS") or data.get(
        "max_context_tokens", 12000
    )
    try:
        max_context_tokens = int(raw_context)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"max_context_tokens 必须是整数，当前为 {raw_context!r}") from exc
    if max_context_tokens < 1000:
        raise ConfigError("max_context_tokens 不能小于 1000")

    raw_short = os.environ.get("MINIAGENT_SHORT_TERM_TOKENS") or data.get(
        "short_term_tokens", 2000
    )
    try:
        short_term_tokens = int(raw_short)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"short_term_tokens 必须是整数，当前为 {raw_short!r}"
        ) from exc
    if short_term_tokens < 1:
        raise ConfigError("short_term_tokens 必须大于 0")

    raw_retries = os.environ.get("MINIAGENT_MAX_RETRIES") or data.get("max_retries", 3)
    try:
        max_retries = int(raw_retries)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"max_retries 必须是整数，当前为 {raw_retries!r}") from exc
    if max_retries < 1:
        raise ConfigError("max_retries 必须大于 0（含首次尝试）")

    raw_rpm = os.environ.get("MINIAGENT_RATE_LIMIT_RPM") or data.get("rate_limit_rpm", 60)
    try:
        rate_limit_rpm = float(raw_rpm)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"rate_limit_rpm 必须是数字，当前为 {raw_rpm!r}") from exc
    if rate_limit_rpm < 0:
        raise ConfigError("rate_limit_rpm 不能为负数（0 表示不限流）")

    memory_enabled = _to_bool(
        os.environ.get("MINIAGENT_MEMORY_ENABLED")
        if os.environ.get("MINIAGENT_MEMORY_ENABLED") is not None
        else data.get("memory_enabled", True),
        "memory_enabled",
    )

    raw_batch = os.environ.get("MINIAGENT_MEMORY_BATCH_SIZE") or data.get(
        "memory_batch_size", 6
    )
    try:
        memory_batch_size = int(raw_batch)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"memory_batch_size 必须是整数，当前为 {raw_batch!r}") from exc
    if memory_batch_size < 1:
        raise ConfigError("memory_batch_size 必须大于 0")

    raw_top_k = os.environ.get("MINIAGENT_MEMORY_TOP_K") or data.get("memory_top_k", 3)
    try:
        memory_top_k = int(raw_top_k)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"memory_top_k 必须是整数，当前为 {raw_top_k!r}") from exc
    if memory_top_k < 1:
        raise ConfigError("memory_top_k 必须大于 0")

    prompt_guard = (
        os.environ.get("MINIAGENT_PROMPT_GUARD") or str(data.get("prompt_guard", "annotate"))
    ).strip().lower()
    if prompt_guard not in GUARD_MODES:
        raise ConfigError(
            f"未知的 prompt_guard：{prompt_guard}（可选 {' / '.join(GUARD_MODES)}）"
        )

    tool_deps = (
        os.environ.get("MINIAGENT_TOOL_DEPS") or str(data.get("tool_deps", "model"))
    ).strip().lower()
    if tool_deps not in DEPS_MODES:
        raise ConfigError(f"未知的 tool_deps：{tool_deps}（可选 {' / '.join(DEPS_MODES)}）")

    plan_mode = (
        os.environ.get("MINIAGENT_PLAN_MODE") or str(data.get("plan_mode", "off"))
    ).strip().lower()
    if plan_mode not in PLAN_MODES:
        raise ConfigError(f"未知的 plan_mode：{plan_mode}（可选 {' / '.join(PLAN_MODES)}）")

    plan_verify = (
        os.environ.get("MINIAGENT_PLAN_VERIFY") or str(data.get("plan_verify", "off"))
    ).strip().lower()
    if plan_verify not in VERIFY_MODES:
        raise ConfigError(
            f"未知的 plan_verify：{plan_verify}（可选 {' / '.join(VERIFY_MODES)}）"
        )

    raw_verify_rounds = os.environ.get("MINIAGENT_PLAN_VERIFY_ROUNDS") or data.get(
        "plan_verify_rounds", DEFAULT_VERIFY_ROUNDS
    )
    try:
        plan_verify_rounds = int(raw_verify_rounds)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"plan_verify_rounds 必须是整数，当前为 {raw_verify_rounds!r}"
        ) from exc
    if plan_verify_rounds < 0:
        raise ConfigError("plan_verify_rounds 不能为负数（0 表示不做补充执行）")

    no_progress = (
        os.environ.get("MINIAGENT_NO_PROGRESS") or str(data.get("no_progress", "on"))
    ).strip().lower()
    if no_progress not in NO_PROGRESS_MODES:
        raise ConfigError(
            f"未知的 no_progress：{no_progress}（可选 {' / '.join(NO_PROGRESS_MODES)}）"
        )

    raw_no_progress_repeat = os.environ.get(
        "MINIAGENT_NO_PROGRESS_REPEAT"
    ) or data.get("no_progress_repeat", DEFAULT_NO_PROGRESS_REPEAT)
    try:
        no_progress_repeat = int(raw_no_progress_repeat)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"no_progress_repeat 必须是整数，当前为 {raw_no_progress_repeat!r}"
        ) from exc
    if no_progress_repeat < 2:
        raise ConfigError("no_progress_repeat 不能小于 2")

    hitl = (os.environ.get("MINIAGENT_HITL") or str(data.get("hitl", "off"))).strip().lower()
    if hitl not in HITL_MODES:
        raise ConfigError(f"未知的 hitl：{hitl}（可选 {' / '.join(HITL_MODES)}）")

    raw_rules = os.environ.get("MINIAGENT_HITL_CONFIRM") or data.get(
        "hitl_confirm", ",".join(DEFAULT_CONFIRM_RULES)
    )
    hitl_confirm = tuple(
        part.strip().lower() for part in str(raw_rules).split(",") if part.strip()
    )
    unknown_rules = [rule for rule in hitl_confirm if rule not in RULE_NAMES]
    if unknown_rules:
        raise ConfigError(
            f"未知的 hitl_confirm 规则：{'、'.join(unknown_rules)}"
            f"（可选 {' / '.join(RULE_NAMES)}，逗号分隔）"
        )

    search_api_key = str(
        os.environ.get("TAVILY_API_KEY") or data.get("tavily_api_key", "")
    ).strip()
    workdir = str(
        os.environ.get("MINIAGENT_WORKDIR") or data.get("workdir", "")
    ).strip()
    snapshot = (
        os.environ.get("MINIAGENT_SNAPSHOT") or str(data.get("snapshot", "on"))
    ).strip().lower()
    if snapshot not in SNAPSHOT_MODES:
        raise ConfigError(
            f"未知的 snapshot：{snapshot}（可选 {' / '.join(SNAPSHOT_MODES)}）"
        )

    embedding_base_url = (
        os.environ.get("MINIAGENT_EMBEDDING_BASE_URL")
        or str(data.get("embedding_base_url", EMBEDDING_DEFAULT_BASE_URL))
    ).strip().rstrip("/")
    embedding_model = (
        os.environ.get("MINIAGENT_EMBEDDING_MODEL")
        or str(data.get("embedding_model", EMBEDDING_DEFAULT_MODEL))
    ).strip()
    # 只从环境变量或 config 读；没有就关闭向量通道，不借用主对话的 key
    embedding_api_key = (
        os.environ.get("SILICONFLOW_API_KEY")
        or str(data.get("embedding_api_key", ""))
    ).strip()
    raw_embedding_timeout = os.environ.get(
        "MINIAGENT_EMBEDDING_TIMEOUT"
    ) or data.get("embedding_timeout", EMBEDDING_TIMEOUT)
    try:
        embedding_timeout = float(raw_embedding_timeout)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"embedding_timeout 必须是数字，当前为 {raw_embedding_timeout!r}"
        ) from exc
    if embedding_timeout <= 0:
        raise ConfigError("embedding_timeout 必须大于 0")

    return Config(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        short_term_tokens=short_term_tokens,
        max_retries=max_retries,
        rate_limit_rpm=rate_limit_rpm,
        soul_path=str(os.environ.get("MINIAGENT_SOUL_PATH") or data.get("soul_path", "")),
        user_path=str(os.environ.get("MINIAGENT_USER_PATH") or data.get("user_path", "")),
        memory_enabled=memory_enabled,
        memory_dir=str(os.environ.get("MINIAGENT_MEMORY_DIR") or data.get("memory_dir", "")),
        memory_batch_size=memory_batch_size,
        memory_top_k=memory_top_k,
        gate_model=str(os.environ.get("MINIAGENT_GATE_MODEL") or data.get("gate_model", "")),
        prompt_guard=prompt_guard,
        tool_deps=tool_deps,
        plan_mode=plan_mode,
        plan_verify=plan_verify,
        plan_verify_rounds=plan_verify_rounds,
        no_progress=no_progress,
        no_progress_repeat=no_progress_repeat,
        hitl=hitl,
        hitl_confirm=hitl_confirm,
        tavily_api_key=search_api_key,
        workdir=workdir,
        snapshot=snapshot,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_timeout=embedding_timeout,
    )


def _to_bool(value: object, key: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} 必须是布尔值，当前为 {value!r}")
