"""配置：环境变量 + 可选的 config.toml，环境变量优先。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from miniagent.dependencies import DEPS_MODES
from miniagent.errors import ConfigError
from miniagent.guard import MODES as GUARD_MODES

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Config:
    provider: str = "mock"  # mock | openai
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    max_turns: int = 10
    max_context_tokens: int = 12000
    request_timeout: float = 60.0
    max_retries: int = 3  # 含首次请求的最大尝试次数
    rate_limit_rpm: float = 60.0  # 每分钟最多模型请求数，0 表示不限流
    soul_path: str = ""  # 空 = 自动查找 ./soul.md 与 ~/.miniagent/soul.md
    memory_enabled: bool = True
    memory_dir: str = ""  # 空 = ~/.miniagent/memory
    memory_batch_size: int = 6  # 攒够多少轮再蒸馏
    memory_top_k: int = 3
    gate_model: str = ""  # 记忆门用的模型名，空 = 与主模型相同
    prompt_guard: str = "annotate"  # off | annotate | block
    tool_deps: str = "model"  # off | model


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

    return Config(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        max_retries=max_retries,
        rate_limit_rpm=rate_limit_rpm,
        soul_path=str(os.environ.get("MINIAGENT_SOUL_PATH") or data.get("soul_path", "")),
        memory_enabled=memory_enabled,
        memory_dir=str(os.environ.get("MINIAGENT_MEMORY_DIR") or data.get("memory_dir", "")),
        memory_batch_size=memory_batch_size,
        memory_top_k=memory_top_k,
        gate_model=str(os.environ.get("MINIAGENT_GATE_MODEL") or data.get("gate_model", "")),
        prompt_guard=prompt_guard,
        tool_deps=tool_deps,
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
