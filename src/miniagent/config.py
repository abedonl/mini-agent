"""配置：环境变量 + 可选的 config.toml，环境变量优先。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from miniagent.errors import ConfigError

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

    return Config(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        max_retries=max_retries,
        rate_limit_rpm=rate_limit_rpm,
    )
