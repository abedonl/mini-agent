"""人格与画像层：soul.md 与 user.md 的定位与加载。

两者职责不同，不要混用：
- soul.md：Agent 自身的人格、原则与行为底线（固定身份），对应 system prompt 最前面；
- user.md：对话对象的用户画像、偏好、背景与沟通习惯，回答"对话对象是谁"。

它们都不是记忆：长期记忆是 memory.jsonl / memory.md 的蒸馏结果，
短期上下文是按 short_term_tokens 裁剪出来的请求窗口。

优先级：显式路径 > 项目目录 ./xxx.md > 用户目录 ~/.miniagent/xxx.md > 默认。
文件缺失或为空时回退到默认值，不抛错。
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SOUL = (
    "你是 miniagent，一个运行在用户终端里的命令行 agent。\n"
    "行为约束：\n"
    "1. 需要事实（文件内容、命令结果）时调用工具，不要编造。\n"
    "2. 工具返回错误时，先读错误信息再决定是否修正参数重试。\n"
    "3. 每次只做必要的最小步骤，完成任务后直接给出结论。\n"
    "4. 用中文回答，简洁直接。"
)


def project_soul_path(cwd: Path | str | None = None) -> Path:
    return Path(cwd or Path.cwd()) / "soul.md"


def user_soul_path() -> Path:
    return Path.home() / ".miniagent" / "soul.md"


def project_user_path(cwd: Path | str | None = None) -> Path:
    return Path(cwd or Path.cwd()) / "user.md"


def user_user_path() -> Path:
    return Path.home() / ".miniagent" / "user.md"


def load_soul(
    explicit: Path | str | None = None,
    *,
    cwd: Path | str | None = None,
) -> tuple[str, str]:
    """返回 (人格文本, 来源)。来源取值：explicit / project / user / default。"""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text, "explicit"
        return DEFAULT_SOUL, "default"

    for path, source in (
        (project_soul_path(cwd), "project"),
        (user_soul_path(), "user"),
    ):
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text, source
    return DEFAULT_SOUL, "default"


def load_user_profile(
    explicit: Path | str | None = None,
    *,
    cwd: Path | str | None = None,
) -> tuple[str, str]:
    """返回 (用户画像文本, 来源)。来源取值：explicit / project / user / none。

    没有内置默认画像：缺失时返回空字符串，system prompt 里不出现这一段。
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text, "explicit"
        return "", "none"

    for path, source in (
        (project_user_path(cwd), "project"),
        (user_user_path(), "user"),
    ):
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text, source
    return "", "none"
