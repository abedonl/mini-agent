"""安全边界：路径白名单与危险命令黑名单。

这不是容器沙箱：只拦截明确的越界读写与常见破坏性命令，
边界写在 README，真正执行仍发生在本机。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from miniagent.errors import ConfigError, ToolError

# 只做粗粒度匹配，宁可漏掉变体，也不误伤日常命令（如 rm -rf ./build）
DANGEROUS_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\brm\s+(?:-[a-zA-Z]+\s+)*(?:/|~|/\*|~/\*|\./\*|\*)(?:\s|$)",
            re.I,
        ),
        "删除根目录/家目录/全部文件",
    ),
    (re.compile(r"\bmkfs(?:\.\w+)?\b", re.I), "格式化文件系统"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/", re.I), "直接写入块设备"),
    (re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.I), "fork bomb"),
    (re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.I), "关机或重启系统"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.I), "格式化磁盘"),
    (re.compile(r"\b(?:del|erase|rd|rmdir)\s+/[sfq]\b[^\n]*[a-zA-Z]:", re.I), "递归删除磁盘内容"),
    (
        re.compile(
            r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|pwsh|powershell)\b",
            re.I,
        ),
        "下载内容直接交给 shell 执行",
    ),
    (
        re.compile(r"\bremove-item\b[^\n]*-recurse[^\n]*[a-zA-Z]:\\?\s*$", re.I),
        "递归删除磁盘内容",
    ),
)


def check_command(command: str) -> None:
    """命中黑名单时抛 ToolError，由注册表转成错误结果回传给模型。"""
    for pattern, reason in DANGEROUS_RULES:
        if pattern.search(command):
            raise ToolError(f"危险命令已被拦截（{reason}）：{command[:120]}")


class PathPolicy:
    """路径白名单：resolve 后必须落在允许目录内。

    resolve 会展开 ~、消除 .. 并跟随已存在的符号链接，
    因此白名单内的软链指向外部时同样会被拒绝。
    """

    def __init__(self, roots: Iterable[Path | str]) -> None:
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)
        if not self.roots:
            raise ConfigError("路径白名单不能为空")

    def resolve(self, raw: str, *, for_write: bool = False) -> Path:
        target = Path(raw).expanduser().resolve()
        if not any(target == root or root in target.parents for root in self.roots):
            action = "写入" if for_write else "读取"
            allowed = "、".join(str(root) for root in self.roots)
            raise ToolError(f"拒绝{action}白名单外路径：{target}（允许目录：{allowed}）")
        return target
