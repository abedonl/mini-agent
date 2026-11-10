"""安全边界：路径白名单与危险命令黑名单。

这不是容器沙箱：只拦截明确的越界读写与常见破坏性命令，
边界写在 README，真正执行仍发生在本机。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from miniagent.errors import ConfigError, PermissionDenied

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
            raise PermissionDenied(
                f"危险命令已被拦截（{reason}）：{command[:120]}",
                hint="该命令属于危险命令黑名单，需要用户手动执行或换成安全等价命令",
            )


def _normalize_roots(roots: Iterable[Path | str]) -> tuple[Path, ...]:
    normalized = tuple(Path(root).expanduser().resolve() for root in roots)
    if not normalized:
        raise ConfigError("路径白名单不能为空")
    return normalized


def _inside(target: Path, roots: tuple[Path, ...]) -> bool:
    return any(target == root or root in target.parents for root in roots)


def check_workdir(raw: str | Path, roots: Iterable[Path | str]) -> Path:
    """校验工作目录：必须存在、是目录，且落在授权根内。

    前端传来的路径同样走这里，因此不能靠"界面上填一个路径"绕过白名单；
    越权时把授权根和授权方式写进错误信息，便于用户自己补 --allow-dir。
    """
    normalized = _normalize_roots(roots)
    target = Path(raw).expanduser()
    if not str(target).strip():
        raise ConfigError("工作目录不能为空")
    if not target.is_absolute():
        target = Path.cwd() / target
    resolved = target.resolve()
    if not resolved.exists():
        raise ConfigError(f"工作目录不存在：{resolved}")
    if not resolved.is_dir():
        raise ConfigError(f"工作目录不是目录：{resolved}")
    if not _inside(resolved, normalized):
        allowed = "、".join(str(root) for root in normalized)
        raise ConfigError(
            f"工作目录不在授权根内：{resolved}（授权根：{allowed}）；"
            f"如确需使用，请启动时追加 --allow-dir {resolved}"
        )
    return resolved


class PathPolicy:
    """路径白名单：resolve 后必须落在允许目录内。

    resolve 会展开 ~、消除 .. 并跟随已存在的符号链接，
    因此白名单内的软链指向外部时同样会被拒绝。
    工作目录只决定相对路径基准和 run_shell 的 cwd，不会扩大白名单。
    """

    def __init__(self, roots: Iterable[Path | str], workdir: str | Path | None = None) -> None:
        self.roots = _normalize_roots(roots)
        if workdir is not None:
            self.workdir = check_workdir(workdir, self.roots)
            return
        cwd = Path.cwd().resolve()
        # 未显式指定时默认 cwd；cwd 不在白名单内（如库调用只传 roots）
        # 就退回第一个授权根，且不强制它此刻已存在。
        self.workdir = cwd if _inside(cwd, self.roots) else self.roots[0]

    def resolve(self, raw: str, *, for_write: bool = False) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.workdir / path
        target = path.resolve()
        if not _inside(target, self.roots):
            action = "写入" if for_write else "读取"
            allowed = "、".join(str(root) for root in self.roots)
            raise PermissionDenied(
                f"拒绝{action}白名单外路径：{target}（允许目录：{allowed}）",
                hint="把目标目录加入 --allow-dir 白名单后才能继续",
            )
        return target
