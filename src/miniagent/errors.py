"""异常分类：每类错误对应 CLI 的一个退出码。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecoverySummary:
    """一次失败前已经做过的恢复动作，用于向用户解释"不是直接失败"。

    由请求层填写 attempts / last_status / last_error，Agent 再补上本轮发生的
    降级动作（记忆门放行、摘要退硬裁、依赖判定退串行等）。
    """

    attempts: int = 0
    last_status: int | None = None
    last_error: str = ""
    degraded: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts: list[str] = []
        if self.attempts > 1:
            parts.append(f"重试 {self.attempts - 1} 次（共 {self.attempts} 次尝试）")
        elif self.attempts == 1:
            parts.append("未重试（共 1 次尝试）")
        if self.last_status is not None:
            parts.append(f"最后状态码 HTTP {self.last_status}")
        elif self.last_error:
            parts.append(f"最后错误：{self.last_error}")
        if self.degraded:
            # 去重但保持出现顺序：同一降级在一轮里可能重复记录
            parts.append("降级：" + "、".join(dict.fromkeys(self.degraded)))
        return "；".join(parts)


class MiniAgentError(Exception):
    """所有可预期错误的基类。"""

    exit_code = 1


class ConfigError(MiniAgentError):
    """配置缺失或非法，例如未设置模型密钥。"""

    exit_code = 2


class ProviderError(MiniAgentError):
    """模型接口调用失败（网络、HTTP 状态码、响应格式）。

    recovery 记录失败前已经尝试过的恢复动作；没有经过重试的错误为 None，
    打印时会退化成只有一行消息，保持与旧版本一致的输出。
    """

    exit_code = 3

    def __init__(self, message: str, *, recovery: RecoverySummary | None = None) -> None:
        super().__init__(message)
        self.recovery = recovery

    @property
    def recovery_text(self) -> str:
        return self.recovery.describe() if self.recovery is not None else ""


class ToolError(MiniAgentError):
    """工具执行层面的错误。工具错误不会中断循环，通常回传给模型。"""

    exit_code = 4


class PermissionDenied(ToolError):
    """硬边界拦截：路径白名单、危险命令黑名单。

    hint 说明需要用户放行什么，由注册表拼进回传模型的错误文本里，
    避免只抛一句"失败"而没人知道该改哪个配置。
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint
