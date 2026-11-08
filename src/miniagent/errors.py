"""异常分类：每类错误对应 CLI 的一个退出码。"""


class MiniAgentError(Exception):
    """所有可预期错误的基类。"""

    exit_code = 1


class ConfigError(MiniAgentError):
    """配置缺失或非法，例如未设置模型密钥。"""

    exit_code = 2


class ProviderError(MiniAgentError):
    """模型接口调用失败（网络、HTTP 状态码、响应格式）。"""

    exit_code = 3


class ToolError(MiniAgentError):
    """工具执行层面的错误。工具错误不会中断循环，通常回传给模型。"""

    exit_code = 4
