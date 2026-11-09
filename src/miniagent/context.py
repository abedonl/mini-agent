"""上下文控制：粗粒度 token 估算与裁剪。

裁剪只作用于"即将发给模型的消息视图"，不修改会话文件与 Agent 内存里的完整历史。
"""

from __future__ import annotations

from dataclasses import replace

from miniagent.llm import Message

DEFAULT_MAX_TOKENS = 12000
DEFAULT_KEEP_TURNS = 6
OLD_TOOL_OUTPUT_LIMIT = 600
HARD_TOOL_OUTPUT_LIMIT = 200


def estimate_tokens(text: str) -> int:
    """粗略估算：中日韩字符按 1 token，其余按 4 字符 1 token。

    不追求精确，只用于判断"是否明显超长"，避免引入 tokenizer 依赖。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u2e80" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af")
    return cjk + (len(text) - cjk) // 4


def estimate_messages(messages: list[Message]) -> int:
    total = 0
    for message in messages:
        total += 4  # 每条消息的角色与结构开销
        total += estimate_tokens(message.content or "")
        for call in message.tool_calls:
            total += estimate_tokens(call.name) + estimate_tokens(call.arguments)
    return total


def split_turns(messages: list[Message]) -> list[list[Message]]:
    """以 user 消息为界切分轮次，保证 assistant.tool_calls 与其 tool 结果不被拆散。"""
    turns: list[list[Message]] = []
    for message in messages:
        if message.role == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def _flatten(turns: list[list[Message]]) -> list[Message]:
    return [message for turn in turns for message in turn]


def _truncate_tool_outputs(turns: list[list[Message]], limit: int) -> None:
    for turn in turns:
        for index, message in enumerate(turn):
            content = message.content or ""
            if message.role == "tool" and len(content) > limit:
                turn[index] = replace(
                    message, content=content[:limit] + "\n...[历史工具输出已截断]"
                )


def trim_messages(
    messages: list[Message],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    keep_turns: int = DEFAULT_KEEP_TURNS,
) -> list[Message]:
    """保留 system + 最近若干轮；仍超预算则继续丢最旧的轮次，并截断旧工具输出。

    返回新列表，原始 Message 对象不被修改。
    """
    if estimate_messages(messages) <= max_tokens:
        return list(messages)

    system = [message for message in messages if message.role == "system"][:1]
    turns = split_turns([message for message in messages if message.role != "system"])
    kept = turns[-keep_turns:]

    # 先截断较早轮次的工具输出，尽量保住轮次结构；仍超预算再丢最旧的整轮
    _truncate_tool_outputs(kept[:-2], OLD_TOOL_OUTPUT_LIMIT)
    while len(kept) > 1 and estimate_messages(system + _flatten(kept)) > max_tokens:
        kept.pop(0)

    # 只剩最后一轮仍然超预算：硬截断其工具输出，避免请求直接失败
    if estimate_messages(system + _flatten(kept)) > max_tokens:
        _truncate_tool_outputs(kept, HARD_TOOL_OUTPUT_LIMIT)

    return system + _flatten(kept)
