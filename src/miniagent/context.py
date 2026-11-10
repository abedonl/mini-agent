"""上下文控制：粗粒度 token 估算与裁剪。

裁剪只作用于"即将发给模型的消息视图"，不修改会话文件与 Agent 内存里的完整历史。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from miniagent.llm import Message, ToolCall

DEFAULT_MAX_TOKENS = 12000
DEFAULT_SHORT_TERM_TOKENS = 2000
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


PAIRING_PLACEHOLDER = (
    "【轨迹修复：缺失的工具结果】\n"
    "工具 {name}（call_id={call_id}）在上一条 assistant 消息里被调用，"
    "但没有找到对应的结果消息（可能进程中断或结果未落盘）。"
    "这里由系统补一条占位，不代表执行成功或失败；需要时请重新调用或向用户核实。"
)


def _default_placeholder(call: ToolCall) -> str:
    return PAIRING_PLACEHOLDER.format(name=call.name or "未知工具", call_id=call.id)


@dataclass
class PairingReport:
    """一次轨迹体检的结果；messages 是新列表，原列表与 Message 对象不被修改。"""

    messages: list[Message]
    filled: list[str] = field(default_factory=list)
    dropped_calls: int = 0
    dropped_results: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.filled or self.dropped_calls or self.dropped_results)


def repair_pairing(
    messages: list[Message],
    *,
    placeholder: Callable[[ToolCall], str] | None = None,
) -> PairingReport:
    """修复工具调用与结果的结构配对，供"注入前体检"和断点恢复共用。

    规则（按顺序、幂等）：
    - 丢掉没有名字的工具调用；调用全被丢掉时该 assistant 降级为普通文本消息；
    - 丢掉找不到对应调用的孤儿 tool 结果；
    - assistant.tool_calls 后面缺少的 tool 结果就地补占位，占位文本明确标注是补的。

    只读不写：返回新列表，不修改传入的 messages，也不碰会话文件与 checkpoint。
    """
    fill = placeholder or _default_placeholder
    report = PairingReport(messages=[])
    out = report.messages
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        index += 1

        if message.role == "tool":
            # 不在任何 assistant.tool_calls 之后的 tool 消息没有配对对象
            report.dropped_results += 1
            continue

        if message.role != "assistant" or not message.tool_calls:
            out.append(message)
            continue

        calls = [call for call in message.tool_calls if (call.name or "").strip()]
        if len(calls) != len(message.tool_calls):
            report.dropped_calls += len(message.tool_calls) - len(calls)
            message = replace(message, tool_calls=calls)
        if not calls:
            out.append(replace(message, tool_calls=[]))
            continue

        out.append(message)
        expected = {call.id for call in calls}
        answered: set[str] = set()
        while index < total and messages[index].role == "tool":
            result = messages[index]
            index += 1
            call_id = result.tool_call_id or ""
            if call_id in expected and call_id not in answered:
                out.append(result)
                answered.add(call_id)
            else:
                report.dropped_results += 1

        for call in calls:
            if call.id in answered:
                continue
            out.append(
                Message(
                    role="tool",
                    content=fill(call),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
            report.filled.append(call.id)
    return report


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


@dataclass
class WindowStats:
    """短期窗口的账本：预算只约束对话历史，不含 system 与记忆注入。"""

    budget: int
    from_tokens: int
    to_tokens: int
    total_messages: int
    kept_messages: int
    total_turns: int
    kept_turns: int
    trimmed: bool

    def detail(self) -> str:
        """trace detail：与 trim 事件同风格，额外给出消息/轮次与是否裁剪。"""
        return (
            f"stage=short_term budget={self.budget} from={self.from_tokens} "
            f"to={self.to_tokens} messages={self.kept_messages}/{self.total_messages} "
            f"turns={self.kept_turns}/{self.total_turns} trimmed={int(self.trimmed)}"
        )


@dataclass
class WindowResult:
    messages: list[Message]
    stats: WindowStats


def apply_short_term_window(
    messages: list[Message],
    budget: int = DEFAULT_SHORT_TERM_TOKENS,
) -> WindowResult:
    """把"最近对话历史"收进短期预算；system 与记忆提示不占额度。

    - 以 user 消息切分轮次，assistant.tool_calls 与对应 tool 结果始终同进同出；
    - 至少保留最近一轮完整对话，即使它本身已超预算；
    - 只改请求视图，原始消息对象与会话历史不变。
    """
    system = [message for message in messages if message.role == "system"]
    turns = split_turns([message for message in messages if message.role != "system"])
    history = _flatten(turns)
    from_tokens = estimate_messages(history)
    total_messages = len(messages)

    kept = turns[-1:]
    if len(turns) > 1:
        for turn in reversed(turns[:-1]):
            # 多保留一轮后仍在预算内才留；否则从这一轮开始全部丢弃
            if estimate_messages(_flatten([turn, *kept])) > budget:
                break
            kept = [turn, *kept]

    kept_history = _flatten(kept)
    trimmed = len(kept) < len(turns)
    stats = WindowStats(
        budget=budget,
        from_tokens=from_tokens,
        to_tokens=estimate_messages(kept_history),
        total_messages=total_messages,
        kept_messages=len(system) + len(kept_history),
        total_turns=len(turns),
        kept_turns=len(kept),
        trimmed=trimmed,
    )
    if not trimmed:
        return WindowResult(list(messages), stats)
    return WindowResult(system + kept_history, stats)
