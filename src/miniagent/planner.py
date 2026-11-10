"""Plan-and-Execute：结构化计划生成、本地校验、逐步执行与重规划。

计划由模型生成，但"能不能执行"由本地校验决定：
工具名、参数 Schema、步骤上限、前序引用都必须过；校验不过只给模型一次修正机会，
再不过就退回 ReAct，避免把预算耗在格式纠正上。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from miniagent.llm import Message
from miniagent.tools.registry import ToolRegistry, validate_args

PLAN_MODES = ("off", "on")
VERIFY_MODES = ("off", "on")
DEFAULT_MAX_PLAN_STEPS = 8
DEFAULT_MAX_REPLANS = 2
DEFAULT_VERIFY_ROUNDS = 1
REVIEW_ACTIONS = ("continue", "replan", "finish")
VERDICTS = ("pass", "fail")

MAX_OUTPUT_CHARS = 800  # 回喂给模型的单步输出上限


@dataclass(frozen=True)
class PlanStep:
    index: int  # 从 1 开始的步骤序号，depends_on 用它引用
    intent: str
    tool: str
    args: dict[str, Any]
    depends_on: tuple[int, ...] = ()


@dataclass(frozen=True)
class Plan:
    steps: tuple[PlanStep, ...]


@dataclass(frozen=True)
class PlanCheck:
    plan: Plan | None
    code: str = ""
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.plan is not None


@dataclass(frozen=True)
class Review:
    action: str
    steps: tuple[PlanStep, ...] = ()
    answer: str = ""
    code: str = ""
    errors: tuple[str, ...] = ()
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.action in {"continue", "finish"} or bool(self.steps)


@dataclass
class StepRun:
    """一步的执行记录，用于回喂模型与重规划。"""

    step: PlanStep
    ok: bool
    output: str
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class VerifyCheck:
    """一次收尾校验的结果。unavailable 表示校验调用本身失败，不能当作通过。"""

    verdict: str  # pass | fail
    reason: str = ""
    evidence: tuple[str, ...] = ()
    code: str = ""
    unavailable: bool = False

    @property
    def ok(self) -> bool:
        return self.verdict == "pass" and not self.code and not self.unavailable


def _extract_json(text: str) -> Any | None:
    """容忍 ```json 围栏与前后解释文字，取出第一个 JSON 值。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    fenced = re.sub(r"^```[A-Za-z]*\s*", "", cleaned)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    for candidate in (fenced, cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    start = cleaned.find("{")
    if start == -1:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None
    return value


def validate_steps(
    raw_steps: Any,
    registry: ToolRegistry,
    max_steps: int = DEFAULT_MAX_PLAN_STEPS,
) -> tuple[tuple[PlanStep, ...] | None, str, tuple[str, ...]]:
    """逐步校验；任何一步不合格就整体不接受（避免半份计划执行到一半才发现问题）。"""
    if not isinstance(raw_steps, list):
        return None, "bad_shape", ("steps 必须是数组",)
    if not raw_steps:
        return None, "empty_plan", ("计划为空：steps 不能为空",)
    if len(raw_steps) > max_steps:
        return None, "too_many_steps", (f"步骤数 {len(raw_steps)} 超过上限 {max_steps}",)

    known = set(registry.names())
    steps: list[PlanStep] = []
    errors: list[str] = []
    code = ""

    for position, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            errors.append(f"第 {position} 步必须是对象")
            code = code or "bad_step"
            continue

        intent = str(item.get("intent") or "").strip()
        tool = str(item.get("tool") or "").strip()
        args = item.get("args", {})
        depends = item.get("depends_on") or []

        if not intent:
            errors.append(f"第 {position} 步缺少 intent")
            code = code or "bad_step"
        if not tool:
            errors.append(f"第 {position} 步缺少 tool")
            code = code or "bad_step"
        if not isinstance(args, dict):
            errors.append(f"第 {position} 步 args 必须是对象")
            code = code or "bad_args"
            args = {}
        if not isinstance(depends, list) or any(
            isinstance(dep, bool) or not isinstance(dep, int) for dep in depends
        ):
            errors.append(f"第 {position} 步 depends_on 必须是整数数组")
            code = code or "bad_dep"
            depends = []

        if tool and tool not in known:
            errors.append(f"第 {position} 步工具不存在：{tool}")
            code = code or "unknown_tool"
        elif tool:
            tool_obj = registry.get(tool)
            if tool_obj is not None:
                for problem in validate_args(tool_obj.parameters, args):
                    errors.append(f"第 {position} 步 {problem}")
                    code = code or "bad_args"

        for dep in depends:
            if dep < 1 or dep >= position:
                errors.append(f"第 {position} 步 depends_on 引用了不存在的前序步骤：{dep}")
                code = code or "bad_dep"

        if intent and tool and tool in known and isinstance(args, dict):
            steps.append(PlanStep(position, intent, tool, args, tuple(depends)))

    if errors:
        return None, code or "bad_step", tuple(errors)
    return tuple(steps), "", ()


def parse_plan(text: str, registry: ToolRegistry, max_steps: int) -> PlanCheck:
    if not (text or "").strip():
        return PlanCheck(None, "empty_text", ("模型没有返回计划内容",))
    data = _extract_json(text)
    if data is None:
        return PlanCheck(None, "parse_error", ("计划不是合法 JSON",))
    if not isinstance(data, dict) or "steps" not in data:
        return PlanCheck(None, "bad_shape", ("计划必须是包含 steps 字段的对象",))
    steps, code, errors = validate_steps(data["steps"], registry, max_steps)
    if steps is None:
        return PlanCheck(None, code, errors)
    return PlanCheck(Plan(steps))


def parse_review(text: str, registry: ToolRegistry, max_steps: int) -> Review:
    raw = text or ""
    if not raw.strip():
        return Review("", code="empty_text", errors=("模型没有返回决策",), raw=raw)
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return Review("", code="parse_error", errors=("决策不是合法 JSON 对象",), raw=raw)
    action = str(data.get("action") or "").strip().lower()
    if action not in REVIEW_ACTIONS:
        return Review(
            "", code="bad_action", errors=(f"action 必须是 {' / '.join(REVIEW_ACTIONS)}",), raw=raw
        )
    if action == "continue":
        return Review("continue", raw=raw)
    if action == "finish":
        answer = str(data.get("answer") or "").strip()
        if not answer:
            return Review("", code="bad_review", errors=("finish 必须给出 answer",), raw=raw)
        return Review("finish", answer=answer, raw=raw)
    steps, code, errors = validate_steps(data.get("steps"), registry, max_steps)
    if steps is None:
        return Review("", code=code, errors=errors, raw=raw)
    return Review("replan", steps=steps, raw=raw)


# ---- 提示词 ---------------------------------------------------------------


def _tool_catalog(registry: ToolRegistry) -> str:
    lines = []
    for spec in registry.specs():
        params = json.dumps(spec["parameters"], ensure_ascii=False)
        lines.append(f"- {spec['name']}：{spec['description']}\n  参数 Schema：{params}")
    return "\n".join(lines)


PLAN_HEADER = """你是任务规划器。只输出一个 JSON 对象，不要解释、不要 Markdown、不要代码块。
输出格式：
{"steps":[{"intent":"本步要做什么","tool":"工具名","args":{...},"depends_on":[1]}]}
规则：
- 每步只调用一个工具；需要多个工具时拆成多个步骤。
- depends_on 可省略，只能引用序号更小的步骤（步骤从 1 开始）。
- tool 必须来自下面的可用工具；args 必须符合该工具的 JSON Schema。
- 按执行顺序排列，只规划完成任务必需的步骤。"""

REVIEW_HEADER = """你是计划执行的监督者。只输出一个 JSON 对象，不要解释、不要 Markdown。
每次只能三选一：
{"action":"continue"} —— 按原计划继续下一步；
{"action":"replan","steps":[...]} —— 剩余步骤不再适用，给出新的剩余步骤
（格式同计划，只能引用新列表内的前序步骤）；
{"action":"finish","answer":"给用户的最终答复"} —— 任务已完成或无法继续。
所有步骤都执行完后必须 action=finish。"""


def _plan_rules(max_steps: int) -> str:
    return f"- 最多 {max_steps} 步。"


def build_plan_messages(
    context: list[Message],
    registry: ToolRegistry,
    max_steps: int,
    user_input: str,
) -> list[Message]:
    instruction = (
        f"{PLAN_HEADER}\n{_plan_rules(max_steps)}\n\n任务：{user_input}\n\n"
        f"可用工具：\n{_tool_catalog(registry)}"
    )
    return [*context, Message(role="user", content=instruction)]


def build_repair_messages(
    messages: list[Message],
    draft: str,
    errors: tuple[str, ...],
    max_steps: int,
) -> list[Message]:
    detail = "\n".join(f"- {error}" for error in errors)
    return [
        *messages,
        Message(role="assistant", content=draft),
        Message(
            role="user",
            content=(
                "上面的计划未通过本地校验，问题如下：\n"
                f"{detail}\n"
                f"请修正后重新只输出 JSON 计划（最多 {max_steps} 步），不要输出其他内容。"
            ),
        ),
    ]


def _format_step(step: PlanStep) -> str:
    depends = f" depends_on={list(step.depends_on)}" if step.depends_on else ""
    return f"步骤 {step.index}：{step.intent}（工具 {step.tool}{depends}）"


def build_review_messages(
    registry: ToolRegistry,
    plan: Plan,
    remaining: list[PlanStep],
    done: list[StepRun],
    user_input: str,
    max_steps: int,
    max_replans: int,
    replans: int,
) -> list[Message]:
    executed = "\n".join(
        f"- 步骤 {run.step.index} {run.step.intent}："
        f"{'成功' if run.ok else '失败'}，输出：{run.output or '(空)'}"
        for run in done
    )
    todo = "\n".join(f"- {_format_step(step)}" for step in remaining) or "（无）"
    payload = (
        f"{REVIEW_HEADER}\n{_plan_rules(max_steps)}\n"
        f"- 本轮最多允许重规划 {max_replans} 次，已用 {replans} 次。\n\n"
        f"原始任务：{user_input}\n\n"
        f"原计划：\n" + "\n".join(f"- {_format_step(step)}" for step in plan.steps) + "\n\n"
        f"已执行：\n{executed or '（无）'}\n\n"
        f"剩余步骤：\n{todo}\n\n"
        f"可用工具：\n{_tool_catalog(registry)}"
    )
    return [Message(role="system", content=payload)]


def summarize_run(done: list[StepRun], user_input: str) -> str:
    """兜底总结：不额外调模型，直接把已完成步骤拼成答复。"""
    lines = [f"已完成 {len(done)} 个步骤（任务：{user_input}）："]
    for run in done:
        status = "成功" if run.ok else "失败"
        preview = " ".join((run.output or "").split())
        if len(preview) > 200:
            preview = preview[:200] + "..."
        lines.append(f"{run.step.index}. {run.step.intent}（{run.step.tool}，{status}）：{preview}")
    return "\n".join(lines)


# ---- 收尾校验 -------------------------------------------------------------


def observations(done: list[StepRun]) -> str:
    """把工具的实际输出拼成证据池；校验结论里的证据必须能在这里找到。"""
    blocks = []
    for run in done:
        args = json.dumps(run.step.args, ensure_ascii=False) if run.step.args else "{}"
        blocks.append(
            f"[步骤 {run.step.index}] {run.step.tool}({args})\n"
            f"状态：{'成功' if run.ok else '失败'}\n"
            f"输出：{run.output or '(空)'}"
        )
    return "\n\n".join(blocks)


def evidence_pool(done: list[StepRun]) -> str:
    """只把工具的实际返回当作可引用证据：计划文本与参数不算"观察到的结果"。"""
    blocks = []
    for run in done:
        blocks.append(
            f"状态：{'成功' if run.ok else '失败'}\n输出：{run.output or '(空)'}"
        )
    return "\n\n".join(blocks)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def parse_verify(text: str, evidence_text: str) -> VerifyCheck:
    """判定必须给出可核对证据；证据缺失或对不上实际输出，一律算不通过。"""
    if not (text or "").strip():
        return VerifyCheck("fail", "校验没有返回内容", code="empty_text")
    data = _extract_json(text)
    if not isinstance(data, dict):
        return VerifyCheck("fail", "校验结果不是合法 JSON", code="parse_error")
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        return VerifyCheck("fail", "verdict 必须是 pass 或 fail", code="bad_verdict")
    reason = str(data.get("reason") or "").strip()
    if not reason:
        return VerifyCheck("fail", "校验没有给出判定依据", code="no_reason")
    raw_evidence = data.get("evidence") or []
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    evidence = tuple(str(item).strip() for item in raw_evidence if str(item).strip())
    if verdict == "fail":
        return VerifyCheck("fail", reason, evidence)

    if not evidence:
        return VerifyCheck("fail", "判定通过但没有给出证据", code="evidence_missing")
    haystack = _normalize(evidence_text)
    for item in evidence:
        if _normalize(item) not in haystack:
            return VerifyCheck(
                "fail",
                f"证据无法与实际工具输出对应：{item[:120]}",
                evidence,
                code="evidence_ungrounded",
            )
    return VerifyCheck("pass", reason, evidence)


VERIFY_HEADER = """你是结果校验员。你的任务是判断"原始用户意图"是否已经变成"实际可交付的最终状态"，
而不是复述执行过程。只输出一个 JSON 对象，不要解释、不要 Markdown：
{"verdict":"pass|fail","reason":"一句话判定依据","evidence":["必须逐字引用观察记录里的片段"]}
规则：
- verdict=pass 时 evidence 至少一条，且必须能在观察记录里逐字找到；找不到就判 fail。
- 观察记录为空、或不足以支撑结论时判 fail。
- 只依据观察记录判断，不得推测没被观察到的结果。
- fail 时 reason 要写明差在哪一步、缺什么。"""


def build_verify_messages(
    user_input: str,
    done: list[StepRun],
    round_index: int,
) -> list[Message]:
    payload = (
        f"{VERIFY_HEADER}\n\n"
        f"原始用户意图：{user_input}\n\n"
        f"观察记录（工具实际返回，第 {round_index} 次校验）：\n"
        f"{observations(done) or '（空）'}"
    )
    return [Message(role="system", content=payload)]


def build_supplement_messages(
    registry: ToolRegistry,
    user_input: str,
    done: list[StepRun],
    check: VerifyCheck,
) -> list[Message]:
    payload = (
        "收尾校验未通过，请只针对未达成项补充执行。可以用工具，也可以直接结束。\n"
        f"原始用户意图：{user_input}\n"
        f"未达成原因：{check.reason}\n\n"
        f"已有的观察记录：\n{observations(done) or '（空）'}\n\n"
        f"可用工具：\n{_tool_catalog(registry)}"
    )
    return [Message(role="system", content=payload)]


def format_verification_failure(done: list[StepRun], check: VerifyCheck) -> str:
    """校验没过时的最终答复：只陈述做了什么、差在哪，不包装成成功。"""
    lines = ["收尾校验未通过，任务没有达到可交付状态。", f"未达成：{check.reason}"]
    if done:
        lines.append("已执行：")
        for run in done:
            status = "成功" if run.ok else "失败"
            preview = " ".join((run.output or "").split())
            if len(preview) > 160:
                preview = preview[:160] + "..."
            lines.append(
                f"{run.step.index}. {run.step.intent}"
                f"（{run.step.tool}，{status}）：{preview}"
            )
    return "\n".join(lines)
