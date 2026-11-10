"""Plan-and-Execute：结构化计划、本地校验、逐步执行、重规划与退回 ReAct。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniagent.agent import Agent
from miniagent.config import load_config
from miniagent.errors import ConfigError, ProviderError
from miniagent.llm import LLMResponse, MockProvider, ToolCall
from miniagent.logging import TraceLogger
from miniagent.planner import (
    DEFAULT_VERIFY_ROUNDS,
    parse_plan,
    parse_review,
    validate_steps,
)
from miniagent.tools import ToolRegistry


class EchoTool:
    """最小工具：记录调用顺序，参数只要求一个 path。"""

    def __init__(self, name: str, output: str | None = None) -> None:
        self.name = name
        self.description = f"工具 {name}"
        self.read_only = True
        self.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        self.output = output or f"{name} ok"
        self.calls: list[dict] = []

    def run(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self.output


def make_registry(*names: str) -> tuple[ToolRegistry, dict[str, EchoTool]]:
    registry = ToolRegistry()
    tools: dict[str, EchoTool] = {}
    for name in names or ("read_file", "write_file"):
        tool = EchoTool(name)
        tools[name] = tool
        registry.register(tool)
    return registry, tools


def plan_json(*steps: dict) -> str:
    return json.dumps({"steps": list(steps)}, ensure_ascii=False)


def step(tool: str, path: str = "a.txt", intent: str = "看一下", depends: list[int] | None = None):
    data: dict = {"intent": intent, "tool": tool, "args": {"path": path}}
    if depends:
        data["depends_on"] = depends
    return data


def review_json(action: str, **extra) -> str:
    return json.dumps({"action": action, **extra}, ensure_ascii=False)


def verify_json(verdict: str, reason: str, evidence: list[str] | None = None) -> str:
    data: dict = {"verdict": verdict, "reason": reason}
    if evidence is not None:
        data["evidence"] = evidence
    return json.dumps(data, ensure_ascii=False)


class VerifyFailProvider(MockProvider):
    """在第 fail_at 次调用上抛 ProviderError，用来模拟校验调用失败。"""

    def __init__(self, script: list[LLMResponse], fail_at: int) -> None:
        super().__init__(script)
        self.fail_at = fail_at

    def chat(self, messages, tools, *, on_text=None):
        if len(self.calls) + 1 == self.fail_at:
            self.calls.append(list(messages))
            raise ProviderError("模型接口不可用")
        return super().chat(messages, tools, on_text=on_text)


def build_agent(
    script: list[LLMResponse],
    *,
    tracer: TraceLogger | None = None,
    plan_mode: str = "on",
    max_steps: int = 8,
    max_replans: int = 2,
    plan_verify: str = "off",
    plan_verify_rounds: int = DEFAULT_VERIFY_ROUNDS,
    tools: tuple[str, ...] = ("read_file", "write_file"),
    provider: MockProvider | None = None,
) -> tuple[Agent, MockProvider, dict[str, EchoTool]]:
    registry, table = make_registry(*tools)
    mock = provider if provider is not None else MockProvider(list(script))
    agent = Agent(
        provider=mock,
        registry=registry,
        tool_deps="off",
        plan_mode=plan_mode,
        plan_max_steps=max_steps,
        plan_max_replans=max_replans,
        plan_verify=plan_verify,
        plan_verify_rounds=plan_verify_rounds,
        tracer=tracer,
    )
    return agent, mock, table


def plan_events(tracer: TraceLogger, log_dir: Path, event: str) -> list:
    records = TraceLogger.load(tracer.session_id, log_dir)
    return [record for record in records if record.event == event]


# ---- 校验 -----------------------------------------------------------------


def test_validate_accepts_valid_plan() -> None:
    registry, _ = make_registry()
    check = parse_plan(
        plan_json(step("read_file", "a.txt"), step("write_file", "b.txt", depends=[1])),
        registry,
        8,
    )
    assert check.ok
    assert check.plan is not None
    assert [item.tool for item in check.plan.steps] == ["read_file", "write_file"]
    assert check.plan.steps[1].depends_on == (1,)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("", "empty_text"),
        ("我先看看文件", "parse_error"),
        ('{"plan": []}', "bad_shape"),
        (plan_json(), "empty_plan"),
        (plan_json(step("nope")), "unknown_tool"),
        (plan_json({"intent": "缺参数", "tool": "read_file", "args": {}}), "bad_args"),
        (plan_json(step("read_file", depends=[3])), "bad_dep"),
        (plan_json(step("read_file", depends=[1])), "bad_dep"),
        (plan_json({"intent": "", "tool": "read_file", "args": {"path": "a"}}), "bad_step"),
        (plan_json(*[step("read_file") for _ in range(3)]), "too_many_steps"),
    ],
)
def test_validate_rejects_bad_plans(content: str, code: str) -> None:
    registry, _ = make_registry()
    check = parse_plan(content, registry, max_steps=2 if code == "too_many_steps" else 8)
    assert not check.ok
    assert check.code == code
    assert check.errors


def test_validate_is_all_or_nothing() -> None:
    """一步不合格就整体不执行，避免半份计划跑到一半才发现。"""
    registry, _ = make_registry()
    steps, code, errors = validate_steps(
        [step("read_file"), step("ghost")], registry, 8
    )
    assert steps is None
    assert code == "unknown_tool"
    assert any("第 2 步" in error for error in errors)


def test_review_parsing() -> None:
    registry, _ = make_registry()
    assert parse_review(review_json("continue"), registry, 8).action == "continue"
    finish = parse_review(review_json("finish", answer="做完了"), registry, 8)
    assert finish.action == "finish" and finish.answer == "做完了"
    replan = parse_review(
        review_json("replan", steps=[step("write_file")]), registry, 8
    )
    assert replan.action == "replan" and len(replan.steps) == 1

    assert parse_review("", registry, 8).code == "empty_text"
    assert parse_review("继续", registry, 8).code == "parse_error"
    assert parse_review(review_json("skip"), registry, 8).code == "bad_action"
    assert parse_review(review_json("finish"), registry, 8).code == "bad_review"
    bad_replan = parse_review(review_json("replan", steps=[step("ghost")]), registry, 8)
    assert bad_replan.action == "" and bad_replan.code == "unknown_tool"


# ---- 开关 -----------------------------------------------------------------


def test_plan_off_keeps_react_without_extra_calls(tmp_path: Path) -> None:
    tracer = TraceLogger("off", tmp_path)
    agent, provider, _ = build_agent(
        [LLMResponse(content="直接回答")], tracer=tracer, plan_mode="off"
    )
    result = agent.run("你好")

    assert result.ok
    assert result.text == "直接回答"
    assert len(provider.calls) == 1  # 关闭时零额外调用
    assert plan_events(tracer, tmp_path, "plan") == []


def test_plan_off_with_tools_keeps_call_count(tmp_path: Path) -> None:
    tracer = TraceLogger("off-tools", tmp_path)
    tool_round = LLMResponse(
        tool_calls=[
            ToolCall(id="c1", name="read_file", arguments='{"path": "a.txt"}')
        ]
    )
    agent, provider, table = build_agent(
        [tool_round, LLMResponse(content="读完了")], tracer=tracer, plan_mode="off"
    )
    result = agent.run("读 a.txt")

    assert result.ok
    assert len(provider.calls) == 2
    assert table["read_file"].calls == [{"path": "a.txt"}]


# ---- 生成与重试 -----------------------------------------------------------


def test_plan_retry_after_invalid_then_executes(tmp_path: Path) -> None:
    tracer = TraceLogger("retry", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(content="我觉得应该先读文件"),  # 非 JSON
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="完成")),
        ],
        tracer=tracer,
    )
    result = agent.run("读 a.txt")

    assert result.ok
    assert result.text == "完成"
    assert table["read_file"].calls == [{"path": "a.txt"}]
    assert len(provider.calls) == 3

    validates = [
        record
        for record in plan_events(tracer, tmp_path, "plan")
        if "phase=validate" in record.detail
    ]
    assert [record.ok for record in validates] == [False, True]
    assert "source=retry" in validates[-1].detail
    assert validates[0].error_code == "parse_error"


def test_plan_invalid_twice_falls_back_to_react(tmp_path: Path) -> None:
    tracer = TraceLogger("fallback", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(content="不是 JSON"),
            LLMResponse(content=plan_json(step("ghost"))),
            LLMResponse(content="ReAct 兜底回答"),
        ],
        tracer=tracer,
    )
    result = agent.run("看一下")

    assert result.ok
    assert result.text == "ReAct 兜底回答"
    assert len(provider.calls) == 3  # 两次计划 + 一次 ReAct
    assert table["read_file"].calls == []
    fallback = plan_events(tracer, tmp_path, "plan_fallback")
    assert len(fallback) == 1
    assert fallback[0].error_code == "unknown_tool"
    assert plan_events(tracer, tmp_path, "plan_step") == []


# ---- 执行 -----------------------------------------------------------------


def test_plan_executes_steps_and_feeds_results_back(tmp_path: Path) -> None:
    tracer = TraceLogger("full", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(
                content=plan_json(
                    step("read_file", "a.txt", intent="读输入"),
                    step("write_file", "b.txt", intent="写结果", depends=[1]),
                )
            ),
            LLMResponse(content=review_json("continue")),
            LLMResponse(content=review_json("finish", answer="两步都做完了")),
        ],
        tracer=tracer,
    )
    result = agent.run("把 a 处理后写到 b")

    assert result.ok
    assert result.text == "两步都做完了"
    assert [call["path"] for call in table["read_file"].calls] == ["a.txt"]
    assert [call["path"] for call in table["write_file"].calls] == ["b.txt"]
    assert len(provider.calls) == 3  # 1 次计划 + 每步 1 次评估

    # 执行顺序记录在会话里：system、user，然后每步 assistant(tool_calls) -> tool，最后 assistant
    roles = [message.role for message in agent.messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]

    steps = plan_events(tracer, tmp_path, "plan_step")
    assert [record.tool for record in steps] == ["read_file", "write_file"]
    assert all(record.ok for record in steps)
    reviews = plan_events(tracer, tmp_path, "plan_review")
    assert [record.ok for record in reviews] == [True, True]


def test_plan_can_finish_early(tmp_path: Path) -> None:
    agent, provider, table = build_agent(
        [
            LLMResponse(
                content=plan_json(step("read_file", "a.txt"), step("write_file", "b.txt"))
            ),
            LLMResponse(content=review_json("finish", answer="读一下就够了")),
        ]
    )
    result = agent.run("看看就好")

    assert result.ok
    assert result.text == "读一下就够了"
    assert table["read_file"].calls == [{"path": "a.txt"}]
    assert table["write_file"].calls == []  # 提前结束，第二步没跑


def test_plan_replans_remaining_steps(tmp_path: Path) -> None:
    tracer = TraceLogger("replan", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(
                content=plan_json(
                    step("read_file", "a.txt"),
                    step("write_file", "b.txt"),
                    step("write_file", "c.txt"),
                )
            ),
            LLMResponse(
                content=review_json(
                    "replan", steps=[step("write_file", "d.txt", intent="改写到 d")]
                )
            ),
            LLMResponse(content=review_json("finish", answer="按新计划完成")),
        ],
        tracer=tracer,
    )
    result = agent.run("先读再写")

    assert result.ok
    assert [call["path"] for call in table["read_file"].calls] == ["a.txt"]
    assert [call["path"] for call in table["write_file"].calls] == ["d.txt"]
    reviews = plan_events(tracer, tmp_path, "plan_review")
    assert any("action=replan" in record.detail for record in reviews)
    assert any("replans=1" in record.detail for record in reviews)


def test_replan_limit_falls_back_to_original_steps(tmp_path: Path) -> None:
    tracer = TraceLogger("replan-limit", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file", "a.txt"), step("write_file", "b.txt"))),
            LLMResponse(
                content=review_json(
                    "replan", steps=[step("write_file", "d.txt")]
                )
            ),
            LLMResponse(content=review_json("finish", answer="按原计划完成")),
        ],
        tracer=tracer,
        max_replans=0,
    )
    result = agent.run("先读再写")

    assert result.ok
    assert [call["path"] for call in table["write_file"].calls] == ["b.txt"]
    reviews = plan_events(tracer, tmp_path, "plan_review")
    assert reviews[0].error_code == "replan_limit"
    assert reviews[0].ok is False


def test_review_failure_continues_with_original_plan(tmp_path: Path) -> None:
    tracer = TraceLogger("review-bad", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file", "a.txt"), step("write_file", "b.txt"))),
            LLMResponse(content="我决定继续"),  # 不是 JSON
            LLMResponse(content=review_json("finish", answer="都做完了")),
        ],
        tracer=tracer,
    )
    result = agent.run("先读再写")

    assert result.ok
    assert table["write_file"].calls == [{"path": "b.txt"}]
    reviews = plan_events(tracer, tmp_path, "plan_review")
    assert reviews[0].ok is False and reviews[0].error_code == "parse_error"


def test_plan_all_steps_failed_returns_failure(tmp_path: Path) -> None:
    class BrokenTool(EchoTool):
        def run(self, args: dict) -> str:
            raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(BrokenTool("broken"))
    provider = MockProvider(
        [
            LLMResponse(content=plan_json(step("broken", "a.txt"))),
            LLMResponse(content=review_json("continue")),
        ]
    )
    agent = Agent(
        provider=provider,
        registry=registry,
        tool_deps="off",
        plan_mode="on",
    )
    result = agent.run("试一下")

    assert result.ok is False  # 退出码语义仍按失败处理
    assert "失败" in result.text


# ---- 收尾校验 -------------------------------------------------------------


def test_verify_off_makes_no_extra_calls(tmp_path: Path) -> None:
    tracer = TraceLogger("verify-off", tmp_path)
    agent, provider, _ = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="读完了")),
        ],
        tracer=tracer,
        plan_verify="off",
    )
    result = agent.run("读 a.txt")

    assert result.ok and result.text == "读完了"
    assert len(provider.calls) == 2  # 计划 + 一次评估，没有额外校验调用
    assert plan_events(tracer, tmp_path, "plan_verify") == []
    run_end = plan_events(tracer, tmp_path, "run_end")[-1]
    assert "verify=skipped" in run_end.detail


def test_verify_pass_with_grounded_evidence(tmp_path: Path) -> None:
    tracer = TraceLogger("verify-pass", tmp_path)
    agent, provider, _ = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="读完了")),
            LLMResponse(content=verify_json("pass", "已读到文件内容", ["read_file ok"])),
        ],
        tracer=tracer,
        plan_verify="on",
    )
    result = agent.run("读 a.txt")

    assert result.ok and result.text == "读完了"
    assert len(provider.calls) == 3
    checks = plan_events(tracer, tmp_path, "plan_verify")
    assert any(
        "verdict=pass" in record.detail and "evidence=1" in record.detail
        for record in checks
    )
    run_end = plan_events(tracer, tmp_path, "run_end")[-1]
    assert "verify=pass" in run_end.detail
    assert "supplements=0" in run_end.detail


def test_verify_fail_then_supplement_passes(tmp_path: Path) -> None:
    tracer = TraceLogger("verify-supplement", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="读完了")),
            LLMResponse(content=verify_json("fail", "还没有写入 b.txt")),
            LLMResponse(
                tool_calls=[
                    ToolCall(id="s1", name="write_file", arguments='{"path": "b.txt"}')
                ]
            ),
            LLMResponse(content=verify_json("pass", "已补写 b.txt", ["write_file ok"])),
        ],
        tracer=tracer,
        plan_verify="on",
    )
    result = agent.run("把 a 处理成 b")

    assert result.ok and result.text == "读完了"
    assert table["write_file"].calls == [{"path": "b.txt"}]
    assert len(provider.calls) == 5
    checks = plan_events(tracer, tmp_path, "plan_verify")
    supplements = [record for record in checks if "phase=supplement" in record.detail]
    assert len(supplements) == 1 and "tools=1" in supplements[0].detail
    run_end = plan_events(tracer, tmp_path, "run_end")[-1]
    assert "verify=pass" in run_end.detail
    assert "supplements=1" in run_end.detail


def test_verify_fail_after_supplement_reports_honestly(tmp_path: Path) -> None:
    tracer = TraceLogger("verify-still-fail", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="读完了")),
            LLMResponse(content=verify_json("fail", "还没有写入 b.txt")),
            LLMResponse(
                tool_calls=[
                    ToolCall(id="s1", name="write_file", arguments='{"path": "b.txt"}')
                ]
            ),
            LLMResponse(content=verify_json("fail", "b.txt 仍然缺少目标内容")),
        ],
        tracer=tracer,
        plan_verify="on",
    )
    result = agent.run("把 a 处理成 b")

    assert result.ok is False  # 不包装成成功
    assert "收尾校验未通过" in result.text
    assert "未达成：b.txt 仍然缺少目标内容" in result.text
    assert "已执行：" in result.text
    assert table["write_file"].calls == [{"path": "b.txt"}]
    assert len(provider.calls) == 5  # 校验、补充、复校各一次，不无限修补
    run_end = plan_events(tracer, tmp_path, "run_end")[-1]
    assert run_end.error_code == "verify_failed"
    assert "verify=fail" in run_end.detail
    assert "supplements=1" in run_end.detail


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (verify_json("pass", "看起来完成了"), "evidence_missing"),
        (verify_json("pass", "完成了", ["这句是我编的"]), "evidence_ungrounded"),
    ],
)
def test_verify_pass_without_grounded_evidence_fails(
    tmp_path: Path, content: str, code: str
) -> None:
    tracer = TraceLogger(f"verify-{code}", tmp_path)
    agent, provider, _ = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="读完了")),
            LLMResponse(content=content),
        ],
        tracer=tracer,
        plan_verify="on",
        plan_verify_rounds=0,
    )
    result = agent.run("读 a.txt")

    assert result.ok is False
    assert "收尾校验未通过" in result.text
    checks = plan_events(tracer, tmp_path, "plan_verify")
    assert checks[-1].error_code == code
    assert "verdict=fail" in checks[-1].detail
    assert len(provider.calls) == 3  # 证据不成立时不触发补充执行
    run_end = plan_events(tracer, tmp_path, "run_end")[-1]
    assert run_end.error_code == "verify_failed"


def test_verify_call_error_is_not_silent_pass(tmp_path: Path) -> None:
    tracer = TraceLogger("verify-error", tmp_path)
    provider = VerifyFailProvider(
        [
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="读完了")),
        ],
        fail_at=3,
    )
    agent, _, _ = build_agent(
        [], tracer=tracer, plan_verify="on", provider=provider
    )
    result = agent.run("读 a.txt")

    assert result.ok is False  # 校验失败按未通过处理，绝不静默视为成功
    assert "收尾校验调用失败" in result.text
    assert len(provider.calls) == 3  # 无法校验时不再补充执行
    checks = plan_events(tracer, tmp_path, "plan_verify")
    assert checks[-1].error_code == "verify_unavailable"
    assert "verdict=unverified" in checks[-1].detail
    run_end = plan_events(tracer, tmp_path, "run_end")[-1]
    assert run_end.error_code == "verify_unavailable"


def test_verify_supplement_round_limit_zero_skips_supplement(tmp_path: Path) -> None:
    tracer = TraceLogger("verify-rounds-0", tmp_path)
    agent, provider, table = build_agent(
        [
            LLMResponse(content=plan_json(step("read_file"))),
            LLMResponse(content=review_json("finish", answer="读完了")),
            LLMResponse(content=verify_json("fail", "还没有写入 b.txt")),
        ],
        tracer=tracer,
        plan_verify="on",
        plan_verify_rounds=0,
    )
    result = agent.run("把 a 处理成 b")

    assert result.ok is False
    assert table["write_file"].calls == []
    assert len(provider.calls) == 3  # 不通过即停，不做补充
    checks = plan_events(tracer, tmp_path, "plan_verify")
    assert all("phase=supplement" not in record.detail for record in checks)
    run_end = plan_events(tracer, tmp_path, "run_end")[-1]
    assert "supplements=0" in run_end.detail


# ---- 配置 -----------------------------------------------------------------


def test_config_plan_mode_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIAGENT_PLAN_MODE", "on")
    monkeypatch.chdir(tmp_path)
    assert load_config(tmp_path / "config.toml").plan_mode == "on"

    monkeypatch.setenv("MINIAGENT_PLAN_MODE", "maybe")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "config.toml")


def test_config_plan_verify_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    assert load_config(config_path).plan_verify == "off"  # 默认关闭，零额外调用

    monkeypatch.setenv("MINIAGENT_PLAN_VERIFY", "on")
    monkeypatch.setenv("MINIAGENT_PLAN_VERIFY_ROUNDS", "2")
    config = load_config(config_path)
    assert config.plan_verify == "on"
    assert config.plan_verify_rounds == 2

    monkeypatch.setenv("MINIAGENT_PLAN_VERIFY", "maybe")
    with pytest.raises(ConfigError):
        load_config(config_path)

    monkeypatch.setenv("MINIAGENT_PLAN_VERIFY", "on")
    monkeypatch.setenv("MINIAGENT_PLAN_VERIFY_ROUNDS", "-1")
    with pytest.raises(ConfigError):
        load_config(config_path)
