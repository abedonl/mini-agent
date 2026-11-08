# 小型 Agent 框架二次开发选型调研

> 调研日期：2026-11-08  
> 目标场景：后端开发 / Agent 开发方向实习作品集，希望在现成的小项目上深入改造，而不是从零实现  
> 结论口径：先看“内核是否读得完、接缝是否清楚”，再看“改造后能否量化、能否讲出工程判断”

---

## 1. 结论摘要

**首选：`SWE-agent/mini-swe-agent`。**

它的 Agent 内核只有一个约 190 行的类，`Model` 和 `Environment` 都用 `typing.Protocol` 定义，MIT 许可证，维护活跃，且有 SWE-bench 成绩作为对照基准。它留给二次开发的空间恰好是后端能力：工具抽象、会话恢复、并发策略、成本预算、可观测性、评测回归、服务化。

**备选：`huggingface/smolagents`。**

如果目标不是“把一个小内核做成运行时”，而是“扩展一个通用 Agent 框架”，smolagents 的扩展点更齐全：`Tool` / `Model` / `AgentMemory` / 沙箱执行器 / MCP / Hub 都已现成。代价是包内约 1.28 万行源码、1,050 个未关闭 Issue，改造更容易变成“给上游打补丁”，而不是“拥有一个系统”。

**特定场景可选：`Eigenwise/atomic-agents`。**

适合做“结构化输出 + MCP + 可观测性”的后端 Agent 流水线。它的 Pydantic Schema 和 OTel 埋点很干净，但 Agent 运行时很薄，能改造的循环/并发/恢复空间小于前两者。

**不推荐作为二次开发基座：**

- `openai/openai-agents-python`：官方定位 lightweight，但 0.24.0 的 `src/agents` 已有 317 个文件、约 13.45 万行；厂商主导、迭代快，适合当依赖，不适合当作品集基座。
- `openai/swarm`：512 行的教学原型，README 已明确由 Agents SDK 取代，2026-04-15 后无提交。只适合读，不适合改。

一句话选择：

- 想讲“我读透了一个 Agent 内核，并把它补成生产可用的运行时” → **mini-swe-agent**。
- 想讲“我扩展了一个通用 Agent 框架的工具/模型/沙箱生态” → **smolagents**。

---

## 2. 调研口径与“小型”的判定依据

### 2.1 数据来源

- 仓库元数据（Star、Fork、未关闭 Issue、最近提交时间、许可证）：GitHub REST API，查询日期 2026-11-08。[S14]
- 版本与发布日期：各仓库 Releases 页面。[S2][S6][S8][S11]
- 代码量：2026-11-08 浅克隆默认分支，统计 `src` 下发布包内的 `.py` 文件；**不含仓库级 `tests/`、`docs/`、`examples/`**。总行数含空行与注释，非空行用于辅助判断。
- 下载量：pypistats 最近 30 天数据；该数据包含镜像与 CI 流量，只作趋势参考，不作质量结论。[S15]

### 2.2 “小型”分三层看

| 层级 | 判断标准 | 为什么重要 |
| --- | --- | --- |
| 内核 | 主循环/Agent 类是否在数百行内 | 决定你能否真正读透并独立改造 |
| 发布包 | `src` 包内总行数 | 决定你需要理解、测试、维护的边界 |
| 仓库 | 测试、文档、基准、示例的总量 | 决定上手成本和“对照基准”是否可用 |

只看 README 的“100 行”“1,000 行”容易被营销口径误导：它们通常只指主循环文件，不等于整个包。下面的数字统一按发布包口径统计。

---

## 3. 候选与规模对比

| 候选 | 定位 | 发布包规模（文件 / 总行 / 非空行） | 核心文件规模 | 许可证 | 最近提交 / 最近正式版 | Star / Fork / 未关闭 Issue | PyPI 月下载（参考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | HF 的通用 code-agent 库 | 18 / 12,774 / 10,998 | `agents.py` 1,813；`models.py` 2,102；`local_python_executor.py` 1,768；`tools.py` 1,422 | Apache-2.0 | 2026-11-03 / v1.26.0（2026-05-29） | 30,266 / 3,201 / 1,050 | 425,605 |
| [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | Princeton/Stanford 的极简 coding agent | 59 / 5,349 / 4,531；剔除基准与包内测试后 51 / 4,316 / 3,660 | `agents/default.py` 190；`models/litellm_model.py` 164；`environments/local.py` 92；`run/hello_world.py` 42（核心 4 文件约 488 行） | MIT | 2026-11-02 / v2.4.6（2026-07-23） | 9,372 / 1,296 / 139 | 11,605,679 |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | OpenAI 官方多 Agent SDK | 317 / 134,534 / 117,775 | `agent.py` 1,047；`run.py` 2,785；`tool.py` 2,444；`run_state.py` 5,130 | MIT | 2026-11-07 / v0.24.0（2026-10-15） | 30,547 / 5,005 / 43 | 12,449,784 |
| [Eigenwise/atomic-agents](https://github.com/Eigenwise/atomic-agents) | Instructor/Pydantic 驱动的模块化 Agent 库 | 30 / 5,944 / 4,869 | `agents/atomic_agent.py` 918；`connectors/mcp/mcp_factory.py` 982；`context/chat_history.py` 387 | MIT | 2026-10-29 / v2.15.0（2026-10-21） | 6,320 / 562 / 29 | 11,711 |
| [openai/swarm](https://github.com/openai/swarm) | OpenAI 教学用多 Agent 原型 | 6 / 512 / 432 | `core.py` 258 | MIT | 2026-04-15 / 无正式 Release | 22,108 / 2,350 / 38 | 不适用（已停止演进） |

数据来源：[S1][S2][S3][S4][S5][S6][S7][S8][S10][S11][S13][S14][S15]。

几个容易被忽略的事实：

1. **smolagents 的“约 1,000 行”只指主循环逻辑。** README 写的是 “the logic for agents fits in ~1,000 lines of code (see agents.py)”[S3]，但 `agents.py` 实测 1,813 行，整个发布包 12,774 行。它仍然算“小型框架”，但不是“千行框架”。
2. **mini-swe-agent 的“100 行”基本属实，但只是 Agent 类。** README 原文是 “Just some 100 lines of python for the agent class”[S5]；`DefaultAgent` 文件 190 行，核心四文件约 488 行。整个运行时包 4,316 行，主要增量在模型适配、环境适配、CLI 与基准脚本。
3. **openai-agents-python 不属于小型项目。** 13.45 万行、317 个文件，是本次候选中体量最大的；它的“lightweight”是相对传统编排框架而言。把它当基座，改造工作量会集中在读懂 5,130 行的 `run_state.py` 和 2,785 行的 `run.py` 上。
4. **Star 数不等于改造价值。** swarm 有 2.2 万 Star，但已停止演进；mini-swe-agent 只有 9,372 Star，却有约 1,160 万月下载量，说明它被大量评测/CI 流水线当作基线使用。

---

## 4. 评估维度对比

| 维度 | smolagents | mini-swe-agent | openai-agents-python | atomic-agents | swarm |
| --- | --- | --- | --- | --- | --- |
| 代码可读性 | 中：主循环可读，但模型层与执行器各 1.7–2.1k 行 | **高**：核心 4 文件约 490 行 | 低—中：317 文件，需靠文档导航 | 中：抽象层多，但文件职责清楚 | 极高：`core.py` 258 行 |
| 核心抽象清晰度 | 高：`MultiStepAgent` → `CodeAgent`/`ToolCallingAgent`；`Model`、`BaseTool`、`AgentMemory` 分层明确 | **高**：`DefaultAgent` + `Model` Protocol + `Environment` Protocol，三件事各就各位 | 高但层次多：`Agent`/`Runner`/`RunConfig`/`Session`/`ModelProvider` | 中高：`AtomicAgent` + `BaseTool[In,Out]` + `BaseIOSchema` | 高：`Agent` + functions + handoff，极简 |
| 自定义工具 | **很强**：`Tool`、`@tool`、`ToolCollection`、MCP、LangChain、Hub | 弱：原生只有 bash，**需要自己补 Tool 层** | 很强：`function_tool`、MCP、托管工具 | 强：`BaseTool` 强制 Pydantic 输入/输出 Schema | 弱：普通函数 + handoff |
| 模型 Provider | **很强**：`Model` 子类覆盖 HF/vLLM/MLX/Transformers/OpenAI/Azure/Bedrock/LiteLLM | 强：`Model` Protocol + LiteLLM/OpenRouter/Portkey 适配 | 中：OpenAI 优先；兼容端点可用，真正多 Provider 要自己适配 | 强：Instructor + LiteLLM | 弱：仅 OpenAI |
| 记忆/状态 | 中：`AgentMemory` 在内存中，无内建持久化 | 中：线性消息历史 + `serialize()`/`save()`，可恢复但无会话库 | **强**：`Session`/`SQLiteSession`/压缩会话 | 中：`ChatHistory`，偏上下文拼装 | 无 |
| 依赖轻重 | **轻**：6 个必需依赖；OpenAI/LiteLLM/沙箱均为 extras | 重：14 个必需依赖，含 `litellm`、`datasets`、`textual` | 中：13 个，含 `mcp`、`starlette`、`websockets` | 重：12 个，含 `instructor`、`litellm`、`mcp`、`textual` | 极轻：仅 `openai` |
| 维护状态 | 活跃但发布节奏慢：提交至 2026-11，正式版停在 2026-05；1,050 个未关闭 Issue | 活跃：提交至 2026-11；v2 仍标 Alpha | **最活跃**：提交至 2026-11-07，版本按月推进 | 活跃：提交与发布均至 2026-10 | 已停止：最后提交 2026-04 |
| 二次开发叙事分量 | 高：可讲“沙箱安全、Provider 路由、工具治理、记忆后端” | **很高**：可讲“从 190 行内核到可恢复、可观测、可并发的运行时” | 低：体量大、厂商主导，容易变成“提 PR”而非“做系统” | 中：可讲“类型化工具契约、MCP、OTel”，但循环改造空间小 | 低：已废弃，改完没有对照意义 |
| 主要风险 | 上游迭代快、Issue 存量大、代码执行安全边界复杂 | 原生无工具抽象、依赖偏重、v2 仍 Alpha | 体量与厂商绑定；升级会持续侵入你的改动 | Python ≥3.12；依赖重；Agent 偏单次调用 | 已停止演进 |

---

## 5. 逐项判断

### 5.1 mini-swe-agent：最适合“读透 + 重造”

**规模证据。** `src/minisweagent/agents/default.py` 190 行，`models/litellm_model.py` 164 行，`environments/local.py` 92 行，`run/hello_world.py` 42 行；核心四文件约 488 行。发布包 51 个运行时文件、4,316 行，其余主要是 CLI、基准脚本和模型适配。[S4][S5]

**核心抽象。** `DefaultAgent` 只做四件事：渲染模板、追加消息、调用模型、执行动作。`Model` 与 `Environment` 都是 `Protocol`，所以替换模型、替换执行环境（本地/Docker/Singularity/bubblewrap）不需要改 Agent。历史是线性的，每一步只是追加消息；轨迹即消息，便于调试与后续微调。这是它最大的设计优点。

**扩展点。**

- 工具：原生只有 bash，没有 Tool 抽象。对“通用 Agent 框架”是缺点，对“二次开发作品集”反而是最大的改造入口。
- 模型：`Model` Protocol + LiteLLM/OpenRouter/Portkey 适配，新增 Provider 的成本低。
- 状态：`serialize()`/`save()` 已提供基础，但没有会话库、恢复策略、幂等控制。
- 环境：`Environment` Protocol 已把本地/Docker/Singularity 统一，适合讲“执行环境可替换”。

**依赖。** README 说 “no fancy dependencies”[S5]，但 `pyproject.toml` 声明了 14 个必需依赖：`pyyaml`、`requests`、`jinja2`、`pydantic`、`litellm`、`tenacity`、`rich`、`python-dotenv`、`typer`、`platformdirs`、`textual`、`prompt_toolkit`、`datasets`、`openai`。其中 `datasets`、`textual` 主要服务基准与 TUI，二次开发时可以裁剪；`litellm` 是模型层的实际依赖，不容易完全去掉。

**维护。** 最近提交 2026-11-02，正式版 v2.4.6（2026-07-23），MIT，9,372 Star / 1,296 Fork / 139 个未关闭 Issue。[S4][S6][S14] `pyproject.toml` 标注 `Development Status :: 3 - Alpha`，说明它还没把自己包装成稳定框架；这对“二次开发”不是坏事，但对“直接用于生产”要谨慎。

**为什么适合做作品集。** 它的空白正好对应后端岗位的评分点：接口设计、会话状态、并发、重试、限流、可观测性、评测。你不是在别人写好的 13 万行代码里找入口，而是在一个读得完的内核上做系统性扩展；改造前后的差异可以被 SWE-bench 子集或自建任务集量化。

### 5.2 smolagents：扩展点最齐全的“通用型”备选

**规模证据。** 发布包 18 个文件、12,774 行；`agents.py` 1,813 行，`models.py` 2,102 行，`local_python_executor.py` 1,768 行，`tools.py` 1,422 行，`remote_executors.py` 1,076 行。[S1]

**核心抽象。** `MultiStepAgent(ABC)` 派生 `CodeAgent`（用代码表达动作）与 `ToolCallingAgent`（走传统 tool calling）。`Model` 是一等抽象，覆盖 HF Inference、Transformers、MLX、vLLM、OpenAI、Azure、Bedrock、LiteLLM；`BaseTool`/`Tool`/`ToolCollection` 负责工具与 MCP；`AgentMemory` 负责步骤记录；`PythonExecutor` 与远程执行器负责沙箱。

**扩展点。**

- 工具：`@tool` 装饰器、`Tool` 子类、MCP、LangChain 工具、Hub 工具，是这次候选中工具生态最强的。
- 模型：新增 Provider 只需实现 `Model`；OpenAI/LiteLLM 都是可选 extras。
- 状态：`AgentMemory` 可替换，但没有内建会话持久化；要做恢复/多租户，需要自己补。
- 执行：代码动作可跑在 E2B、Modal、Docker、Blaxel 等沙箱；这是很好的安全工程素材。

**依赖。** 6 个必需依赖：`huggingface-hub`、`requests`、`rich`、`jinja2`、`pillow`、`python-dotenv`；OpenAI、LiteLLM、MCP、Docker 等全部是 extras。基础安装很轻，但用到哪个能力就装哪个 extra。[S1]

**维护。** Apache-2.0；最近提交 2026-11-03，但正式版停在 v1.26.0（2026-05-29），主分支已是 `1.27.0.dev0`；30,266 Star / 3,201 Fork / 1,050 个未关闭 Issue。[S1][S2][S14] Issue 存量高，说明需求旺，也说明维护带宽被摊薄。

**适合谁。** 如果 JD 更强调“Agent 平台 / 工具生态 / 多模型接入”，或者你想直接复用 code-as-action、沙箱、MCP、Hub，那么 smolagents 比 mini 少走很多路。代价是你要在 1.28 万行里建立自己的边界，且上游迭代会持续侵入你的改动。

### 5.3 openai-agents-python：适合当依赖，不适合当基座

**规模证据。** `src/agents` 317 个文件、134,534 行；`run_state.py` 5,130 行、`run.py` 2,785 行、`tool.py` 2,444 行、`agent.py` 1,047 行。[S7]

**核心抽象。** `Agent`、`Runner`、`RunConfig`、`function_tool`、`handoff`、`guardrail`、`Session`/`SQLiteSession`、`ModelProvider`、tracing，覆盖多 Agent 编排、会话、护栏、可观测性，文档与示例也是候选中最好的。[S9]

**为什么不推荐做基座。**

- 体量：13.45 万行，核心状态机 `run_state.py` 就有 5,130 行，读透成本远高于前两者。
- 厂商绑定：OpenAI 是一等公民；接其他 Provider 需要适配层，作品集容易被解读为“OpenAI SDK 的使用者”。
- 迭代速度：版本按月推进，内部模块仍在演化；你做的改造很容易在下一个版本冲突。
- 叙事：面试官更可能问“你为什么不用官方 SDK”，而不是“你改造了什么”。把它当依赖、在业务层做工程化，反而更合理。

许可证 MIT，维护最活跃（提交至 2026-11-07，v0.24.0 于 2026-10-15），Star 30,547 / Fork 5,005 / 仅 43 个未关闭 Issue。[S7][S8][S14] 质量不等于适合改造。

### 5.4 atomic-agents：类型化后端流水线的可选底座

**规模证据。** 30 个文件、5,944 行；`atomic_agent.py` 918 行，`mcp_factory.py` 982 行，`chat_history.py` 387 行，另有 OpenTelemetry 语义约定埋点。[S10]

**核心抽象。** `AtomicAgent` + `BaseTool[InputSchema, OutputSchema]` + `BaseIOSchema`（Pydantic）+ `ChatHistory` + MCP 连接器 + OTel GenAI spans。它把“工具输入输出必须是类型化 Schema”做成了硬约束，适合结构化数据管道。[S12]

**优点。** MIT；维护活跃（提交 2026-10-29，v2.15.0 于 2026-10-21）；29 个未关闭 Issue；可观测性跟 OTel 标准对齐；MCP 支持完整；依赖 `instructor`，结构化输出省事。

**限制。**

- 要求 Python ≥3.12；依赖 12 个，含 `instructor`、`litellm`、`mcp`、`textual`、`gitpython`，比 mini 还重。
- Agent 运行时偏“单次调用 + Schema 校验”，没有 mini 那种可改的循环/环境/恢复体系；你想讲“并发、预算、断点恢复”，能改的地方不多。
- PyPI 月下载约 1.17 万，社区规模明显小于另外三个（Star 6,320）。[S15]

**结论。** 如果你的项目定位是“类型化工具 + MCP + 可观测性的后端 Agent 流水线”，它是合理选择；如果定位是“Agent 运行时”，它不如 mini/smolagents。

### 5.5 swarm：只适合阅读

512 行、6 个文件，`core.py` 258 行；`Agent` + `functions` + `handoff` + `run()`，没有持久化、没有 Session、只支持 OpenAI。README 明确写着 “Swarm is now replaced by the OpenAI Agents SDK”，并建议迁移。[S13]

它的价值是 20 分钟读懂“handoff 是怎么实现的”；作为二次开发基座没有意义：功能太少，且上游已停止演进，改完也无法证明工程判断。Star 22,108 只说明它曾经有教学影响力。

---

## 6. 推荐结论

### 6.1 首选：mini-swe-agent

**推荐理由：**

1. **内核足够小，改造边界清楚。** 190 行 Agent + 两个 Protocol，读完只需半天；你做的每一处扩展都能对应到具体代码，而不是在框架里猜。
2. **空白正好是后端能力。** 原生没有 Tool 抽象、没有会话库、没有并发策略、没有结构化 trace；这四块恰好是后端/Agent 岗的评分点。
3. **有可量化基线。** 项目自述 SWE-bench verified >74%[S5]，并自带 benchmark 脚本；改造前后可以对比成功率、步数、成本，而不是只讲“我加了功能”。
4. **许可证与维护友好。** MIT，提交活跃，Star/Fork 规模适中，Issue 存量 139，不会被上游 PR 洪流淹没。
5. **故事完整。** “从一个 190 行的 bash Agent，长成可恢复、可观测、可并发、可评测的 Agent 运行时”，比“我基于某框架写了业务”更有分量。

**必须先接受的前提：**

- 它不是通用 Agent 框架，原生只有 bash；你要自己设计并实现 Tool 层。这既是工作量，也是作品集的核心证据。
- 必需依赖 14 个，和“运行时依赖 ≤2”的轻量目标冲突；可以通过裁剪 CLI/基准依赖降低，但 `litellm` 仍在。
- v2 仍是 Alpha；跟上游同步时要保留自己的适配层，避免被重构拖走。

### 6.2 备选：smolagents

**什么时候选它：**

- 目标岗位偏“Agent 平台 / 工具生态 / 多模型接入”，你想直接复用 code-as-action、沙箱、MCP、Hub。
- 你希望改造集中在“安全与治理”，比如代码执行沙箱、工具权限、Provider 路由与降级，而不是从零补 Tool 层。
- 你能接受 1.28 万行的阅读成本和 1,050 个 Issue 的上游噪声。

**改造方向：** 把 `CodeAgent` 的代码执行收敛到受控沙箱 + 工具白名单/权限模型；给 `Model` 加路由、fallback、成本与限流；把 `AgentMemory` 换成可持久化后端；补结构化 trace 与评测。面试叙事是“在通用框架上做安全与治理”，不是“从零造运行时”。

### 6.3 明确不推荐

| 候选 | 不推荐理由 |
| --- | --- |
| openai-agents-python | 13.45 万行、厂商主导、版本快；适合当依赖，不适合当二次开发基座。你很难在面试中主张“拥有”这套系统。 |
| swarm | 已由 Agents SDK 取代，2026-04 后无提交；功能过少，改完没有对照价值。仅作 handoff 实现的教学阅读。 |

---

## 7. 拿到手后的改造路线（以 mini-swe-agent 为首选）

目标不是“加很多功能”，而是让每个改造都对应一个可验证的工程判断。

### 阶段 0：读透与基线（先别改）

- 跑通 `hello_world` 和一个真实小任务；读 `agents/default.py`、`models/litellm_model.py`、`environments/local.py`、`run/hello_world.py`。
- 画一张调用图：`Agent.run → Model.query → Environment.execute → 消息追加`。
- 记录基线：任务成功率、平均步数、单步 token/费用、失败类型分布。
- 产出：一页架构笔记 + 基线数据；这是后续所有改造的对照。

### 阶段 1：Tool 层与接口设计（对应“新增工具零改动”）

- 增加 `Tool` Protocol、注册表、JSON Schema 校验、权限标记（只读/写/执行）。
- 把 bash 包装成一个工具；再实现 `read_file`、`write_file`、`web_fetch`，保持核心循环不变。
- 关键设计：工具实现不反向依赖 Agent；参数错误、未知工具、超时统一走“错误回传模型”的路径。
- 可讲点：接口最小契约、依赖单向、错误分类；验收是“新增第 5 个工具只改一个文件 + 一行注册”。

### 阶段 2：会话、状态与恢复（对应“可恢复、可回放”）

- 把线性消息历史落到 JSONL/SQLite；实现 `--resume`、断点续跑、步骤幂等。
- 增加 token 估算与上下文裁剪：保留 system + 最近 N 轮 + 工具输出截断。
- 保留“轨迹即消息”的特性，确保调试、回放、后续微调都还能用。
- 可讲点：状态模型、恢复语义、裁剪副作用；验收是“中断后会话完整、恢复后能正确续答”。

### 阶段 3：可靠性、并发与成本（后端岗核心证据）

- 模型请求：错误分类、指数退避 + 抖动、尊重 `Retry-After`、超时预算；**工具不自动重试**（有副作用）。
- 并发策略：同轮多工具“只读并发、写/执行串行”，结果按 `tool_call_id` 原序回填。
- 成本与配额：在 `GlobalModelStats` 基础上做会话级/租户级预算，超限可中止而不是静默烧钱。
- 可讲点：重试边界、并发竞态、顺序保证、预算控制；验收是“429/超时/多工具并发的测试全绿”。

### 阶段 4：可观测性与评测（让改造可量化）

- 结构化 JSONL：`ts`、`session_id`、`turn`、`tool`、`duration_ms`、`prompt_tokens`、`completion_tokens`、`cost`、`ok`、`error_code`；默认不记录正文与密钥。
- 提供 `trace <session_id>` 回放：按轮次看工具、耗时、错误；可选接 OTel。
- 建一个小评测集：SWE-bench verified 子集或自建 10–20 个任务；对比改造前后的成功率、步数、成本、p95 延迟。
- 可讲点：字段设计、脱敏、可回放、回归评测；这是把“我改了”变成“我证明有效”的关键。

### 阶段 5：服务化（可选，视 JD 决定）

- 用 FastAPI + SSE 把同一内核暴露为服务；会话隔离、并发上限、取消/中断。
- 利用现有 `Environment` Protocol 切换 Local/Docker/Singularity，做执行环境池与资源限制。
- 可讲点：CLI/HTTP 双入口、同一内核不分裂、隔离与配额；不要为了“显得后端”堆 Redis/MQ，除非确有需求。

### 改造项与面试证据映射

| 改造项 | 面试可讲的点 | 验收证据 |
| --- | --- | --- |
| Tool Protocol + 注册表 | 接口设计与依赖方向 | 新增工具零改循环；坏参数/未知工具测试 |
| 会话持久化 + 恢复 | 状态机与恢复语义 | 中断后可恢复；JSONL 可回放 |
| 重试 + 超时 + 限流 | 可靠性边界 | 429/5xx/超时测试；工具不自动重试 |
| 只读并发 + 顺序回填 | 并发正确性 | 并发测试 + 顺序断言 |
| 成本预算 | 资源治理 | 超限中止；费用统计 |
| 结构化日志 + trace | 可观测性 | 凭日志定位失败轮次/工具 |
| 评测集 | 工程闭环 | 改造前后指标对比 |

---

## 8. 与现有 PLAN.md 的关系

本次调研不修改 `PLAN.md`。如果采纳“基于 mini-swe-agent 改造”，需要提前想清楚两处冲突：

1. **依赖约束。** `PLAN.md` 要求运行时依赖 ≤2；mini 声明 14 个必需依赖。要么放宽约束（保留核心依赖并解释取舍），要么只借鉴架构、自己重写（但那就不再是“二次开发”）。
2. **叙事。** `PLAN.md` 的面试叙事是“从零实现、所以可控”；改成“读透 190 行内核后做生产化改造”，故事同样成立，但证据重心从“我写了循环”转为“我设计了接口、状态、可靠性与评测”。

不变的是评分表：循环与状态、并发、可靠性、可观测性、接口与测试。变的只是起点。若不想动 `PLAN.md` 的约束，另一条路是把 mini/smolagents 仅作为架构参照，仍按原计划从零写；本调研的对比数据可用于“为什么不用框架”的答辩。

---

## 9. 参考来源

- [S1] smolagents 仓库与源码：[https://github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)
- [S2] smolagents v1.26.0 发布页：[https://github.com/huggingface/smolagents/releases/tag/v1.26.0](https://github.com/huggingface/smolagents/releases/tag/v1.26.0)
- [S3] smolagents README（“~1,000 lines”与能力说明）：[https://github.com/huggingface/smolagents/blob/main/README.md](https://github.com/huggingface/smolagents/blob/main/README.md)
- [S4] mini-swe-agent 仓库与源码：[https://github.com/SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
- [S5] mini-swe-agent README（“100 lines”、SWE-bench >74%、依赖说明）：[https://github.com/SWE-agent/mini-swe-agent/blob/main/README.md](https://github.com/SWE-agent/mini-swe-agent/blob/main/README.md)
- [S6] mini-swe-agent v2.4.6 发布页：[https://github.com/SWE-agent/mini-swe-agent/releases/tag/v2.4.6](https://github.com/SWE-agent/mini-swe-agent/releases/tag/v2.4.6)
- [S7] openai-agents-python 仓库与源码：[https://github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python)
- [S8] openai-agents-python v0.24.0 发布页：[https://github.com/openai/openai-agents-python/releases/tag/v0.24.0](https://github.com/openai/openai-agents-python/releases/tag/v0.24.0)
- [S9] OpenAI Agents SDK 文档：[https://openai.github.io/openai-agents-python/](https://openai.github.io/openai-agents-python/)
- [S10] atomic-agents 仓库与源码：[https://github.com/Eigenwise/atomic-agents](https://github.com/Eigenwise/atomic-agents)
- [S11] atomic-agents v2.15.0 发布页：[https://github.com/Eigenwise/atomic-agents/releases/tag/v2.15.0](https://github.com/Eigenwise/atomic-agents/releases/tag/v2.15.0)
- [S12] atomic-agents 文档：[https://eigenwise.github.io/atomic-agents/](https://eigenwise.github.io/atomic-agents/)
- [S13] swarm 仓库与 README（已被 Agents SDK 取代）：[https://github.com/openai/swarm](https://github.com/openai/swarm)
- [S14] GitHub REST API（仓库元数据查询，2026-11-08）：[https://docs.github.com/en/rest/repos/repos](https://docs.github.com/en/rest/repos/repos)
- [S15] pypistats（PyPI 下载趋势，含镜像/CI 流量）：[https://pypistats.org/](https://pypistats.org/)
- [S16] SWE-bench 官方榜单：[https://www.swebench.com/](https://www.swebench.com/)
- [S17] mini-swe-agent 文档：[https://mini-swe-agent.com/latest/](https://mini-swe-agent.com/latest/)

> 代码量说明：本文行数为发布包内 `.py` 文件的物理行数（含空行与注释）。统计日期 2026-11-08，默认分支浅克隆；不同统计工具（cloc、tokei）会因注释/空行口径产生差异，量级判断不受影响。
