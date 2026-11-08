# miniagent

一个从零手写的命令行 AI agent：模型能自己读文件、写文件、跑命令，并把结果接回推理循环。

不依赖 LangChain 等 agent 框架，核心循环与工具层均为自研，约 1500 行 Python（含测试约 2700 行）。目标是做一个**可读、可控、可扩展**的最小 harness，而不是功能堆砌。

当前进度：**M0–M4 全部完成（Mock 下完整可演示）**：多轮工具循环、会话持久化、上下文裁剪、SSE 流式、重试/限流/工具并发、结构化 trace、路径白名单与 CI。

## 快速开始

需要 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync

# 离线体验，不请求任何网络
uv run miniagent --mock -p "你好"

# 交互模式（默认 Mock）
uv run miniagent --mock
```

接入真实模型时，创建 `config.toml`（已在 `.gitignore` 中）：

```toml
provider = "openai"
base_url = "https://api.deepseek.com/v1"   # 任何 OpenAI 兼容端点
model = "deepseek-chat"
api_key = "sk-..."
```

也可以用环境变量覆盖：`MINIAGENT_PROVIDER`、`MINIAGENT_BASE_URL`、`MINIAGENT_MODEL`、`MINIAGENT_API_KEY`（兼容 `OPENAI_API_KEY`）、`MINIAGENT_MAX_CONTEXT_TOKENS`（默认 12000）、`MINIAGENT_MAX_RETRIES`（默认 3，含首次请求）、`MINIAGENT_RATE_LIMIT_RPM`（默认 60，0 表示关闭限流）、`MINIAGENT_LOG_DIR`（trace 目录，默认 `~/.miniagent/logs`）。

## 能做什么

### 单次执行

```bash
$ uv run miniagent --mock -p "你好"
[mock] 收到：你好
[mock] 离线回显，未调用真实模型；本次上下文 2 条消息。可用工具：read_file、write_file、run_shell。
session: 20261109-001933-0a7f34（--resume 20261109-001933-0a7f34 可继续）
```

### 工具循环（离线可复现）

一次任务里模型连续调用多个工具，结果按原顺序回填：

```text
> 写一个文件再读回来
  -> write_file {"path": "note.txt", "content": "hello miniagent"}
  <- write_file ok 0ms | 已写入 note.txt（15 字节）
  -> read_file {"path": "note.txt"}
  <- read_file ok 3ms | hello miniagent
文件已写好，读回内容一致
```

终端里始终能看到工具名、参数摘要、耗时与结果预览；工具报错不会中断进程，而是作为结果回传给模型自修正。

### 会话持久化与恢复

每条消息立即追加到 `~/.miniagent/sessions/<id>.jsonl`，进程退出或中断都不丢历史：

```bash
# 第一次
uv run miniagent --mock -p "记住我叫小明"
# session: 20261109-001933-0a7f34

# 新进程恢复，模型能看到之前 3 条历史
uv run miniagent --mock --resume 20261109-001933-0a7f34 -p "回看历史"
# [mock] ... 本次上下文 4 条消息。
```

REPL 命令：

| 命令 | 作用 |
| --- | --- |
| `/tools` | 列出已注册工具 |
| `/sessions` | 列出最近会话 |
| `/resume <id>` | 切换到指定会话 |
| `/clear` | 开一个新会话 |
| `/exit` | 退出 |

### 流式输出与上下文裁剪

- 模型文本通过增量回调送到 CLI，边生成边打印；`tool_calls` 的 SSE 分片按 `index` 拼接。流式与非流式共用同一个 `Agent` 循环。
- 流式请求在收到第一个字之前失败时，自动降级为非流式重试一次；已经输出过内容则直接报错，避免重复输出。
- 历史超过 `max_context_tokens`（默认 12000）时，只裁剪“发给模型的消息视图”：保留 system + 最近 6 轮，先截断较早轮次的工具输出，仍超预算再丢最旧整轮。会话文件始终保留完整历史。

```bash
$ MINIAGENT_MAX_CONTEXT_TOKENS=1000 uv run miniagent --mock
> 第 1 条 ...
...
> [context] 历史过长，本次请求只带 12/22 条消息（估算 1051 -> 646 tokens）
```

`Ctrl+C` 中断时，已接收的半截文本不会写入会话，恢复后历史仍然一致。

### 重试、限流与工具并发

- **错误分类**：429/408/5xx 与超时、连接失败可重试；其他 4xx（参数错误、鉴权失败）立即失败，不做无用重试。
- **指数退避**：0.5s 起翻倍、带抖动、上限 8s；服务端返回 `Retry-After` 时优先服从。默认最多尝试 3 次（`MINIAGENT_MAX_RETRIES`）。
- **单进程令牌桶限流**：按 `MINIAGENT_RATE_LIMIT_RPM`（默认 60）平滑放行；预计等待超过 30s 直接报错，不会无声挂住。
- **只读并发、写串行**：同一轮里 `read_file` 这类只读工具用线程池并发执行，`write_file` / `run_shell` 按原序串行；`tool_start` 先全部发出，`tool_end` 与回填消息仍按原序，模型看到的结果顺序稳定。未知工具按“有副作用”处理。
- 工具调用**永不自动重试**：避免写文件、跑命令重复产生副作用；失败作为错误文本回传，由模型自行修正。

```text
max_tool_workers=1: 0.81s   # 3 个只读 + 1 个写，全部串行
max_tool_workers=4: 0.40s   # 只读并行（0.2s），写仍串行（0.2s），结果顺序不变
```

### 结构化日志与 trace 回放

每次运行都会把固定字段追加到 `~/.miniagent/logs/trace.jsonl`（可用 `MINIAGENT_LOG_DIR` 或 `--log-dir` 改路径）：

```bash
$ uv run miniagent trace 20261109-001933-0a7f34
session: 20261109-001933-0a7f34（10 条记录）
[turn 0] run_start prompt≈243 tokens
[turn 1] request  prompt≈243 tokens
[turn 1] response 0ms completion≈34 tokens
[turn 1] tool     write_file ok 0ms
[turn 2] request  prompt≈311 tokens
[turn 2] response 0ms completion≈26 tokens
[turn 2] tool     read_file ok 1ms
[turn 3] request  prompt≈348 tokens
[turn 3] response 0ms completion≈12 tokens
[turn 3] run_end  ok 6ms
```

- 字段固定为 `ts / session_id / turn / event / tool / duration_ms / prompt_tokens / completion_tokens / ok / error_code`，一行一个 JSON 对象。
- 事件覆盖 `run_start / request / response / tool / trim / error / run_end`；失败轮次带 `error_code`（`provider_error` / `tool_error` / `max_turns`）。
- **只记元数据**：不写对话正文、工具参数/输出、API 密钥；排查时凭轮次、工具名、耗时与错误码定位。
- `prompt_tokens` / `completion_tokens` 由 `context.estimate_tokens` 粗略估算，不依赖 tokenizer。

### 安全边界

- **路径白名单**：CLI 默认只允许读写当前工作目录；`--allow-dir DIR` 可追加（可重复）。`PathPolicy` 先 `resolve`，`..` 与指向外部的软链都会被拒绝。
- **危险命令黑名单**：`run_shell` 执行前拦截 `rm -rf /`、`mkfs`、`shutdown`、`format C:`、`curl ... | sh` 等模式；命中后作为工具错误回传模型，不中断进程。
- **写入上限**：单次 `write_file` 内容不超过 1 MiB；工具输出仍截断到 8000 字符。
- 这是**应用层护栏，不是容器沙箱**：`run_shell` 仍然是真实执行，只建议在受信本机运行。

## 架构

```
CLI (cli.py)                 参数 / REPL / 事件渲染 / trace 子命令 / 退出码
  |
Agent (agent.py)             多轮循环 / 轮次上限 / 工具调度 / trace 埋点 / 事件回调
  |-- LLMProvider (llm.py)    Protocol；Mock 与 OpenAI 兼容实现
  |     `-- reliability.py    错误分类 / 令牌桶 / 退避策略
  |-- ToolRegistry            Schema 导出 / 参数校验 / 执行与错误兜底
  |     `-- builtin.py        read_file / write_file / run_shell
  |           `-- security.py 路径白名单 / 危险命令黑名单
  |-- Session (session.py)    JSONL 追加写入 / 加载 / 列举
  |-- context.py              token 估算 / 裁剪请求视图（不改历史）
  `-- logging.py              固定字段 trace / 回放数据源
```

依赖单向向下：CLI → Agent → Provider / 工具抽象；工具实现不依赖 Agent。Agent 只发出事件，不负责打印，接入 HTTP/SSE 服务时无需改循环。

核心数据结构：

- `Message`：`role` / `content` / `tool_calls` / `tool_call_id`，贴齐 OpenAI 兼容协议。
- `ToolCall`：`id` / `name` / `arguments`（保留原始 JSON 字符串，先校验再解析）。
- `AgentEvent`：`text` / `text_delta` / `text_end` / `tool_start` / `tool_end` / `context_trimmed` / `error`。
- `TraceRecord`：`ts` / `session_id` / `turn` / `event` / `tool` / `duration_ms` / `prompt_tokens` / `completion_tokens` / `ok` / `error_code`。

## 岗位能力映射

| 能力点 | 代码位置 | 对应测试 |
| --- | --- | --- |
| 工具调用循环与控制流 | `agent.py` | `tests/test_agent_loop.py` |
| 会话持久化与上下文裁剪 | `session.py`、`context.py` | `tests/test_session.py`、`tests/test_context.py` |
| 并发与顺序回填 | `agent.py` 的 `_run_tools`、`tools/registry.py` | `tests/test_reliability.py` |
| 流式输出与中断安全 | `llm.py`、`cli.py` | `tests/test_streaming.py` |
| 错误分类、重试与限流 | `reliability.py`、`llm.py`、`errors.py` | `tests/test_reliability.py` |
| 可观测性与 trace 回放 | `logging.py`、`cli.py` 的 `trace_main` | `tests/test_logging.py` |
| 路径白名单与命令黑名单 | `security.py`、`tools/builtin.py` | `tests/test_security.py` |
| 接口设计与模块边界 | `tools/base.py`、`llm.py` 的 Protocol | `tests/test_tools.py` |
| CLI、配置与退出码 | `cli.py`、`config.py` | `tests/test_cli.py` |

> 上表每一行都能在仓库里找到实现与测试；面试时按行指代码即可。

## 新增一个工具

写一个类，注册一行，核心循环零改动：

```python
# src/miniagent/tools/builtin.py
class ListDirTool:
    name = "list_dir"
    description = "列出目录下的文件，用于了解项目结构。"
    read_only = True
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def run(self, args: dict) -> str:
        return "\n".join(p.name for p in Path(args["path"]).iterdir())
```

```python
# cli.py 的 build_registry()
for tool in [*builtin_tools(policy), ListDirTool()]:
    registry.register(tool)
```

参数校验、错误回传、事件渲染、会话落盘都会自动生效。

## 设计取舍

1. **不用 agent 框架**：核心循环不到百行，出问题能定位到每一行；框架的抽象成本大于收益。
2. **结构化 tool calling，bash 只是其中一个工具**：新增工具只加一个类 + 一行注册，才能证明接口可扩展；同时保留 `run_shell` 作通用兜底。
3. **消息格式贴齐 OpenAI 协议**：换模型只改 `base_url + model`，协议差异收敛在 `llm.py`。
4. **工具错误不终止进程**：错误文本回传模型，由模型决定修正；但工具不自动重试，避免副作用重复执行。
5. **会话与日志都用 JSONL**：追加写、可读、可 grep，不需要数据库。
6. **MockProvider 是一等公民**：所有测试与演示都能离线跑，面试官不需要密钥就能复现。
7. **流式用回调而不是第二套接口**：`chat(..., on_text=...)` 一个入口，Agent 循环、Mock 与真实实现共用；代价是 Provider 多了一个可选职责。
8. **可靠性放在单进程内**：令牌桶与退避都是进程内实现，不引 Redis/MQ；多进程部署时它是下限，需要全局限流再替换 `TokenBucket` 即可，接口不变。
9. **日志只记元数据**：trace 不写对话正文、工具参数/输出与密钥，换取"可回放又不泄露隐私"；排查粒度到轮次、工具、耗时与错误码。
10. **安全边界放在工具层**：路径策略由工具构造参数注入，Agent 与 Provider 不感知；新增工具可以按需复用，也可显式选择不挂策略（测试与库调用场景）。

## 测试

```bash
uv run pytest        # 83 个用例，全部离线
uv run ruff check .  # lint
```

覆盖：正常调用、坏参数、幻觉工具名、非法 JSON、工具异常、轮次上限、单轮多工具顺序、跨进程会话恢复、会话文件损坏、SSE 解析、流式降级、中断安全、上下文裁剪、错误分类、Retry-After 与指数退避、限流阻塞、只读并发与写串行、trace 字段与脱敏、trace 回放、路径白名单、危险命令拦截、写入大小上限。

CI 配置在 [.github/workflows/ci.yml](.github/workflows/ci.yml)：push / PR 触发，Python 3.11 与 3.12 双版本跑 `ruff check .` 和 `pytest -q`。

## 路线图

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 工程骨架、模型调用、CLI 单次执行 | 完成 |
| M1 | 工具注册表、多轮循环、会话持久化与恢复 | 完成（录屏暂缓） |
| M2 | 上下文裁剪、SSE 流式输出、中断安全 | 完成（Mock 验收） |
| M3 | 错误分类、指数退避重试、令牌桶限流、只读工具并发 | 完成（Mock 验收） |
| M4 | 结构化日志、trace 回放、路径白名单、CI | 完成（Mock 验收） |

## 边界

- `run_shell` 会真实执行命令，仅建议在本机受信环境使用；已有路径白名单、危险命令黑名单与写入上限，但**不是容器沙箱**。
- 会话文件包含完整对话内容，默认写在用户目录；trace 只含元数据。不要把这两个目录提交到公开仓库。
