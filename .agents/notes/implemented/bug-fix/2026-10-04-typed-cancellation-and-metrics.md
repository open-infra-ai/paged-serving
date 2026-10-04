# Agent Note: 类型化取消与 HTTP 错误计数

Status: implemented

## Problem

取消与后端故障共用 Failed，监控无法区分客户端退出和计算错误。JSON 提取错误和 SSE 终态错误没有完整进入 HTTP 错误计数；一个 HTTP 请求启动多个候选时，直接按候选累加又会重复报警。

## Decision

使用 `CancellationReason` 和 `RequestState::Cancelled` 表达客户端退出与停机，`CompletedRequest.cancellation` 携带类型化原因。引擎按候选分别累计 completed、failed、cancelled；慢消费者溢出属于失败，不属于正常取消。HTTP 请求共享 `HttpRequestOutcome`，准入错误、候选终态与 SSE 提前关闭使用同一个原子去重标记；未读取 body 的后端失败仍可观测。SSE body 所有权由既有 inflight guard 管理。

## Counting units

| 情况 | 引擎 completed / failed / cancelled | HTTP errors |
|------|------------------------------------|-------------|
| JSON、参数或 429 拒绝 | 未准入候选不增加；已准入 sibling 可取消 | 每 HTTP 请求一次 |
| 三个候选正常完成 | 3 / 0 / 0 | 0 |
| 客户端退出、unary abort 或 shutdown | 每个尚未完成候选 cancelled +1 | 0 |
| 后端失败 | 每个失败候选 failed +1；未失败 sibling 可取消 | 每 HTTP 请求最多一次，body 未读也计 |
| 完成前文本溢出 | 0 / 1 / 0（每溢出候选） | 每 HTTP 请求最多一次 |
| 最后一步已计算成功、文本溢出 | 1 / 0 / 0（不回写引擎历史） | 每 HTTP 请求最多一次 |
| 事件通道消失但无 Done | 不虚构候选终态 | body 被消费时计一次 |

requests_total 只计 completion/chat HTTP 请求，streaming_total 只计成功构建的 SSE 响应；两者不乘候选数。inflight 覆盖 handler 或 SSE body 存活期。token 总数在排出终态时累计，含失败/取消前的部分输出。各项都提供 HELP/TYPE，但独立原子读取与步末刷新不是事务快照，也不是网络交付确认。

## Alternatives considered

字符串前缀分类能保持 public Rust 类型不变，但文案调整、后端错误恰好包含相同前缀都会改变统计。类型化状态用于分类，字符串只用于错误信封。

仅在 SSE 被读取时累加容易实现，但慢消费者不读取 body 时会漏报，n>1 的多个错误入口还会重复。共享每 HTTP 请求的去重标记保留候选计数与 HTTP 计数的不同单位。

用统一 outcome enum 替换 CompletedRequest 全部字段能强制全部不变量，但要求调用方全面迁移。保留 success/error/finish_reason，补 cancellation 字段，限制这批 Rust API 的迁移范围；C ABI 不变。

## Verification

Rust 1.88 的 locked all-target check 和完整默认测试通过：260 个默认测试与 17 个 doc tests。25 个服务内联与 48 个 HTTP/SSE 测试重复 10 轮通过（730 次用例执行），覆盖未读 body、n>1 去重、静默取消、shutdown unary、提前关闭、溢出和成功终态。scheduler 测试确实构造 pending/prefill/decode 三种状态；引擎测试证明错误文案包含 cancellation 前缀仍是 Failed，并检查所有终态资源恰好回收一次。stable clippy 无警告，fmt 与笔记门禁使用仓内命令验证。真实 CUDA 与真实网络故障注入不在这组 CPU 验收中。

## Consequences

指标能够区分主动退出、计算故障和应用层交付失败；调度器复用相同终态资源清理，不复制三个阶段的取消路径。代价是每 HTTP 请求增加一个共享记录对象，以及 public enum 变体/struct 字段的 Rust source-breaking change；调用方需要更新穷尽匹配和 struct literal，C ABI 和 Cargo.lock 保持原样。failed 排除取消后不可与旧口径直接拼接。取消保留 500 / internal_error 错误信封但不增加 HTTP errors，因此此 counter 不能直接充当 HTTP 5xx 数。同步 backend step 与网络排空 deadline 的限制仍存在；真实 TCP 故障与 CUDA 回收需要独立验收。

## Related notes

[有界事件与取消](../feature/2026-10-04-bounded-events-and-cancellation.md) 部分重叠：其队列与资源回收决定保留，本篇接管 typed cancellation 和完整指标；工具链与真实 HTTP 回归笔记分别处理编译门禁与传输测试，没有相同决定。
