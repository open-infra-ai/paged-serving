# Agent Note: 有界事件队列与请求取消

Status: implemented

## Problem

服务层每请求事件与多候选 fan-in 都使用无界队列；消费端消失时，取消还依赖下一次非空文本发送。满队列、末步完成与断连并发时，终态投递和资源回收必须有独立保证。

## Decision

复用 PR #23（head `25811e35c2d978f39efd6eb9731dc6fadc9c8a25`）的 `RequestGuard`、watch 取消与 shutdown 入口，在独立整改分支集成，不合并或改动该 PR。

- 每候选文本 mailbox 容量由 `EngineConfig.event_channel_capacity` 指定，默认 64，零容量拒绝；旧 JSON 缺省该字段仍可读取。Rust 全字段 struct literal 需要补新字段，无 C ABI 变化。
- 引擎仅 `try_send`；队列满即终止该候选，错误文本为 `slow consumer: event channel overflow`。独立 oneshot 保证终态不受满队列阻挡；失败与取消分类见 [类型化终态](../bug-fix/2026-10-04-typed-cancellation-and-metrics.md)。
- 失败终态优先，丢弃剩余文本并输出 error 与 `[DONE]`；成功终态必须先排空已产生文本，再发送 usage 与 `[DONE]`。末步已经成功但文本投递溢出时，HTTP 仍失败，不能宣称完整成功。
- unary 不订阅文本，直接等待终态，避免没有消费需求的 mailbox 误触发 overflow。
- 多候选直接使用已有 futures-util 的 `SelectAll` 拉取各候选事件，删除 fan-in 队列与转发任务。队列项上界为 `n × capacity`，再加至多每候选一个合并器持有的事件；这不是字节上界，logprobs、输出历史与网络缓冲另计。
- 引擎检出 shutdown 真值后不可逆退出：关闭 submission 接收端，取消已有请求，排空未准入队列并拒绝它们。watch 的版本变化不等于真值，`send(false)` 不应误取消；调用方触发 shutdown 后必须保持真值，不能把它当可撤销开关。
- `inflight` 覆盖 SSE body lifetime；typed Cancelled 与 cancelled/errors 指标口径由 [指标决策](../bug-fix/2026-10-04-typed-cancellation-and-metrics.md) 接管，不改变队列和主动取消所有权。

## Review

沿用 [取消与背压设计包](../../../../docs/architecture/cancellation-backpressure-design.md) 的所有权、非阻塞生产者和带外终态决定。修正其中“成功终态忽略残余 chunk”会丢文本的规则；fan-in 改为直接拉取，减少第二级缓冲和任务退出路径。G0-G8 范围关闭于 CPU 控制面；真实 CUDA free 与性能仍需单独验证，不声称同步 GPU step 可被中断。

现有活跃笔记检索未发现相同取消/背压归属；MSRV 笔记仅涉及工具链，无重叠。backend `sequences_finished` 与 scheduler property tests 提供现有回收验证，本次增加真实服务入口与受控时序测试。

## Alternatives considered

独立 forwarding task 加有界 fan-in 能复用已有组织方式，且生产者可 await；但它引入第二层缓冲、满队列上的终态等待与任务回收。直接拉取保持每候选次序且减少所有权节点。

引擎 await 有界发送可无损等待客户端，但唯一 engine worker 会被单个慢请求阻塞，其他请求无法推进；因此选择局部取消慢消费者，而不承诺无损等待任意慢客户端。

Done 与 Chunk 共用有界 mailbox 最简单、天然保序；但满队列时 Done 无法进入，只能依靠 sender-drop 报泛化错误。独立终态通道允许准确失败原因与成功排空。

## Verification

容量 1 的确定性测试覆盖 overflow、末步 overflow、成功排空、失败抢占、unary 长输出、多候选丢弃/准入失败、无文本取消、shutdown 后拒绝；probe 记录 backend release 恰好一次，KV 与活跃槽位回基线。聚合错误首次发出时 sibling cancel 已置位，不依赖客户端继续读取 `[DONE]`。scheduler 原有 property tests 继续覆盖随机取消/完成/失败的资源守恒。

Rust 1.88 的 `cargo check --locked --all-targets` 和 `cargo test --locked` 通过：255 个默认测试与 17 个 doc tests（较第一批新增 25 个测试）。24 个服务内联测试和 45 个 HTTP/SSE 集成测试重复运行 10 轮通过。stable fmt/clippy 通过；Cargo.lock 与 C ABI 无变化。远端 CI 状态以本分支 Actions 为准，真实 CUDA 测试未运行。

## Consequences

文本队列具备可检查的项数上限，取消与终态不依赖文本生产；删除 fan-in 转发任务减少独立生命周期。代价是慢客户端获得明确失败，而非无限等待；Rust 全字段构造和错误枚举穷尽匹配需要更新。候选在引擎中已成功、但末步文本溢出时，引擎 completed 与 HTTP failed 是不同层级，不据此推导交付成功率。默认容量 64 未经真实网络负载调优；同步 backend step 结束前无法响应取消，HTTP 网络排空未设置 deadline。真实后端回收和网络负载调优仍是验证缺口；独立取消与错误统计的验证由指标笔记维护。
