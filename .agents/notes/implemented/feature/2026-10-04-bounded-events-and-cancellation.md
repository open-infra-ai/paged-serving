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

CPU 传输回归使用真实 axum listener、HTTP/1.1 socket 和既有 engine loop，不把
Router oneshot 的 body drop 等同网络断连。测试执行器在每个同步步骤等待有界许可；
客户端分别在首文本后、空文本 decode 期间和 unary 响应头前关闭 socket。网络侧
inflight 先归零，才允许已在途步骤返回；再观察取消终态、逻辑 KV/active 回基线和
后端释放通知恰好一次。同步在途步骤仍可完成，不要求或声称 kernel 抢占。
同一服务实例的四个真实 HTTP 探针成功；shutdown 另验证存活 SSE 的错误终态与 DONE。

## Review

沿用 [取消与背压设计包](../../../../docs/architecture/cancellation-backpressure-design.md) 的所有权、非阻塞生产者和带外终态决定。修正其中“成功终态忽略残余 chunk”会丢文本的规则；fan-in 改为直接拉取，减少第二级缓冲和任务退出路径。G0-G8 范围关闭于 CPU 控制面；真实 CUDA free 与性能仍需单独验证，不声称同步 GPU step 可被中断。

现有活跃笔记检索未发现相同取消/背压归属；MSRV 笔记仅涉及工具链，无重叠。backend `sequences_finished` 与 scheduler property tests 提供现有回收验证，本次增加真实服务入口与受控时序测试。

## Alternatives considered

独立 forwarding task 加有界 fan-in 能复用已有组织方式，且生产者可 await；但它引入第二层缓冲、满队列上的终态等待与任务回收。直接拉取保持每候选次序且减少所有权节点。

引擎 await 有界发送可无损等待客户端，但唯一 engine worker 会被单个慢请求阻塞，其他请求无法推进；因此选择局部取消慢消费者，而不承诺无损等待任意慢客户端。

Done 与 Chunk 共用有界 mailbox 最简单、天然保序；但满队列时 Done 无法进入，只能依靠 sender-drop 报泛化错误。独立终态通道允许准确失败原因与成功排空。

Router oneshot 可直接控制 body 的持有/丢弃，适合确定性 mailbox 溢出回归；但它跳过
hyper 与 socket 生命周期。真实 TCP 用例与这些测试并存，不从“客户端停止读取”猜
队列必然已满：内核缓冲、HTTP 写出与应用 mailbox 不是同一层。

固定慢步骤加 sleep 能扩大断连时窗、实现最短；但自然完成与 CI 调度延迟会影响
断言。测试专用有界许可冻结步骤，指标作为 HTTP owner 消失的屏障；五秒只是
异常等待上限，不是生成/取消预算，也不改变生产执行器或引擎协议。
测试许可等待经 `block_in_place` 交还 runtime worker，使 socket/handler 能在步骤
暂停时独立推进；这不是生产 executor 的 offload 方案。直接在 async worker 等待
许可可能让已唤醒的 HTTP 工作和测试屏障互相等待，不能把这种夹具失败称为产品泄漏。

## Verification

容量 1 的确定性测试覆盖 overflow、末步 overflow、成功排空、失败抢占、unary 长输出、多候选丢弃/准入失败、无文本取消、shutdown 后拒绝；probe 记录 backend release 恰好一次，KV 与活跃槽位回基线。聚合错误首次发出时 sibling cancel 已置位，不依赖客户端继续读取 `[DONE]`。scheduler 原有 property tests 继续覆盖随机取消/完成/失败的资源守恒。

Rust 1.88 的 `cargo check --locked --all-targets` 和 `cargo test --locked` 通过：255 个默认测试与 17 个 doc tests（较第一批新增 25 个测试）。24 个服务内联测试和 45 个 HTTP/SSE 集成测试重复运行 10 轮通过。stable fmt/clippy 通过；Cargo.lock 与 C ABI 无变化。远端 CI 状态以本分支 Actions 为准，真实 CUDA 测试未运行。

`tests/server_tcp_lifecycle.rs` 的四个真实 TCP 用例覆盖首段文本后、无文本 decode 与
unary 响应头前的断连，以及 shutdown 的 SSE 终态。每个断连用例先确认在途逻辑
序列和存活 HTTP owner，关闭 socket 后等待 inflight=0，再放行当前同步步骤；
取消=1、failed/completed=0、逻辑 KV/active=0，执行过的序列收到恰好一次释放通知。
原实例的四个两-token unary 探针全部返回 AA/usage=2，累计 completed=4、cancelled=1，
五个不同序列各释放一次；不声称四个探针一定同时占满调度槽位。
shutdown 用例读取真实 socket 的一个 error 与一个 DONE、无 usage，readyz=503；
取消不计服务错误。整个测试模块只使用受控 CPU 探针，没有真实模型输入。

许可直接占住 runtime worker 时，重复批次发生 HTTP read 超时并中断，未记作通过。
夹具用 `block_in_place` 交还 worker 后，四个用例连续 50 轮通过（200 次场景执行，
800 个 completion HTTP 请求，不含指标/readiness 请求）；五秒上限与验收断言保持。
该批完整默认套件实际 280 个测试加 17 个 doc tests，真实 tokenizer 明确 1 个 ignored；
Rust 1.88 locked all-target check、stable clippy/fmt、40 个离线结果审计测试与 notes
门禁通过。生产实现、C ABI、锁文件、构建/CI 和历史 raw 没有修改。

## Consequences

文本队列具备可检查的项数上限，取消与终态不依赖文本生产；删除 fan-in 转发任务减少独立生命周期。代价是慢客户端获得明确失败，而非无限等待；Rust 全字段构造和错误枚举穷尽匹配需要更新。候选在引擎中已成功、但末步文本溢出时，引擎 completed 与 HTTP failed 是不同层级，不据此推导交付成功率。默认容量 64 未经真实网络负载调优；同步 backend step 结束前无法响应取消，HTTP 网络排空未设置 deadline。真实后端回收和网络负载调优仍是验证缺口；独立取消与错误统计的验证由指标笔记维护。

真实 TCP 回归只证明受控 CPU 后端的 HTTP/控制面与释放通知，probe 是测试记录，
不是 tiny-llm 原生登记表或显存 oracle。OBS/真实 GPU HTTP/持续 lane 的批准与验收
仍独立待办，不能凭这组 CPU 测试关闭它们。测试占用临时本地端口并配置四个 runtime worker；
显式 shutdown/join 回收 listener，panic 的 Drop 兜底关闭许可并停止服务。
