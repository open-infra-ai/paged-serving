# PSRV-CANCEL-BP 设计包：请求取消与有界背压

> 对应任务：PSRV-P0-001（主动取消）、PSRV-P0-002（有界背压）、PSRV-P0-003
> （指标语义冻结）。评审包来源：`ai-infra-interview-prep/L3_L4_DESIGN_REVIEW_PACKAGES.md` §7。
> 本文是设计冻结稿，未经 reviewer 批准前不得修改生产实现。

## 1. Decision summary

- **Chosen design**：
  1. 取消：`Submission` 携带 `watch::Sender<bool>` 取消信号；handler 侧持有
     `RequestGuard`（RAII），任何消费路径退出（receiver drop、SSE stream drop、
     unary future abort、n>1 部分准入失败）都触发 `send(true)`；engine loop
     每步检查 `has_changed()`/`borrow()` 主动取消，不再依赖 `tx.send()` 失败
     的偶然时点。
  2. 背压：engine → 单请求事件通道改为**有界 mailbox**（容量可配，默认 64），
     engine loop 用 `try_send`；overflow → 取消该慢消费者请求并排出稳定终态。
     n>1 fan-in 用有界 channel + 专用转发 task `send().await`（允许 await）。
  3. 终态：`RequestState` 新增 `Cancelled` 变体（与 `Failed` 区分），
     `CompletedRequest` 增加 `cancelled: bool`；metrics 新增
     `paged_requests_cancelled_total`。
- **Why**：send-failure 检测只在下一次非空 chunk 时才生效——pending/prefill
  断连、HF 解码空窗、unary abort 都会让请求继续占用 KV block 与调度槽位。
  unbounded channel 让慢/僵尸 consumer 无界堆积 `Chunk(String, ...)`，
  内存无上界且违反 G4「使用 unbounded channel」拒绝条件。
- **Rejected alternatives**：
  - 只保留 send-failure 被动取消（§7.2 明确禁止只依赖它）；
  - engine loop 对有界 channel `send().await`（会阻塞整个 engine loop，
    违反 §7.3 策略 1 的前置证明）；
  - `RequestState::Failed` + 字符串前缀区分 cancelled（字符串匹配脆弱，
    metrics 无法可靠区分）；
  - fan-in 用 unbounded（内存无上界，正是本任务要消除的问题）。
- **Explicit non-goals**：
  - 不实现 preemption/chunked prefill/prefix cache；
  - 不改 OpenAI API 响应形状（SSE chunk 格式、错误信封不变）；
  - 不引入 request timeout 新特性（本包只定义超时若存在时走 cancel 路径）；
  - metrics sampler 周期采集属 PSRV-P1-003，本包只冻结其 channel 策略。

## 2. Base evidence (G0)

- **Repository**：`open-infra-ai/paged-serving`
- **Base branch**：`master` @ `4986a05019f81ca53d1b4b909e9c7d4fda142a04`
- **Dirty state**：clean
- **代码锚点**：
  - `src/server.rs:189` — 每请求 `mpsc::unbounded_channel::<RequestEvent>`；
  - `src/server.rs:657` — submission queue 已有界
    `mpsc::channel(SUBMISSION_QUEUE_CAPACITY)`；
  - `src/server.rs:720-731` — 断连检测：engine loop 在 `tx.send()` 失败时
    `cancel_request`（**被动**，依赖下一次非空 chunk）；
  - `src/server.rs:1280` — n>1 fan-in `unbounded_channel::<(usize, RequestEvent)>`；
  - `src/server.rs:1568-1572` — `generate_many` 首个失败时 `set.abort_all()`
    （abort task → receiver drop → 仍靠被动 send-failure）；
  - `src/engine.rs:603` — `cancel_request` → `scheduler.cancel_by_request_id`；
  - `src/scheduler.rs:440` — `cancel_by_request_id` → `fail_by_request_id`，
    覆盖 pending_queue / prefill_sequences / decode_sequences，KV 随
    `fail_sequence` 释放；终态为 `Failed("request cancelled: ...")`；
  - `src/main.rs:228,282-305` — axum graceful shutdown（停收新连接、排空在途）。
- **现有 tests**：
  - `tests/server_integration.rs:684` `test_client_disconnect_cancels_generation`
    （被动路径）；
  - `src/engine.rs` 内联测试：`cancel_request` 停生成、`sequences_finished`
    on cancel、decoder state 清理（:874/:1261/:1362）。
- **现有 metrics**（`src/server.rs:41-100`）：`paged_requests_total`、
  `paged_errors_total`、`paged_inflight_requests`（handler lifetime，
  `InflightGuard` RAII）、`paged_streaming_requests_total`；engine 侧
  active_sequences / kv_utilization / completed / failed / tokens_generated。
- **已知缺口**：
  - pending/prefill 阶段 client 断连 → 请求存活到首个 chunk send 才取消；
  - HF decoder 暂无文本（空 chunk）期间断连 → 不检测；
  - unary `generate()` 的 events receiver 随 handler abort drop → 被动；
  - n>1 streaming 第 k 个准入失败 `return err` → 已准入 k-1 个靠被动回收；
  - `completions`/`chat_completions` 中 malformed JSON 返回 400 但不计
    `errors_total`（指标口径待 §7.4 冻结）。
- **GPU/外部依赖**：本任务纯 tokio/host 侧，无 GPU、无模型依赖；
  tiny-llm backend 经 `EngineBackend` trait 抽象，取消经 scheduler 层，
  backend sequence 释放由既有 `free`/`drop` 路径承担（P0-001 验收要求
  backend sequence 回基线，测试用 probe 验证）。
- **Unknown（需在实现前关闭或由 reviewer 接受）**：
  - tiny-llm backend 的 sequence free 是否幂等（重复 cancel → 重复 release）；
  - `n>1` fan-in 中单个 candidate overflow-cancel 后聚合流终态形状
    （设计 §5 定为：任一 candidate 终态失败 → 整体 error chunk）。

## 3. Request 状态机与所有权（§7.1）

```text
received → admitted → pending → prefill → decode → completed | failed | cancelled
```

| 状态 | HTTP owner | consumer owner | scheduler seq | KV blocks | backend seq | event channel | cancel 信号 | 终态事件 | cleanup owner |
|---|---|---|---|---|---|---|---|---|---|
| received | handler future | — | — | — | — | — | token 未建 | — | handler |
| admitted | handler | guard | pending_queue | — | — | tx/rx 已建 | `watch::Sender` 存活 | — | guard |
| pending | handler | guard/SSE | pending_queue | 0 | 未分配 | tx→mailbox | token | Done(fail/cancel) | engine loop |
| prefill | handler | guard/SSE | prefill_sequences | 已分配 | 已建 | 同上 | token | 同上 | engine loop |
| decode | handler | guard/SSE | decode_sequences | 已分配 | 活跃 | 同上 | token | 同上 | engine loop |
| completed | SSE drain | — | 已释放 | 已归还 | 已 free | Done 已发 | — | Done(success) | engine loop |
| failed | SSE drain | — | 已释放 | 已归还 | 已 free | Done 已发 | — | Done(error) | engine loop |
| cancelled | SSE drain | — | 已释放 | 已归还 | 已 free | Done 已发 | token 已发 | Done(cancelled) | engine loop |

**Invariants**（与 §7.1 要求一一对应）：

1. 每个已准入 request 恰好一个 terminal state —— `fail_sequence`/complete 后
   sequence 从三张表移除，Done 只经 `dispatch_completed` 发一次；
2. 每个 sequence 恰好一次 release —— cancel 是幂等查找（`cancel_by_request_id`
  找不到返回 false），KV 归还发生在 `fail_sequence` 单点；
3. terminal 后不再产生 chunk —— 终态后 request 不在 decode 表，无新 chunk；
4. cancel/timeout/disconnect 后资源回基线 —— 测试用 BlockPool/metrics
   前后对比验证（§7.5）。

## 4. Cancellation 触发矩阵（§7.2）

| 触发 | 当前行为 | 目标行为 |
|---|---|---|
| client 首 token 前断开 | 被动：首个 chunk send 失败才取消 | `events_rx` drop → guard `cancel()` → engine 下一步检出 |
| HF 空文本窗口断开 | 被动：下一个非空 chunk 才检测 | 同上，与 chunk 产生解耦 |
| 多 chunk 后断开 | 被动 send-failure → cancel（已测） | 主动 token + 保留 send-failure 兜底 |
| unary handler abort | `generate()` future drop → rx drop → 被动 | rx 由 guard 持有，guard Drop 发 cancel |
| `n>1` 第 k 个准入失败 | `return err` → 已准入 stream drop → 被动 | 对已准入 token 显式 `cancel()` 后返回 |
| server shutdown | graceful drain 只在途自然完成，SSE 流可任意长 → shutdown 可无限挂起（**原稿"guard 随连接 drop"有误**：graceful shutdown 不主动 drop 连接） | shutdown 信号 → engine loop 广播取消全部在途请求（`Done{cancelled}`），流终止、连接关闭，排空有界 |
| engine/backend error | `Failed` 终态 + Done（已有） | 不变 |
| request timeout | 不存在 | 若未来加入，走同一 cancel 路径 |
| channel overflow | 不存在（unbounded） | overflow → 对该请求发 cancel + Done(cancelled)，
  不影响其他请求 |

**机制**：`Submission` 新增 `cancel: watch::Receiver<bool>`（初始 false）。
handler 侧 `RequestGuard { cancel_tx }` 实现 `Drop → let _ = tx.send(true)`。
engine loop 在 `step_events` 前后各检查一次 `waiters` 中
`rx.has_changed()` 的请求。**判定口径：`has_changed() != Ok(false)` 即取消**
——`Ok(true)` 为显式 cancel，`Err` 为 sender 全部 drop（owner 已消失，
同样必须取消，覆盖 guard 未正常触发 cancel 的异常路径）。
`admit_submission` 前也检查一次：submission 排队期间已断连的请求不再准入，
省一次 prefill 分配。保留 `tx.send().is_err()` 检测作为兜底
（rx 被 drop 但 guard 未覆盖的路径）。

**post-terminal 安全性**：Done 投递后 guard Drop 仍会 `send(true)`——
此时 waiter 已移除、`cancel_by_request_id` 返回 false，是无害 no-op。
request_id 不复用，无串扰。

`watch` 语义：单值、覆盖式、多 receiver 可观察；n>1 每候选独立 token。
`CancellationToken`（tokio-util）等价但引入新依赖——选 `watch`（现有
tokio "sync" feature 已含）。

**shutdown 广播**：`main` 持有一个 engine 级 `watch::Sender<bool>`；
`shutdown_signal()` 触发后置位。engine loop 检出后遍历 `waiters`
全部 `cancel_request` + 投递 `Done{cancelled}`，SSE 流随即终止、连接
关闭，axum graceful drain 因而有界（不再被任意长的流挂起）。

## 5. Bounded channel 策略（§7.3）

| channel | capacity | element 上界 | producer 可否 await | overflow 行为 | 全局影响 |
|---|---|---|---|---|---|
| engine → 单请求 chunk | `config.event_channel_capacity`，默认 64 | `Chunk`：单步 detok 文本（≪1KB）+ logprobs | **否**（`try_send`，engine loop 不得阻塞） | `Full` → cancel 该请求（见下方终态通道） | 慢消费者只丢自己 |
| engine → 单请求 **终态** | `oneshot`（恰 1 条，固有上界） | `CompletedRequest` 定长 | 否（`send`，永不阻塞/拒绝） | 不可能 overflow | 终态必达 |
| n>1 child → fan-in | `n × event_channel_capacity` | chunk/Done 标记 + usize index | **可**（专用转发 task，`send().await`） | 上游 mailbox 先溢出 → child 被 cancel → task 退出 | 背压逐候选传导 |
| submission queue | 1024（现状，冻结） | `Submission` 定长 | 可（handler await = 准入延迟） | 队列满 → `send().await` 背压到 handler（engine stall 时 handler 悬挂而非 429——记录为已知行为，准入层 overload→429 仍发生在 admit 点） | — |
| metrics sampler（P1-003 预留） | 1（`watch`/`latest`） | 快照定长 | 否 | coalesce（只保留最新） | 无背压 |

**终态带外通道（评审修订）**：`Done` 不能与 `Chunk` 共用有界 mailbox——
overflow-cancel 时 mailbox 已满，`try_send(Done)` 同样失败，client 只会
观察到 sender-drop 得到泛化错误而非精确终态。waiter 改为
`{ events_tx: mpsc::Sender<Chunk>, done_tx: oneshot::Sender<CompletedRequest>,
cancel_rx }`；`dispatch_completed` 与 overflow-cancel 均经 `done_tx` 投递
（oneshot 单发，无背压问题；send 失败即 receiver 已 drop，静默丢弃同现状）。
SSE/转发侧对 `events` 与 `done` `select!`；done 到达后忽略残余 chunk。

- **失败终态形状**：overflow-cancel 的 Done 携带 `cancelled: true` +
  `error = "slow consumer: event channel overflow"`；SSE 端收到 error
  envelope + `[DONE]`，与现终态形状一致。
- **n>1 单候选 overflow**：该候选 Done(cancelled) 到达 fan-in → 聚合法
  维持现状（任一候选终态失败 → 整体 error chunk）。
- **内存上界**：单请求 ≤ 64 × ~1KB ≈ 64KB；全局 ≤ 64KB × max_concurrent。
- **为什么不开新 forwarding task**（§7.3 策略 2）：consumer 慢时问题只是
  从 engine mailbox 转移到 forwarding task 的输出端，仍要回答同一 overflow
  问题；直接 bounded mailbox + overflow-cancel 少一个移动部件。

## 6. 指标语义冻结（§7.4，P0-003 范围）

| 问题 | 冻结口径 |
|---|---|
| `inflight` 含义 | **response body lifetime**（修订）：现状 `InflightGuard` 随 handler future 返回即 drop，而流式路径构建 SSE stream 后不 await 生成——gauge 对在途 streaming 恒不计数。改为流式把 guard 移入 stream 体内（stream 结束/drop 时递减）、unary 维持 handler 作用域；统一语义 = "请求仍在被服务"。指标名不变，HELP 写明 |
| `requests_total` 对 n | 一个 API request 计 1（不乘 n） |
| malformed JSON | **计** `errors_total`（现状不计 → 修，breaking 记录 CHANGELOG） |
| 429 / admission failure | 计 `errors_total`（现状已计，冻结） |
| SSE terminal error | 计 `errors_total`（generation failure 路径已计） |
| cancelled vs failed | 独立：`paged_requests_cancelled_total` 新增 counter；cancelled **不计入** `errors_total`（client 主动行为非服务故障） |
| active_sequences / KV | engine loop 每步结束后快照（现状，冻结时点） |
| HELP 文本 | 每个指标补 `# HELP` 行，单位精确到 request/token/ratio |

## 7. 测试方式（§7.5/G6）

不以 sleep 为主；用确定性原语：

1. **状态控制**：`oneshot`/barrier 把 engine 停在 pending（占满 KV 预算）
   与 decode（backend probe 可控 step）；
2. **主动取消**：drop `events_rx` / drop SSE stream / abort handler task 后，
   用 engine probe 断言「下一次 loop 迭代即 cancel」，不依赖 chunk；
3. **overflow**：capacity=1 配置下 server 连发 >1 chunk 且 consumer 不读 →
   断言该请求 Done(cancelled)、其他请求不受影响；
4. **资源基线**：BlockPool free-list 与 active_sequences 在
   cancel/timeout/disconnect 前后对比（复用既有测试模式）；
5. **exactly-once**：proptest 随机交错 cancel/complete/drop，断言
   terminal ≤ 1 且 release 恰好一次；
6. **n>1 准入失败**：第 2 个 submit 注入 admission error → 断言第 1 个
   candidate 的 cancel token 已置位、其 seq/KV 回基线；
7. **fan-in**：单候选 overflow-cancel → 聚合流收到 error + 终态。

## 8. G0-G8 门禁自评

| 门禁 | 结论 |
|---|---|
| G0 事实 | §2 列全；unknowns 已声明 |
| G1 API/ABI | 无 C ABI 变化；`Submission`/`RequestEvent` 为 crate 内部类型；`EngineConfig` 增 `event_channel_capacity`（有默认值，向后兼容） |
| G2 数据布局 | 无数值布局变化 |
| G3 所有权 | §3 表 + invariant；guard Drop 是唯一 cancel 入口，幂等 |
| G4 并发 | engine loop 永不 await 有界 channel（try_send）；转发 task 可 await（专用）；无 unbounded channel 残留 |
| G5 错误语义 | overflow → cancelled 终态（非 success）；malformed JSON 计入 errors（breaking 记录）；SSE 终态形状不变 |
| G6 correctness | §7 矩阵：probe/barrier/proptest，非 sleep；CPU 可测，无需 GPU |
| G7 性能 | 非性能任务。预期：内存上界化；`try_send` 为 O(1)。不做 benchmark 声明，正式 serving 矩阵属 PSRV-P1-004 |
| G8 拆分 | PR-1 本设计文档；PR-2 取消所有权（P0-001）；PR-3 有界 channel（P0-002）；PR-4 指标语义（P0-003）。回滚：revert 单 PR 即恢复 unbounded+被动行为 |

## 9. 开放问题（待 reviewer）

1. `RequestState::Cancelled` 新变体 vs `Failed`+前缀：本文选新变体
   （语义干净、metrics 可靠区分）；若 reviewer 倾向最小 diff 可退回
   前缀方案，但需接受 metrics 字符串匹配。
2. `event_channel_capacity` 默认值 64 是否合理：单步 chunk ≪1KB、
   SSE consumer 为 axum 写出（通常远快于 decode step），64 提供
   ~64 step 抖动缓冲；可调。
3. overflow 终态 error 文案与 `type` 字段值（`internal_error` vs
   新 `cancelled` 类型）——影响 API 可观察面，需冻结。

## 10. 评审修订记录（self-review 2026-09-15）

1. **终态带外通道**：`Done` 原设计与 `Chunk` 共用有界 mailbox——
   overflow-cancel 时 mailbox 已满、终态无法投递。改为 `oneshot` 终态
   通道（§5）。
2. **`inflight` 口径修正**：原冻结 handler lifetime 对流式恒不计数；
   改为 response body lifetime（§6）。
3. **shutdown 语义**：原稿误称"guard 随连接 drop"——graceful shutdown
   不主动断开连接，长 SSE 会使 shutdown 无限挂起。改为 shutdown 信号
   广播 cancel-all（§4 触发表）。
4. **`watch` Err 分支**：`has_changed()` 在 sender 全 drop 时返回 `Err`；
   冻结为 `!= Ok(false)` 即取消，覆盖 guard 异常路径（§4）。