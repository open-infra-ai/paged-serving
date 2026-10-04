# Paged-Serving 后续开发计划

> **历史执行档案**：本文记录 v0.2.0 正确性修复开始前的基线和任务拆解，文中的
> 测试数量、失败状态与“冻结”步骤不代表当前 HEAD。当前状态与下一步以
> [`README.md`](README.md) 和 [`ROADMAP.md`](ROADMAP.md) 为准；保留本文是为了让
> 修复决策和验证过程可追溯。

> 本文档基于仓库 `HEAD 58778e1` 的代码审查结论编写。
> 目标读者：一个按任务执行的 AI 编程模型（包括便宜的模型）。
> 每个任务都应独立提交、独立验证，不要一次性重写整个项目。

---

## 0. 执行规则（重要）

1. **一次只做一个 Task**。完成并验证后再做下一个。
2. **先跑基线命令**，确认当前状态：
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test
   ```
   当前预期：`cargo test` 通过（185 个测试），fmt 和 clippy 失败。
3. 每个 Task 完成后至少运行：
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test
   ```
4. 不要顺手重构无关代码；如果发现计划外问题，在任务说明中记录为 NOTE，不要扩大修改范围。
5. 涉及 `tiny-llm` 的测试默认跳过；没有 `TINY_LLM_DIR` / `TINY_LLM_MODEL` 环境时，**不要**运行 `--all-features`。
6. 每个 Task 完成后用独立 commit，建议格式：
   ```
   fix(scheduler): memory watermark leaves decode growth reserve
   ```
7. 计划中的“验收标准”必须逐条满足，未满足不允许标记任务完成。

---

## 1. 当前基线

| 项目 | 状态 |
|---|---|
| 默认测试 | `cargo test`：185 个测试全部通过 |
| 格式检查 | `cargo fmt --check` 失败 |
| Clippy | `cargo clippy --all-targets -- -D warnings` 失败 |
| tiny-llm feature | 未设置 `TINY_LLM_DIR` 时 `--all-features` 链接失败（预期，但 CI/文档说法不一致） |
| 主要正确性风险 | 内存水位线缺失、pending 队头阻塞、执行输出无校验、后端 KV 生命周期未闭环 |

---

## 2. 任务总览

### P0：正确性与工程基线（必须完成）

| ID | 任务 | 规模 | 依赖 |
|---|---|---|---|
| T0 | 修复 fmt / clippy，恢复 CI 绿色 | S | 无 |
| T1 | 执行输出契约校验，坏后端不得卡死请求 | M | T0 |
| T2 | 调度器内存水位线 + decode 增长预留 | M | T0 |
| T3 | 修复 pending 队头阻塞（HOL） | S | T2 |
| T4 | 后端序列生命周期钩子 + tiny-llm KV 释放 | M | T0 |
| T5 | 超时重试必须声明幂等性 | S | T0 |
| T6 | 浮点参数 NaN 校验 | S | T0 |
| T7 | Unicode stop 序列字节偏移修复 | S | T0 |
| T8 | 文档与代码事实对齐 | M | T0–T7 |

### P1：mini-vLLM 故事补全（建议完成）

| ID | 任务 | 规模 | 依赖 |
|---|---|---|---|
| T9 | Chat Completions 应用真实 chat template | M | T8 |
| T10 | 引擎指标接入 /metrics + CB on/off benchmark | M | T8 |
| T11 | tiny-llm 策略 1：分页 KV C ABI 与适配器 | L | T4，需 tiny-llm 仓库配合 |
| T12 | 可选：BPE 安全增量解码或明确降级 | M | T8 |

### P2：冻结与发布

| ID | 任务 | 规模 | 依赖 |
|---|---|---|---|
| T13 | 最终验收、版本发布、冻结声明 | S | P0 全部 |

---

## 3. P0 任务详细说明

### T0：修复 fmt / clippy，恢复 CI 绿色

**问题**
- `src/tokenizer.rs`：`token_to_id()` 已返回 `u32`，多余的 `as u32` 触发 `clippy::unnecessary_cast`。
- `benches/concurrency_benchmark.rs`：`GPUExecutorTrait` 未使用。
- `tests/concurrency_stress.rs`：手写取模判断触发 `clippy::manual_is_multiple_of`。
- 当前 `.github/workflows/ci.yml` 实际要求 fmt + clippy，所以 CI 是红的。

**实施步骤**
1. `src/tokenizer.rs` 的 `first_token_id` 改为：
   ```rust
   fn first_token_id(inner: &Tokenizer, names: &[&str]) -> Option<u32> {
       names.iter().find_map(|name| inner.token_to_id(name))
   }
   ```
2. 删除 `benches/concurrency_benchmark.rs` 导入中的 `GPUExecutorTrait`。
3. `tests/concurrency_stress.rs`：
   ```rust
   // before
   if self.counter % self.fail_mod == 0 {
   // after
   if self.counter.is_multiple_of(self.fail_mod) {
   ```
4. 运行 `cargo fmt --all`，让 rustfmt 处理导入排序和换行。
5. 运行 `cargo clippy --all-targets -- -D warnings`，修复剩余 warning，直到退出码为 0。

**验收**
```bash
cargo fmt --all -- --check   # 无输出，退出码 0
cargo clippy --all-targets -- -D warnings  # 无 warning/error，退出码 0
cargo test                   # 仍为 185 passed
```

**不要做**
- 不要在本任务给 CI 加 `--all-features`。
- 不要修改 CHANGELOG 的 CI 描述（留给 T8）。

---

### T1：执行输出契约校验，坏后端不得卡死请求

**问题**
`BatchExecutionPipeline::execute()` 对后端返回的 `ExecutionOutput` 完全信任。一个返回空输出的后端会让 `engine.step()` 永远返回空结果，请求永久卡在调度器中，`run()` 无限循环。

**实施步骤**
1. 在 `src/execution_pipeline.rs` 增加两个私有校验函数：
   ```rust
   fn validate_batch_shape(batch: &ExecutionBatch) -> Result<(), EngineError>
   fn validate_execution_output(
       batch: &ExecutionBatch,
       output: &ExecutionOutput,
   ) -> Result<(), EngineError>
   ```
2. `validate_batch_shape` 至少检查：
   - `input_tokens.len() == positions.len()`
   - `seq_ids.len() == seq_lens.len() == is_prefill.len() == context_lens.len() == block_tables.len()`
   - `seq_lens.iter().sum::<u32>() as usize == input_tokens.len()`
   - 任意 `seq_len == 0` 或对应 `block_tables` 为空 → 错误
   - 错误统一返回：
     ```rust
     Err(EngineError::BackendError(format!("invalid execution batch: ...")))
     ```
3. `validate_execution_output` 至少检查：
   - `output.next_tokens.len() == output.seq_ids.len()`
   - `output.seq_ids.len() == batch.seq_ids.len()`
   - `output.seq_ids` 无重复
   - `output.seq_ids` 的集合与 `batch.seq_ids` 集合完全一致（顺序可以不同，但必须一一对应）
   - `output.logprobs` 为空，或长度等于 `output.seq_ids.len()`
   - 不合法时返回：
     ```rust
     Err(EngineError::BackendError(format!("malformed execution output: ...")))
     ```
4. `BatchExecutionPipeline::execute()`：
   - 构建 batch 后先 `validate_batch_shape(&execution_batch)?`
   - 后端返回 `Ok(output)` 后先 `validate_execution_output(&execution_batch, &output)?` 再返回。
5. 在 `src/gpu_executor.rs` 的 `GPUExecutorTrait` 文档中明确写出输出长度与 seq id 集合契约。
6. 增加测试：
   - `src/execution_pipeline.rs` 单元测试：合法输出通过；空输出、重复 seq id、少 token、logprobs 长度错误均返回 `Err`。
   - `tests/integration_tests.rs` 增加 `test_malformed_backend_output_fails_request_and_terminates`：
     - 自定义 executor 返回 `ExecutionOutput::default()`；
     - 提交 1 个请求；
     - 调用 **一次** `engine.step()`（不要用 `run()`，避免修复前测试挂死）；
     - 断言该请求以失败终态排出，且 `engine.has_pending_work() == false`。

**验收**
```bash
cargo test execution_pipeline
cargo test --test integration_tests test_malformed_backend_output_fails_request_and_terminates
cargo test
```

---

### T2：调度器内存水位线 + decode 增长预留

**问题**
`Scheduler::schedule()` 启动 pending prefill 时只检查 `can_allocate`，不检查启动后是否超过 `memory_threshold`，也不为 decode 下一步增长预留块。最小复现：64 个请求、每个 prompt 正好 1 个 block、64 块池、threshold=0.9 → 第一步 util=1.0，第二步 32 个请求 `OutOfBlocks` 失败。

**目标行为**
- 启动新 prefill 前同时满足：
  1. **高水位线**：启动后 `used_blocks / total_blocks <= memory_threshold`；
  2. **即时 decode reserve**：启动后剩余空闲块数，不少于“本步已调度序列 + 候选序列”在下一步可能需要的增长块数。
- 预算不足的候选请求延后到后续步骤，而不是失败或把池子打满。

**实施步骤**
1. 在 `src/scheduler.rs` 增加私有辅助函数：
   ```rust
   /// 一个 prefill 在首 token 生成后的下一步，是否需要多分配 1 个 block。
   fn prefill_needs_decode_reserve(&self, prompt_tokens: u32) -> bool {
       let prompt_blocks = self.config.blocks_for_tokens(prompt_tokens);
       self.config
           .blocks_for_tokens(prompt_tokens.saturating_add(2))
           > prompt_blocks
   }

   /// 当前已调度/在跑的序列在“本步执行完成后、下下步 grow 时”需要的新增块数。
   fn next_step_growth_reserve(&self) -> u32 {
       self.prefill_sequences
           .values()
           .chain(self.decode_sequences.values())
           .filter(|seq| {
               let after_this_step = seq.context_len().saturating_add(1);
               self.config
                   .blocks_for_tokens(after_this_step.saturating_add(1))
                   > seq.logical_blocks.len() as u32
           })
           .count() as u32
   }

   /// 启动一个 prefill 是否安全。
   fn has_prefill_budget(&self, blocks_needed: u32, prompt_tokens: u32) -> bool {
       let stats = self.kv_cache.get_memory_stats();
       let used_after = stats.used_blocks.saturating_add(blocks_needed);
       let free_after = stats.free_blocks.saturating_sub(blocks_needed);
       let reserve = self
           .next_step_growth_reserve()
           .saturating_add(u32::from(self.prefill_needs_decode_reserve(prompt_tokens)));
       let watermark_ok = (used_after as f32)
           <= (stats.total_blocks as f32 * self.config.memory_threshold).floor();
       watermark_ok && free_after >= reserve
   }
   ```
2. 在 `schedule()` 的 pending 循环中，`try_start_prefill` 之前增加：
   ```rust
   if !self.has_prefill_budget(blocks_needed, prefill_tokens) {
       self.pending_queue.push_back(PendingRequest { seq_id, request });
       continue; // 先跳过，尝试后面的小请求；T3 会统一这个扫描逻辑
   }
   ```
3. `try_start_prefill` 内部保留 `can_allocate` 作为最终防线。
4. `update_memory_pressure()` 保留，用于 `add_request` 的提交侧拒绝。
5. 增加回归测试（放 `tests/integration_tests.rs`）：
   `test_memory_pressure_leaves_decode_growth_reserve`：
   - `block_size=16, max_num_blocks=64, max_batch_size=64, max_num_seqs=64, max_total_tokens=2048, memory_threshold=0.9`
   - 自定义 executor 恒输出非 EOS token；
   - 使用 `SimpleTokenizer::without_special_tokens()`；
   - 提交 64 个请求，prompt 为 16 个 ASCII 字符（正好 1 block），`max_tokens=2`；
   - 跑完 `engine.run()`；
   - 断言 **64 个全部成功**，运行过程中任意时刻 `memory_utilization <= 1.0`；
   - 断言第一步后利用率为 **0.5**（最多启动 32 个 prefill），且没有请求因 `OutOfBlocks` 失败。
6. 在 `src/scheduler.rs` 单元测试中补：
   - `has_prefill_budget` 在“剩余 1 块且候选需要 decode reserve”时返回 false；
   - `next_step_growth_reserve` 对刚好整块的 prefill/decode 序列计数正确。

**验收**
```bash
cargo test scheduler
cargo test --test integration_tests test_memory_pressure_leaves_decode_growth_reserve
cargo test --test concurrency_stress
```

**注意**
- 该策略只解决“下一步马上需要增长”的 OOM；本项目明确不实现抢占，长期最坏情况仍可能 OOM。这个边界要写进 README/T8。
- 不要在本任务引入 chunked prefill 或 preemption。

---

### T3：修复 pending 队头阻塞（HOL）

**问题**
`schedule()` 对装不下的 pending 请求执行 `push_front + break`，导致大 prefill 永远挡住后面的小 prefill。复现：8 个长 decode 占满 batch 预算，96 token 的大 pending 后面跟着 1 token 的小请求，20 步后小请求仍未开始。

**实施步骤**
1. 将 pending 处理从 `while let Some(pending) = pop_front()` 改为**按当前队列长度扫描一轮**：
   ```rust
   let pending_count = self.pending_queue.len();
   for _ in 0..pending_count {
       let Some(pending) = self.pending_queue.pop_front() else { break };
       // ...
   }
   ```
2. 对以下“本步装不下”的情况，改为 `push_back` 后 `continue`（不要 `break`）：
   - `num_sequences >= max_batch_size`
   - `total_tokens + prefill_tokens > max_total_tokens`
   - `try_start_prefill` 因内存预算/分配失败返回 Err
3. 只有请求本身非法（`prefill_tokens > max_total_tokens` 或 `blocks_needed > max_num_blocks`）才失败并继续。
4. `total_tokens` 和 `num_sequences` 更新逻辑不变。
5. 增加测试：
   - `src/scheduler.rs` 单元测试 `test_pending_queue_skips_large_prefill_for_smaller_one`。
   - `tests/integration_tests.rs` 增加 `test_small_pending_request_not_blocked_by_large_one`，使用 T2 场景类似配置：
     - 8 个长 decode（max_tokens 较大）占住 batch；
     - 提交 96 token 大 pending，再提交 1 token 小 pending；
     - 断言小请求在 5 个 step 内完成或进入 prefill，而大请求仍可等待。
6. 运行原内存压力测试，确保 T2 不受影响。

**验收**
```bash
cargo test scheduler
cargo test --test integration_tests test_small_pending_request_not_blocked_by_large_one
cargo test --test concurrency_stress
```

**说明**
这个策略放弃了严格 FCFS：只保证“本步能装下”的请求大致按到达顺序处理。项目不实现 chunked prefill，跳过是正确取舍；T8 文档要写明。

---

### T4：后端序列生命周期钩子 + tiny-llm KV 释放

**问题**
`GPUExecutorTrait` 只有 `execute()`。引擎侧释放逻辑块后，执行器不知道序列结束。`TinyLlmExecutor` 从未调用 `tinyllm_free_sequence`，真实后端 slot 会耗尽。

**实施步骤**
1. `src/gpu_executor.rs` 的 trait 增加默认空实现：
   ```rust
   /// 引擎在序列到达终态（完成/失败/取消）并释放逻辑 KV 块后调用。
   /// 后端应在此时释放该序列占用的物理 KV 资源。
   fn sequences_finished(&mut self, _seq_ids: &[SeqId]) {}
   ```
2. `src/scheduler.rs`：
   - 将 `completed_requests: Vec<Request>` 改为 `Vec<(SeqId, Request)>`；
   - 保留现有 `get_completed(&mut self) -> Vec<Request>` 兼容测试和调用方；
   - 新增：
     ```rust
     pub fn take_completed_with_seq_ids(&mut self) -> Vec<(SeqId, Request)> {
         std::mem::take(&mut self.completed_requests)
     }
     ```
   - 修改 `get_request_by_id` 中对 completed 的查找；
   - 修改 `complete_sequence` / `fail_sequence` / 直接失败 pending 的 push 点，带上 `seq_id`。
3. `src/execution_pipeline.rs` 增加：
   ```rust
   pub fn sequences_finished(&mut self, seq_ids: &[SeqId]) {
       self.gpu_executor.sequences_finished(seq_ids);
   }
   ```
4. `src/engine.rs` 的 `collect_completed_requests`：
   - 改用 `take_completed_with_seq_ids`；
   - 取出所有 `seq_id`；
   - 在构造 `CompletedRequest` 前后调用一次 `self.execution_pipeline.sequences_finished(&finished_seq_ids)`。
5. `src/tiny_llm_executor.rs` 实现：
   ```rust
   fn sequences_finished(&mut self, seq_ids: &[SeqId]) {
       for &sid in seq_ids {
           let sid_i = sid as c_int;
           if self.allocated.remove(&sid_i) {
               let rc = unsafe { symbols::tinyllm_free_sequence(self.handle, sid_i) };
               if rc != 0 {
                   log::warn!("tinyllm_free_sequence({sid}) failed rc={rc}");
               }
           }
       }
   }
   ```
6. 测试：
   - `src/engine.rs` 增加 `RecordingExecutor`：记录 `sequences_finished` 收到的 seq id；
   - 覆盖三个路径：正常完成、执行失败、`cancel_request()` 取消；
   - `tests/tiny_llm_backend.rs` 在真实环境测试中增加“两波请求”：先跑 3 个完成，再跑 3 个完成，全部成功。没有真实模型时该测试仍按现有方式跳过。

**验收**
```bash
cargo test engine
cargo test --test integration_tests
cargo test
# 如有真实环境：
TINY_LLM_DIR=... TINY_LLM_MODEL=... cargo test --features tiny-llm --test tiny_llm_backend -- --nocapture
```

**注意**
- 默认 CPU/Mock 后端可以保持空实现，因为物理块按索引复用、写入时覆盖。
- 不要改 `get_completed` 的现有签名，避免大范围调用方改动。

---

### T5：超时重试必须声明幂等性

**问题**
`BatchExecutionPipeline` 对 `GpuTimeout` 无条件重试同一 batch。真实 CUDA kernel 超时时 KV 可能已部分写入，重放不安全。

**实施步骤**
1. `src/gpu_executor.rs` 的 `ExecutorCapabilities` 增加字段：
   ```rust
   pub struct ExecutorCapabilities {
       pub sampling: bool,
       /// 后端执行是否幂等：GpuTimeout 后重放同一 batch 是否安全。
       pub retry_safe: bool,
   }
   pub const GREEDY_ONLY: Self = Self { sampling: false, retry_safe: false };
   ```
2. `BatchExecutionPipeline::execute` 的重试条件增加：
   ```rust
   let retry_safe = self.gpu_executor.capabilities().retry_safe;
   // ...
   Err(EngineError::GpuTimeout)
       if retry_safe && retries < self.max_retry_attempts => { ... }
   Err(EngineError::GpuTimeout) => return Err(EngineError::GpuTimeout),
   ```
3. 更新 `src/engine.rs` 测试：
   - `TimeoutThenSuccessExecutor::capabilities()` 返回 `ExecutorCapabilities { sampling: false, retry_safe: true }`；
   - `AlwaysTimeoutExecutor::capabilities()` 同样返回 retry_safe=true，保留“重试耗尽”测试；
   - 新增 `NonRetrySafeTimeoutExecutor`（attempts 计数器，capabilities retry_safe=false），断言 `execute` 只被调用 1 次且请求失败。
4. `TinyLlmExecutor::capabilities()` 显式返回 `GREEDY_ONLY`（retry_safe=false）。
5. 更新 `GPUExecutorTrait` / `ExecutorCapabilities` 文档。

**验收**
```bash
cargo test engine
cargo test
```

---

### T6：浮点参数 NaN 校验

**问题**
- `EngineConfig::validate()` 中 `NaN` 的 `memory_threshold` 会通过校验，导致内存保护永久失效。
- `GenerationParams::validate()` 中 `top_p=NaN` 会通过范围校验。

**实施步骤**
1. `src/config.rs`：
   ```rust
   if !self.memory_threshold.is_finite()
       || self.memory_threshold <= 0.0
       || self.memory_threshold > 1.0
   { return Err(ConfigError::InvalidMemoryThreshold(self.memory_threshold)); }
   ```
2. `src/types/request.rs`：
   ```rust
   if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) { ... }
   if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 { ... }
   ```
3. 测试：
   - `src/config.rs` 单测增加 `NaN` / `infinity` 被拒绝；
   - `src/types/request.rs` 单测增加 `top_p=NaN`、`temperature=NaN` 被拒绝。
4. 运行 property tests，确认原有范围逻辑不回归。

**验收**
```bash
cargo test config
cargo test types
cargo test
```

---

### T7：Unicode stop 序列字节偏移修复

**问题**
`find_stop_sequence` 使用 `str::find` 得到**字节偏移**，`tokens_before_char` 却用 `chars().count()` 累加字符数。非 ASCII 文本（如中文）在 stop 序列之前时，偏移不一致，可能保留 stop 序列。

**实施步骤**
1. `src/engine.rs` 的 `tokens_before_char`：
   - 参数改名为 `byte_offset: usize`；
   - 累加长度从 `segment.chars().count()` 改为 `segment.len()`（字节数）。
2. 更新函数文档。
3. 增加 `src/engine.rs` 单元测试：
   - 构造一个测试 tokenizer，单 token 解码分别返回 `"好"`、`"A"`、`"B"`；
   - tokens 为 3 个，byte_offset 为 `"好".len()`（即 3）；
   - 断言返回 `1`（只保留 `"好"`）。
4. 增加一个 stop 序列端到端测试（如果现有测试工具足够）：输出 `好AB`，stop=`"AB"`，断言最终输出为 `"好"` 且 `finish_reason="stop"`。若测试 tokenizer 构造复杂，至少保留步骤 3 的单元测试，并在代码注释说明。
5. 检查 `apply_stop_sequences` 中 offset 变量命名一致。

**验收**
```bash
cargo test engine
cargo test --test server_integration test_completions_honors_stop_sequences
```

---

### T8：文档与代码事实对齐

**问题清单**
- `src/lib.rs`、`src/main.rs` 仍出现 “Hetero-Paged-Serving / Heterogeneous Inference System / mock compute backend”。
- CLI `--tokenizer` 帮助写 Qwen2.5 词表 151936，但 `HuggingFaceTokenizer::vocab_size()` 实际是 151665。
- README 快速开始使用中文输入，但默认 SimpleTokenizer 只支持 ASCII，中文会变 UNK。
- README/CHANGELOG 声称 “true token-level SSE streaming”，对 HF tokenizer 不成立（`BufferedDecoder` 在 finish 时才输出）。
- CHANGELOG 声称 CI 跑 `--all-features`，实际 ci.yml 没有。
- README 内存压力描述与实现细节不一致（提交即拒绝，而非“pending 保留”的新语义需要重新表述）。

**实施步骤**
1. `src/lib.rs`：模块文档改为 `paged-serving`、CPU reference backend，不再叫 Hetero/mock。
2. `src/main.rs`：
   - `#[command(about = "...")]` 改为 paged-serving + CPU reference；
   - 启动日志与打印去掉 Heterogeneous；
   - `--tokenizer` 帮助改为“完整有效词表 151665；GGUF embedding 可能为 151936 并含 padding 行”。
3. `README.md`：
   - 快速开始示例改成 ASCII 文本（如 `"Hello, world!"`）；
   - 增加一句：默认 SimpleTokenizer 仅支持 ASCII，中文请用 `--tokenizer <tokenizer.json>`；
   - 在“性能边界”或“限制”中说明：HF tokenizer 流式模式当前为“完成前一次性文本 chunk”，SimpleTokenizer 才是逐 token；
   - 明确：本项目无抢占，内存压力策略是“拒绝新 prefill + 保留在途 decode + 预留即时 decode 增长块”；
   - 明确：Chat Completions 当前 chat template 状态（T9 前如实说明是 role 文本拼接）。
4. `CHANGELOG.md`：
   - 删除或改写“CI 现在运行 --all-features / bench smoke”的错误声明；
   - 把“true token-level streaming”改为限定范围后的表述；
   - Unreleased 段增加本次 P0 修复记录（随各任务逐步补充）。
5. `ROADMAP.md`：勾选已完成项，新增“P0 正确性修复”完成记录。
6. 全局检查：
   ```bash
   grep -R "Hetero\|mock compute backend\|151936\|all-features" -n src README.md CHANGELOG.md ROADMAP.md .github || true
   ```
   历史版本/CHANGELOG 旧记录可保留，但当前描述不得再有错误。

**验收**
```bash
cargo test --doc
cargo doc --no-deps
grep -R "Hetero" -n src README.md || true   # src/README 当前描述中应为空
```

---

## 4. P1 任务详细说明

### T9：Chat Completions 应用真实 chat template

**背景**
当前 `prepare_chat_request` 只做 `"role: content\n"` 拼接。HF 模型（如 Qwen2）需要 `<|im_start|>` 模板。

**实施步骤**
1. 在 `src/server.rs` 增加：
   ```rust
   fn qwen2_chat_prompt(messages: &[ChatMessage]) -> String
   fn build_chat_prompt(kind: &TokenizerKind, messages: &[ChatMessage]) -> String
   ```
2. Qwen2 模板规则：
   - 有 system 消息时：
     ```
     <|im_start|>system
     {content}<|im_end|>
     ```
   - 每个 user/assistant 消息：
     ```
     <|im_start|>{role}
     {content}<|im_end|>
     ```
   - 最后追加：
     ```
     <|im_start|>assistant
     ```
3. `prepare_chat_request` 中：
   - `TokenizerKind::HuggingFace` → `qwen2_chat_prompt`；
   - `TokenizerKind::Simple` → 保留现有 `role: content` 拼接（现有 server 测试不破坏）。
4. 增加单元测试：
   - `test_qwen2_chat_prompt_with_system`；
   - `test_qwen2_chat_prompt_without_system`；
   - 断言精确字符串。
5. 在 README/T8 文档中写明：目前 chat template 硬编码 Qwen2，其他模型需扩展。
6. 如有真实 `PSERV_TOKENIZER_JSON`，可加门控集成测试，验证 `HuggingFaceTokenizer::try_encode(chat_prompt)` 编码后的首 token 为 `<|im_start|>` 对应 ID。

**验收**
```bash
cargo test server
cargo test --test server_integration test_chat_completions
```

---

### T10：引擎指标接入 /metrics + CB on/off benchmark

**实施步骤**
1. `src/server.rs`：
   - 定义共享快照：
     ```rust
     #[derive(Default)]
     struct SharedEngineMetrics {
         active_sequences: AtomicU64,
         kv_utilization_bp: AtomicU64, // basis points，避免 f32 原子
         completed_requests: AtomicU64,
         failed_requests: AtomicU64,
         total_tokens_generated: AtomicU64,
     }
     ```
   - `engine_loop` 每步结束后用 `engine.get_metrics()` 更新快照。
   - `create_router_with_engine` 创建 `Arc<SharedEngineMetrics>`，放入 `AppState`。
   - `/metrics` 输出 Prometheus 格式，新增：
     ```
     paged_engine_active_sequences
     paged_engine_kv_utilization
     paged_engine_completed_requests
     paged_engine_failed_requests
     paged_engine_tokens_generated_total
     ```
2. 更新 `tests/server_integration.rs::test_metrics_endpoint_exposes_prometheus_counters`，断言新指标名存在。
3. `benches/concurrency_benchmark.rs` 增加一组 CB on/off 对比：
   - 相同 N 并发请求；
   - `cb_off`：`config.max_batch_size = 1`；
   - `cb_on`：`config.max_batch_size = N`；
   - 输出总排空时间和 p50/p95/p99。
   - 说明：使用 Mock 后端，只反映调度开销，不是 GPU 吞吐。
4. README 指标表更新。

**验收**
```bash
cargo test --test server_integration test_metrics_endpoint_exposes_prometheus_counters
cargo bench --bench concurrency_benchmark -- --test   # smoke
```

---

### T11：tiny-llm 策略 1：分页 KV C ABI 与适配器 ✅（2026-08-18，Batch D）

> 已完成：ABI v2（9 int config + `num_blocks`）、`TinyLlmExecutor` 默认策略 1
> （真实块表扁平化上传）、`PAGED_SERVING_TINY_LLM_STRATEGY=2` 回退策略 2。
> llama.cpp 逐 token 对齐与 3 并发 e2e（`qwen2_three_concurrent_paged_requests_match_llama_cpp`）
> 通过；tiny-llm 侧实现见其仓库 Batch D 提交。

**背景**
当前策略 2 连续 KV，`block_tables` 传 NULL。要形成完整 PagedAttention 故事，需要策略 1。

**paged-serving 侧实施**
1. 修改 `src/tiny_llm_ffi.rs` 的 C ABI 为 v2（与 tiny-llm 侧共同确认）：
   ```c
   int tinyllm_step(
       TinyLlmHandle* handle,
       const int* seq_ids,
       const int* input_tokens,
       const int* positions,
       const int* seq_lens,
       const int* context_lens,      // 新增
       const int* block_tables,      // 扁平化
       const int* block_counts,      // 新增：每序列块数
       const unsigned char* is_prefill,
       int num_sequences,
       int* next_tokens,
       float* logprobs,
       int logprobs_k);
   ```
2. `src/tiny_llm_executor.rs`：
   - 构造扁平 `block_tables`、`block_counts`、`context_lens`；
   - 不再调用 `tinyllm_allocate_sequence` 为每个序列预留连续 KV；
   - 仍保留 `sequences_finished` 钩子，通知后端清理每序列元数据。
3. `tests/tiny_llm_text_e2e.rs` 保持 llama.cpp 逐 token 对齐测试。
4. 新增压力测试：多波并发 + 完成后再提交，确认物理块复用不产生串扰。

**tiny-llm 侧（另开任务，不在本仓库内）**
- 实现按 `block_tables[seq]` 写入/读取 K/V；
- 块池大小使用 `block_size * max_num_blocks`；
- 与 llama.cpp 对齐测试必须继续通过。

**验收**
- 真实环境运行：
  ```bash
  TINY_LLM_DIR=... TINY_LLM_MODEL=... PSERV_TOKENIZER_JSON=... \
  cargo test --features tiny-llm --test tiny_llm_backend --test tiny_llm_text_e2e
  ```
- 多波请求无 slot 泄漏；
- 输出与 llama.cpp 逐 token 一致。

---

### T12（可选）：BPE 安全增量解码或明确降级

**选项 A（推荐，成本低）**
- 保持 `BufferedDecoder`；
- 文档明确：HF tokenizer 流式响应是“请求结束时的一个完整文本 chunk”；
- 所有“token-level streaming”表述改为 SimpleTokenizer 限定。
- 验收：README/CHANGELOG 无夸大表述。

**选项 B（进阶）**
- 为 byte-level BPE 实现安全增量 decoder：
  - 维护字节缓冲；
  - `push` 时追加 token 字节，能完整解码 UTF-8 序列才输出；
  - 需要处理 tokenizers 非 byte-level 模型的降级路径；
- 增加差分测试：流式片段拼接 == 一次性 decode，并覆盖中文、emoji、跨 token 多字节字符。

**建议**
先做选项 A。选项 B 只有在时间充足且想深入 tokenizer 时再做。

---

## 5. P2：最终验收与冻结

### T13：版本发布与冻结声明

1. 完成 P0 全部任务并保持 CI 绿色。
2. 更新 `Cargo.toml` 版本为 `0.2.0`（如果 T11 策略 1 完成且验证过，可考虑 `1.0.0`）。
3. CHANGELOG 增加 `[0.2.0]`，列出 P0/P1 已完成项。
4. ROADMAP 更新：
   - 已完成的勾选；
   - 明确“不继续实现”的边界：无抢占、无 chunked prefill、无 prefix caching、无生产 CUDA kernel；
   - 指向 tiny-llm / cuflash 后续工作。
5. README 增加“开发结束标准”一节，声明本仓库进入低优先级维护状态。
6. 打 tag：`git tag v0.2.0`。

**最终验收命令（全部必须通过）**
```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --doc
cargo doc --no-deps
cargo bench --bench concurrency_benchmark -- --test
cargo bench --bench engine_benchmark -- --test
```

---

## 6. 完成后建议的叙事与边界

本项目完成后，面试时应按以下口径描述：

1. 这是一个 **LLM Serving 控制面** 项目，不是 CUDA kernel 项目；
2. 核心实现：分页 KV 块表、continuous batching 调度、内存水位线、OpenAI 兼容服务、CPU 参考前向；
3. 明确边界：无抢占、无 chunked prefill、真实 kernel 在 tiny-llm 仓库；
4. 性能数字必须标注是 Mock/CPU 参考后端，不能声称 GPU 吞吐；
5. 后续重心转向 tiny-llm 分页 KV kernel 或 vLLM/SGLang 上游 PR。

---

## 7. 任务执行日志

> 执行模型每完成一个任务，在下面填写 commit hash 与验证结果。

| Task | 状态 | commit | 验证结果 |
|---|---|---|---|
| T0 | [x] | 9f9d5f9 | cargo fmt/clippy/test 全绿 |
| T1 | [x] | 58b29ea | execution_pipeline / integration 测试通过 |
| T2 | [x] | eb1a787 | scheduler + integration 内存压力测试通过 |
| T3 | [x] | 89cc1f0 | scheduler + integration HOL 测试通过 |
| T4 | [x] | a3f2019 | engine / integration 通过；tiny-llm feature 下 check 通过 |
| T5 | [x] | 85fd794 | engine 测试通过（含 NonRetrySafeTimeoutExecutor） |
| T6 | [x] | fad84d9 | config / types 测试通过 |
| T7 | [x] | ce33374 | engine + server_integration stop 测试通过 |
| T8 | [x] | 46c63e7 | doc/build/test 通过；grep Hetero 为空 |
| T9 | [x] | 2455a41 | server 单测 + server_integration 通过 |
| T10 | [x] | c527960 | server_integration metrics + bench smoke 通过 |
| T11 | [x] | 9e8f6c7 | 分页 KV（策略 1）接入：ABI v2 + 真实块表；llama.cpp 逐 token 对齐、3 并发 e2e 与资源守恒通过 |
| T12 | [x] | 61f3bd0 | 文档明确降级（选项 A），随 T8/T13 完成 |
| T13 | [x] | (本 commit) | 最终验收全部通过，v0.2.0 |
