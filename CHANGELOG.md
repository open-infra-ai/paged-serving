# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 请求取消与有界背压设计包（`docs/architecture/cancellation-backpressure-design.md`）：
  冻结 request 状态机所有权表、主动取消触发矩阵、四条 channel 的容量与
  overflow 策略、指标语义口径与测试方式；待评审后分 PR 实现（P0-001/002/003）。
- 请求所有权驱动的主动取消（PSRV-P0-001 / 设计包 PR-2）：每请求
  `watch` 取消信号 + handler 侧 `RequestGuard`（Drop 置位），引擎循环每步
  检出 `has_changed() != Ok(false)`（含 sender 全 drop 的 `Err`——owner
  消失即取消）；submission 准入预检跳过排队期已取消请求；SSE 流、unary
  future abort 与 n>1 部分准入失败均经 guard 主动取消；新增
  `create_router_with_engine_and_shutdown` 返回 shutdown 广播触发端，
  graceful shutdown 时取消全部在途请求（引擎循环自持保活 sender，避免
  router 先于响应体析构被误判为 shutdown）。

### Fixed
- 服务端不再因仅启用 `tiny-llm` 编译 feature 就被误认为正在使用真实 CUDA
  后端：新增显式 `--backend tiny-llm --model-path <model.gguf>` 运行时选择，并在
  feature、后端与模型参数不匹配时直接报错，避免性能实验静默落到 CPU reference。
- HuggingFace tokenizer 不再把所有生成 token 缓冲到请求结束才发送：升级到
  tokenizers 0.21，并使用其安全逐步流式 decode 状态机处理 BPE/WordPiece/
  byte-fallback 边界；中间片段与最终一次性 decode 保持严格等价。

### Changed
- README 与 serving benchmark 操作手册同步真实后端启动命令；`build.rs` 不再监听
  仅供测试使用、不会改变链接产物的 `TINY_LLM_MODEL` 环境变量。
- 归档首份真实 CUDA serving 结果（21 个 run、原始请求、模型 SHA-256、硬件与
  双仓 commit），如实记录吞吐平台、429 和 Poisson 未收敛，而不做跨硬件外推。
- 归档 P2 当前流式语义的 21-run CUDA 矩阵：首文本 TTFT、TPOT、inter-chunk 与
  固定 Poisson seed 均可追溯；未收敛重复与 429 保留在报告中，不将均值表述为 SLO。
- tiny-llm 的正常 greedy C ABI 路径现在在每序列 layer forward 后把末层 hidden 写入
  GPU batch buffer，并在 step 末尾批量执行 final RMSNorm、LM head 与 argmax 后一次回传
  整批 token；控制面 API 与 ABI 布局不变。Transformer layer forward 仍逐序列执行，且
  本项尚未重新采集 serving 矩阵，故不作为吞吐提升声明。
- 归档上述干净提交的真实 CUDA HTTP 功能 canary：closed c=4、4 个 smoke 请求均成功，
  结果包绑定双仓 commit、模型 SHA-256 与原始请求；`n=1`、无预热，不作为性能结果。
- 归档批量末端后处理后的真实 CUDA HTTP 正式矩阵：21 个 run 绑定 RTX 3060 Laptop、
  模型 SHA-256、双仓 clean commit、逐请求记录、固定 Poisson seed、CSV 与图表；closed-loop
  全部成功，Poisson 0.64 / 1.28 req/s 的 429 与未收敛重复如实保留，不作跨版本速度归因。

## [0.2.1] - 2026-08-28

### Added
- **Serving 压测客户端 `loadgen`**（`src/bin/loadgen.rs`，W1 评测地基）：
  对 OpenAI 兼容 `/v1/completions` SSE 端点做可复现负载实验；闭环饱和
  （`--mode closed`）与开环泊松（`--mode poisson`）双负载模型；指标口径
  TTFT（首个非空文本 chunk）/ ITL / 逐请求 TPOT / 失败归类（timeout /
  http_429 / http_4xx / http_5xx / connection / stream_error / no_done）；
  同一二进制零改动覆盖 paged-serving / llama-server / vLLM，保证横向可比；
  输出 per_request.jsonl + stdout 分位汇总。冒烟数据集
  `benchmarks/serving/datasets/synth/smoke.jsonl`。CPU 后端双模式端到端验证通过。
- tiny-llm 后端容量调节环境变量：`PAGED_SERVING_TINY_LLM_MAX_SEQS`（最大并发序列，
  默认 4，下限 1）与 `PAGED_SERVING_TINY_LLM_DECODE_RESERVE`（decode 预留 token，
  默认 512，下限 0）；原硬编码常量改为默认值，非法值记警告并回退默认。
  动机：6GB 卡跑 0.5B 模型时 KV 池仅 ~300MB，并发 8 可容纳，压测不再被
  保守上限挡住；长生成场景可按 `max_tokens` 上调预留。README 能力表同步。
- 分页 KV 策略 1 默认启用（**T11 完成**，`9e8f6c7`）：tiny-llm 后端经 C ABI 真实
  上传 `block_tables`/`num_blocks`（ABI v2），默认启用；`PAGED_SERVING_TINY_LLM_STRATEGY=2`
  可回退连续 KV。同步更新 README / ROADMAP / DEVELOPMENT_PLAN（0.2.0 发布时的
  CHANGELOG 仍写 T11 未实施，此处订正）。
- 优先级调度（**E2 完成**，`a69b146`）：`GenerationParams::priority` 高优先级先调度
  （同级 FCFS）；server 层透传优先级，支持单请求指定优先级。

### Fixed
- 与 tiny-llm 同步澄清 `tinyllm_step` 的 logprobs 缓冲区契约：每个候选占两个
  `f32`，总容量至少为 `num_sequences * logprobs_k * 2`；本次不改变 ABI 布局。
- 修复 serving 评测的指标与产物契约：TTFT 从请求发送前开始计时，不再漏掉
  连接、排队、prefill 与响应头耗时；SSE 解析同时支持 LF/CRLF，并在完整 event
  到齐后解码 UTF-8；非法 SSE/JSON 明确归类为 `protocol_error`。取消用文本
  chunk 数冒充 completion token 数，usage 缺失时仅允许显式 tokenizer 对完整
  输出文本计数；token coverage 不足时 tok/s 留空。新增权威 `summary.json` 与
  loadgen 回归测试，绘图只读取全局测量墙钟，不再用单请求最大时长高估吞吐，
  并按引擎/数据集分系列，避免把不同后端平均到同一条曲线。
- `tiny_llm_executor` 门控单测 `test_alloc_tokens_for_capacity_formula` 的陈旧断言：
  原断言小 context 时容量恒为 64，与公式 `(context + reserve).max(64)` 矛盾
  （该测试在默认 CI 不启用 tiny-llm feature，长期未执行）。现断言与公式及
  B15 越界回归一致，并补下限生效边界与 reserve 可配置用例。
- 修复 all-features Clippy 门禁：文档列表续行缩进、测试配置初始化与边界断言
  遵循当前 lint；并把 Rust 1.87 才稳定的 `is_multiple_of` 改为兼容声明 MSRV 1.82
  的取模表达式。

### Changed
- Serving 评测归档改为根环境 metadata + 每 run 模型/量化 metadata +
  `per_request.jsonl` + `summary.json` + `stdout.log`；方法论明确 SSE chunk 间隔
  不是 token 级 ITL，当前协议无法可靠测得 ITL 时必须显示不可用。README 与
  ROADMAP 统一为“控制面核心稳定、可信评测与上游转化仍 active”，并明确
  推理加速主线属于 tiny-llm。Cargo 包元数据改为真实维护者、仓库、许可证和
  Rust 版本，并设置 `paged-serving` 为默认 binary，恢复 README 中
  `cargo run -- --serve` 的可用性；FFI rustdoc 同步当前真实接入状态。
- README IN/OUT：tokenizer 改为 HTTP 边界适配器；词表/BPE 权威仍在 tiny-llm
- 架构图改为控制面 + CPU 参考 / tiny-llm 策略 1 双后端
- 面向用户的 GitHub 链接统一为 `github.com/open-infra-ai/...`

## [0.2.0] - 2026-08-17

### Changed

- 仓库由 `hetero-paged-serving` 更名为 `paged-serving`：原名中的 "hetero"（异构）与当前纯 CPU 参考后端的实际状态不符。crate 更名为 `paged-serving`/`paged_serving`，指标名同步更名为 `paged_*`。旧仓库地址自动重定向。

### Added

- tiny-llm 真实后端接入（里程碑 3/4）：
  - `build.rs`：`TINY_LLM_DIR` 指向 tiny-llm 构建目录时链接 `libtiny_llm.a` +
    spdlog + CUDA runtime（监听库文件变更自动重链）
  - `src/tiny_llm_executor.rs`：`TinyLlmExecutor` 实现 `GPUExecutorTrait`，
    把引擎调度的 `ExecutionBatch` 经 C ABI 交给 tiny-llm 步进执行
    （策略 2 连续 KV，位置由后端跟踪，greedy only）
  - `tiny_llm_ffi.rs`：契约同步（`tinyllm_step` 增加 `seq_ids`，支持任意 id 混批）
  - 测试：`tests/tiny_llm_backend.rs`（feature + `TINY_LLM_MODEL` 门控，
    能力声明、3 并发端到端、资源守恒）
- 真实 tokenizer 词表对齐（打通文本质量验证）：
  - `HuggingFaceTokenizer` 从词表探测真实 BOS/EOS/PAD（Qwen2：BOS/PAD=151643、
    EOS=151645），encode 不再自动追加特殊 token（与 tiny-llm `add_bos=false`
    语义一致），`vocab_size()` 返回含 added tokens 的完整词表（Qwen2.5 为 151665）
  - CLI 新增 `--tokenizer <path>`：启用 HuggingFace tokenizer（也可经
    `config.json` 的 `tokenizer.kind=huggingface` 配置）
  - 差分验证 `tests/tokenizer_real_diff.rs`：paged-serving(HF) 与 tiny-llm 权威
    fixture 逐 id 对齐（30/30，`PSERV_TOKENIZER_JSON` + `PSERV_TOKENIZER_FIXTURE` 门控）
  - 文本质量端到端 `tests/tiny_llm_text_e2e.rs`：真实后端 + 真实 tokenizer，
    与 llama.cpp 同 prompt greedy 输出逐 token 完全一致（24/24），EOS 正确终止
- 并发压测框架（ROADMAP 选项 A 第三项）：
  - `tests/concurrency_stress.rs`：并发突发尾延迟分布、高并发资源守恒、
    失败隔离与失败后资源归还、内存压力优雅处理（4 个场景断言）
  - `benches/concurrency_benchmark.rs`：并发突发吞吐基线
    （n8/16/32/64 全部排空耗时，Mock 后端）
- Engine event API: `InferenceEngine::step_events()` / `StepEvents` (per-step completions plus per-token text events), `create_router_with_engine()` for injecting custom executors into the HTTP layer (used by tests), and `submit_request()` returning `(request_id, prompt_tokens)` from a single tokenization pass.
- Server: token-level SSE streaming (with SimpleTokenizer), 429 overload rejection with `Retry-After`, graceful shutdown (SIGTERM / Ctrl+C), a real `/readyz` probe, `model` validation (404 on mismatch), chat role whitelist, and a JSON error envelope with a `type` field on every error path (including malformed bodies and unknown routes).
- Tests: GPU-timeout retry-exhaustion path, server concurrency / 500 / 429 / 404 coverage, and strengthened assertions replacing tautological ones. CI runs fmt + clippy + test + doc (no `--all-features`; the tiny-llm feature needs a real model and is not part of default CI).
- `test-utils` cargo feature: test helpers are now compiled only under `cfg(test)` or this feature, keeping them out of release builds.
- `InferenceEngine::cancel_request()` / `Scheduler::cancel_by_request_id()`: cancel in-flight requests (any stage) and free their KV blocks.
- `--host` / `--port` CLI overrides for serve mode (work with or without `--config`).
- Integration test proving the engine stops executing a request after its client disconnects.

### Removed

- Command bridge serving path and its related config, tests, and exported API surface.
- Local AI-tool residue in `.gitignore` (`.claude/`, `CLAUDE.local.md`, `.omc/`).
- GitHub Pages sections for whitepaper, benchmarks, comparison, references, and speculative advanced configuration.
- Repository-scoped AI workflow scaffolding: `CLAUDE.md` and the full legacy spec tree.
- Changelog mirrors under GitHub Pages (`docs/en/changelog/`, `docs/zh/changelog/`).
- Unused docs dependency `vitepress-plugin-llms`.
- Unused direct Rust dependencies `tracing` and `tracing-subscriber`.
- Dead code: CUDA Graph trait methods (never invoked by the engine), `Request::created_at`, `Scheduler::get_sequence_mut`, `PageTable::get_physical`, `RequestState::is_active`/`is_terminal`, the never-occurring unmapped `LogicalBlock` state, and the misleading copy-on-write comment on `PhysicalBlock::ref_count`.
- Fake after-the-fact "streaming" (chunking finished text into 32-char slices), replaced by token-level streaming (SimpleTokenizer; HF tokenizer uses a buffered chunk emitted at finish).
- Dead fields `Sequence::num_computed_tokens` and `ExecutionOutput::logits` (never read).
- `LogicalBlock` wrapper type: `PageTable` / `Sequence` now hold `Vec<PhysicalBlockRef>` directly.
- Unused `KVCacheManager::has_sequence` and `Scheduler::num_prefill_sequences`.
- `InferenceEngine::count_prompt_tokens` (folded into `submit_request`'s return value, eliminating per-request double tokenization).
- Redundant `config.validate()` calls in `main` / `create_router` (constructors already validate).
- `usize_from_u32` helper (inlined).

### Changed

- Server concurrency model: the engine is now owned by a single background loop; handlers submit via channels and await per-request events. Continuous batching now works under concurrent HTTP load, and the async runtime is no longer blocked (previously a global mutex serialized every request behind a full `engine.run()`).
- Scheduler determinism: active sequences use `BTreeMap` (FCFS by submission order) instead of `HashMap`, making scheduling fair and reproducible.
- `temperature == 0.0` (greedy decoding) is now accepted; the valid range is `[0.0, 2.0]`.
- `build_execution_batch` fails fast on an inconsistent decode sequence instead of silently skipping it (which would have left the request stuck forever).
- Usage reporting uses the real tokenizer: `prompt_tokens` / `completion_tokens` are exact counts, not whitespace estimates.
- Overflow-safe token arithmetic in the scheduler (`saturating_add`) and an always-safe `Sequence::context_len`.
- Benchmarks rebuilt with `iter_batched` so they no longer panic or degrade into measuring idle/error paths; filler benches removed.
- Crate metadata, CLI about text, README, and docs now describe the project as a paged-memory + continuous-batching scaffold with a mock compute backend; unsupported claims ("CPU-GPU co-execution", fabricated throughput charts, fictional preemption API) were removed.
- The engine loop now yields once per step; previously it had no await point during active generation.
- Client disconnect now cancels the in-flight request instead of generating to `max_tokens` for nobody; the SSE stream reports an error event rather than a bare `[DONE]` when the event channel closes without a terminal event.
- `InferenceEngine::run()` drains completions buffered before the loop (e.g. cancelled requests).
- A too-large prefill no longer head-of-line-blocks smaller prefills (`continue` instead of `break`).
- `step_events` attributes generated tokens via an O(1) map instead of a per-token linear scan.
- `ExecutionError::CudaError` renamed to `ExecutionError::BackendError` (the mock backend is not CUDA).
- CLI config built from `EngineConfig::default()` + overrides instead of duplicated default literals.
- Docs and examples updated for the new `submit_request` signature, removed fields, and `LogicalBlock` removal.

### Fixed

- Benchmark suite crashed during warmup (`bench_request_submission` exhausted `max_num_seqs`); the whole suite is now runnable and smoke-tested in CI.
- Mock executor divide-by-zero panic on an empty vocabulary (now an error, consistent with the CUDA backend).
- Double-free and zero `block_size` misuse are now caught by debug assertions with clear messages.
- Engine loop had no await point during active generation, starving handlers / SSE streams on single-threaded runtimes (first token never reached the client until the batch finished) and pinning a worker on multi-threaded runtimes; now yields each step.

### P0 正确性修复（T0–T7）

- **T0** 修复 fmt / clippy，恢复 CI 绿色（`unnecessary_cast`、未使用 import、`manual_is_multiple_of`）。
- **T1** 执行输出契约校验：坏后端（空输出、重复/缺失 seq id、错位 logprobs）不再让请求
  卡死在调度器中，而是快速失败并以终态排出。
- **T2** 调度器内存水位线 + decode 增长预留：启动新 prefill 前同时检查高水位线与
  "下一步 decode 增长"的预留块，避免把块池打满导致下一步 `OutOfBlocks`。
- **T3** 修复 pending 队头阻塞（HOL）：每步按当前队列长度扫描一轮，装不下的请求
  `push_back` 延后，不再用 `push_front + break` 让大 prefill 挡住小 prefill。
- **T4** 后端序列生命周期钩子 `sequences_finished`：引擎在序列完成/失败/取消并释放
  逻辑 KV 块后通知后端，tiny-llm 据此释放物理 KV 槽位（修复 slot 耗尽）。
- **T5** 超时重试声明幂等性：仅 `retry_safe=true` 的后端才会在 `GpuTimeout` 后重放
  同一 batch；真实 CUDA 后端默认不重放。
- **T6** 浮点参数 NaN 校验：`memory_threshold`、`temperature`、`top_p` 的 NaN/infinity
  均被拒绝，避免内存保护与采样范围校验被绕过。
- **T7** Unicode stop 序列字节偏移修复：`tokens_before_char` 按字节长度累加，
  中文等非 ASCII 文本不再错误保留 stop 序列。

### P1（T9–T10、T12）

- **T9** Chat Completions 应用真实 chat template：使用 HuggingFace tokenizer 时
  采用 Qwen2 的 `<|im_start|>` 模板（system/user/assistant + 末尾 assistant 引导）；
  `SimpleTokenizer` 保持原 `role: content` 拼接。模板当前硬编码 Qwen2。
- **T10** 引擎指标接入 `/metrics`：新增 `paged_engine_active_sequences` /
  `paged_engine_kv_utilization` / `paged_engine_completed_requests` /
  `paged_engine_failed_requests` / `paged_engine_tokens_generated_total`；
  `concurrency_benchmark` 增加 CB on/off 对比组（`max_batch_size=1` vs `=N`）。
- **T12（选项 A）** 明确流式降级：保留 `BufferedDecoder`，文档明确 HF tokenizer
  流式为"请求结束时的一个完整文本 chunk"，"token-level streaming" 表述限定
  `SimpleTokenizer`（随 T8 完成）。
- **T11** 在 0.2.0 发布时未实施（需 tiny-llm 仓库同步改造分页 KV C ABI）；
  已于 2026-08-18 完成并默认启用（策略 1），详见 [0.2.1]。

## [0.1.0] - 2026-04-16

### Added

- Bilingual project docs (README, docs site, API references, deployment and development guides).
- Core Rust inference engine modules for paged KV cache, scheduler, tokenizer, and execution pipeline.
- OpenAI-compatible serving endpoints (`/v1/completions`, `/v1/chat/completions`) with health/readiness/metrics endpoints.
- Mock GPU executor and broad automated test coverage across unit/property/integration/doc tests.

### Changed

- 2026-03-13: CPU-safe CI fixes and clippy-driven code cleanup (`div_ceil`, `HashMap::entry` usage).
- 2026-03-10: GitHub Actions workflow standardization for permissions, concurrency, and docs pipeline reliability.

## Historical Notes

Legacy spec archive records were removed during repository simplification.
Durable project history is now condensed in this changelog and GitHub Releases.

[0.2.1]: https://github.com/open-infra-ai/paged-serving/releases/tag/v0.2.1
[0.2.0]: https://github.com/aicl-lab/paged-serving/releases/tag/v0.2.0
[0.1.0]: https://github.com/aicl-lab/paged-serving/releases/tag/v0.1.0
