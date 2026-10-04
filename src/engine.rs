//! 推理引擎 - 主编排器
//!
//! 协调所有组件实现端到端推理：
//! - Tokenizer 用于文本处理
//! - Scheduler 用于请求管理
//! - GPU Executor 用于计算
//! - KV Cache Manager 用于内存管理
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │            InferenceEngine              │
//! │  ┌──────────┐  ┌──────────┐  ┌───────┐  │
//! │  │Tokenizer │  │Scheduler │  │  GPU  │  │
//! │  │          │  │          │  │Executor│ │
//! │  └──────────┘  └────┬─────┘  └───────┘  │
//! │                     │                    │
//! │              ┌──────▼──────┐            │
//! │              │ KV Cache    │            │
//! │              │ Manager     │            │
//! │              └─────────────┘            │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # 示例
//!
//! ```rust
//! use paged_serving::{EngineConfig, GenerationParams, InferenceEngine};
//!
//! // 创建引擎
//! let config = EngineConfig::default();
//! let mut engine = InferenceEngine::new(config)?;
//!
//! // 提交请求（CPU 参考后端当前仅支持 greedy：temperature = 0.0, top_p = 1.0）
//! let params = GenerationParams {
//!     max_tokens: 50,
//!     ..GenerationParams::default()
//! };
//! let (request_id, _prompt_tokens) = engine.submit_request("你好，世界！", params)?;
//!
//! // 运行推理
//! let completed = engine.run();
//!
//! for result in completed {
//!     println!("输出: {}", result.output_text);
//! }
//! # Ok::<(), paged_serving::EngineError>(())
//! ```

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::execution_pipeline::BatchExecutionPipeline;
use crate::gpu_executor::{create_default_gpu_executor, GPUExecutorTrait};
use crate::scheduler::Scheduler;
use crate::tokenizer::{build_tokenizer, IncrementalDecoder, TokenizerTrait};
use crate::types::{
    CompletedRequest, FinishReason, GenerationParams, Request, RequestId, RequestState, SeqId,
    TokenId, TokenLogprobs,
};
use std::collections::HashMap;

/// 本步为单个请求生成的文本片段：`(request_id, 解码文本, 该 token 的 logprob)`。
pub type StepChunk = (RequestId, String, Option<TokenLogprobs>);

/// 文本中第一个 stop 序列出现的**字节偏移**（`str::find` 语义，取最早命中者）。
fn find_stop_sequence(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// 输出 token 中位于字节偏移 `byte_offset` 之前（不含）的 token 数。
///
/// `str::find` 返回的是**字节偏移**（而非字符偏移），因此这里必须按
/// 每个 token 解码后的**字节长度**（`str::len`）累加，才能与非 ASCII
/// 文本精确对齐。对逐字符 tokenizer（SimpleTokenizer）两者等价；对
/// 子词 tokenizer（HuggingFace）保守下取整——stop 序列开头的部分
/// token 也会一并移除，保证输出中绝不残留 stop 序列。
fn tokens_before_char(
    tokens: &[TokenId],
    tokenizer: &dyn TokenizerTrait,
    byte_offset: usize,
) -> usize {
    let mut len = 0usize;
    for (i, &token) in tokens.iter().enumerate() {
        if len >= byte_offset {
            return i;
        }
        if let Ok(segment) = tokenizer.try_decode(&[token]) {
            len += segment.len();
        }
    }
    tokens.len()
}

/// 推理引擎
///
/// 主编排器，协调所有组件实现端到端推理。
///
/// # 组件
///
/// - **Tokenizer** - 文本与 token 之间的转换
/// - **Scheduler** - 请求调度和批次管理
/// - **GPU Executor** - GPU 计算执行
/// - **KV Cache Manager** - KV Cache 内存管理
///
/// # 示例
///
/// ```rust
/// use paged_serving::{EngineConfig, InferenceEngine};
///
/// let config = EngineConfig::default();
/// let engine = InferenceEngine::new(config)?;
/// # Ok::<(), paged_serving::EngineError>(())
/// ```
pub struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn TokenizerTrait>,
    scheduler: Scheduler,
    execution_pipeline: BatchExecutionPipeline,
    /// 每请求增量解码器：生命周期与请求一致（完成/取消/失败时销毁），
    /// 保证流式片段拼接 == 最终一次性 decode 文本。
    decoders: HashMap<RequestId, Box<dyn IncrementalDecoder>>,
    eos_token_id: u32,
    total_requests: u64,
    completed_requests_count: u64,
    failed_requests_count: u64,
    cancelled_requests_count: u64,
    total_tokens_generated: u64,
    next_request_id: RequestId,
}

impl InferenceEngine {
    /// 创建新的推理引擎（使用默认组件）
    ///
    /// # 参数
    ///
    /// * `config` - 引擎配置
    ///
    /// # 错误
    ///
    /// 如果配置无效，返回 [`EngineError::Config`]。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_serving::{EngineConfig, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let engine = InferenceEngine::new(config)?;
    /// # Ok::<(), paged_serving::EngineError>(())
    /// ```
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        config.validate()?;

        let tokenizer = build_tokenizer(&config)?;
        let vocab_size = tokenizer.vocab_size();
        let eos_token_id = tokenizer.eos_token_id();

        let scheduler = Scheduler::new(config.clone());
        let gpu_executor = create_default_gpu_executor(config.clone(), vocab_size)?;
        let execution_pipeline = BatchExecutionPipeline::new(gpu_executor, &config);

        Ok(Self {
            config,
            tokenizer,
            scheduler,
            execution_pipeline,
            decoders: HashMap::new(),
            eos_token_id,
            total_requests: 0,
            completed_requests_count: 0,
            failed_requests_count: 0,
            cancelled_requests_count: 0,
            total_tokens_generated: 0,
            next_request_id: 1,
        })
    }

    /// 使用自定义组件创建引擎（用于测试）
    ///
    /// # 参数
    ///
    /// * `config` - 引擎配置
    /// * `tokenizer` - 自定义分词器
    /// * `scheduler` - 自定义调度器
    /// * `gpu_executor` - 自定义 GPU 执行器
    pub fn with_components(
        config: EngineConfig,
        tokenizer: Box<dyn TokenizerTrait>,
        scheduler: Scheduler,
        gpu_executor: Box<dyn GPUExecutorTrait>,
    ) -> Result<Self, EngineError> {
        config.validate()?;

        let eos_token_id = tokenizer.eos_token_id();
        let execution_pipeline = BatchExecutionPipeline::new(gpu_executor, &config);

        Ok(Self {
            config,
            tokenizer,
            scheduler,
            execution_pipeline,
            decoders: HashMap::new(),
            eos_token_id,
            total_requests: 0,
            completed_requests_count: 0,
            failed_requests_count: 0,
            cancelled_requests_count: 0,
            total_tokens_generated: 0,
            next_request_id: 1,
        })
    }

    /// 提交新的推理请求
    ///
    /// # 参数
    ///
    /// * `text` - 输入文本
    /// * `params` - 生成参数
    ///
    /// # 返回
    ///
    /// `(请求唯一标识符, prompt token 数)`。prompt token 数来自提交时的
    /// 同一次分词，供服务层报告精确 usage，无需二次分词。
    ///
    /// # 错误
    ///
    /// - [`EngineError::EmptyInput`] / [`EngineError::InputTooLong`] / [`EngineError::TotalLengthTooLong`] - 参数无效或输入为空
    /// - [`EngineError::MemoryPressure`] / [`EngineError::MaxConcurrentSequencesReached`] - 内存压力或达到序列上限
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_serving::{EngineConfig, GenerationParams, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let mut engine = InferenceEngine::new(config)?;
    ///
    /// let params = GenerationParams {
    ///     max_tokens: 50,
    ///     ..GenerationParams::default()
    /// };
    ///
    /// let (request_id, prompt_tokens) = engine.submit_request("你好", params)?;
    /// assert!(request_id > 0 && prompt_tokens > 0);
    /// # Ok::<(), paged_serving::EngineError>(())
    /// ```
    pub fn submit_request(
        &mut self,
        text: &str,
        params: GenerationParams,
    ) -> Result<(RequestId, usize), EngineError> {
        // 验证参数
        params.validate()?;

        // 后端能力校验：不支持的采样语义在 submit 阶段诚实失败，
        // 而不是在执行时被后端静默降级为 greedy。
        if !self.execution_pipeline.capabilities().sampling && !params.is_greedy() {
            return Err(EngineError::UnsupportedGenerationMode(format!(
                "backend only supports greedy decoding (temperature == 0.0, top_p == 1.0), \
                 got temperature = {}, top_p = {}",
                params.temperature, params.top_p
            )));
        }

        // 验证输入
        if text.is_empty() {
            return Err(EngineError::EmptyInput);
        }

        // 分词
        let input_tokens = self
            .tokenizer
            .try_encode(text)
            .map_err(EngineError::Tokenization)?;
        let prompt_tokens = input_tokens.len();

        // 检查 prompt 长度
        if input_tokens.len() > self.config.max_model_len as usize {
            return Err(EngineError::InputTooLong(
                input_tokens.len(),
                self.config.max_model_len,
            ));
        }

        let total_requested_tokens = input_tokens.len() + params.max_tokens as usize;
        if total_requested_tokens > self.config.max_model_len as usize {
            return Err(EngineError::TotalLengthTooLong(
                total_requested_tokens,
                self.config.max_model_len,
            ));
        }

        // 创建请求（使用实例级 ID）
        let request_id = self.next_request_id;
        let request = Request::new(request_id, input_tokens, params);

        // 添加到调度器；成功后才推进 request_id，避免 add_request 失败
        // （如 MemoryPressure）时产生 id 空洞、metrics 与外层 request_id 不连续。
        self.scheduler.add_request(request)?;
        self.next_request_id += 1;
        self.total_requests += 1;
        self.decoders
            .insert(request_id, self.tokenizer.create_decoder());

        Ok((request_id, prompt_tokens))
    }

    /// 执行一步推理
    ///
    /// 调度下一批次并通过执行流水线完成 GPU 计算。
    ///
    /// # 返回
    ///
    /// 本次步骤完成的请求列表。
    pub fn step(&mut self) -> Result<Vec<CompletedRequest>, EngineError> {
        self.step_events().map(|events| events.completed)
    }

    /// 执行一步推理，并返回细粒度事件
    ///
    /// 除完成请求外，还报告本步为每个请求新生成的 token（已解码为文本片段），
    /// 供服务层驱动 token 级流式响应。
    pub fn step_events(&mut self) -> Result<StepEvents, EngineError> {
        // 调度下一批次
        let scheduler_output = self.scheduler.schedule();
        let mut generated: Vec<(RequestId, TokenId)> = Vec::new();

        if !scheduler_output.is_empty() {
            // seq_id → request_id 映射，用于把生成的 token 归属到请求
            let request_id_by_seq: HashMap<SeqId, RequestId> = scheduler_output
                .prefill_sequences
                .iter()
                .chain(scheduler_output.decode_sequences.iter())
                .map(|seq| (seq.seq_id, seq.request.id))
                .collect();
            let seq_ids = scheduler_output.seq_ids();

            // 通过批次执行流水线完成 GPU 计算（含重试）
            match self.execution_pipeline.execute(&scheduler_output) {
                Ok(execution_output) => {
                    for (seq_id, token) in execution_output
                        .seq_ids
                        .iter()
                        .zip(execution_output.next_tokens.iter())
                    {
                        if let Some(&request_id) = request_id_by_seq.get(seq_id) {
                            generated.push((request_id, *token));
                        }
                    }
                    self.scheduler
                        .update_sequences(&execution_output, self.eos_token_id);
                }
                Err(engine_error) => {
                    let reason = engine_error.to_string();
                    self.scheduler.fail_sequences(seq_ids, &reason);
                    // 失败路径同样排出终态并销毁对应 decoder（不回放缓冲文本）
                    let (completed, tail_chunks) = self.collect_completed_requests();
                    return if completed.is_empty() {
                        Err(engine_error)
                    } else {
                        Ok(StepEvents {
                            completed,
                            chunks: tail_chunks,
                        })
                    };
                }
            }
        }

        // PSERV-105：stop 序列生成端检测（在 decoder push 之前，
        // 命中后触发 token 不会作为片段推送给客户端）。
        self.apply_stop_sequences(&generated);

        // 用每请求增量解码器把新 token 解码为文本片段。
        // 解码失败时让请求诚实失败（tokenizer 错误），而不是静默丢失文本。
        let mut chunks: Vec<StepChunk> = Vec::new();
        for (request_id, token) in generated {
            let Some(decoder) = self.decoders.get_mut(&request_id) else {
                continue;
            };
            // 该 token 的 logprob（update_sequences 已把 logprobs 累积到请求）
            let token_logprobs = self
                .scheduler
                .get_request_by_id(request_id)
                .filter(|r| r.params.logprobs.is_some())
                .and_then(|r| r.logprobs.last().cloned());
            match decoder.push(token) {
                Ok(Some(text)) if !text.is_empty() => {
                    chunks.push((request_id, text, token_logprobs))
                }
                Ok(_) => {}
                Err(msg) => {
                    self.scheduler
                        .fail_by_request_id(request_id, &format!("tokenizer decode failed: {msg}"));
                }
            }
        }

        let (completed, tail_chunks) = self.collect_completed_requests();
        chunks.extend(tail_chunks);

        Ok(StepEvents { completed, chunks })
    }

    /// PSERV-105：对本次生成过 token 且配置了 stop 序列的请求做生成端检测。
    ///
    /// 一旦输出文本命中任一 stop 序列：把输出 token 截断到序列之前、
    /// 标记请求完成（`finish_reason="stop"`），并销毁其增量解码器——
    /// 已推送的片段无法撤回，避免 `finish()` 把未截断的文本回放给客户端。
    fn apply_stop_sequences(&mut self, generated: &[(RequestId, TokenId)]) {
        let mut to_stop: Vec<(RequestId, usize)> = Vec::new();
        for &(request_id, _) in generated {
            let Some(req) = self.scheduler.get_request_by_id(request_id) else {
                continue;
            };
            if req.params.stop.is_empty() {
                continue;
            }
            let Ok(text) = self.tokenizer.try_decode(&req.output_tokens) else {
                continue;
            };
            if let Some(start) = find_stop_sequence(&text, &req.params.stop) {
                let keep = tokens_before_char(&req.output_tokens, &*self.tokenizer, start);
                to_stop.push((request_id, keep));
            }
        }
        for (request_id, keep) in to_stop {
            self.scheduler.complete_by_stop_sequence(request_id, keep);
            self.decoders.remove(&request_id);
        }
    }

    /// 排出所有到达终态的请求；同时销毁其增量解码器。
    ///
    /// 对成功完成的请求调用 `finish()` 冲刷末尾文本，作为本步附加片段返回
    /// （保证在 Done 事件之前送达）；失败/取消的请求直接丢弃解码器状态。
    /// 到达终态的序列会先通知后端释放物理 KV 资源（[`GPUExecutorTrait::sequences_finished`]）。
    pub(crate) fn collect_completed_requests(&mut self) -> (Vec<CompletedRequest>, Vec<StepChunk>) {
        let completed_requests = self.scheduler.take_completed_with_seq_ids();
        if completed_requests.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // 序列已释放逻辑 KV 块，通知后端同步释放其物理 KV 资源。
        let finished_seq_ids: Vec<SeqId> = completed_requests
            .iter()
            .map(|(seq_id, _)| *seq_id)
            .collect();
        self.execution_pipeline
            .sequences_finished(&finished_seq_ids);

        let mut tail_chunks: Vec<StepChunk> = Vec::new();
        let results = completed_requests
            .into_iter()
            .map(|(_, req)| {
                let decoder = self.decoders.remove(&req.id);
                let input_text = self.tokenizer.try_decode(&req.input_tokens).ok();
                let decoded_output = self.tokenizer.try_decode(&req.output_tokens);
                let tokenization_error = decoded_output.as_ref().err().cloned();
                let output_text = decoded_output.unwrap_or_default();
                let success =
                    matches!(req.state, RequestState::Completed) && tokenization_error.is_none();
                // 成功时区分停止原因：末 token 为 EOS → 自然停止；否则是达到
                // max_tokens 被截断。失败/取消的请求没有 finish_reason。
                let finish_reason = if success {
                    let stopped_on_eos = req.output_tokens.last() == Some(&self.eos_token_id);
                    // stop 序列命中（可能已把 EOS 之后的 token 截断）或 EOS → 自然停止
                    Some(if req.stop_sequence_hit || stopped_on_eos {
                        FinishReason::Stop
                    } else {
                        FinishReason::Length
                    })
                } else {
                    None
                };
                let error = match (&req.state, tokenization_error) {
                    (RequestState::Failed(msg), _) => Some(msg.clone()),
                    (RequestState::Cancelled(reason), _) => Some(reason.to_string()),
                    (_, Some(msg)) => Some(format!("tokenizer decode failed: {msg}")),
                    _ => None,
                };
                let cancellation = match req.state {
                    RequestState::Cancelled(reason) => Some(reason),
                    _ => None,
                };
                // logprobs：请求启用且后端实际提供了时才返回
                let logprobs = if req.params.logprobs.is_some() && !req.logprobs.is_empty() {
                    Some(req.logprobs.clone())
                } else {
                    None
                };

                if success {
                    if let Some(mut decoder) = decoder {
                        match decoder.finish() {
                            Ok(Some(tail)) if !tail.is_empty() => {
                                // finish() 冲刷的末尾文本不对应单个新 token，logprobs 置空
                                tail_chunks.push((req.id, tail, None));
                            }
                            Ok(_) => {}
                            Err(msg) => {
                                log::warn!("decoder finish failed for request {}: {msg}", req.id);
                            }
                        }
                    }
                }

                self.total_tokens_generated += req.output_tokens.len() as u64;
                if success {
                    self.completed_requests_count += 1;
                } else if cancellation.is_some() {
                    self.cancelled_requests_count += 1;
                } else {
                    self.failed_requests_count += 1;
                }

                CompletedRequest {
                    request_id: req.id,
                    input_text,
                    output_text,
                    output_tokens: req.output_tokens,
                    success,
                    error,
                    cancellation,
                    finish_reason,
                    logprobs,
                }
            })
            .collect();

        (results, tail_chunks)
    }

    /// 运行推理循环直到所有请求完成
    ///
    /// # 返回
    ///
    /// 所有完成的请求列表。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_serving::{EngineConfig, GenerationParams, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let mut engine = InferenceEngine::new(config)?;
    ///
    /// let params = GenerationParams::default();
    /// engine.submit_request("测试", params)?;
    ///
    /// let completed = engine.run();
    /// for result in completed {
    ///     println!("输出: {}", result.output_text);
    /// }
    /// # Ok::<(), paged_serving::EngineError>(())
    /// ```
    pub fn run(&mut self) -> Vec<CompletedRequest> {
        // 先排出循环前已缓冲的终态（如 cancel_request 产生的失败请求）；
        // 非流式调用方不需要增量片段，丢弃 tail chunks（完整文本在 output_text）。
        let (mut all_completed, _) = self.collect_completed_requests();

        // 防死循环（B12）：pending 因 KV 预算/块不足而永远无法启动时，
        // has_pending_work() 恒真但每步空转（无序列执行、无完成排出、无在途序列）。
        // 连续 STALL_LIMIT 步无任何进展即判定卡死：记录错误并退出，而非无限自旋。
        const STALL_LIMIT: u32 = 100;
        let mut idle_steps: u32 = 0;

        while self.scheduler.has_pending_work() {
            let completed_before = all_completed.len();
            match self.step() {
                Ok(completed) => {
                    all_completed.extend(completed);
                }
                Err(e) => {
                    log::error!("推理步骤失败: {e}");
                }
            }
            let progressed = all_completed.len() != completed_before
                || self.scheduler.num_active_sequences() > 0;
            if progressed {
                idle_steps = 0;
            } else {
                idle_steps += 1;
                if idle_steps >= STALL_LIMIT {
                    log::error!(
                        "引擎连续 {STALL_LIMIT} 步无进展：pending 请求因 KV 预算/块不足 \
                         永远无法启动，将其标记失败后结束 run()（避免死循环）"
                    );
                    // 不遗留：把卡死的 pending 请求标记失败并排出，
                    // 调用方必须能拿到每个提交请求的终态。
                    self.scheduler
                        .fail_pending_requests("request 因 KV 预算/块不足永远无法启动");
                    break;
                }
            }
        }

        // 循环可能因 break 提前退出，补收一次终态（含 fail_pending_requests 排出的失败请求）。
        let (final_completed, _) = self.collect_completed_requests();
        all_completed.extend(final_completed);

        all_completed
    }

    /// 按 request_id 取消请求（客户端断连时由服务层调用）。
    ///
    /// 序列无论处于哪个阶段都会被标记取消并释放 KV 资源；
    /// 其终态会在下一步经常规完成通道排出。返回是否取消成功。
    pub fn cancel_request(&mut self, request_id: RequestId) -> bool {
        self.scheduler.cancel_by_request_id(request_id)
    }

    pub fn cancel_request_with_reason(
        &mut self,
        request_id: RequestId,
        reason: crate::types::CancellationReason,
    ) -> bool {
        self.scheduler
            .cancel_by_request_id_with_reason(request_id, reason)
    }

    pub(crate) fn fail_request(&mut self, request_id: RequestId, reason: &str) -> bool {
        self.scheduler.fail_by_request_id(request_id, reason)
    }

    /// 当前存活的每请求增量解码器数量（测试用：验证状态随终态清理）。
    #[cfg(test)]
    fn num_live_decoders(&self) -> usize {
        self.decoders.len()
    }

    /// 检查是否有待处理的工作
    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

    /// 把等待队列中的全部请求标记为失败（stall 兜底）。
    ///
    /// 连续空转（无完成、无活跃序列、pending 非空）意味着这些请求因
    /// KV 预算/块不足而永远无法启动；主动失败并排出，避免调用方
    /// （服务层 engine_loop / run()）无限忙等。返回被失败处理的请求数。
    pub fn fail_pending_requests(&mut self, reason: &str) -> usize {
        self.scheduler.fail_pending_requests(reason)
    }

    /// 获取内存利用率
    pub fn memory_utilization(&self) -> f32 {
        self.scheduler.get_memory_utilization()
    }

    /// 获取配置
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

/// 单步推理事件
///
/// 由 [`InferenceEngine::step_events`] 产生，供服务层驱动流式响应。
#[derive(Debug, Clone, Default)]
pub struct StepEvents {
    /// 本步到达终态（成功、失败或取消）的请求
    pub completed: Vec<CompletedRequest>,
    /// 本步为各请求新生成的文本片段（见 [`StepChunk`]）
    pub chunks: Vec<StepChunk>,
}

/// 引擎指标
///
/// 运行时统计信息。
#[derive(Debug, Clone, Default)]
pub struct EngineMetrics {
    /// 已提交请求总数
    pub total_requests: u64,
    /// 成功完成请求总数
    pub completed_requests: u64,
    /// 失败请求总数（不含主动取消）
    pub failed_requests: u64,
    /// 主动取消请求总数
    pub cancelled_requests: u64,
    /// 已生成 token 总数
    pub total_tokens_generated: u64,
    /// 当前内存利用率
    pub memory_utilization: f32,
    /// 当前活跃序列数
    pub active_sequences: u32,
}

impl InferenceEngine {
    /// 获取当前指标
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_serving::{EngineConfig, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let engine = InferenceEngine::new(config)?;
    ///
    /// let metrics = engine.get_metrics();
    /// println!("完成请求: {}", metrics.completed_requests);
    /// # Ok::<(), paged_serving::EngineError>(())
    /// ```
    pub fn get_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            total_requests: self.total_requests,
            completed_requests: self.completed_requests_count,
            failed_requests: self.failed_requests_count,
            cancelled_requests: self.cancelled_requests_count,
            total_tokens_generated: self.total_tokens_generated,
            memory_utilization: self.memory_utilization(),
            active_sequences: self.scheduler.num_active_sequences() as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TokenizerConfig, TokenizerKind};
    use crate::error::EngineError;
    use crate::gpu_executor::ExecutorCapabilities;
    use crate::test_utils::{
        create_test_config, test_params, write_test_tokenizer_json, AlwaysFailExecutor,
        ConstantTokenExecutor, SequenceExecutor,
    };
    use crate::tokenizer::SimpleTokenizer;
    use crate::types::{ExecutionBatch, ExecutionOutput, SeqId};
    use std::fs;
    use std::sync::{Arc, Mutex};

    /// 记录 `sequences_finished` 收到的 seq id；可配置 execute 失败与否。
    struct RecordingExecutor {
        fail: bool,
        finished: Arc<Mutex<Vec<SeqId>>>,
    }

    impl RecordingExecutor {
        fn new(fail: bool) -> (Self, Arc<Mutex<Vec<SeqId>>>) {
            let finished = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    fail,
                    finished: finished.clone(),
                },
                finished,
            )
        }
    }

    impl GPUExecutorTrait for RecordingExecutor {
        fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
            if self.fail {
                return Err(EngineError::KernelLaunchFailed(
                    "recorded executor failure".to_string(),
                ));
            }
            Ok(ExecutionOutput {
                next_tokens: vec![123; batch.seq_ids.len()],
                seq_ids: batch.seq_ids.clone(),
                logprobs: Vec::new(),
            })
        }

        fn sequences_finished(&mut self, seq_ids: &[SeqId]) {
            self.finished.lock().unwrap().extend_from_slice(seq_ids);
        }
    }

    struct TimeoutThenSuccessExecutor {
        attempts: u32,
    }

    impl GPUExecutorTrait for TimeoutThenSuccessExecutor {
        fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
            if self.attempts == 0 {
                self.attempts += 1;
                Err(EngineError::GpuTimeout)
            } else {
                Ok(ExecutionOutput {
                    next_tokens: vec![123; batch.seq_ids.len()],
                    seq_ids: batch.seq_ids.clone(),
                    logprobs: Vec::new(),
                })
            }
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            // 确定性测试后端：超时后重放安全，允许重试。
            ExecutorCapabilities {
                sampling: false,
                retry_safe: true,
            }
        }
    }

    struct AlwaysTimeoutExecutor;

    impl GPUExecutorTrait for AlwaysTimeoutExecutor {
        fn execute(&mut self, _batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
            Err(EngineError::GpuTimeout)
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            // 保持 retry_safe=true，以便保留"重试耗尽后失败"的既有测试。
            ExecutorCapabilities {
                sampling: false,
                retry_safe: true,
            }
        }
    }

    /// 非幂等超时后端：即使有重试预算也绝不重放同一 batch。
    struct NonRetrySafeTimeoutExecutor {
        attempts: Arc<Mutex<u32>>,
    }

    impl NonRetrySafeTimeoutExecutor {
        fn new() -> (Self, Arc<Mutex<u32>>) {
            let attempts = Arc::new(Mutex::new(0));
            (
                Self {
                    attempts: attempts.clone(),
                },
                attempts,
            )
        }
    }

    impl GPUExecutorTrait for NonRetrySafeTimeoutExecutor {
        fn execute(&mut self, _batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
            *self.attempts.lock().unwrap() += 1;
            Err(EngineError::GpuTimeout)
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            ExecutorCapabilities {
                sampling: false,
                retry_safe: false,
            }
        }
    }

    #[test]
    fn test_engine_creation() {
        let config = create_test_config();
        let engine = InferenceEngine::new(config);

        assert!(engine.is_ok());
    }

    #[test]
    fn test_engine_creation_fails_when_huggingface_tokenizer_file_is_missing() {
        let config = EngineConfig {
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some("/tmp/does-not-exist-tokenizer.json".into()),
            },
            ..create_test_config()
        };

        let engine = InferenceEngine::new(config);
        assert!(matches!(engine, Err(EngineError::Tokenization(_))));
    }

    #[test]
    fn test_submit_request_uses_configured_huggingface_tokenizer() {
        let path = write_test_tokenizer_json();
        let config = EngineConfig {
            max_model_len: 6,
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some(path.clone()),
            },
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let result = engine.submit_request(
            "hello world",
            GenerationParams {
                max_tokens: 1,
                ..GenerationParams::default()
            },
        );

        assert!(
            result.is_ok(),
            "configured HuggingFace tokenizer should be used"
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_cancel_request_stops_generation_and_surfaces_typed_cancellation() {
        let mut engine = InferenceEngine::new(create_test_config()).unwrap();
        let (request_id, _) = engine
            .submit_request(
                "Hello world",
                GenerationParams {
                    max_tokens: 50,
                    ..GenerationParams::default()
                },
            )
            .unwrap();

        // 推进一步让请求进入 decode
        engine.step().unwrap();
        assert!(engine.has_pending_work());

        assert!(engine.cancel_request(request_id));

        // 即使没有其他工作，run 也必须排出取消产生的失败终态
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(completed[0]
            .error
            .as_deref()
            .unwrap_or("")
            .contains("cancelled"));
        assert!(!engine.has_pending_work());

        assert!(!engine.cancel_request(999));
    }

    #[test]
    fn test_submit_request() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 10,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(result.is_ok());
        assert!(engine.has_pending_work());
    }

    #[test]
    fn test_submit_request_rejects_non_greedy_params_on_greedy_only_backend() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        // temperature != 0.0：CPU 参考后端没有实现采样，必须诚实失败
        let sampled = GenerationParams {
            max_tokens: 10,
            temperature: 1.0,
            top_p: 1.0,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        let result = engine.submit_request("Hello", sampled);
        assert!(
            matches!(result, Err(EngineError::UnsupportedGenerationMode(_))),
            "non-zero temperature must be rejected, got {result:?}"
        );

        // top_p != 1.0：同样拒绝（即使 temperature == 0.0）
        let top_p = GenerationParams {
            max_tokens: 10,
            temperature: 0.0,
            top_p: 0.9,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        let result = engine.submit_request("Hello", top_p);
        assert!(
            matches!(result, Err(EngineError::UnsupportedGenerationMode(_))),
            "top_p != 1.0 must be rejected, got {result:?}"
        );

        // greedy 组合仍然接受
        let greedy = GenerationParams {
            max_tokens: 10,
            temperature: 0.0,
            top_p: 1.0,
            stop: Vec::new(),
            logprobs: None,
            priority: 0,
        };
        assert!(engine.submit_request("Hello", greedy).is_ok());
    }

    #[test]
    fn test_submit_empty_request() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams::default();
        let result = engine.submit_request("", params);

        assert!(result.is_err());
    }

    #[test]
    fn test_submit_invalid_params() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 0,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(result.is_err());
    }

    #[test]
    fn test_submit_request_rejects_total_length_over_limit() {
        let config = EngineConfig {
            max_model_len: 8,
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 4,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(matches!(result, Err(EngineError::TotalLengthTooLong(_, 8))));
    }

    #[test]
    fn test_submit_request_allows_total_length_at_limit() {
        let config = EngineConfig {
            max_model_len: 11,
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 4,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_submit_request_rejects_prompt_too_long_before_generation_check() {
        let config = EngineConfig {
            max_model_len: 3,
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let result = engine.submit_request("Hello", GenerationParams::default());
        assert!(matches!(result, Err(EngineError::InputTooLong(_, 3))));
    }

    #[test]
    fn test_step() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 5,
            ..GenerationParams::default()
        };

        engine.submit_request("Hi", params).unwrap();

        let mut completed = Vec::new();
        for _ in 0..10 {
            completed.extend(engine.step().unwrap());
        }

        assert_eq!(
            completed.len(),
            1,
            "request should complete within 10 steps"
        );
        assert!(completed[0].success);
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_completed_request_preserves_input_text() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].input_text.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_run_to_completion() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 3,
            ..GenerationParams::default()
        };

        engine.submit_request("Test", params).unwrap();

        let completed = engine.run();
        assert_eq!(completed.len(), 1, "the submitted request must complete");
        assert!(completed[0].success);
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_run_oversized_prompt_fails_not_stranded() {
        // B12 回归：所需 KV 块数超过内存水位线的长 prompt 会被调度器
        // 立即失败并排出——run() 必须返回该请求的失败终态，而不是遗留
        // 在 pending（调用方拿不到结果）。
        // 1000 字符 → 1002 tokens（BOS/EOS）→ ceil(1002/16)=63 块
        // > floor(64*0.9)=57 水位线，永久不可调度。
        let config = EngineConfig {
            max_num_blocks: 64,
            max_total_tokens: 1024,
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        engine
            .submit_request(
                &"a".repeat(1000),
                GenerationParams {
                    max_tokens: 4,
                    ..GenerationParams::default()
                },
            )
            .unwrap();

        let completed = engine.run();
        assert_eq!(completed.len(), 1, "请求不能被遗留，必须返回失败终态");
        assert!(!completed[0].success, "超预算请求应失败");
        let err = completed[0].error.as_ref().expect("失败请求应有错误信息");
        assert!(
            err.contains("watermark"),
            "错误应说明水位线限制，实际: {err}"
        );
        assert!(!engine.has_pending_work(), "run() 后不应残留 pending");
    }

    #[test]
    fn test_multiple_requests() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 2,
            ..GenerationParams::default()
        };

        for i in 0..3 {
            engine
                .submit_request(&format!("Request {}", i), params.clone())
                .unwrap();
        }

        let completed = engine.run();
        assert_eq!(completed.len(), 3, "all three requests must complete");
        assert!(completed.iter().all(|c| c.success));
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_executor_failure_marks_request_failed_and_clears_pending_work() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysFailExecutor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();

        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(completed[0].error.is_some());
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_executor_failure_updates_failure_metrics() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysFailExecutor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let _ = engine.run();

        let metrics = engine.get_metrics();
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.completed_requests, 0);
    }

    #[test]
    fn test_sequences_finished_called_on_normal_completion() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let (executor, finished) = RecordingExecutor::new(false);
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(executor),
        )
        .unwrap();

        engine
            .submit_request(
                "Hello",
                GenerationParams {
                    max_tokens: 1,
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(completed[0].success);
        assert_eq!(
            *finished.lock().unwrap(),
            vec![1],
            "normal completion must notify backend of the finished sequence"
        );
    }

    #[test]
    fn test_sequences_finished_called_on_execution_failure() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let (executor, finished) = RecordingExecutor::new(true);
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(executor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert_eq!(
            *finished.lock().unwrap(),
            vec![1],
            "execution failure must also notify backend to release KV"
        );
    }

    #[test]
    fn test_sequences_finished_called_on_cancel() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let (executor, finished) = RecordingExecutor::new(false);
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(executor),
        )
        .unwrap();

        let (request_id, _) = engine
            .submit_request(
                "Hello",
                GenerationParams {
                    max_tokens: 50,
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        engine.step().unwrap(); // 进入 decode
        assert!(engine.cancel_request(request_id));

        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert_eq!(
            completed[0].cancellation,
            Some(crate::CancellationReason::ClientDisconnected)
        );
        assert_eq!(engine.get_metrics().cancelled_requests, 1);
        assert_eq!(engine.get_metrics().failed_requests, 0);
        assert_eq!(
            *finished.lock().unwrap(),
            vec![1],
            "cancel must also notify backend to release KV"
        );
    }

    #[test]
    fn test_terminal_metrics_use_state_not_error_text() {
        use crate::CancellationReason;

        let config = create_test_config();
        let (executor, finished) = RecordingExecutor::new(false);
        let mut engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            Scheduler::new(config),
            Box::new(executor),
        )
        .unwrap();
        let (success, _) = engine.submit_request("success", test_params(1)).unwrap();
        let (client, _) = engine.submit_request("client", test_params(4)).unwrap();
        let (shutdown, _) = engine.submit_request("shutdown", test_params(4)).unwrap();
        let (failed, _) = engine.submit_request("failed", test_params(4)).unwrap();
        assert!(engine.cancel_request(client));
        assert!(engine.cancel_request_with_reason(shutdown, CancellationReason::ServerShutdown));
        // 同样的文案仍是 Failed，不能被字符串前缀误判为取消。
        assert!(engine.fail_request(failed, "request cancelled: backend failure"));
        let completed = engine.run();
        assert_eq!(completed.len(), 4);
        assert!(
            completed
                .iter()
                .find(|r| r.request_id == success)
                .unwrap()
                .success
        );
        assert_eq!(
            completed
                .iter()
                .find(|r| r.request_id == shutdown)
                .unwrap()
                .cancellation,
            Some(CancellationReason::ServerShutdown)
        );
        assert!(completed
            .iter()
            .find(|r| r.request_id == failed)
            .unwrap()
            .cancellation
            .is_none());
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_requests, 4);
        assert_eq!(metrics.completed_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.cancelled_requests, 2);
        assert_eq!(metrics.total_tokens_generated, 1);
        assert_eq!(metrics.active_sequences, 0);
        assert_eq!(metrics.memory_utilization, 0.0);
        assert!(engine.run().is_empty());
        assert!(!engine.cancel_request(client));
        let mut ids = finished.lock().unwrap().clone();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4],
            "all terminal resources released exactly once"
        );
        assert_eq!(engine.get_metrics().cancelled_requests, 2);
    }

    #[test]
    fn test_gpu_timeout_retries_once_then_succeeds() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(TimeoutThenSuccessExecutor { attempts: 0 }),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(completed[0].success);
        assert!(completed[0].error.is_none());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.completed_requests, 1);
    }

    #[test]
    fn test_streaming_chunks_concat_equals_final_output() {
        // PSERV-102 核心性质：所有流式片段拼接 == 最终一次性 decode 文本
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let (request_id, _) = engine
            .submit_request(
                "Hello stream",
                GenerationParams {
                    max_tokens: 8,
                    ..GenerationParams::default()
                },
            )
            .unwrap();

        let mut streamed = String::new();
        let mut final_text = None;
        while engine.has_pending_work() {
            let events = engine.step_events().unwrap();
            for (rid, chunk, _logprobs) in &events.chunks {
                assert_eq!(*rid, request_id);
                streamed.push_str(chunk);
            }
            for completed in events.completed {
                assert_eq!(completed.request_id, request_id);
                assert!(completed.success);
                final_text = Some(completed.output_text);
            }
        }

        assert_eq!(streamed, final_text.expect("request must complete"));
        assert!(!streamed.is_empty());
        assert_eq!(
            engine.num_live_decoders(),
            0,
            "decoder state must be dropped on completion"
        );
    }

    #[test]
    fn test_decoder_state_cleaned_up_on_cancel_and_failure() {
        // 取消路径：decoder 与 KV 资源一并清理
        let mut engine = InferenceEngine::new(create_test_config()).unwrap();
        let (request_id, _) = engine
            .submit_request(
                "cancel me",
                GenerationParams {
                    max_tokens: 50,
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        engine.step().unwrap();
        assert_eq!(engine.num_live_decoders(), 1);

        assert!(engine.cancel_request(request_id));
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert_eq!(engine.num_live_decoders(), 0);

        // 后端失败路径：decoder 同样被清理
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysFailExecutor),
        )
        .unwrap();
        engine
            .submit_request("fail me", GenerationParams::default())
            .unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert_eq!(engine.num_live_decoders(), 0);
    }

    /// push 阶段解码失败的 tokenizer：引擎必须让请求失败，而不是静默丢文本。
    struct FailingDecodeTokenizer(SimpleTokenizer);

    impl TokenizerTrait for FailingDecodeTokenizer {
        fn try_encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
            self.0.try_encode(text)
        }
        fn try_decode(&self, tokens: &[TokenId]) -> Result<String, String> {
            self.0.try_decode(tokens)
        }
        fn vocab_size(&self) -> u32 {
            self.0.vocab_size()
        }
        fn bos_token_id(&self) -> TokenId {
            self.0.bos_token_id()
        }
        fn eos_token_id(&self) -> TokenId {
            self.0.eos_token_id()
        }
        fn pad_token_id(&self) -> TokenId {
            self.0.pad_token_id()
        }
        fn create_decoder(&self) -> Box<dyn crate::tokenizer::IncrementalDecoder> {
            struct AlwaysFailDecoder;
            impl crate::tokenizer::IncrementalDecoder for AlwaysFailDecoder {
                fn push(&mut self, _token: TokenId) -> Result<Option<String>, String> {
                    Err("simulated incremental decode failure".to_string())
                }
                fn finish(&mut self) -> Result<Option<String>, String> {
                    Ok(None)
                }
            }
            Box::new(AlwaysFailDecoder)
        }
    }

    /// 单 token 解码返回固定字符串的测试 tokenizer：token 0 → "好"（3 字节）、
    /// 1 → "A"、2 → "B"。用于验证 stop 序列的**字节偏移**处理。
    struct FixedSegmentTokenizer;

    impl TokenizerTrait for FixedSegmentTokenizer {
        fn try_encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
            // 非空文本编码为单个 token（让请求有合法输入）。
            if text.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![0])
            }
        }
        fn try_decode(&self, tokens: &[TokenId]) -> Result<String, String> {
            let mut s = String::new();
            for &t in tokens {
                match t {
                    0 => s.push('好'),
                    1 => s.push('A'),
                    2 => s.push('B'),
                    _ => s.push('?'),
                }
            }
            Ok(s)
        }
        fn vocab_size(&self) -> u32 {
            3
        }
        fn bos_token_id(&self) -> TokenId {
            0
        }
        fn eos_token_id(&self) -> TokenId {
            0
        }
        fn pad_token_id(&self) -> TokenId {
            0
        }
        fn create_decoder(&self) -> Box<dyn crate::tokenizer::IncrementalDecoder> {
            struct NoopDecoder;
            impl crate::tokenizer::IncrementalDecoder for NoopDecoder {
                fn push(&mut self, _token: TokenId) -> Result<Option<String>, String> {
                    Ok(None)
                }
                fn finish(&mut self) -> Result<Option<String>, String> {
                    Ok(None)
                }
            }
            Box::new(NoopDecoder)
        }
    }

    #[test]
    fn test_decoder_push_failure_fails_request_honestly() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(FailingDecodeTokenizer(SimpleTokenizer::new())),
            scheduler,
            Box::new(crate::test_utils::ConstantTokenExecutor { token: 65 }),
        )
        .unwrap();

        engine
            .submit_request("decode failure", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(
            completed[0]
                .error
                .as_deref()
                .unwrap_or("")
                .contains("tokenizer decode failed"),
            "push failure must surface as tokenizer error, got {:?}",
            completed[0].error
        );
        assert_eq!(engine.num_live_decoders(), 0);
    }

    /// HuggingFace 流式解码产生的中间片段与请求完成时的最终文本必须严格等价。
    #[test]
    fn test_streaming_equivalence_with_huggingface_decoder() {
        let path = write_test_tokenizer_json();
        let config = EngineConfig {
            max_model_len: 64,
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some(path.clone()),
            },
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let (request_id, _) = engine
            .submit_request(
                "hello world",
                GenerationParams {
                    max_tokens: 4,
                    ..GenerationParams::default()
                },
            )
            .unwrap();

        let mut streamed = String::new();
        let mut final_text = None;
        while engine.has_pending_work() {
            let events = engine.step_events().unwrap();
            for (rid, chunk, _logprobs) in &events.chunks {
                assert_eq!(*rid, request_id);
                streamed.push_str(chunk);
            }
            for completed in events.completed {
                final_text = Some(completed.output_text);
            }
        }

        assert_eq!(streamed, final_text.expect("request must complete"));
        assert_eq!(engine.num_live_decoders(), 0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_gpu_timeout_exhausts_retries_then_fails_request() {
        // 执行器持续超时：重试 max_retry_attempts 次后必须放弃，
        // 将请求标记为失败并释放资源，而不是无限重试或静默挂起。
        let config = create_test_config(); // max_retry_attempts: 2
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysTimeoutExecutor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(
            completed[0].error.as_deref().unwrap_or("").contains("GPU"),
            "error should surface the GPU timeout, got {:?}",
            completed[0].error
        );
        assert!(!engine.has_pending_work());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.completed_requests, 0);
    }

    #[test]
    fn test_non_retry_safe_timeout_fails_without_replay() {
        // 非幂等后端（retry_safe=false）：GpuTimeout 后不得重放同一 batch，
        // execute 必须只被调用 1 次，请求直接失败。
        let config = create_test_config(); // max_retry_attempts: 2
        let scheduler = Scheduler::new(config.clone());
        let (executor, attempts) = NonRetrySafeTimeoutExecutor::new();
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(executor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(
            completed[0].error.as_deref().unwrap_or("").contains("GPU"),
            "error should surface the GPU timeout, got {:?}",
            completed[0].error
        );
        assert!(!engine.has_pending_work());
        assert_eq!(
            *attempts.lock().unwrap(),
            1,
            "non-retry-safe backend must not replay the batch on timeout"
        );
    }

    #[test]
    fn test_completed_request_reports_finish_reason() {
        // PSERV-103：成功完成的请求必须区分停止原因——
        // 达到 max_tokens 截断 → Length，末 token 为 EOS → Stop。
        let config = create_test_config();
        // EOS 动态取自 tokenizer 而非硬编码，避免依赖默认 special token 配置。
        let eos = SimpleTokenizer::new().eos_token_id();
        // 恒生成一个确定 != EOS 的合法字符 id（65 是 'A'；若冲突则换 66）。
        let non_eos = if eos == 65 { 66 } else { 65 };

        // Length：执行器生成非 EOS token，请求被 max_tokens 截断。
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(ConstantTokenExecutor { token: non_eos }),
        )
        .unwrap();
        engine.submit_request("hi", test_params(3)).unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].finish_reason, Some(FinishReason::Length));

        // Stop：执行器每一步都生成 EOS，请求在首个 token 后自然停止。
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(ConstantTokenExecutor { token: eos }),
        )
        .unwrap();
        engine.submit_request("hi", test_params(3)).unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_stop_sequence_truncates_output_and_reports_stop() {
        // PSERV-105：输出文本命中 stop 序列时请求立即停止；stop 序列本身
        // 从输出中移除（保留此前缀），finish_reason 为 Stop。
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            scheduler,
            // 逐 token 输出 A, A, B（SimpleTokenizer: 'A'=37, 'B'=38）：
            // 第 3 步文本 "AAB" 命中 "AB"@1
            Box::new(SequenceExecutor::new(vec![37, 37, 38])),
        )
        .unwrap();
        engine
            .submit_request(
                "hi",
                GenerationParams {
                    max_tokens: 10,
                    stop: vec!["AB".to_string()],
                    logprobs: None,
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        let result = &completed[0];
        assert!(result.success);
        // "AAB" 中 "AB" 起始于字符偏移 1 → 保留第 1 个 A，移除 AB
        assert_eq!(result.output_tokens, vec![37]);
        assert_eq!(result.output_text, "A");
        assert_eq!(result.finish_reason, Some(FinishReason::Stop));
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_tokens_before_char_uses_byte_offset_for_unicode() {
        // tokens: 好(0), A(1), B(2)；"好" 占 3 字节。
        // `str::find("AB")` 在 "好AB" 中返回字节偏移 3，应只保留第一个 token。
        let tokens = vec![0u32, 1, 2];
        let tokenizer = FixedSegmentTokenizer;
        let keep = tokens_before_char(&tokens, &tokenizer, "好".len());
        assert_eq!(keep, 1, "only the first token (好) should be kept");
    }

    #[test]
    fn test_stop_sequence_unicode_byte_offset_end_to_end() {
        // 输出 好AB，stop="AB"：按字节偏移截断后最终输出应为 "好"，
        // finish_reason="stop"（修复前会保留 stop 序列 "AB"）。
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(FixedSegmentTokenizer),
            scheduler,
            // 逐 token 输出 0(好), 1(A), 2(B)
            Box::new(SequenceExecutor::new(vec![0, 1, 2])),
        )
        .unwrap();
        engine
            .submit_request(
                "hi",
                GenerationParams {
                    max_tokens: 10,
                    stop: vec!["AB".to_string()],
                    logprobs: None,
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        let result = &completed[0];
        assert!(result.success);
        assert_eq!(result.output_text, "好");
        assert_eq!(result.finish_reason, Some(FinishReason::Stop));
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_stop_sequence_unmatched_reports_length() {
        // PSERV-105：stop 序列未命中时，请求照常以 max_tokens 截断结束（Length）。
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(ConstantTokenExecutor { token: 37 }), // 'A' A A ...
        )
        .unwrap();
        engine
            .submit_request(
                "hi",
                GenerationParams {
                    max_tokens: 3,
                    stop: vec!["Z".to_string()], // 永不出现
                    logprobs: None,
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(completed[0].success);
        assert_eq!(completed[0].output_text, "AAA");
        assert_eq!(completed[0].finish_reason, Some(FinishReason::Length));
    }
}
