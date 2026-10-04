//! Continuous Batching 调度器
//!
//! 实现请求调度，支持 prefill/decode 阶段管理和内存感知的批次构建。
//!
//! # 核心特性
//!
//! - **Decode 优先调度** - 优先调度 decode 请求以降低延迟
//! - **内存压力感知** - 内存超阈值时拒绝新 prefill
//! - **连续批处理** - 动态组合 prefill 和 decode 请求
//! - **优先级调度（PSERV-112）** - `GenerationParams::priority` 越大越先调度
//!   （prefill 启动、在途 prefill 组合与 decode 内部排序均生效）；
//!   同级保持 FCFS，默认优先级 0
//!
//! # 状态机
//!
//! ```text
//! Pending → Prefill → Decode → Completed
//!                   ↘ Failed
//! ```

use std::collections::{BTreeMap, VecDeque};

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::kv_cache::KVCacheManager;
use crate::types::{
    CancellationReason, ExecutionOutput, PhysicalBlockRef, Request, RequestId, RequestState,
    SchedulerOutput, SeqId, Sequence, TokenId,
};

/// 保留错误文案前缀；终态分类使用 CancellationReason，不解析此字符串。
pub const CANCEL_REASON_PREFIX: &str = "request cancelled";

#[derive(Debug, Clone)]
struct PendingRequest {
    seq_id: SeqId,
    request: Request,
}

/// Continuous Batching 调度器
///
/// 管理序列生命周期（Pending → Prefill → Decode → Completed/Failed）、
/// KV Cache 资源、以及调度策略（decode 优先、批次约束、内存压力门控）。
pub struct Scheduler {
    config: EngineConfig,
    kv_cache: KVCacheManager,
    pending_queue: VecDeque<PendingRequest>,
    // BTreeMap（而非 HashMap）使调度迭代顺序按 seq_id 单调递增，
    // 即先提交先调度（FCFS），保证公平性并让调度行为可复现。
    prefill_sequences: BTreeMap<SeqId, Sequence>,
    decode_sequences: BTreeMap<SeqId, Sequence>,
    // 携带 seq_id 的终态请求：引擎需要据此通知后端释放物理 KV 资源。
    completed_requests: Vec<(SeqId, Request)>,
    next_seq_id: SeqId,
    under_memory_pressure: bool,
}

impl Scheduler {
    /// 创建调度器
    ///
    /// # Panics
    ///
    /// 要求 `config.block_size > 0`（否则块数计算会除零 panic）。
    /// 经由 `InferenceEngine` 构造时 `EngineConfig::validate` 已保证这一点；
    /// 直接构造底层调度器的调用方需自行确保配置有效。
    pub fn new(config: EngineConfig) -> Self {
        debug_assert!(
            config.block_size > 0,
            "config.block_size must be greater than 0"
        );
        let kv_cache = KVCacheManager::new(config.max_num_blocks, config.block_size);
        Self {
            config,
            kv_cache,
            pending_queue: VecDeque::new(),
            prefill_sequences: BTreeMap::new(),
            decode_sequences: BTreeMap::new(),
            completed_requests: Vec::new(),
            next_seq_id: 1,
            under_memory_pressure: false,
        }
    }

    // === 公共接口 ===

    pub fn add_request(&mut self, request: Request) -> Result<SeqId, EngineError> {
        self.update_memory_pressure();

        if self.under_memory_pressure {
            return Err(EngineError::MemoryPressure);
        }

        let total_sequences = self.pending_queue.len() + self.num_active_sequences();
        if total_sequences >= self.config.max_num_seqs as usize {
            return Err(EngineError::MaxConcurrentSequencesReached(
                self.config.max_num_seqs,
            ));
        }

        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        self.pending_queue
            .push_back(PendingRequest { seq_id, request });
        Ok(seq_id)
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();
        let mut total_tokens: u32 = 0;
        let mut num_sequences: u32 = 0;

        self.update_memory_pressure();

        // Priority 1: Schedule decode sequences first (lower latency for in-flight requests)
        for seq_id in self.decode_seq_ids() {
            if num_sequences >= self.config.max_batch_size {
                break;
            }

            if total_tokens.saturating_add(1) > self.config.max_total_tokens {
                break;
            }

            if let Err(reason) = self.grow_decode_sequence(seq_id) {
                self.fail_sequence(seq_id, &reason);
                continue;
            }

            if let Some(sequence) = self.decode_sequences.get(&seq_id) {
                output.decode_sequences.push(sequence.clone());
                total_tokens = total_tokens.saturating_add(1);
                num_sequences += 1;
            }
        }

        // Priority 2: Schedule prefill sequences（PSERV-112：高优先级先，
        // 同级保持 seq_id 顺序 = FCFS）
        let mut prefill_candidates: Vec<SeqId> = self.prefill_seq_ids();
        prefill_candidates.sort_by_key(|&sid| {
            let prio = self
                .prefill_sequences
                .get(&sid)
                .map(|sq| sq.request.params.priority)
                .unwrap_or(0);
            (std::cmp::Reverse(prio), sid)
        });
        for seq_id in prefill_candidates {
            if num_sequences >= self.config.max_batch_size {
                break;
            }

            let (prefill_tokens, blocks_needed) = match self.prefill_sequences.get(&seq_id) {
                Some(sequence) => {
                    let tokens = sequence.request.input_tokens.len() as u32;
                    (tokens, self.config.blocks_for_tokens(tokens))
                }
                None => continue,
            };

            if prefill_tokens > self.config.max_total_tokens {
                let reason = format!(
                    "Input tokens {} exceed max_total_tokens {}",
                    prefill_tokens, self.config.max_total_tokens
                );
                self.fail_sequence(seq_id, &reason);
                continue;
            }

            if blocks_needed > self.config.max_num_blocks {
                let reason = format!(
                    "Required blocks {} exceed max_num_blocks {}",
                    blocks_needed, self.config.max_num_blocks
                );
                self.fail_sequence(seq_id, &reason);
                continue;
            }

            // continue 而非 break：装不下本步的大 prefill 不应阻塞后面的小 prefill
            if total_tokens.saturating_add(prefill_tokens) > self.config.max_total_tokens {
                continue;
            }

            if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                output.prefill_sequences.push(sequence.clone());
                total_tokens = total_tokens.saturating_add(prefill_tokens);
                num_sequences += 1;
            }
        }

        // Priority 3: Start new prefills from pending queue (if not under memory pressure)
        if !self.under_memory_pressure {
            // PSERV-112：先按优先级稳定排序（高优先级在前，同级保持 FCFS），
            // 再做一轮扫描。
            let mut pending_vec: Vec<PendingRequest> = self.pending_queue.drain(..).collect();
            pending_vec.sort_by_key(|p| (std::cmp::Reverse(p.request.params.priority), p.seq_id));
            self.pending_queue = pending_vec.into();

            // 只扫描"进入本步时"的队列长度一轮：装不下的请求 push_back 留到后续步骤，
            // 而不是 push_front + break——否则一个大 prefill 会永久挡住后面的小 prefill。
            // 这也让内存预算不足的候选能被跳过，尝试后面的小请求。
            let pending_count = self.pending_queue.len();
            for _ in 0..pending_count {
                let Some(pending) = self.pending_queue.pop_front() else {
                    break;
                };
                let (seq_id, request) = (pending.seq_id, pending.request);

                if num_sequences >= self.config.max_batch_size {
                    self.pending_queue
                        .push_back(PendingRequest { seq_id, request });
                    continue;
                }

                let prefill_tokens = request.input_tokens.len() as u32;
                if prefill_tokens > self.config.max_total_tokens {
                    let mut failed_request = request;
                    failed_request.state = RequestState::Failed(format!(
                        "Input tokens {} exceed max_total_tokens {}",
                        prefill_tokens, self.config.max_total_tokens
                    ));
                    self.completed_requests.push((seq_id, failed_request));
                    continue;
                }

                let blocks_needed = self.config.blocks_for_tokens(prefill_tokens);
                if blocks_needed > self.config.max_num_blocks {
                    let mut failed_request = request;
                    failed_request.state = RequestState::Failed(format!(
                        "Required blocks {} exceed max_num_blocks {}",
                        blocks_needed, self.config.max_num_blocks
                    ));
                    self.completed_requests.push((seq_id, failed_request));
                    continue;
                }

                if total_tokens.saturating_add(prefill_tokens) > self.config.max_total_tokens {
                    self.pending_queue
                        .push_back(PendingRequest { seq_id, request });
                    continue;
                }

                // 永久不可调度：所需块数超过内存水位线上限，池子即使完全空闲
                // 也无法容纳（has_prefill_budget 会永远返回 false）。若放回
                // pending_queue，has_pending_work() 恒真而每步零进展，服务层
                // 的 engine_loop 将无限空转（B12 服务层）。因此直接失败并给出
                // 明确错误，而不是反复重试。
                let stats = self.kv_cache.get_memory_stats();
                let watermark_blocks =
                    (stats.total_blocks as f32 * self.config.memory_threshold).floor() as u32;
                if blocks_needed > watermark_blocks {
                    let mut failed_request = request;
                    failed_request.state = RequestState::Failed(format!(
                        "Required blocks {blocks_needed} exceed memory watermark \
                         {watermark_blocks} (max_num_blocks {}, threshold {}); \
                         this prompt can never be scheduled",
                        self.config.max_num_blocks, self.config.memory_threshold
                    ));
                    self.completed_requests.push((seq_id, failed_request));
                    continue;
                }

                // 内存水位线 + decode 增长预留：预算不足的候选延后，而不是把池子打满
                // 或让下一步 decode 增长时 OOM。（此处仅剩"其他序列占用中"的临时等待。）
                if !self.has_prefill_budget(blocks_needed, prefill_tokens) {
                    self.pending_queue
                        .push_back(PendingRequest { seq_id, request });
                    continue;
                }

                match self.try_start_prefill(seq_id, request) {
                    Ok(seq_id) => {
                        if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                            output.prefill_sequences.push(sequence.clone());
                            total_tokens = total_tokens.saturating_add(prefill_tokens);
                            num_sequences += 1;
                        }
                    }
                    Err((seq_id, request)) => {
                        self.pending_queue.push_back(PendingRequest {
                            seq_id,
                            request: *request,
                        });
                        continue;
                    }
                }
            }
        }

        output.total_tokens = total_tokens;
        output
    }

    pub fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId) {
        let mut to_complete = Vec::new();
        let mut to_fail = Vec::new();
        let max_model_len = self.config.max_model_len as usize;

        for (i, &seq_id) in outputs.seq_ids.iter().enumerate() {
            let Some(next_token) = outputs.next_tokens.get(i).copied() else {
                to_fail.push((
                    seq_id,
                    format!("Malformed execution output: missing next token for seq_id {seq_id}"),
                ));
                continue;
            };

            if self.prefill_sequences.contains_key(&seq_id) {
                self.transition_prefill_to_decode(seq_id);
            }

            if let Some(sequence) = self.decode_sequences.get_mut(&seq_id) {
                sequence.request.output_tokens.push(next_token);
                // 累积 logprobs（仅请求启用时；后端未提供则跳过）
                if sequence.request.params.logprobs.is_some() {
                    if let Some(Some(lp)) = outputs.logprobs.get(i) {
                        sequence.request.logprobs.push(lp.clone());
                    }
                }
                sequence.num_generated_tokens += 1;

                if sequence.request.total_tokens() >= max_model_len
                    || sequence.request.is_complete(eos_token_id)
                {
                    to_complete.push(seq_id);
                }
            } else {
                to_fail.push((
                    seq_id,
                    format!("Malformed execution output: unknown seq_id {seq_id}"),
                ));
            }
        }

        for (seq_id, reason) in to_fail {
            self.fail_sequence(seq_id, &reason);
        }
        for seq_id in to_complete {
            self.complete_sequence(seq_id);
        }
    }

    /// 排出所有终态请求（兼容旧调用方，丢弃 seq_id）。
    pub fn get_completed(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.completed_requests)
            .into_iter()
            .map(|(_, request)| request)
            .collect()
    }

    /// 排出所有终态请求并携带其 seq_id（引擎据此通知后端释放物理 KV）。
    pub fn take_completed_with_seq_ids(&mut self) -> Vec<(SeqId, Request)> {
        std::mem::take(&mut self.completed_requests)
    }

    /// 按 request_id 只读查询请求（pending / prefill / decode / 已完成的缓冲）。
    ///
    /// 完成但尚未被 `get_completed` 取走的请求也可查到：本步 push 循环
    /// 需要在请求完成（序列已移出）后仍能读取其最后一个 token 的 logprobs。
    pub fn get_request_by_id(&self, request_id: RequestId) -> Option<&Request> {
        self.pending_queue
            .iter()
            .find(|p| p.request.id == request_id)
            .map(|p| &p.request)
            .or_else(|| {
                self.prefill_sequences
                    .values()
                    .chain(self.decode_sequences.values())
                    .find(|seq| seq.request.id == request_id)
                    .map(|seq| &seq.request)
            })
            .or_else(|| {
                self.completed_requests
                    .iter()
                    .find(|(_, r)| r.id == request_id)
                    .map(|(_, r)| r)
            })
    }

    /// 因命中 stop 序列而终止请求（PSERV-105）：
    /// 把输出 token 截断到 stop 序列之前、标记 `stop_sequence_hit` 并完成。
    /// 仅可能命中已开始生成的 prefill/decode 请求（pending 无输出 token）。
    /// 返回是否找到了对应请求。
    pub fn complete_by_stop_sequence(&mut self, request_id: RequestId, keep_tokens: usize) -> bool {
        let seq_id = self
            .prefill_sequences
            .values()
            .chain(self.decode_sequences.values())
            .find(|seq| seq.request.id == request_id)
            .map(|seq| seq.seq_id);
        match seq_id {
            Some(seq_id) => {
                if let Some(seq) = self
                    .prefill_sequences
                    .get_mut(&seq_id)
                    .or_else(|| self.decode_sequences.get_mut(&seq_id))
                {
                    seq.request.output_tokens.truncate(keep_tokens);
                    seq.request.logprobs.truncate(keep_tokens);
                    seq.request.stop_sequence_hit = true;
                }
                self.complete_sequence(seq_id);
                true
            }
            None => false,
        }
    }

    pub fn has_pending_work(&self) -> bool {
        !self.pending_queue.is_empty()
            || !self.prefill_sequences.is_empty()
            || !self.decode_sequences.is_empty()
    }

    pub fn get_memory_utilization(&self) -> f32 {
        self.kv_cache.get_memory_stats().utilization()
    }

    pub fn fail_sequences<I>(&mut self, seq_ids: I, reason: &str)
    where
        I: IntoIterator<Item = SeqId>,
    {
        for seq_id in seq_ids {
            self.fail_sequence(seq_id, reason);
        }
    }

    #[cfg(test)]
    fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.prefill_sequences
            .get(&seq_id)
            .or_else(|| self.decode_sequences.get(&seq_id))
    }

    pub fn num_active_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }

    /// 按 request_id 取消请求：无论它处于 pending、prefill 还是 decode，
    /// 都将其标记取消并释放 KV 资源。返回是否真的取消到了东西。
    pub fn cancel_by_request_id(&mut self, request_id: RequestId) -> bool {
        self.cancel_by_request_id_with_reason(request_id, CancellationReason::ClientDisconnected)
    }

    pub fn cancel_by_request_id_with_reason(
        &mut self,
        request_id: RequestId,
        reason: CancellationReason,
    ) -> bool {
        self.terminate_by_request_id(request_id, RequestState::Cancelled(reason))
    }

    /// 按 request_id 将请求标记为失败（释放 KV 资源），失败原因随请求排出。
    /// 返回是否找到了对应请求。
    pub fn fail_by_request_id(&mut self, request_id: RequestId, reason: &str) -> bool {
        self.terminate_by_request_id(request_id, RequestState::Failed(reason.to_string()))
    }

    fn terminate_by_request_id(&mut self, request_id: RequestId, state: RequestState) -> bool {
        let seq_id = self
            .pending_queue
            .iter()
            .find(|p| p.request.id == request_id)
            .map(|p| p.seq_id)
            .or_else(|| {
                self.prefill_sequences
                    .values()
                    .chain(self.decode_sequences.values())
                    .find(|seq| seq.request.id == request_id)
                    .map(|seq| seq.seq_id)
            });
        match seq_id {
            Some(seq_id) => {
                self.terminate_sequence(seq_id, state);
                true
            }
            None => false,
        }
    }

    /// 把等待队列中的全部请求标记为失败（stall 兜底）。
    ///
    /// 服务层检测到连续空转（无完成排出、无活跃序列，pending 却非空）时调用：
    /// 此时 pending 请求因 KV 预算/块不足而永远无法启动，若不主动失败，
    /// `has_pending_work()` 恒真而每步零进展，engine_loop 会无限忙等。
    /// 返回被失败处理的请求数。
    pub fn fail_pending_requests(&mut self, reason: &str) -> usize {
        let mut failed = 0;
        for pending in self.pending_queue.drain(..) {
            let mut request = pending.request;
            request.state = RequestState::Failed(format!("{reason}（request {}）", request.id));
            self.completed_requests.push((pending.seq_id, request));
            failed += 1;
        }
        failed
    }

    #[cfg(test)]
    fn is_in_exactly_one_queue(&self, seq_id: SeqId) -> bool {
        let in_pending = self.pending_queue.iter().any(|p| p.seq_id == seq_id);
        let in_prefill = self.prefill_sequences.contains_key(&seq_id);
        let in_decode = self.decode_sequences.contains_key(&seq_id);
        let count = [in_pending, in_prefill, in_decode]
            .iter()
            .filter(|&&flag| flag)
            .count();
        count == 1
    }

    #[cfg(test)]
    fn has_prefill_sequence(&self, seq_id: SeqId) -> bool {
        self.prefill_sequences.contains_key(&seq_id)
    }

    #[cfg(test)]
    fn has_decode_sequence(&self, seq_id: SeqId) -> bool {
        self.decode_sequences.contains_key(&seq_id)
    }

    #[cfg(test)]
    fn has_pending_request(&self, seq_id: SeqId) -> bool {
        self.pending_queue.iter().any(|p| p.seq_id == seq_id)
    }

    #[cfg(test)]
    fn num_decode_sequences(&self) -> usize {
        self.decode_sequences.len()
    }

    // === 内部方法 ===

    fn update_memory_pressure(&mut self) {
        let stats = self.kv_cache.get_memory_stats();
        self.under_memory_pressure = stats.utilization() >= self.config.memory_threshold;
    }

    /// 一个 prefill 在首 token 生成后的下一步，是否需要多分配 1 个 block。
    fn prefill_needs_decode_reserve(&self, prompt_tokens: u32) -> bool {
        let prompt_blocks = self.config.blocks_for_tokens(prompt_tokens);
        self.config
            .blocks_for_tokens(prompt_tokens.saturating_add(2))
            > prompt_blocks
    }

    /// 当前已调度/在跑的序列在"本步执行完成后、下下步 grow 时"需要的新增块数。
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

    /// 启动一个 prefill 是否安全（高水位线 + 即时 decode 增长预留）。
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

    fn try_start_prefill(
        &mut self,
        seq_id: SeqId,
        request: Request,
    ) -> Result<SeqId, (SeqId, Box<Request>)> {
        let num_tokens = request.input_tokens.len() as u32;
        let blocks_needed = self.config.blocks_for_tokens(num_tokens);

        if !self.kv_cache.can_allocate(blocks_needed) {
            return Err((seq_id, Box::new(request)));
        }

        if self.kv_cache.allocate_sequence(seq_id, num_tokens).is_err() {
            return Err((seq_id, Box::new(request)));
        }

        let mut sequence = Sequence::new(seq_id, request);
        sequence.request.state = RequestState::Prefill;

        if let Some(block_table) = self.kv_cache.get_block_table(seq_id) {
            sequence.logical_blocks = block_table
                .iter()
                .map(|&block_idx| PhysicalBlockRef { block_idx })
                .collect();
        }

        self.prefill_sequences.insert(seq_id, sequence);
        Ok(seq_id)
    }

    fn grow_decode_sequence(&mut self, seq_id: SeqId) -> Result<(), String> {
        let Some(seq) = self.decode_sequences.get(&seq_id) else {
            return Err(format!("Decode sequence not found: {seq_id}"));
        };

        let current_tokens = seq.context_len();
        let current_blocks = seq.logical_blocks.len() as u32;
        let blocks_needed = self
            .config
            .blocks_for_tokens(current_tokens.saturating_add(1));

        if blocks_needed > current_blocks {
            match self.kv_cache.allocate_block(seq_id) {
                Ok(physical_ref) => {
                    let Some(sequence) = self.decode_sequences.get_mut(&seq_id) else {
                        return Err(format!("Decode sequence not found: {seq_id}"));
                    };
                    sequence.logical_blocks.push(physical_ref);
                }
                Err(err) => {
                    return Err(format!("Failed to allocate KV block: {}", err));
                }
            }
        }
        Ok(())
    }

    fn transition_prefill_to_decode(&mut self, seq_id: SeqId) {
        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Decode;
            self.decode_sequences.insert(seq_id, sequence);
        }
    }

    fn complete_sequence(&mut self, seq_id: SeqId) {
        if let Some(mut sequence) = self.decode_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Completed;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push((seq_id, sequence.request));
            return;
        }

        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Completed;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push((seq_id, sequence.request));
        }
    }

    fn fail_sequence(&mut self, seq_id: SeqId, reason: &str) {
        self.terminate_sequence(seq_id, RequestState::Failed(reason.to_string()));
    }

    fn terminate_sequence(&mut self, seq_id: SeqId, state: RequestState) {
        if let Some(mut sequence) = self.decode_sequences.remove(&seq_id) {
            sequence.request.state = state;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push((seq_id, sequence.request));
            return;
        }

        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = state;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push((seq_id, sequence.request));
            return;
        }

        if let Some(index) = self.pending_queue.iter().position(|p| p.seq_id == seq_id) {
            if let Some(mut pending) = self.pending_queue.remove(index) {
                pending.request.state = state;
                self.completed_requests.push((seq_id, pending.request));
            }
        }
    }

    /// decode 序列按 priority 降序（同级保持 seq_id 顺序 = FCFS），
    /// 与 prefill 的优先级语义一致，避免高优先级在途请求的尾延迟被同级排队掩盖。
    fn decode_seq_ids(&self) -> Vec<SeqId> {
        let mut ids: Vec<SeqId> = self.decode_sequences.keys().copied().collect();
        ids.sort_by_key(|&sid| {
            let prio = self
                .decode_sequences
                .get(&sid)
                .map(|sq| sq.request.params.priority)
                .unwrap_or(0);
            (std::cmp::Reverse(prio), sid)
        });
        ids
    }

    fn prefill_seq_ids(&self) -> Vec<SeqId> {
        self.prefill_sequences.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        create_test_config, create_test_request, create_test_request_with_params,
    };
    use crate::types::GenerationParams;

    #[test]
    fn test_add_request() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(42, 32);
        let result = scheduler.add_request(request);

        assert_eq!(result.unwrap(), 1);
        assert!(scheduler.has_pending_work());
    }

    #[test]
    fn test_add_request_returns_real_seq_id() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(999, 16);
        let seq_id = scheduler.add_request(request).unwrap();

        assert_eq!(seq_id, 1);
    }

    #[test]
    fn test_schedule_prefill_uses_returned_seq_id() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(999, 16);
        let seq_id = scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1);
        assert_eq!(output.prefill_sequences[0].seq_id, seq_id);
    }

    #[test]
    fn test_pending_queue_counts_toward_max_sequences() {
        let config = EngineConfig {
            max_num_seqs: 1,
            ..create_test_config()
        };
        let mut scheduler = Scheduler::new(config);

        let first = create_test_request(1, 16);
        let second = create_test_request(2, 16);

        assert!(scheduler.add_request(first).is_ok());
        assert!(matches!(
            scheduler.add_request(second),
            Err(EngineError::MaxConcurrentSequencesReached(1))
        ));
    }

    #[test]
    fn test_add_request_sequence_ids_are_monotonic() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let seq_id1 = scheduler.add_request(create_test_request(100, 8)).unwrap();
        let seq_id2 = scheduler.add_request(create_test_request(200, 8)).unwrap();

        assert_eq!(seq_id1, 1);
        assert_eq!(seq_id2, 2);
    }

    #[test]
    fn test_schedule_prefill() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(1, 32);
        scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();

        assert_eq!(output.prefill_sequences.len(), 1);
        assert_eq!(output.decode_sequences.len(), 0);
    }

    #[test]
    fn test_prefill_to_decode_transition() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(1, 32);
        scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1);

        let seq_id = output.prefill_sequences[0].seq_id;

        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            seq_ids: vec![seq_id],
            logprobs: Vec::new(),
        };

        scheduler.update_sequences(&exec_output, 0);

        assert!(scheduler.has_decode_sequence(seq_id));
        assert!(!scheduler.has_prefill_sequence(seq_id));
    }

    #[test]
    fn test_decode_priority() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request1 = create_test_request(1, 16);
        scheduler.add_request(request1).unwrap();
        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;

        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            seq_ids: vec![seq_id],
            logprobs: Vec::new(),
        };
        scheduler.update_sequences(&exec_output, 0);

        let request2 = create_test_request(2, 16);
        scheduler.add_request(request2).unwrap();

        let output = scheduler.schedule();

        assert!(!output.decode_sequences.is_empty());
    }

    #[test]
    fn test_completion() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let mut request = create_test_request(1, 16);
        request.params.max_tokens = 2;
        scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;

        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            seq_ids: vec![seq_id],
            logprobs: Vec::new(),
        };
        scheduler.update_sequences(&exec_output, 0);

        scheduler.schedule();
        let exec_output = ExecutionOutput {
            next_tokens: vec![101],
            seq_ids: vec![seq_id],
            logprobs: Vec::new(),
        };
        scheduler.update_sequences(&exec_output, 0);

        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 1);
    }

    #[test]
    fn test_decode_priority_with_small_batch_keeps_pending_request_queued() {
        let config = EngineConfig {
            max_batch_size: 1,
            max_total_tokens: 64,
            ..create_test_config()
        };
        let mut scheduler = Scheduler::new(config);

        let decode_request = create_test_request(1, 16);
        let pending_request = create_test_request(2, 16);

        scheduler.add_request(decode_request).unwrap();
        let output = scheduler.schedule();
        let decode_seq_id = output.prefill_sequences[0].seq_id;

        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![100],
                seq_ids: vec![decode_seq_id],
                logprobs: Vec::new(),
            },
            0,
        );

        let pending_seq_id = scheduler.add_request(pending_request).unwrap();
        let scheduled = scheduler.schedule();

        assert_eq!(scheduled.decode_sequences.len(), 1);
        assert_eq!(scheduled.prefill_sequences.len(), 0);
        assert_eq!(scheduled.decode_sequences[0].seq_id, decode_seq_id);
        assert!(scheduler.has_pending_request(pending_seq_id));
    }

    #[test]
    fn test_priority_higher_prefill_starts_first() {
        // PSERV-112：预算只够一个 prefill 时，高优先级请求先被调度启动，
        // 即使它后提交。
        let mut config = create_test_config();
        config.max_total_tokens = 40; // 只够一个 32-token prefill
        let mut scheduler = Scheduler::new(config);

        scheduler
            .add_request(Request::new(1, vec![1; 32], GenerationParams::default()))
            .unwrap();
        let high_params = GenerationParams {
            priority: 5,
            ..GenerationParams::default()
        };
        scheduler
            .add_request(Request::new(2, vec![1; 32], high_params))
            .unwrap();

        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1, "budget fits one prefill");
        assert_eq!(
            output.prefill_sequences[0].seq_id, 2,
            "high priority scheduled first"
        );
        assert!(scheduler.has_pending_work(), "low priority still pending");
    }

    #[test]
    fn test_priority_same_level_keeps_fcfs() {
        // PSERV-112：同级优先级保持 FCFS（先提交先调度）。
        let mut config = create_test_config();
        config.max_total_tokens = 40;
        let mut scheduler = Scheduler::new(config);

        scheduler
            .add_request(Request::new(1, vec![1; 32], GenerationParams::default()))
            .unwrap();
        scheduler
            .add_request(Request::new(2, vec![1; 32], GenerationParams::default()))
            .unwrap();

        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1);
        assert_eq!(output.prefill_sequences[0].seq_id, 1, "FCFS preserved");
    }

    #[test]
    fn test_priority_field_defaults_to_zero_and_validates() {
        let params = GenerationParams::default();
        assert_eq!(params.priority, 0);
        assert!(params.validate().is_ok());

        let mut p = params.clone();
        p.priority = 200; // 调度提示不设上限，validate 只做采样参数校验
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_decode_seq_ids_orders_by_priority_desc() {
        // PSERV-112 回归：decode 序列按 priority 降序（同级 FCFS）。
        // 已有 test_decode_priority 只断言非空；此处直接锁定排序契约。
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        // 请求 1（低优先级）先进 decode
        let mut low = create_test_request(1, 16);
        low.params.priority = 1;
        scheduler.add_request(low).unwrap();
        let out = scheduler.schedule();
        let low_seq = out.prefill_sequences[0].seq_id;
        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![100],
                seq_ids: vec![low_seq],
                logprobs: Vec::new(),
            },
            0,
        );

        // 请求 2（高优先级）后进 decode
        let mut high = create_test_request(2, 16);
        high.params.priority = 9;
        scheduler.add_request(high).unwrap();
        let out = scheduler.schedule();
        let high_seq = out.prefill_sequences[0].seq_id;
        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![101],
                seq_ids: vec![high_seq],
                logprobs: Vec::new(),
            },
            0,
        );

        assert_eq!(
            scheduler.decode_seq_ids(),
            vec![high_seq, low_seq],
            "decode 应按 priority 降序"
        );
    }

    #[test]
    fn test_oversized_prompt_fails_instead_of_wedging_pending() {
        // B12 回归：所需块数超过内存水位线的请求必须立即失败并排出，
        // 而不是放回 pending_queue 造成 has_pending_work() 恒真的空转死循环。
        // 1000 tokens → ceil(1000/16)=63 块 > floor(64*0.9)=57 水位线。
        let mut config = create_test_config();
        config.max_num_blocks = 64;
        config.max_total_tokens = 1024;
        let mut scheduler = Scheduler::new(config);

        scheduler.add_request(create_test_request(1, 1000)).unwrap();

        let output = scheduler.schedule();
        assert!(output.prefill_sequences.is_empty(), "不应启动该 prefill");

        // 关键断言：请求不留在 pending（否则引擎将永远空转），而是已失败排出
        assert!(
            !scheduler.has_pending_work(),
            "永久不可调度的请求不应滞留 pending"
        );
        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 1, "请求应被失败并排出");
        assert!(matches!(
            completed[0].state,
            RequestState::Failed(ref msg) if msg.contains("watermark")
        ));
    }

    #[test]
    fn test_fail_pending_requests_drains_and_reports() {
        // stall 兜底 API：把等待队列全部标记失败并排出（run()/engine_loop 使用）。
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        scheduler.add_request(create_test_request(1, 16)).unwrap();
        scheduler.add_request(create_test_request(2, 16)).unwrap();
        assert!(scheduler.has_pending_work());

        let failed = scheduler.fail_pending_requests("test stall");
        assert_eq!(failed, 2);
        assert!(!scheduler.has_pending_work(), "pending 应被清空");

        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 2);
        assert!(
            completed
                .iter()
                .all(|r| matches!(r.state, RequestState::Failed(_))),
            "全部应标记失败"
        );
    }

    #[test]
    fn test_add_request_counts_pending_toward_max_num_seqs() {
        let mut scheduler = Scheduler::new(EngineConfig {
            max_num_seqs: 1,
            block_size: 4,
            max_num_blocks: 32,
            max_batch_size: 8,
            max_model_len: 32,
            max_total_tokens: 32,
            memory_threshold: 0.9,
            ..Default::default()
        });

        let req = |id: u64| {
            Request::new(
                id,
                vec![10, 11, 12],
                crate::types::GenerationParams {
                    max_tokens: 2,
                    ..Default::default()
                },
            )
        };

        assert!(scheduler.add_request(req(1)).is_ok());
        let result = scheduler.add_request(req(2));

        assert!(matches!(
            result,
            Err(EngineError::MaxConcurrentSequencesReached(1))
        ));
    }

    #[test]
    fn test_cancel_by_request_id_covers_all_stages_and_frees_kv() {
        let mut scheduler = Scheduler::new(create_test_config());

        // request 1 保持 prefill；request 2 推进到 decode。
        let pending_seq = scheduler
            .add_request(create_test_request_with_params(1, 8, 10))
            .unwrap();
        let active_seq = scheduler
            .add_request(create_test_request_with_params(2, 8, 10))
            .unwrap();

        let output = scheduler.schedule();
        assert!(output
            .prefill_sequences
            .iter()
            .any(|s| s.seq_id == active_seq));
        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![100],
                seq_ids: vec![active_seq],
                logprobs: Vec::new(),
            },
            0,
        );
        assert!(scheduler.has_decode_sequence(active_seq));
        assert!(scheduler.get_memory_utilization() > 0.0);

        assert!(scheduler.prefill_sequences.contains_key(&pending_seq));
        // request 3 在 schedule 之后提交，确实保持 pending。
        let queued_seq = scheduler
            .add_request(create_test_request_with_params(3, 8, 10))
            .unwrap();
        assert!(scheduler.has_pending_request(queued_seq));
        assert!(scheduler.cancel_by_request_id(3));
        assert!(!scheduler.has_pending_request(queued_seq));

        // 取消 prefill 请求
        assert!(scheduler.cancel_by_request_id(1));
        assert!(!scheduler.has_pending_request(pending_seq));

        // 取消 decode 请求：KV 必须释放
        assert!(scheduler.cancel_by_request_id(2));
        assert!(!scheduler.has_decode_sequence(active_seq));
        assert_eq!(scheduler.get_memory_utilization(), 0.0);

        // 三个阶段的取消都应经正常完成通道以类型化取消终态排出
        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 3);
        assert!(completed.iter().all(|r| matches!(
            &r.state,
            RequestState::Cancelled(CancellationReason::ClientDisconnected)
        )));

        // 未知 request_id 返回 false
        assert!(!scheduler.cancel_by_request_id(999));
    }

    #[test]
    fn test_grow_decode_sequence_fails_for_missing_sequence() {
        let mut scheduler = Scheduler::new(create_test_config());

        let result = scheduler.grow_decode_sequence(999);

        assert!(result.is_err(), "missing decode sequence must return Err");
    }

    #[test]
    fn test_update_sequences_fails_request_when_output_missing_token() {
        let mut scheduler = Scheduler::new(EngineConfig {
            block_size: 4,
            max_num_blocks: 32,
            max_batch_size: 8,
            max_num_seqs: 4,
            max_model_len: 32,
            max_total_tokens: 32,
            memory_threshold: 0.9,
            ..Default::default()
        });

        let request = Request::new(
            1,
            vec![10, 11, 12],
            crate::types::GenerationParams {
                max_tokens: 1,
                ..Default::default()
            },
        );
        let seq_id = scheduler.add_request(request).unwrap();
        let pending = scheduler.pending_queue.pop_front().unwrap();
        scheduler
            .try_start_prefill(pending.seq_id, pending.request)
            .unwrap();
        scheduler.transition_prefill_to_decode(seq_id);

        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: Vec::new(),
                seq_ids: vec![seq_id],
                logprobs: Vec::new(),
            },
            2,
        );

        assert!(scheduler.get_sequence(seq_id).is_none());
        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 1);
        assert!(
            matches!(&completed[0].state, RequestState::Failed(message) if message.contains("missing next token")),
            "malformed output should fail request instead of fabricating token"
        );
        assert!(
            completed[0].output_tokens.is_empty(),
            "malformed output must not append synthetic tokens"
        );
    }

    #[test]
    fn test_has_prefill_budget_false_when_only_one_block_left_and_candidate_needs_decode_reserve() {
        let config = EngineConfig {
            block_size: 16,
            max_num_blocks: 64,
            max_batch_size: 8,
            max_num_seqs: 32,
            max_model_len: 2048,
            max_total_tokens: 512,
            memory_threshold: 0.9,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        // 占满 63 块，只剩 1 块。
        scheduler.kv_cache.allocate_sequence(999, 1008).unwrap();
        assert_eq!(scheduler.kv_cache.get_memory_stats().free_blocks, 1);

        // 候选需要 1 块且其下一步需要 decode 增长：必须拒绝（水位线与预留都不过）。
        assert!(!scheduler.has_prefill_budget(1, 16));
    }

    #[test]
    fn test_next_step_growth_reserve_counts_boundary_sequences() {
        let mut scheduler = Scheduler::new(create_test_config());

        // 16 tokens = 恰好 1 个 block：prefill 阶段就应在下一步需要增长。
        let seq1 = scheduler.add_request(create_test_request(1, 16)).unwrap();
        let p1 = scheduler.pending_queue.pop_front().unwrap();
        scheduler.try_start_prefill(p1.seq_id, p1.request).unwrap();
        assert_eq!(scheduler.next_step_growth_reserve(), 1);

        // 推进到 decode（context 17，仍 1 个 block）：仍需要增长。
        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![100],
                seq_ids: vec![seq1],
                logprobs: Vec::new(),
            },
            0,
        );
        assert!(scheduler.has_decode_sequence(seq1));
        assert_eq!(scheduler.next_step_growth_reserve(), 1);

        // 40 tokens = 3 个 block：+2 tokens 后仍 3 块，短期无需增长，不计入。
        let seq2 = scheduler.add_request(create_test_request(2, 40)).unwrap();
        let p2 = scheduler.pending_queue.pop_front().unwrap();
        scheduler.try_start_prefill(p2.seq_id, p2.request).unwrap();
        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![101],
                seq_ids: vec![seq2],
                logprobs: Vec::new(),
            },
            0,
        );
        // 只剩 seq1（decode, 1 block）计数；seq2（context 41, 3 blocks）不计数。
        assert_eq!(scheduler.next_step_growth_reserve(), 1);
    }

    #[test]
    fn test_pending_queue_skips_large_prefill_for_smaller_one() {
        let config = EngineConfig {
            block_size: 16,
            max_num_blocks: 200,
            max_batch_size: 8,
            max_num_seqs: 64,
            max_model_len: 2048,
            max_total_tokens: 100,
            memory_threshold: 0.9,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        // 7 个 decode 序列占住批次（每步各 1 token）。
        for i in 0..7 {
            let seq = scheduler
                .add_request(create_test_request(i + 1, 16))
                .unwrap();
            let pending = scheduler.pending_queue.pop_front().unwrap();
            scheduler
                .try_start_prefill(pending.seq_id, pending.request)
                .unwrap();
            scheduler.transition_prefill_to_decode(seq);
        }

        // 96 token 的大 pending 排在 1 token 的小 pending 之前。
        let large_seq = scheduler.add_request(create_test_request(100, 96)).unwrap();
        let small_seq = scheduler.add_request(create_test_request(200, 1)).unwrap();

        let output = scheduler.schedule();

        // 小请求本步应被启动（进入 prefill），大请求仍留在 pending 等待后续步骤。
        assert!(
            output
                .prefill_sequences
                .iter()
                .any(|s| s.seq_id == small_seq),
            "small pending request must be started this step"
        );
        assert!(
            !output
                .prefill_sequences
                .iter()
                .any(|s| s.seq_id == large_seq),
            "large pending request must not be started this step"
        );
        assert!(scheduler.has_pending_request(large_seq));
        assert!(!scheduler.has_pending_request(small_seq));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::test_utils::{create_test_config_with_limits, create_test_request_with_params};
    use proptest::prelude::*;
    use std::collections::HashSet;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_request_id_uniqueness(
            num_requests in 1usize..50,
            tokens_per_request in 1usize..64,
        ) {
            let config = create_test_config_with_limits(32, 4096, 500);
            let mut scheduler = Scheduler::new(config);
            let mut assigned_ids: HashSet<SeqId> = HashSet::new();

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 10);
                let _ = scheduler.add_request(request);
            }

            loop {
                let output = scheduler.schedule();
                if output.is_empty() {
                    break;
                }

                for seq in &output.prefill_sequences {
                    prop_assert!(
                        !assigned_ids.contains(&seq.seq_id),
                        "Duplicate sequence ID: {}",
                        seq.seq_id
                    );
                    assigned_ids.insert(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    assigned_ids.insert(seq.seq_id);
                }

                let seq_ids: Vec<SeqId> = output.prefill_sequences.iter()
                    .chain(output.decode_sequences.iter())
                    .map(|s| s.seq_id)
                    .collect();
                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens: vec![100; seq_ids.len()],
                        logprobs: Vec::new(),
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        #[test]
        fn prop_scheduler_queue_state_consistency(
            num_requests in 1usize..20,
            tokens_per_request in 1usize..32,
            num_steps in 1usize..10,
        ) {
            let config = create_test_config_with_limits(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 50);
                let _ = scheduler.add_request(request);
            }

            for _ in 0..num_steps {
                let output = scheduler.schedule();

                for seq in &output.prefill_sequences {
                    prop_assert!(
                        scheduler.is_in_exactly_one_queue(seq.seq_id),
                        "Sequence {} is not in exactly one queue",
                        seq.seq_id
                    );
                }

                for seq in &output.decode_sequences {
                    prop_assert!(
                        scheduler.is_in_exactly_one_queue(seq.seq_id),
                        "Sequence {} is not in exactly one queue",
                        seq.seq_id
                    );
                }

                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();

                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }

                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        seq_ids,
                        logprobs: Vec::new(),
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        #[test]
        fn prop_batch_size_constraints(
            max_batch_size in 1u32..16,
            max_total_tokens in 64u32..512,
            num_requests in 1usize..30,
            tokens_per_request in 1usize..64,
        ) {
            let config = create_test_config_with_limits(max_batch_size, max_total_tokens, 500);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 10);
                let _ = scheduler.add_request(request);
            }

            for _ in 0..5 {
                let output = scheduler.schedule();

                let num_sequences = output.num_sequences();
                prop_assert!(
                    num_sequences <= max_batch_size as usize,
                    "Batch has {} sequences, max is {}",
                    num_sequences,
                    max_batch_size
                );

                prop_assert!(
                    output.total_tokens <= max_total_tokens,
                    "Batch has {} tokens, max is {}",
                    output.total_tokens,
                    max_total_tokens
                );

                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();

                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }

                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        seq_ids,
                        logprobs: Vec::new(),
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        #[test]
        fn prop_decode_priority_over_prefill(
            num_decode in 1usize..10,
            num_pending in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_limits(32, 2048, 500);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_decode {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }

            for _ in 0..num_decode {
                let output = scheduler.schedule();

                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();

                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }

                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        seq_ids,
                        logprobs: Vec::new(),
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }

            for i in num_decode..(num_decode + num_pending) {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }

            let output = scheduler.schedule();

            // decode 优先且批次上限（32）远大于 decode 数量（<10）：
            // 每个在途 decode 序列都必须被调度，一个都不能落下。
            let decode_count = scheduler.num_decode_sequences();
            prop_assert_eq!(
                output.decode_sequences.len(),
                decode_count,
                "every in-flight decode sequence must be scheduled each step"
            );
        }

        #[test]
        fn prop_prefill_to_decode_transition(
            num_requests in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_limits(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 50);
                scheduler.add_request(request).unwrap();
            }

            let output = scheduler.schedule();
            let prefill_seq_ids: Vec<SeqId> = output.prefill_sequences.iter().map(|s| s.seq_id).collect();

            let next_tokens: Vec<u32> = prefill_seq_ids.iter().map(|_| 100).collect();
            let exec_output = ExecutionOutput {
                next_tokens,
                seq_ids: prefill_seq_ids.clone(),
                logprobs: Vec::new(),
            };

            scheduler.update_sequences(&exec_output, 0);

            for seq_id in &prefill_seq_ids {
                prop_assert!(
                    scheduler.has_decode_sequence(*seq_id),
                    "Sequence {} should be in decode after prefill",
                    seq_id
                );
                prop_assert!(
                    !scheduler.has_prefill_sequence(*seq_id),
                    "Sequence {} should not be in prefill after transition",
                    seq_id
                );
            }
        }

        #[test]
        fn prop_completion_conditions(
            max_tokens in 1u32..20,
            tokens_per_request in 4usize..16,
            eos_position in 0usize..25,
        ) {
            let config = create_test_config_with_limits(8, 512, 100);
            let mut scheduler = Scheduler::new(config);

            let request = create_test_request_with_params(1, tokens_per_request, max_tokens);
            scheduler.add_request(request).unwrap();

            let output = scheduler.schedule();
            let seq_id = output.prefill_sequences[0].seq_id;

            let exec_output = ExecutionOutput {
                next_tokens: vec![100],
                seq_ids: vec![seq_id],
                logprobs: Vec::new(),
            };
            scheduler.update_sequences(&exec_output, 0);

            let eos_token: TokenId = 0;
            let mut generated = 1u32;

            while scheduler.has_decode_sequence(seq_id) && generated < max_tokens + 5 {
                scheduler.schedule();

                let token = if generated as usize == eos_position {
                    eos_token
                } else {
                    100 + generated
                };

                let exec_output = ExecutionOutput {
                    next_tokens: vec![token],
                    seq_ids: vec![seq_id],
                    logprobs: Vec::new(),
                };
                scheduler.update_sequences(&exec_output, eos_token);
                generated += 1;
            }

            let completed = scheduler.get_completed();

            if !completed.is_empty() {
                let req = &completed[0];
                let hit_max = req.output_tokens.len() >= max_tokens as usize;
                let hit_eos = req.output_tokens.last() == Some(&eos_token);

                prop_assert!(
                    hit_max || hit_eos,
                    "Completion should be due to max_tokens or EOS"
                );
            }
        }

        #[test]
        fn prop_memory_pressure_response(
            num_initial_requests in 5usize..15,
            tokens_per_request in 16usize..64,
        ) {
            let config = EngineConfig {
                block_size: 16,
                max_num_blocks: 20,
                max_batch_size: 16,
                max_num_seqs: 32,
                max_model_len: 2048,
                max_total_tokens: 1024,
                memory_threshold: 0.5,
                ..Default::default()
            };
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_initial_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                let _ = scheduler.add_request(request);
                let _ = scheduler.schedule();
            }

            let utilization = scheduler.get_memory_utilization();

            if utilization >= 0.5 {
                let new_request = create_test_request_with_params(999, tokens_per_request, 100);
                let result = scheduler.add_request(new_request);

                // 利用率达到阈值后，add_request 必须确定性地拒绝（MemoryPressure），
                // 而不是"可能拒绝也可能接受"。
                prop_assert!(
                    matches!(result, Err(EngineError::MemoryPressure)),
                    "utilization {} >= threshold 0.5 must reject with MemoryPressure, got {:?}",
                    utilization,
                    result
                );
            }
        }

        /// **穷举场景：请求在任意阶段被取消/失败后资源必须全部归还**
        ///
        /// 交替执行「随机取消/失败某个请求」与「推进一步」，覆盖 pending /
        /// prefill / decode 各阶段的终止路径；最终所有请求必须到达终态
        /// （无悬挂、无遗留 pending），KV 块全部归还，内存利用率回到基线。
        #[test]
        fn prop_resources_reclaimed_after_cancel_and_failure(
            num_requests in 1usize..8,
            tokens_per_request in 1usize..16,
            max_tokens in 1u32..6,
            ops in prop::collection::vec(0u8..3, 0..40),
            targets in prop::collection::vec(0usize..8, 0..40),
        ) {
            let config = create_test_config_with_limits(16, 4096, 500);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_requests {
                scheduler
                    .add_request(create_test_request_with_params(
                        i as u64,
                        tokens_per_request,
                        max_tokens,
                    ))
                    .unwrap();
            }

            // 交替执行「取消/失败某个请求」与「推进一步」，直到没有待处理工作。
            let mut step = 0usize;
            let mut completed_count = 0usize;
            while scheduler.has_pending_work() {
                if let Some(&op) = ops.get(step) {
                    let target = if targets.is_empty() {
                        0
                    } else {
                        (targets[step % targets.len()] % num_requests) as u64
                    };
                    match op {
                        0 => {} // 仅推进
                        1 => {
                            let _ = scheduler.cancel_by_request_id(target);
                        }
                        _ => {
                            scheduler.fail_by_request_id(target, "property-test failure");
                        }
                    }
                }

                let output = scheduler.schedule();
                if !output.is_empty() {
                    let seq_ids = output.seq_ids();
                    let exec_output = ExecutionOutput {
                        next_tokens: vec![100; seq_ids.len()],
                        seq_ids,
                        logprobs: Vec::new(),
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
                // 排出本步到达终态的请求（取消/失败/正常完成）并累计
                completed_count += scheduler.get_completed().len();
                step += 1;
                prop_assert!(step < 200, "generation must terminate");
            }

            // 不变式：所有请求到达终态，且资源全部归还
            prop_assert_eq!(
                completed_count,
                num_requests,
                "every request (completed, cancelled or failed) must reach a terminal state"
            );
            prop_assert_eq!(
                scheduler.num_active_sequences(),
                0,
                "no sequence may remain active"
            );
            let stats = scheduler.kv_cache.get_memory_stats();
            prop_assert_eq!(stats.used_blocks, 0, "all KV blocks must be reclaimed");
            prop_assert_eq!(
                stats.used_blocks + stats.free_blocks,
                stats.total_blocks,
                "block count invariant must hold"
            );
            prop_assert_eq!(
                scheduler.get_memory_utilization(),
                0.0,
                "memory utilization must return to baseline"
            );
        }
    }
}
