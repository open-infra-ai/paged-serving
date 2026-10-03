//! 并发压测：资源守恒、尾延迟、失败传播
//!
//! ROADMAP 选项 A 第三项。使用 Mock/CPU 后端先行验证调度器在高并发下的
//! 资源不变量、延迟分布与失败隔离；真实 tiny-llm 后端接入后直接切换
//! executor 复用本套场景（executor 可替换是 EngineBackend 契约的设计目标）。

use paged_serving::test_utils::create_test_config;
use paged_serving::types::{CompletedRequest, ExecutionBatch, ExecutionOutput, RequestId, TokenId};
use paged_serving::{
    EngineConfig, EngineError, GPUExecutorTrait, GenerationParams, InferenceEngine,
    MockGPUExecutor, Scheduler, SimpleTokenizer,
};
use std::collections::HashMap;
use std::time::Instant;

// ── 统计辅助 ──────────────────────────────────────────────

/// 有序样本的百分位（p ∈ [0,1]，线性插值取最近秩）。
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── 自定义 executor：周期性注入失败 ────────────────────────

/// 每 `fail_mod` 次 execute 返回一次错误，模拟后端偶发故障。
/// 用于验证失败隔离（单个/部分请求失败不污染其余请求）与失败后资源归还。
struct PeriodicFailExecutor {
    counter: usize,
    fail_mod: usize,
    token: TokenId,
}

impl PeriodicFailExecutor {
    fn new(fail_mod: usize, token: TokenId) -> Self {
        Self {
            counter: 0,
            fail_mod: fail_mod.max(1),
            token,
        }
    }
}

impl GPUExecutorTrait for PeriodicFailExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        if batch.is_empty() {
            return Ok(ExecutionOutput::default());
        }
        self.counter += 1;
        if self.counter.is_multiple_of(self.fail_mod) {
            return Err(EngineError::KernelLaunchFailed(
                "injected periodic failure".to_string(),
            ));
        }
        Ok(ExecutionOutput {
            next_tokens: vec![self.token; batch.seq_ids.len()],
            seq_ids: batch.seq_ids.clone(),
            logprobs: Vec::new(),
        })
    }
}

// ── 引擎构造与驱动辅助 ────────────────────────────────────

fn make_mock_engine(config: EngineConfig) -> InferenceEngine {
    let executor = MockGPUExecutor::new(config.clone(), 1000);
    InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config),
        Box::new(executor),
    )
    .unwrap()
}

/// 提交 n 个并发请求（max_tokens 1..=10 循环），记录提交时刻。
fn submit_burst(engine: &mut InferenceEngine, n: usize) -> HashMap<RequestId, Instant> {
    let mut starts = HashMap::new();
    for i in 0..n {
        let max_tokens = (i as u32 % 10) + 1;
        let params = GenerationParams {
            max_tokens,
            ..GenerationParams::default()
        };
        let (id, _) = engine
            .submit_request(&format!("prompt {}", i), params)
            .unwrap_or_else(|e| panic!("submit #{} failed: {e}", i));
        starts.insert(id, Instant::now());
    }
    starts
}

/// 驱动引擎直到无 pending 工作，返回全部完成请求与逐请求延迟（ms）。
fn drain(
    engine: &mut InferenceEngine,
    starts: &mut HashMap<RequestId, Instant>,
) -> (Vec<CompletedRequest>, Vec<f64>) {
    let mut completed = Vec::new();
    let mut latencies = Vec::new();
    let mut guard = 0;
    while engine.has_pending_work() {
        let done = engine.step().unwrap_or_else(|e| panic!("step failed: {e}"));
        for c in &done {
            if let Some(t0) = starts.remove(&c.request_id) {
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        completed.extend(done);
        guard += 1;
        assert!(guard < 100_000, "引擎未排空（疑似死循环）");
    }
    (completed, latencies)
}

// ── 场景 1：并发突发全完成 + 尾延迟分布 ───────────────────

#[test]
fn test_concurrent_burst_tail_latency_and_success() {
    let mut config = create_test_config();
    config.max_num_seqs = 64; // 40 并发请求需要（默认 32 会在 submit 时拒绝）
    let mut engine = make_mock_engine(config);
    let mut starts = submit_burst(&mut engine, 40);

    let (completed, mut latencies) = drain(&mut engine, &mut starts);

    assert_eq!(completed.len(), 40, "全部请求都应完成");
    for c in &completed {
        assert!(c.success, "request {} failed: {:?}", c.request_id, c.error);
    }
    assert!(starts.is_empty(), "仍有未完成请求: {:?}", starts.keys());

    // 尾延迟分布：p50 <= p95 <= p99，且量级合理（Mock 后端应亚毫秒级）
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);
    assert!(
        p50 <= p95 + 1e-9 && p95 <= p99 + 1e-9,
        "延迟单调性破坏: p50={p50:.2}ms p95={p95:.2}ms p99={p99:.2}ms"
    );
    assert!(p99 < 5_000.0, "p99 延迟异常: {p99:.2}ms");

    let m = engine.get_metrics();
    assert_eq!(m.completed_requests, 40);
    assert_eq!(m.failed_requests, 0);
}

// ── 场景 2：高并发下资源守恒 ─────────────────────────────

#[test]
fn test_resource_conservation_under_high_concurrency() {
    // 小 KV 池 + 大并发，强制分配/释放路径在压力下运行
    let mut config = create_test_config();
    config.max_num_blocks = 64;
    config.max_num_seqs = 64;
    config.max_model_len = 512;
    let mut engine = make_mock_engine(config);
    let mut starts = submit_burst(&mut engine, 50);

    // 运行中内存利用率始终在 [0, 1]（不越界、不 panic）
    let mut guard = 0;
    while engine.has_pending_work() {
        let util = engine.memory_utilization();
        assert!((0.0..=1.0).contains(&util), "运行中 util 越界: {util}");
        let done = engine.step().unwrap();
        for c in &done {
            starts.remove(&c.request_id);
        }
        guard += 1;
        assert!(guard < 100_000, "引擎未排空");
    }

    // 所有请求结束后 KV 块必须全部归还（无泄漏）
    let final_util = engine.memory_utilization();
    assert!(final_util < 0.05, "KV 块未归还（疑似泄漏）: {final_util}");
    assert!(starts.is_empty(), "仍有未完成请求: {:?}", starts.keys());
}

// ── 场景 3：失败隔离 + 失败后资源归还 ────────────────────

#[test]
fn test_failure_isolation_and_reclamation() {
    let config = create_test_config();
    let executor = PeriodicFailExecutor::new(5, 42);
    let mut engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config),
        Box::new(executor),
    )
    .unwrap();
    let mut starts = submit_burst(&mut engine, 30);

    let (completed, _) = drain(&mut engine, &mut starts);

    assert_eq!(completed.len(), 30, "请求不应丢失");
    let ok = completed.iter().filter(|c| c.success).count();
    let failed = completed.iter().filter(|c| !c.success).count();
    assert_eq!(ok + failed, 30);
    assert!(failed > 0, "注入失败应产生失败请求");
    assert!(ok > 0, "失败不应污染其余请求（失败隔离）");
    for c in completed.iter().filter(|c| !c.success) {
        assert!(c.error.is_some(), "失败请求应有错误信息");
    }

    // 失败请求的资源也必须归还
    let util = engine.memory_utilization();
    assert!(util < 0.05, "失败请求的 KV 块未归还: {util}");

    let m = engine.get_metrics();
    assert_eq!(m.failed_requests as usize, failed);
    assert_eq!(m.completed_requests as usize, ok);
}

// ── 场景 4：内存压力优雅处理 ─────────────────────────────

#[test]
fn test_memory_pressure_rejects_or_queues_gracefully() {
    // 极小 KV 池 + 大量提交：引擎必须优雅拒绝（而非 panic/OOM）
    let mut config = create_test_config();
    config.max_num_blocks = 16;
    config.max_num_seqs = 8;
    config.max_model_len = 256;
    let mut engine = make_mock_engine(config);

    let mut submitted = 0;
    for i in 0..100 {
        let params = GenerationParams {
            max_tokens: 5,
            ..GenerationParams::default()
        };
        match engine.submit_request(&format!("p{}", i), params) {
            Ok(_) => submitted += 1,
            Err(e) => {
                assert!(
                    matches!(
                        e,
                        EngineError::MemoryPressure | EngineError::MaxConcurrentSequencesReached(_)
                    ),
                    "压力下应返回内存/并发类错误，实际: {e}"
                );
                break;
            }
        }
    }
    assert!(submitted > 0, "至少应接受部分请求");

    // 已接受请求全部完成，无卡死
    let mut completed = 0;
    let mut guard = 0;
    while engine.has_pending_work() {
        completed += engine.step().unwrap().len();
        guard += 1;
        assert!(guard < 100_000, "引擎未排空");
    }
    assert_eq!(completed, submitted);

    // 压力解除后新请求可正常提交并完成
    let (id, _) = engine
        .submit_request(
            "after pressure",
            GenerationParams {
                max_tokens: 3,
                ..GenerationParams::default()
            },
        )
        .unwrap();
    let done = engine.run();
    assert!(id > 0);
    assert_eq!(done.len(), 1);
    assert!(done[0].success);
}
