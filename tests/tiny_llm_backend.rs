//! tiny-llm 真实后端接入的端到端测试（里程碑 4）。
//!
//! 门控：
//! - 编译期：`cargo test --features tiny-llm`（启用 C ABI 符号与适配器）
//! - 链接期：`TINY_LLM_DIR` 指向 tiny-llm 构建目录（build.rs）
//! - 运行期：`TINY_LLM_MODEL` 指向真实 GGUF 模型
//! - 启用 feature 后缺少模型直接失败，不将未执行记为通过
//!
//! 使用 paged-serving 自带的 SimpleTokenizer（词表语义与 Qwen 不同），
//! 本测试验证的是**接入流程正确性**（引擎驱动、KV 生命周期、资源守恒、
//! 能力声明），而非文本质量；文本质量验证需接入与模型词表一致的 tokenizer。
//!
//! 使用 `--test-threads=1` 串行执行本目标，避免两个测试同时加载模型。
//! 每个测试复用自己的后端实例；能力检查与两波请求合并验证实例复用。

#![cfg(feature = "tiny-llm")]

use paged_serving::test_utils::create_test_config;
use paged_serving::{
    EngineError, GenerationParams, InferenceEngine, Scheduler, SimpleTokenizer, TinyLlmExecutor,
};

fn model_path() -> String {
    std::env::var("TINY_LLM_MODEL").expect("真实后端测试必须设置 TINY_LLM_MODEL")
}

#[test]
fn tiny_llm_backend_end_to_end() {
    let path = model_path();

    let mut config = create_test_config();
    config.max_num_blocks = 256;
    config.max_model_len = 256; // 与 tiny-llm KV 预留（context+512）匹配
    config.max_total_tokens = 1024;

    let executor = TinyLlmExecutor::new(&path, config.clone()).expect("tinyllm_load failed");
    let mut engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config),
        Box::new(executor),
    )
    .unwrap();

    // 能力声明：非 greedy 参数应在 submit 阶段被诚实拒绝
    let params = GenerationParams {
        max_tokens: 4,
        temperature: 0.8,
        ..GenerationParams::default()
    };
    let err = engine.submit_request("sampling", params).unwrap_err();
    assert!(matches!(err, EngineError::UnsupportedGenerationMode(_)));

    // 并发提交 3 个请求（greedy），驱动到全部完成
    for i in 0..3 {
        let params = GenerationParams {
            max_tokens: 4,
            ..GenerationParams::default()
        };
        engine
            .submit_request(&format!("prompt {}", i), params)
            .unwrap();
    }

    let completed = engine.run();
    assert_eq!(completed.len(), 3, "全部请求应完成");
    for c in &completed {
        assert!(c.success, "request {} failed: {:?}", c.request_id, c.error);
    }

    // 资源守恒：全部完成后 KV 归还
    let util = engine.memory_utilization();
    assert!(util < 0.05, "tiny-llm KV 未归还: {util}");

    let m = engine.get_metrics();
    assert_eq!(m.completed_requests, 3);
    assert_eq!(m.failed_requests, 0);

    // 第二波请求：复用同一后端实例，验证 `sequences_finished` 释放物理 KV 后
    // 没有 slot 泄漏（再跑 3 个也应全部成功）。
    for i in 3..6 {
        let params = GenerationParams {
            max_tokens: 4,
            ..GenerationParams::default()
        };
        engine
            .submit_request(&format!("prompt {}", i), params)
            .unwrap();
    }

    let completed2 = engine.run();
    assert_eq!(completed2.len(), 3, "第二波全部请求应完成");
    for c in &completed2 {
        assert!(
            c.success,
            "第二波 request {} failed: {:?}",
            c.request_id, c.error
        );
    }

    let util2 = engine.memory_utilization();
    assert!(util2 < 0.05, "第二波后 tiny-llm KV 未归还: {util2}");

    let m2 = engine.get_metrics();
    assert_eq!(m2.completed_requests, 6);
    assert_eq!(m2.failed_requests, 0);
}

/// B15 回归：decode 越过 tiny-llm 预留 KV 容量（context + decode 预留，默认 512）时，
/// 请求必须携带清晰的越界错误失败（而非后端 rc 神秘报错或挂起）。
///
/// 触发方式：短 prompt + 大 max_tokens，使总长度越过 submit 的
/// TotalLengthTooLong 校验上限，但解码步数超过首次分配的 KV 容量。
#[test]
fn tiny_llm_backend_decode_overrun_reports_clear_error() {
    let path = model_path();

    let mut config = create_test_config();
    // max_model_len 需容纳 prompt + 超过 decode 预留默认值(512) 的 max_tokens，
    // 才能让 decode 实际越过预留容量；KV 池与总额同步放大。
    config.max_num_blocks = 256;
    config.max_model_len = 1024;
    config.max_total_tokens = 1024;

    let executor = TinyLlmExecutor::new(&path, config.clone()).expect("tinyllm_load failed");
    let mut engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config),
        Box::new(executor),
    )
    .unwrap();

    // 短 prompt + 远超预留容量的 max_tokens（1 + 900 <= max_model_len）
    let params = GenerationParams {
        max_tokens: 900,
        ..GenerationParams::default()
    };
    engine.submit_request("hi", params).unwrap();

    let completed = engine.run();
    assert_eq!(completed.len(), 1, "请求应到达终态而非挂起");
    let c = &completed[0];
    assert!(!c.success, "越界请求应失败");
    let err = c.error.as_ref().expect("失败请求应有错误信息");
    assert!(
        err.contains("超出 tiny-llm 预留 KV 容量"),
        "应返回清晰的越界错误，实际: {err}"
    );

    // 资源守恒：失败后 KV 也应归还
    let util = engine.memory_utilization();
    assert!(util < 0.05, "失败后 tiny-llm KV 未归还: {util}");

    let m = engine.get_metrics();
    assert_eq!(m.completed_requests, 0);
    assert_eq!(m.failed_requests, 1);
}
