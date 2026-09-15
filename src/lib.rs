//! # Paged-Serving
//!
//! 基于 `PagedAttention` 分页内存与 Continuous Batching 的推理引擎脚手架；
//! 默认计算后端为 CPU 参考执行器（真实前向、greedy 采样）。
//!
//! ## 概述
//!
//! 本库提供了一个推理引擎脚手架，实现了以下核心技术（默认计算后端为 CPU 参考执行器）：
//!
//! - **`PagedAttention`**: 分页式 KV Cache 管理，按需分配/释放显存块
//! - **Continuous Batching**: 连续批处理调度，prefill/decode 分阶段管理
//! - **内存压力感知**: 可配置阈值，自动拒绝新请求防止 OOM
//! - **模块化设计**: 所有核心组件通过 trait 抽象，便于替换实现
//!
//! ## 架构
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
//! ## 快速开始
//!
//! ```rust
//! use paged_serving::{EngineConfig, GenerationParams, InferenceEngine};
//!
//! // 创建引擎
//! let config = EngineConfig::default();
//! let mut engine = InferenceEngine::new(config)?;
//!
//! // 提交请求（CPU 参考后端当前仅支持 greedy：temperature = 0.0, top_p = 1.0；
//! // 默认 SimpleTokenizer 仅支持 ASCII，中文请改用 HuggingFace tokenizer）
//! let params = GenerationParams {
//!     max_tokens: 50,
//!     ..GenerationParams::default()
//! };
//! let (request_id, _prompt_tokens) = engine.submit_request("Hello, world!", params)?;
//!
//! // 运行推理
//! let completed = engine.run();
//!
//! for result in completed {
//!     println!("输出: {}", result.output_text);
//! }
//! # Ok::<(), paged_serving::EngineError>(())
//! ```
//!
//! ## 核心组件
//!
//! ### 配置
//!
//! - [`EngineConfig`] - 引擎配置参数，包括块大小、批次大小、内存阈值等
//!
//! ### 推理引擎
//!
//! - [`InferenceEngine`] - 主编排器，协调所有组件
//! - [`EngineMetrics`] - 运行时指标收集
//!
//! ### 调度器
//!
//! - [`Scheduler`] - Continuous Batching 调度器
//!
//! ### KV Cache 管理
//!
//! - [`KVCacheManager`] - `PagedAttention` KV Cache 管理器
//!
//! ### GPU 执行器
//!
//! - [`MockGPUExecutor`] - Mock GPU 执行器（测试用）
//! - [`GPUExecutorTrait`] - GPU 执行器 trait 接口
//! - [`build_execution_batch`] - 构建执行批次
//!
//! ### 分词器
//!
//! - [`SimpleTokenizer`] - 简单字符级分词器（测试用，`without_special_tokens()` 可精确往返）
//! - [`TokenizerTrait`] - 分词器 trait 接口
//!
//! ### 类型
//!
//! - [`Request`] - 推理请求
//! - [`Sequence`] - 活跃序列（含 KV Cache）
//! - [`GenerationParams`] - 生成参数
//! - [`CompletedRequest`] - 完成的请求
//! - [`ExecutionBatch`] - GPU 执行批次
//! - [`ExecutionOutput`] - GPU 执行输出
//!
//! ### 错误处理
//!
//! - [`EngineError`] - 引擎运行时错误（扁平化，覆盖内存 / 验证 / 执行 / 调度 / 分词）
//! - [`ConfigError`] - 配置错误

pub mod config;
pub mod cpu_executor;
pub mod engine;
pub mod error;
pub mod execution_pipeline;
pub mod gpu_executor;
pub mod kv_cache;
pub mod scheduler;
pub mod server;
#[cfg(feature = "tiny-llm")]
pub mod tiny_llm_executor;
pub mod tiny_llm_ffi;
pub mod tokenizer;
pub mod types;

// 测试辅助模块：仅在单元测试（cfg(test)）或 `test-utils` feature 下编译，
// 不会进入发布构建。集成测试与 bench 经由 dev-dependencies 启用该 feature。
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_utils;

// 选择性导出，避免命名空间污染（如 error::Result 遮蔽 std::Result）
pub use config::{EngineConfig, ServingConfig, SpecialTokenIds, TokenizerConfig, TokenizerKind};
pub use cpu_executor::CpuReferenceExecutor;
pub use engine::{EngineMetrics, InferenceEngine, StepEvents};
pub use error::{ConfigError, EngineError};
pub use execution_pipeline::{build_execution_batch, BatchExecutionPipeline};
pub use gpu_executor::{create_default_gpu_executor, GPUExecutorTrait, MockGPUExecutor};
pub use kv_cache::KVCacheManager;
pub use scheduler::Scheduler;
pub use server::{
    create_router, create_router_with_engine, create_router_with_engine_and_shutdown,
};
#[cfg(feature = "tiny-llm")]
pub use tiny_llm_executor::TinyLlmExecutor;
pub use tokenizer::{
    build_tokenizer, HuggingFaceTokenizer, IncrementalDecoder, SimpleTokenizer, TokenizerTrait,
};
pub use types::{
    BlockIdx, CompletedRequest, ExecutionBatch, ExecutionOutput, GenerationParams, MemoryStats,
    PhysicalBlockRef, Request, RequestId, RequestState, SchedulerOutput, SeqId, Sequence, TokenId,
};
