//! 错误类型定义
//!
//! 本模块定义两类错误：
//!
//! - [`ConfigError`]：配置加载与校验错误（在引擎构造前发生，main 单独处理）
//! - [`EngineError`]：引擎运行时错误（扁平化，覆盖内存 / 验证 / 执行 / 调度 / 分词）
//!
//! # 错误处理策略
//!
//! | 错误变体 | 恢复策略 |
//! |----------|----------|
//! | `OutOfBlocks` | 等待序列完成释放内存 |
//! | `GpuTimeout` | 重试最多 2 次 |
//! | `BackendError` | 失败该批次序列 |
//! | `MemoryPressure` | 拒绝新请求，等待内存释放 |
//! | `Validation*` / `Input*` | 直接返回错误 |
//!
//! # 示例
//!
//! ```rust
//! use paged_serving::{EngineError, GenerationParams};
//!
//! let params = GenerationParams {
//!     max_tokens: 0,
//!     ..GenerationParams::default()
//! };
//!
//! match params.validate() {
//!     Ok(()) => println!("参数有效"),
//!     Err(EngineError::InvalidMaxTokens(0)) => println!("max_tokens 无效"),
//!     Err(e) => println!("其他错误: {}", e),
//! }
//! ```

use thiserror::Error;

/// 配置相关错误
///
/// 表示配置参数验证或加载过程中发生的错误。
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// 无效的 block_size：必须 > 0，实际值为 {0}
    #[error("无效的 block_size: 必须大于 0，实际值为 {0}")]
    InvalidBlockSize(u32),

    /// 无效的 max_num_blocks：必须 > 0，实际值为 {0}
    #[error("无效的 max_num_blocks: 必须大于 0，实际值为 {0}")]
    InvalidMaxNumBlocks(u32),

    /// 无效的 max_batch_size：必须 > 0，实际值为 {0}
    #[error("无效的 max_batch_size: 必须大于 0，实际值为 {0}")]
    InvalidMaxBatchSize(u32),

    /// 无效的 max_num_seqs：必须 > 0，实际值为 {0}
    #[error("无效的 max_num_seqs: 必须大于 0，实际值为 {0}")]
    InvalidMaxNumSeqs(u32),

    /// 无效的 max_model_len：必须 > 0，实际值为 {0}
    #[error("无效的 max_model_len: 必须大于 0，实际值为 {0}")]
    InvalidMaxModelLen(u32),

    /// 无效的 max_total_tokens：必须 > 0，实际值为 {0}
    #[error("无效的 max_total_tokens: 必须大于 0，实际值为 {0}")]
    InvalidMaxTotalTokens(u32),

    #[error("无效的 event_channel_capacity: 必须大于 0，实际值为 {0}")]
    InvalidEventChannelCapacity(u32),

    /// 无效的 memory_threshold：必须在 (0.0, 1.0] 范围内，实际值为 {0}
    #[error("无效的 memory_threshold: 必须在 (0.0, 1.0] 范围内，实际值为 {0}")]
    InvalidMemoryThreshold(f32),

    /// 加载配置文件失败：{0}
    #[error("加载配置文件失败: {0}")]
    FileLoadError(String),

    /// 保存配置文件失败：{0}
    #[error("保存配置文件失败: {0}")]
    FileSaveError(String),

    /// 解析配置失败：{0}
    #[error("解析配置失败: {0}")]
    ParseError(String),

    /// 缺少 HuggingFace tokenizer 路径
    #[error("缺少 HuggingFace tokenizer 路径")]
    MissingTokenizerPath,

    /// 无效的服务端口：{0}
    #[error("无效的服务端口: 必须大于 0，实际值为 {0}")]
    InvalidServerPort(u16),

    /// 无效的模型名称
    #[error("无效的模型名称: 不能为空")]
    InvalidModelName,

    /// max_batch_size 大于 max_num_seqs
    #[error("max_batch_size ({0}) 大于 max_num_seqs ({1})，配置不一致")]
    BatchSizeExceedNumSeqs(u32, u32),
}

/// 引擎运行时错误（扁平化，覆盖内存 / 验证 / 执行 / 调度 / 分词）
///
/// # 示例
///
/// ```rust
/// use paged_serving::EngineError;
///
/// fn handle_error(error: EngineError) {
///     match error {
///         EngineError::Config(_) => eprintln!("配置错误: {}", error),
///         EngineError::OutOfBlocks => eprintln!("内存错误: 物理块耗尽"),
///         EngineError::GpuTimeout => eprintln!("执行错误: GPU 超时"),
///         EngineError::MemoryPressure => eprintln!("调度错误: 内存压力"),
///         EngineError::Tokenization(msg) => eprintln!("分词错误: {}", msg),
///         _ => eprintln!("其他错误: {}", error),
///     }
/// }
/// ```
#[derive(Error, Debug)]
pub enum EngineError {
    /// 配置错误
    #[error("配置错误: {0}")]
    Config(#[from] ConfigError),

    // --- 内存 ---
    /// 物理块耗尽：没有可用的空闲块
    #[error("物理块耗尽：没有可用的空闲块")]
    OutOfBlocks,

    /// 序列不存在：{0}
    #[error("序列不存在: {0}")]
    SequenceNotFound(u64),

    /// 无效的块索引：{0}
    #[error("无效的块索引: {0}")]
    InvalidBlockIndex(u32),

    // --- 验证 ---
    /// 无效的 max_tokens：必须 > 0，实际值为 {0}
    #[error("无效的 max_tokens: 必须大于 0，实际值为 {0}")]
    InvalidMaxTokens(u32),

    /// 无效的 temperature：必须在 [0.0, 2.0] 范围内，实际值为 {0}
    #[error("无效的 temperature: 必须在 [0.0, 2.0] 范围内，实际值为 {0}")]
    InvalidTemperature(f32),

    /// 无效的 top_p：必须在 (0.0, 1.0] 范围内，实际值为 {0}
    #[error("无效的 top_p: 必须在 (0.0, 1.0] 范围内，实际值为 {0}")]
    InvalidTopP(f32),

    /// 后端不支持的生成模式：{0}
    ///
    /// 参数本身在合法范围内，但当前执行后端没有实现对应语义
    /// （例如 CPU 参考执行器只支持 greedy：`temperature == 0.0` 且 `top_p == 1.0`）。
    /// 在 submit 阶段拒绝，而不是在执行时静默忽略采样参数。
    #[error("后端不支持的生成模式: {0}")]
    UnsupportedGenerationMode(String),

    /// 停止序列数量超出限制：{0}（OpenAI 允许最多 4 个）
    #[error("停止序列数量超出限制: {0}（最多 4 个）")]
    TooManyStopSequences(usize),

    /// top logprobs 数量超出限制：{0}（OpenAI 允许最多 5 个）
    #[error("top logprobs 数量超出限制: {0}（最多 5 个）")]
    InvalidLogprobs(usize),

    /// 输入文本为空
    #[error("输入文本为空")]
    EmptyInput,

    /// 输入超出最大模型长度：{0} > {1}
    #[error("输入超出最大模型长度: {0} > {1}")]
    InputTooLong(usize, u32),

    /// 请求总长度超出最大模型长度：{0} > {1}
    #[error("请求总长度超出最大模型长度: {0} > {1}")]
    TotalLengthTooLong(usize, u32),

    // --- 执行 ---
    /// 计算后端错误：{0}
    #[error("计算后端错误: {0}")]
    BackendError(String),

    #[error("server shutting down")]
    ShuttingDown,

    /// GPU 超时
    #[error("GPU 超时")]
    GpuTimeout,

    /// 无效输出：检测到 NaN 或 Inf
    #[error("无效输出: 检测到 NaN 或 Inf")]
    InvalidOutput,

    /// Kernel 启动失败：{0}
    #[error("Kernel 启动失败: {0}")]
    KernelLaunchFailed(String),

    // --- 调度 ---
    /// 内存压力：无法接受新的 prefill 请求
    #[error("内存压力: 无法接受新的 prefill 请求")]
    MemoryPressure,

    /// 超出最大并发序列数：{0}
    #[error("超出最大并发序列数: {0}")]
    MaxConcurrentSequencesReached(u32),

    /// 请求不存在：{0}
    #[error("请求不存在: {0}")]
    RequestNotFound(u64),

    /// 无效的状态转换：{0}
    #[error("无效的状态转换: {0}")]
    InvalidStateTransition(String),

    // --- 分词 ---
    /// 分词错误：{0}
    #[error("分词错误: {0}")]
    Tokenization(String),
}
