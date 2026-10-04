//! 核心类型和数据结构
//!
//! 按子系统组织：
//! - [`request`] — 请求与生成参数
//! - [`scheduler`] — 序列与调度输出
//! - [`execution`] — GPU 执行批次与输出
//! - [`memory`] — 内存统计与块引用

/// 请求唯一标识符
pub type RequestId = u64;

/// 序列唯一标识符
pub type SeqId = u64;

/// Token ID 类型
pub type TokenId = u32;

/// 物理块索引
pub type BlockIdx = u32;

/// 请求状态
///
/// 表示请求在推理流水线中的当前状态。
///
/// # 状态转换
///
/// ```text
/// Pending → Prefill → Decode → Completed
/// 每个非终态也可转为 Failed 或 Cancelled
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    /// 等待调度
    Pending,

    /// Prefill 阶段（处理输入 tokens）
    Prefill,

    /// Decode 阶段（生成 tokens）
    Decode,

    /// 成功完成
    Completed,

    /// 失败，包含错误信息
    Failed(String),

    /// 主动取消，不计入后端失败
    Cancelled(CancellationReason),
}

/// 请求主动取消的原因；错误文案不参与指标分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationReason {
    ClientDisconnected,
    ServerShutdown,
}

impl std::fmt::Display for CancellationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ClientDisconnected => "request cancelled: client disconnected",
            Self::ServerShutdown => "request cancelled: server shutting down",
        })
    }
}

pub mod execution;
pub mod memory;
pub mod request;
pub mod scheduler;

pub use execution::*;
pub use memory::*;
pub use request::*;
pub use scheduler::*;
