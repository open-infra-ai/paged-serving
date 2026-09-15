//! HTTP 服务层
//!
//! 提供 OpenAI 兼容的最小服务接口、健康检查与指标暴露。
//!
//! # 并发模型
//!
//! [`InferenceEngine`] 由唯一的后台任务（"引擎循环"）独占持有，HTTP handler
//! 不持锁、也不在 async 运行时上执行同步推理：
//!
//! - handler 通过 mpsc 通道提交请求，并等待该请求的专属事件流；
//! - 引擎循环在每一步之间清空提交队列，使调度器能看到并发请求、
//!   组成真正的 continuous batching 批次；
//! - 流式响应随 token 生成逐片段推送；文本片段由 tokenizer 在可安全追加时产生。

use crate::config::{EngineConfig, TokenizerKind};
use crate::engine::EngineMetrics;
use crate::error::EngineError;
use crate::tokenizer::{build_tokenizer, TokenizerTrait};
use crate::types::{CompletedRequest, FinishReason, GenerationParams, RequestId, TokenLogprobs};
use crate::InferenceEngine;
use async_stream::stream;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;

/// 提交队列容量；队列满时 handler 异步等待（背压），不会丢失请求。
const SUBMISSION_QUEUE_CAPACITY: usize = 1024;

#[derive(Default)]
struct ServerMetrics {
    requests_total: AtomicU64,
    errors_total: AtomicU64,
    inflight_requests: AtomicU64,
    streaming_requests_total: AtomicU64,
}

impl ServerMetrics {
    fn render(&self) -> String {
        format!(
            "# TYPE paged_requests_total counter\npaged_requests_total {}\n# TYPE paged_errors_total counter\npaged_errors_total {}\n# TYPE paged_inflight_requests gauge\npaged_inflight_requests {}\n# TYPE paged_streaming_requests_total counter\npaged_streaming_requests_total {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.errors_total.load(Ordering::Relaxed),
            self.inflight_requests.load(Ordering::Relaxed),
            self.streaming_requests_total.load(Ordering::Relaxed),
        )
    }
}

/// 引擎指标共享快照：由引擎循环每步结束后写入，/metrics 读取。
/// `kv_utilization_bp` 用 basis points（万分之）存储，避免 f32 原子。
#[derive(Default)]
struct SharedEngineMetrics {
    active_sequences: AtomicU64,
    kv_utilization_bp: AtomicU64,
    completed_requests: AtomicU64,
    failed_requests: AtomicU64,
    total_tokens_generated: AtomicU64,
}

impl SharedEngineMetrics {
    fn update(&self, metrics: &EngineMetrics) {
        self.active_sequences
            .store(metrics.active_sequences as u64, Ordering::Relaxed);
        self.kv_utilization_bp.store(
            (metrics.memory_utilization.clamp(0.0, 1.0) * 10_000.0).round() as u64,
            Ordering::Relaxed,
        );
        self.completed_requests
            .store(metrics.completed_requests, Ordering::Relaxed);
        self.failed_requests
            .store(metrics.failed_requests, Ordering::Relaxed);
        self.total_tokens_generated
            .store(metrics.total_tokens_generated, Ordering::Relaxed);
    }

    fn render(&self) -> String {
        format!(
            "# TYPE paged_engine_active_sequences gauge\npaged_engine_active_sequences {}\n\
             # TYPE paged_engine_kv_utilization gauge\npaged_engine_kv_utilization {}\n\
             # TYPE paged_engine_completed_requests counter\npaged_engine_completed_requests {}\n\
             # TYPE paged_engine_failed_requests counter\npaged_engine_failed_requests {}\n\
             # TYPE paged_engine_tokens_generated_total counter\npaged_engine_tokens_generated_total {}\n",
            self.active_sequences.load(Ordering::Relaxed),
            (self.kv_utilization_bp.load(Ordering::Relaxed) as f64) / 10_000.0,
            self.completed_requests.load(Ordering::Relaxed),
            self.failed_requests.load(Ordering::Relaxed),
            self.total_tokens_generated.load(Ordering::Relaxed),
        )
    }
}

/// RAII guard：保证 inflight gauge 在任何退出路径（含提前返回）都会递减。
/// 注意：不能派生 `Clone`——clone 只复制 Arc 而不 `fetch_add`，Drop 却必
/// `fetch_sub`，会造成计数下溢（`AtomicU64` 绕回）。guard 应始终通过
/// `InflightGuard::new` 构造并保持单一实例。
struct InflightGuard {
    metrics: Arc<ServerMetrics>,
}

impl InflightGuard {
    fn new(metrics: &Arc<ServerMetrics>) -> Self {
        metrics.inflight_requests.fetch_add(1, Ordering::Relaxed);
        Self {
            metrics: metrics.clone(),
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.metrics
            .inflight_requests
            .fetch_sub(1, Ordering::Relaxed);
    }
}

/// 提交给引擎循环的请求。
struct Submission {
    prompt: String,
    params: GenerationParams,
    /// 回传准入回执，或准入阶段的错误。
    admit: oneshot::Sender<Result<Admission, EngineError>>,
    /// 该请求的事件流（token 片段 + 终态）。
    events: mpsc::UnboundedSender<RequestEvent>,
    /// 请求生命周期取消信号：handler 侧 `RequestGuard` Drop 置位。
    /// 引擎循环每步检出 `has_changed() != Ok(false)`（含 sender 全 drop 的
    /// `Err`——owner 消失即取消），主动取消而非依赖下一次 chunk 发送失败。
    cancel: watch::Receiver<bool>,
}

/// handler 侧的请求所有权 guard：任何消费路径退出（事件 receiver drop、
/// SSE stream drop、unary future abort、n>1 部分准入失败时显式 drop）
/// 都在 Drop 中置位取消信号。
///
/// 对已到达终态的请求是**无害 no-op**：此时 waiter 已移除，
/// `cancel_by_request_id` 查无此项返回 false；request_id 不复用，无串扰。
struct RequestGuard {
    cancel: watch::Sender<bool>,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let _ = self.cancel.send(true);
    }
}

/// 引擎循环侧的每请求等待者：事件发送端 + 取消信号接收端。
struct Waiter {
    events: mpsc::UnboundedSender<RequestEvent>,
    cancel: watch::Receiver<bool>,
    /// 已对该请求签发过 cancel：token 置位状态会持续可读，避免每步重复
    /// 调用 cancel_by_request_id（终态经 Done 事件排出后才移除 waiter）。
    cancelled: bool,
}

/// 引擎循环接受请求后返回的回执。
struct Admission {
    request_id: RequestId,
    /// 用真实 tokenizer 统计的 prompt token 数（用于精确的 usage 报告）。
    prompt_tokens: usize,
}

/// 引擎循环推送给等待 handler 的每请求事件。
enum RequestEvent {
    /// 新生成的文本片段（每生成一个 token 推送一次）及该 token 的 logprob。
    Chunk(String, Option<TokenLogprobs>),
    /// 请求到达终态（成功或失败）。
    Done(CompletedRequest),
}

/// 把到达终态的请求路由给其等待者（Done 事件）；对端已断开时静默丢弃。
fn dispatch_completed(waiters: &mut HashMap<RequestId, Waiter>, completed: Vec<CompletedRequest>) {
    for completed in completed {
        if let Some(waiter) = waiters.remove(&completed.request_id) {
            let _ = waiter.events.send(RequestEvent::Done(completed));
        }
    }
}

struct AppState {
    config: EngineConfig,
    submit_tx: mpsc::Sender<Submission>,
    metrics: Arc<ServerMetrics>,
    engine_metrics: Arc<SharedEngineMetrics>,
    response_counter: Arc<AtomicU64>,
    /// 用于把 token id 解码为文本（logprobs 的 tokens 字段等）。
    tokenizer: Arc<dyn TokenizerTrait>,
}

impl AppState {
    fn next_id(&self, prefix: &str) -> String {
        let value = self.response_counter.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{value}")
    }

    /// 向引擎循环提交请求，返回准入回执、事件流与请求所有权 guard。
    /// guard 必须在请求消费方存活期间持有；其 Drop 主动置位取消信号。
    async fn submit(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<
        (
            Admission,
            mpsc::UnboundedReceiver<RequestEvent>,
            RequestGuard,
        ),
        ApiError,
    > {
        let (admit_tx, admit_rx) = oneshot::channel();
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        // guard 在 Submission 入队前创建：排队等待准入期间 client 断连，
        // guard Drop 同样置位，`admit_submission` 检出后跳过准入。
        let guard = RequestGuard { cancel: cancel_tx };
        self.submit_tx
            .send(Submission {
                prompt: prompt.to_string(),
                params,
                admit: admit_tx,
                events: events_tx,
                cancel: cancel_rx,
            })
            .await
            .map_err(|_| ApiError::internal("engine loop is not running"))?;
        let admission = admit_rx
            .await
            .map_err(|_| ApiError::internal("engine loop dropped the request"))??;
        Ok((admission, events_rx, guard))
    }

    /// 非流式生成：等待请求到达终态并返回完整结果。
    async fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationResult, ApiError> {
        // _guard 覆盖整个等待窗口：future 被 abort（client 断连）或任何
        // 提前返回时 Drop 置位，引擎循环下一步检出并释放资源。
        let (admission, mut events, _guard) = self.submit(prompt, params).await?;
        while let Some(event) = events.recv().await {
            if let RequestEvent::Done(completed) = event {
                return generation_result(admission.prompt_tokens, completed);
            }
            // 非流式模式下忽略 token 片段事件
        }
        Err(ApiError::internal(format!(
            "engine loop ended before request {} completed",
            admission.request_id
        )))
    }
}

/// 将终态请求转换为 API 结果；失败的请求映射为 500 错误。
fn generation_result(
    prompt_tokens: usize,
    completed: CompletedRequest,
) -> Result<GenerationResult, ApiError> {
    if completed.success {
        Ok(GenerationResult {
            text: completed.output_text,
            prompt_tokens,
            completion_tokens: completed.output_tokens.len(),
            finish_reason: completed
                .finish_reason
                .unwrap_or(FinishReason::Stop)
                .as_str()
                .to_string(),
            logprobs: completed.logprobs,
        })
    } else {
        Err(ApiError::internal(
            completed
                .error
                .unwrap_or_else(|| "generation failed".to_string()),
        ))
    }
}

/// HTTP API 错误，输出 OpenAI 风格的错误信封。
///
/// 将引擎分层错误映射到恰当的状态码：
/// 验证错误 → 400，过载（内存压力 / 并发上限）→ 429（含 `Retry-After`），
/// 资源不存在 → 404，其余内部错误 → 500。
enum ApiError {
    BadRequest(String),
    NotFound(String),
    Overloaded(String),
    Internal(String),
}

impl ApiError {
    fn internal(message: impl Into<String>) -> Self {
        ApiError::Internal(message.into())
    }
}

impl From<EngineError> for ApiError {
    fn from(err: EngineError) -> Self {
        match err {
            EngineError::EmptyInput
            | EngineError::InputTooLong(_, _)
            | EngineError::TotalLengthTooLong(_, _)
            | EngineError::InvalidMaxTokens(_)
            | EngineError::InvalidTemperature(_)
            | EngineError::InvalidTopP(_)
            | EngineError::TooManyStopSequences(_)
            | EngineError::UnsupportedGenerationMode(_) => ApiError::BadRequest(err.to_string()),
            EngineError::MemoryPressure | EngineError::MaxConcurrentSequencesReached(_) => {
                ApiError::Overloaded(err.to_string())
            }
            _ => ApiError::Internal(err.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "invalid_request_error", m),
            ApiError::Overloaded(m) => (StatusCode::TOO_MANY_REQUESTS, "overloaded_error", m),
            ApiError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", m),
        };
        let mut response = Json(serde_json::json!({
            "error": { "message": message, "type": error_type }
        }))
        .into_response();
        *response.status_mut() = status;
        if status == StatusCode::TOO_MANY_REQUESTS {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

/// 服务层内部的生成结果（HTTP 层私有类型，不属于库公共 API）。
#[derive(Debug)]
struct GenerationResult {
    /// 生成的文本
    text: String,
    /// prompt token 数
    prompt_tokens: usize,
    /// 生成 token 数
    completion_tokens: usize,
    /// 生成结束原因（stop / length）
    finish_reason: String,
    /// 每个生成 token 的 logprob（请求启用时）
    logprobs: Option<Vec<TokenLogprobs>>,
}

/// `stop` 停止序列：单个字符串或字符串数组（OpenAI 兼容形态）。
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    /// 是否等价于"不停止"（null / 空字符串 / 空数组）。
    fn is_empty(&self) -> bool {
        match self {
            StopSequence::Single(s) => s.is_empty(),
            StopSequence::Multiple(v) => v.is_empty(),
        }
    }

    /// 展开为字符串列表（PSERV-105：stop 已支持，交给引擎生成端检测）。
    fn to_vec(&self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s.clone()],
            StopSequence::Multiple(v) => v.clone(),
        }
    }
}

/// OpenAI `logprobs` 参数：bool 或 int（新版允许 int 指定 top 候选数）。
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LogprobsParam {
    Bool(bool),
    Count(u64),
}

/// 解析 `logprobs` 为引擎参数：`None` 表示不返回；
/// `Some(k)` 表示返回每个 token 的 logprob 及前 k 个候选（`1..=5`）。
fn logprobs_param(p: Option<LogprobsParam>) -> Result<Option<usize>, ApiError> {
    match p {
        None => Ok(None),
        Some(LogprobsParam::Bool(false)) => Ok(None),
        Some(LogprobsParam::Bool(true)) => Ok(Some(1)),
        Some(LogprobsParam::Count(0)) => Ok(None),
        Some(LogprobsParam::Count(k)) if k <= 5 => Ok(Some(k as usize)),
        Some(LogprobsParam::Count(k)) => Err(ApiError::BadRequest(format!(
            "logprobs must be between 0 and 5, got {k}"
        ))),
    }
}

/// CPU 后端未实现的 OpenAI 兼容参数（PSERV-104）。
///
/// 与 PSERV-101 的原则一致：不支持的参数在准入阶段显式拒绝（400），
/// 而不是被 serde 静默忽略。仅"非默认值"被拒绝——默认值（例如
/// `frequency_penalty=0`、`n=1`、`stop=null`/`[]`、`echo=false`）语义上
/// 无害，直接放行。其余完全未知的字段（如 `user`）仍被忽略，不影响
/// 生成语义。
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct UnsupportedParams {
    seed: Option<i64>,
    echo: Option<bool>,
    suffix: Option<String>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    best_of: Option<u32>,
    stream_options: Option<serde_json::Value>,
    // chat/completions 特有
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    response_format: Option<serde_json::Value>,
    function_call: Option<serde_json::Value>,
    functions: Option<serde_json::Value>,
}

/// 值是否为空数组（等价于"未提供"）。
fn is_empty_array(v: &serde_json::Value) -> bool {
    matches!(v, serde_json::Value::Array(a) if a.is_empty())
}

/// 拒绝未支持参数的非默认值（PSERV-104）。
///
/// 返回 400 + `invalid_request_error`，消息带参数名以便客户端定位。
fn reject_unsupported_params(p: &UnsupportedParams) -> Result<(), ApiError> {
    let unsupported = |name: &str| {
        Err(ApiError::BadRequest(format!(
            "parameter '{name}' is not supported by the CPU reference backend"
        )))
    };

    if p.seed.is_some() {
        return unsupported("seed");
    }
    if p.echo == Some(true) {
        return unsupported("echo");
    }
    if p.suffix.as_ref().is_some_and(|s| !s.is_empty()) {
        return unsupported("suffix");
    }
    // NaN 会绕过 `!= 0.0` 的「未启用」判断而被误当成 unsupported，这里先显式拒绝，
    // 给出明确的非法参数错误而非误导性的「不支持」。
    if p.frequency_penalty.is_some_and(|v| !v.is_finite()) {
        return Err(ApiError::BadRequest(
            "frequency_penalty must be a finite number".to_string(),
        ));
    }
    if p.presence_penalty.is_some_and(|v| !v.is_finite()) {
        return Err(ApiError::BadRequest(
            "presence_penalty must be a finite number".to_string(),
        ));
    }
    if p.frequency_penalty.is_some_and(|v| v != 0.0) {
        return unsupported("frequency_penalty");
    }
    if p.presence_penalty.is_some_and(|v| v != 0.0) {
        return unsupported("presence_penalty");
    }
    if p.best_of.is_some_and(|v| v != 1) {
        return unsupported("best_of");
    }
    if p.stream_options.is_some() {
        return unsupported("stream_options");
    }
    if p.tools.as_ref().is_some_and(|v| !is_empty_array(v)) {
        return unsupported("tools");
    }
    if p.tool_choice.is_some() {
        return unsupported("tool_choice");
    }
    if p.response_format.is_some() {
        return unsupported("response_format");
    }
    if p.function_call.is_some() {
        return unsupported("function_call");
    }
    if p.functions.as_ref().is_some_and(|v| !is_empty_array(v)) {
        return unsupported("functions");
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    stop: Option<StopSequence>,
    n: Option<u32>,
    logprobs: Option<LogprobsParam>,
    /// 调度优先级（PSERV-112）：数值越大越先调度，缺省 0。
    priority: Option<u8>,
    #[serde(flatten)]
    unsupported: UnsupportedParams,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    stop: Option<StopSequence>,
    n: Option<u32>,
    logprobs: Option<LogprobsParam>,
    /// 调度优先级（PSERV-112）：数值越大越先调度，缺省 0。
    priority: Option<u8>,
    #[serde(flatten)]
    unsupported: UnsupportedParams,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<LogprobsOutput>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<LogprobsOutput>,
}

/// OpenAI `logprobs` 响应块（每个 choice 一个）。
#[derive(Serialize)]
struct LogprobsOutput {
    /// 每个生成 token 的文本表示
    tokens: Vec<String>,
    /// 每个生成 token 的对数概率
    token_logprobs: Vec<f32>,
    /// 每位置前 k 个候选（token 文本 → logprob）
    top_logprobs: Vec<serde_json::Map<String, serde_json::Value>>,
    /// 每 token 在输出文本中的起始字节偏移
    /// （字节口径与引擎侧 [`crate::engine::tokens_before_char`] 的 stop 截断一致，
    /// 多字节 UTF-8（中文/emoji）下按字节累计才与 OpenAI 规范对齐）
    text_offset: Vec<usize>,
}

/// 从引擎返回的 logprob 信息构造 OpenAI logprobs 响应块。
///
/// `k` 为请求指定的 top 候选数；token 文本用服务层持有的 tokenizer 解码，
/// 特殊 token（解码为空）以 `<token_id=N>` 占位。
fn logprobs_output(
    logprobs: &[TokenLogprobs],
    tokenizer: &dyn TokenizerTrait,
    k: usize,
) -> LogprobsOutput {
    let mut tokens = Vec::with_capacity(logprobs.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.len());
    let mut text_offset = Vec::with_capacity(logprobs.len());
    let mut offset = 0usize;

    for lp in logprobs {
        let token_text = tokenizer.try_decode(&[lp.token]).unwrap_or_default();
        // 字节长度（str::len），与引擎 stop 截断的字节偏移口径一致。
        let token_len = token_text.len();
        let token_label = if token_text.is_empty() {
            format!("<token_id={}>", lp.token)
        } else {
            token_text
        };
        tokens.push(token_label);
        token_logprobs.push(lp.logprob);
        text_offset.push(offset);
        offset += token_len;

        let top = lp
            .top_logprobs
            .iter()
            .take(k)
            .map(|(id, logprob)| {
                let text = tokenizer.try_decode(&[*id]).unwrap_or_default();
                let label = if text.is_empty() {
                    format!("<token_id={id}>")
                } else {
                    text
                };
                (label, serde_json::json!(logprob))
            })
            .collect();
        top_logprobs.push(top);
    }

    LogprobsOutput {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

/// 为单个生成结果构造 logprobs 响应块（请求未启用时为 None）。
fn choice_logprobs(
    state: &Arc<AppState>,
    result: &GenerationResult,
    k: usize,
) -> Option<LogprobsOutput> {
    result
        .logprobs
        .as_ref()
        .map(|lps| logprobs_output(lps, state.tokenizer.as_ref(), k))
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

struct PreparedGenerationRequest {
    model: String,
    prompt: String,
    params: GenerationParams,
    stream: bool,
    /// 候选数（OpenAI `n`）：同一 prompt 生成 n 个独立候选。
    n: usize,
}

/// 根据配置创建 router（构造默认引擎并启动引擎循环）
///
/// 必须在 tokio 运行时内调用（后台引擎循环经由 `tokio::spawn` 启动）。
pub fn create_router(config: EngineConfig) -> Result<Router, EngineError> {
    let engine = InferenceEngine::new(config.clone())?;
    create_router_with_engine(config, engine)
}

/// 使用预先构造的引擎创建 router（用于测试注入自定义执行器）
///
/// 必须在 tokio 运行时内调用。
pub fn create_router_with_engine(
    config: EngineConfig,
    engine: InferenceEngine,
) -> Result<Router, EngineError> {
    let (router, _shutdown_tx) = create_router_with_engine_and_shutdown(config, engine)?;
    Ok(router)
}

/// 创建 router 并返回 shutdown 广播触发端。
///
/// 对返回的 sender `send(true)` 会令引擎循环取消全部在途请求：SSE 流
/// 收到终态后结束，axum graceful drain 因而有界，不会被任意长的生成
/// 挂起。引擎循环自身持有保活 sender 克隆，仅显式 `send(true)` 触发
/// shutdown；单纯 drop 该句柄不会误触发（router 可先于流式响应体析构）。
pub fn create_router_with_engine_and_shutdown(
    config: EngineConfig,
    engine: InferenceEngine,
) -> Result<(Router, watch::Sender<bool>), EngineError> {
    config.validate()?;
    let tokenizer: Arc<dyn TokenizerTrait> = Arc::from(build_tokenizer(&config)?);
    let (submit_tx, submit_rx) = mpsc::channel(SUBMISSION_QUEUE_CAPACITY);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let engine_metrics = Arc::new(SharedEngineMetrics::default());
    let engine_metrics_loop = engine_metrics.clone();
    // 保活 sender 由引擎循环自身持有：通道只在显式 send(true) 时置位。
    // （若把保活 sender 放在 AppState，router 先于响应体流析构——如
    // oneshot 测试——sender 全 drop 会被 has_changed 的 Err 误读为 shutdown。）
    let shutdown_keepalive = shutdown_tx.clone();
    tokio::spawn(engine_loop(
        engine,
        submit_rx,
        engine_metrics_loop,
        shutdown_rx,
        shutdown_keepalive,
    ));

    let state = Arc::new(AppState {
        config,
        submit_tx,
        metrics: Arc::new(ServerMetrics::default()),
        engine_metrics,
        response_counter: Arc::new(AtomicU64::new(1)),
        tokenizer,
    });

    let router = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(not_found)
        .with_state(state);

    Ok((router, shutdown_tx))
}

/// 检出每请求取消信号与 shutdown 广播，主动取消对应请求。
///
/// 判定口径 `has_changed() != Ok(false)`：`Ok(true)` 为显式置位，
/// `Err` 为 sender 全部 drop（owner 已消失，同样必须取消）。
/// cancel 后请求立即进入 completed_requests，此时 has_pending_work 可能
/// 已为 false——必须当步排出 Done，否则下一轮循环阻塞在 `recv()` 上，
/// 终态事件滞留（shutdown 时 SSE 流悬挂）。
fn cancel_flagged(
    engine: &mut InferenceEngine,
    waiters: &mut HashMap<RequestId, Waiter>,
    shutdown_rx: &watch::Receiver<bool>,
) {
    let shutdown = !matches!(shutdown_rx.has_changed(), Ok(false));
    let mut flagged: Vec<RequestId> = Vec::new();
    for (id, waiter) in waiters.iter_mut() {
        if !waiter.cancelled && (shutdown || !matches!(waiter.cancel.has_changed(), Ok(false))) {
            waiter.cancelled = true;
            flagged.push(*id);
        }
    }
    if flagged.is_empty() {
        return;
    }
    for id in flagged {
        engine.cancel_request(id);
    }
    let (completed, _) = engine.collect_completed_requests();
    dispatch_completed(waiters, completed);
}

/// 后台引擎循环：引擎的唯一所有者。
///
/// 每一步之间清空提交队列（使调度器能看到并发请求并组批），并把每请求事件
/// 推送给对应 handler。所有提交端被丢弃（服务器关闭）时退出。
async fn engine_loop(
    mut engine: InferenceEngine,
    mut submit_rx: mpsc::Receiver<Submission>,
    engine_metrics: Arc<SharedEngineMetrics>,
    shutdown_rx: watch::Receiver<bool>,
    // 保活 sender：与信号消费方同寿，通道不会因 router/handler 先析构
    // 而意外关闭（sender 全 drop → has_changed Err → 误判 shutdown）。
    _shutdown_keepalive: watch::Sender<bool>,
) {
    // request_id → 等待者（事件发送端 + 取消信号接收端）
    let mut waiters: HashMap<RequestId, Waiter> = HashMap::new();

    // 防死循环（B12 服务层）：pending 因 KV 预算/块不足而永远无法启动时，
    // has_pending_work() 恒真但每步空转（无序列执行、无完成排出、无在途序列）。
    // 连续 STALL_LIMIT 步无任何进展即判定卡死：把卡住的 pending 请求标记失败
    // 并排出（客户端拿到明确错误），而不是 100% CPU 忙等阻塞整个服务。
    // 与 InferenceEngine::run 的 STALL_LIMIT 语义一致（run 侧会同时兜底）。
    const STALL_LIMIT: u32 = 100;
    let mut idle_steps: u32 = 0;

    loop {
        // 空闲时阻塞等待首个提交，避免忙等。
        if !engine.has_pending_work() {
            match submit_rx.recv().await {
                Some(submission) => admit_submission(&mut engine, submission, &mut waiters),
                None => return, // 所有提交端已丢弃
            }
        }

        // 非阻塞地清空队列中所有提交，让调度器看到并发请求。
        while let Ok(submission) = submit_rx.try_recv() {
            admit_submission(&mut engine, submission, &mut waiters);
        }

        // 主动取消检出：guard drop（SSE 断开、unary abort、n>1 部分准入
        // 失败、排队期 owner 消失）或 shutdown 广播 → 当步取消并排出终态，
        // 不再依赖下一次 chunk 发送失败的偶然时点。
        cancel_flagged(&mut engine, &mut waiters, &shutdown_rx);

        let mut progressed = false;
        match engine.step_events() {
            Ok(events) => {
                let completed_len = events.completed.len();
                let chunks_len = events.chunks.len();
                let mut disconnected: Vec<RequestId> = Vec::new();
                for (request_id, chunk, logprobs) in events.chunks {
                    if let Some(waiter) = waiters.get(&request_id) {
                        // 发送失败 == 对端已断开：登记取消，释放调度资源。
                        // （兜底路径：正常断开已由 cancel token 检出。）
                        if waiter
                            .events
                            .send(RequestEvent::Chunk(chunk, logprobs))
                            .is_err()
                        {
                            disconnected.push(request_id);
                        }
                    }
                }
                for request_id in disconnected {
                    waiters.remove(&request_id);
                    engine.cancel_request(request_id);
                }
                dispatch_completed(&mut waiters, events.completed);
                // 有请求到达终态、生成过 token 片段、或仍有序列在途（执行中）
                // 即视为有进展（后两者覆盖"解码 token 落到空文本"的边界）
                progressed = completed_len > 0
                    || chunks_len > 0
                    || engine.get_metrics().active_sequences > 0;
            }
            Err(err) => {
                // 可归属的序列已在 step_events 内被标记失败并经 Done 事件上报；
                // 此处仅记录无法归属的步骤级错误。
                log::error!("engine step failed: {err}");
            }
        }

        // 防死循环（B12 服务层）：连续 STALL_LIMIT 步无进展即判定卡死。
        // 把无法启动的 pending 请求标记失败并排出（客户端拿到明确错误），
        // 而不是无限忙等阻塞整个服务。
        if progressed {
            idle_steps = 0;
        } else {
            idle_steps += 1;
            if idle_steps >= STALL_LIMIT {
                log::error!(
                    "服务引擎连续 {STALL_LIMIT} 步无进展：pending 请求因 KV 预算/块不足 \
                     永远无法启动，将其标记失败并排出（避免忙等阻塞服务）"
                );
                engine.fail_pending_requests("request 因 KV 预算/块不足永远无法启动");
                // 立即把失败终态排出并路由：若留到下一轮，循环会在
                // has_pending_work()==false 时阻塞在 recv()，Done 事件将滞留。
                let (stalled, _) = engine.collect_completed_requests();
                dispatch_completed(&mut waiters, stalled);
                idle_steps = 0;
            }
        }

        // 每步结束后刷新引擎指标快照，供 /metrics 读取。
        engine_metrics.update(&engine.get_metrics());

        // 每步让出一次：生成循环本身没有 await 点，单线程 runtime 上
        // 会饿死 handler / SSE 流任务（首 token 永远到不了客户端）；
        // 多线程 runtime 上也避免独占 worker。
        tokio::task::yield_now().await;
    }
}

fn admit_submission(
    engine: &mut InferenceEngine,
    submission: Submission,
    waiters: &mut HashMap<RequestId, Waiter>,
) {
    // owner 已在 submission 排队期间消失（guard Drop / cancel 置位 /
    // sender 全 drop）→ 跳过准入，省一次调度/KV 分配；admit 回执无人
    // 接收，静默丢弃。
    if !matches!(submission.cancel.has_changed(), Ok(false)) {
        return;
    }
    let result = engine
        .submit_request(&submission.prompt, submission.params)
        .map(|(request_id, prompt_tokens)| Admission {
            request_id,
            prompt_tokens,
        });
    if let Ok(admission) = &result {
        waiters.insert(
            admission.request_id,
            Waiter {
                events: submission.events,
                cancel: submission.cancel,
                cancelled: false,
            },
        );
    }
    let _ = submission.admit.send(result);
}

async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// 就绪探针：仅当引擎循环仍在运行时返回 ready。
async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    if state.submit_tx.is_closed() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "not_ready" })),
        )
            .into_response()
    } else {
        Json(serde_json::json!({ "status": "ready" })).into_response()
    }
}

async fn not_found(uri: Uri) -> ApiError {
    ApiError::NotFound(format!("unknown route: {}", uri.path()))
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = format!(
        "{}{}",
        state.metrics.render(),
        state.engine_metrics.render()
    );
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        body,
    )
        .into_response()
}

async fn completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let _guard = InflightGuard::new(&state.metrics);
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    let Json(request) = match body {
        Ok(json) => json,
        Err(rejection) => return ApiError::BadRequest(rejection.body_text()).into_response(),
    };

    let prepared = match prepare_completion_request(&state, request) {
        Ok(prepared) => prepared,
        Err(err) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return err.into_response();
        }
    };

    respond_generation(&state, StreamKind::Completion, "cmpl", prepared).await
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let _guard = InflightGuard::new(&state.metrics);
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    let Json(request) = match body {
        Ok(json) => json,
        Err(rejection) => return ApiError::BadRequest(rejection.body_text()).into_response(),
    };

    let prepared = match prepare_chat_request(&state, request) {
        Ok(prepared) => prepared,
        Err(err) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return err.into_response();
        }
    };

    respond_generation(&state, StreamKind::Chat, "chatcmpl", prepared).await
}

/// completions 与 chat_completions 共用的生成/响应流程。
async fn respond_generation(
    state: &Arc<AppState>,
    kind: StreamKind,
    id_prefix: &str,
    prepared: PreparedGenerationRequest,
) -> Response {
    if prepared.stream {
        // 流式：为每个候选提交并 fan-in 为单一 SSE 流；guards 随流持有，
        // 流 drop（client 断开）时全部候选被主动取消。
        let mut streams = Vec::with_capacity(prepared.n);
        let mut guards = Vec::with_capacity(prepared.n);
        let mut prompt_tokens = 0;
        for _ in 0..prepared.n {
            match state
                .submit(&prepared.prompt, prepared.params.clone())
                .await
            {
                Ok((admission, events, guard)) => {
                    if streams.is_empty() {
                        prompt_tokens = admission.prompt_tokens;
                    }
                    streams.push(events);
                    guards.push(guard);
                }
                Err(err) => {
                    // n>1 部分准入失败：drop 已收集的 guards → 主动取消
                    // 已准入候选，不再等下一次 chunk send 失败回收。
                    drop(guards);
                    state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                    return err.into_response();
                }
            }
        }
        state
            .metrics
            .streaming_requests_total
            .fetch_add(1, Ordering::Relaxed);
        let k = prepared.params.logprobs.unwrap_or(1);
        let ctx = StreamContext {
            state,
            kind,
            id_prefix,
            model: &prepared.model,
            prompt_tokens,
            k,
        };
        if prepared.n == 1 {
            stream_response(
                ctx,
                streams.pop().expect("n == 1 has exactly one stream"),
                guards.pop().expect("n == 1 has exactly one guard"),
            )
        } else {
            stream_response_multi(ctx, streams, guards)
        }
    } else if prepared.n == 1 {
        let k = prepared.params.logprobs.unwrap_or(1);
        match state.generate(&prepared.prompt, prepared.params).await {
            Ok(generated) => match kind {
                StreamKind::Completion => {
                    let logprobs = choice_logprobs(state, &generated, k);
                    let response_usage = usage(&generated);
                    Json(CompletionResponse {
                        id: state.next_id(id_prefix),
                        object: "text_completion",
                        created: unix_timestamp(),
                        model: prepared.model,
                        choices: vec![CompletionChoice {
                            text: generated.text,
                            index: 0,
                            finish_reason: generated.finish_reason,
                            logprobs,
                        }],
                        usage: response_usage,
                    })
                    .into_response()
                }
                StreamKind::Chat => {
                    let logprobs = choice_logprobs(state, &generated, k);
                    let response_usage = usage(&generated);
                    Json(ChatCompletionResponse {
                        id: state.next_id(id_prefix),
                        object: "chat.completion",
                        created: unix_timestamp(),
                        model: prepared.model,
                        choices: vec![ChatCompletionChoice {
                            index: 0,
                            message: ChatMessage {
                                role: "assistant".to_string(),
                                content: generated.text,
                            },
                            finish_reason: generated.finish_reason,
                            logprobs,
                        }],
                        usage: response_usage,
                    })
                    .into_response()
                }
            },
            Err(err) => {
                state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                err.into_response()
            }
        }
    } else {
        // 非流式 n>1：并行生成 n 个候选，响应含 n 个 choices
        match generate_many(state, &prepared).await {
            Ok(generated) => {
                match kind {
                    StreamKind::Completion => {
                        let response_usage = usage_multi(&generated);
                        let k = prepared.params.logprobs.unwrap_or(1);
                        Json(CompletionResponse {
                            id: state.next_id(id_prefix),
                            object: "text_completion",
                            created: unix_timestamp(),
                            model: prepared.model,
                            choices: generated
                                .into_iter()
                                .enumerate()
                                .map(|(i, g)| CompletionChoice {
                                    text: g.text,
                                    index: i as u32,
                                    finish_reason: g.finish_reason,
                                    logprobs: g.logprobs.as_ref().map(|lps| {
                                        logprobs_output(lps, state.tokenizer.as_ref(), k)
                                    }),
                                })
                                .collect(),
                            usage: response_usage,
                        })
                        .into_response()
                    }
                    StreamKind::Chat => {
                        let response_usage = usage_multi(&generated);
                        let k = prepared.params.logprobs.unwrap_or(1);
                        Json(ChatCompletionResponse {
                            id: state.next_id(id_prefix),
                            object: "chat.completion",
                            created: unix_timestamp(),
                            model: prepared.model,
                            choices: generated
                                .into_iter()
                                .enumerate()
                                .map(|(i, g)| ChatCompletionChoice {
                                    index: i as u32,
                                    message: ChatMessage {
                                        role: "assistant".to_string(),
                                        content: g.text,
                                    },
                                    finish_reason: g.finish_reason,
                                    logprobs: g.logprobs.as_ref().map(|lps| {
                                        logprobs_output(lps, state.tokenizer.as_ref(), k)
                                    }),
                                })
                                .collect(),
                            usage: response_usage,
                        })
                        .into_response()
                    }
                }
            }
            Err(err) => {
                state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                err.into_response()
            }
        }
    }
}

/// OpenAI `n` 候选数的上限（架构练习默认 16，避免无界资源占用）。
const MAX_CANDIDATES: usize = 16;

/// SSE payload 的公共元数据（避免逐方法重复的 id/created/model 参数）。
struct SseMeta<'a> {
    id: &'a str,
    created: u64,
    model: &'a str,
}

/// 流式响应的两种载荷形状。
#[derive(Clone, Copy)]
enum StreamKind {
    Completion,
    Chat,
}

impl StreamKind {
    /// 单 choice 的中间 chunk（`n == 1` 特例）。
    fn chunk_payload(
        &self,
        meta: &SseMeta,
        chunk: &str,
        logprobs: Option<&LogprobsOutput>,
    ) -> serde_json::Value {
        self.multi_chunk_payload(meta, 1, 0, chunk, logprobs)
    }

    /// `n` 候选的中间 chunk：n 个 choice，仅事件来源 `index` 携带增量。
    fn multi_chunk_payload(
        &self,
        meta: &SseMeta,
        n: usize,
        index: usize,
        chunk: &str,
        logprobs: Option<&LogprobsOutput>,
    ) -> serde_json::Value {
        let object = match self {
            StreamKind::Completion => "text_completion",
            StreamKind::Chat => "chat.completion.chunk",
        };
        let id = meta.id;
        let created = meta.created;
        let model = meta.model;
        let logprobs_value = logprobs
            .map(serde_json::to_value)
            .transpose()
            .unwrap_or(None);
        let choices: Vec<serde_json::Value> = (0..n)
            .map(|i| {
                let mut choice = match self {
                    StreamKind::Completion if i == index => serde_json::json!({
                        "text": chunk, "index": i, "finish_reason": serde_json::Value::Null
                    }),
                    StreamKind::Completion => serde_json::json!({
                        "text": "", "index": i, "finish_reason": serde_json::Value::Null
                    }),
                    StreamKind::Chat if i == index => serde_json::json!({
                        "index": i, "delta": {"content": chunk}, "finish_reason": serde_json::Value::Null
                    }),
                    StreamKind::Chat => serde_json::json!({
                        "index": i, "delta": {}, "finish_reason": serde_json::Value::Null
                    }),
                };
                if i == index {
                    if let Some(lp) = &logprobs_value {
                        choice["logprobs"] = lp.clone();
                    }
                }
                choice
            })
            .collect();
        serde_json::json!({
            "id": id,
            "object": object,
            "created": created,
            "model": model,
            "choices": choices,
        })
    }

    /// 单 choice 的终止 chunk（`n == 1` 特例）。
    fn final_payload(
        &self,
        id: &str,
        created: u64,
        model: &str,
        usage: &Usage,
        finish_reason: &str,
    ) -> serde_json::Value {
        self.multi_final_payload(id, created, model, &[finish_reason], usage)
    }

    /// `n` 候选的终止 chunk：n 个 finish_reason + 聚合 usage。
    fn multi_final_payload(
        &self,
        id: &str,
        created: u64,
        model: &str,
        finish_reasons: &[&str],
        usage: &Usage,
    ) -> serde_json::Value {
        let object = match self {
            StreamKind::Completion => "text_completion",
            StreamKind::Chat => "chat.completion.chunk",
        };
        let choices: Vec<serde_json::Value> = finish_reasons
            .iter()
            .enumerate()
            .map(|(i, finish_reason)| match self {
                StreamKind::Completion => serde_json::json!({
                    "text": "", "index": i, "finish_reason": finish_reason
                }),
                StreamKind::Chat => serde_json::json!({
                    "index": i, "delta": {}, "finish_reason": finish_reason
                }),
            })
            .collect();
        let mut payload = serde_json::json!({
            "id": id,
            "object": object,
            "created": created,
            "model": model,
            "choices": choices,
        });
        payload["usage"] = serde_json::json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        });
        payload
    }
}

/// 流式响应的公共上下文（把参数数收敛到 clippy 上限内）。
struct StreamContext<'a> {
    state: &'a Arc<AppState>,
    kind: StreamKind,
    id_prefix: &'a str,
    model: &'a str,
    prompt_tokens: usize,
    k: usize,
}

/// token 级流式响应：随引擎循环推送的事件逐片段发送，
/// 首字节延迟 = 首 token 生成时间（而非完整生成时间）。
fn stream_response(
    ctx: StreamContext<'_>,
    mut events: mpsc::UnboundedReceiver<RequestEvent>,
    guard: RequestGuard,
) -> Response {
    let StreamContext {
        state,
        kind,
        id_prefix,
        model,
        prompt_tokens,
        k,
    } = ctx;
    let id = state.next_id(id_prefix);
    let created = unix_timestamp();
    let model = model.to_string();
    let tokenizer = state.tokenizer.clone();
    Sse::new(stream! {
        // 持有请求 guard 覆盖流的整个生命周期：client 断开 → stream drop
        // → guard Drop 置位取消信号，engine loop 下一步检出（主动取消）。
        let _guard = guard;
        let mut failure: Option<String> = None;
        let mut terminated = false;
        while let Some(event) = events.recv().await {
            match event {
                RequestEvent::Chunk(chunk, logprobs) => {
                    let lp_out = logprobs
                        .as_ref()
                        .map(|lp| logprobs_output(std::slice::from_ref(lp), tokenizer.as_ref(), k));
                    let meta = SseMeta {
                        id: &id,
                        created,
                        model: &model,
                    };
                    yield Ok::<Event, Infallible>(Event::default()
                        .data(kind.chunk_payload(&meta, &chunk, lp_out.as_ref()).to_string()));
                }
                RequestEvent::Done(completed) => {
                    terminated = true;
                    if completed.success {
                        let usage = Usage {
                            prompt_tokens,
                            completion_tokens: completed.output_tokens.len(),
                            total_tokens: prompt_tokens + completed.output_tokens.len(),
                        };
                        let finish_reason = completed
                            .finish_reason
                            .unwrap_or(FinishReason::Stop)
                            .as_str();
                        yield Ok::<Event, Infallible>(Event::default()
                            .data(kind.final_payload(&id, created, &model, &usage, finish_reason)
                                .to_string()));
                    } else {
                        failure = completed.error;
                    }
                    break;
                }
            }
        }
        if !terminated {
            // 通道关闭但从未收到终态（如引擎循环异常退出）：
            // 明确报错而不是伪装成正常结束。
            let payload = serde_json::json!({
                "error": {
                    "message": "engine loop ended before request completed",
                    "type": "internal_error"
                }
            });
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        if let Some(message) = failure {
            let payload = serde_json::json!({
                "error": { "message": message, "type": "internal_error" }
            });
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    })
    .keep_alive(KeepAlive::default())
    .into_response()
}

/// 流式 n>1：把 n 个候选事件流 fan-in 为单一 SSE 流（PSERV-106）。
///
/// 每个事件发出一个含 n 个 choice 的 chunk（仅事件来源 index 携带增量，
/// 其余为空占位）；全部候选到达终态后发出终止 chunk（n 个 finish_reason
/// + 聚合 usage）。任一候选失败 → error chunk 整体失败。
fn stream_response_multi(
    ctx: StreamContext<'_>,
    streams: Vec<mpsc::UnboundedReceiver<RequestEvent>>,
    guards: Vec<RequestGuard>,
) -> Response {
    let StreamContext {
        state,
        kind,
        id_prefix,
        model,
        prompt_tokens,
        k,
    } = ctx;
    let id = state.next_id(id_prefix);
    let created = unix_timestamp();
    let model = model.to_string();
    let n = streams.len();
    let tokenizer = state.tokenizer.clone();
    Sse::new(stream! {
        // 全部候选的 guard 随聚合流持有：流结束/drop 时统一取消。
        let _guards = guards;
        // fan-in：每个候选一个转发任务，事件带候选 index 进入统一通道
        let (tx, mut rx) = mpsc::unbounded_channel::<(usize, RequestEvent)>();
        let mut tasks = JoinSet::new();
        for (i, mut events) in streams.into_iter().enumerate() {
            let tx = tx.clone();
            tasks.spawn(async move {
                while let Some(event) = events.recv().await {
                    if tx.send((i, event)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);

        let mut failure: Option<String> = None;
        let mut terminated = false;
        let mut remaining = n;
        let mut finish_reasons: Vec<Option<String>> = vec![None; n];
        let mut completion_tokens: Vec<usize> = vec![0; n];

        while let Some((i, event)) = rx.recv().await {
            match event {
                RequestEvent::Chunk(chunk, logprobs) => {
                    let lp_out = logprobs
                        .as_ref()
                        .map(|lp| logprobs_output(std::slice::from_ref(lp), tokenizer.as_ref(), k));
                    let meta = SseMeta {
                        id: &id,
                        created,
                        model: &model,
                    };
                    let payload =
                        kind.multi_chunk_payload(&meta, n, i, &chunk, lp_out.as_ref());
                    yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                }
                RequestEvent::Done(completed) => {
                    remaining -= 1;
                    if completed.success {
                        finish_reasons[i] = Some(
                            completed
                                .finish_reason
                                .unwrap_or(FinishReason::Stop)
                                .as_str()
                                .to_string(),
                        );
                        completion_tokens[i] = completed.output_tokens.len();
                    } else {
                        failure = completed.error;
                        break;
                    }
                    if remaining == 0 {
                        terminated = true;
                        let completion_total: usize = completion_tokens.iter().sum();
                        let usage = Usage {
                            prompt_tokens,
                            completion_tokens: completion_total,
                            total_tokens: prompt_tokens + completion_total,
                        };
                        let reasons: Vec<&str> = finish_reasons
                            .iter()
                            .map(|r| r.as_deref().unwrap_or("stop"))
                            .collect();
                        let payload = kind.multi_final_payload(&id, created, &model, &reasons, &usage);
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                        break;
                    }
                }
            }
        }
        if !terminated && failure.is_none() {
            // 通道关闭但未收到全部终态：明确报错而非伪装成正常结束
            let payload = serde_json::json!({
                "error": {
                    "message": "engine loop ended before all candidates completed",
                    "type": "internal_error"
                }
            });
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        if let Some(message) = failure {
            let payload = serde_json::json!({
                "error": { "message": message, "type": "internal_error" }
            });
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    })
    .keep_alive(KeepAlive::default())
    .into_response()
}

fn prepare_completion_request(
    state: &AppState,
    request: CompletionRequest,
) -> Result<PreparedGenerationRequest, ApiError> {
    reject_unsupported_params(&request.unsupported)?;
    let prompt = validate_prompt(request.prompt)?;
    Ok(PreparedGenerationRequest {
        model: resolve_model(state, request.model)?,
        prompt,
        params: generation_params(
            request.max_tokens,
            request.temperature,
            request.top_p,
            stop_sequences(request.stop),
            logprobs_param(request.logprobs)?,
            request.priority,
        )?,
        stream: request.stream.unwrap_or(false),
        n: candidate_count(request.n)?,
    })
}

fn prepare_chat_request(
    state: &AppState,
    request: ChatCompletionRequest,
) -> Result<PreparedGenerationRequest, ApiError> {
    reject_unsupported_params(&request.unsupported)?;
    if request.messages.is_empty() {
        return Err(ApiError::BadRequest(
            "messages must not be empty".to_string(),
        ));
    }
    for message in &request.messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err(ApiError::BadRequest(format!(
                "invalid role '{}': expected one of system, user, assistant",
                message.role
            )));
        }
    }

    let prompt = validate_prompt(build_chat_prompt(
        &state.config.tokenizer.kind,
        &request.messages,
    ))?;

    Ok(PreparedGenerationRequest {
        model: resolve_model(state, request.model)?,
        prompt,
        params: generation_params(
            request.max_tokens,
            request.temperature,
            request.top_p,
            stop_sequences(request.stop),
            logprobs_param(request.logprobs)?,
            request.priority,
        )?,
        stream: request.stream.unwrap_or(false),
        n: candidate_count(request.n)?,
    })
}

/// 构建 Chat Completions 的 prompt 文本。
///
/// - [`TokenizerKind::HuggingFace`]：应用 Qwen2 的 chat template（`<|im_start|>`），
///   与 HF 模型词表对齐；其他模型需按需扩展。
/// - [`TokenizerKind::Simple`]：保持简单的 `role: content` 文本拼接（默认/测试）。
fn build_chat_prompt(kind: &TokenizerKind, messages: &[ChatMessage]) -> String {
    match kind {
        TokenizerKind::HuggingFace => qwen2_chat_prompt(messages),
        TokenizerKind::Simple => messages
            .iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Qwen2 系模型的 chat template：
///
/// - 有 system 消息时以 `<|im_start|>system\n{content}<|im_end|>` 开头；
/// - 每个 user/assistant 消息为 `<|im_start|>{role}\n{content}<|im_end|>`；
/// - 最后追加 `<|im_start|>assistant`，提示模型开始回复。
fn qwen2_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::with_capacity(messages.len() + 1);
    for (i, message) in messages.iter().enumerate() {
        // 仅第一条 system 消息按 system 特殊处理；后续重复按普通消息（防御）。
        if i == 0 && message.role == "system" {
            parts.push(format!("<|im_start|>system\n{}<|im_end|>", message.content));
        } else {
            parts.push(format!(
                "<|im_start|>{}\n{}<|im_end|>",
                message.role, message.content
            ));
        }
    }
    parts.push("<|im_start|>assistant".to_string());
    parts.join("\n")
}

/// 将请求的 `stop` 字段展开为引擎使用的停止序列列表。
/// 空序列（null / 空字符串 / 空数组）等价于未提供。
fn stop_sequences(stop: Option<StopSequence>) -> Vec<String> {
    match stop {
        Some(s) if s.is_empty() => Vec::new(),
        Some(s) => s.to_vec(),
        None => Vec::new(),
    }
}

/// 校验请求的 model 字段：缺省时使用配置的模型名，
/// 显式指定但不匹配时返回 404（OpenAI 惯例）。
fn resolve_model(state: &AppState, requested: Option<String>) -> Result<String, ApiError> {
    match requested {
        Some(name) if name == state.config.serving.model_name => Ok(name),
        Some(name) => Err(ApiError::NotFound(format!("model '{name}' not found"))),
        None => Ok(state.config.serving.model_name.clone()),
    }
}

fn generation_params(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Vec<String>,
    logprobs: Option<usize>,
    priority: Option<u8>,
) -> Result<GenerationParams, ApiError> {
    // 缺省值即 greedy（后端当前唯一支持的生成模式）；显式传入其他
    // 采样参数的请求会在 submit 阶段以 400 拒绝，而非静默降级。
    let params = GenerationParams {
        max_tokens: max_tokens.unwrap_or(16),
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        stop,
        logprobs,
        // priority 直接透传（PSERV-112），缺省 0。
        priority: priority.unwrap_or(0),
    };
    params
        .validate()
        .map_err(|err| ApiError::BadRequest(err.to_string()))?;
    Ok(params)
}

fn validate_prompt(prompt: String) -> Result<String, ApiError> {
    if prompt.trim().is_empty() {
        Err(ApiError::BadRequest("prompt must not be empty".to_string()))
    } else {
        Ok(prompt)
    }
}

/// n 候选数解析与校验（OpenAI `n`，默认 1，上限 [`MAX_CANDIDATES`]）。
fn candidate_count(n: Option<u32>) -> Result<usize, ApiError> {
    let n = n.unwrap_or(1) as usize;
    if n == 0 {
        return Err(ApiError::BadRequest("n must be at least 1".to_string()));
    }
    if n > MAX_CANDIDATES {
        return Err(ApiError::BadRequest(format!(
            "n must be at most {MAX_CANDIDATES}"
        )));
    }
    Ok(n)
}

/// n 个候选的聚合 usage：prompt 只计一次，completion 为各候选之和。
fn usage_multi(results: &[GenerationResult]) -> Usage {
    let prompt = results.first().map(|r| r.prompt_tokens).unwrap_or(0);
    let completion = results.iter().map(|r| r.completion_tokens).sum();
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

/// 非流式 n：并行生成 n 个候选；任一候选失败即整体失败。
async fn generate_many(
    state: &Arc<AppState>,
    prepared: &PreparedGenerationRequest,
) -> Result<Vec<GenerationResult>, ApiError> {
    let mut set = JoinSet::new();
    for _ in 0..prepared.n {
        let state = state.clone();
        let prompt = prepared.prompt.clone();
        let params = prepared.params.clone();
        set.spawn(async move { state.generate(&prompt, params).await });
    }
    let mut generated = Vec::with_capacity(prepared.n);
    while let Some(result) = set.join_next().await {
        match result {
            Ok(Ok(g)) => generated.push(g),
            Ok(Err(err)) => {
                // 任一候选失败即整体失败：显式取消剩余候选，避免它们继续占用
                // 调度器序列槽位与 KV，直到自然完成。
                set.abort_all();
                return Err(err);
            }
            Err(join_err) => {
                set.abort_all();
                return Err(ApiError::internal(format!(
                    "candidate task failed: {join_err}"
                )));
            }
        }
    }
    Ok(generated)
}

fn usage(result: &GenerationResult) -> Usage {
    Usage {
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        total_tokens: result.prompt_tokens + result.completion_tokens,
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn test_qwen2_chat_prompt_with_system() {
        let messages = vec![
            msg("system", "You are helpful"),
            msg("user", "Hello"),
            msg("assistant", "Hi there"),
        ];
        let prompt = qwen2_chat_prompt(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are helpful<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\nHi there<|im_end|>\n\
             <|im_start|>assistant"
        );
    }

    #[test]
    fn test_qwen2_chat_prompt_without_system() {
        let messages = vec![msg("user", "Hello"), msg("assistant", "Hi")];
        let prompt = qwen2_chat_prompt(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\nHi<|im_end|>\n\
             <|im_start|>assistant"
        );
    }

    #[test]
    fn test_build_chat_prompt_simple_keeps_role_concat() {
        let messages = vec![msg("user", "Hello"), msg("assistant", "Hi")];
        let prompt = build_chat_prompt(&TokenizerKind::Simple, &messages);
        assert_eq!(prompt, "user: Hello\nassistant: Hi");
    }

    #[test]
    fn test_build_chat_prompt_huggingface_uses_qwen2() {
        let messages = vec![msg("user", "Hello")];
        let prompt = build_chat_prompt(&TokenizerKind::HuggingFace, &messages);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant"
        );
    }

    /// B4 回归：HTTP 请求中的 `priority` 字段必须被解析并透传到
    /// `GenerationParams`，而不是被静默丢弃。
    #[test]
    fn test_completion_request_parses_priority_field() {
        let body: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"hi","priority":7}"#).unwrap();
        assert_eq!(body.priority, Some(7), "priority 字段应被解析");

        // 缺省时 priority 为 None（后续在 generation_params 中落为 0）
        let no_prio: CompletionRequest = serde_json::from_str(r#"{"prompt":"hi"}"#).unwrap();
        assert_eq!(no_prio.priority, None);
    }

    #[test]
    fn test_chat_request_parses_priority_field() {
        let body: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"priority":3}"#)
                .unwrap();
        assert_eq!(body.priority, Some(3));
    }

    #[test]
    fn test_generation_params_passes_priority() {
        // greedy 参数必然合法，直接解构（ApiError 不派生 Debug/Display）
        let params =
            match generation_params(Some(8), Some(0.0), Some(1.0), Vec::new(), None, Some(7)) {
                Ok(p) => p,
                Err(_) => panic!("greedy 参数应通过校验"),
            };
        assert_eq!(params.priority, 7, "priority 应原样透传");

        // 缺省落为 0（调度器的默认优先级）
        let params = match generation_params(Some(8), Some(0.0), Some(1.0), Vec::new(), None, None)
        {
            Ok(p) => p,
            Err(_) => panic!("greedy 参数应通过校验"),
        };
        assert_eq!(params.priority, 0, "缺省 priority 应为 0");
    }

    // ===== PSRV-P0-001：取消所有权回归 =====
    //
    // 直接驱动 engine_loop / admit_submission / cancel_flagged：guard Drop
    // 置位 watch → 引擎循环下一步检出 → 取消 + 终态排出 + 资源释放。
    // 事件通道本身是同步屏障（Done 到达时 fail_sequence 已执行、槽位已
    // 释放），无需 sleep。token 37 = 'A'（SimpleTokenizer 词表内，
    // 每步产出非空文本片段）。

    use crate::test_utils::{create_test_config, test_params, ConstantTokenExecutor};
    use crate::{GPUExecutorTrait, Scheduler, SimpleTokenizer};
    use std::time::Duration;
    use tokio::time::timeout;

    const TEST_TOKEN: crate::types::TokenId = 37; // 'A'
    const EVENT_TIMEOUT: Duration = Duration::from_secs(10);

    fn loop_engine(config: &EngineConfig, executor: Box<dyn GPUExecutorTrait>) -> InferenceEngine {
        InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::without_special_tokens()),
            Scheduler::new(config.clone()),
            executor,
        )
        .expect("test engine must build")
    }

    /// 绕过 HTTP 层直接驱动引擎循环的测试夹具：
    /// 真实的 Submission/RequestGuard/Waiter 路径，submit 走 AppState::submit。
    struct LoopHarness {
        state: Arc<AppState>,
        shutdown_tx: watch::Sender<bool>,
        engine_metrics: Arc<SharedEngineMetrics>,
        join: tokio::task::JoinHandle<()>,
    }

    impl Drop for LoopHarness {
        fn drop(&mut self) {
            self.join.abort();
        }
    }

    fn spawn_loop(engine: InferenceEngine, config: EngineConfig) -> LoopHarness {
        let (submit_tx, submit_rx) = mpsc::channel(SUBMISSION_QUEUE_CAPACITY);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let engine_metrics = Arc::new(SharedEngineMetrics::default());
        let join = tokio::spawn(engine_loop(
            engine,
            submit_rx,
            engine_metrics.clone(),
            shutdown_rx,
            // 与 create_router_with_engine_and_shutdown 相同：循环自持保活
            // sender，通道只在显式 send(true) 时置位。
            shutdown_tx.clone(),
        ));
        let state = Arc::new(AppState {
            config,
            submit_tx,
            metrics: Arc::new(ServerMetrics::default()),
            engine_metrics: engine_metrics.clone(),
            response_counter: Arc::new(AtomicU64::new(1)),
            tokenizer: Arc::new(SimpleTokenizer::without_special_tokens()),
        });
        LoopHarness {
            state,
            shutdown_tx,
            engine_metrics,
            join,
        }
    }

    async fn submit_ok(
        state: &Arc<AppState>,
        prompt: &str,
        params: GenerationParams,
    ) -> (
        Admission,
        mpsc::UnboundedReceiver<RequestEvent>,
        RequestGuard,
    ) {
        match state.submit(prompt, params).await {
            Ok(v) => v,
            Err(_) => panic!("submit must succeed"),
        }
    }

    async fn next_event(
        events: &mut mpsc::UnboundedReceiver<RequestEvent>,
    ) -> Option<RequestEvent> {
        timeout(EVENT_TIMEOUT, events.recv())
            .await
            .expect("timed out waiting for request event")
    }

    /// 屏障：等到该请求的第一个 Chunk（说明已进入 decode）。
    async fn await_chunk(events: &mut mpsc::UnboundedReceiver<RequestEvent>) {
        match next_event(events).await {
            Some(RequestEvent::Chunk(..)) => {}
            Some(RequestEvent::Done(done)) => {
                panic!("expected chunk, got terminal (success={})", done.success)
            }
            None => panic!("event channel closed before any chunk"),
        }
    }

    /// 屏障：等到终态 Done。
    async fn await_done(events: &mut mpsc::UnboundedReceiver<RequestEvent>) -> CompletedRequest {
        loop {
            match next_event(events).await {
                Some(RequestEvent::Done(done)) => return done,
                Some(RequestEvent::Chunk(..)) => continue,
                None => panic!("event channel closed without Done"),
            }
        }
    }

    fn assert_cancelled(done: &CompletedRequest) {
        assert!(!done.success, "cancelled request must not be success");
        let msg = done.error.as_deref().unwrap_or("");
        assert!(
            msg.contains("cancelled"),
            "expected cancellation reason, got: {msg}"
        );
    }

    /// 有界轮询指标快照：引擎循环每步结束刷新 SharedEngineMetrics。
    async fn await_metric(counter: &AtomicU64, target: u64, what: &str) {
        for _ in 0..10_000 {
            if counter.load(Ordering::Relaxed) == target {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("timed out waiting for {what} == {target}");
    }

    /// guard Drop → decode 中请求被主动取消：终态恰好一次、原因含
    /// cancelled、序列槽位释放（Done 到达即已释放）。
    #[tokio::test]
    async fn cancel_decode_request_via_guard_drop() {
        let config = create_test_config();
        let harness = spawn_loop(
            loop_engine(
                &config,
                Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
            ),
            config,
        );
        let (admission, mut events, guard) =
            submit_ok(&harness.state, "hello", test_params(1000)).await;
        await_chunk(&mut events).await; // 已进入 decode

        drop(guard);
        let done = await_done(&mut events).await;
        assert_eq!(done.request_id, admission.request_id);
        assert_cancelled(&done);
        // 恰好一次终态：Done 之后 waiter 移除、发送端 drop，通道关闭。
        assert!(next_event(&mut events).await.is_none());
        await_metric(
            &harness.engine_metrics.active_sequences,
            0,
            "active_sequences",
        )
        .await;
        drop(harness);
    }

    /// guard Drop → pending 中请求被取消：不产生任何 token、终态正常排出。
    #[tokio::test]
    async fn cancel_pending_request_via_guard_drop() {
        // batch=1 + 长 decode 占用者：第二个请求永远停在 pending。
        let mut config = create_test_config();
        config.max_batch_size = 1;
        let harness = spawn_loop(
            loop_engine(
                &config,
                Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
            ),
            config,
        );
        let (_adm_a, mut events_a, guard_a) =
            submit_ok(&harness.state, "hog", test_params(1000)).await;
        await_chunk(&mut events_a).await; // A 已进入 decode，占住每步唯一 batch 槽

        let (_adm_b, mut events_b, guard_b) =
            submit_ok(&harness.state, "pending", test_params(8)).await;
        drop(guard_b);
        let done_b = await_done(&mut events_b).await;
        assert_cancelled(&done_b);
        assert!(
            done_b.output_tokens.is_empty(),
            "pending-stage cancel must not produce tokens"
        );

        // A 不受影响，仍在产出片段（无串扰）。
        await_chunk(&mut events_a).await;
        drop(guard_a);
        assert_cancelled(&await_done(&mut events_a).await);
        drop(harness);
    }

    /// 排队期 owner 消失 → 准入预检跳过：不分配调度/KV 资源。
    /// 同时覆盖 `has_changed()` 的 Err 分支（sender 全 drop）。
    #[tokio::test]
    async fn admission_precheck_skips_cancelled_submission() {
        let config = create_test_config();
        let mut engine = loop_engine(
            &config,
            Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
        );
        let mut waiters: HashMap<RequestId, Waiter> = HashMap::new();
        let (admit_tx, mut admit_rx) = oneshot::channel();
        let (events_tx, _events_rx) = mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        // 等价于 guard Drop：sender 消失即 owner 消失。
        drop(cancel_tx);

        admit_submission(
            &mut engine,
            Submission {
                prompt: "x".to_string(),
                params: test_params(4),
                admit: admit_tx,
                events: events_tx,
                cancel: cancel_rx,
            },
            &mut waiters,
        );

        assert!(
            waiters.is_empty(),
            "cancelled submission must not get a waiter"
        );
        assert!(
            !engine.has_pending_work(),
            "cancelled submission must not allocate a sequence"
        );
        // admit 回执被丢弃：sender 已 drop，对端收 Closed。
        assert!(matches!(
            admit_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
    }

    /// cancel_flagged 单元级：pending 态请求被取消、终态排出、waiter 移除。
    #[tokio::test]
    async fn cancel_flagged_dispatches_terminal_for_pending() {
        let config = create_test_config();
        let mut engine = loop_engine(
            &config,
            Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
        );
        let (request_id, _prompt_tokens) = engine
            .submit_request("hello", test_params(8))
            .expect("admission must succeed"); // Pending：从未 step
        let (events_tx, mut events_rx) = mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let guard = RequestGuard { cancel: cancel_tx };
        let mut waiters = HashMap::new();
        waiters.insert(
            request_id,
            Waiter {
                events: events_tx,
                cancel: cancel_rx,
                cancelled: false,
            },
        );
        let (_shutdown_keepalive, shutdown_rx) = watch::channel(false);

        drop(guard);
        cancel_flagged(&mut engine, &mut waiters, &shutdown_rx);

        assert!(waiters.is_empty(), "terminal dispatch must remove waiter");
        let done = await_done(&mut events_rx).await;
        assert_eq!(done.request_id, request_id);
        assert_cancelled(&done);
        assert_eq!(engine.get_metrics().active_sequences, 0);
        assert!(!engine.has_pending_work());
    }

    /// shutdown 广播：全部在途请求被取消，终态逐一排出。
    #[tokio::test]
    async fn shutdown_broadcast_cancels_all_inflight() {
        let config = create_test_config();
        let harness = spawn_loop(
            loop_engine(
                &config,
                Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
            ),
            config,
        );
        let (_adm_a, mut events_a, _guard_a) =
            submit_ok(&harness.state, "a", test_params(1000)).await;
        let (_adm_b, mut events_b, _guard_b) =
            submit_ok(&harness.state, "b", test_params(1000)).await;
        await_chunk(&mut events_a).await;
        await_chunk(&mut events_b).await;

        harness.shutdown_tx.send(true).expect("loop alive");

        assert_cancelled(&await_done(&mut events_a).await);
        assert_cancelled(&await_done(&mut events_b).await);
        await_metric(
            &harness.engine_metrics.active_sequences,
            0,
            "active_sequences",
        )
        .await;
        drop(harness);
    }

    /// 无跨请求串扰：取消 A 不影响 B 正常完成。
    #[tokio::test]
    async fn cancelling_one_request_does_not_affect_others() {
        let config = create_test_config();
        let harness = spawn_loop(
            loop_engine(
                &config,
                Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
            ),
            config,
        );
        let (_adm_a, mut events_a, guard_a) =
            submit_ok(&harness.state, "long", test_params(1000)).await;
        let (_adm_b, mut events_b, _guard_b) =
            submit_ok(&harness.state, "short", test_params(2)).await;
        await_chunk(&mut events_a).await;
        await_chunk(&mut events_b).await;

        drop(guard_a);
        assert_cancelled(&await_done(&mut events_a).await);

        let done_b = await_done(&mut events_b).await;
        assert!(done_b.success, "sibling request must complete normally");
        drop(harness);
    }

    /// n>1 部分准入失败：已准入候选被显式取消且槽位立即释放。
    /// Done 到达时 fail_sequence 已执行——新请求立即准入成功即证明。
    #[tokio::test]
    async fn partial_admission_failure_releases_admitted_candidate() {
        let mut config = create_test_config();
        config.max_num_seqs = 1;
        config.max_batch_size = 1; // 同步收紧：满足 validate 关系校验
        let harness = spawn_loop(
            loop_engine(
                &config,
                Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
            ),
            config,
        );
        // 候选 1 准入并占住唯一序列槽。
        let (_adm1, mut events1, guard1) =
            submit_ok(&harness.state, "first", test_params(1000)).await;
        await_chunk(&mut events1).await;

        // 候选 2 准入必然失败（并发上限）。
        match harness.state.submit("second", test_params(8)).await {
            Err(ApiError::Overloaded(_)) => {}
            _ => panic!("candidate 2 must fail admission with Overloaded"),
        }

        // respond_generation 的对应动作：drop 已收集 guard → 主动取消。
        drop(guard1);
        assert_cancelled(&await_done(&mut events1).await);

        // Done 到达 = 槽位已释放：新请求必须立即准入并正常完成。
        let (_adm3, mut events3, _guard3) =
            submit_ok(&harness.state, "third", test_params(2)).await;
        assert!(await_done(&mut events3).await.success);
        drop(harness);
    }

    /// 终态后 drop guard 是无害 no-op：waiter 已移除、id 不复用，
    /// 后续请求不受影响。
    #[tokio::test]
    async fn guard_drop_after_completion_is_harmless() {
        let config = create_test_config();
        let harness = spawn_loop(
            loop_engine(
                &config,
                Box::new(ConstantTokenExecutor { token: TEST_TOKEN }),
            ),
            config,
        );
        let (_adm, mut events, guard) = submit_ok(&harness.state, "done", test_params(1)).await;
        assert!(await_done(&mut events).await.success);

        drop(guard); // 终态后 drop：不得产生幻影失败
        let (_adm2, mut events2, _guard2) = submit_ok(&harness.state, "next", test_params(1)).await;
        assert!(await_done(&mut events2).await.success);
        await_metric(
            &harness.engine_metrics.failed_requests,
            0,
            "failed_requests",
        )
        .await;
        drop(harness);
    }
}
