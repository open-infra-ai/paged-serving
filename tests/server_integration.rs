use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use paged_serving::{
    create_router, create_router_with_engine,
    test_utils::{AlwaysFailExecutor, ConstantTokenExecutor, SequenceExecutor},
    EngineConfig, EngineError, ExecutionBatch, ExecutionOutput, GPUExecutorTrait, InferenceEngine,
    Scheduler, ServingConfig, SimpleTokenizer, TokenizerTrait,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn create_test_config() -> EngineConfig {
    EngineConfig {
        max_total_tokens: 512,
        serving: ServingConfig {
            model_name: "test-model".to_string(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Assert the `usage` block is present, counts a non-empty prompt,
/// and reports `total_tokens == prompt_tokens + completion_tokens`.
fn assert_usage_consistent(json: &Value) {
    let prompt = json["usage"]["prompt_tokens"]
        .as_u64()
        .expect("usage.prompt_tokens must be an integer");
    let completion = json["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens must be an integer");
    let total = json["usage"]["total_tokens"]
        .as_u64()
        .expect("usage.total_tokens must be an integer");
    assert!(prompt > 0, "prompt_tokens should be non-zero");
    assert_eq!(
        total,
        prompt + completion,
        "total_tokens must equal prompt_tokens + completion_tokens"
    );
}

#[tokio::test]
async fn test_health_and_ready_endpoints() {
    let app = create_router(create_test_config()).unwrap();

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let ready = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_endpoint_exposes_prometheus_counters() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("paged_requests_total"));
    assert!(body.contains("paged_inflight_requests"));
    // 引擎指标快照
    assert!(body.contains("paged_engine_active_sequences"));
    assert!(body.contains("paged_engine_kv_utilization"));
    assert!(body.contains("paged_engine_completed_requests"));
    assert!(body.contains("paged_engine_failed_requests"));
    assert!(body.contains("paged_engine_tokens_generated_total"));
}

#[tokio::test]
async fn test_completions_returns_openai_shape() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["model"], "test-model");
    assert!(json["choices"][0]["text"].is_string());
    assert_usage_consistent(&json);
}

#[tokio::test]
async fn test_chat_completions_returns_assistant_message() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [
                            {"role": "user", "content": "say hi"}
                        ],
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert!(json["choices"][0]["message"]["content"].is_string());
    assert_usage_consistent(&json);
}

#[tokio::test]
async fn test_completions_stream_returns_done_event() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 2,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("data: [DONE]"));
    // 流必须以带 finish_reason 的终止 chunk 结束，且其 usage 字段完整。
    // 该请求 max_tokens=2 且 CPU 执行器不生成 EOS，终止原因应为 length。
    assert!(
        body.contains("\"finish_reason\":\"length\""),
        "stream must end with a length chunk, got: {body}"
    );
}

#[tokio::test]
async fn test_completions_rejects_non_greedy_sampling_params() {
    // PSERV-101：CPU 参考后端只有 greedy 一条路径；显式传入其他采样参数
    // 必须在准入阶段返回 400，而不是在执行时被静默忽略。
    let app = create_router(create_test_config()).unwrap();

    for body in [
        json!({"prompt": "hello", "max_tokens": 2, "temperature": 0.8}),
        json!({"prompt": "hello", "max_tokens": 2, "temperature": 0.0, "top_p": 0.9}),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "non-greedy params must be rejected: {body}"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("greedy"));
    }
}

#[tokio::test]
async fn test_streaming_chunks_concat_equals_non_streaming_text() {
    // PSERV-102 端到端性质：SSE 片段拼接 == 同一请求非流式返回的完整文本。
    // CPU 执行器固定种子 → 两个全新引擎对同一 prompt 产生相同 token 序列。
    let stream_app = create_router(create_test_config()).unwrap();
    let unary_app = create_router(create_test_config()).unwrap();

    let stream_response = stream_app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 6,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stream_response.status(), StatusCode::OK);

    let body = to_bytes(stream_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    let mut streamed_text = String::new();
    let mut saw_final = false;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let payload: Value = serde_json::from_str(data).unwrap();
        let choice = &payload["choices"][0];
        if choice["finish_reason"].is_null() {
            streamed_text.push_str(choice["text"].as_str().unwrap());
        } else {
            saw_final = true;
        }
    }
    assert!(
        saw_final,
        "stream must terminate with a finish_reason chunk"
    );

    let unary_response = unary_app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 6
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unary_response.status(), StatusCode::OK);
    let body = to_bytes(unary_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let unary_text = json["choices"][0]["text"].as_str().unwrap();

    assert_eq!(
        streamed_text, unary_text,
        "concatenated SSE chunks must equal the one-shot completion text"
    );
    assert!(!streamed_text.is_empty());
}

#[tokio::test]
async fn test_completions_rejects_invalid_sampling_params() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_completions_rejects_empty_prompt() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_rejects_empty_messages() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [],
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_rejects_invalid_role() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "robot", "content": "hi"}],
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_completions_rejects_unknown_model() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "no-such-model",
                        "prompt": "hello",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_unknown_route_returns_json_envelope() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_completions_rejects_malformed_json_with_envelope() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("this is not json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"]["message"].is_string(),
        "rejection must use the JSON error envelope"
    );
}

#[tokio::test]
async fn test_concurrent_requests_both_complete() {
    // 两个并发请求都必须完成：引擎循环应在两步之间接收第二个请求，
    // 而不是把它锁在第一个请求的完整生成之后。
    let app = create_router(create_test_config()).unwrap();

    let make_request = || {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "test-model",
                    "prompt": "hello world",
                    "max_tokens": 3
                })
                .to_string(),
            ))
            .unwrap()
    };

    let (first, second) = tokio::join!(
        app.clone().oneshot(make_request()),
        app.clone().oneshot(make_request()),
    );

    for response in [first.unwrap(), second.unwrap()] {
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(json["choices"][0]["text"].is_string());
        assert_usage_consistent(&json);
    }
}

#[tokio::test]
async fn test_completions_maps_execution_failure_to_500() {
    // 注入永远失败的执行器：生成失败必须映射为 500 + internal_error 信封，
    // 而不是被字符串化吞掉。
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(AlwaysFailExecutor),
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "internal_error");
    assert!(!json["error"]["message"].as_str().unwrap().is_empty());
}

/// 每次执行都阻塞一小段时间的执行器，用于制造可复现的过载窗口。
struct SlowExecutor;

impl GPUExecutorTrait for SlowExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        std::thread::sleep(std::time::Duration::from_millis(50));
        Ok(ExecutionOutput {
            next_tokens: vec![100; batch.seq_ids.len()],
            seq_ids: batch.seq_ids.clone(),
            logprobs: Vec::new(),
        })
    }
}

#[tokio::test]
async fn test_completions_returns_429_when_overloaded() {
    // max_num_seqs=1 + 慢执行器：第一个请求独占序列槽位期间，
    // 第二个请求必须收到 429（而非 500），并带 Retry-After。
    // max_batch_size 同步收紧为 1（单批上限不得超过并发上限，满足 validate 关系校验）。
    let config = EngineConfig {
        max_num_seqs: 1,
        max_batch_size: 1,
        serving: ServingConfig {
            model_name: "test-model".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(SlowExecutor),
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let make_request = || {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "test-model",
                    "prompt": "hello world",
                    "max_tokens": 5
                })
                .to_string(),
            ))
            .unwrap()
    };

    let (first, second) = tokio::join!(
        app.clone().oneshot(make_request()),
        app.clone().oneshot(make_request()),
    );

    let responses = [first.unwrap(), second.unwrap()];
    let statuses: Vec<StatusCode> = responses.iter().map(|r| r.status()).collect();
    assert!(
        statuses.contains(&StatusCode::OK),
        "one request should succeed, got {statuses:?}"
    );
    assert!(
        statuses.contains(&StatusCode::TOO_MANY_REQUESTS),
        "the other should be rejected as overloaded, got {statuses:?}"
    );

    for response in responses {
        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            assert!(
                response.headers().contains_key("retry-after"),
                "429 must carry Retry-After"
            );
        }
    }
}

/// 统计被执行的序列数并人为放慢每步，用于证明：
/// 客户端断开后引擎不再为它继续消耗算力。
struct CountingExecutor {
    executed_sequences: Arc<AtomicU64>,
    step_delay: Duration,
}

impl GPUExecutorTrait for CountingExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        std::thread::sleep(self.step_delay);
        self.executed_sequences
            .fetch_add(batch.num_sequences() as u64, Ordering::SeqCst);
        Ok(ExecutionOutput {
            next_tokens: vec![100; batch.num_sequences()],
            seq_ids: batch.seq_ids.clone(),
            logprobs: Vec::new(),
        })
    }
}

/// 拉取 /metrics 文本（oneshot 会消费 router，调用方需自行 clone 保留）。
async fn metrics_text(app: &axum::Router) -> String {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(body.to_vec()).unwrap()
}

/// 有界轮询 /metrics 直至出现目标行：指标快照由引擎循环每步刷新，
/// 观测到 `paged_engine_failed_requests 1` 即证明终态已排出、槽位已释放。
async fn wait_for_metric(app: &axum::Router, needle: &str) {
    for _ in 0..300 {
        if metrics_text(app).await.contains(needle) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for metric: {needle}");
}

#[tokio::test]
async fn test_client_disconnect_cancels_generation() {
    let executed = Arc::new(AtomicU64::new(0));
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::without_special_tokens()),
        Scheduler::new(config.clone()),
        Box::new(CountingExecutor {
            executed_sequences: executed.clone(),
            step_delay: Duration::from_millis(5),
        }),
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();
    let metrics_app = app.clone();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "disconnect me",
                        "max_tokens": 500,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 读到第一个 token 片段后立刻丢弃 body，模拟客户端断连
    let mut body = response.into_body();
    let mut saw_chunk = false;
    while let Some(Ok(frame)) = body.frame().await {
        if let Ok(data) = frame.into_data() {
            if data.windows(15).any(|w| w == b"text_completion") {
                saw_chunk = true;
                break;
            }
        }
    }
    assert!(saw_chunk, "stream should produce at least one token chunk");
    drop(body);

    // 确定性屏障：取消终态被排出（取消计入 failed_requests）。
    wait_for_metric(&metrics_app, "paged_engine_failed_requests 1").await;

    // 断连后执行计数必须停下来：间隔采样两次，不再增长。
    // （若无取消机制，5ms/步的引擎会在两次采样间推进上百步。）
    tokio::time::sleep(Duration::from_millis(300)).await;
    let first_sample = executed.load(Ordering::SeqCst);
    assert!(
        first_sample < 500,
        "request should be cancelled long before max_tokens"
    );
    tokio::time::sleep(Duration::from_millis(500)).await;
    let second_sample = executed.load(Ordering::SeqCst);
    assert_eq!(
        first_sample, second_sample,
        "engine must stop executing a request after its client disconnects"
    );
}

/// PSRV-P0-001：非流式请求 future 被 abort（client 断连/超时取消）时，
/// RequestGuard Drop 必须主动取消在途请求并释放资源。
#[tokio::test]
async fn test_unary_future_abort_cancels_request() {
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::without_special_tokens()),
        Scheduler::new(config.clone()),
        Box::new(SlowExecutor),
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();
    let metrics_app = app.clone();
    let req_app = app.clone();

    // unary 请求整体在一个 spawn 的任务里跑：abort 即模拟 handler future
    // 被丢弃（连接断开）。
    let task = tokio::spawn(async move {
        req_app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "prompt": "abort me",
                            "max_tokens": 500
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
    });

    // 屏障：请求真正进入执行（prefill+decode 序列存在）。
    wait_for_metric(&metrics_app, "paged_engine_active_sequences 1").await;
    task.abort();

    // guard Drop → 主动取消 → 终态排出 → 序列槽位释放。
    wait_for_metric(&metrics_app, "paged_engine_failed_requests 1").await;
    wait_for_metric(&metrics_app, "paged_engine_active_sequences 0").await;
}

/// PSRV-P0-001：n>1 流式请求在部分准入失败时，已准入候选必须被显式取消
/// 并释放序列槽位（不依赖下一次 chunk 发送失败的偶然时点）。
#[tokio::test]
async fn test_streaming_n2_partial_admission_cancels_admitted_candidate() {
    // 并发上限 1：n=2 的第二个候选必然准入失败。
    let config = EngineConfig {
        max_num_seqs: 1,
        max_batch_size: 1, // 同步收紧：单批上限不得超过并发上限
        serving: ServingConfig {
            model_name: "test-model".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::without_special_tokens()),
        Scheduler::new(config.clone()),
        Box::new(ConstantTokenExecutor { token: 37 }), // 'A'：非空片段
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();
    let metrics_app = app.clone();

    // n=2 stream：候选 1 准入占住唯一槽位，候选 2 必然 429。
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "two candidates",
                        "max_tokens": 500,
                        "n": 2,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    // 屏障：候选 1 的取消终态已排出（槽位此刻已释放）。
    wait_for_metric(&metrics_app, "paged_engine_failed_requests 1").await;

    // 行为级证明：新请求立即准入成功（若槽位未释放会得到 429）。
    let follow_up = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "after",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        follow_up.status(),
        StatusCode::OK,
        "slot from the cancelled candidate must be released"
    );
}

#[tokio::test]
async fn test_completions_reports_length_finish_reason_when_truncated() {
    // PSERV-103：CPU 执行器生成非 EOS token，达到 max_tokens 截断时，
    // 非流式响应必须报告 finish_reason="length"，而不是始终 "stop"。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["choices"][0]["finish_reason"], "length",
        "truncated completion must report length"
    );
    assert_usage_consistent(&json);
}

#[tokio::test]
async fn test_completions_reports_stop_finish_reason_on_eos() {
    // PSERV-103：注入恒生成 EOS 的执行器，请求在首个 token 后自然停止，
    // 非流式响应必须报告 finish_reason="stop"。
    let config = create_test_config();
    let eos = SimpleTokenizer::new().eos_token_id();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(ConstantTokenExecutor { token: eos }), // EOS 动态取自 tokenizer
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["choices"][0]["finish_reason"], "stop",
        "EOS-terminated completion must report stop"
    );
}

#[tokio::test]
async fn test_streaming_reports_stop_finish_reason_on_eos() {
    // PSERV-103：流式场景下 EOS 自然停止，终止 chunk 的 finish_reason 必须是 "stop"。
    let config = create_test_config();
    let eos = SimpleTokenizer::new().eos_token_id();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(ConstantTokenExecutor { token: eos }), // EOS 动态取自 tokenizer
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 8,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    let mut final_reason = None;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let payload: Value = serde_json::from_str(data).unwrap();
        let reason = &payload["choices"][0]["finish_reason"];
        if !reason.is_null() {
            final_reason = Some(reason.as_str().unwrap().to_string());
        }
    }
    assert_eq!(
        final_reason.as_deref(),
        Some("stop"),
        "EOS-terminated stream must end with a stop chunk, got: {final_reason:?}"
    );
}

#[tokio::test]
async fn test_completions_rejects_unsupported_params() {
    // PSERV-104：CPU 后端未实现的参数在准入阶段返回 400 + invalid_request_error，
    // 消息带参数名——而不是被 serde 静默忽略。
    let app = create_router(create_test_config()).unwrap();

    let cases = [
        // stop（PSERV-105）、n（PSERV-106）、logprobs（PSERV-107）已支持，不再在此拒绝
        (
            "seed",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"seed":42}),
        ),
        (
            "echo",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"echo":true}),
        ),
        (
            "suffix",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"suffix":"!"}),
        ),
        (
            "frequency_penalty",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"frequency_penalty":0.5}),
        ),
        (
            "presence_penalty",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"presence_penalty":0.5}),
        ),
        (
            "best_of",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"best_of":2}),
        ),
        (
            "stream_options",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"stream_options":{"include_usage":true}}),
        ),
        (
            "tools",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"tools":[{"type":"function"}]}),
        ),
        (
            "tool_choice",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"tool_choice":"auto"}),
        ),
        (
            "response_format",
            json!({"model":"test-model","prompt":"hi","max_tokens":2,"response_format":{"type":"json_object"}}),
        ),
    ];

    for (param, body) in cases {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "param {param} must be rejected: {body}"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(
            json["error"]["message"].as_str().unwrap().contains(param),
            "message must name param {param}: {}",
            json["error"]["message"]
        );
    }
}

#[tokio::test]
async fn test_chat_completions_rejects_unsupported_params() {
    // PSERV-104：chat/completions 同样在准入阶段拒绝未支持参数。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2,
                        "tools": [{"type": "function", "function": {"name": "f"}}],
                        "response_format": {"type": "json_object"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(json["error"]["message"].as_str().unwrap().contains("tools"));
}

#[tokio::test]
async fn test_completions_accepts_unsupported_params_at_defaults() {
    // PSERV-104：未支持参数取默认值时（frequency_penalty=0、n=1、stop=null/[]、
    // echo=false、logprobs=false/0、best_of=1）语义无害，仍应放行；
    // 完全未知的字段（如 user）也应忽略而非拒绝。
    let app = create_router(create_test_config()).unwrap();

    let cases = [
        json!({"model":"test-model","prompt":"hi","max_tokens":2}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"frequency_penalty":0.0}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"presence_penalty":0.0}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"n":1}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"echo":false}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"logprobs":false}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"logprobs":0}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"stop":[]}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"best_of":1}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"user":"client-42"}),
    ];

    for body in cases {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::OK,
            "default values must pass: {body}"
        );
    }
}

#[tokio::test]
async fn test_completions_honors_stop_sequences() {
    // PSERV-105：stop 序列命中时，非流式响应移除 stop 序列并报告
    // finish_reason="stop"（而非 length）。
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(SequenceExecutor::new(vec![37, 37, 38])), // A, A, B
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 10,
                        "stop": "AB"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    // 生成 "AAB" 命中 "AB"@1 → 保留前缀 "A"
    assert_eq!(json["choices"][0]["text"], "A");
    assert_eq!(
        json["choices"][0]["finish_reason"], "stop",
        "stop sequence hit must report stop"
    );
    assert_usage_consistent(&json);
}

#[tokio::test]
async fn test_streaming_honors_stop_sequences() {
    // PSERV-105：流式下 stop 命中后不再推送新内容，终止 chunk 报 "stop"；
    // 拼接文本 == 非流式文本（单字符 stop 序列保证无跨 token 前缀残留）。
    let config = create_test_config();
    let make_app = || {
        let engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            Scheduler::new(config.clone()),
            Box::new(SequenceExecutor::new(vec![37, 37, 38])), // A, A, B
        )
        .unwrap();
        create_router_with_engine(config.clone(), engine).unwrap()
    };
    let stream_app = make_app();
    let unary_app = make_app();

    let stream_response = stream_app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 10,
                        "stop": "B",
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stream_response.status(), StatusCode::OK);

    let body = to_bytes(stream_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    let mut streamed_text = String::new();
    let mut final_reason = None;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let payload: Value = serde_json::from_str(data).unwrap();
        let choice = &payload["choices"][0];
        if choice["finish_reason"].is_null() {
            streamed_text.push_str(choice["text"].as_str().unwrap());
        } else {
            final_reason = Some(choice["finish_reason"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(
        final_reason.as_deref(),
        Some("stop"),
        "stop-terminated stream must end with a stop chunk, got: {final_reason:?}"
    );

    let unary_response = unary_app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 10,
                        "stop": "B"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let body = to_bytes(unary_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let unary_text = json["choices"][0]["text"].as_str().unwrap();

    // 生成 A, A, B，stop="B" 命中 → 移除 B，保留 "AA"
    assert_eq!(streamed_text, "AA");
    assert_eq!(streamed_text, unary_text);
}

#[tokio::test]
async fn test_completions_rejects_too_many_stop_sequences() {
    // PSERV-105：stop 超过 4 个（OpenAI 限制）在准入阶段返回 400。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 2,
                        "stop": ["a", "b", "c", "d", "e"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_chat_completions_honors_stop_sequences() {
    // PSERV-105：chat/completions 同样支持 stop 序列。
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(SequenceExecutor::new(vec![37, 37, 38])), // A, A, B
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 10,
                        "stop": "AB"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "A");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn test_completions_returns_n_candidates() {
    // PSERV-106：非流式 n>1 返回 n 个 choices；usage.completion_tokens 为各
    // 候选之和（prompt 只计一次）。greedy 后端所有候选内容相同。
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(ConstantTokenExecutor { token: 37 }), // 'A'
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 3,
                        "n": 3
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3, "n=3 must yield 3 choices");
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], i as u64);
        assert_eq!(choice["text"], "AAA");
        assert_eq!(choice["finish_reason"], "length");
    }
    assert_eq!(
        json["usage"]["completion_tokens"], 9,
        "completion_tokens must sum across candidates (3 x 3 tokens)"
    );
    assert_usage_consistent(&json);
}

#[tokio::test]
async fn test_streaming_merges_n_candidates() {
    // PSERV-106：流式 n=2 fan-in 为单一 SSE；中间 chunk 含 2 个 choice，
    // 终止 chunk 带 2 个 finish_reason 与聚合 usage。
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(ConstantTokenExecutor { token: 37 }), // 'A'
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 2,
                        "n": 2,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    let mut per_index = vec![String::new(); 2];
    let mut final_reasons: Vec<String> = Vec::new();
    let mut saw_usage = false;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let payload: Value = serde_json::from_str(data).unwrap();
        let choices = payload["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2, "each chunk must carry n=2 choices");
        for choice in choices {
            let index = choice["index"].as_u64().unwrap() as usize;
            if choice["finish_reason"].is_null() {
                per_index[index].push_str(choice["text"].as_str().unwrap());
            } else {
                final_reasons.push(choice["finish_reason"].as_str().unwrap().to_string());
            }
        }
        if !payload["usage"].is_null() {
            saw_usage = true;
        }
    }

    assert_eq!(per_index, vec!["AA".to_string(), "AA".to_string()]);
    assert_eq!(
        final_reasons,
        vec!["length".to_string(), "length".to_string()]
    );
    assert!(saw_usage, "final chunk must carry aggregated usage");
}

#[tokio::test]
async fn test_chat_completions_returns_n_candidates() {
    // PSERV-106：chat/completions 同样支持 n。
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(ConstantTokenExecutor { token: 37 }), // 'A'
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2,
                        "n": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["message"]["content"], "AA");
    assert_eq!(choices[1]["message"]["content"], "AA");
}

#[tokio::test]
async fn test_completions_rejects_invalid_n() {
    // PSERV-106：n=0 与超上限（16）在准入阶段返回 400。
    let app = create_router(create_test_config()).unwrap();

    for body in [
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"n":0}),
        json!({"model":"test-model","prompt":"hi","max_tokens":2,"n":17}),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "invalid n must be rejected: {body}"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }
}

#[tokio::test]
async fn test_completions_returns_logprobs() {
    // PSERV-107：logprobs=true 时每个 choice 携带 logprobs 块
    // （tokens / token_logprobs / top_logprobs / text_offset）。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 3,
                        "logprobs": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let lp = &json["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"].as_array().unwrap().len(), 3);
    assert_eq!(lp["token_logprobs"].as_array().unwrap().len(), 3);
    assert_eq!(lp["top_logprobs"].as_array().unwrap().len(), 3);
    assert_eq!(lp["text_offset"].as_array().unwrap().len(), 3);
    // 每个 token_logprob 应为有限数值
    for v in lp["token_logprobs"].as_array().unwrap() {
        assert!(v.as_f64().unwrap().is_finite());
    }
    // top_logprobs 每项是 {token文本: logprob} 的 map，且包含选中 token
    let first_top = lp["top_logprobs"][0].as_object().unwrap();
    assert!(!first_top.is_empty());
    assert_eq!(lp["text_offset"][0], 0);
}

#[tokio::test]
async fn test_completions_omits_logprobs_by_default() {
    // PSERV-107：未请求 logprobs 时响应不携带该字段（而非空对象）。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"test-model","prompt":"hi","max_tokens":2}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["choices"][0].get("logprobs").is_none(),
        "logprobs must be omitted when not requested"
    );
}

#[tokio::test]
async fn test_streaming_carries_logprobs() {
    // PSERV-107：流式每个携带内容的 chunk 同时携带该 token 的 logprobs。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 2,
                        "logprobs": true,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    let mut saw_content_logprobs = false;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let payload: Value = serde_json::from_str(data).unwrap();
        let choice = &payload["choices"][0];
        if choice["finish_reason"].is_null() && !choice["text"].as_str().unwrap().is_empty() {
            // 携带内容的 chunk 必须带 logprobs（单 token 块）
            let lp = &choice["logprobs"];
            assert_eq!(lp["tokens"].as_array().unwrap().len(), 1);
            assert_eq!(lp["token_logprobs"].as_array().unwrap().len(), 1);
            saw_content_logprobs = true;
        }
    }
    assert!(
        saw_content_logprobs,
        "at least one content chunk must carry logprobs"
    );
}

#[tokio::test]
async fn test_chat_completions_returns_logprobs() {
    // PSERV-107：chat/completions 同样支持 logprobs。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2,
                        "logprobs": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let lp = &json["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"].as_array().unwrap().len(), 2);
    assert_eq!(lp["top_logprobs"].as_array().unwrap().len(), 2);
    // logprobs=2：top_logprobs 每项含至多 2 个候选
    assert!(lp["top_logprobs"][0].as_object().unwrap().len() <= 2);
}

#[tokio::test]
async fn test_completions_rejects_logprobs_above_limit() {
    // PSERV-107：logprobs > 5（OpenAI 上限）在准入阶段返回 400。
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hi",
                        "max_tokens": 2,
                        "logprobs": 6
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_oversized_prompt_gets_error_and_server_stays_responsive() {
    // B12 服务层回归：所需 KV 块数超过内存水位线的长 prompt 必须及时得到
    // 明确错误响应（而非 engine_loop 无限空转导致请求挂起、服务阻塞），
    // 且后续请求仍能正常服务。
    // 1000 字符 → 1002 tokens（BOS/EOS）→ ceil(1002/16)=63 块
    // > floor(64*0.9)=57 水位线，永久不可调度。
    let config = EngineConfig {
        max_num_blocks: 64,
        max_total_tokens: 1024,
        max_model_len: 1024,
        serving: ServingConfig {
            model_name: "test-model".to_string(),
            ..Default::default()
        },
        ..create_test_config()
    };
    let app = create_router(config).unwrap();

    let big_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "prompt": "a".repeat(1000),
                "max_tokens": 4
            })
            .to_string(),
        ))
        .unwrap();

    // 必须及时返回错误而非挂起：用超时把"回归死循环"暴露为测试失败
    let response = tokio::time::timeout(Duration::from_secs(5), app.clone().oneshot(big_request))
        .await
        .expect("超预算长 prompt 请求不应挂起（engine_loop 死循环回归）")
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "永久不可调度的请求应返回 500"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("watermark"),
        "错误应说明水位线限制，实际: {}",
        json["error"]["message"]
    );

    // 服务仍可用：后续普通请求正常完成
    let ok_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "prompt": "hi",
                "max_tokens": 4
            })
            .to_string(),
        ))
        .unwrap();
    let response = app.oneshot(ok_request).await.expect("服务应保持可用");
    assert_eq!(response.status(), StatusCode::OK);
}
