use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PROMPTS: [&str; 6] = ["usage", "unknown", "429", "error", "no_done", "invalid"];

struct TestOutput(PathBuf);

impl TestOutput {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "paged-serving-loadgen-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // 必须成功创建新目录，不能复用/清理一个不属于本测试的已有目录。
        std::fs::create_dir(&path).unwrap();
        let dataset = PROMPTS
            .iter()
            .enumerate()
            .map(|(index, prompt)| json!({"prompt":prompt, "prompt_tokens":index + 1}).to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path.join("dataset.jsonl"), dataset).unwrap();
        Self(path)
    }
}

impl Drop for TestOutput {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct TestServer {
    base_url: String,
    received: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl TestServer {
    async fn start() -> Self {
        let received = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route("/v1/completions", post(respond))
            .with_state(received.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            base_url,
            received,
            task,
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn respond(State(received): State<Arc<AtomicUsize>>, Json(body): Json<Value>) -> Response {
    received.fetch_add(1, Ordering::Relaxed);
    assert_eq!(body["stream"], true);
    assert_eq!(body["temperature"].as_f64(), Some(0.0));
    assert_eq!(body["model"], "cli-test-model");
    let chunk = r#"{"choices":[{"text":"A","finish_reason":null}]}"#;
    let done = "[DONE]";
    let events = match body["prompt"].as_str().unwrap() {
        "usage" => vec![
            chunk,
            r#"{"choices":[{"text":"B","finish_reason":"length"}],"usage":{"completion_tokens":2}}"#,
            done,
        ],
        "unknown" => vec![chunk, done],
        "429" => return (StatusCode::TOO_MANY_REQUESTS, "admission rejected").into_response(),
        "error" => vec![r#"{"error":{"message":"injected backend failure"}}"#, done],
        "no_done" => vec![chunk],
        "invalid" => vec!["{"],
        "timeout" => {
            let partial = format!(
                "data: {chunk}\n\ndata: {{\"choices\":[{{\"text\":\"B\",\"finish_reason\":\"length\"}}],\"usage\":{{\"completion_tokens\":2}}}}\n\n"
            );
            let stream =
                futures_util::stream::once(async { Ok::<_, std::convert::Infallible>(partial) })
                    .chain(futures_util::stream::pending());
            return (
                [("content-type", "text/event-stream")],
                axum::body::Body::from_stream(stream),
            )
                .into_response();
        }
        prompt => panic!("unexpected test prompt: {prompt}"),
    };
    let body = events
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
    ([("content-type", "text/event-stream")], body).into_response()
}

fn bounded_output(mut command: Command) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if started.elapsed() > Duration::from_secs(10) {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "loadgen CLI timed out: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn command(
    server: &TestServer,
    output: &TestOutput,
    mode: &str,
    warmup: u64,
    seed: u64,
) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_loadgen"));
    command
        .args([
            "--base-url",
            &server.base_url,
            "--mode",
            mode,
            "--concurrency",
            "3",
            "--rate",
            "80",
            "--requests",
            "6",
            "--timeout-secs",
            "1",
            "--max-tokens",
            "4",
            "--engine",
            "cli-test",
            "--model",
            "cli-test-model",
            "--seed",
            &seed.to_string(),
            "--warmup-secs",
            &warmup.to_string(),
            "--dataset",
        ])
        .arg(output.0.join("dataset.jsonl"))
        .arg("--out")
        .arg(output.0.join("nested/per_request.jsonl"));
    command
}

async fn run_cli(
    server: &TestServer,
    output: &TestOutput,
    mode: &str,
    warmup: u64,
    seed: u64,
) -> (Vec<Value>, Value) {
    let command = command(server, output, mode, warmup, seed);
    let result = tokio::task::spawn_blocking(move || bounded_output(command))
        .await
        .unwrap();
    assert!(
        result.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    artifacts(output, "nested/summary.json")
}

fn artifacts(output: &TestOutput, summary_path: &str) -> (Vec<Value>, Value) {
    let records = std::fs::read_to_string(output.0.join("nested/per_request.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let summary =
        serde_json::from_slice(&std::fs::read(output.0.join(summary_path)).unwrap()).unwrap();
    (records, summary)
}

fn assert_artifacts(records: &[Value], summary: &Value) {
    assert_eq!(records.len(), 6);
    let mut ids = BTreeSet::new();
    let mut errors = std::collections::BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["measured_index"], index);
        assert_eq!(
            record["prompt_tokens_meta"],
            index + 1,
            "prompt order must not depend on warmup count"
        );
        assert!(ids.insert(record["request_id"].as_u64().unwrap()));
        assert!(record["dispatch_offset_ms"].as_f64().unwrap() >= 0.0);
        if !record["ok"].as_bool().unwrap() {
            assert!(!record["error_detail"].as_str().unwrap().is_empty());
            *errors
                .entry(record["error_class"].as_str().unwrap().to_string())
                .or_insert(0usize) += 1;
        }
    }
    assert_eq!(summary["schema_version"], 1);
    assert_eq!(summary["config"]["engine"], "cli-test");
    assert_eq!(summary["config"]["requests"], 6);
    assert_eq!(summary["requests"]["total"], 6);
    assert_eq!(summary["requests"]["success"], 2);
    assert_eq!(summary["requests"]["failed"], 4);
    assert_eq!(summary["errors"], json!(errors));
    assert_eq!(summary["completion_tokens"]["known_requests"], 1);
    assert_eq!(summary["completion_tokens"]["successful_requests"], 2);
    assert_eq!(
        summary["completion_tokens"]["coverage_pct"].as_f64(),
        Some(50.0)
    );
    assert_eq!(summary["completion_tokens"]["total"], 2);
    assert!(summary["throughput"]["output_tokens_per_second"].is_null());
    assert!(summary["itl_ms"].is_null());
    assert_eq!(summary["ttft_ms"]["samples"], 2);
    assert!(
        records[1]["completion_tokens"].is_null(),
        "chunks are not tokens"
    );
    assert_eq!(records[2]["error_class"], "http_429");
    assert_eq!(records[3]["error_class"], "stream_error");
    assert_eq!(records[4]["error_class"], "no_done");
    assert_eq!(records[5]["error_class"], "protocol_error");
}

#[tokio::test]
async fn closed_cli_excludes_real_warmup_and_writes_exact_artifacts() {
    let server = TestServer::start().await;
    let output = TestOutput::new();
    let (records, summary) = run_cli(&server, &output, "closed", 1, 42).await;
    assert_artifacts(&records, &summary);
    assert!(server.received.load(Ordering::Relaxed) > 6);
    assert!(records[0]["request_id"].as_u64().unwrap() > 0);
    assert_eq!(summary["config"]["concurrency"], 3);
    assert!(summary["config"]["arrival_seed"].is_null());
    assert!(summary["config"]["arrival_schedule"].is_null());
    assert!(records
        .iter()
        .all(|record| record["scheduled_arrival_ms"].is_null()));
}

#[tokio::test]
async fn poisson_cli_replays_planned_load_independently_of_warmup() {
    let server = TestServer::start().await;
    let mut schedules = Vec::new();
    for (warmup, seed) in [(1, 42), (0, 42), (0, 77)] {
        let output = TestOutput::new();
        let (records, summary) = run_cli(&server, &output, "poisson", warmup, seed).await;
        assert_artifacts(&records, &summary);
        assert_eq!(summary["config"]["arrival_seed"], seed);
        assert_eq!(
            summary["config"]["arrival_schedule"],
            "absolute_deadline_seed_reset"
        );
        let schedule: Vec<f64> = records
            .iter()
            .map(|record| {
                let planned = record["scheduled_arrival_ms"].as_f64().unwrap();
                assert!(record["dispatch_offset_ms"].as_f64().unwrap() >= planned);
                planned
            })
            .collect();
        assert!(schedule.windows(2).all(|pair| pair[1] >= pair[0]));
        if warmup > 0 {
            assert!(records[0]["request_id"].as_u64().unwrap() > 0);
        }
        schedules.push(schedule);
    }
    assert_eq!(
        schedules[0], schedules[1],
        "same seed must ignore warmup RNG consumption"
    );
    assert_ne!(schedules[1], schedules[2]);
}

#[tokio::test]
async fn invalid_cli_arguments_fail_without_result_files() {
    let server = TestServer::start().await;
    let output = TestOutput::new();
    let mut command = command(&server, &output, "invalid", 0, 42);
    command
        .arg("--summary-out")
        .arg(output.0.join("custom/summary.json"));
    let result = tokio::task::spawn_blocking(move || bounded_output(command))
        .await
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(!output.0.join("nested/per_request.jsonl").exists());
    assert!(!output.0.join("custom/summary.json").exists());
    assert_eq!(server.received.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn complete_usage_cli_reports_wall_time_throughput_and_custom_summary() {
    let server = TestServer::start().await;
    let output = TestOutput::new();
    std::fs::write(
        output.0.join("dataset.jsonl"),
        r#"{"prompt":"usage","prompt_tokens":4}"#,
    )
    .unwrap();
    let mut command = command(&server, &output, "closed", 0, 42);
    command
        .arg("--summary-out")
        .arg(output.0.join("custom/summary.json"));
    let result = tokio::task::spawn_blocking(move || bounded_output(command))
        .await
        .unwrap();
    assert!(
        result.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let (records, summary) = artifacts(&output, "custom/summary.json");
    assert_eq!(records.len(), 6);
    assert!(records.iter().all(|record| record["ok"] == true));
    assert_eq!(summary["requests"]["success"], 6);
    assert_eq!(summary["completion_tokens"]["total"], 12);
    assert_eq!(
        summary["completion_tokens"]["coverage_pct"].as_f64(),
        Some(100.0)
    );
    assert_eq!(summary["completion_tokens"]["source_counts"]["usage"], 6);
    let wall = summary["measurement_wall_secs"].as_f64().unwrap();
    let throughput = summary["throughput"]["output_tokens_per_second"]
        .as_f64()
        .unwrap();
    assert!(wall > 0.0);
    assert!((throughput - 12.0 / wall).abs() < 1e-8 * throughput);
    assert!(!output.0.join("nested/summary.json").exists());
    assert_eq!(server.received.load(Ordering::Relaxed), 6);
}

#[tokio::test]
async fn body_timeouts_cli_preserves_raw_output_and_excludes_failed_metrics() {
    let server = TestServer::start().await;
    let output = TestOutput::new();
    std::fs::write(
        output.0.join("dataset.jsonl"),
        r#"{"prompt":"timeout","prompt_tokens":4}"#,
    )
    .unwrap();
    let (records, summary) = run_cli(&server, &output, "closed", 0, 42).await;
    assert_eq!(records.len(), 6);
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["measured_index"], index);
        assert_eq!(record["ok"], false);
        assert_eq!(record["error_class"], "timeout");
        assert!(!record["error_detail"].as_str().unwrap().is_empty());
        assert_eq!(record["chunks"], 2);
        assert!(record["ttft_ms"].as_f64().is_some());
        assert_eq!(
            record["inter_chunk_latency_ms"].as_array().unwrap().len(),
            1
        );
        assert_eq!(record["completion_tokens"], 2);
        assert_eq!(record["tokens_source"], "usage");
        assert_eq!(record["finish_reason"], "length");
    }
    assert_eq!(summary["requests"]["total"], 6);
    assert_eq!(summary["requests"]["success"], 0);
    assert_eq!(summary["requests"]["failed"], 6);
    assert_eq!(summary["errors"], json!({"timeout": 6}));
    for metric in ["ttft_ms", "inter_chunk_latency_ms", "tpot_ms"] {
        assert_eq!(summary[metric]["samples"], 0);
        assert!(summary[metric]["p50"].is_null());
        assert!(summary[metric]["p95"].is_null());
        assert!(summary[metric]["p99"].is_null());
    }
    assert_eq!(summary["completion_tokens"]["known_requests"], 0);
    assert_eq!(summary["completion_tokens"]["total"], 0);
    assert_eq!(summary["throughput"]["successful_requests_per_second"], 0.0);
    assert!(summary["throughput"]["output_tokens_per_second"].is_null());
    assert_eq!(server.received.load(Ordering::Relaxed), 6);
}
