//! 真实 HTTP/1.1 传输回归；执行器仅为受控 CPU 探针，不是 CUDA 回收证据。

use paged_serving::types::SeqId;
use paged_serving::{
    create_router_with_engine_and_shutdown, EngineConfig, EngineError, ExecutionBatch,
    ExecutionOutput, GPUExecutorTrait, InferenceEngine, Scheduler, ServingConfig, SimpleTokenizer,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::net::{Shutdown, SocketAddr};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{oneshot, watch};

const WAIT: Duration = Duration::from_secs(5);
const TEXT_TOKEN: u32 = 37; // SimpleTokenizer 的 'A'。
const SILENT_TOKEN: u32 = 0;

#[derive(Default)]
struct Probe {
    entered: u64,
    executed: BTreeSet<SeqId>,
    released: Vec<SeqId>,
}

struct ControlledExecutor {
    probe: Arc<Mutex<Probe>>,
    entered: watch::Sender<u64>,
    steps: mpsc::Receiver<u32>,
}

impl GPUExecutorTrait for ControlledExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        let entered = {
            let mut probe = self.probe.lock().unwrap();
            probe.entered += 1;
            probe.executed.extend(batch.seq_ids.iter().copied());
            probe.entered
        };
        self.entered.send_replace(entered);
        // 许可暂停不是计算工作；将 worker 交还 runtime，让 HTTP owner 可独立退出。
        let token =
            tokio::task::block_in_place(|| self.steps.recv_timeout(WAIT)).map_err(|error| {
                EngineError::KernelLaunchFailed(format!("test step control unavailable: {error}"))
            })?;
        Ok(ExecutionOutput {
            next_tokens: vec![token; batch.seq_ids.len()],
            seq_ids: batch.seq_ids.clone(),
            logprobs: Vec::new(),
        })
    }

    fn sequences_finished(&mut self, seq_ids: &[SeqId]) {
        self.probe
            .lock()
            .unwrap()
            .released
            .extend_from_slice(seq_ids);
    }
}

struct TcpServer {
    address: SocketAddr,
    client: reqwest::Client,
    probe: Arc<Mutex<Probe>>,
    entered: watch::Receiver<u64>,
    steps: Option<mpsc::SyncSender<u32>>,
    engine_shutdown: watch::Sender<bool>,
    http_shutdown: Option<oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
}

impl TcpServer {
    async fn start() -> Self {
        let config = EngineConfig {
            max_num_seqs: 4,
            max_batch_size: 4,
            max_num_blocks: 64,
            max_model_len: 1024,
            max_total_tokens: 512,
            serving: ServingConfig {
                model_name: "tcp-test".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let probe = Arc::new(Mutex::new(Probe::default()));
        let (entered_tx, entered) = watch::channel(0);
        // 4 个探针各最多 2 步；许可有界且不影响生产 mailbox。
        let (steps, step_rx) = mpsc::sync_channel(8);
        let engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::without_special_tokens()),
            Scheduler::new(config.clone()),
            Box::new(ControlledExecutor {
                probe: probe.clone(),
                entered: entered_tx,
                steps: step_rx,
            }),
        )
        .unwrap();
        let (router, engine_shutdown) =
            create_router_with_engine_and_shutdown(config, engine).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (http_shutdown, stop) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stop.await;
                })
                .await
        });
        Self {
            address,
            client: reqwest::Client::builder().timeout(WAIT).build().unwrap(),
            probe,
            entered,
            steps: Some(steps),
            engine_shutdown,
            http_shutdown: Some(http_shutdown),
            task: Some(task),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }

    async fn request(&self, streaming: bool) -> TcpStream {
        let body = json!({
            "model":"tcp-test", "prompt":"network", "max_tokens":500, "stream":streaming,
        })
        .to_string();
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            self.address, body.len(),
        );
        tokio::time::timeout(WAIT, async {
            let mut stream = TcpStream::connect(self.address).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            stream
        })
        .await
        .expect("opening HTTP request timed out")
    }

    fn advance(&self, token: u32, count: usize) {
        for _ in 0..count {
            self.steps.as_ref().unwrap().try_send(token).unwrap();
        }
    }

    async fn wait_entered(&mut self, target: u64) {
        tokio::time::timeout(WAIT, async {
            while *self.entered.borrow_and_update() < target {
                self.entered.changed().await.unwrap();
            }
        })
        .await
        .expect("executor did not reach controlled step");
    }

    async fn wait_metrics(&self, expected: &[(&str, u64)]) {
        tokio::time::timeout(WAIT, async {
            loop {
                let response = self.client.get(self.url("/metrics")).send().await.unwrap();
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                let text = response.text().await.unwrap();
                if expected
                    .iter()
                    .all(|(name, value)| text.lines().any(|line| line == format!("{name} {value}")))
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("metrics did not reach {expected:?}"));
    }

    fn assert_releases_once(&self, count: usize) {
        let probe = self.probe.lock().unwrap();
        assert_eq!(probe.released.len(), count);
        let unique: BTreeSet<_> = probe.released.iter().copied().collect();
        assert_eq!(
            unique.len(),
            count,
            "duplicate backend release notification"
        );
        assert_eq!(
            unique, probe.executed,
            "executed sequence missing its release"
        );
    }

    async fn assert_cancelled_and_reusable(&self) {
        self.wait_metrics(&[
            ("paged_engine_cancelled_requests", 1),
            ("paged_engine_completed_requests", 0),
            ("paged_engine_failed_requests", 0),
            ("paged_errors_total", 0),
            ("paged_engine_active_sequences", 0),
            ("paged_engine_kv_utilization", 0),
            ("paged_inflight_requests", 0),
        ])
        .await;
        self.assert_releases_once(1);
        // 原实例、原执行器；无 reset。最坏每请求单独 prefill/decode，共八步。
        self.advance(TEXT_TOKEN, 8);
        let requests = (0..4).map(|_| async {
            let response = self
                .client
                .post(self.url("/v1/completions"))
                .json(&json!({"model":"tcp-test", "prompt":"probe", "max_tokens":2}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["choices"][0]["text"], "AA");
            assert_eq!(body["usage"]["completion_tokens"], 2);
        });
        futures_util::future::join_all(requests).await;
        self.wait_metrics(&[
            ("paged_requests_total", 5),
            ("paged_engine_cancelled_requests", 1),
            ("paged_engine_completed_requests", 4),
            ("paged_engine_failed_requests", 0),
            ("paged_errors_total", 0),
            ("paged_engine_active_sequences", 0),
            ("paged_engine_kv_utilization", 0),
            ("paged_inflight_requests", 0),
        ])
        .await;
        self.assert_releases_once(5);
    }

    async fn stop(mut self) {
        self.engine_shutdown.send_replace(true);
        self.steps.take();
        self.http_shutdown.take().unwrap().send(()).unwrap();
        tokio::time::timeout(WAIT, self.task.take().unwrap())
            .await
            .expect("HTTP server shutdown timed out")
            .unwrap()
            .unwrap();
    }
}

impl Drop for TcpServer {
    fn drop(&mut self) {
        let _ = self.engine_shutdown.send(true);
        self.steps.take();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn read_until(stream: &mut TcpStream, marker: &[u8]) -> Vec<u8> {
    tokio::time::timeout(WAIT, async {
        let mut output = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            if output.windows(marker.len()).any(|bytes| bytes == marker) {
                return output;
            }
            let count = stream.read(&mut buffer).await.unwrap();
            assert!(count > 0, "HTTP connection ended before expected bytes");
            output.extend_from_slice(&buffer[..count]);
            assert!(output.len() <= 65536, "test response exceeded 64 KiB");
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "HTTP read timed out waiting for {:?}",
            String::from_utf8_lossy(marker)
        )
    })
}

fn disconnect(stream: TcpStream) {
    stream.into_std().unwrap().shutdown(Shutdown::Both).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tcp_disconnect_after_first_text_cancels_and_reuses_same_engine() {
    let mut server = TcpServer::start().await;
    let mut stream = server.request(true).await;
    let headers = read_until(&mut stream, b"\r\n\r\n").await;
    assert!(headers.starts_with(b"HTTP/1.1 200"));
    server.wait_entered(1).await;
    server.advance(TEXT_TOKEN, 1);
    read_until(&mut stream, br#""text":"A""#).await;
    server.wait_entered(2).await;
    server
        .wait_metrics(&[
            ("paged_inflight_requests", 1),
            ("paged_engine_active_sequences", 1),
        ])
        .await;
    assert!(server.probe.lock().unwrap().released.is_empty());
    disconnect(stream);
    // HTTP owner 已消失后，才允许同步在途步骤返回。
    server.wait_metrics(&[("paged_inflight_requests", 0)]).await;
    server.advance(SILENT_TOKEN, 1);
    server.assert_cancelled_and_reusable().await;
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tcp_disconnect_during_silent_decode_cancels_without_next_text() {
    let mut server = TcpServer::start().await;
    let mut stream = server.request(true).await;
    let headers = read_until(&mut stream, b"\r\n\r\n").await;
    assert!(headers.starts_with(b"HTTP/1.1 200"));
    assert!(!headers.windows(5).any(|bytes| bytes == b"data:"));
    server.wait_entered(1).await;
    server.advance(SILENT_TOKEN, 1);
    server.wait_entered(2).await;
    server
        .wait_metrics(&[
            ("paged_inflight_requests", 1),
            ("paged_engine_active_sequences", 1),
        ])
        .await;
    assert!(server.probe.lock().unwrap().released.is_empty());
    disconnect(stream);
    server.wait_metrics(&[("paged_inflight_requests", 0)]).await;
    server.advance(SILENT_TOKEN, 1);
    server.assert_cancelled_and_reusable().await;
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tcp_unary_disconnect_before_headers_cancels_and_reuses_same_engine() {
    let mut server = TcpServer::start().await;
    let stream = server.request(false).await;
    server.wait_entered(1).await;
    server.advance(SILENT_TOKEN, 1);
    server.wait_entered(2).await;
    server
        .wait_metrics(&[
            ("paged_inflight_requests", 1),
            ("paged_engine_active_sequences", 1),
        ])
        .await;
    disconnect(stream);
    server.wait_metrics(&[("paged_inflight_requests", 0)]).await;
    server.advance(SILENT_TOKEN, 1);
    server.assert_cancelled_and_reusable().await;
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tcp_shutdown_terminates_silent_sse_with_one_error_and_done() {
    let mut server = TcpServer::start().await;
    let mut stream = server.request(true).await;
    let headers = read_until(&mut stream, b"\r\n\r\n").await;
    assert!(headers.starts_with(b"HTTP/1.1 200"));
    server.wait_entered(1).await;
    server.advance(SILENT_TOKEN, 1);
    server.wait_entered(2).await;
    server.engine_shutdown.send(true).unwrap();
    server.advance(SILENT_TOKEN, 1);
    let mut body = Vec::new();
    tokio::time::timeout(WAIT, stream.read_to_end(&mut body))
        .await
        .unwrap()
        .unwrap();
    let body = String::from_utf8(body).unwrap();
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    let errors: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str(payload).unwrap())
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(
        errors[0]["error"]["message"],
        "request cancelled: server shutting down"
    );
    assert!(!body.contains("\"usage\""));
    server
        .wait_metrics(&[
            ("paged_engine_cancelled_requests", 1),
            ("paged_engine_failed_requests", 0),
            ("paged_engine_active_sequences", 0),
            ("paged_engine_kv_utilization", 0),
            ("paged_inflight_requests", 0),
            ("paged_errors_total", 0),
        ])
        .await;
    server.assert_releases_once(1);
    let ready = server
        .client
        .get(server.url("/readyz"))
        .send()
        .await
        .unwrap();
    assert_eq!(ready.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    server.stop().await;
}
