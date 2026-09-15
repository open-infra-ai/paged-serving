//! Serving 压测客户端（loadgen）：对 OpenAI 兼容 `/v1/completions`（SSE 流式）
//! 端点做可复现负载实验。
//!
//! 同一二进制零改动覆盖三个后端：paged-serving / llama-server / vLLM，
//! 保证横向可比（口径定义见 `benchmarks/serving/methodology.md`）。
//!
//! # 负载模型
//! - `--mode closed`：闭环饱和——固定 `--concurrency` 个并发槽，一个请求完成
//!   立即补发下一个。测最大吞吐与尾延迟。
//! - `--mode poisson`：开环泊松——按 `--rate` req/s 的指数间隔到达，请求独立
//!   并发。测 SLO 曲线（TTFT p95 vs 到达率）。
//!
//! # 指标口径（客户端侧）
//! - TTFT：请求开始发送 → 首个**非空文本** SSE chunk 的墙钟。空文本 chunk
//!   （如部分服务器的前导帧）不计，口径是用户可观察到的首段文本延迟。
//! - inter-chunk latency：相邻非空文本 chunk 的到达间隔。SSE chunk 不保证
//!   一一对应 token，因此该指标不会冒充 ITL；没有 token 级时间戳时 ITL 不可得。
//! - completion_tokens：优先取最终帧 `usage.completion_tokens`；缺失且显式提供
//!   `--tokenizer` 时，按完整输出文本重新分词（记录为 `tokenizer_text`）。绝不以
//!   chunk 数冒充 token 数。
//! - 失败归类：timeout / http_429 / http_4xx / http_5xx / connection /
//!   stream_error（SSE 读取或服务端 error）/ protocol_error（非法 SSE/JSON）/
//!   no_done（流提前结束）。
//!
//! # 输出
//! - `--out` 指定 per_request.jsonl（每行一条请求记录）；
//! - 与逐请求文件同目录写 `summary.json`，它是墙钟窗口、分位和吞吐的权威汇总；
//! - stdout 打印同口径的人类可读摘要。
//!
//! # 示例
//! ```bash
//! cargo run --release --bin loadgen -- \
//!   --base-url http://127.0.0.1:3000 --mode closed --concurrency 4 \
//!   --dataset benchmarks/serving/datasets/synth/work.jsonl \
//!   --requests 64 --warmup-secs 30 --max-tokens 128 \
//!   --out benchmarks/serving/results/run1/per_request.jsonl
//! ```

use clap::Parser;
use futures_util::StreamExt;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "OpenAI 兼容 serving 端点压测客户端（闭环/泊松双模式）")]
struct Args {
    /// 目标服务 base URL，如 http://127.0.0.1:3000
    #[arg(long)]
    base_url: String,

    /// 负载模式：closed（闭环饱和）| poisson（开环泊松到达）
    #[arg(long, default_value = "closed")]
    mode: String,

    /// 闭环并发槽数（closed 模式）
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// 平均到达率 req/s（poisson 模式）
    #[arg(long, default_value_t = 1.0)]
    rate: f64,

    /// Poisson 到达间隔的随机种子；省略时生成随机种子并写入 summary.json
    #[arg(long)]
    seed: Option<u64>,

    /// 数据集文件（jsonl，每行 {"prompt": "...", "prompt_tokens": 可选}）
    #[arg(long)]
    dataset: String,

    /// 测量窗口内总请求数（warmup 不计入）
    #[arg(long, default_value_t = 64)]
    requests: usize,

    /// warmup 秒数：以相同负载形态运行并丢弃结果（0 = 跳过）
    #[arg(long, default_value_t = 30)]
    warmup_secs: u64,

    /// 每请求生成 token 上限
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,

    /// 单请求超时（秒）
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,

    /// per_request.jsonl 输出路径
    #[arg(long, default_value = "per_request.jsonl")]
    out: String,

    /// summary.json 输出路径；缺省时写到 --out 同目录
    #[arg(long)]
    summary_out: Option<String>,

    /// 可选 tokenizer.json；仅在服务端未返回 usage 时统计完整输出文本 token 数
    #[arg(long)]
    tokenizer: Option<String>,

    /// 被测引擎标签，只写入 summary.json，不参与请求路由
    #[arg(long, default_value = "unknown")]
    engine: String,

    /// model 字段（透传给 API，不影响路由）
    #[arg(long, default_value = "paged-serving")]
    model: String,
}

#[derive(Deserialize, Clone)]
struct DatasetEntry {
    prompt: String,
    /// 离线 tokenize 的 prompt token 数（元数据，仅用于报表核对）
    #[serde(default)]
    prompt_tokens: Option<u32>,
}

#[derive(Serialize, Clone)]
struct RequestRecord {
    request_id: usize,
    /// 测量窗口内序号（warmup 请求为 null）
    measured_index: Option<usize>,
    ok: bool,
    error_class: Option<String>,
    /// 错误详情（stream_error 的服务端消息等；聚合统计仍用 error_class）
    error_detail: Option<String>,
    ttft_ms: Option<f64>,
    /// 相邻非空 SSE 文本 chunk 的到达间隔；不是 token 级 ITL。
    inter_chunk_latency_ms: Vec<f64>,
    duration_ms: f64,
    /// 非空文本 chunk 数
    chunks: u32,
    completion_tokens: Option<u32>,
    /// completion_tokens 来源："usage" | "tokenizer_text"
    tokens_source: Option<String>,
    finish_reason: Option<String>,
    prompt_tokens_meta: Option<u32>,
}

#[derive(Serialize)]
struct MetricSummary {
    samples: usize,
    p50: Option<f64>,
    p95: Option<f64>,
    p99: Option<f64>,
}

#[derive(Serialize)]
struct RequestSummary {
    total: usize,
    success: usize,
    failed: usize,
    success_rate_pct: f64,
}

#[derive(Serialize)]
struct TokenSummary {
    known_requests: usize,
    successful_requests: usize,
    coverage_pct: f64,
    total: u64,
    source_counts: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct ThroughputSummary {
    output_tokens_per_second: Option<f64>,
    successful_requests_per_second: f64,
}

#[derive(Serialize)]
struct RunConfigSummary {
    engine: String,
    mode: String,
    base_url: String,
    model: String,
    dataset: String,
    requests: usize,
    concurrency: Option<usize>,
    rate: Option<f64>,
    /// 实际用于 Poisson 指数间隔的随机种子；closed 模式为 null。
    arrival_seed: Option<u64>,
    max_tokens: u32,
    warmup_secs: u64,
    timeout_secs: u64,
    tokenizer: Option<String>,
}

#[derive(Serialize)]
struct RunSummary {
    schema_version: u32,
    config: RunConfigSummary,
    measurement_wall_secs: f64,
    requests: RequestSummary,
    ttft_ms: MetricSummary,
    inter_chunk_latency_ms: MetricSummary,
    /// 标准 ITL 需要 token 级时间戳；普通 OpenAI SSE 不提供该信息。
    itl_ms: Option<MetricSummary>,
    tpot_ms: MetricSummary,
    completion_tokens: TokenSummary,
    throughput: ThroughputSummary,
    errors: BTreeMap<String, usize>,
}

fn failure_record(
    request_id: usize,
    measured_index: Option<usize>,
    error_class: &str,
    error_detail: Option<String>,
    started_at: Instant,
    prompt_tokens_meta: Option<u32>,
) -> RequestRecord {
    RequestRecord {
        request_id,
        measured_index,
        ok: false,
        error_class: Some(error_class.to_string()),
        error_detail,
        ttft_ms: None,
        inter_chunk_latency_ms: Vec::new(),
        duration_ms: started_at.elapsed().as_secs_f64() * 1000.0,
        chunks: 0,
        completion_tokens: None,
        tokens_source: None,
        finish_reason: None,
        prompt_tokens_meta,
    }
}

/// 从字节缓冲区取出一个完整 SSE event 的 data 载荷。
///
/// 同时支持 LF 与 CRLF 分隔，并在完整 event 到齐后才做 UTF-8 解码，避免网络
/// chunk 恰好切在多字节字符中间时产生替换字符。
fn take_sse_data(buffer: &mut Vec<u8>) -> Result<Option<String>, std::str::Utf8Error> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|pos| (pos, 2));
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|pos| (pos, 4));
    let Some((pos, delimiter_len)) = [lf, crlf].into_iter().flatten().min_by_key(|(pos, _)| *pos)
    else {
        return Ok(None);
    };

    let consumed: Vec<u8> = buffer.drain(..pos + delimiter_len).collect();
    let event = std::str::from_utf8(&consumed[..pos])?;
    let data = event
        .lines()
        .filter_map(|line| {
            line.trim_end_matches('\r')
                .strip_prefix("data:")
                .map(str::trim_start)
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Some(data))
}

/// 单请求执行：发送流式 completions 请求，逐 chunk 打时间戳。
async fn run_request(
    client: &reqwest::Client,
    args: &Args,
    request_id: usize,
    measured_index: Option<usize>,
    entry: &DatasetEntry,
    output_tokenizer: Option<&Tokenizer>,
) -> RequestRecord {
    let body = serde_json::json!({
        "model": args.model,
        "prompt": entry.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": true,
    });

    let t_start = Instant::now();
    let resp = client
        .post(format!(
            "{}/v1/completions",
            args.base_url.trim_end_matches('/')
        ))
        .json(&body)
        .timeout(Duration::from_secs(args.timeout_secs))
        .send()
        .await;

    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            let class = if e.is_timeout() {
                "timeout"
            } else {
                "connection"
            };
            return failure_record(
                request_id,
                measured_index,
                class,
                Some(e.to_string()),
                t_start,
                entry.prompt_tokens,
            );
        }
    };

    let status = resp.status();
    if !status.is_success() {
        let class = match status.as_u16() {
            429 => "http_429",
            400..=499 => "http_4xx",
            _ => "http_5xx",
        };
        return failure_record(
            request_id,
            measured_index,
            class,
            Some(format!("HTTP {status}")),
            t_start,
            entry.prompt_tokens,
        );
    }

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    let mut chunk_times: Vec<Instant> = Vec::new();
    let mut output_text = String::new();
    let mut completion_tokens: Option<u32> = None;
    let mut finish_reason: Option<String> = None;
    let mut stream_failure: Option<(String, String)> = None;
    let mut saw_done = false;

    'outer: while let Some(item) = stream.next().await {
        let bytes = match item {
            Ok(b) => b,
            Err(e) => {
                stream_failure = Some((
                    "stream_error".to_string(),
                    format!("stream read error: {e}"),
                ));
                break;
            }
        };
        buf.extend_from_slice(&bytes);

        // SSE event 可能跨多个网络 chunk；仅解析已完整到达的 event。
        loop {
            let payload = match take_sse_data(&mut buf) {
                Ok(Some(payload)) => payload,
                Ok(None) => break,
                Err(e) => {
                    stream_failure = Some((
                        "protocol_error".to_string(),
                        format!("SSE event is not valid UTF-8: {e}"),
                    ));
                    break 'outer;
                }
            };
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                saw_done = true;
                break 'outer;
            }
            let v = match serde_json::from_str::<serde_json::Value>(&payload) {
                Ok(value) => value,
                Err(e) => {
                    stream_failure = Some((
                        "protocol_error".to_string(),
                        format!("invalid JSON in SSE data: {e}"),
                    ));
                    break 'outer;
                }
            };
            if let Some(err) = v.get("error") {
                stream_failure = Some((
                    "stream_error".to_string(),
                    err.get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown stream error")
                        .to_string(),
                ));
                break 'outer;
            }
            // 最终帧：usage / finish_reason（text 通常为空）
            if let Some(usage) = v.get("usage") {
                if let Some(ct) = usage
                    .get("completion_tokens")
                    .and_then(|c| c.as_u64())
                    .and_then(|ct| u32::try_from(ct).ok())
                {
                    completion_tokens = Some(ct);
                }
            }
            if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
                if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                    if !fr.is_empty() {
                        finish_reason = Some(fr.to_string());
                    }
                }
                let text = choice
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default();
                if !text.is_empty() {
                    chunk_times.push(Instant::now());
                    output_text.push_str(text);
                }
            }
        }
    }

    let duration_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    // 失败归类：流内 error > 无 [DONE]（即使已收到部分 chunk 也判失败，
    // 成功率口径不掺水）> 正常
    let (ok, error_class, error_detail) = if let Some((class, message)) = &stream_failure {
        (false, Some(class.clone()), Some(message.clone()))
    } else if !saw_done {
        (
            false,
            Some("no_done".to_string()),
            Some("SSE stream ended without [DONE]".to_string()),
        )
    } else {
        (true, None, None)
    };

    let ttft_ms = chunk_times
        .first()
        .map(|t| t.duration_since(t_start).as_secs_f64() * 1000.0);
    let inter_chunk_latency_ms: Vec<f64> = chunk_times
        .windows(2)
        .map(|w| (w[1].duration_since(w[0])).as_secs_f64() * 1000.0)
        .collect();

    let (tokens, tokens_source) = match completion_tokens {
        Some(ct) => (Some(ct), Some("usage".to_string())),
        None => output_tokenizer
            .and_then(|tokenizer| tokenizer.encode(output_text.as_str(), false).ok())
            .and_then(|encoding| u32::try_from(encoding.len()).ok())
            .map(|count| (Some(count), Some("tokenizer_text".to_string())))
            .unwrap_or((None, None)),
    };

    RequestRecord {
        request_id,
        measured_index,
        ok,
        error_class,
        error_detail,
        ttft_ms,
        inter_chunk_latency_ms,
        duration_ms,
        chunks: chunk_times.len() as u32,
        completion_tokens: tokens,
        tokens_source,
        finish_reason,
        prompt_tokens_meta: entry.prompt_tokens,
    }
}

fn load_dataset(path: &str) -> Result<Vec<DatasetEntry>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("读取数据集失败: {e}"))?;
    let mut entries = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: DatasetEntry = serde_json::from_str(line)
            .map_err(|e| format!("数据集第 {} 行解析失败: {e}", i + 1))?;
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err("数据集为空".to_string());
    }
    Ok(entries)
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    // 最近秩法（与评测口径一致，避免插值引入的歧义）
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

fn metric_summary(mut values: Vec<f64>) -> MetricSummary {
    values.sort_by(f64::total_cmp);
    MetricSummary {
        samples: values.len(),
        p50: percentile(&values, 50.0),
        p95: percentile(&values, 95.0),
        p99: percentile(&values, 99.0),
    }
}

fn poisson_interval_secs(rng: &mut impl Rng, rate: f64) -> f64 {
    -(1.0 - rng.gen::<f64>()).ln() / rate
}

fn build_summary(records: &[RequestRecord], wall_secs: f64, args: &Args) -> RunSummary {
    // warmup 记录（measured_index = null）不进正式 summary。main 本就不会把
    // warmup 结果 push 进 records，这里把该不变量固化在聚合入口，防止未来
    // 调用方误传。
    let measured: Vec<&RequestRecord> = records
        .iter()
        .filter(|r| r.measured_index.is_some())
        .collect();
    let total = measured.len();
    let ok_records: Vec<&RequestRecord> = measured.iter().copied().filter(|r| r.ok).collect();
    let ok = ok_records.len();

    let ttfts = ok_records.iter().filter_map(|r| r.ttft_ms).collect();
    let inter_chunk_latencies = ok_records
        .iter()
        .flat_map(|r| r.inter_chunk_latency_ms.iter().copied())
        .collect();
    // 逐请求 TPOT：(duration - ttft) / (tokens - 1)
    let tpots = ok_records
        .iter()
        .filter_map(|r| {
            let tokens = r.completion_tokens.unwrap_or(0);
            match (r.ttft_ms, tokens > 1) {
                (Some(ttft), true) => Some((r.duration_ms - ttft) / (tokens as f64 - 1.0)),
                _ => None,
            }
        })
        .collect();

    let known_token_records: Vec<&RequestRecord> = ok_records
        .iter()
        .copied()
        .filter(|r| r.completion_tokens.is_some())
        .collect();
    let total_tokens: u64 = known_token_records
        .iter()
        .map(|r| u64::from(r.completion_tokens.unwrap_or(0)))
        .sum();
    let mut source_counts = BTreeMap::new();
    for record in &known_token_records {
        if let Some(source) = &record.tokens_source {
            *source_counts.entry(source.clone()).or_insert(0) += 1;
        }
    }

    // 失败归类计数
    let mut error_counts: BTreeMap<String, usize> = Default::default();
    for r in measured.iter().filter(|r| !r.ok) {
        *error_counts
            .entry(r.error_class.clone().unwrap_or_else(|| "unknown".into()))
            .or_insert(0) += 1;
    }

    let token_coverage_pct = if ok > 0 {
        100.0 * known_token_records.len() as f64 / ok as f64
    } else {
        0.0
    };
    let output_tokens_per_second = if wall_secs > 0.0 && ok > 0 && known_token_records.len() == ok {
        Some(total_tokens as f64 / wall_secs)
    } else {
        None
    };

    RunSummary {
        schema_version: 1,
        config: RunConfigSummary {
            engine: args.engine.clone(),
            mode: args.mode.clone(),
            base_url: args.base_url.clone(),
            model: args.model.clone(),
            dataset: args.dataset.clone(),
            requests: args.requests,
            concurrency: (args.mode == "closed").then_some(args.concurrency),
            rate: (args.mode == "poisson").then_some(args.rate),
            arrival_seed: (args.mode == "poisson").then_some(args.seed).flatten(),
            max_tokens: args.max_tokens,
            warmup_secs: args.warmup_secs,
            timeout_secs: args.timeout_secs,
            tokenizer: args.tokenizer.clone(),
        },
        measurement_wall_secs: wall_secs,
        requests: RequestSummary {
            total,
            success: ok,
            failed: total - ok,
            success_rate_pct: if total > 0 {
                100.0 * ok as f64 / total as f64
            } else {
                0.0
            },
        },
        ttft_ms: metric_summary(ttfts),
        inter_chunk_latency_ms: metric_summary(inter_chunk_latencies),
        itl_ms: None,
        tpot_ms: metric_summary(tpots),
        completion_tokens: TokenSummary {
            known_requests: known_token_records.len(),
            successful_requests: ok,
            coverage_pct: token_coverage_pct,
            total: total_tokens,
            source_counts,
        },
        throughput: ThroughputSummary {
            output_tokens_per_second,
            successful_requests_per_second: if wall_secs > 0.0 {
                ok as f64 / wall_secs
            } else {
                0.0
            },
        },
        errors: error_counts,
    }
}

fn print_summary(summary: &RunSummary) {
    let fmt = |v: Option<f64>| v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "-".into());
    let fmt_metric = |metric: &MetricSummary| {
        format!(
            "{} / {} / {}（n={}）",
            fmt(metric.p50),
            fmt(metric.p95),
            fmt(metric.p99),
            metric.samples
        )
    };

    println!("=== loadgen 汇总 ===");
    println!(
        "引擎: {} | 模式: {} | 目标: {}",
        summary.config.engine, summary.config.mode, summary.config.base_url
    );
    println!(
        "请求: {}（成功 {}，成功率 {:.2}%）| 测量窗口: {:.3}s",
        summary.requests.total,
        summary.requests.success,
        summary.requests.success_rate_pct,
        summary.measurement_wall_secs
    );
    if !summary.errors.is_empty() {
        let errs: Vec<String> = summary
            .errors
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        println!("失败归类: {}", errs.join(", "));
    }
    println!("TTFT ms p50/p95/p99: {}", fmt_metric(&summary.ttft_ms));
    println!(
        "chunk 间隔 ms p50/p95/p99: {}（不是 ITL）",
        fmt_metric(&summary.inter_chunk_latency_ms)
    );
    println!("ITL ms: -（协议未提供 token 级时间戳）");
    println!("TPOT ms p50/p95/p99: {}", fmt_metric(&summary.tpot_ms));
    println!(
        "token 计数覆盖: {}/{}（{:.2}%）",
        summary.completion_tokens.known_requests,
        summary.completion_tokens.successful_requests,
        summary.completion_tokens.coverage_pct
    );
    match summary.throughput.output_tokens_per_second {
        Some(tokens_per_second) => println!(
            "吞吐: {tokens_per_second:.1} tok/s | {:.2} req/s",
            summary.throughput.successful_requests_per_second
        ),
        None => println!(
            "吞吐: tok/s 不可用（token 计数覆盖不足）| {:.2} req/s",
            summary.throughput.successful_requests_per_second
        ),
    }
}

fn default_summary_path(request_path: &str) -> PathBuf {
    Path::new(request_path)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("summary.json")
}

fn write_summary(path: &Path, summary: &RunSummary) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut output = serde_json::to_string_pretty(summary)?;
    output.push('\n');
    std::fs::write(path, output)?;
    println!("权威汇总已写入: {}", path.display());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();
    env_logger::init();

    if args.mode != "closed" && args.mode != "poisson" {
        eprintln!("--mode 必须是 closed 或 poisson");
        std::process::exit(2);
    }
    if args.mode == "closed" && args.concurrency == 0 {
        eprintln!("closed 模式 --concurrency 必须 >= 1");
        std::process::exit(2);
    }
    if args.mode == "poisson" && (!args.rate.is_finite() || args.rate <= 0.0) {
        eprintln!("poisson 模式 --rate 必须是有限正数");
        std::process::exit(2);
    }
    if args.requests == 0 {
        eprintln!("--requests 必须 >= 1");
        std::process::exit(2);
    }
    if args.max_tokens == 0 {
        eprintln!("--max-tokens 必须 >= 1");
        std::process::exit(2);
    }
    if args.timeout_secs == 0 {
        eprintln!("--timeout-secs 必须 >= 1");
        std::process::exit(2);
    }
    if args.mode == "poisson" && args.seed.is_none() {
        args.seed = Some(rand::random());
    }

    let dataset = load_dataset(&args.dataset)?;
    let output_tokenizer = match &args.tokenizer {
        Some(path) => Some(Arc::new(Tokenizer::from_file(path).map_err(|e| {
            std::io::Error::other(format!("加载 tokenizer 失败（{path}）: {e}"))
        })?)),
        None => None,
    };
    println!(
        "数据集: {} 条 | 模式: {} | 并发/速率: {} | 测量请求数: {} | warmup: {}s",
        dataset.len(),
        args.mode,
        if args.mode == "closed" {
            args.concurrency.to_string()
        } else {
            format!("{} req/s", args.rate)
        },
        args.requests,
        args.warmup_secs
    );
    if let Some(seed) = (args.mode == "poisson").then_some(args.seed).flatten() {
        println!("Poisson arrival seed: {seed}");
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(args.concurrency.max(16))
        .build()?;

    let args = Arc::new(args);
    let dataset = Arc::new(dataset);
    let records: Arc<std::sync::Mutex<Vec<RequestRecord>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let next_id = Arc::new(AtomicUsize::new(0));

    // 取下一个数据集条目（环绕）；返回 (全局请求 id, 条目)。
    let pick = |next_id: &Arc<AtomicUsize>, dataset: &Arc<Vec<DatasetEntry>>| {
        let id = next_id.fetch_add(1, Ordering::Relaxed);
        (id, dataset[id % dataset.len()].clone())
    };

    let measured_count = Arc::new(AtomicUsize::new(0));

    let wall = if args.mode == "closed" {
        // warmup 阶段：闭环跑 warmup_secs，结果丢弃。
        if args.warmup_secs > 0 {
            let stop = Arc::new(AtomicBool::new(false));
            let mut warmup_handles = Vec::new();
            for _ in 0..args.concurrency {
                let (client, args, dataset, next_id, stop, output_tokenizer) = (
                    client.clone(),
                    args.clone(),
                    dataset.clone(),
                    next_id.clone(),
                    stop.clone(),
                    output_tokenizer.clone(),
                );
                warmup_handles.push(tokio::spawn(async move {
                    while !stop.load(Ordering::Relaxed) {
                        let (id, entry) = pick(&next_id, &dataset);
                        let rec = run_request(
                            &client,
                            &args,
                            id,
                            None,
                            &entry,
                            output_tokenizer.as_deref(),
                        )
                        .await;
                        if !rec.ok {
                            log::warn!("warmup 请求 {id} 失败: {:?}", rec.error_class);
                        }
                    }
                }));
            }
            tokio::time::sleep(Duration::from_secs(args.warmup_secs)).await;
            stop.store(true, Ordering::Relaxed);
            for h in warmup_handles {
                let _ = h.await;
            }
            println!("warmup 完成（{}s），开始测量窗口", args.warmup_secs);
        }
        let measure_start = Instant::now();

        // 测量阶段：闭环直到完成 args.requests 个请求。
        let mut handles = Vec::new();
        for _ in 0..args.concurrency {
            let (client, args, dataset, next_id, records, measured_count, output_tokenizer) = (
                client.clone(),
                args.clone(),
                dataset.clone(),
                next_id.clone(),
                records.clone(),
                measured_count.clone(),
                output_tokenizer.clone(),
            );
            handles.push(tokio::spawn(async move {
                loop {
                    let idx = measured_count.fetch_add(1, Ordering::Relaxed);
                    if idx >= args.requests {
                        break;
                    }
                    let (id, entry) = pick(&next_id, &dataset);
                    let rec = run_request(
                        &client,
                        &args,
                        id,
                        Some(idx),
                        &entry,
                        output_tokenizer.as_deref(),
                    )
                    .await;
                    records.lock().unwrap().push(rec);
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        measure_start.elapsed().as_secs_f64()
    } else {
        // poisson 模式：指数间隔到达；warmup 期间同样到达但丢弃。
        let seed = args.seed.expect("poisson mode always has an arrival seed");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut warmup_handles: Vec<tokio::task::JoinHandle<RequestRecord>> = Vec::new();
        let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
        let mut issued = 0usize;

        // 先发射 warmup 流量（不计入 issued 上限）。
        if args.warmup_secs > 0 {
            let warmup_until = Instant::now() + Duration::from_secs(args.warmup_secs);
            while Instant::now() < warmup_until {
                let dt = poisson_interval_secs(&mut rng, args.rate);
                tokio::time::sleep(Duration::from_secs_f64(dt)).await;
                let (id, entry) = pick(&next_id, &dataset);
                let (client, args, output_tokenizer) =
                    (client.clone(), args.clone(), output_tokenizer.clone());
                warmup_handles.push(tokio::spawn(async move {
                    run_request(
                        &client,
                        &args,
                        id,
                        None,
                        &entry,
                        output_tokenizer.as_deref(),
                    )
                    .await
                }));
            }
            for h in warmup_handles {
                let _ = h.await; // warmup 结果丢弃，仅起预热作用
            }
            println!("warmup 完成（{}s），开始测量窗口", args.warmup_secs);
        }
        let measure_start = Instant::now();

        while issued < args.requests {
            let dt = poisson_interval_secs(&mut rng, args.rate);
            tokio::time::sleep(Duration::from_secs_f64(dt)).await;
            let (id, entry) = pick(&next_id, &dataset);
            let idx = issued;
            issued += 1;
            let (client, args, records, output_tokenizer) = (
                client.clone(),
                args.clone(),
                records.clone(),
                output_tokenizer.clone(),
            );
            handles.push(tokio::spawn(async move {
                let rec = run_request(
                    &client,
                    &args,
                    id,
                    Some(idx),
                    &entry,
                    output_tokenizer.as_deref(),
                )
                .await;
                records.lock().unwrap().push(rec);
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        measure_start.elapsed().as_secs_f64()
    };

    let recs = finalize_records(&records);
    write_records(&args.out, &recs)?;
    let summary = build_summary(&recs, wall, &args);
    let summary_path = args
        .summary_out
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_summary_path(&args.out));
    write_summary(&summary_path, &summary)?;
    print_summary(&summary);

    Ok(())
}

/// 按测量窗口序号排序后取出全部记录。
fn finalize_records(records: &Arc<std::sync::Mutex<Vec<RequestRecord>>>) -> Vec<RequestRecord> {
    let mut g = records.lock().unwrap();
    g.sort_by_key(|r| r.measured_index);
    (*g).clone()
}

fn write_records(path: &str, records: &[RequestRecord]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut out = String::new();
    for r in records {
        out.push_str(&serde_json::to_string(r)?);
        out.push('\n');
    }
    std::fs::write(path, out)?;
    println!("逐请求记录已写入: {path}（{} 条）", records.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            base_url: "http://127.0.0.1:3000".to_string(),
            mode: "closed".to_string(),
            concurrency: 2,
            rate: 1.0,
            seed: None,
            dataset: "smoke.jsonl".to_string(),
            requests: 2,
            warmup_secs: 0,
            max_tokens: 8,
            timeout_secs: 10,
            out: "results/per_request.jsonl".to_string(),
            summary_out: None,
            tokenizer: None,
            engine: "paged-serving".to_string(),
            model: "test-model".to_string(),
        }
    }

    fn successful_record(completion_tokens: Option<u32>) -> RequestRecord {
        RequestRecord {
            request_id: 0,
            measured_index: Some(0),
            ok: true,
            error_class: None,
            error_detail: None,
            ttft_ms: Some(10.0),
            inter_chunk_latency_ms: vec![2.0, 3.0],
            duration_ms: 30.0,
            chunks: 3,
            completion_tokens,
            tokens_source: completion_tokens.map(|_| "usage".to_string()),
            finish_reason: Some("length".to_string()),
            prompt_tokens_meta: Some(4),
        }
    }

    fn failure_record_with(class: &str) -> RequestRecord {
        RequestRecord {
            request_id: 0,
            measured_index: Some(0),
            ok: false,
            error_class: Some(class.to_string()),
            error_detail: Some("detail".to_string()),
            ttft_ms: None,
            inter_chunk_latency_ms: Vec::new(),
            duration_ms: 5.0,
            chunks: 0,
            completion_tokens: None,
            tokens_source: None,
            finish_reason: None,
            prompt_tokens_meta: None,
        }
    }

    // ---- 真实 HTTP/SSE 回归：本地一次性 server ----
    // 用 std::net 而非 tokio::net：本 crate 未启用 tokio "net" feature，
    // 测试 server 只需阻塞写若干字节。

    /// 读完一个 HTTP 请求（headers + Content-Length body）。若未消费请求体
    /// 就提前写响应并关连接，client 可能先读到 RST 而非响应。
    fn read_http_request(stream: &mut std::net::TcpStream) {
        use std::io::Read;
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
            let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]).into_owned();
            let content_length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|v| v.trim().parse().ok())
                })
                .unwrap_or(0);
            let body_end = header_end + 4 + content_length;
            while buf.len() < body_end {
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
            return;
        }
    }

    /// 启动一次性测试 server：accept 一个连接 → 读请求 → 执行 respond
    /// （写完整响应，或模拟故障如不响应）。返回 (base_url, join)。
    fn spawn_server(
        respond: impl FnOnce(&mut std::net::TcpStream) + Send + 'static,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let port = listener.local_addr().unwrap().port();
        let join = std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            read_http_request(&mut stream);
            respond(&mut stream);
        });
        (format!("http://127.0.0.1:{port}"), join)
    }

    /// 最小 HTTP 响应。Connection: close 且不带长度 → body 以 EOF 结束，
    /// 与 SSE 流式语义一致（服务端发完即关）。
    fn http_response(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut out = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n"
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    }

    /// LF 分隔的 SSE 响应体，每个元素一个 `data:` event。
    fn sse_response(events: &[&str]) -> Vec<u8> {
        let mut body = Vec::new();
        for event in events {
            body.extend_from_slice(b"data: ");
            body.extend_from_slice(event.as_bytes());
            body.extend_from_slice(b"\n\n");
        }
        http_response("200 OK", "text/event-stream", &body)
    }

    fn test_entry() -> DatasetEntry {
        DatasetEntry {
            prompt: "hello".to_string(),
            prompt_tokens: Some(2),
        }
    }

    /// 对一次性 server 执行一次真实 run_request，返回聚合后的 record。
    async fn run_against(response: Vec<u8>, mut args: Args) -> RequestRecord {
        let (base_url, join) = spawn_server(move |stream| {
            use std::io::Write;
            let _ = stream.write_all(&response);
            let _ = stream.flush();
        });
        args.base_url = base_url;
        let rec = run_request(
            &reqwest::Client::new(),
            &args,
            0,
            Some(0),
            &test_entry(),
            None,
        )
        .await;
        join.join().expect("test server thread");
        rec
    }

    #[tokio::test]
    async fn run_request_success_with_usage_over_real_http() {
        let rec = run_against(
            sse_response(&[
                "{\"choices\":[{\"text\":\"Hel\",\"finish_reason\":null}]}",
                "{\"choices\":[{\"text\":\"lo\",\"finish_reason\":null}]}",
                "{\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":7}}",
                "[DONE]",
            ]),
            test_args(),
        )
        .await;
        assert!(rec.ok, "unexpected failure: {:?}", rec.error_class);
        assert_eq!(rec.chunks, 2);
        assert_eq!(rec.completion_tokens, Some(7));
        assert_eq!(rec.tokens_source.as_deref(), Some("usage"));
        assert_eq!(rec.finish_reason.as_deref(), Some("stop"));
        assert!(rec.ttft_ms.is_some());
        assert_eq!(rec.inter_chunk_latency_ms.len(), 1);
        assert!(rec.error_class.is_none() && rec.error_detail.is_none());
    }

    #[tokio::test]
    async fn run_request_accepts_crlf_sse_end_to_end() {
        let body = b"data: {\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"stop\"}]}\r\n\r\ndata: [DONE]\r\n\r\n";
        let rec = run_against(
            http_response("200 OK", "text/event-stream", body),
            test_args(),
        )
        .await;
        assert!(rec.ok, "unexpected failure: {:?}", rec.error_class);
        assert_eq!(rec.chunks, 1);
        assert_eq!(rec.finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn run_request_handles_utf8_split_across_tcp_writes() {
        let (base_url, join) = spawn_server(|stream| {
            use std::io::Write;
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            );
            // "你" = E4 BD A0；首字节单独发出，模拟网络 chunk 切在多字节
            // 字符中间。缓冲层若按到达边界解码会产生 protocol_error。
            let _ = stream.write_all(b"data: {\"choices\":[{\"text\":\"\xE4");
            let _ = stream.flush();
            std::thread::sleep(Duration::from_millis(30));
            let _ =
                stream.write_all(b"\xBD\xA0\",\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
            let _ = stream.flush();
        });
        let mut args = test_args();
        args.base_url = base_url;
        let rec = run_request(
            &reqwest::Client::new(),
            &args,
            0,
            Some(0),
            &test_entry(),
            None,
        )
        .await;
        join.join().expect("test server thread");
        assert!(rec.ok, "split UTF-8 must not fail: {:?}", rec.error_class);
        assert_eq!(rec.chunks, 1);
    }

    #[tokio::test]
    async fn run_request_classifies_http_error_statuses() {
        for (status, expected) in [
            ("429 Too Many Requests", "http_429"),
            ("400 Bad Request", "http_4xx"),
            ("500 Internal Server Error", "http_5xx"),
        ] {
            let rec = run_against(
                http_response(status, "application/json", b"{\"error\":\"x\"}"),
                test_args(),
            )
            .await;
            assert!(!rec.ok, "{status} must fail");
            assert_eq!(rec.error_class.as_deref(), Some(expected));
            assert!(
                rec.error_detail.as_deref().is_some_and(|d| !d.is_empty()),
                "{expected} must keep non-empty detail"
            );
        }
    }

    #[tokio::test]
    async fn run_request_classifies_connection_refused() {
        // 绑定后立即释放端口，得到确定性的"无监听者"地址。
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut args = test_args();
        args.base_url = format!("http://127.0.0.1:{port}");
        let rec = run_request(
            &reqwest::Client::new(),
            &args,
            0,
            Some(0),
            &test_entry(),
            None,
        )
        .await;
        assert!(!rec.ok);
        assert_eq!(rec.error_class.as_deref(), Some("connection"));
        assert!(rec.error_detail.as_deref().is_some_and(|d| !d.is_empty()));
    }

    #[tokio::test]
    async fn run_request_classifies_timeout() {
        let (base_url, _join) = spawn_server(|_stream| {
            // 读完请求后保持连接不响应；客户端 1s 超时先触发。
            // 不 join：server 线程睡完即退，与测试无共享状态。
            std::thread::sleep(Duration::from_secs(3));
        });
        let mut args = test_args();
        args.base_url = base_url;
        args.timeout_secs = 1;
        let rec = run_request(
            &reqwest::Client::new(),
            &args,
            0,
            Some(0),
            &test_entry(),
            None,
        )
        .await;
        assert!(!rec.ok);
        assert_eq!(rec.error_class.as_deref(), Some("timeout"));
        assert!(rec.error_detail.as_deref().is_some_and(|d| !d.is_empty()));
    }

    #[tokio::test]
    async fn run_request_classifies_invalid_json_as_protocol_error() {
        let rec = run_against(sse_response(&["{not json}", "[DONE]"]), test_args()).await;
        assert!(!rec.ok);
        assert_eq!(rec.error_class.as_deref(), Some("protocol_error"));
        assert!(rec.error_detail.as_deref().is_some_and(|d| !d.is_empty()));
    }

    #[tokio::test]
    async fn run_request_classifies_invalid_utf8_as_protocol_error() {
        let mut body = b"data: ".to_vec();
        body.extend_from_slice(&[0xFF, 0xFE]);
        body.extend_from_slice(b"\n\n");
        let rec = run_against(
            http_response("200 OK", "text/event-stream", &body),
            test_args(),
        )
        .await;
        assert!(!rec.ok);
        assert_eq!(rec.error_class.as_deref(), Some("protocol_error"));
        assert!(rec.error_detail.as_deref().is_some_and(|d| !d.is_empty()));
    }

    #[tokio::test]
    async fn run_request_classifies_error_frame_as_stream_error() {
        let rec = run_against(
            sse_response(&[
                "{\"choices\":[{\"text\":\"x\",\"finish_reason\":null}]}",
                "{\"error\":{\"message\":\"engine overloaded\"}}",
                "[DONE]",
            ]),
            test_args(),
        )
        .await;
        assert!(!rec.ok);
        assert_eq!(rec.error_class.as_deref(), Some("stream_error"));
        // 服务端错误消息透传为 detail（样例 record 保留非空 detail）。
        assert_eq!(rec.error_detail.as_deref(), Some("engine overloaded"));
    }

    #[tokio::test]
    async fn run_request_classifies_missing_done_as_failure() {
        let rec = run_against(
            sse_response(&["{\"choices\":[{\"text\":\"x\",\"finish_reason\":null}]}"]),
            test_args(),
        )
        .await;
        assert!(!rec.ok);
        assert_eq!(rec.error_class.as_deref(), Some("no_done"));
        assert!(rec.error_detail.as_deref().is_some_and(|d| !d.is_empty()));
    }

    #[tokio::test]
    async fn run_request_without_usage_leaves_token_count_unknown() {
        let rec = run_against(
            sse_response(&[
                "{\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"stop\"}]}",
                "[DONE]",
            ]),
            test_args(),
        )
        .await;
        assert!(rec.ok);
        // 无 usage 且无 tokenizer → token 数未知，coverage 不完整时汇总层
        // 不得产出 tok/s。
        assert_eq!(rec.completion_tokens, None);
        assert_eq!(rec.tokens_source, None);
        let summary = build_summary(&[rec], 1.0, &test_args());
        assert_eq!(summary.requests.success, 1);
        assert_eq!(summary.completion_tokens.known_requests, 0);
        assert_eq!(summary.completion_tokens.coverage_pct, 0.0);
        assert!(summary.throughput.output_tokens_per_second.is_none());
    }

    #[test]
    fn summary_excludes_warmup_records() {
        let args = test_args();
        let measured = successful_record(Some(4));
        let mut warmup = successful_record(Some(4));
        warmup.measured_index = None;
        let summary = build_summary(&[measured, warmup], 1.0, &args);
        assert_eq!(summary.requests.total, 1);
        assert_eq!(summary.requests.success, 1);
        assert_eq!(summary.completion_tokens.known_requests, 1);
        assert_eq!(summary.completion_tokens.total, 4);
    }

    #[test]
    fn summary_aggregates_error_classes() {
        let args = test_args();
        let summary = build_summary(
            &[
                failure_record_with("no_done"),
                failure_record_with("no_done"),
                failure_record_with("http_429"),
            ],
            1.0,
            &args,
        );
        assert_eq!(summary.requests.total, 3);
        assert_eq!(summary.requests.failed, 3);
        assert_eq!(summary.requests.success, 0);
        assert_eq!(summary.errors.get("no_done"), Some(&2));
        assert_eq!(summary.errors.get("http_429"), Some(&1));
    }

    #[test]
    fn sse_parser_waits_for_complete_utf8_event_and_accepts_crlf() {
        let event = "data: {\"text\":\"你\"}\r\n\r\n".as_bytes();
        let split = event.iter().position(|byte| *byte >= 0x80).unwrap() + 1;
        let mut buffer = event[..split].to_vec();
        assert!(take_sse_data(&mut buffer).unwrap().is_none());

        buffer.extend_from_slice(&event[split..]);
        assert_eq!(
            take_sse_data(&mut buffer).unwrap().as_deref(),
            Some("{\"text\":\"你\"}")
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn sse_parser_joins_multiple_data_lines() {
        let mut buffer = b"event: message\ndata: first\ndata: second\n\n".to_vec();
        assert_eq!(
            take_sse_data(&mut buffer).unwrap().as_deref(),
            Some("first\nsecond")
        );
    }

    #[test]
    fn token_throughput_requires_complete_successful_request_coverage() {
        let args = test_args();
        let partial = build_summary(
            &[successful_record(Some(4)), successful_record(None)],
            1.0,
            &args,
        );
        assert_eq!(partial.completion_tokens.known_requests, 1);
        assert_eq!(partial.completion_tokens.coverage_pct, 50.0);
        assert!(partial.throughput.output_tokens_per_second.is_none());

        let complete = build_summary(
            &[successful_record(Some(4)), successful_record(Some(6))],
            2.0,
            &args,
        );
        assert_eq!(complete.throughput.output_tokens_per_second, Some(5.0));
        assert_eq!(complete.tpot_ms.samples, 2);
    }

    #[test]
    fn summary_defaults_to_request_record_directory() {
        assert_eq!(
            default_summary_path("results/run/per_request.jsonl"),
            PathBuf::from("results/run/summary.json")
        );
        assert_eq!(
            default_summary_path("per_request.jsonl"),
            PathBuf::from("summary.json")
        );
    }

    #[test]
    fn seeded_poisson_intervals_are_reproducible() {
        let mut first = StdRng::seed_from_u64(42);
        let mut second = StdRng::seed_from_u64(42);
        let first_intervals: Vec<f64> = (0..4)
            .map(|_| poisson_interval_secs(&mut first, 0.64))
            .collect();
        let second_intervals: Vec<f64> = (0..4)
            .map(|_| poisson_interval_secs(&mut second, 0.64))
            .collect();

        assert_eq!(first_intervals, second_intervals);
        assert!(first_intervals.iter().all(|interval| *interval > 0.0));
    }
}
