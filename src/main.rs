//! Paged-Serving - Main Entry Point

use clap::{Parser, ValueEnum};
use log::info;
#[cfg(feature = "tiny-llm")]
use paged_serving::{build_tokenizer, Scheduler, TinyLlmExecutor};
use paged_serving::{
    create_router_with_engine_and_shutdown, EngineConfig, GenerationParams, InferenceEngine,
    TokenizerConfig, TokenizerKind,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    #[default]
    Cpu,
    TinyLlm,
}

#[derive(Parser, Debug)]
#[command(name = "paged-serving")]
#[command(
    about = "Paged-memory, continuously-batched inference engine scaffold with a CPU reference backend"
)]
struct Args {
    /// Path to configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Block size (tokens per block)
    #[arg(long)]
    block_size: Option<u32>,

    /// Maximum number of blocks
    #[arg(long)]
    max_num_blocks: Option<u32>,

    /// Maximum batch size
    #[arg(long)]
    max_batch_size: Option<u32>,

    /// Maximum number of sequences
    #[arg(long)]
    max_num_seqs: Option<u32>,

    /// Maximum model length
    #[arg(long)]
    max_model_len: Option<u32>,

    /// Maximum total tokens per batch
    #[arg(long)]
    max_total_tokens: Option<u32>,

    /// Memory pressure threshold (0.0 - 1.0)
    #[arg(long)]
    memory_threshold: Option<f32>,

    /// Serve listen host (overrides config file / default 127.0.0.1)
    #[arg(long)]
    host: Option<String>,

    /// Serve listen port (overrides config file / default 3000)
    #[arg(long)]
    port: Option<u16>,

    /// Input text to process
    #[arg(short, long)]
    input: Option<String>,

    /// Start OpenAI-compatible HTTP server
    #[arg(long)]
    serve: bool,

    /// Execution backend. tiny-llm requires --features tiny-llm at build time
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,

    /// GGUF model used by --backend tiny-llm
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "100")]
    max_tokens: u32,

    /// Sampling temperature (only 0.0 = greedy is supported by the CPU backend)
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Top-p sampling parameter (only 1.0 is supported by the CPU backend)
    #[arg(long, default_value = "1.0")]
    top_p: f32,

    /// HuggingFace tokenizer.json 路径；设置后引擎改用 HF tokenizer（完整有效词表
    /// 151665，GGUF embedding 可能为 151936 并含 padding 行），替代默认的 SimpleTokenizer
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

fn create_engine(
    config: EngineConfig,
    backend: BackendKind,
    model_path: Option<&Path>,
) -> Result<InferenceEngine, Box<dyn std::error::Error>> {
    match (backend, model_path) {
        (BackendKind::Cpu, None) => Ok(InferenceEngine::new(config)?),
        (BackendKind::Cpu, Some(_)) => Err("--model-path requires --backend tiny-llm".into()),
        (BackendKind::TinyLlm, None) => Err("--backend tiny-llm requires --model-path".into()),
        (BackendKind::TinyLlm, Some(model_path)) => {
            #[cfg(feature = "tiny-llm")]
            {
                let model_path = model_path
                    .to_str()
                    .ok_or("--model-path must be valid UTF-8")?;
                let tokenizer = build_tokenizer(&config)?;
                let scheduler = Scheduler::new(config.clone());
                let executor = TinyLlmExecutor::new(model_path, config.clone())?;
                Ok(InferenceEngine::with_components(
                    config,
                    tokenizer,
                    scheduler,
                    Box::new(executor),
                )?)
            }

            #[cfg(not(feature = "tiny-llm"))]
            {
                let _ = model_path;
                Err("--backend tiny-llm requires a binary built with --features tiny-llm".into())
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    let mut config = if let Some(config_path) = args.config {
        // 之前 --config 会静默忽略所有单项 CLI 参数，极易误配；现在显式报错。
        let has_overrides = [
            args.block_size.is_some(),
            args.max_num_blocks.is_some(),
            args.max_batch_size.is_some(),
            args.max_num_seqs.is_some(),
            args.max_model_len.is_some(),
            args.max_total_tokens.is_some(),
            args.memory_threshold.is_some(),
        ]
        .contains(&true);
        if has_overrides {
            return Err(
                "--config cannot be combined with individual engine-config flags \
                        (--block-size, --max-num-blocks, --max-batch-size, --max-num-seqs, \
                         --max-model-len, --max-total-tokens, --memory-threshold)"
                    .into(),
            );
        }
        EngineConfig::from_file(&config_path)?
    } else {
        let mut config = EngineConfig::default();
        if let Some(v) = args.block_size {
            config.block_size = v;
        }
        if let Some(v) = args.max_num_blocks {
            config.max_num_blocks = v;
        }
        if let Some(v) = args.max_batch_size {
            config.max_batch_size = v;
        }
        if let Some(v) = args.max_num_seqs {
            config.max_num_seqs = v;
        }
        if let Some(v) = args.max_model_len {
            config.max_model_len = v;
        }
        if let Some(v) = args.max_total_tokens {
            config.max_total_tokens = v;
        }
        if let Some(v) = args.memory_threshold {
            config.memory_threshold = v;
        }
        config
    };

    // serving 覆盖对两种配置来源都生效（--host/--port 可与 --config 组合）
    if let Some(host) = args.host {
        config.serving.host = host;
    }
    if let Some(port) = args.port {
        config.serving.port = port;
    }

    // --tokenizer 可与 --config 组合：显式覆盖配置文件的 tokenizer 设置
    if let Some(path) = args.tokenizer {
        config.tokenizer = TokenizerConfig {
            kind: TokenizerKind::HuggingFace,
            path: Some(path),
        };
    }

    info!("Starting Paged-Serving");
    info!("Configuration: {:?}", config);

    println!("Paged-Serving");
    println!("===========");
    println!("Configuration:");
    println!("  Block size: {}", config.block_size);
    println!("  Max blocks: {}", config.max_num_blocks);
    println!("  Max batch size: {}", config.max_batch_size);
    println!("  Max sequences: {}", config.max_num_seqs);
    println!();

    let engine = create_engine(config.clone(), args.backend, args.model_path.as_deref())?;

    if args.serve {
        let bind_addr = format!("{}:{}", config.serving.host, config.serving.port);
        info!("Starting OpenAI-compatible server on {}", bind_addr);
        println!("Server mode: {}", bind_addr);
        println!("Model name: {}", config.serving.model_name);
        println!("Backend: {:?}", args.backend);

        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
        let (app, shutdown_trigger) = create_router_with_engine_and_shutdown(config, engine)?;
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown_signal().await;
                // 广播引擎循环取消全部在途请求：graceful shutdown 不主动
                // 断开连接，没有 cancel-all 时长 SSE 流会使排空无限挂起。
                let _ = shutdown_trigger.send(true);
            })
            .await?;
        info!("Server shut down gracefully");
        return Ok(());
    }

    let mut engine = engine;

    // Process input if provided
    if let Some(input_text) = args.input {
        let params = GenerationParams {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            stop: Vec::new(), // CLI 暂不暴露 stop 参数
            logprobs: None,
            priority: 0,
        };

        println!("Input: {}", input_text);
        println!("Generating up to {} tokens...", args.max_tokens);
        println!();

        // Submit request
        let (request_id, prompt_tokens) = engine.submit_request(&input_text, params)?;
        info!(
            "Submitted request: {} ({} prompt tokens)",
            request_id, prompt_tokens
        );

        // Run inference
        let completed = engine.run();

        // Print results
        for result in completed {
            if result.success {
                println!("Output: {}", result.output_text);
                println!("Tokens generated: {}", result.output_tokens.len());
            } else {
                println!("Error: {:?}", result.error);
            }
        }
    } else {
        println!("No input provided. Use --input to specify text to process.");
        println!();
        println!("Example:");
        println!("  paged-serving --input \"Hello, world!\" --max-tokens 50");
    }

    Ok(())
}

/// 监听 Ctrl+C（以及 Unix 平台的 SIGTERM），触发后让服务器优雅关闭：
/// 停止接受新连接，排空在途请求。
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Shutdown signal received, draining in-flight requests");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_arguments_fail_on_mismatched_model_path() {
        let config = EngineConfig::default();
        let cpu_error = create_engine(
            config.clone(),
            BackendKind::Cpu,
            Some(Path::new("model.gguf")),
        )
        .err()
        .expect("CPU backend with a model path must fail");
        assert_eq!(
            cpu_error.to_string(),
            "--model-path requires --backend tiny-llm"
        );
        let tiny_error = create_engine(config, BackendKind::TinyLlm, None)
            .err()
            .expect("tiny-llm backend without a model path must fail");
        assert_eq!(
            tiny_error.to_string(),
            "--backend tiny-llm requires --model-path"
        );
    }
}
