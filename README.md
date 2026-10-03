# Paged-Serving

> 📚 Portfolio map: https://github.com/open-infra-ai/open-infra-ai

<div align="center">

[![CI](https://github.com/open-infra-ai/paged-serving/actions/workflows/ci.yml/badge.svg)](https://github.com/open-infra-ai/paged-serving/actions/workflows/ci.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.88%2B-orange?logo=rust)](https://www.rust-lang.org/)

**面向学习与验证的 LLM Serving 控制面：Paged KV 调度、Continuous Batching 与 OpenAI API**

> **开发状态**：**active**（Serving 评测口径与跨引擎验证持续进行）；控制面核心
> （分页 KV / continuous batching / 调度 / API）v0.2.0 已稳定；
> 计算后端双路径：默认 CPU 参考执行器（确定性，供测试/CI），`tiny-llm` cargo feature
> 下接入 [tiny-llm](https://github.com/open-infra-ai/tiny-llm) 真实 CUDA 后端，并已启用
> **分页 KV（策略 1：block_tables 真实上传）**——3 并发 e2e 与 llama.cpp greedy
> 逐 token 对齐、资源守恒成立。

**[文档](#文档) | [更新日志](CHANGELOG.md)**

</div>

---

## 项目概述

Paged-Serving 是一个基于 Rust 构建的 LLM Serving 控制面，以模块化、可测试的架构
练习分页 KV 内存管理与连续批处理调度。计算后端默认是 CPU 参考执行器（随机初始化
小型 Transformer，确定性输出）；`tiny-llm` feature 下接入真实 CUDA Runtime，
并把每序列 `block_tables` 上传到 tiny-llm 的分页 KV 池（策略 1）。本仓库不把
控制面、参考执行器和外部 Runtime 的组合包装成生产级推理引擎。

| 特性 | 说明 | 状态 |
|------|------|:----:|
| **Paged KV 控制面** | BlockPool、PageTable、块表上传与资源守恒；不宣称固定碎片率 | ✅ |
| **优先级调度** | `GenerationParams::priority` 高优先级先调度（同级 FCFS） | ✅ |
| **连续批处理** | 动态 prefill/decode 调度 | ✅ |
| **内存压力感知** | 可配置的 OOM 防护 | ✅ |
| **模块化架构** | 基于 Trait 的抽象设计 | ✅ |
| **OpenAI 兼容服务器** | `/v1/completions` + `/v1/chat/completions` + SSE | ✅ |
| **自动化验证** | unit、integration、server integration 与 property tests | ✅ |
| **tiny-llm 真实后端** | `tiny-llm` feature 下接入 CUDA 后端，分页 KV（策略 1）默认启用，`PAGED_SERVING_TINY_LLM_STRATEGY=2` 可回退连续 KV；正常 greedy 把各序列末层 hidden 写入 GPU batch buffer，再批量执行 final RMSNorm、LM head 与 argmax，并一次回传整批结果；Transformer layer 仍逐序列；`PAGED_SERVING_TINY_LLM_MAX_SEQS`（默认 4）与 `PAGED_SERVING_TINY_LLM_DECODE_RESERVE`（默认 512）可按显存/生成长度调节容量 | ✅ |

在五仓学习路径中，本仓库只练习 LLM Serving 控制面；真实模型权重加载与 token 计算属于 `tiny-llm`。整体顺序见 [`LEARNING_PATH.md`](https://github.com/open-infra-ai/open-infra-ai/blob/master/LEARNING_PATH.md)（meta 仓）。

## 项目边界（IN / OUT）

**IN（本仓库负责）**：
- Paged KV 控制面（BlockPool / 页表 / 资源不变量）
- continuous batching（动态 prefill/decode 调度）
- 准入控制 / 内存水位线 / 队头阻塞处理
- OpenAI 兼容 API（/v1/completions、/v1/chat/completions、SSE）
- HTTP 边界的 tokenizer **适配器**（默认 `SimpleTokenizer`；`--tokenizer` 走 HF `tokenizer.json`）
- 属性测试与资源不变量验证

**OUT（明确不做，见对应仓库）**：
- 计算 kernel（GEMM/attention/RoPE/W8A16）→ [tiny-llm](https://github.com/open-infra-ai/tiny-llm)
- 模型加载、词表与 BPE 算法权威 → [tiny-llm](https://github.com/open-infra-ai/tiny-llm)（本仓不重新实现 GGUF tokenizer；HF 路径必须与 tiny-llm fixture 逐 id 对齐）
- FlashAttention 深挖 → [cuflash](https://github.com/open-infra-ai/cuflash)

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                     InferenceEngine（控制面）                          │
├──────────────────────────────────────────────────────────────────────┤
│  Tokenizer 适配器     Scheduler              KV Cache Manager          │
│  Simple / HF JSON     状态机 + 准入          BlockPool + PageTable    │
│         │                  │                         │                │
│         └────────┬─────────┴─────────────────────────┘                │
│                  ▼                                                    │
│         GPUExecutorTrait                                              │
│           ├─ CPUExecutor          默认：确定性参考，CI / 单测            │
│           └─ TinyLlmExecutor      feature tiny-llm：策略 1 分页 KV     │
│                  │                （block_tables 经 C ABI 上传）        │
│                  ▼                                                    │
│         tiny-llm 数据面（GGUF / W8A16 / 分页 KV 池）                   │
└──────────────────────────────────────────────────────────────────────┘
```

## 调度器设计讲解

> 面试讲述用。用 2 分钟讲清状态机、准入控制与调度优先级；同时主动说明
> 当前架构练习的边界（无抢占），避免被追问时措手不及。

### 请求生命周期状态机

```
            准入通过
  ┌──────┐  add_request  ┌─────────┐   首批调度   ┌─────────┐
  │ 新请求 ├──────────────►│ Pending │─────────────►│ Prefill │
  └──────┘                └─────────┘              └────┬────┘
       │                                               │ 生成首个 token
       │ 取消/失败（任意阶段）                           ▼
       │                                               ┌─────────┐
       │         ┌─────────────────────────────────────►│ Decode  │
       │         │     每步生成 1 token                 └────┬────┘
       │         │                                          │
       ▼         ▼                                          │ EOS / stop / max_tokens
  ┌─────────────────┐                                       ▼
  │ Failed（释放KV）│                                ┌─────────────┐
  └─────────────────┘                                │  Completed   │
                                                     └─────────────┘
```

- **Pending**：已通过准入、等待首个调度步。仅占一个序列槽位，不占 KV 块。
- **Prefill**：一次性处理整个 prompt，计算首个输出 token，分配 KV 块。
- **Decode**：每步生成一个 token（自回归），KV 逐块增长。
- **Completed / Failed**：终态。请求移出调度器，KV 块全部归还自由池。

### 准入控制（三层）

1. **请求级校验**（`submit_request` 阶段）：
   - 参数合法（`max_tokens`、greedy 采样、`stop` ≤ 4、`logprobs` ≤ 5）
   - 总长度不超 `max_model_len`
   - 并发序列不超 `max_num_seqs`（否则 `MaxConcurrentSequencesReached` → HTTP 429）
2. **批次预算**（每步 `schedule`）：
   - 序列数 ≤ `max_batch_size`
   - 本步总 token ≤ `max_total_tokens`（decode 每序列 1，prefill 按输入长度）
   - 单请求块需求 ≤ `max_num_blocks`（超出直接失败，而非悄悄截断）
3. **内存压力**：KV 池利用率 ≥ `memory_threshold` 时，**新提交的请求直接拒绝**
   （`MemoryPressure` → HTTP 429 + `Retry-After`）；已解码序列继续推进；
   每步启动新 prefill 前还检查高水位线与"下一步 decode 增长"的预留块，
   预算不足的候选延后到后续步骤，而不是把池子打满导致 OOM。

### 每步调度优先级

```
1. Decode 序列   —— 优先，降低在途请求的尾延迟
2. Prefill 序列  —— 已开始 prefill 的继续推进
3. Pending 队列  —— 新请求（非内存压力下），先来先服务
```

### 抢占策略与边界

**当前实现没有抢占**（vLLM 式的 swap / preempt-resume 未实现）。遇到内存
压力时的策略是"拒绝新 prefill、保住已解码序列"，而非驱逐旧序列。这是本仓库
（架构练习）的明确边界——面试时主动说明，并解释真实系统如何用
`swap`（KV 换出到 CPU 内存）与 `preempt-resume`（按 sequence group 抢占）
应对长尾负载。

### 资源不变量

- KV 块池恒满足 `used_blocks + free_blocks == total_blocks`
- 任何终止路径（完成 / 取消 / 失败 / 客户端断开）都归还 KV 块，内存利用率
  回到基线 —— 由穷举属性测试覆盖

## 快速开始

### 环境要求

- **Rust 1.88+** (2021 edition；以 `Cargo.lock` 为复现依赖集，CI 在 1.88.0 验证全部默认目标)
- **Linux** (推荐 Ubuntu 20.04+) 或 **macOS**

### 安装

```bash
# 克隆仓库
git clone https://github.com/open-infra-ai/paged-serving.git
cd paged-serving

# 以 release 模式构建
cargo build --release

# 运行测试套件
cargo test
```

### 命令行用法

> 默认 `SimpleTokenizer` 只支持 ASCII；中文等非 ASCII 文本请配合
> `--tokenizer <tokenizer.json>` 使用 HuggingFace tokenizer（否则会变成 UNK）。

```bash
# 基本用法
./target/release/paged-serving --input "Hello, world!" --max-tokens 50

# 使用自定义参数（当前 CPU 后端仅支持 greedy：--temperature 0.0 --top-p 1.0，
# 其他采样参数会在提交时返回错误，而不是被静默忽略）
./target/release/paged-serving \
  --input "Explain quantum computing" \
  --max-tokens 100

# 启动 OpenAI 兼容 HTTP 服务
./target/release/paged-serving --serve

# 启动 tiny-llm 真实 CUDA 后端；显式选择避免压测时误用 CPU reference
TINY_LLM_DIR=../tiny-llm/build cargo build --locked --release --features tiny-llm
./target/release/paged-serving --serve --backend tiny-llm \
  --model-path ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokenizer ../models/tokenizer.json
```

### OpenAI 兼容服务

```bash
# 启动服务，默认地址 127.0.0.1:3000
cargo run -- --serve

# 健康检查 / 就绪检查 / 指标
curl http://127.0.0.1:3000/healthz
curl http://127.0.0.1:3000/readyz
curl http://127.0.0.1:3000/metrics

# Completions 接口
curl http://127.0.0.1:3000/v1/completions \
  -H "content-type: application/json" \
  -d '{"model":"paged-serving","prompt":"Hello","max_tokens":8}'

# Chat Completions 接口
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"paged-serving","messages":[{"role":"user","content":"Say hello"}],"max_tokens":8}'
```

### 指标（/metrics，Prometheus 格式）

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `paged_requests_total` | counter | 累计 HTTP 请求数 |
| `paged_errors_total` | counter | 累计错误响应数 |
| `paged_inflight_requests` | gauge | 当前在途 HTTP 请求数 |
| `paged_streaming_requests_total` | counter | 累计流式请求数 |
| `paged_engine_active_sequences` | gauge | 引擎当前活跃序列数 |
| `paged_engine_kv_utilization` | gauge | KV 块池利用率（0.0–1.0） |
| `paged_engine_completed_requests` | counter | 累计成功完成请求数 |
| `paged_engine_failed_requests` | counter | 累计失败请求数 |
| `paged_engine_tokens_generated_total` | counter | 累计生成 token 数 |

### 库用法

```rust
use paged_serving::{EngineConfig, GenerationParams, InferenceEngine};

// 使用默认配置创建引擎
let mut engine = InferenceEngine::new(EngineConfig::default())?;

// 提交生成请求（greedy 解码，当前后端唯一支持的模式）
// 注意：默认 SimpleTokenizer 仅支持 ASCII，中文请改用 HuggingFace tokenizer。
let request_id = engine.submit_request(
    "Hello, world!",
    GenerationParams {
        max_tokens: 100,
        ..GenerationParams::default()
    }
)?;

// 运行推理并收集结果
let results = engine.run();
for result in results {
    println!("生成结果: {}", result.output_text);
}
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--block-size` | 16 | 每物理块 token 数 |
| `--max-num-blocks` | 1024 | 物理块总数 |
| `--max-batch-size` | 32 | 每批次最大序列数 |
| `--max-num-seqs` | 256 | 最大序列数 |
| `--max-model-len` | 2048 | 最大模型上下文长度 |
| `--max-total-tokens` | 4096 | 每批次最大 token 总数 |
| `--backend` | `cpu` | 执行后端：`cpu` 或编译 feature 后可用的 `tiny-llm` |
| `--model-path` | 无 | `tiny-llm` 后端使用的 GGUF；选择该后端时必填 |
| `--memory-threshold` | 0.9 | 内存压力阈值 (0.0-1.0) |
| `--max-tokens` | 100 | 最大生成 token 数 |
| `--temperature` | 0.0 | 采样温度；CPU 后端仅支持 0.0（greedy），其他值提交时返回错误 |
| `--top-p` | 1.0 | 核采样阈值；CPU 后端仅支持 1.0，其他值提交时返回错误 |
| `--tokenizer` | 无 | HuggingFace tokenizer.json 路径；设置后引擎改用 HF tokenizer（完整有效词表 151665；GGUF embedding 可能为 151936 并含 padding 行），替代默认的 SimpleTokenizer |

配置文件 (`config.json`):

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "max_total_tokens": 4096,
  "memory_threshold": 0.9,
  "max_retry_attempts": 2
}
```

加载：`./paged-serving --config config.json`

## 文档

| 资源 | 链接 |
|------|------|
| **API 文档** | `cargo doc --open` |
| **贡献指南** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **更新日志** | [CHANGELOG.md](CHANGELOG.md) |

## 性能边界

默认计算后端是 CPU 参考执行器（随机权重小模型），因此它只用于测试、CI 和协议/调度
回归；不能产生真实 token 吞吐或 GPU 利用率结论。`tiny-llm` feature 已接入真实 CUDA
后端与分页 KV（策略 1），目前 3 并发 e2e 只证明跨语言生命周期与 greedy 输出正确性。

首份真实 CUDA Serving 结果已归档于
[`benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving/`](benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving/)。
它绑定硬件、双仓 commit、模型 SHA-256、逐请求记录和负结果；在 RTX 3060 Laptop 6GB
上，closed-loop 并发 1–8 的吞吐约为 82 tok/s 且未随并发扩展，Poisson 约 1.0x 饱和
请求容量时已出现 429。该结果只适用于所归档的模型、硬件和负载，不能外推为通用容量或
生产成熟度结论。评测入口与产物要求见
[`benchmarks/serving/README.md`](benchmarks/serving/README.md)。

这份 P1 归档采集于 HF 流式解码修复之前，因而只能作为调度/后端饱和的历史基线，不能
用于描述当前代码的流式 TTFT 或 TPOT。当前流式语义的 21-run P2 矩阵已归档于
[`2026-09-04-RTX3060Laptop-paged-serving-p2-streaming/`](benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving-p2-streaming/)；
它首次记录真实首文本 TTFT、TPOT 与 inter-chunk 分布，并将 Poisson 种子写入双层元数据。
其中 closed c1/c2/c4 与全部 Poisson 档的 TTFT p95 重复波动超过 10%，所以它是当前
路径的可追溯边界证据，不是精确 SLO 或 P1→P2 速度提升声明。该矩阵也早于当前的批量
末端后处理：ABI 虽可一次接收多个序列，但
[`tinyllm_step`](https://github.com/open-infra-ai/tiny-llm/blob/master/src/ffi.cpp)
的 Transformer layer forward 仍逐序列推进。正常 greedy 仅在每序列末层 hidden 写入
GPU batch buffer 后，批量执行 final RMSNorm、LM head 与 argmax，并在 step 末尾一次回传
token id；`logprobs` 仍走主机完整 logits / top-k 路径。因此 continuous batching 在控制面
语义上成立，但尚不是 fused compute batch。该正确性改动尚未重采 serving 矩阵，不能据此
声明吞吐或 TTFT 改善；下一项性能工作是逐层 batch decode，并以新的原始结果包验证。
当前干净提交还有一份 [closed c=4 HTTP 功能 canary](benchmarks/serving/results/2026-09-05-RTX3060Laptop-paged-serving-p2-batch-postprocess-canary/)，
其 4 个 smoke 请求均成功；它是可运行性证据，不替代重复性能矩阵。

批量末端后处理后的当前干净提交已重新采集
[21-run 正式矩阵](benchmarks/serving/results/2026-09-07-RTX3060Laptop-paged-serving-p2-batch-postprocess-streaming/)。
它绑定 RTX 3060 Laptop 6GB、模型 SHA-256、双仓 commit、21 份逐请求记录、固定 Poisson
种子、CSV 与图表：closed-loop 的 12 个 run 均为 64/64 成功，Poisson 0.64 / 1.28 req/s
分别累计出现 9 / 74 个 HTTP 429。closed c1/c4 的 TTFT p95 和吞吐通过 10% 重复波动检查，
而 c2/c8 与三档 Poisson 仍存在未收敛指标；因此该包是当前路径的可追溯边界，不是稳定 SLO、
通用容量或批量末端后处理的 before/after 速度提升。Transformer layer forward 仍逐序列，
故也不能将它表述为 fused compute batch。

### 流式（SSE）与分词器

- `SimpleTokenizer` 对每个可见 token 直接产生一个 SSE 文本片段。
- HuggingFace tokenizer 使用 tokenizers 0.21 的官方逐步流式 decode 状态机：BPE、
  WordPiece 和 byte-fallback 只有在文本能安全追加时才产生片段；特殊 token 或不完整
  UTF-8 可暂不产生片段。所有片段拼接严格等于一次性 decode，且无需等到请求结束。
  因此 TTFT 定义为首个**非空文本**片段，而不是任意 token 或 HTTP 响应头。

### 内存压力与无抢占

本项目**没有抢占**（vLLM 式的 swap / preempt-resume 未实现）。内存压力下的
策略是：

1. **拒绝新 prefill**（`add_request` 提交侧：利用率 ≥ 阈值时返回 `MemoryPressure`）；
2. **保留在途 decode**（已开始的序列继续推进，不驱逐）；
3. **预留即时 decode 增长块**：启动新 prefill 前既检查高水位线
   （启动后 `used_blocks / total_blocks ≤ memory_threshold`），也为
   "本步已调度/在跑序列 + 候选序列"下一步的 decode 增长预留空闲块。

该策略只解决"下一步马上需要增长"的 OOM；长期最坏情况（大量长序列同时需要
多个增长块）仍可能 OOM——这是无抢占实现的固有边界。

### Chat Completions 的 chat template

- 使用 `SimpleTokenizer` 时，`prepare_chat_request` 保持简单的 `role: content`
  文本拼接。
- 使用 HuggingFace tokenizer 时，应用 **Qwen2 的 chat template**
  （`<|im_start|>` / `<|im_end|>`，末尾追加 `<|im_start|>assistant`），
  与 Qwen2 系模型词表对齐。当前模板是**硬编码 Qwen2** 的，其他模型需扩展
  `build_chat_prompt`。

### 为什么选择 PagedAttention？

传统 LLM 服务为每个请求的 KV 缓存分配连续内存块，导致严重的内存碎片和浪费。PagedAttention 通过以下方式解决：

1. **块级分配**：将 KV 缓存分割为固定大小的块
2. **按需分页**：仅在需要时分配块
3. **写时复制**（尚未实现，未来方向）：跨序列共享块，实现高效的 beam search

## 测试

```bash
# 运行所有测试
cargo test

# 运行覆盖率测试
cargo llvm-cov --html

# 运行属性测试
cargo test -- --test-threads=1
```

| 类型 | 覆盖范围 | 说明 |
|------|:--------:|------|
| 单元测试 | 核心模块 | 分页、调度、执行和配置 |
| 属性测试 | 状态不变量 | 资源守恒、队列唯一性和容量上限 |
| 集成测试 | 端到端工作流 | engine 与请求生命周期 |
| Server 集成 | HTTP/SSE | API、取消、健康检查与指标 |

## 贡献指南

欢迎贡献！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

```bash
# 提交前运行所有检查
cargo test && cargo fmt --check && cargo clippy
```

## 当前路线

- [x] PagedAttention KV Cache
- [x] Continuous Batching 调度器
- [x] 内存压力感知
- [x] 属性测试
- [x] OpenAI 兼容 HTTP 服务
- [x] CPU 参考执行器（paged KV cache + transformer 前向）
- [x] HuggingFace Tokenizer 集成
- [ ] 完成可信的 closed-loop / Poisson 跨引擎评测与原始数据归档
- [ ] 把调度、KV cache 与评测经验转化为上游社区贡献

## 当前阶段：核心稳定，评测仍在推进

本项目的 v0.2.0 控制面核心已经稳定，P0 正确性修复（T0–T8）与部分 P1
（T9、T10、T12）已经完成；仓库保持 active，是因为 serving 评测工具和跨引擎
证据仍在完善。以下功能边界继续冻结：

- **无抢占**（无 swap / preempt-resume）
- **无 chunked prefill**、**无 prefix caching**
- **不拥有 CUDA kernel**（真实 kernel 属于 tiny-llm；本仓只通过 C ABI 调度）
- 不发布缺少可信 token 计数、完整墙钟、硬件和 commit 绑定的吞吐数字

推理加速主线位于 [tiny-llm](https://github.com/open-infra-ai/tiny-llm)；本仓库只负责
把 Runtime 能力置于真实请求、调度和 KV 生命周期中验证。FlashAttention 的独立
kernel 学习仍在 [cuflash](https://github.com/open-infra-ai/cuflash)。

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)。

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention 概念和灵感来源
- [Rust](https://www.rust-lang.org/) - 系统编程语言
- [Criterion](https://github.com/bheisler/criterion.rs) - 统计基准测试

---

<p align="center"><b>由 open-infra-ai 用 ❤️ 构建</b></p>
