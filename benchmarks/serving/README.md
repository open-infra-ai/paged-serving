# Serving 评测体系

对 paged-serving（及对照基线 llama-server / vLLM）做可复现的 serving 级
负载实验。**指标口径与方法论的唯一权威是
[`methodology.md`](methodology.md)**——任何数字引用必须与其口径一致。

## 目录结构

```
benchmarks/serving/
├── README.md              # 本文件：入口与运行方法
├── methodology.md         # 指标口径 + 实验协议（唯一权威）
├── datasets/
│   └── synth/
│       ├── gen_synth.py   # 三档合成分布生成器（固定种子）
│       ├── smoke.jsonl    # 3 条冒烟 prompt
│       └── {short,work,long}.jsonl   # gen_synth.py 产出
├── run_sweep.sh           # 矩阵编排：dirty 检查 + 双层 metadata + loadgen
├── plots.py               # 语义校验后读取 summary，生成图表与 audit.json
├── validate_results.py    # 原始请求/summary/metadata 一致性与收敛诊断
├── test_validate_results.py # 标准库离线正/负夹具门禁
├── RESULT_REPORT_TEMPLATE.md # 人工结论、限制与复现命令模板
└── results/<date>-<gpu>/  # 原始请求 + run summary + 环境/模型 metadata + 图表
```

压测客户端是 [`src/bin/loadgen.rs`](../../src/bin/loadgen.rs)（闭环饱和 /
开环泊松双模式，同一二进制零改动覆盖三个后端）。

## CLI 回归验证

```bash
cargo test --locked --test loadgen_cli
```

测试启动真实 loadgen 子进程和本地 TCP/SSE 夹具服务器，联合核对 CLI 参数、warmup、
closed/Poisson 发压、原始 JSONL 和 summary。相同 seed 的计划及测量 prompt 顺序
不受预热次数影响；未知 token 保留 null，错误详情保留，tok/s 只在成功请求 token
coverage 完整时输出。计划/实际 dispatch 字段的意义见 [方法论](methodology.md)。
这组测试不需要 GPU，不是 CUDA serving 性能结果，也不验收生产服务端回收或网络压力。

## 结果语义回归验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
    -s benchmarks/serving -p 'test_*.py' -v
```

标准库测试覆盖原始请求、聚合数字和配置被损坏后的拒绝行为，不需要 GPU 或 matplotlib。
CI 单独运行此门禁并只读重验历史正式结果；不会因为 429 或 `non_converged` 而隐藏负结果。

## 快速开始

```bash
# 0. 构建（仓库根目录）
cargo build --locked --release --bin loadgen
# 真实后端服务（CPU 参考后端可直接 --serve）
TINY_LLM_DIR=../tiny-llm/build cargo build --locked --release --features tiny-llm

# 1. 生成数据集
python3 benchmarks/serving/datasets/synth/gen_synth.py \
    --outdir benchmarks/serving/datasets/synth

# 2. 启动被测服务（三选一）
# paged-serving（tiny-llm 真实后端）：
#   PAGED_SERVING_TINY_LLM_MAX_SEQS=8 ./target/release/paged-serving --serve \
#       --backend tiny-llm --model-path <model.gguf> \
#       --port 3000 --tokenizer <tokenizer.json>
# llama-server（基线）：
#   llama-server -m <model.gguf> -c 2048 --parallel 8 --cont-batching --port 8080
# vLLM（若显存允许）：
#   vllm serve Qwen/Qwen2.5-0.5B-Instruct --enforce-eager \
#       --gpu-memory-utilization 0.75 --max-model-len 2048 --port 8000

# 3. 冒烟（3 条 prompt，验证管线连通）
./target/release/loadgen --base-url http://127.0.0.1:3000 \
    --engine paged-serving --model paged-serving \
    --mode closed --concurrency 2 \
    --dataset benchmarks/serving/datasets/synth/smoke.jsonl \
    --requests 4 --warmup-secs 0 --max-tokens 16 --out /tmp/smoke.jsonl

# 4. 完整矩阵（默认闭环并发 1/2/4/8 × work 分布 × 3 次重复）
cd benchmarks/serving
./run_sweep.sh --base-url http://127.0.0.1:3000 --engine paged-serving \
    --model paged-serving --model-path ../../../models/<model>.gguf \
    --backend-quant W8A16 --tokenizer <tokenizer.json> --cuda-archs 86 \
    --poisson-seed 20260904

# 5. 检查产物并生成图表
python3 validate_results.py results/<date>-<gpu>/
python3 plots.py results/<date>-<gpu>/

# 6. 准备发布正式结果前（要求填写 report.md、模型 SHA-256 和汇总图表）
cp RESULT_REPORT_TEMPLATE.md results/<date>-<gpu>/report.md
python3 validate_results.py --formal results/<date>-<gpu>/
```

重验历史结果时把新图表写到独立目录，保留历史 CSV/PNG：

```bash
python3 validate_results.py --formal --json results/<date>-<gpu>/
python3 plots.py results/<date>-<gpu>/ --out-dir /tmp/serving-reaudit
```

`--json` 只向 stdout 输出机器可读诊断，不修改结果包。数值/配置矛盾退出 1，参数错误
退出 2；校验成功退出 0 只表示内部一致，不代表收敛、GPU correctness 或稳定 SLO。
绘图在校验通过后才写产物，CSV 附带引擎 commit、模型 SHA、量化和收敛状态；
`audit.json` 保留逐组合诊断。任一重复缺少 token 吞吐时，整个组合不平均已知子集。

## 结果索引

| 日期 | 硬件 | 内容 | 目录 |
|------|------|------|------|
| 2026-09-04 | RTX 3060 Laptop 6GB | paged-serving + tiny-llm 真实 CUDA：21 个 run 的 P1 基线；含吞吐平台、429 与流式限制 | [正式结果](results/2026-09-04-RTX3060Laptop-paged-serving/) |
| 2026-09-04 | RTX 3060 Laptop 6GB | P2 HuggingFace 流式与 Poisson 种子功能 canary；`n=1`，不含性能结论 | [功能证据](results/2026-09-04-RTX3060Laptop-paged-serving-p2-stream-canary/) |
| 2026-09-04 | RTX 3060 Laptop 6GB | P2 当前流式语义：21 个 run，真实 TTFT / TPOT / inter-chunk；重复收敛限制已写入报告 | [正式结果](results/2026-09-04-RTX3060Laptop-paged-serving-p2-streaming/) |
| 2026-09-05 | RTX 3060 Laptop 6GB | P2 批量末端后处理后的真实 CUDA HTTP canary：closed c=4，4/4 成功；`n=1`、无预热，不含性能结论 | [功能证据](results/2026-09-05-RTX3060Laptop-paged-serving-p2-batch-postprocess-canary/) |
| 2026-09-07 | RTX 3060 Laptop 6GB | P2 批量末端后处理后的真实 CUDA HTTP：21 个 run；closed 均全成功，Poisson 0.64/1.28 出现 429，重复收敛限制已写入报告 | [正式结果](results/2026-09-07-RTX3060Laptop-paged-serving-p2-batch-postprocess-streaming/) |

已有 P1/P2 正式报告都只覆盖 paged-serving；下一份**跨引擎**正式报告必须同时覆盖：

- 正确性 canary 后的真实 CUDA 后端；
- closed-loop 并发 1/2/4/8 与 Poisson 到达率矩阵；
- `paged-serving`、`llama-server`、可运行时的 vLLM 使用同一 loadgen、数据集与请求参数；
- 每个数字绑定硬件、驱动、双仓 commit、模型 SHA-256、量化格式和原始请求数据；
- 失败、OOM、429、无法启动的对照与 token coverage 不足均写进 `report.md`。

`validate_results.py` 检查数据一致性和必需产物，不证明声明的硬件/模型实际参与执行，
也不评判性能优劣；只有人工填写 `report.md` 的结论和限制后，结果才可被 README、
简历或面试材料引用。历史可选字段缺失会显式诊断，不回填为零或随机 seed。

## 纪律提醒（摘要，全文见 methodology.md）

- 无实测不写数字；每个数字绑定双仓 commit + 硬件 + 复现命令
- 失败归类不掺水：无 `[DONE]` 判失败；429 单独计数
- TTFT 从请求发送前开始计时；SSE chunk 间隔不冒充 token 级 ITL
- usage 缺失时必须提供 tokenizer；token coverage 不足则 tok/s 留空
- 量化格式差异（W8A16 vs Q4_K_M vs FP16）必须在表头声明
- 负结果归档（vLLM 跑不起来也是结果）
- 笔记本卡散热/功耗墙写进口径，不外推
