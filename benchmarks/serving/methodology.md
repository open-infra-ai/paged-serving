# Serving 评测方法论（指标口径唯一权威）

本文件定义 `benchmarks/serving/` 全部实验的指标口径与实验协议。
任何进入 README / 结果页 / 面试材料的数字，口径必须与此一致；
不一致的数字禁止引用。

通用测量纪律（环境模板 / 复现要求 / 无实测不写数字）沿用
[`tiny-llm/docs/performance/benchmark-methodology.md`](https://github.com/open-infra-ai/tiny-llm/blob/master/docs/performance/benchmark-methodology.md)
§1-§3，本文不复制，只定义 serving 层增量。

## 1. 指标定义（客户端侧，loadgen 产出）

| 指标 | 口径 | 分位 |
|------|------|------|
| **TTFT** | 客户端开始发送请求 → 首个**非空文本** SSE chunk 的墙钟。包含连接、排队、prefill 与响应头时间；空文本前导帧不计，口径是用户可观察的首段文本延迟 | p50/p95/p99 |
| **inter-chunk latency** | 同一 SSE 流相邻非空文本 chunk 的到达间隔。它只描述传输粒度，**不是 ITL** | p50/p95/p99 |
| **ITL** | 相邻输出 token 的到达间隔。OpenAI SSE chunk 可能聚合多个 token，协议又不提供 token 级时间戳，因此当前跨引擎 loadgen 将其明确记为不可用，不从 chunk 猜测 | 当前不产出 |
| **TPOT** | 逐请求 `(请求总耗时 − TTFT) / (输出 tokens − 1)`；仅统计具有可信 `completion_tokens` 且 tokens > 1 的成功请求，并报告样本覆盖 | p50/p95/p99 |
| **生成吞吐** | 完整测量窗口内 Σ输出 tokens / Δt（客户端侧）。只有所有成功请求都有可信 token 计数时才产出，否则为 `null`，禁止用已知子集外推 | 均值 |
| **请求吞吐** | 稳态窗口内 completed req/s | 均值 |
| **成功率** | 完成（200 + `[DONE]`）/ 提交。失败归类：`timeout` / `http_429`（内存压力或并发上限的配置性拒绝，单独计数）/ `http_4xx` / `http_5xx` / `connection` / `stream_error`（SSE 内 error 载荷）/ `no_done`（流提前结束，即使已收到部分 chunk 也判失败——成功率不掺水） | 计数 |
| **KV 利用率** | 计划从 `/metrics` 定时采样 `paged_engine_kv_utilization`；当前 sweep 尚未实现采样器，不能声称已有结果 | 待实现 |
| **调度延迟** | Criterion Mock 后端纯调度路径已经存在；服务侧 step duration 指标尚未接入 | 分布 / 待实现 |

**completion_tokens 来源**：优先最终帧 `usage.completion_tokens`
（`tokens_source="usage"`）；缺失且显式传入 `--tokenizer <tokenizer.json>` 时，
对完整输出文本重新分词（`tokens_source="tokenizer_text"`）。后者不包含未解码的
EOS 等特殊 token，必须在报表中标注。未提供 tokenizer 时 token 数与 tok/s 留空，
绝不以 chunk 数代替。

## 2. 负载模型（两种都必须在档）

1. **闭环饱和**（`--mode closed`）：固定并发槽 1/2/4/8，一个请求完成立即补发。
   测最大吞吐与尾延迟。
2. **开环泊松**（`--mode poisson`）：到达率 λ 按指数间隔，请求独立并发。
   λ 取 0.5×/1.0×/2.0× 饱和容量，画 **TTFT p95 vs λ 的 SLO 曲线**——
   这是 serving 岗位最常问的负载形态。

每个 Poisson run 都必须记录实际 `arrival_seed`：直接调用 `loadgen` 时，省略
`--seed` 会生成随机种子并写入 `summary.json`；`run_sweep.sh` 默认以
`20260904 + repeat - 1` 生成可复现种子，同时写入 `summary.json` 和
`run_metadata.json`。种子固定的是计划到达间隔；GPU 时钟、操作系统调度和网络抖动仍会使
实际完成时刻存在波动，不能把它误说成完全确定性实验。

测量窗口从 `arrival_seed` 重新初始化 RNG，Poisson 使用相对测量起点的累积
绝对 deadline，而不是每次发压后再相对 sleep。预热消耗不推进测量窗口的 RNG；两个模式
都按 `measured_index` 选测量 prompt，不能让预热请求数改变正式数据集顺序。
deadline 落后时照常发出到期请求，不丢请求、不等待前一响应；这可能形成迟到后的集中发压，
因此必须查看实际 dispatch，不能把目标 λ 当作已实现的服务端到达率。

`per_request.jsonl` 的 `scheduled_arrival_ms` 是 Poisson 的计划时间，closed 为 null；
`dispatch_offset_ms` 是开始执行该请求的客户端时间。两者均相对测量起点，后者包含客户端
调度抖动，但不是服务器收到请求的时间。`summary.config.arrival_schedule` 的
`absolute_deadline_seed_reset` 标记这一执行口径；closed 为 null。
这些是 schema v1 的可选扩展，历史记录缺字段时按“未采集”解释，禁止回填或猜测。
与没有该标记的历史负载比较时需重新配对实验，不将调度方式变化解释成服务端加速。

闭环与开环回答不同问题：闭环给"上限"，开环给"给定到达率下的延迟代价"。
只报闭环是 serving 评测的常见缺陷，本体系两者强制并列。

## 3. 数据集

| 数据集 | 用途 |
|--------|------|
| `datasets/synth/{short,work,long}.jsonl` | 受控实验：三档输入长度分布（32-64 / 128-256 / 512-1024 token 级），固定种子可复现；`prompt_tokens` 为 1.35 token/word 估算值，仅用于分布描述 |
| `datasets/synth/smoke.jsonl` | 冒烟：3 条手工 prompt，验证管线连通 |
| ShareGPT-1000 子集（W2 引入） | 真实性交叉验证：长尾输入分布最能暴露调度问题（大 prefill 挤兑小请求）；若获取受限，以合成分布为准并声明 |
| 重复 prefix 集（W5 引入） | prefix caching 专用：同一 system prompt × 200 条不同问句 |

统一参数：`max_tokens=128`、greedy（`temperature=0`，全链路一致）、
条目上限 prompt tokens ≤ `max_model_len − max_tokens`。

## 4. 实验协议

1. **三件套绑定**：实验根 `metadata.json` 记录 paged-serving/tiny-llm commit、
   `nvidia-smi` 快照、驱动/CUDA 与构建口径；每次 run 的
   `run_metadata.json` 记录被测引擎 commit、模型路径、量化格式、完整负载参数和
   （Poisson 模式的）`arrival_seed`。
   `run_sweep.sh` 默认拒绝 dirty worktree（`--allow-dirty` 显式放行并在根
   metadata 记录 `dirty: true`）。
2. **预热**：`--warmup-secs ≥ 30`（warmup 流量与测量窗口同负载形态，结果丢弃）。
3. **重复与收敛**：每个 (并发, 分布) 组合跑 3 次，报告均值与 min/max 波动；
   按 `(max − min) / mean > 10%` 视为未收敛，须排查（笔记本卡散热降频是常见原因，
   写入结果说明而非隐藏）。
4. **横向可比**：三个后端（paged-serving / llama-server / vLLM）使用
   同一 `loadgen` 二进制、同一数据集、同一参数矩阵；量化格式差异
   （W8A16 vs Q4_K_M vs FP16）必须在结果表头声明——比值是完整路径差，
   不是同量化对比。
5. **硬件口径**：所有数字标注硬件（如 RTX 3060 Laptop 6GB / 驱动 / CUDA）。
   笔记本卡的功耗墙与散热限制写进口径声明，不外推到桌面/数据中心卡。
6. **负结果归档**：vLLM 启动失败、某并发档 429 风暴、偏差未收敛——
   全部作为结果归档（命令 + 输出 + 原因），不改写不隐藏。

## 5. 结果归档结构

```
results/<date>-<gpu-slug>/
├── metadata.json        # 三件套 + 参数矩阵（schema 见下）
├── report.md            # 人工结论、限制、负结果与完整复现命令（正式结果必需）
├── <engine>_<mode>_c<N|rate>_<dataset>_r<N>/
│   ├── run_metadata.json   # 本后端 commit、模型/量化、负载参数
│   ├── per_request.jsonl   # loadgen 逐请求记录
│   ├── summary.json        # 权威墙钟、分位、coverage 与吞吐
│   └── stdout.log          # 人类可读运行日志
├── *.csv                # 跨组合汇总表（plots.py 生成）
└── *.png                # 图表（每张带 commit+日期+硬件 caption）
```

`metadata.json` schema：

```json
{
  "schema_version": 1,
  "date": "2026-08-30",
  "hardware": {"gpu": "RTX 3060 Laptop", "vram": "6144 MiB", "driver": "…"},
  "software": {"cuda_toolkit": "Cuda compilation tools, release 12.0, …"},
  "commits": {"paged_serving": "sha", "tiny_llm": "sha", "dirty": false},
  "build": {"profile": "release", "cuda_archs": "86"}
}
```

后端、模型、量化和具体矩阵属于单次 run，写入各子目录的
`run_metadata.json`；这样同一实验根目录可以安全容纳多个后端，根 metadata
不会因第二次 sweep 被某个后端的 URL/模型覆盖。

`run_metadata.json` 的 `model.sha256` 是本地模型文件的 SHA-256；正式报告缺少该值时，
不得用“同名模型”或远程 revision 替代。远程模型须先固定到本地文件并记录文件哈希，
再进入正式对照。

运行 `python3 validate_results.py <result-root>` 检查基础产物；发布前运行
`python3 validate_results.py --formal <result-root>`。两种模式均联合重算 JSONL 与 summary，
检查完整测量序号、唯一 request_id、成功/失败与错误分类、token 来源与 coverage、
样本数和分位、成功请求数/输出 token 总量除以 `measurement_wall_secs` 的吞吐。
失败流可携带部分文本与 token，但不进入成功性能样本或成功 token 总量。
分位选取与 loadgen 一致：排序后取 `round(p/100 × (n−1))`，非负索引的 0.5 向上舍入；
浮点指标重算容差为 `rel_tol=1e-9`、`abs_tol=1e-6`，整数计数和配置参数必须完全一致。

目录标签、run metadata 和 summary.config 必须一致；同一引擎/模式/数据集的曲线只允许
并发或 rate 改变，模型文件/量化、commit/dirty、数据集路径、请求数、预热、超时、
tokenizer 和到达执行口径保持一致。Poisson 的 seed 可随重复变化，但各 run 的声明要一致。
旧 Poisson seed、计划/dispatch 字段未采集时诊断为 `legacy_*_unavailable`；不回填，
不能据此声称计划负载可复现。带新到达执行标记的结果必须提供实际 seed 与完整时间字段。

`--formal` 额外要求至少三次完整重复、预热至少 30 秒、完整双仓/引擎 commit、模型
SHA-256 和非空报告/CSV、至少两张 PNG。CSV/PNG 与人工结论仅检查产物存在性；
声明的 SHA 与环境也不等于已核验文件内容或真实 GPU 执行，仍需原始运行与 correctness 证据。

`--json` 输出校验错误、历史限制和各组合 TTFT p95、token/request 吞吐的收敛状态。
完整三次以上重复采用上述 10% 门槛；全零的相对波动为 0；无可用值为 `unavailable`，
可用重复少于三次或覆盖不完整为 `insufficient_repeats`，不得声称已收敛。
`non_converged`、失败请求或低成功率是有效结果，不导致校验失败；数据矛盾、缺少正式
门槛或配置混用才返回非零。校验通过不自动批准性能结论或稳定 SLO。

## 6. 图表规范（plots.py）

- TTFT p95 vs 并发折线（三引擎/数据集分系列，阴影为重复 min/max）
- 输出 token 吞吐 vs 并发折线（token coverage 100% 才绘制）
- SLO 曲线：TTFT p95 vs λ（泊松档）
- 每张图 caption：`<engine> @ <commit>, <date>, <gpu>`
- 绘图复用语义校验，写文件前拒绝坏数据与不兼容系列；`--out-dir` 支持历史只读重验。
- CSV/图例声明被测引擎 commit 和量化，标题中的 loadgen commit 是发压代码来源。
  `audit.json` 与 CSV 的收敛列保留 `non_converged`；任一重复的 token 吞吐不可用时，
  整个组合的平均吞吐与 min/max 留空，不择取已知子集。

CUDA Graph on/off 的配对 TPOT 图属于 tiny-llm 的 engine 层报告，不混入 serving 曲线。
KV 利用率图属于后续能力，只有采样器真正实现并归档原始数据后才加入本规范。
