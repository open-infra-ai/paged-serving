# Serving 结果报告模板

> 将此文件复制到 `results/<date>-<gpu>/report.md` 后填写。方括号是待填写项，不能原样发布。
> 只有 `validate_results.py --formal` 通过且以下限制写清楚后，结果才能进入 README、简历或面试材料。

## 范围与结论

- 日期 / 硬件：[GPU、显存、驱动、CUDA、CPU、发压机位置]
- 被测 commit：[paged-serving、tiny-llm、外部引擎]
- 模型与量化：[模型文件 SHA-256、量化格式、tokenizer]
- 结论：[只陈述本次矩阵实际支持的观察；不外推到其他 GPU、模型或负载]

## 正确性门控

- 服务启动与 `/health`：[命令和结果]
- 真实 CUDA 后端 canary：[请求、输出检查、退出状态]
- token 计数来源与 coverage：[usage / tokenizer_text / 不可用]
- 对照后端：[可运行、失败或不适用；失败时保留原始日志]

## 负载与结果

### Closed-loop

说明并发 1/2/4/8、数据集、重复次数、warmup 和 max tokens；引用 `summary_table.csv` 与
`ttft_by_concurrency.png`，同时报告成功率、429/错误和 token coverage。

### Poisson

说明到达率、与闭环容量的关系、数据集、重复次数和 warmup；引用 `slo_curve.png`。不要把
SSE chunk 间隔写成 ITL。

## 对照公平性与限制

- 语义审计：[命令、退出码、`audit.json` 路径；通过仅说明内部一致]
- 收敛：[各组合 `(max−min)/mean`、`non_converged` / `insufficient_repeats`；保留全部重复]
- 历史字段缺失：[seed/计划时间/dispatch 未采集及其影响，不补造]
- 各后端的模型、量化、上下文上限、并发上限和启动参数。
- W8A16、Q4_K_M、FP16 等非同量化只能叫完整路径对照，不能叫同量化性能比较。
- 未实现的 KV 利用率采样、prefix cache、抢占或 chunked prefill 不得用推测补全。
- OOM、429、连接错误、token coverage 不足和未收敛重复必须保留。

## 完整复现

```bash
# [构建、启动服务、运行 sweep、校验产物、生成图表的完整命令]
```

## 产物清单

- [ ] `metadata.json`
- [ ] 每个 run 的 `run_metadata.json`、`per_request.jsonl`、`summary.json`、`stdout.log`
- [ ] `summary_table.csv` 与图表
- [ ] `audit.json` 中的校验诊断与收敛限制（新生成图表产物，历史包不回填）
- [ ] 模型 SHA-256、双仓 commit、硬件/软件版本
- [ ] 本报告中的结论、限制和负结果
