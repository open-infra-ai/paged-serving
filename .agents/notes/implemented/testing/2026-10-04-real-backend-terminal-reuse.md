# Agent Note: 真实后端终态后复用与释放通知负对照

Status: implemented

## Problem

终态后的逻辑 KV 利用率为零，不等于后端仍能接收完整下一批请求。既有越界回归
只查看低于 5% 的计数，没有继续复用同一实例；真实 GPU 取消缺少终态后再服务的
验证。分页 KV 的序列登记与连续 KV 的固定槽位也不能共用一个泄漏判据。

## Decision

测试在同一后端实例上验证正常完成、越界失败、prefill 后取消和 decode 后取消，
每个异常终态后继续提交四请求批次。检查逻辑资源精确归零、取消独立分类与终态
恰好一次，并分别在策略 1/2 的隔离进程运行。

测试专用 wrapper 将计算委托真实 TinyLlmExecutor，但故意不转发释放通知。
连续 KV 的后续满容量批次必须因分配失败而被测试识别；分页 KV 的后续批次仍能
成功，作为反例证明此探针不能鉴别其序列登记泄漏。生产实现与 C ABI 不改动。

[真实测试执行语义](../../implemented/testing/2026-10-04-real-test-execution-gates.md)
只部分重叠，继续维护显式输入与执行计数；
[取消所有权](../../implemented/feature/2026-10-04-bounded-events-and-cancellation.md)
只部分重叠，继续维护服务层 CPU 控制面。本笔记不关闭 HTTP 断连、持续 GPU CI
或完整 L3 设计审批，不将同步 step 之后的取消称作 kernel 中途抢占。

## Alternatives considered

只断言逻辑 KV 归零最快，也可在 CPU 跑，但释放通知缺失时逻辑计数仍归零，不能
鉴别后端槽位是否泄漏。增加同实例满容量复用和真实计算的负对照。

读取后端内部登记表能直接检查分页元数据，但需要新增生产诊断接口或暴露 handle，
超出本次测试增强范围；先明确记录分页探针的判别上限，留给联合设计评审。

## Verification

2026-10-04，策略 1 的四个真实后端用例实际通过，
[原始输出](../../../../tests/evidence/2026-10-04-terminal-reuse-strategy1.log)记录正常
两波各三请求，以及异常终态后的满四请求复用。策略 2 的相同四用例也实际通过，
其[完整 feature 输出](../../../../tests/evidence/2026-10-04-terminal-reuse-strategy2-full-feature.log)
记录 274 个测试与 17 个 doc tests、0 ignored，包含七个实际加载模型的 GPU 用例
（其中一个为释放通知故障对照），另有 30 条 tokenizer fixture。

两种策略的越界之后四请求均成功；prefill 后、decode 后各取消四个已有 1/2 个
输出 token 的请求，每个阶段随后四请求成功。两个阶段累计 8 cancelled、
8 completed、0 failed，重复 cancel 返回 false，取消无 finish_reason、无尾文本；
终态 ID 集合与提交 ID 一致且后续步骤无重复终态，逻辑 KV 和 active 都为零。

拦截释放通知时两种策略仍报告逻辑 KV 与 active 为零，但结果不同：

| 隔离运行 | 对照下一批结果 | 可支持的结论 |
|----------|----------------|--------------|
| 策略 1，分页 KV | 四请求成功，4 cancelled、4 completed、0 failed | 同实例复用不能鉴别分页序列登记泄漏 |
| 策略 2，连续 KV | 四请求失败，4 cancelled、0 completed、4 failed；错误含 `tinyllm_allocate_sequence` | 满容量复用能检出本对照的连续槽位耗尽 |

对照 wrapper 的 execute 与 capabilities 委托真实 TinyLlmExecutor；只有
sequences_finished 使用 trait 默认空实现。它没有 mock 计算或预造失败消息，
同一正向配置能再服务，反向配置实际在新序列分配处失败。对照的预期错误不是
正常生产路径故障，也不证明所有部分释放、元数据或显存泄漏都已覆盖。

输入、工具链与 tiny-llm 库哈希均复核，与
[执行语义记录](2026-10-04-real-test-execution-gates.md#verification)相同：
RTX 3060 Laptop 6144 MiB、driver 610.88、nvcc 12.0.140、GCC 13.3.0，Rust 1.88.0。
tiny-llm commit 为 `b9cbdf922dcfaf0dfc8c704bd285b4b04e757f51`，worktree 干净；
`cmake --build ../tiny-llm/build --target tiny_llm --parallel 2` 返回目标为最新，
库 SHA-256 仍为 `a0f095664a22b47742f8eae03e0c1f2a8370813486f50ce3aede8cdbc1842460`，
spdlog SHA-256 仍为 `06507e43ceb20a24218ec18224b050a7f21becfe0badd191f579b3a2ddb8c192`。
模型、tokenizer 与 fixture SHA-256 均与所链接记录相同，没有下载新输入。

Paged 测试基于 `b83dcf843906f5ab8a2eed18ab7077be074d8e62` 加未提交测试补丁，
执行时 dirty；`tests/tiny_llm_backend.rs` 的 SHA-256 为
`2e4d6ee8adec5fee30a979d85e8af5bdc6d68c13aeff126d400779cf527bbda1`，
生产 src、build.rs、C ABI、Cargo.toml、Cargo.lock 和既有实验结果均无 diff。
新输出与同一测试内容随提交保存，不称 clean-commit 性能实验。

策略矩阵命令如下；策略 2 的完整输出以同样变量运行
`cargo +1.88.0 test --locked --features tiny-llm -- --include-ignored --test-threads=1 --nocapture`。

```bash
export TINY_LLM_DIR=../tiny-llm/build
export TINY_LLM_MODEL=../models/qwen2.5-0.5b-instruct-q4_k_m.gguf
export PAGED_SERVING_TINY_LLM_MAX_SEQS=4
export PAGED_SERVING_TINY_LLM_DECODE_RESERVE=512
for strategy in 1 2; do
  PAGED_SERVING_TINY_LLM_STRATEGY="$strategy" \
    cargo +1.88.0 test --locked --features tiny-llm --test tiny_llm_backend \
    -- --test-threads=1 --nocapture
done
```

完整 suite 另需 `PSERV_TOKENIZER_JSON=../models/tokenizer.json` 与
`PSERV_TOKENIZER_FIXTURE=../tiny-llm/tests/data/tokenizer_fixture.json`。默认 CPU suite
263 个测试与 17 个 doc tests 通过，1 ignored；缺模型时后端目标明确 4 failed、
Cargo 101。默认/feature stable locked Clippy、fmt 与 36 个 Serving Python 回归通过。

## Consequences

额外模型加载延长 GPU 验证，必须串行。固定批次为四、decode reserve 为 512；
这是资源生命周期 canary，不测 HTTP 断连、CUDA 显存字节释放或取消响应时间。
测试探针与正常路径共用计算实现，但释放通知负对照独立于生产回收逻辑。
复用与反例使账面归零的局限可以复现；代价是分页元数据回收仍不能由这个探针
独立认证，需要单独设计可观察的登记验证，不自动新增生产诊断接口。
