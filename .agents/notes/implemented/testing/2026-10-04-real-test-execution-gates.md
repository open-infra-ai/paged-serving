# Agent Note: 真实后端测试必须区分未执行与通过

Status: implemented

## Problem

真实 tokenizer 与 tiny-llm 集成测试在缺少运行时输入时直接返回，测试框架却将其
记为 passed。默认 CPU 测试计数包含未执行的差分验证；显式启用真实后端也不能
证明模型加载和 GPU 请求实际运行。

## Decision

外部 tokenizer 差分测试使用 Rust 的显式 ignore，默认套件如实显示未执行。
指定 `--ignored` 后两个输入均为必填；启用 `tiny-llm` feature 后，两个后端测试与
三个文本测试缺少输入即失败。CPU CI 验证 tokenizer 缺输入的失败出口，不能把
编译错误或零测试当成门禁通过。真实模型测试串行运行，不并发创建多个实例。

此修改仅修复测试执行语义，不修改生产实现、C ABI、模型格式或 GPU runner。
PSRV-P1-002 的完整 L3 设计审批、持续 GPU lane 与服务端取消回收仍独立待验收。

[Rust ABI 笔记](../../implemented/architecture/2026-08-23-ffi-dual-source-rust-side.md)
只部分重叠，继续维护布局双源；
[显式后端选择](../../implemented/architecture/2026-09-04-explicit-backend-selection.md)
只部分重叠，继续维护服务端运行时选择；HTTP/SSE 与 Serving 结果笔记不拥有本决定。

## Alternatives considered

默认套件强制要求外部 tokenizer 可以避免遗漏，但会破坏无模型环境的 CPU 回归。
选择显式 ignored，并要求请求执行时失败可见。

将所有 GPU 测试设为 ignored 能保留宽松的 feature 套件，但启用 feature 已是明确
选择真实后端，进一步隐藏执行会延续假绿问题。GPU 测试直接要求环境变量。

## Verification

2026-10-04，默认 CPU 套件实际执行 263 个测试与 17 个 doc tests，真实 tokenizer
显示 0 passed、1 ignored；两个 GPU 目标为 0 tests，表示 feature 未启用。
历史记录中的 264 个默认 passed 包含 tokenizer 缺输入的直接返回，不能全称为
实际完成的验证。历史测试输出不改写，当前执行计数按此拆分。

两个 tokenizer 输入分别移除时，显式 `--ignored` 返回 Cargo 101，报告 0 passed、
1 failed 并点名变量；CI 同时要求错误文本和实际 failed 数，编译失败不能冒充此验收。
空 cases fixture 也返回 101。真实输入跑过 30 条 fixture 的逐 id 比较。

真实 GPU 聚焦验证实际通过 2 个接入/错误回收、3 个文本和 1 个 tokenizer 用例，
详见[聚焦测试输出](../../../../tests/evidence/2026-10-04-real-backend-focused.log)。
Hello 匹配固定 24-token 历史 oracle；数学输出为 `2+2 is 4.`，只满足公共前缀
与 EOS；三并发结束后 active_sequences 为 0、memory_utilization 为 0。
未修改 token 期望，也未启动独立 llama.cpp 对照。

固定策略/容量、包含 ignored 的完整 feature 套件实际通过 272 个测试与 17 个
doc tests，0 ignored，详见[完整 feature 输出](../../../../tests/evidence/2026-10-04-real-backend-full-feature.log)。
其中只有五个现有集成用例实际加载 GPU 模型；其他测试不得按 GPU 用例计数。
缺模型时两后端/三文本用例分别报告 2/3 failed，缺 tokenizer 时三文本用例报告
3 failed；`CUDA_VISIBLE_DEVICES=-1` 的隔离子进程报告 CUDA error 100 和 1 failed，
均返回 Cargo 101，没有把缺设备变成 skipped 或 passed。

硬件为 RTX 3060 Laptop 6144 MiB，driver 610.88；WSL2 Linux 6.18.40.1，
nvcc 12.0.140、GCC 13.3.0，测试使用 Rust 1.88.0。构建为 Release、CUDA arch 86。
tiny-llm 源码 commit 为 `b9cbdf922dcfaf0dfc8c704bd285b4b04e757f51`，构建前后
worktree 干净；静态库由 `cmake --build ../tiny-llm/build --target tiny_llm --parallel 2`
从该源码增量重建，不能把先前存量库的日期绑定到此 commit。

Paged 测试执行基于 `a7fef1e523132fe5e52bf22141afdc97db53b682` 加当前未提交的
测试补丁，源码状态为 dirty，不称作 clean-commit 实验。三个测试文件以以下哈希
固定；生产 `src/`、ABI、build.rs、Cargo.toml、Cargo.lock 与历史结果均无 diff。
提交此笔记的 commit 包含相同测试内容；构建产物不提交。

| 文件 | SHA-256 |
|------|---------|
| `tests/tiny_llm_backend.rs` | `fda7b986e8d1ff01870749193d26b4e8491b11da626317f04f6241423b7f3d9e` |
| `tests/tiny_llm_text_e2e.rs` | `c70caa014bab6cf6e31f3d477fa750f776de1113ed9757bc27e0696fab8c2cc1` |
| `tests/tokenizer_real_diff.rs` | `22d190891808b8870482acc43ebcd070e056a04ee11b14be9a01cbacc189923a` |
| `tiny-llm/build/libtiny_llm.a` | `a0f095664a22b47742f8eae03e0c1f2a8370813486f50ce3aede8cdbc1842460` |
| `tiny-llm/build/_deps/spdlog-build/libspdlog.a` | `06507e43ceb20a24218ec18224b050a7f21becfe0badd191f579b3a2ddb8c192` |
| `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| `models/tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tiny-llm/tests/data/tokenizer_fixture.json` | `11902d29c3018813716f02d0664ee89ab102733f4598dea000c1ad6ca263441f` |

以下命令生成完整 feature 输出；聚焦输出来自同一组输入与默认容量参数，额外
指定三个 `--test` 目标与 `--nocapture`。两次均串行执行；此输出中的用时不是 benchmark。

```bash
TINY_LLM_DIR=../tiny-llm/build \
TINY_LLM_MODEL=../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
PSERV_TOKENIZER_JSON=../models/tokenizer.json \
PSERV_TOKENIZER_FIXTURE=../tiny-llm/tests/data/tokenizer_fixture.json \
PAGED_SERVING_TINY_LLM_STRATEGY=1 \
PAGED_SERVING_TINY_LLM_MAX_SEQS=4 \
PAGED_SERVING_TINY_LLM_DECODE_RESERVE=512 \
cargo +1.88.0 test --locked --features tiny-llm --quiet -- \
  --include-ignored --test-threads=1
```

`cargo fmt --all -- --check`、默认与 feature 的 locked stable Clippy（`-D warnings`）
及 36 个 Serving Python 回归均通过。CPU CI 的输入拒绝检查不使用 GPU，也不能
代替真实 GPU runner 的持续门禁。

## Consequences

启用 feature 但未配置模型的开发者会看到失败，需要按 README 设置输入。
固定 llama.cpp token 序列是历史 oracle，复跑不能称为本次独立 llama.cpp 对照，
部分前缀断言也不能扩写为三个并发请求全部逐 token 一致。
显式忽略保留纯 CPU 开发入口，严格输入失败使请求执行可审计；代价是调用者
需要配置输入并串行执行。此局部修复和单机功能验证不关闭 PSRV-P1-002 的 L3
审批、完整持续 GPU lane、独立审阅、HTTP 取消回收或默认分支集成缺口。
