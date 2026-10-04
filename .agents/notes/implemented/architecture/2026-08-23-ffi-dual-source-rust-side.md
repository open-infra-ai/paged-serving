# Agent Note: `src/tiny_llm_ffi.rs` 是双源 C ABI 的 Rust 侧

Status: implemented

## Problem

本仓经 C ABI 调用 tiny-llm 引擎。Rust 侧结构体定义若与 C 头文件漂移，
repr(C) 布局错位不会在编译期报警，只会在运行时表现为错位读写。

## Decision

`src/tiny_llm_ffi.rs` 与 `tiny-llm/include/tiny_llm/ffi.h` 构成 ABI 代码双源：

- `TinyLlmConfig` repr(C) 布局守卫测试断言 `size_of == 9 * 4`，
  与 C 侧 9 个 int 字段一一对应。
- 缓冲区契约按 C 侧澄清实现：`logprobs` 为 `(token_id, logprob)` 交错
  float 对，容量 `num_sequences * logprobs_k * 2`。
- 序列生命周期由后端拥有（allocate/free），调度侧只驱动 step。
- ABI 或布局变化是 breaking change：先改 meta 仓契约文档，双仓同批，
  两仓 CHANGELOG 各记一条。

## Alternatives considered

- **bindgen 自动生成绑定** — 单源最强；但 ABI 面很窄（~9 字段 config +
  少量函数），守卫测试已能捕获漂移，引入生成链反而增加构建复杂度。
- **serde + IPC 序列化** — 彻底隔离；但同进程静态链接是本作品集刻意练习
  的窄接口形态，IPC 会掩盖要学的调度边界。

## Consequences

- **收益**：布局错位在 `cargo test` 当场报警；契约澄清（如 2026-08-23
  logprobs 容量）有明确落点。
- **代价**：双源靠守卫与纪律维持，新增字段必须同批改两仓。

## Verification

布局守卫测试在 `tests/` 与 crate 内 `size_of` 断言；e2e 差分见
`tests/tiny_llm_backend.rs`、`tests/tiny_llm_text_e2e.rs`。
