# Agent Note: 最低 Rust 版本覆盖锁定的默认依赖图

Status: implemented

## Problem

Cargo.toml 声明 Rust 1.82，但 Cargo.lock 的 ICU 2.3 系列需要 1.88，criterion 0.8.2
需要 1.86。只测 stable 不能证明声明的最低版本可以构建测试与 benchmark。

## Decision

保持依赖与锁文件不变，声明 Rust 1.88，并用固定 1.88.0 在 CI 执行
`cargo check --locked --all-targets` 和 `cargo test --locked`。stable 的构建、lint、
测试和文档也使用 --locked。CUDA FFI feature 的双仓环境要求仍独立验证。
`fix/**` 推送也执行该门禁，修复分支不必先合入默认分支才能获得 CI 验证。

## Alternatives considered

降级并精确固定依赖可保留 1.82，但会同时更换 CLI、测试、URL/Unicode 等依赖，
扩大行为回归面；当前没有必须支持 1.82 的消费者约束，选择已锁定依赖的实际下限。

只提高数字并继续测 stable 维护成本最低，但未来依赖升级仍会悄悄突破最低版本。
固定最低工具链与 --locked 同时约束编译器和依赖解析。

## Verification

1.88.0 的 `cargo check --locked --all-targets` 与 `cargo test --locked` 本地通过：
第一批基线为 230 个默认测试和 17 个 doc tests；README/Cargo.toml/CI 同步，Cargo.lock 无变化。
stable Clippy/fmt 通过；PeriodicFailExecutor 的整除判断用 is_multiple_of，fail_mod
由构造器保证 >=1，同一测试在 1.88.0 再运行通过。CI 配置相同命令为独立 MSRV job，
远端实际执行状态以对应修复分支的 Actions run 为准。

## Consequences

锁定依赖的最低版本可被门禁验证；1.82 用户需升级，新增 CI job 增加编译成本。
最低版本门禁不覆盖实际 CUDA 链接或
所有平台目标，相关结论不得由默认 CPU 测试外推。
