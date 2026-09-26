# Agent Note: 分页 KV 策略 1 为默认后端路径

Status: implemented

## Problem

tiny-llm 提供两种 KV 布局：连续 cache（策略 2）与分页 KV（策略 1，
block_tables + scatter/gather 池）。控制面要做的是 Paged KV 调度练习，
后端默认走哪条路径决定了「块表上传、资源守恒」这些控制面概念是否真实生效。

## Decision

`tiny-llm` cargo feature 下，每序列 `block_tables` 真实上传到 tiny-llm 的
分页 KV 池（策略 1）为**默认路径**；`PAGED_SERVING_TINY_LLM_STRATEGY=2`
可显式回退连续 KV。容量由 `PAGED_SERVING_TINY_LLM_MAX_SEQS`（默认 4）与
`PAGED_SERVING_TINY_LLM_DECODE_RESERVE`（默认 512）调节。

## Alternatives considered

- **默认连续 KV、分页作为可选** — 实现最稳、对齐最容易；但那样控制面练的
  就是「调度了一个不用分页的运行时」，Paged KV 块表上传与资源守恒无从体现。
- **只实现策略 1** — 最少代码；但失去 A/B 对照面，策略 2 保留为回退与
  正确性参照（tiny-llm 侧有两策略逐 token 差分）。

## Consequences

- **收益**：控制面声明的 Paged KV 语义在真实 CUDA 后端上可验证——
  3 并发 e2e 与 llama.cpp greedy 对齐即走此路径。
- **代价**：策略 1 的块表契约（长度校验、非法块 id 语义）变成控制面必须
  遵守的真实约束，不是抽象接口。

## Verification

commit `fb9d670` 标记策略 1 为默认（2026-08-18）；`--backend tiny-llm`
路径下块表真实上传；回退开关经 `PAGED_SERVING_TINY_LLM_STRATEGY`。
