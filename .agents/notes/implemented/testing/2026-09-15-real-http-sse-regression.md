# Agent Note: loadgen 回归用真实 HTTP/SSE，不用 mock 传输

Status: implemented

## Problem

`loadgen`（serving 压测客户端）的失败归类——timeout、http_429、stream_error、
no_done——必须经得起真实服务器行为的检验。用 mock 传输层测失败路径，
测的是测试自己编的失败，不是 SSE 协议与服务器交互出的失败。

## Decision

loadgen 回归用一次性本地真实服务器做 HTTP/SSE 失败回归：真实 TCP、真实 SSE
帧、真实状态码，覆盖 timeout / 429 / 4xx / 5xx / connection / stream_error /
no_done 各归类。指标口径固定为 TTFT（首个非空文本 chunk）/ ITL / 逐请求
TPOT，同一二进制零改动覆盖 paged-serving / llama-server / vLLM 横向可比。

## Alternatives considered

- **mock HTTP 层单测** — 最快最稳；但分帧、半包、429 页体形态都是真实
  服务器行为，mock 只会复刻测试作者的理解。
- **只靠真实 GPU benchmark 验证** — 最真；但昂贵且不可进 CI，回归需要
  能在每次提交跑的形态。

## Consequences

- **收益**：失败归类的口径由真实协议行为锚定；benchmark 报告的失败分类
  有回归保护。
- **代价**：一次性本地服务器增加测试基建；仍是 CI 可跑的轻量形态。

## Verification

commit `54e7ae8`（PR #21，2026-09-15）；测试在 `cargo test` 内自起
本地服务器完成真实 HTTP/SSE 往返。

CLI 参数、预热/测量窗口与结果落盘由
[CLI 复现笔记](2026-10-04-loadgen-cli-reproducibility.md) 单独维护；本篇的单请求分类
与 TTFT/inter-chunk/token coverage 口径保持不变。
