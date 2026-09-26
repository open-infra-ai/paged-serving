# Agent Note: 后端显式选择——feature ≠ 正在用真实后端

Status: implemented

## Problem

此前只要编译时启用 `tiny-llm` feature，服务端就被当成「正在用真实 CUDA
后端」。实测中性能实验可能静默落到 CPU reference 执行器：feature 在场
不等于后端被选，产物报告里的「真实 CUDA」标签因此不可信。

## Decision

运行时显式选择后端：`--backend tiny-llm --model-path <model.gguf>`。
feature、后端参数与模型路径不匹配时**直接报错**，不做静默回退。CPU
reference 执行器仍是默认后端（确定性、供测试与 CI）。

## Alternatives considered

- **feature 启用即默认 CUDA 后端** — 最少参数最强；但「装了」≠「在用」，
  已实测产生误导性性能记录。
- **自动探测模型文件存在与否** — 看似智能；路径存在与否不是语义选择，
  静默推断同上。

## Consequences

- **收益**：每条性能/benchmark 记录的 backend 身份来自显式命令行，
  可审计、可复现。
- **代价**：启动命令更长；老脚本若依赖隐式回退会报错（刻意的，报错即护栏）。

## Verification

commit `11b2297 fix: make real serving backend explicit`（2026-09-04）；
feature 与 `--backend` 不匹配时进程报错退出。
