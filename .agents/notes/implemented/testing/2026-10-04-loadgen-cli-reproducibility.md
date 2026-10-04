# Agent Note: loadgen CLI 结果与测量窗口复现

Status: implemented

## Problem

真实 TCP 的单请求分类测试无法证明 CLI 的 warmup、负载调度、请求排序与结果文件正确衔接。Poisson 预热消耗测量 RNG，测量 prompt 由预热后的全局 request_id 选取；同一 seed 和数据集可能因为预热吞吐变化产生不同测量负载。相对 sleep 还会累计调度延迟，worker JoinError 被丢弃可能掩盖不完整执行。

## Decision

Poisson 测量窗口重新从 arrival seed 初始化 RNG，使用测量起点的累积绝对 deadline；warmup 最后的计划到达不能越过预热窗口，唤醒后也检查窗口是否结束。两个模式的测量 prompt 按 measured_index 选取，request_id 继续标识包括预热的全局请求。任务 join 错误向 CLI 传播，不写入伪完整的新报告。

per_request 携带 nullable scheduled_arrival_ms 和 dispatch_offset_ms；summary.config 的 arrival_schedule 使用 `absolute_deadline_seed_reset` 标记。计划时间用于核对同二进制/锁文件下的 seed，实际时间保留调度抖动，不把二者混成服务器收到请求的时间。旧结果包不改写，新增字段属于 schema v1 的可选扩展；没有字段表示未采集，而不是零。

## Alternatives considered

提取公开 loadgen 模块能直接控制时间、运行状态并写细粒度测试，但不能证明 CLI 参数到落盘的真实连接，也扩大 API 面。测试启动 cargo 构建的真实二进制和临时本地服务器，不新增公共模块。

保留相对 sleep 和贯穿预热的 RNG 最小改动，但 seed 的意义会依赖预热执行次数，且调度开销累计进入到达间隔。测量窗口单独 RNG 与绝对 deadline 明确可复现的只是计划负载，不承诺 OS/网络时序相同。

只比对 summary 足够验证聚合总数，但无法发现原始请求缺失、乱序、错误详情或未知 token 被猜测。本测试联合核对 JSONL 与 summary，以及预热实际到达但未被记录。

## Verification

`tests/loadgen_cli.rs` 启动 cargo 构建的真实 loadgen 子进程与本地 TCP/SSE 夹具，校验 closed/Poisson 的 6 条测量记录、序号/输入顺序、错误详情和 summary。预热请求实际被接收但没有进入文件；两次相同 seed、有/无预热的计划完全一致，不同 seed 的计划不同，实际 dispatch 不早于计划。50% 成功请求 token coverage 时 tok/s 为 null；100% coverage 时按 measurement wall 计算，支持自定义 summary 路径。非法参数非零退出且不创建结果。

Rust 1.88 locked all-target check 和完整测试通过：264 个默认测试与 17 个 doc tests。4 个 CLI 用例重复 10 轮通过，共 40 次用例执行、60 个子进程；测试子进程有 10 秒超时与 kill/wait 回收，本地 TCP server 和独占临时目录由测试释放。stable clippy 无警告，fmt 与 notes 使用仓内门禁。没有真实 GPU 负载或性能结论。

## Consequences

测量输入与预热吞吐解耦，原始请求足以区分计划到达和实际客户端 dispatch；CLI 到文件的错误计数和 token 口径有真实传输保护。代价是绝对 deadline 落后时可能紧接发压，不丢请求也不改成 closed-loop；高负载的调度抖动须从实际 dispatch 观察，不声称目标到达率完全实现。新增 nullable 字段让严格未知字段消费者需要升级，但历史结果仍能使用；与旧执行口径做性能比较须重新配对实验。测试 warmup 仅 1 秒，不证明长负载性能、GPU 回收与生产服务端网络压力。

## Related notes

[真实 HTTP/SSE 分类](2026-09-15-real-http-sse-regression.md) 部分重叠，其单请求分类和指标口径保留，本篇定义 CLI 与测量窗口；有界事件/类型化指标笔记处理服务端，非同一决定。
