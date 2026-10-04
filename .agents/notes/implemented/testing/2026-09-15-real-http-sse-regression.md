# Agent Note: loadgen 回归用真实 HTTP/SSE，不用 mock 传输

Status: implemented

## Problem

`loadgen`（serving 压测客户端）的失败归类——timeout、http_429、stream_error、
no_done——必须经得起真实服务器行为的检验。用 mock 传输层测失败路径，
测的是测试自己编的失败，不是 SSE 协议与服务器交互出的失败。

## Decision

loadgen 回归用一次性本地真实服务器做 HTTP/SSE 失败回归：真实 TCP、真实 SSE
帧、真实状态码，覆盖 timeout / 429 / 4xx / 5xx / connection / stream_error /
no_done 各归类。指标口径固定为 TTFT（首个非空文本 chunk）/ inter-chunk / 逐请求
TPOT；标准 ITL 未采集，保持 null，不把 SSE chunk 当 token。跨引擎仍须分别验证协议
canary，本地夹具不证明 llama-server / vLLM 已实际运行。

`timeout_secs` 是请求的总预算，覆盖响应头和流式正文。正文传输错误的 `is_timeout()`
为真时记为 `timeout`，其余传输异常记为 `stream_error`。服务端 SSE error 帧即使消息
含 timeout 也仍是 `stream_error`，不根据错误文字猜分类；按 Content-Length 未收齐的
正文属于传输异常，完整 EOF 但缺 `[DONE]` 保留 `no_done`。

正文超时后的原始记录保留已经收到的 chunk、TTFT、间隔、usage 与 finish_reason，
`ok` 仍为 false。收到 finish_reason/usage 不能替代 `[DONE]`；失败记录参与错误分布，
不进入成功请求的 TTFT/inter-chunk/TPOT、token total 或 coverage。没有新增错误类、
schema 字段、重试或第二套计时器，也不修改请求总预算。

completions 数据帧通过私有类型解析：choices、choice、usage 和 error 必须是对应的
JSON 数组/对象，text 是字符串，显式 usage 的 completion_tokens 是 u32；参与统计的
字段重复出现属于 protocol_error，扩展字段仍可忽略。请求只生成一个候选，响应不得
包含多个 choice 或非零 index；省略 index 兼容已有夹具。合法文本/空文本 choice 或
usage 帧必须至少出现一次，只有 `[DONE]` 或空 choices 的流不是成功的 completion。
usage-only（包括零 token）、usage:null、空文本终态和服务端扩展字段仍合法；不要求
所有服务端提供 finish_reason。缺 usage 仍表示未知或显式 tokenizer 的文本重分词，
非法 usage 则必须失败，不能靠 tokenizer fallback 将请求洗成成功。

## Alternatives considered

- **mock HTTP 层单测** — 最快最稳；但分帧、半包、429 页体形态都是真实
  服务器行为，mock 只会复刻测试作者的理解。
- **只靠真实 GPU benchmark 验证** — 最真；但昂贵且不可进 CI，回归需要
  能在每次提交跑的形态。
- **按错误字符串识别 timeout** — 能兼容一些包装层，且不需要新的时间控制；但会把
  服务端自行声明的 backend timeout 混成客户端总预算到期。使用 reqwest 原始传输
  错误的类型标记，并以正文截断和服务端 timeout 消息作为反例。
- **给每次 stream read 添加独立 timeout** — 可定义空闲连接的等待上限；但会引入
  与现有请求总预算不同的语义。此项只修分类，不引入 read-idle timeout 或重设 deadline。
- **继续用 Value 做可选字段访问** — 扩展字段兼容性最好，但合法 JSON 不等于合法
  completion，且末值覆盖让重复统计字段有歧义。私有类型只约束解释过的字段。
- **拒绝所有未知字段并要求 finish_reason** — 可冻结一个服务端的完整格式，但会拒绝
  合法 provider 扩展和现有无终态原因的响应。只约束计数、文本与单候选语义，不做
  provider 全字段白名单；只收到 `[DONE]` 则没有可观察 completion，必须失败。

## Consequences

- **收益**：失败归类的口径由真实协议行为锚定；benchmark 报告的失败分类
  有回归保护。
- **代价**：一次性本地服务器增加测试基建；仍是 CI 可跑的轻量形态。

新结果将客户端正文总预算到期归入 timeout；旧结果的 stream_error 可能混有这类
超时，不能仅凭聚合计数重分桶，更不能改写历史 raw。需要比较失败分布时应固定新
loadgen 来源重新测量；这项客户端分类不证明服务端已完成取消或 GPU 资源回收。

类型化解析让格式错误的帧成为显式负结果，而不是低 token coverage 的成功；忽略的
扩展字段不参与统计，也不承诺校验其语义或重复键。旧 raw 不重判、不改写；跨版本
比较成功率时需固定 loadgen 来源重新测量。此解析不验证模型文本正确性，也不是
完整 OpenAI 协议认证；后续支持多候选必须先明确逐候选输出和聚合 token 的口径。

## Verification

commit `54e7ae8`（PR #21，2026-09-15）；测试在 `cargo test` 内自起
本地服务器完成真实 HTTP/SSE 往返。

正文超时回归使用本地 TCP 保持连接：分别在响应前、响应头后首 chunk 前、部分文本
和 usage 后等待客户端真实的一秒总预算到期。容量为 1 的通知通道在请求返回后释放
并 join 服务器线程，五秒仅是夹具异常的等待上限，不用后台三秒 sleep 制造超时。
正文截断与消息为 backend timeout 的 error 帧仍验证为 stream_error。修复前，两个
正文超时分类测试及真实 CLI 落盘测试分别因误记 stream_error 而失败，退出码为 101；
修复后单请求分类、部分输出保留与成功样本排除全部通过。

正文超时批次的 Rust 1.88 locked 21 个 loadgen 测试与 5 个 CLI 用例连续执行 10 轮
均通过，该批完整默认套件为实际 267 个测试加 17 个 doc tests，真实 tokenizer 明确
1 个 ignored。
fmt、default clippy、40 个离线结果审计测试与 notes gate 通过。C ABI、Cargo.lock、
构建/CI 配置及历史实验包没有改动；本批没有新的真实 GPU 或服务端回收证据。

类型化帧回归覆盖无意义 JSON、字段类型与 u32 范围、单候选、数组伪装对象、重复
已知字段（含 null 首值、同值与 Unicode 转义键）、仅 DONE、非法 error 和 tokenizer
fallback。修复前 22 个单请求测试有 6 个失败，新增真实 CLI 用例也因 ok=true 失败，
两条 cargo 命令均退出 101；合法空输出与 usage-only 的两个正向用例通过。
修复后 29 个 loadgen 测试与 6 个 CLI 用例连续 10 轮通过，完整默认套件实际执行
276 个测试加 17 个 doc tests，真实 tokenizer 为 1 个明确 ignored；fmt、default
clippy 与 40 个离线审计测试通过。此批仍没有 GPU、模型输出正确性或性能收益结论。

CLI 参数、预热/测量窗口与结果落盘由
[CLI 复现笔记](2026-10-04-loadgen-cli-reproducibility.md) 单独维护；本篇的单请求分类
与 TTFT/inter-chunk/token coverage 口径保持不变。
