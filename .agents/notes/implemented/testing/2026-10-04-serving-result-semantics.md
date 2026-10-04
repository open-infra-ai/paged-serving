# Agent Note: Serving 原始结果的语义校验与收敛诊断

Status: implemented

## Problem

结果文件齐全不能证明 summary 与原始请求一致，也不能阻止绘图把同标签但不同模型、commit 或负载参数的 run 平均在一起。没有机器可读的收敛诊断，历史报告中的波动限制容易在二次引用时丢失。

## Decision

现有 validator 联合校验 JSONL、summary 与双层 metadata；绘图复用同一校验入口，在写 CSV/图前拒绝矛盾或不兼容系列。重算成功/失败、错误分类、token coverage、样本数、分位和墙钟吞吐，不把失败流的部分文本计入成功性能样本。任一重复的 token 吞吐不可用时，绘图的整个组合留空，不只平均已知子集。

按历史报告已使用的 `(max - min) / mean > 10%` 生成收敛状态。结构或数值矛盾返回非零；负结果和未收敛不构成数据无效，输出 machine-readable `non_converged` 并保留原始重复。正式结果至少三次重复、预热至少 30 秒；基础校验允许 canary。缺少历史可选字段报告未采集，不回填；Poisson 重复的 arrival_seed 可变化，但执行口径和其余参数必须一致。

四类证据文件共用严格 JSON 入口：对象的重复键在任意嵌套层级被拒绝，包括相同值和 Unicode 转义后同名的键；浮点字面量溢出为无穷大也被拒绝，包含未参与指标计算的扩展字段。诊断使用 `invalid_json`、文件路径及重复键或字面量；JSONL 另有实际行号。解析不替用户选择首值或末值，也不修写输入。

原始浮点指标有重算容差，配置必须精确一致；rate 标签按浮点精确表示归组，防止格式化截断把相邻到达率合并。CSV 声明被测引擎 commit、模型 SHA 和量化，图例标明引擎 commit 和量化，标题单独声明 loadgen 来源；`audit.json` 保留规则与逐组合状态，`--out-dir` 让历史重验不覆盖已有 CSV/PNG。

## Alternatives considered

独立 JSON Schema 可以清晰声明类型，但不能单独表达跨 JSONL/summary 的重算、重复一致性或收敛；使用 Python 标准库的共享校验逻辑和离线单测，避免新增运行依赖。

让绘图在参数不一致时自动拆成不同系列能保留所有数据，但当前目录标签会继续暗示同一实验条件，容易把差异解释为重复波动；同一标签内拒绝混合并给出字段诊断，由实验者显式整理新实验，不移动历史结果。

只检查总数和 coverage 的实现最小，但篡改分位、吞吐分母或错误分类仍能通过；联合重算成功样本，使用与 Rust 相同的分位选取规则。

Python 默认的末值覆盖容易兼容一般 JSON 消费者，也无需额外代码，但会把同时声明失败和成功的请求解释成单一成功值。只对已知字段查重复又会漏掉嵌套 metadata；标准库 `object_pairs_hook` 与 `parse_float` 在共享入口检查完整文档，不增加依赖或另一套 schema。

## Verification

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmarks/serving -p 'test_*.py' -v` 的 36 个离线测试通过，含合法/坏 JSON、字段类型、请求序号、warmup 泄漏、错误分类、失败部分输出、token 来源、样本/分位、墙钟吞吐、metadata 冲突、配置混用、相邻 rate、历史字段、未收敛与 CLI 退出码。Python 门禁和历史正式包重验配置在独立 `serving-evidence` CI job。

三个正式包用 `--formal`、两个 canary 用基础模式，只读验证全部 66 个 run；2026-09-07 的 21-run 包重现 c2/c8 与三档 Poisson 的 TTFT p95 未收敛、83 个 429，以及历史 CSV 的核心均值。图表生成到 `/tmp/paged-serving-audit-X5wS8w`，包含三张二维图、CSV 和 audit；本机 matplotlib 的 Axes3D 环境警告不影响二维产物，未修改依赖环境。历史 results 的 git diff 为空。

Rust 1.88 `cargo test --locked` 的 264 个默认测试与 17 个 doc tests 通过，stable clippy、fmt 通过。没有启用 tiny-llm feature，默认 suite 中的 feature-only 文件有零测试，不作为真实 CUDA 验收。

严格 JSON 解析的完整回归为 40 个离线测试。四个新增方法覆盖所有证据文件、嵌套对象/数组、相同值重复、转义同名、正负浮点溢出与有限大数；CLI 校验失败退出 1，绘图拒绝创建输出目录，输入逐字节保持不变。未修复解析入口时，新回归出现 17 个失败断言，含真实 validator CLI 错误退出 0；修复后全部通过。

同一严格入口只读重验五个存量包共 66 个 run，三个正式包使用 `--formal`、两个 canary 使用基础模式，全部通过；历史 results 的 git diff 为空。解析限制不要求迁移有效存量数据。

解析修复的 Rust 1.88 默认回归实际执行 263 个测试加 17 个 doc tests，真实 tokenizer 明确 1 个 ignored；fmt、default clippy 与 notes gate 通过。没有启用真实后端 feature，本项不重复采集或扩大 GPU 验证声明。

## Consequences

原始数据的二次引用具有可执行门禁，未收敛与 coverage 缺口进入 CSV/审计而不是丢失；代价是同一系列改变模型、commit 或负载控制参数需要显式分开实验。正式校验比产物存在性检查严格，canary 保持使用基础模式；发现历史矛盾时须给出诊断，不能放宽重算或改写 raw。

metadata 仅声明实验来源，校验不能证明 GPU 实际执行、模型内容或服务端配置；CSV/PNG 与人工报告只检查存在性，不认证历史图表内容。独立审阅、真实 CUDA 取消回收、网络压力与稳定 SLO 仍是另外的验收，不因本批自动完成。

重复键即使值相同也构成歧义证据，严格入口拒绝这种输入；未来引入高精度数值格式时须显式设计表示与计算方式，不能把溢出的浮点值当作可忽略扩展。schema v1 字段、有效数据的统计口径及未收敛规则保持原样。

## Related notes

[CLI 复现](2026-10-04-loadgen-cli-reproducibility.md) 部分重叠：它定义生产字段和负载执行，本篇验证存量/新增结果的二次引用；不改变其 schema v1 可选扩展决定。其余取消、ABI、后端选择与单请求传输笔记不改变本项校验口径。
