#!/usr/bin/env python3
"""从 serving sweep 结果生成跨引擎汇总与图表。

单次 run 的权威聚合来自 loadgen 生成的 ``summary.json``。本脚本不再从
逐请求 duration 反推测量墙钟，避免并发多波次时高估吞吐。

用法：
    python3 plots.py results/2026-08-30-RTX-3060-Laptop

输出：
    ttft_by_concurrency.png       闭环 TTFT p95 × 并发
    throughput_by_concurrency.png 闭环输出 token 吞吐 × 并发
    slo_curve.png                 泊松 TTFT p95 × 到达率
    summary_table.csv             全指标与重复波动
    audit.json                    校验、收敛状态与声明的模型/引擎来源
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from validate_results import audit_package


def nested(data, *path):
    current = data
    for key in path:
        current = current[key]
    return current


def collect_groups(root: Path):
    report, runs, metadata = audit_package(root)
    if report["errors"]:
        details = "\n".join(f"{i['code']}: {i['location']}: {i['message']}" for i in report["errors"])
        raise ValueError(f"拒绝绘制无效或不兼容结果：\n{details}")
    groups = {}
    for run in runs:
        groups.setdefault(run["group"], []).append(run["summary"])
    return groups, report, metadata


def values_at(runs, *path):
    values = [nested(run, *path) for run in runs]
    # 不能只选 token coverage 完整的重复，偷偷排除其余 run 后平均。
    return [] if any(value is None for value in values) else values


def aggregate(runs, *path):
    values = values_at(runs, *path)
    if not values:
        return None
    return sum(value / len(values) for value in values)


def spread(runs, *path):
    values = values_at(runs, *path)
    if not values:
        return None, None, None
    return sum(value / len(values) for value in values), min(values), max(values)


def tag_value(tag: str) -> float:
    if tag.startswith("rate"):
        return float(tag[4:])
    if tag.startswith("c"):
        return float(tag[1:])
    raise ValueError(f"未知负载标签：{tag}")


def series_label(engine: str, dataset: str, row) -> str:
    return f"{engine}/{dataset} @ {row['engine_commit'][:7]}, {row['backend_quant']}"


def write_csv(root: Path, rows):
    csv_path = root / "summary_table.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.3f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    print(f"written {csv_path}")


def plot_closed_ttft(root: Path, rows, caption: str, plt):
    closed = [row for row in rows if row["mode"] == "closed"]
    if not closed:
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    series = sorted({(row["engine"], row["dataset"]) for row in closed})
    plotted = False
    for engine, dataset in series:
        subset = sorted(
            (
                row
                for row in closed
                if row["engine"] == engine
                and row["dataset"] == dataset
                and row["ttft_p95"] is not None
            ),
            key=lambda row: tag_value(row["tag"]),
        )
        if not subset:
            continue
        x_values = [tag_value(row["tag"]) for row in subset]
        means = [row["ttft_p95"] for row in subset]
        lows = [row["ttft_p95_min"] for row in subset]
        highs = [row["ttft_p95_max"] for row in subset]
        axis.plot(x_values, means, marker="o", label=series_label(engine, dataset, subset[0]))
        axis.fill_between(x_values, lows, highs, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_yscale("log")
    axis.set_xlabel("concurrency")
    axis.set_ylabel("TTFT p95 (ms, log)")
    axis.set_title(f"TTFT p95 by concurrency — {caption}")
    axis.legend()
    fig.tight_layout()
    path = root / "ttft_by_concurrency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"written {path}")


def plot_closed_throughput(root: Path, rows, caption: str, plt):
    closed = [row for row in rows if row["mode"] == "closed"]
    if not closed:
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    series = sorted({(row["engine"], row["dataset"]) for row in closed})
    plotted = False
    for engine, dataset in series:
        subset = sorted(
            (
                row
                for row in closed
                if row["engine"] == engine
                and row["dataset"] == dataset
                and row["throughput_tok_s"] is not None
            ),
            key=lambda row: tag_value(row["tag"]),
        )
        if not subset:
            continue
        x_values = [tag_value(row["tag"]) for row in subset]
        means = [row["throughput_tok_s"] for row in subset]
        lows = [row["throughput_tok_s_min"] for row in subset]
        highs = [row["throughput_tok_s_max"] for row in subset]
        axis.plot(
            x_values,
            means,
            marker="o",
            label=series_label(engine, dataset, subset[0]),
        )
        axis.fill_between(x_values, lows, highs, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        print("skip throughput_by_concurrency.png: token 计数覆盖不足")
        return
    axis.set_xlabel("concurrency")
    axis.set_ylabel("output throughput (tok/s)")
    axis.set_title(f"Output throughput by concurrency — {caption}")
    axis.legend()
    fig.tight_layout()
    path = root / "throughput_by_concurrency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"written {path}")


def plot_poisson_slo(root: Path, rows, caption: str, plt):
    poisson = [row for row in rows if row["mode"] == "poisson"]
    if not poisson:
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    series = sorted({(row["engine"], row["dataset"]) for row in poisson})
    plotted = False
    for engine, dataset in series:
        subset = sorted(
            (
                row
                for row in poisson
                if row["engine"] == engine
                and row["dataset"] == dataset
                and row["ttft_p95"] is not None
            ),
            key=lambda row: tag_value(row["tag"]),
        )
        if not subset:
            continue
        x_values = [tag_value(row["tag"]) for row in subset]
        means = [row["ttft_p95"] for row in subset]
        lows = [row["ttft_p95_min"] for row in subset]
        highs = [row["ttft_p95_max"] for row in subset]
        axis.plot(
            x_values,
            means,
            marker="s",
            label=series_label(engine, dataset, subset[0]),
        )
        axis.fill_between(x_values, lows, highs, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xlabel("arrival rate λ (req/s)")
    axis.set_ylabel("TTFT p95 (ms)")
    axis.set_title(f"SLO curve — {caption}")
    axis.legend()
    fig.tight_layout()
    path = root / "slo_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"written {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-dir", type=Path, help="产物输出目录；重验历史结果时使用独立目录")
    args = parser.parse_args()
    try:
        groups, report, metadata = collect_groups(args.root)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    # 校验先于任何输出和可选 matplotlib 依赖，坏数据不会留下半套图表。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.out_dir or args.root
    root.mkdir(parents=True, exist_ok=True)
    commit = metadata["commits"]["paged_serving"][:10]
    date = metadata["date"]
    gpu = metadata["hardware"]["gpu"]
    caption = f"loadgen @ {commit}, {date}, {gpu}"

    convergence = {tuple(group["key"]): group["metrics"] for group in report["groups"]}
    provenance = {tuple(group["key"]): group["provenance"] for group in report["groups"]}
    rows = []
    for (engine, mode, tag, dataset), runs in sorted(groups.items()):
        ttft_p95, ttft_p95_min, ttft_p95_max = spread(runs, "ttft_ms", "p95")
        throughput, throughput_min, throughput_max = spread(
            runs, "throughput", "output_tokens_per_second"
        )
        rows.append(
            {
                "engine": engine,
                "mode": mode,
                "tag": tag,
                "dataset": dataset,
                "repeats": len(runs),
                **provenance[(engine, mode, tag, dataset)],
                "success_rate_pct": aggregate(runs, "requests", "success_rate_pct"),
                "ttft_p50": aggregate(runs, "ttft_ms", "p50"),
                "ttft_p95": ttft_p95,
                "ttft_p95_min": ttft_p95_min,
                "ttft_p95_max": ttft_p95_max,
                "ttft_p99": aggregate(runs, "ttft_ms", "p99"),
                "inter_chunk_p50": aggregate(
                    runs, "inter_chunk_latency_ms", "p50"
                ),
                "tpot_p50": aggregate(runs, "tpot_ms", "p50"),
                "tpot_p95": aggregate(runs, "tpot_ms", "p95"),
                "token_coverage_pct": aggregate(
                    runs, "completion_tokens", "coverage_pct"
                ),
                "throughput_tok_s": throughput,
                "throughput_tok_s_min": throughput_min,
                "throughput_tok_s_max": throughput_max,
                "throughput_req_s": aggregate(
                    runs, "throughput", "successful_requests_per_second"
                ),
                "ttft_convergence": convergence[(engine, mode, tag, dataset)]["ttft_p95_ms"]["status"],
                "throughput_convergence": convergence[(engine, mode, tag, dataset)]["output_tokens_per_second"]["status"],
            }
        )

    write_csv(root, rows)
    (root / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plot_closed_ttft(root, rows, caption, plt)
    plot_closed_throughput(root, rows, caption, plt)
    plot_poisson_slo(root, rows, caption, plt)
    for group in report["groups"]:
        if any(metric["status"] == "non_converged" for metric in group["metrics"].values()):
            print(f"non_converged: {'/'.join(group['key'])}；原始重复全部保留，详见 audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
