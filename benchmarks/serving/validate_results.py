#!/usr/bin/env python3
"""联合校验 Serving 原始请求、summary、metadata 与重复收敛。

python3 validate_results.py [--formal] [--json] <result-root>
数值矛盾返回 1；未收敛仍是有效负结果，详见 methodology.md。
"""

import argparse
from collections import Counter
from datetime import date
import json
import math
from pathlib import Path
import re
import sys


RUN_DIR_RE = re.compile(
    r"^(?P<engine>.+)_(?P<mode>closed|poisson)_"
    r"(?P<tag>c\d+|rate\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)_"
    r"(?P<ds>\w+)_r(?P<rep>\d+)$"
)
RUN_FILES = ("run_metadata.json", "per_request.jsonl", "summary.json", "stdout.log")
CONFIG_FIELDS = (
    "engine", "mode", "base_url", "model", "dataset", "requests", "concurrency",
    "rate", "max_tokens", "warmup_secs", "timeout_secs", "tokenizer",
)
CONVERGENCE_METRICS = {
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "output_tokens_per_second": ("throughput", "output_tokens_per_second"),
    "successful_requests_per_second": ("throughput", "successful_requests_per_second"),
}


class EvidenceError(ValueError):
    def __init__(self, code, location, message):
        super().__init__(message)
        self.issue = {"code": code, "location": str(location), "message": message}


def require(condition, code, location, message):
    if not condition:
        raise EvidenceError(code, location, message)


def field(obj, key, location):
    require(key in obj, "missing_field", f"{location}.{key}", "缺少必需字段")
    return obj[key]


def mapping(value, location):
    require(isinstance(value, dict), "invalid_type", location, "必须是 JSON 对象")
    return value


def number(value, location):
    require(type(value) in (int, float) and 0 <= value <= sys.float_info.max,
            "invalid_number", location, "必须是有限非负数，bool 不算数字")
    return value


def integer(value, location, minimum=0):
    require(type(value) is int and minimum <= value <= 2**64 - 1, "invalid_integer", location,
            f"必须是 {minimum}..2^64-1 范围的整数")
    return value


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), "invalid_type", location,
            "必须是非空字符串")
    return value


def same(actual, expected, location, exact=False):
    if type(expected) is float:
        number(actual, location)
        matches = actual == expected if exact else math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)
    elif type(expected) is int:
        matches = type(actual) is int and actual == expected
    else:
        matches = actual == expected
    require(matches, "value_mismatch", location,
            f"声明值 {actual!r} 与原始数据/配置预期 {expected!r} 不一致")


def reject_constant(value):
    raise ValueError(f"JSON 禁止非有限常量 {value}")


def decode_json(content, location):
    try:
        value = json.loads(content, parse_constant=reject_constant)
    except ValueError as error:
        raise EvidenceError("invalid_json", location, str(error)) from error
    return mapping(value, location)


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EvidenceError("unreadable_file", path, str(error)) from error


def load_json(path):
    return decode_json(read_text(path), path)


def percentile(values, percent):
    ordered = sorted(values)
    # Rust f64::round 对非负值用 half-up，Python round 的 ties-to-even 不等价。
    return float(ordered[math.floor(percent / 100 * (len(ordered) - 1) + 0.5)]) if ordered else None


def check_metric(summary, key, values, location):
    metric = mapping(field(summary, key, location), f"{location}.{key}")
    same(field(metric, "samples", location), len(values), f"{location}.{key}.samples")
    for percent in (50, 95, 99):
        label = f"p{percent}"
        same(field(metric, label, location), percentile(values, percent), f"{location}.{key}.{label}")


def validate_records(path, config, wall_secs, warnings):
    records = [decode_json(line, f"{path}:{index}")
               for index, line in enumerate(read_text(path).splitlines(), 1) if line.strip()]
    require(bool(records), "empty_requests", path, "原始请求文件为空")
    ids = set()
    timing_present = any("dispatch_offset_ms" in record or "scheduled_arrival_ms" in record
                         for record in records)
    new_schedule = config.get("arrival_schedule") == "absolute_deadline_seed_reset"
    previous_schedule = 0.0
    for index, record in enumerate(records):
        loc = f"{path}:record[{index}]"
        same(field(record, "measured_index", loc), index, f"{loc}.measured_index")
        request_id = integer(field(record, "request_id", loc), f"{loc}.request_id")
        require(request_id not in ids, "duplicate_request_id", loc, "request_id 重复")
        ids.add(request_id)
        ok = field(record, "ok", loc)
        require(type(ok) is bool, "invalid_type", f"{loc}.ok", "必须是 bool")
        error_class = field(record, "error_class", loc)
        if ok:
            same(error_class, None, f"{loc}.error_class")
            same(record.get("error_detail"), None, f"{loc}.error_detail")
        else:
            text(error_class, f"{loc}.error_class")
            if "error_detail" in record:
                text(record["error_detail"], f"{loc}.error_detail")
        duration = number(field(record, "duration_ms", loc), f"{loc}.duration_ms")
        require(duration <= wall_secs * 1000 + 1e-6, "invalid_timeline", loc,
                "单请求耗时超过完整测量墙钟")
        chunks = integer(field(record, "chunks", loc), f"{loc}.chunks")
        ttft = field(record, "ttft_ms", loc)
        if chunks:
            number(ttft, f"{loc}.ttft_ms")
            require(ttft <= duration + 1e-6, "invalid_timeline", loc, "TTFT 超过请求总耗时")
        else:
            same(ttft, None, f"{loc}.ttft_ms")
        intervals = field(record, "inter_chunk_latency_ms", loc)
        require(isinstance(intervals, list), "invalid_type", loc, "inter-chunk 必须是数组")
        same(len(intervals), max(chunks - 1, 0), f"{loc}.inter_chunk_latency_ms.length")
        for interval in intervals:
            number(interval, f"{loc}.inter_chunk_latency_ms")
        require(sum(intervals) + (ttft or 0) <= duration + 1e-6, "invalid_timeline", loc,
                "TTFT 加 chunk 间隔超过请求耗时")
        tokens = field(record, "completion_tokens", loc)
        source = field(record, "tokens_source", loc)
        if tokens is None:
            same(source, None, f"{loc}.tokens_source")
        else:
            integer(tokens, f"{loc}.completion_tokens")
            require(source in ("usage", "tokenizer_text"), "invalid_token_source", loc,
                    "已知 tokens 必须注明 usage 或 tokenizer_text，不能由 chunk 数猜测")
        if timing_present or new_schedule:
            dispatch = number(field(record, "dispatch_offset_ms", loc), f"{loc}.dispatch_offset_ms")
            require(dispatch + duration <= wall_secs * 1000 + 1e-6, "invalid_timeline", loc,
                    "dispatch 加请求耗时超过测量窗口")
            scheduled = field(record, "scheduled_arrival_ms", loc)
            if config["mode"] == "closed":
                same(scheduled, None, f"{loc}.scheduled_arrival_ms")
            else:
                number(scheduled, f"{loc}.scheduled_arrival_ms")
                require(previous_schedule <= scheduled <= dispatch + 1e-6,
                        "invalid_timeline", loc, "计划时间必须单调且不晚于实际 dispatch")
                previous_schedule = scheduled
    if not timing_present and not new_schedule:
        warnings.append({"code": "legacy_timing_unavailable", "location": str(path),
                         "message": "历史计划/dispatch 时间未采集，不补写或解释为零"})
    return records


def validate_run(run, root_metadata, formal, warnings):
    for name in RUN_FILES:
        require((run / name).is_file(), "missing_file", run / name, "缺少 run 必需产物")
    loc = run / "summary.json"
    summary = load_json(loc)
    meta = load_json(run / "run_metadata.json")
    same(field(summary, "schema_version", loc), 1, f"{loc}.schema_version")
    same(field(meta, "schema_version", run), 1, f"{run}/run_metadata.json.schema_version")
    config = mapping(field(summary, "config", loc), f"{loc}.config")
    for key in CONFIG_FIELDS:
        field(config, key, f"{loc}.config")
    for key in ("engine", "mode", "base_url", "model", "dataset"):
        text(config[key], f"{loc}.config.{key}")
    for key in ("requests", "max_tokens", "timeout_secs"):
        integer(config[key], f"{loc}.config.{key}", 1)
    integer(config["warmup_secs"], f"{loc}.config.warmup_secs")
    if config["tokenizer"] is not None:
        text(config["tokenizer"], f"{loc}.config.tokenizer")
    match = RUN_DIR_RE.fullmatch(run.name)
    same(config["engine"], match["engine"], f"{loc}.config.engine")
    same(config["mode"], match["mode"], f"{loc}.config.mode")
    same(Path(config["dataset"]).stem, match["ds"], f"{loc}.config.dataset")
    if config["mode"] == "closed":
        concurrency = integer(config["concurrency"], f"{loc}.config.concurrency", 1)
        same(match["tag"], f"c{concurrency}", f"{run}.tag")
        same(config["rate"], None, f"{loc}.config.rate")
        same(config.get("arrival_seed"), None, f"{loc}.config.arrival_seed")
        same(config.get("arrival_schedule"), None, f"{loc}.config.arrival_schedule")
    else:
        rate = number(config["rate"], f"{loc}.config.rate")
        require(rate > 0 and match["tag"].startswith("rate"), "invalid_load", run,
                "Poisson rate 必须 > 0，目录标签须为 rate")
        same(rate, float(match["tag"][4:]), f"{loc}.config.rate", exact=True)
        same(config["concurrency"], None, f"{loc}.config.concurrency")
        if config.get("arrival_seed") is not None:
            integer(config["arrival_seed"], f"{loc}.config.arrival_seed")
        else:
            require(config.get("arrival_schedule") is None, "missing_field", loc,
                    "新到达执行口径必须记录实际 arrival_seed")
            warnings.append({"code": "legacy_seed_unavailable", "location": str(loc),
                             "message": "Poisson seed 未采集，不能声明计划负载可复现"})
        require(config.get("arrival_schedule") in (None, "absolute_deadline_seed_reset"),
                "unsupported_schedule", loc, "不支持的到达执行口径")
    engine = mapping(field(meta, "engine", run), f"{run}.engine")
    model = mapping(field(meta, "model", run), f"{run}.model")
    load = mapping(field(meta, "load", run), f"{run}.load")
    same(field(engine, "name", run), config["engine"], f"{run}.engine.name")
    text(field(engine, "commit", run), f"{run}.engine.commit")
    require(type(field(engine, "dirty", run)) is bool, "invalid_type", run, "engine.dirty 必须是 bool")
    same(field(model, "api_name", run), config["model"], f"{run}.model.api_name")
    text(field(model, "backend_quant", run), f"{run}.model.backend_quant")
    same(field(model, "tokenizer", run), config["tokenizer"], f"{run}.model.tokenizer")
    same(field(load, "dataset", run), match["ds"], f"{run}.load.dataset")
    repeat = integer(field(load, "repeat", run), f"{run}.load.repeat", 1)
    same(repeat, int(match["rep"]), f"{run}.load.repeat")
    for key in ("mode", "requests", "warmup_secs", "max_tokens", "concurrency", "rate"):
        expected = float(config[key]) if key == "rate" and config[key] is not None else config[key]
        same(field(load, key, run), expected, f"{run}.load.{key}", exact=True)
    same(load.get("arrival_seed"), config.get("arrival_seed"), f"{run}.load.arrival_seed")
    if root_metadata is not None and config["engine"] == "paged-serving":
        same(engine["commit"], root_metadata["commits"]["paged_serving"], f"{run}.engine.commit")
        if "paged_serving_dirty" in root_metadata["commits"]:
            same(engine["dirty"], root_metadata["commits"]["paged_serving_dirty"], f"{run}.engine.dirty")
    if formal:
        require(re.fullmatch(r"[a-fA-F0-9]{40}", engine["commit"]) is not None,
                "invalid_provenance", run, "正式结果缺少完整引擎 commit")
        require(isinstance(model.get("sha256"), str) and re.fullmatch(r"[a-fA-F0-9]{64}", model["sha256"]),
                "invalid_provenance", run, "正式结果缺少 64 位模型 SHA-256")
        require(config["warmup_secs"] >= 30, "insufficient_warmup", run, "正式结果预热必须 >= 30 秒")
    wall = number(field(summary, "measurement_wall_secs", loc), f"{loc}.measurement_wall_secs")
    require(wall > 0, "invalid_wall_time", loc, "完整测量墙钟必须 > 0")
    records = validate_records(run / "per_request.jsonl", config, wall, warnings)
    same(len(records), config["requests"], f"{loc}.config.requests")
    successful = [record for record in records if record["ok"]]
    requests = mapping(field(summary, "requests", loc), f"{loc}.requests")
    for key, value in (("total", len(records)), ("success", len(successful)),
                       ("failed", len(records) - len(successful)),
                       ("success_rate_pct", 100.0 * len(successful) / len(records))):
        same(field(requests, key, loc), value, f"{loc}.requests.{key}")
    errors = mapping(field(summary, "errors", loc), f"{loc}.errors")
    expected_errors = dict(Counter(record["error_class"] for record in records if not record["ok"]))
    for key, count in errors.items():
        integer(count, f"{loc}.errors.{key}", 1)
    same(errors, expected_errors, f"{loc}.errors")
    known = [record for record in successful if record["completion_tokens"] is not None]
    token_summary = mapping(field(summary, "completion_tokens", loc), f"{loc}.completion_tokens")
    token_total = sum(record["completion_tokens"] for record in known)
    for key, value in (("known_requests", len(known)), ("successful_requests", len(successful)),
                       ("coverage_pct", 100.0 * len(known) / len(successful) if successful else 0.0),
                       ("total", token_total)):
        same(field(token_summary, key, loc), value, f"{loc}.completion_tokens.{key}")
    source_counts = mapping(field(token_summary, "source_counts", loc), f"{loc}.completion_tokens.source_counts")
    for key, count in source_counts.items():
        integer(count, f"{loc}.completion_tokens.source_counts.{key}", 1)
    same(source_counts, dict(Counter(record["tokens_source"] for record in known)),
         f"{loc}.completion_tokens.source_counts")
    check_metric(summary, "ttft_ms", [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None], loc)
    check_metric(summary, "inter_chunk_latency_ms", [v for r in successful for v in r["inter_chunk_latency_ms"]], loc)
    check_metric(summary, "tpot_ms", [(r["duration_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
                                   for r in known if r["ttft_ms"] is not None and r["completion_tokens"] > 1], loc)
    same(field(summary, "itl_ms", loc), None, f"{loc}.itl_ms")
    throughput = mapping(field(summary, "throughput", loc), f"{loc}.throughput")
    same(field(throughput, "successful_requests_per_second", loc), len(successful) / wall,
         f"{loc}.throughput.successful_requests_per_second")
    same(field(throughput, "output_tokens_per_second", loc),
         token_total / wall if successful and len(known) == len(successful) else None,
         f"{loc}.throughput.output_tokens_per_second")
    tag = match["tag"] if config["mode"] == "closed" else f"rate{float(config['rate'])!r}"
    group = (match["engine"], match["mode"], tag, match["ds"])
    signature = {f"config.{key}": config[key] for key in CONFIG_FIELDS}
    signature["config.arrival_schedule"] = config.get("arrival_schedule")
    signature.update({f"engine.{key}": engine[key] for key in ("commit", "dirty")})
    signature.update({f"model.{key}": model.get(key)
                      for key in ("path", "sha256", "backend_quant", "tokenizer")})
    return {"path": run, "group": group, "repeat": repeat, "summary": summary, "signature": signature}


def convergence(values, repeats):
    if not values:
        return {"status": "unavailable", "samples": 0, "relative_range_pct": None}
    mean = sum(value / len(values) for value in values)
    relative_range = (max(values) - min(values)) / mean if mean else 0.0
    status = ("insufficient_repeats" if len(values) < 3 or len(values) != repeats else
              "non_converged" if relative_range > 0.1 else "converged")
    return {"status": status, "samples": len(values), "mean": mean, "min": min(values),
            "max": max(values), "relative_range_pct": 100 * relative_range}


def audit_package(root, formal=False):
    root = Path(root)
    report = {"schema_version": 1, "root": str(root), "formal": formal,
              "errors": [], "warnings": [], "groups": [],
              "convergence_rule": {"statistic": "(max-min)/mean", "threshold_pct": 10,
                                   "minimum_repeats": 3}}
    metadata = None
    try:
        metadata = load_json(root / "metadata.json")
        same(field(metadata, "schema_version", root), 1, f"{root}/metadata.json.schema_version")
        text(field(metadata, "date", root), f"{root}.date")
        try:
            date.fromisoformat(metadata["date"])
        except ValueError as error:
            raise EvidenceError("invalid_date", root, "date 必须是有效 ISO 日期") from error
        for key in ("hardware", "software", "commits", "build"):
            mapping(field(metadata, key, root), f"{root}.{key}")
        for section, keys in (("hardware", ("gpu", "vram", "driver")),
                              ("software", ("cuda_toolkit",)),
                              ("build", ("profile", "cuda_archs"))):
            for key in keys:
                text(field(metadata[section], key, root), f"{root}.{section}.{key}")
        for key in ("dirty", "paged_serving_dirty", "tiny_llm_dirty"):
            if key == "dirty" or key in metadata["commits"]:
                require(type(field(metadata["commits"], key, root)) is bool,
                        "invalid_type", f"{root}.commits.{key}", "dirty 状态必须是 bool")
        if metadata["commits"]["dirty"]:
            report["warnings"].append({"code": "dirty_provenance", "location": str(root),
                                       "message": "实验声明 dirty，commit 不能单独重建当时源码"})
        for key in ("paged_serving", "tiny_llm"):
            value = text(field(metadata["commits"], key, root), f"{root}.commits.{key}")
            if formal:
                require(re.fullmatch(r"[a-fA-F0-9]{40}", value) is not None,
                        "invalid_provenance", root, f"正式结果缺少完整 {key} commit")
    except EvidenceError as error:
        report["errors"].append(error.issue)
        metadata = None
    directories = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    runs = [path for path in directories if RUN_DIR_RE.fullmatch(path.name)]
    for path in directories:
        if not RUN_DIR_RE.fullmatch(path.name) and any((path / name).exists() for name in RUN_FILES[:3]):
            report["errors"].append({"code": "invalid_run_name", "location": str(path),
                                     "message": "含 run 结果的目录名不合法，不能静默跳过"})
    validated = []
    if not runs:
        report["errors"].append({"code": "missing_runs", "location": str(root), "message": "没有匹配的 run 目录"})
    for run in runs:
        try:
            validated.append(validate_run(run, metadata, formal, report["warnings"]))
        except EvidenceError as error:
            report["errors"].append(error.issue)
    grouped = {}
    series = {}
    for run in validated:
        grouped.setdefault(run["group"], []).append(run)
        engine, mode, _, dataset = run["group"]
        series.setdefault((engine, mode, dataset), []).append(run)
    for members in series.values():
        reference = members[0]["signature"]
        for member in members[1:]:
            for key, value in reference.items():
                if key not in ("config.concurrency", "config.rate") and member["signature"][key] != value:
                    report["errors"].append({"code": "incompatible_repeats", "location": str(member["path"]),
                                             "message": f"同引擎/模式/数据集系列的 {key} 不同：{value!r} / {member['signature'][key]!r}"})
    for group, members in sorted(grouped.items()):
        repeats = [member["repeat"] for member in members]
        if len(set(repeats)) != len(repeats):
            report["errors"].append({"code": "duplicate_repeat", "location": "/".join(group), "message": "重复编号不能重复"})
        if formal and len(members) < 3:
            report["errors"].append({"code": "insufficient_repeats", "location": "/".join(group), "message": "正式组合至少需要 3 个完整重复"})
        metrics = {}
        for name, (section, key) in CONVERGENCE_METRICS.items():
            values = [member["summary"][section][key] for member in members
                      if member["summary"][section][key] is not None]
            metrics[name] = convergence(values, len(members))
        signature = members[0]["signature"]
        report["groups"].append({"key": list(group), "repeats": repeats, "metrics": metrics,
                                 "provenance": {"engine_commit": signature["engine.commit"],
                                                "model_sha256": signature["model.sha256"],
                                                "backend_quant": signature["model.backend_quant"],
                                                "arrival_schedule": signature["config.arrival_schedule"]}})
    if formal:
        for name in ("report.md", "summary_table.csv"):
            path = root / name
            try:
                require(path.is_file() and bool(read_text(path).strip()), "missing_formal_artifact", path,
                        "缺少非空正式产物")
            except EvidenceError as error:
                report["errors"].append(error.issue)
        if len(list(root.glob("*.png"))) < 2:
            report["errors"].append({"code": "missing_formal_artifact", "location": str(root), "message": "正式结果至少需要两张图"})
    report["runs"] = {"discovered": len(runs), "validated": len(validated)}
    report["status"] = "failed" if report["errors"] else "passed"
    return report, validated, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--json", action="store_true", help="只向 stdout 输出机器可读审计 JSON")
    args = parser.parse_args(argv)
    report, _, _ = audit_package(args.root, args.formal)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"结果包语义校验{'失败' if report['errors'] else '通过'}：{args.root}（{report['runs']['validated']}/{report['runs']['discovered']} 个 run）")
        for issue in report["errors"] + report["warnings"]:
            print(f"- {issue['code']}: {issue['location']}: {issue['message']}")
        for group in report["groups"]:
            states = ", ".join(f"{key}={metric['status']}" for key, metric in group["metrics"].items())
            print(f"- {'/'.join(group['key'])}: {states}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
