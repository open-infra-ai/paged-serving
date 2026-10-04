"""离线结果夹具，不产生真实 GPU 性能证据。"""

from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import plots
import validate_results as validator


HERE = Path(__file__).resolve().parent
ROOT_METADATA = {
    "schema_version": 1, "date": "2026-10-04",
    "hardware": {"gpu": "fixture-only", "vram": "n/a", "driver": "n/a"},
    "software": {"cuda_toolkit": "n/a"},
    "commits": {"paged_serving": "a" * 40, "tiny_llm": "b" * 40, "dirty": False},
    "build": {"profile": "fixture", "cuda_archs": "n/a"},
}


def fixture_records():
    records = []
    for index, (ok, tokens, ttft, duration, interval) in enumerate(
        ((True, 3, 10.0, 80.0, 5.0), (True, None, 20.0, 90.0, 10.0),
         (False, 100, 60.0, 120.0, 15.0))
    ):
        records.append({
            "request_id": 20 + index, "measured_index": index, "ok": ok,
            "error_class": None if ok else "no_done",
            "error_detail": None if ok else "stream ended without DONE",
            "ttft_ms": ttft, "duration_ms": duration, "chunks": 2,
            "inter_chunk_latency_ms": [interval], "completion_tokens": tokens,
            "tokens_source": "usage" if tokens is not None else None,
            "finish_reason": "length" if ok else None, "prompt_tokens_meta": 10 + index,
        })
    return records


def fixture_metric(values):
    values = sorted(values)
    return {"samples": len(values), **{
        f"p{p}": values[int(p / 100 * (len(values) - 1) + 0.5)] if values else None
        for p in (50, 95, 99)
    }}


def fixture_summary(records, config, wall):
    successful = [r for r in records if r["ok"]]
    known = [r for r in successful if r["completion_tokens"] is not None]
    tokens = sum(r["completion_tokens"] for r in known)
    return {
        "schema_version": 1, "config": config, "measurement_wall_secs": wall,
        "requests": {"total": len(records), "success": len(successful), "failed": len(records) - len(successful),
                     "success_rate_pct": 100.0 * len(successful) / len(records)},
        "errors": dict(Counter(r["error_class"] for r in records if not r["ok"])),
        "ttft_ms": fixture_metric([r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]),
        "inter_chunk_latency_ms": fixture_metric([v for r in successful for v in r["inter_chunk_latency_ms"]]),
        "tpot_ms": fixture_metric([(r["duration_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
                                  for r in known if r["ttft_ms"] is not None and r["completion_tokens"] > 1]),
        "itl_ms": None,
        "completion_tokens": {"known_requests": len(known), "successful_requests": len(successful),
                              "coverage_pct": 100.0 * len(known) / len(successful) if successful else 0.0,
                              "total": tokens, "source_counts": dict(Counter(r["tokens_source"] for r in known))},
        "throughput": {"successful_requests_per_second": len(successful) / wall,
                       "output_tokens_per_second": tokens / wall if successful and len(known) == len(successful) else None},
    }


class ResultSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="serving-result-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write_json(self.root / "metadata.json", ROOT_METADATA)

    def write_json(self, path, value):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def add_run(self, repeat=1, mode="closed", wall=2.0, records=None, load=2, current=False, engine="paged-serving"):
        records = deepcopy(fixture_records() if records is None else records)
        tag = f"c{load}" if mode == "closed" else f"rate{load}"
        path = self.root / f"{engine}_{mode}_{tag}_work_r{repeat}"
        path.mkdir()
        config = {
            "engine": engine, "mode": mode, "base_url": "http://fixture.invalid",
            "model": "fixture", "dataset": "datasets/synth/work.jsonl", "requests": len(records),
            "concurrency": load if mode == "closed" else None,
            "rate": float(load) if mode == "poisson" else None,
            "arrival_seed": 100 + repeat if mode == "poisson" else None,
            "warmup_secs": 30, "max_tokens": 128, "timeout_secs": 120, "tokenizer": None,
        }
        if current:
            config["arrival_schedule"] = "absolute_deadline_seed_reset" if mode == "poisson" else None
            for index, record in enumerate(records):
                record["scheduled_arrival_ms"] = index * 100.0 if mode == "poisson" else None
                record["dispatch_offset_ms"] = index * 100.0 + 10.0
        summary = fixture_summary(records, config, wall)
        metadata = {
            "schema_version": 1, "engine": {"name": engine, "commit": "a" * 40, "dirty": False},
            "model": {"api_name": "fixture", "path": "fixture.gguf", "sha256": "c" * 64,
                      "backend_quant": "fixture-only", "tokenizer": None},
            "load": {key: config[key] for key in ("mode", "requests", "warmup_secs", "max_tokens", "concurrency", "rate", "arrival_seed")},
        }
        metadata["load"].update({"repeat": repeat, "dataset": "work"})
        self.save_run(path, records, summary, metadata)
        return path, records, summary, metadata

    def save_run(self, path, records, summary, metadata):
        self.write_json(path / "summary.json", summary)
        self.write_json(path / "run_metadata.json", metadata)
        (path / "per_request.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        (path / "stdout.log").write_text("fixture only\n", encoding="utf-8")

    def audit(self, formal=False):
        return validator.audit_package(self.root, formal)[0]

    def assert_rejected(self, code=None):
        report = self.audit()
        self.assertEqual(report["status"], "failed", report)
        if code:
            self.assertIn(code, [i["code"] for i in report["errors"]], report)
        with self.assertRaises(ValueError):
            plots.collect_groups(self.root)

    def formal_artifacts(self):
        (self.root / "report.md").write_text("CPU fixture only\n", encoding="utf-8")
        (self.root / "summary_table.csv").write_text("fixture\n", encoding="utf-8")
        for name in ("one.png", "two.png"):
            (self.root / name).write_bytes(b"fixture presence check, not an actual plot")

    def test_success_failure_partial_tokens_and_samples(self):
        _, _, summary, _ = self.add_run()
        self.assertEqual(summary["completion_tokens"]["total"], 3)
        self.assertEqual(summary["completion_tokens"]["coverage_pct"], 50.0)
        self.assertEqual(summary["ttft_ms"], {"samples": 2, "p50": 20.0, "p95": 20.0, "p99": 20.0})
        self.assertEqual(summary["tpot_ms"]["p50"], 35.0)
        report = self.audit()
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["groups"][0]["metrics"]["output_tokens_per_second"]["status"], "unavailable")
        self.assertIn("legacy_timing_unavailable", [i["code"] for i in report["warnings"]])

    def test_half_up_percentile_not_python_bankers_round(self):
        self.assertEqual(validator.percentile([10.0, 20.0], 50), 20.0)
        self.assertEqual(validator.percentile([1, 2, 3, 4, 5, 6], 50), 4)
        self.assertIsNone(validator.percentile([], 95))

    def test_json_integer_latency_equals_float_latency(self):
        path, records, summary, meta = self.add_run(mode="poisson")
        for record in records:
            record["ttft_ms"] = int(record["ttft_ms"])
        summary["config"]["rate"] = 2
        self.save_run(path, records, summary, meta)
        self.assertEqual(self.audit()["status"], "passed")

    def test_all_failed_is_valid_without_fake_performance(self):
        records = [fixture_records()[2]]
        records[0]["measured_index"] = 0
        self.add_run(records=records)
        self.assertEqual(self.audit()["status"], "passed")

    def test_known_zero_tokens_and_no_text_are_valid(self):
        record = fixture_records()[0]
        record.update({"chunks": 0, "ttft_ms": None, "inter_chunk_latency_ms": [], "completion_tokens": 0})
        _, _, summary, _ = self.add_run(records=[record])
        self.assertEqual(summary["throughput"]["output_tokens_per_second"], 0.0)
        self.assertEqual(self.audit()["status"], "passed")

    def test_corrupt_counts_and_error_breakdown(self):
        path, records, original, metadata = self.add_run()
        for section, key, value in (("requests", "total", 4), ("requests", "failed", 0),
                                    ("requests", "success_rate_pct", 100.0),
                                    ("completion_tokens", "total", 103),
                                    ("completion_tokens", "coverage_pct", 100.0),
                                    ("completion_tokens", "source_counts", {"usage": True}),
                                    ("errors", "http_429", 1), ("errors", "no_done", False)):
            with self.subTest(section=section, key=key):
                summary = deepcopy(original)
                summary[section][key] = value
                self.save_run(path, records, summary, metadata)
                self.assert_rejected()

    def test_percentiles_and_sample_counts_are_recomputed(self):
        path, records, original, metadata = self.add_run()
        for metric in ("ttft_ms", "inter_chunk_latency_ms", "tpot_ms"):
            for key in ("samples", "p50", "p95", "p99"):
                with self.subTest(metric=metric, key=key):
                    summary = deepcopy(original)
                    summary[metric][key] += 1
                    self.save_run(path, records, summary, metadata)
                    self.assert_rejected("value_mismatch")

    def test_partial_token_coverage_cannot_publish_subset_throughput(self):
        path, records, summary, metadata = self.add_run()
        summary["throughput"]["output_tokens_per_second"] = 1.5
        self.save_run(path, records, summary, metadata)
        self.assert_rejected("value_mismatch")

    def test_throughput_uses_wall_not_sum_of_request_durations(self):
        records = fixture_records()[:2]
        records[1].update({"completion_tokens": 3, "tokens_source": "tokenizer_text"})
        path, records, summary, metadata = self.add_run(records=records)
        self.assertEqual(summary["throughput"]["output_tokens_per_second"], 3.0)
        self.assertEqual(self.audit()["status"], "passed")
        summary["throughput"]["output_tokens_per_second"] = 6 / 0.170
        self.save_run(path, records, summary, metadata)
        self.assert_rejected("value_mismatch")

    def test_request_order_duplicates_warmup_and_missing_line(self):
        path, original, summary, metadata = self.add_run()
        for change in ("warmup", "reorder", "duplicate_id", "missing_line"):
            with self.subTest(change=change):
                records = deepcopy(original)
                if change == "warmup":
                    records[0]["measured_index"] = None
                elif change == "reorder":
                    records.reverse()
                elif change == "duplicate_id":
                    records[1]["request_id"] = records[0]["request_id"]
                else:
                    records.pop()
                self.save_run(path, records, summary, metadata)
                self.assert_rejected()

    def test_invalid_record_types_and_timing(self):
        path, original, summary, metadata = self.add_run()
        for key, value in (("ok", 1), ("measured_index", False), ("request_id", "20"),
                           ("duration_ms", -1), ("duration_ms", 3000), ("ttft_ms", 100),
                           ("inter_chunk_latency_ms", [100]), ("chunks", 0),
                           ("tokens_source", "chunk_count"), ("completion_tokens", -1),
                           ("completion_tokens", 2**64), ("duration_ms", 10**400),
                           ("error_class", "http_500")):
            with self.subTest(key=key, value=value):
                records = deepcopy(original)
                records[0][key] = value
                self.save_run(path, records, summary, metadata)
                self.assert_rejected()

    def test_missing_and_invalid_json_are_diagnostics(self):
        path, records, summary, metadata = self.add_run()
        for content in ("{", "[]", '{"duration_ms": NaN}', '{"duration_ms": Infinity}', "null", ""):
            with self.subTest(content=content):
                (path / "per_request.jsonl").write_text(content + "\n", encoding="utf-8")
                self.assert_rejected()
        self.save_run(path, records, summary, metadata)
        (path / "summary.json").unlink()
        self.assert_rejected("missing_file")

    def test_duplicate_keys_are_rejected_at_every_depth(self):
        for content in ('{"key": 1, "key": 1}', '{"outer": {"key": 1, "key": 2}}',
                        '{"outer": [{"key": 1, "key": 2}]}', r'{"key": 1, "\u006bey": 2}'):
            with self.subTest(content=content):
                with self.assertRaises(validator.EvidenceError) as caught:
                    validator.decode_json(content, "fixture.json")
                self.assertEqual(caught.exception.issue["code"], "invalid_json")
                self.assertIn("key", caught.exception.issue["message"])

    def test_duplicate_keys_are_rejected_in_all_evidence_files(self):
        run, _, _, _ = self.add_run()
        for path, key, conflicting in ((self.root / "metadata.json", "schema_version", "2"),
                                       (run / "summary.json", "schema_version", "2"),
                                       (run / "run_metadata.json", "schema_version", "2"),
                                       (run / "per_request.jsonl", "ok", "false")):
            with self.subTest(path=path.name):
                original = path.read_text(encoding="utf-8")
                path.write_text(original.replace("{", f'{{"{key}": {conflicting}, ', 1), encoding="utf-8")
                try:
                    self.assert_rejected("invalid_json")
                    issue = next(i for i in self.audit()["errors"] if i["code"] == "invalid_json")
                    self.assertIn(str(path), issue["location"])
                    self.assertIn(key, issue["message"])
                    if path.suffix == ".jsonl":
                        self.assertTrue(issue["location"].endswith(":1"))
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_overflowing_literals_are_rejected_even_in_uninterpreted_fields(self):
        run, _, _, _ = self.add_run()
        for path in (self.root / "metadata.json", run / "summary.json",
                     run / "run_metadata.json", run / "per_request.jsonl"):
            original = path.read_text(encoding="utf-8")
            for literal in ("1e400", "-1e400"):
                with self.subTest(path=path.name, literal=literal):
                    path.write_text(original.replace("{", f'{{"extra": {{"values": [{literal}]}}, ', 1),
                                    encoding="utf-8")
                    try:
                        self.assert_rejected("invalid_json")
                    finally:
                        path.write_text(original, encoding="utf-8")
        self.assertEqual(validator.decode_json('{"extra": 1e308}', "fixture.json"), {"extra": 1e308})

    def test_ambiguous_json_cli_refuses_audit_and_plot_without_writes(self):
        run, _, _, _ = self.add_run()
        path = run / "summary.json"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("{", '{"schema_version": 2, ', 1), encoding="utf-8")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = subprocess.run([sys.executable, str(HERE / "validate_results.py"), "--json", str(self.root)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "failed")
        self.assertIn("invalid_json", [i["code"] for i in report["errors"]])
        self.assertNotIn("Traceback", result.stderr)
        output = self.root / "not-created"
        result = subprocess.run([sys.executable, str(HERE / "plots.py"), str(self.root), "--out-dir", str(output)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("invalid_json", result.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_unreadable_utf8_is_diagnostic(self):
        path, _, _, _ = self.add_run()
        (path / "per_request.jsonl").write_bytes(b"\xff\xfe")
        self.assert_rejected("unreadable_file")

    def test_root_metadata_and_schema_validation(self):
        path, records, original, meta = self.add_run()
        for key, value in (("hardware", []), ("commits", None), ("date", "not-a-date"),
                           ("schema_version", True), ("build", {})):
            with self.subTest(key=key):
                metadata = deepcopy(ROOT_METADATA)
                metadata[key] = value
                self.write_json(self.root / "metadata.json", metadata)
                self.assert_rejected()
        self.write_json(self.root / "metadata.json", ROOT_METADATA)
        for value in (2, True, None):
            with self.subTest(schema=value):
                summary = deepcopy(original)
                summary["schema_version"] = value
                self.save_run(path, records, summary, meta)
                self.assert_rejected("value_mismatch")

    def test_invalid_wall_and_config(self):
        path, records, original, metadata = self.add_run()
        for wall in (0, -1, True, None, 1e309):
            with self.subTest(wall=wall):
                summary = deepcopy(original)
                summary["measurement_wall_secs"] = wall
                self.save_run(path, records, summary, metadata)
                self.assert_rejected()
        summary = deepcopy(original)
        summary["config"]["requests"] = True
        self.save_run(path, records, summary, metadata)
        self.assert_rejected("invalid_integer")

    def test_metadata_conflicts(self):
        path, records, summary, original = self.add_run(mode="poisson", load=2)
        for section, key, value in (("engine", "commit", "b" * 40), ("engine", "dirty", "false"),
                                    ("model", "api_name", "different"), ("load", "rate", 3.0),
                                    ("load", "arrival_seed", 99), ("load", "repeat", 2)):
            with self.subTest(section=section, key=key):
                metadata = deepcopy(original)
                metadata[section][key] = value
                self.save_run(path, records, summary, metadata)
                self.assert_rejected()

    def test_repeats_cannot_mix_model_quant_commit_or_load(self):
        self.add_run(repeat=1)
        path, records, original_summary, original_meta = self.add_run(repeat=2)
        for change in ("hash", "quant", "timeout", "dataset", "schedule"):
            with self.subTest(change=change):
                summary, meta = deepcopy(original_summary), deepcopy(original_meta)
                if change == "hash":
                    meta["model"]["sha256"] = "d" * 64
                elif change == "quant":
                    meta["model"]["backend_quant"] = "other-quant"
                elif change == "timeout":
                    summary["config"]["timeout_secs"] = 20
                elif change == "dataset":
                    summary["config"]["dataset"] = "another/work.jsonl"
                else:
                    summary["config"]["arrival_schedule"] = "unsupported"
                self.save_run(path, records, summary, meta)
                self.assert_rejected()

    def test_load_curve_cannot_mix_models_at_different_concurrency(self):
        self.add_run(load=1)
        path, records, summary, meta = self.add_run(load=4)
        meta["model"]["sha256"] = "d" * 64
        self.save_run(path, records, summary, meta)
        self.assert_rejected("incompatible_repeats")

    def test_external_engine_repeats_cannot_mix_commits(self):
        self.add_run(engine="llama-server")
        path, records, summary, meta = self.add_run(engine="llama-server", repeat=2)
        meta["engine"]["commit"] = "e" * 40
        self.save_run(path, records, summary, meta)
        self.assert_rejected("incompatible_repeats")

    def test_different_engine_quantization_is_preserved_not_averaged(self):
        self.add_run()
        path, records, summary, meta = self.add_run(engine="llama-server")
        meta["engine"]["commit"] = "e" * 40
        meta["model"]["backend_quant"] = "other-fixture-quant"
        self.save_run(path, records, summary, meta)
        groups, report, _ = plots.collect_groups(self.root)
        self.assertEqual(len(groups), 2)
        declarations = {g["key"][0]: g["provenance"] for g in report["groups"]}
        self.assertEqual(declarations["llama-server"]["backend_quant"], "other-fixture-quant")
        self.assertEqual(declarations["llama-server"]["engine_commit"], "e" * 40)

    def test_poisson_seed_may_vary_by_repeat_but_schedule_may_not(self):
        for repeat in (1, 2, 3):
            self.add_run(repeat=repeat, mode="poisson", current=True)
        report = self.audit()
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["warnings"], [])
        path = self.root / "paged-serving_poisson_rate2_work_r3"
        summary = json.loads((path / "summary.json").read_text())
        summary["config"].pop("arrival_schedule")
        self.write_json(path / "summary.json", summary)
        self.assert_rejected("incompatible_repeats")

    def test_new_timing_must_be_complete_and_inside_window(self):
        path, original, summary, meta = self.add_run(mode="poisson", current=True)
        for key, value in (("scheduled_arrival_ms", 100), ("dispatch_offset_ms", 3000),
                           ("dispatch_offset_ms", None)):
            with self.subTest(key=key, value=value):
                records = deepcopy(original)
                records[0][key] = value
                self.save_run(path, records, summary, meta)
                self.assert_rejected()
        records = deepcopy(original)
        del records[1]["dispatch_offset_ms"]
        self.save_run(path, records, summary, meta)
        self.assert_rejected("missing_field")

    def test_closed_timing_has_no_poisson_schedule(self):
        path, records, summary, meta = self.add_run(current=True)
        self.assertEqual(self.audit()["status"], "passed")
        records[0]["scheduled_arrival_ms"] = 0.0
        self.save_run(path, records, summary, meta)
        self.assert_rejected("value_mismatch")

    def test_legacy_missing_seed_is_warning_not_invented_value(self):
        path, records, summary, meta = self.add_run(mode="poisson")
        del summary["config"]["arrival_seed"]
        del meta["load"]["arrival_seed"]
        self.save_run(path, records, summary, meta)
        report = self.audit()
        self.assertEqual(report["status"], "passed")
        self.assertIn("legacy_seed_unavailable", [i["code"] for i in report["warnings"]])
        self.assertNotIn("arrival_seed", json.loads((path / "summary.json").read_text())["config"])

    def test_current_schedule_cannot_omit_seed(self):
        path, records, summary, meta = self.add_run(mode="poisson", current=True)
        summary["config"].pop("arrival_seed")
        meta["load"].pop("arrival_seed")
        self.save_run(path, records, summary, meta)
        self.assert_rejected("missing_field")

    def test_numeric_tag_alias_cannot_count_as_extra_repeat(self):
        self.add_run(mode="poisson", load=2)
        path, _, _, _ = self.add_run(mode="poisson", load=2, repeat=2)
        path.rename(self.root / "paged-serving_poisson_rate2.0_work_r1")
        meta = json.loads((self.root / "paged-serving_poisson_rate2.0_work_r1/run_metadata.json").read_text())
        meta["load"]["repeat"] = 1
        self.write_json(self.root / "paged-serving_poisson_rate2.0_work_r1/run_metadata.json", meta)
        self.assert_rejected("duplicate_repeat")

    def test_nearby_rates_are_distinct_and_scientific_notation_is_valid(self):
        for rate in (0.1234561, 0.1234562, 1e-6):
            self.add_run(mode="poisson", load=rate)
        report = self.audit()
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(len(report["groups"]), 3)

    def test_config_comparison_does_not_use_metric_tolerance(self):
        path, records, summary, meta = self.add_run(mode="poisson", load=2)
        summary["config"]["rate"] = 2.0000001
        meta["load"]["rate"] = 2.0000001
        self.save_run(path, records, summary, meta)
        self.assert_rejected("value_mismatch")

    def test_invalid_directory_is_not_silently_ignored(self):
        path, _, _, _ = self.add_run()
        path.rename(self.root / "wrong-name")
        self.assert_rejected("invalid_run_name")

    def test_formal_requires_repeats_warmup_hash_and_artifacts(self):
        self.add_run()
        self.assertEqual(self.audit(formal=True)["status"], "failed")
        self.formal_artifacts()
        self.add_run(repeat=2)
        path, records, summary, meta = self.add_run(repeat=3)
        self.assertEqual(self.audit(formal=True)["status"], "passed")
        for change in ("warmup", "hash"):
            with self.subTest(change=change):
                changed_summary, changed_meta = deepcopy(summary), deepcopy(meta)
                if change == "warmup":
                    changed_summary["config"]["warmup_secs"] = 0
                    changed_meta["load"]["warmup_secs"] = 0
                else:
                    changed_meta["model"]["sha256"] = "not-a-sha"
                self.save_run(path, records, changed_summary, changed_meta)
                self.assertEqual(self.audit(formal=True)["status"], "failed")

    def test_non_converged_is_valid_negative_evidence(self):
        for repeat, wall in enumerate((1.0, 2.0, 4.0), 1):
            self.add_run(repeat=repeat, wall=wall)
        self.formal_artifacts()
        report = self.audit(formal=True)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["groups"][0]["metrics"]["successful_requests_per_second"]["status"], "non_converged")
        plots.collect_groups(self.root)

    def test_convergence_boundary_zero_missing_and_few_repeats(self):
        self.assertEqual(validator.convergence([95.0, 100.0, 105.0], 3)["status"], "converged")
        self.assertEqual(validator.convergence([94.0, 100.0, 106.0], 3)["status"], "non_converged")
        self.assertEqual(validator.convergence([0, 0, 0], 3)["relative_range_pct"], 0.0)
        self.assertEqual(validator.convergence([1, 1], 3)["status"], "insufficient_repeats")
        self.assertEqual(validator.convergence([], 3)["status"], "unavailable")
        self.assertEqual(validator.convergence([1e308, 1e308, 1e308], 3)["status"], "converged")

    def test_plot_aggregation_does_not_drop_unknown_repeats(self):
        summaries = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        self.assertIsNone(plots.aggregate(summaries, "x"))
        self.assertEqual(plots.spread(summaries, "x"), (None, None, None))

    def test_cli_json_exit_code_and_no_writes(self):
        path, records, summary, metadata = self.add_run()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        args = [sys.executable, str(HERE / "validate_results.py"), "--json", str(self.root)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "passed")
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        summary["requests"]["success"] = 99
        self.save_run(path, records, summary, metadata)
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["status"], "failed")
        output = self.root / "not-created"
        result = subprocess.run([sys.executable, str(HERE / "plots.py"), str(self.root), "--out-dir", str(output)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(output.exists())

    def test_missing_root_returns_json_diagnostic_not_traceback(self):
        report, _, _ = validator.audit_package(self.root / "absent")
        self.assertEqual(report["status"], "failed")
        self.assertIn("missing_runs", [i["code"] for i in report["errors"]])

    def test_historical_formal_matrix_matches_convergence_and_csv(self):
        root = HERE / "results/2026-09-07-RTX3060Laptop-paged-serving-p2-batch-postprocess-streaming"
        report, runs, _ = validator.audit_package(root, formal=True)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(report["runs"], {"discovered": 21, "validated": 21})
        self.assertEqual(sum(run["summary"]["errors"].get("http_429", 0) for run in runs), 83)
        statuses = {g["key"][2]: g["metrics"]["ttft_p95_ms"]["status"] for g in report["groups"]}
        self.assertEqual(statuses, {"c1": "converged", "c4": "converged", "c2": "non_converged",
                                   "c8": "non_converged", "rate0.32": "non_converged",
                                   "rate0.64": "non_converged", "rate1.28": "non_converged"})
        groups, _, _ = plots.collect_groups(root)
        with (root / "summary_table.csv").open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                summaries = groups[(row["engine"], row["mode"], row["tag"], row["dataset"])]
                for column, path in (("ttft_p95", ("ttft_ms", "p95")),
                                     ("throughput_tok_s", ("throughput", "output_tokens_per_second")),
                                     ("throughput_req_s", ("throughput", "successful_requests_per_second"))):
                    self.assertEqual(f"{plots.aggregate(summaries, *path):.3f}", row[column])


if __name__ == "__main__":
    unittest.main()
