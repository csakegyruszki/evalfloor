"""Independent black-box contract tests for run_eval (stdlib only)."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_eval

METRICS = (
    "tool_calls_total", "subagents", "num_turns", "output_tokens", "cost_usd",
    "duration_ms", "permission_denials", "tool_errors",
)


class ContractBlackBox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.results = self.root / "results"
        self.runs = self.results / "runs"
        self.old_results = getattr(run_eval, "RESULTS_DIR", None)
        self.old_runs = getattr(run_eval, "RUNS_DIR", None)
        run_eval.RESULTS_DIR = self.results
        run_eval.RUNS_DIR = self.runs
        # Never let a test reach the real Claude CLI (a live, billed call).
        self.old_run_once, self.old_bin = run_eval.run_once, getattr(run_eval, 'CLAUDE_BIN', None)
        def _guard(*args, **kwargs):
            raise AssertionError('test attempted to invoke the Claude CLI')
        run_eval.run_once = _guard
        run_eval.CLAUDE_BIN = 'nonexistent-claude-binary-for-tests'

    def tearDown(self):
        run_eval.RESULTS_DIR = self.old_results
        run_eval.RUNS_DIR = self.old_runs
        run_eval.run_once, run_eval.CLAUDE_BIN = self.old_run_once, self.old_bin
        self.tmp.cleanup()

    def invoke(self, *args):
        old_argv = sys.argv
        out = io.StringIO()
        try:
            sys.argv = ["run_eval.py", *args]
            with contextlib.redirect_stdout(out):
                rc = run_eval.main()
        finally:
            sys.argv = old_argv
        return rc, out.getvalue()

    def report(self, text):
        start = text.find("{")
        self.assertNotEqual(start, -1, "compare did not emit a JSON report: " + text)
        return json.loads(text[start:])

    def row(self, run_id, cost=1.0, pair_id=None, denials=0, errors=0):
        row = dict(zip(METRICS, (1, 0, 1, 10, cost, 100, denials, errors)))
        self._sid = getattr(self, "_sid", 0) + 1
        row.update(rc=0, is_error=False, run_id=run_id, warmup=False,
                   manifest_file="manifest.json", session_id="s%d" % self._sid)
        if pair_id is not None:
            row["pair_id"] = pair_id
        return row

    def cohort(self, name, variant, rows, *, pair_id=None, tree="tree", cwd=None):
        directory = self.runs / name
        directory.mkdir(parents=True)
        manifest = {
            "schema_version": 1, "run_id": name, "model": "model", "variant": variant,
            "task_sha256": "task", "claude_version": "cli", "runner_sha256": "runner",
            "cwd": str(cwd or self.root / "work"), "cwd_tree_sha256": tree,
            "cwd_tree_policy": getattr(run_eval, "CWD_TREE_POLICY", "v1"),
            "pair_id": pair_id,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with (directory / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return directory

    def compare(self, baseline, treatment, *extra):
        rc, text = self.invoke("compare", "--baseline", str(baseline), "--treatment",
                               str(treatment), *extra)
        return rc, self.report(text)

    def test_p0_1_failed_rows_invalidate_or_degrade(self):
        base = self.cohort("base", "baseline", [self.row("base") for _ in range(10)])
        rows = [self.row("treat", .1) for _ in range(6)] + [
            {"error": "timeout", "run_id": "treat", "session_id": "x%d" % i, "warmup": False} for i in range(4)]
        treatment = self.cohort("treat", "treatment", rows)
        rc, _ = self.compare(base, treatment)
        self.assertEqual(4, rc)
        rc, report = self.compare(base, treatment, "--allow-failed-runs")
        self.assertNotEqual(0, rc)
        self.assertEqual("DEGRADED_FAILED_RUNS", report["validity"])
        self.assertEqual(4, report["failed_treatment"])
        self.assertNotEqual("PASS", report["verdict"])

    def test_p0_2_zero_tolerance_permission_denials(self):
        base = self.cohort("base", "baseline", [self.row("base", pair_id="p:%d" % i)
                                                   for i in range(6)], pair_id="p")
        treatment = self.cohort("treat", "treatment", [
            self.row("treat", .5, "p:%d" % i, int(i < 5)) for i in range(6)], pair_id="p")
        rc, report = self.compare(base, treatment)
        self.assertEqual(6, rc)
        detail = next(x for x in report["guardrails_detail"]
                      if x.get("metric") == "permission_denials")
        self.assertEqual("zero_tolerance", detail["rule"])
        self.assertEqual(0, detail["baseline_total"])
        self.assertEqual(5, detail["treatment_total"])
        self.assertEqual(5, detail["delta_total"])
        self.assertTrue(detail["fired"])

        equal = self.cohort("equal", "treatment", [self.row("equal", .5, "p:%d" % i)
                                                      for i in range(6)], pair_id="p")
        _, report = self.compare(base, equal)
        detail = next(x for x in report["guardrails_detail"]
                      if x.get("metric") == "permission_denials")
        self.assertFalse(detail["fired"])

    def test_p0_3_failed_warmup_stops_before_measurement(self):
        task = self.root / "task.md"
        task.write_text("task", encoding="utf-8")
        calls = []
        old = run_eval.run_once
        def fails_first(*args, **kwargs):
            calls.append((args, kwargs))
            return {"error": "timeout", "is_error": True, "rc": 1}
        run_eval.run_once = fails_first
        try:
            rc, _ = self.invoke("run", "--task", str(task), "--variant", "v", "--model", "m",
                                "--cwd", str(self.root), "--warmup", "1", "--repeat", "2",
                                "--run-id", "warmup-fail")
        finally:
            run_eval.run_once = old
        self.assertEqual(3, rc)
        metrics = self.runs / "warmup-fail" / "metrics.jsonl"
        if metrics.exists():
            rows = [json.loads(x) for x in metrics.read_text(encoding="utf-8").splitlines() if x]
            self.assertFalse(any(not x.get("warmup", False) for x in rows))
        self.assertEqual(1, len(calls))

    def test_p1_a_cwd_tree_hash_contract_and_compare_gate(self):
        one, two = self.root / "one", self.root / "two"
        one.mkdir(); two.mkdir()
        for root in (one, two):
            (root / "data.txt").write_bytes(b"same")
        first, second = run_eval.compute_cwd_tree_sha256(one), run_eval.compute_cwd_tree_sha256(two)
        self.assertEqual(first, second)
        (two / "data.txt").write_bytes(b"diff")
        self.assertNotEqual(first, run_eval.compute_cwd_tree_sha256(two))
        (two / "data.txt").write_bytes(b"same")
        stable = run_eval.compute_cwd_tree_sha256(two)
        for name in (".pytest_cache", "__pycache__", ".git", "results"):
            d = two / name; d.mkdir(); (d / "ignored").write_bytes(b"new")
        os.utime(two / "data.txt", None)
        self.assertEqual(stable, run_eval.compute_cwd_tree_sha256(two))
        target = self.root / "external-target"; target.write_bytes(b"a")
        link = two / "link"
        try:
            link.symlink_to("../external-target")
        except OSError:
            self.skipTest("OS refuses symlink creation")
        linked = run_eval.compute_cwd_tree_sha256(two)
        target.write_bytes(b"b")
        self.assertEqual(linked, run_eval.compute_cwd_tree_sha256(two))
        link.unlink(); link.symlink_to("data.txt")
        self.assertNotEqual(linked, run_eval.compute_cwd_tree_sha256(two))

        base = self.cohort("base", "b", [self.row("base") for _ in range(6)], tree="a")
        treat = self.cohort("treat", "t", [self.row("treat") for _ in range(6)], tree="b")
        self.assertEqual(4, self.compare(base, treat)[0])
        unavailable = self.cohort("unavailable", "t", [self.row("unavailable") for _ in range(6)],
                                  tree="unavailable:permission", cwd=self.root / "work")
        base2 = self.cohort("base2", "b", [self.row("base2") for _ in range(6)],
                            tree="unavailable:other", cwd=self.root / "work")
        self.assertEqual(4, self.compare(base2, unavailable)[0])
        self.assertNotEqual(4, self.compare(base2, unavailable, "--allow-unverified-cwd")[0])

    def test_p1_b_pairing_mismatch_is_invalid(self):
        base = self.cohort("base", "b", [self.row("base", pair_id="k:%d" % i) for i in range(6)], pair_id="k")
        bad = self.cohort("bad", "t", [self.row("bad", pair_id="z:%d" % i) for i in range(6)], pair_id="z")
        self.assertEqual(4, self.compare(base, bad)[0])
        plain = self.cohort("plain", "t", [self.row("plain") for _ in range(6)])
        plain_base = self.cohort("plain-base", "b", [self.row("plain-base") for _ in range(6)])
        self.assertNotEqual(4, self.compare(plain_base, plain)[0])

    def test_p1_c_run_id_is_safe_basename(self):
        task = self.root / "task.md"; task.write_text("task", encoding="utf-8")
        for value in ("../escaped", "a/b", ".."):
            rc, _ = self.invoke("run", "--task", str(task), "--variant", "v", "--model", "m",
                                "--cwd", str(self.root), "--run-id", value)
            self.assertEqual(2, rc)
        self.assertFalse((self.results / "escaped").exists())
        old = run_eval.run_once
        run_eval.run_once = lambda *args, **kwargs: self.row("ok-1.2_x")
        try:
            rc, _ = self.invoke("run", "--task", str(task), "--variant", "v", "--model", "m",
                                "--cwd", str(self.root), "--warmup", "0", "--repeat", "1",
                                "--run-id", "ok-1.2_x")
        finally:
            run_eval.run_once = old
        self.assertEqual(0, rc)
        self.assertTrue((self.runs / "ok-1.2_x").is_dir())

    def test_p2_paired_estimand_matches_ci(self):
        diffs = [-.9, .1, .1, .1, .1, .1]
        base = self.cohort("base", "b", [self.row("base", 1, "p:%d" % i) for i in range(6)], pair_id="p")
        treat = self.cohort("treat", "t", [self.row("treat", 1 + d, "p:%d" % i)
                                              for i, d in enumerate(diffs)], pair_id="p")
        _, report = self.compare(base, treat)
        wanted = sum(diffs) / len(diffs)
        metric = next(item for item in self._objects(report)
                      if item.get("estimand") == "mean_paired_difference"
                      and abs(item.get("delta", 99) - wanted) < 1e-12)
        self.assertEqual("mean_paired_difference", metric["estimand"])
        ci = [metric["ci_low"], metric["ci_high"]]
        self.assertLessEqual(ci[0], metric["delta"])
        self.assertLessEqual(metric["delta"], ci[1])

    def _objects(self, value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from self._objects(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._objects(child)


if __name__ == "__main__":
    unittest.main()
