"""Offline unit tests for run_eval.py. Never invokes the `claude` binary: run_once() is
monkeypatched wherever a `run` invocation is exercised; `compare` tests use synthetic
on-disk cohorts. Must pass on Python 3.9 and Python 3.13."""
from __future__ import annotations

import json
import pathlib
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_eval as ev

def gauss_sample(rnd, mean, cv, n):
    """n draws from a Gaussian with the given mean and coefficient of variation, floored at 0.01
    (costs can't be negative). Shared helper for the calibration/power simulations below."""
    sd = mean * cv
    return [max(0.01, rnd.gauss(mean, sd)) for _ in range(n)]

def make_row(session_id, **overrides):
    """A synthetic completed row; override any field (e.g. cost_usd, pair_id)."""
    row = {"session_id": session_id, "wall_s": 1.0, "rc": 0, "nonjson_lines": 0,
           "tool_calls_total": 3, "subagents": 0, "num_turns": 4, "output_tokens": 200,
           "cost_usd": 0.5, "duration_ms": 1000, "permission_denials": 0, "tool_errors": 0, "is_error": False}
    row.update(overrides)
    return row

class HarnessTestCase(unittest.TestCase):
    """Redirects RESULTS_DIR/RUNS_DIR to a temp dir so tests never touch the real results/."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ev.RESULTS_DIR = self.tmp / "results"
        ev.RUNS_DIR = ev.RESULTS_DIR / "runs"
        ev.CLAUDE_BIN = "unused-in-tests"
        self._orig_run_once = ev.run_once
        self.addCleanup(setattr, ev, "run_once", self._orig_run_once)

    def patch_run_once(self, fn):
        ev.run_once = fn

    def make_task(self, text="do something"):
        task = self.tmp / "task.md"
        task.write_text(text, encoding="utf-8")
        return task

    def write_cohort(self, run_id, variant, rows, model="m1", task_sha="task-sha",
                      claude_version="v1", runner_sha="runner-sha", pair_id=None):
        """Write a manifest.json + metrics.jsonl directly, bypassing cmd_run (for compare tests)."""
        run_dir = ev.RUNS_DIR / run_id
        run_dir.mkdir(parents=True)
        manifest = {"schema_version": ev.SCHEMA_VERSION, "run_id": run_id,
                    "created_utc": "2026-01-01T00:00:00Z", "model": model, "variant": variant,
                    "task_path": "task.md", "task_sha256": task_sha, "agents_sha256": None,
                    "system_prompt_sha256": None, "cwd": str(self.tmp), "claude_version": claude_version,
                    "runner_sha256": runner_sha, "pair_id": pair_id, "cli_options": {}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                row.setdefault("run_id", run_id)
                row.setdefault("manifest_file", "manifest.json")
                fh.write(json.dumps(row) + "\n")
        return run_dir

class TestRunIsolation(HarnessTestCase):
    def test_two_runs_get_distinct_dirs_with_own_manifest_and_rows(self):
        counter = {"n": 0}

        def fake_run_once(prompt, model, cwd, agents_json, timeout_s, system_prompt_file=None):
            counter["n"] += 1
            return make_row(f"sid{counter['n']}")

        self.patch_run_once(fake_run_once)
        task = self.make_task()
        rc1 = ev.main(["run", "--task", str(task), "--variant", "a", "--repeat", "3", "--model", "m1",
                       "--run-id", "id1", "--warmup", "0"])
        rc2 = ev.main(["run", "--task", str(task), "--variant", "b", "--repeat", "3", "--model", "m1",
                       "--run-id", "id2", "--warmup", "0"])
        self.assertEqual(rc1, ev.EXIT_OK)
        self.assertEqual(rc2, ev.EXIT_OK)
        self.assertEqual(sorted(p.name for p in ev.RUNS_DIR.iterdir()), ["id1", "id2"])
        for rid in ("id1", "id2"):
            manifest = json.loads((ev.RUNS_DIR / rid / "manifest.json").read_text(encoding="utf-8"))
            rows = [json.loads(l) for l in (ev.RUNS_DIR / rid / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["run_id"], rid)
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row["run_id"], rid)
                self.assertEqual(row["manifest_file"], "manifest.json")

    def test_duplicate_run_id_fails_before_execution_and_keeps_existing_data(self):
        self.patch_run_once(lambda *a, **k: make_row("s1"))
        task = self.make_task()
        rc1 = ev.main(["run", "--task", str(task), "--variant", "a", "--repeat", "1", "--model", "m1", "--run-id", "dup"])
        self.assertEqual(rc1, ev.EXIT_OK)
        before = (ev.RUNS_DIR / "dup" / "metrics.jsonl").read_text(encoding="utf-8")
        rc2 = ev.main(["run", "--task", str(task), "--variant", "b", "--repeat", "1", "--model", "m1", "--run-id", "dup"])
        self.assertEqual(rc2, ev.EXIT_ARG_ERROR)
        after = (ev.RUNS_DIR / "dup" / "metrics.jsonl").read_text(encoding="utf-8")
        self.assertEqual(before, after)

    def test_out_must_equal_generated_metrics_path(self):
        self.patch_run_once(lambda *a, **k: make_row("s1"))
        task = self.make_task()
        rc = ev.main(["run", "--task", str(task), "--variant", "a", "--model", "m1",
                      "--run-id", "r", "--out", str(self.tmp / "elsewhere.jsonl")])
        self.assertEqual(rc, ev.EXIT_ARG_ERROR)

class TestFailureSemantics(unittest.TestCase):
    def test_mixed_success_and_error_rows(self):
        rows = [make_row("s1", tool_calls_total=3), make_row("s2", tool_calls_total=2),
                make_row("s3", tool_calls_total=3), {"session_id": "bad", "error": "timeout"}]
        summary = ev.summarize(rows)
        self.assertEqual(summary["n_total"], 4)
        self.assertEqual(summary["n_completed"], 3)
        self.assertEqual(summary["n_failed"], 1)
        tc = summary["metrics"]["tool_calls_total"]
        self.assertAlmostEqual(tc["mean"], 2.6666666666666665, places=9)
        self.assertAlmostEqual(tc["stdev"], 0.5773502691896257, places=9)
        self.assertAlmostEqual(tc["cv"], 0.21650635094610965, places=9)

    def test_is_error_and_nonzero_rc_count_as_failed(self):
        self.assertFalse(ev.is_completed_row(make_row("s", is_error=True)))
        self.assertFalse(ev.is_completed_row(make_row("s", rc=1)))
        self.assertFalse(ev.is_completed_row({"session_id": "s"}))  # missing metrics
        self.assertTrue(ev.is_completed_row(make_row("s")))

    def test_zero_mean_metric_gives_null_cv_not_zero(self):
        rows = [make_row(f"s{i}", subagents=0) for i in range(3)]
        cv = ev.summarize(rows)["metrics"]["subagents"]["cv"]
        self.assertIsNone(cv)

class TestCompareIdentity(HarnessTestCase):
    def _pair(self, **treat_overrides):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}") for i in range(6)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}") for i in range(6)], **treat_overrides)
        return base, treat

    def _compare(self, base, treat):
        report_path = self.tmp / "report.json"
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat),
                      "--primary", "cost_usd:decrease", "--min-n", "6", "--report", str(report_path)])
        return rc, json.loads(report_path.read_text(encoding="utf-8"))

    def test_task_sha_mismatch_named_and_no_verdict(self):
        base, treat = self._pair(task_sha="different")
        rc, report = self._compare(base, treat)
        self.assertEqual(rc, ev.EXIT_COHORT_MISMATCH)
        self.assertIn("task_sha256", report["fields"])
        self.assertNotIn("verdict", report)

    def test_model_mismatch_named(self):
        base, treat = self._pair(model="other-model")
        rc, report = self._compare(base, treat)
        self.assertEqual(rc, ev.EXIT_COHORT_MISMATCH)
        self.assertIn("model", report["fields"])

    def test_claude_version_mismatch_named(self):
        base, treat = self._pair(claude_version="v2")
        rc, report = self._compare(base, treat)
        self.assertEqual(rc, ev.EXIT_COHORT_MISMATCH)
        self.assertIn("claude_version", report["fields"])

    def test_matching_pair_reaches_metric_calculation(self):
        base, treat = self._pair()
        rc, report = self._compare(base, treat)
        self.assertIn(rc, (ev.EXIT_OK, ev.EXIT_INCONCLUSIVE, ev.EXIT_FAIL))
        self.assertIn("verdict", report)
        self.assertIn("cost_usd", report["metrics"])

class TestCompareStatistics(HarnessTestCase):
    def test_same_seed_gives_byte_identical_json(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=0.5 + 0.01 * i) for i in range(10)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=0.3 + 0.01 * i) for i in range(10)])
        p1, p2 = self.tmp / "r1.json", self.tmp / "r2.json"
        for path in (p1, p2):
            ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease",
                     "--min-n", "6", "--seed", "7", "--bootstrap", "1000", "--report", str(path)])
        self.assertEqual(p1.read_bytes(), p2.read_bytes())

    def test_paired_selected_when_pair_ids_cover_both_arms(self):
        base_rows = [make_row(f"b{i}", cost_usd=1.0, pair_id=f"p{i}") for i in range(6)]
        treat_rows = [make_row(f"t{i}", cost_usd=0.5, pair_id=f"p{i}") for i in range(6)]
        self.assertTrue(ev.cohort_is_paired(base_rows, treat_rows))
        diffs = ev.pair_diffs("cost_usd", base_rows, treat_rows)
        self.assertEqual(len(diffs), 6)
        self.assertTrue(all(d == -0.5 for d in diffs))

    def test_independent_selected_without_pair_ids(self):
        base_rows = [make_row(f"b{i}", cost_usd=1.0) for i in range(6)]
        treat_rows = [make_row(f"t{i}", cost_usd=0.5) for i in range(6)]
        self.assertFalse(ev.cohort_is_paired(base_rows, treat_rows))

class TestExitCodes(HarnessTestCase):
    def test_run_all_ok_is_zero(self):
        self.patch_run_once(lambda *a, **k: make_row("s1"))
        rc = ev.main(["run", "--task", str(self.make_task()), "--variant", "a", "--model", "m1"])
        self.assertEqual(rc, ev.EXIT_OK)

    def test_run_with_failures_is_three(self):
        self.patch_run_once(lambda *a, **k: {"session_id": "bad", "error": "timeout"})
        rc = ev.main(["run", "--task", str(self.make_task()), "--variant", "a", "--model", "m1"])
        self.assertEqual(rc, ev.EXIT_RUN_FAILED)

    def test_invalid_task_is_two(self):
        rc = ev.main(["run", "--task", str(self.tmp / "missing.md"), "--variant", "a", "--model", "m1"])
        self.assertEqual(rc, ev.EXIT_ARG_ERROR)

    def test_bad_primary_spec_is_two(self):
        base = self.write_cohort("base", "baseline", [make_row("b1")])
        treat = self.write_cohort("treat", "treatment", [make_row("t1")])
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "not-a-metric:decrease"])
        self.assertEqual(rc, ev.EXIT_ARG_ERROR)

    def test_cohort_mismatch_is_four(self):
        base = self.write_cohort("base", "baseline", [make_row("b1")], task_sha="x")
        treat = self.write_cohort("treat", "treatment", [make_row("t1")], task_sha="y")
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease"])
        self.assertEqual(rc, ev.EXIT_COHORT_MISMATCH)

    def test_inconclusive_is_one(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=0.5) for i in range(3)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=0.5) for i in range(3)])
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease"])
        self.assertEqual(rc, ev.EXIT_INCONCLUSIVE)  # 3 rows < default --min-n 6

    def test_pass_is_zero(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=1.0 + 0.001 * i) for i in range(10)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=0.1 + 0.001 * i) for i in range(10)])
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease"])
        self.assertEqual(rc, ev.EXIT_OK)

    def test_fail_is_six(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=1.0 + 0.001 * i) for i in range(10)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=5.0 + 0.001 * i) for i in range(10)])
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease"])
        self.assertEqual(rc, ev.EXIT_FAIL)

class TestReviewFixes(HarnessTestCase):
    """Regressions found in code review."""

    def test_guardrail_adverse_move_fails_through_cli(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=1.0 + 0.001 * i) for i in range(10)])
        treat = self.write_cohort("treat", "treatment",
                                  [make_row(f"t{i}", cost_usd=0.1 + 0.001 * i, tool_calls_total=12 + i) for i in range(10)])
        args = ["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease"]
        self.assertEqual(ev.main(args), ev.EXIT_OK)  # undeclared metrics are descriptive only
        self.assertEqual(ev.main(args + ["--guardrail", "tool_calls_total:increase"]), ev.EXIT_FAIL)
        self.assertEqual(ev.main(args + ["--guardrail", "tool_calls_total:decrease"]), ev.EXIT_OK)

    def test_min_n_gate_blocks_a_clear_effect(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=1.0 + 0.001 * i) for i in range(4)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=0.1 + 0.001 * i) for i in range(4)])
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease"])
        self.assertEqual(rc, ev.EXIT_INCONCLUSIVE)  # 4 < default --min-n 6

    def test_invalid_numeric_options_are_argument_errors(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}") for i in range(6)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}") for i in range(6)])
        common = ["compare", "--baseline", str(base), "--treatment", str(treat), "--primary", "cost_usd:decrease"]
        self.assertEqual(ev.main(common + ["--min-n", "1"]), ev.EXIT_ARG_ERROR)
        self.assertEqual(ev.main(common + ["--min-n", "4"]), ev.EXIT_ARG_ERROR)  # below the hard floor of 5
        self.assertEqual(ev.main(common + ["--effect", "0"]), ev.EXIT_ARG_ERROR)
        self.assertEqual(ev.main(["run", "--task", str(self.make_task()), "--variant", "a", "--repeat", "0"]),
                         ev.EXIT_ARG_ERROR)
        self.assertEqual(ev.main(["run", "--task", str(self.make_task()), "--variant", "a", "--warmup", "-1"]),
                         ev.EXIT_ARG_ERROR)

    def test_pair_id_via_cli_produces_paired_cohorts(self):
        counter = {"n": 0}

        def fake_run_once(prompt, model, cwd, agents_json, timeout_s, system_prompt_file=None):
            counter["n"] += 1
            return make_row(f"s{counter['n']}")

        self.patch_run_once(fake_run_once)
        task = self.make_task()
        for rid, variant in (("pa", "a"), ("pb", "b")):
            ev.main(["run", "--task", str(task), "--variant", variant, "--repeat", "3", "--model", "m1",
                     "--run-id", rid, "--pair-id", "block1", "--warmup", "0"])
        rows = {rid: [json.loads(l) for l in (ev.RUNS_DIR / rid / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                for rid in ("pa", "pb")}
        self.assertTrue(ev.cohort_is_paired(rows["pa"], rows["pb"]))

    def test_planning_cv_uses_baseline_arm_only(self):
        base = [{"cost_usd": v} for v in (1.0, 1.1, 0.9, 1.0, 1.05)]
        treat = [{"cost_usd": v} for v in (0.1, 0.11, 0.09, 0.1, 0.105)]
        res = ev.compare_metric("cost_usd", base, treat, 2, 0, 1000, 0.20, False)
        self.assertAlmostEqual(res["planning_cv"], res["baseline"]["cv"])

class TestWarmup(HarnessTestCase):
    """Cold prompt-cache order effect:warmup rows exist, are
    flagged, and never contribute to a summary or a comparison."""

    def test_warmup_rows_written_flagged_and_excluded_from_summary(self):
        calls = []

        def fake_run_once(prompt, model, cwd, agents_json, timeout_s, system_prompt_file=None):
            calls.append(1)
            # First call (the warmup) reports an inflated cost, like the measured cold-cache run.
            return make_row(f"s{len(calls)}", cost_usd=(1.0 if len(calls) == 1 else 0.5))

        self.patch_run_once(fake_run_once)
        task = self.make_task()
        rc = ev.main(["run", "--task", str(task), "--variant", "a", "--repeat", "3", "--model", "m1",
                      "--run-id", "wu", "--warmup", "2"])
        self.assertEqual(rc, ev.EXIT_OK)
        self.assertEqual(len(calls), 5)  # 2 warmup + 3 measured
        rows = [json.loads(l) for l in (ev.RUNS_DIR / "wu" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 5)
        warmup_rows = [r for r in rows if r["warmup"]]
        measured_rows = [r for r in rows if not r["warmup"]]
        self.assertEqual(len(warmup_rows), 2)
        self.assertEqual(len(measured_rows), 3)
        self.assertTrue(all("pair_id" not in r for r in warmup_rows))
        self.assertTrue(all(r["cost_usd"] == 0.5 for r in measured_rows))
        manifest = json.loads((ev.RUNS_DIR / "wu" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["cli_options"]["warmup"], 2)

    def test_default_warmup_is_one_and_excluded_from_summary_and_exit_code(self):
        calls = []

        def fake_run_once(prompt, model, cwd, agents_json, timeout_s, system_prompt_file=None):
            calls.append(1)
            if len(calls) == 1:
                return {"session_id": "warm", "error": "cold cache blip"}  # warmup fails
            return make_row(f"s{len(calls)}")

        self.patch_run_once(fake_run_once)
        task = self.make_task()
        rc = ev.main(["run", "--task", str(task), "--variant", "a", "--repeat", "3", "--model", "m1", "--run-id", "wd"])
        self.assertEqual(len(calls), 4)  # default warmup=1 + 3 measured
        self.assertEqual(rc, ev.EXIT_OK)  # the failed warmup row must not affect the exit code

    def test_warmup_rows_excluded_from_compare(self):
        rows_a = [make_row(f"a{i}", cost_usd=0.5, pair_id=f"p{i}", warmup=False) for i in range(6)]
        rows_a.append(make_row("a-warm", cost_usd=99.0, warmup=True))  # no pair_id, extreme value
        rows_b = [make_row(f"b{i}", cost_usd=0.5, pair_id=f"p{i}", warmup=False) for i in range(6)]
        rows_b.append(make_row("b-warm", cost_usd=99.0, warmup=True))
        base = self.write_cohort("base", "baseline", rows_a)
        treat = self.write_cohort("treat", "treatment", rows_b)
        report_path = self.tmp / "report.json"
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat),
                      "--primary", "cost_usd:decrease", "--report", str(report_path)])
        self.assertEqual(rc, ev.EXIT_INCONCLUSIVE)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["metrics"]["cost_usd"]["baseline"]["n"], 6)  # warmup row not counted
        self.assertEqual(report["metrics"]["cost_usd"]["treatment"]["n"], 6)

class TestSignflipPvalue(unittest.TestCase):
    """Exact sign-flip correctness on a hand-computable case."""

    def test_all_negative_six_gives_exact_2_over_64(self):
        diffs = [-1.0] * 6
        self.assertAlmostEqual(ev.signflip_pvalue(diffs, seed=0), 2 / 64)

    def test_mixed_signs_gives_larger_p_than_uniform(self):
        diffs = [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
        p = ev.signflip_pvalue(diffs, seed=0)
        self.assertGreater(p, 2 / 64)

    def test_monte_carlo_branch_above_n16_gives_small_p_for_uniform_sign(self):
        diffs = [-1.0] * 17  # n=17 forces the Monte Carlo branch (n>16); still all one sign
        p = ev.signflip_pvalue(diffs, seed=0, mc_perms=5000)
        self.assertLess(p, 0.01)

    def test_monte_carlo_is_deterministic_for_a_fixed_seed(self):
        diffs = [-2.0, -1.0, -1.5, -0.5, 1.0, -0.8, 0.9, -1.1, -0.3, 0.4, -0.6, -0.2, 0.7, -0.9, -1.3, 0.2, -0.4]
        p1 = ev.signflip_pvalue(diffs, seed=42, mc_perms=2000)
        p2 = ev.signflip_pvalue(diffs, seed=42, mc_perms=2000)
        self.assertEqual(p1, p2)

class TestPairedFloor(HarnessTestCase):
    def test_five_pairs_is_inconclusive_even_with_huge_effect(self):
        base = [make_row(f"b{i}", cost_usd=10.0, pair_id=f"p{i}") for i in range(5)]
        treat = [make_row(f"t{i}", cost_usd=0.01, pair_id=f"p{i}") for i in range(5)]
        result = ev.compare_metric("cost_usd", base, treat, 5, 0, 1000, 0.20, True)
        self.assertEqual(result["result"], "INCONCLUSIVE")
        self.assertIsNone(result["p_value"])

class TestRound3Calibration(unittest.TestCase):
    """A/A false-PASS rate <= 0.05 at the DEFAULT --min-n, in both independent and
    paired mode (not only at n=20), plus a paired power sanity check. Calls compare_metric() /
    overall_verdict() directly, never re-implementing the decision rule."""

    SIMS = 200
    MEAN, CV = 0.613, 0.22  # an early n=3 Sonnet code-review cost measurement (conservative: pilot CV was ~7%)

    def test_aa_false_pass_independent_at_default_min_n(self):
        n = ev.DEFAULT_MIN_N
        passes = 0
        for s in range(self.SIMS):
            rnd = random.Random(70_000 + s)
            base_vals = gauss_sample(rnd, self.MEAN, self.CV, n)
            treat_vals = gauss_sample(rnd, self.MEAN, self.CV, n)
            metric = ev.compare_metric("cost_usd", [{"cost_usd": v} for v in base_vals],
                                        [{"cost_usd": v} for v in treat_vals], n, 0, 1000, 0.20, False)
            if ev.overall_verdict({"cost_usd": metric}, "cost_usd", "decrease", []) == "PASS":
                passes += 1
        self.assertLessEqual(passes / self.SIMS, 0.05)

    def test_aa_false_pass_paired_at_default_min_n(self):
        n = ev.DEFAULT_MIN_N  # == PAIRED_MIN_N == 6: the minimum n where sign-flip can reach p<0.05
        passes = 0
        for s in range(self.SIMS):
            rnd = random.Random(80_000 + s)
            base_vals = gauss_sample(rnd, self.MEAN, self.CV, n)
            treat_vals = gauss_sample(rnd, self.MEAN, self.CV, n)
            pids = [f"p{i}" for i in range(n)]
            rows_b = [{"cost_usd": v, "pair_id": pid} for v, pid in zip(base_vals, pids)]
            rows_t = [{"cost_usd": v, "pair_id": pid} for v, pid in zip(treat_vals, pids)]
            metric = ev.compare_metric("cost_usd", rows_b, rows_t, n, 0, 1000, 0.20, True)
            if ev.overall_verdict({"cost_usd": metric}, "cost_usd", "decrease", []) == "PASS":
                passes += 1
        self.assertLessEqual(passes / self.SIMS, 0.05)

    def test_paired_power_30_percent_decrease_n20_at_least_80_percent(self):
        n = 20
        sims = 50  # fewer sims: n=20 triggers signflip_pvalue's Monte Carlo path
        passes = 0
        for s in range(sims):
            rnd = random.Random(90_000 + s)
            base_vals = gauss_sample(rnd, self.MEAN, self.CV, n)
            treat_vals = gauss_sample(rnd, self.MEAN * 0.70, self.CV, n)
            pids = [f"p{i}" for i in range(n)]
            rows_b = [{"cost_usd": v, "pair_id": pid} for v, pid in zip(base_vals, pids)]
            rows_t = [{"cost_usd": v, "pair_id": pid} for v, pid in zip(treat_vals, pids)]
            metric = ev.compare_metric("cost_usd", rows_b, rows_t, n, 0, 1000, 0.20, True, mc_perms=2000)
            if ev.overall_verdict({"cost_usd": metric}, "cost_usd", "decrease", []) == "PASS":
                passes += 1
        self.assertGreaterEqual(passes / sims, 0.80)

class TestReadmeContent(unittest.TestCase):
    def test_required_strings_present(self):
        text = (pathlib.Path(__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
        for needle in ("vercel-labs/agent-eval", "davidcjw/agentmeter", "yussypu/deja",
                       "Related work", "Limitations", "Python 3.9+", "standard library"):
            self.assertIn(needle, text)

class TestCalibration(unittest.TestCase):
    """Mandatory calibration tests: 200 sims x 1000 bootstrap resamples (kept small for
    runtime while still resolving the target rates). Calls compare_metric() directly."""

    SIMS = 200
    BOOTSTRAP = 1000
    N = 20
    # Calibrated to an early n=3 Sonnet code-review cost measurement: mean 0.613, CV ~0.22 (conservative).
    MEAN = 0.613
    CV = 0.22

    def _sample(self, rnd, mean):
        sd = mean * self.CV
        return [max(0.01, rnd.gauss(mean, sd)) for _ in range(self.N)]

    def test_a_a_false_pass_rate_at_most_5_percent(self):
        passes = 0
        for s in range(self.SIMS):
            rnd = random.Random(10_000 + s)
            base_vals = self._sample(rnd, self.MEAN)
            treat_vals = self._sample(rnd, self.MEAN)
            rows_b = [{"cost_usd": v} for v in base_vals]
            rows_t = [{"cost_usd": v} for v in treat_vals]
            metric = ev.compare_metric("cost_usd", rows_b, rows_t, 5, 0, self.BOOTSTRAP, 0.20, False)
            if ev.overall_verdict({"cost_usd": metric}, "cost_usd", "decrease", []) == "PASS":
                passes += 1
        self.assertLessEqual(passes / self.SIMS, 0.05)

    def test_power_true_30_percent_decrease_detected_at_least_80_percent(self):
        passes = 0
        for s in range(self.SIMS):
            rnd = random.Random(20_000 + s)
            base_vals = self._sample(rnd, self.MEAN)
            treat_vals = self._sample(rnd, self.MEAN * 0.70)
            rows_b = [{"cost_usd": v} for v in base_vals]
            rows_t = [{"cost_usd": v} for v in treat_vals]
            metric = ev.compare_metric("cost_usd", rows_b, rows_t, 5, 0, self.BOOTSTRAP, 0.20, False)
            if ev.overall_verdict({"cost_usd": metric}, "cost_usd", "decrease", []) == "PASS":
                passes += 1
        rate = passes / self.SIMS
        self.assertGreaterEqual(rate, 0.80)
        self.assertLessEqual(rate, 1.0)

    def test_guardrail_violation_never_passes(self):
        for s in range(self.SIMS):
            rnd = random.Random(30_000 + s)
            base_cost = self._sample(rnd, self.MEAN)
            treat_cost = self._sample(rnd, self.MEAN * 0.5)  # primary clearly improves
            base_tools = [3.0 + rnd.gauss(0, 0.01) for _ in range(self.N)]
            treat_tools = [12.0 + rnd.gauss(0, 0.01) for _ in range(self.N)]  # guardrail clearly worsens
            cost_metric = ev.compare_metric("cost_usd", [{"cost_usd": v} for v in base_cost],
                                             [{"cost_usd": v} for v in treat_cost], 5, 0, self.BOOTSTRAP, 0.20, False)
            tools_metric = ev.compare_metric("tool_calls_total", [{"tool_calls_total": v} for v in base_tools],
                                              [{"tool_calls_total": v} for v in treat_tools], 5, 0, self.BOOTSTRAP, 0.20, False)
            verdict = ev.overall_verdict({"cost_usd": cost_metric, "tool_calls_total": tools_metric},
                                         "cost_usd", "decrease", [("tool_calls_total", "increase")])
            self.assertNotEqual(verdict, "PASS")

class TestReleasePolicy(HarnessTestCase):
    """Frozen release contract: primary cost_usd:decrease; guardrails permission_denials and tool_errors."""

    def _cohorts(self, treat_tool_errors=0):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=1.0 + 0.001 * i) for i in range(8)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=0.5 + 0.001 * i,
                                                                  tool_errors=treat_tool_errors + i) for i in range(8)])
        return base, treat

    def _report(self, base, treat, *extra):
        path = self.tmp / "policy.json"
        rc = ev.main(["compare", "--baseline", str(base), "--treatment", str(treat), "--report", str(path), *extra])
        return rc, json.loads(path.read_text(encoding="utf-8"))

    def test_defaults_are_cost_primary_and_two_guardrails(self):
        rc, report = self._report(*self._cohorts())
        self.assertEqual(report["primary"], "cost_usd:decrease")
        self.assertEqual(report["guardrails"], ["permission_denials:increase", "tool_errors:increase"])
        self.assertEqual(rc, ev.EXIT_FAIL)  # tool_errors 0..7 vs 0: a default guardrail fires

    def test_no_tool_errors_passes_with_defaults(self):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=1.0 + 0.001 * i) for i in range(8)])
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=0.5 + 0.001 * i) for i in range(8)])
        rc, report = self._report(base, treat)
        self.assertEqual(rc, ev.EXIT_OK)

    def test_guardrail_none_disables_the_defaults(self):
        rc, report = self._report(*self._cohorts(), "--guardrail", "none")
        self.assertEqual(report["guardrails"], [])
        self.assertEqual(rc, ev.EXIT_OK)

    def test_run_once_counts_failed_tool_results(self):
        stream = "\n".join(json.dumps(o) for o in (
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"},
                                                          {"type": "tool_use", "name": "Bash"}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True},
                                                     {"type": "tool_result", "is_error": False}]}},
            {"type": "result", "num_turns": 2, "duration_ms": 10, "total_cost_usd": 0.01,
             "usage": {"output_tokens": 5}, "permission_denials": []}))

        class Proc:
            stdout, stderr, returncode = stream, "", 0

        orig = ev.subprocess.run
        ev.subprocess.run = lambda *a, **k: Proc()
        try:
            row = ev.run_once("p", "m", pathlib.Path("."), None, 10)
        finally:
            ev.subprocess.run = orig
        self.assertEqual((row["tool_calls_total"], row["tool_errors"]), (2, 1))
        self.assertTrue(ev.is_completed_row(row))

class TestPairedDecision(unittest.TestCase):
    """The paired decision must come from the exact sign-flip test, not from the bootstrap CI."""

    def test_sign_flip_decides_where_bootstrap_would_not(self):
        # Five of six diffs are -1: the bootstrap median CI sits at -1 (excludes 0), but the exact
        # sign-flip p is 4/64 = 0.0625, so the paired result must be INCONCLUSIVE.
        base = [{"cost_usd": 2.0, "pair_id": f"p{i}"} for i in range(6)]
        treat = [{"cost_usd": 2.0 + d, "pair_id": f"p{i}"} for i, d in enumerate([-1, -1, -1, -1, -1, 0.5])]
        res = ev.compare_metric("cost_usd", base, treat, 6, 0, 1000, 0.20, True)
        self.assertLess(res["ci_high"], 0)  # the bootstrap alone would have called this DECREASED
        self.assertAlmostEqual(res["p_value"], 4 / 64)
        self.assertEqual(res["result"], "INCONCLUSIVE")

    def test_exact_p_matches_brute_force_on_asymmetric_diffs(self):
        import itertools
        for diffs in ([-3.0, -1.0, -1.0, -0.5, 0.2, -2.0], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.5]):
            obs = abs(sum(diffs) / len(diffs))
            hits = sum(abs(sum(s * d for s, d in zip(signs, diffs)) / len(diffs)) >= obs - 1e-9
                       for signs in itertools.product((1, -1), repeat=len(diffs)))
            self.assertAlmostEqual(ev.signflip_pvalue(diffs, 0), hits / 2 ** len(diffs))

class TestAuditFixes(HarnessTestCase):
    """Regressions from an independent audit: seed use, identity-field presence, non-finite --effect."""

    def _cohorts(self, **kw):
        base = self.write_cohort("base", "baseline", [make_row(f"b{i}", cost_usd=0.5 + 0.03 * i) for i in range(8)], **kw)
        treat = self.write_cohort("treat", "treatment", [make_row(f"t{i}", cost_usd=0.45 + 0.03 * i) for i in range(8)], **kw)
        return base, treat

    def _cmp(self, base, treat, *extra):
        return ev.main(["compare", "--baseline", str(base), "--treatment", str(treat),
                        "--primary", "cost_usd:decrease", *extra])

    def test_seed_changes_bootstrap_resamples(self):
        a = [0.5 + 0.03 * i for i in range(8)]
        b = [0.45 + 0.03 * i for i in range(8)]
        self.assertEqual(ev.bootstrap_delta_ci(a, b, 1, 1000), ev.bootstrap_delta_ci(a, b, 1, 1000))
        self.assertNotEqual(ev.bootstrap_delta_ci(a, b, 1, 1000), ev.bootstrap_delta_ci(a, b, 2, 1000))

    def test_identity_field_missing_from_both_manifests_is_mismatch(self):
        base, treat = self._cohorts()
        for d in (base, treat):
            manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            del manifest["task_sha256"]
            (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self._cmp(base, treat), ev.EXIT_COHORT_MISMATCH)

    def test_unavailable_claude_version_is_mismatch(self):
        base, treat = self._cohorts(claude_version="unavailable:no binary")
        self.assertEqual(self._cmp(base, treat), ev.EXIT_COHORT_MISMATCH)

    def test_non_finite_or_negative_effect_is_argument_error(self):
        base, treat = self._cohorts()
        for bad in ("nan", "inf", "-1"):
            self.assertEqual(self._cmp(base, treat, "--effect", bad), ev.EXIT_ARG_ERROR)

if __name__ == "__main__":
    unittest.main()
