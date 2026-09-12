# Changelog

## 0.1.1 - 2026-09-12

No change to `run_eval.py` or the tests.

- CI: GitHub Actions runs the unit and contract tests on Linux (Python 3.9-3.14), Windows
  (3.9, 3.14) and macOS (3.14), plus a CLI smoke test. No test makes a model call.
- A/A false-positive table recomputed with 4,000 simulations instead of 400 (Monte Carlo
  standard error about 0.0035 at a 5% rate, down from about 0.011). The conclusion is unchanged:
  the paired bootstrap exceeds 0.05 at every n, so the exact sign-flip test decides in paired
  mode. The sign-flip 0.058 at n = 20 was checked with two further seeds (0.0465, 0.0493).
  `docs/simulate_aa.py` reproduces the table.
- README: `PASS` means cheaper and not operationally worse on the declared guardrails; it does
  not mean a better agent.

## 0.1.0 - 2026-09-12

First public version.

- `run`: executes a Claude Code task N times (`claude -p --output-format stream-json`) into an
  isolated cohort directory with a manifest (task, system prompt, runner and working-tree hashes,
  model, Claude Code version) and one metrics row per run. Warmup runs pay the cold prompt-cache
  cost before measurement; a failed warmup stops the run before any measured run. Run ids are
  restricted to a safe basename under `results/runs/`.
- `compare`: checks that two cohorts are comparable (including the content hash of the working
  tree, `cwd_tree_sha256`, under a versioned exclusion policy), then reports per-metric
  `DECREASED` / `INCREASED` / `INCONCLUSIVE` and an overall `PASS` / `FAIL` / `INCONCLUSIVE`.
  Default policy: primary `cost_usd:decrease`; zero-tolerance guardrails on
  `permission_denials` and `tool_errors` (any increase fails, a policy check, not a significance
  test); other metrics are diagnostic only.
- A failed measured run invalidates the comparison by default; `--allow-failed-runs` reports
  `validity: DEGRADED_FAILED_RUNS` and can never yield `PASS`. A declared pairing that does not
  hold is an error, not a silent fallback to independent mode.
- Independent cohorts use a percentile bootstrap CI of the median difference; paired cohorts use
  an exact sign-flip permutation test of the mean difference, with a bootstrap CI of the same mean.
- Standard library only, Python 3.9+. Tests: unit and regression tests plus independent
  black-box contract tests; the decision rule is checked by A/A and power calibration tests.
