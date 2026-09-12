# Changelog

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
