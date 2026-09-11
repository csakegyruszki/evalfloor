# Changelog

## 0.1.0 - unreleased

First public version.

- `run`: executes a Claude Code task N times (`claude -p --output-format stream-json`) into an
  isolated cohort directory with a manifest (task, system prompt and runner hashes, model,
  Claude Code version) and one metrics row per run. Optional warmup runs pay the cold
  prompt-cache cost before measurement and are excluded from every calculation.
- `compare`: checks that two cohorts are comparable, then reports per-metric
  `DECREASED` / `INCREASED` / `INCONCLUSIVE` and an overall `PASS` / `FAIL` / `INCONCLUSIVE`.
  Default policy: primary `cost_usd:decrease`; guardrails `permission_denials:increase`
  and `tool_errors:increase`; other metrics are diagnostic only.
- Independent cohorts use a percentile bootstrap CI of the median difference; paired
  cohorts use an exact sign-flip permutation test of the mean difference.
- Standard library only, Python 3.9+.
