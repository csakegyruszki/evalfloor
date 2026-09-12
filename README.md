# evalfloor

A small experimental harness for measuring whether a Claude Code configuration changes
operational behaviour enough to distinguish it from normal run-to-run noise.

Claude Code is stochastic: the same task can cost noticeably less in one run and more in the next
with nothing changed. A single before/after run cannot separate a real effect from that noise.
evalfloor runs each configuration several times into an isolated, fingerprinted cohort, compares
the two, and reports a difference only when it is larger than the measured noise. It measures how
the agent works (cost, runtime, failed tool calls, permission denials, tool calls, turns), not
whether its answer is good. Standard library only, Python 3.9+, no dependencies; it drives
`claude -p --output-format stream-json`.

## What it is for

- **Prompt A/B.** Does a system prompt, persona or agent description change what the same task
  costs? Run both variants on the same task and compare. The quick start below is a persona versus
  no-persona example.
- **Skill and CLAUDE.md regression checks.** Run a fixed task before and after editing a skill,
  CLAUDE.md, tool permissions or a workflow. A cost increase, or new failed tool calls or
  permission denials, shows up as FAIL instead of going unnoticed.
- **Cost and reliability tuning.** Choose between configurations (model, tool access, workflow)
  on measured cost, with failed tool calls and permission denials as guardrails, so a cheaper
  configuration that errors more does not pass.

A compare result, summarised (illustrative only, not a measured run):

```
cost_usd            DECREASED
duration_ms         INCONCLUSIVE
tool_errors         no increase (guardrail)
permission_denials  no increase (guardrail)
verdict             PASS
```

## What it is not

It does not judge answer or code quality, and it does not rank agents in general. Fewer tool calls
or tokens can mean less thorough work, so those metrics are reported but never decide. evalfloor
answers a narrow question: did this change alter how the agent operates and what it costs, by more
than its own noise? Pair it with a separate quality check when quality matters.

## Quick start

The `examples/` directory holds a small code-review task, a Python file with planted defects, and
two system prompts that differ only in their persona line. Run the baseline variant into an
isolated cohort:

```
python run_eval.py run --task examples/tasks/code-review.md --variant persona \
    --cwd examples/fixtures/review --model claude-sonnet-5 --repeat 6 --pair-id block1 \
    --system-prompt-file examples/variants/code-reviewer-A-persona.txt
```

Run the treatment (a distinct `--variant` value is required for `compare`):

```
python run_eval.py run --task examples/tasks/code-review.md --variant no-persona \
    --cwd examples/fixtures/review --model claude-sonnet-5 --repeat 6 --pair-id block1 \
    --system-prompt-file examples/variants/code-reviewer-B-nopersona.txt
```

Each `run` adds one warmup run first (`--warmup 1`), so one arm costs seven runs. Compare the two
cohorts (paths are the printed `run dir:` lines from `run`):

```
python run_eval.py compare --baseline results/runs/<baseline-id> --treatment results/runs/<treatment-id>
```

## Release policy (the defaults)

- **Primary: `cost_usd:decrease`.** Cost is a direct, additive operational outcome, and the most
  stable metric measured (about 7% CV in a Sonnet pilot). The planning heuristic suggests the
  paired floor of 6 may be sufficient for roughly 20% cost differences under the measured pilot
  variance; this is not a tested power guarantee.
- **Guardrails: `permission_denials:increase`, `tool_errors:increase`.** A treatment that saves
  money by hitting more denied or failed tool calls fails. These two are **zero-tolerance**: the
  guardrail fires on any adverse mean move, not only a statistically significant one (see below).
  Add `--guardrail duration_ms:increase` if runtime matters (a non-default guardrail metric keeps
  the ordinary statistical rule).
- **Diagnostic only:** `tool_calls_total`, `num_turns`, `subagents`, `output_tokens`. They are
  reported, never decide. Fewer tool calls is not a quality signal: an agent can be better because
  it reads or verifies more. Making tool calls the primary metric is a separate study, needs
  roughly 10-20 runs per arm (22-27% CV measured), and says nothing about quality.

Override with `--primary METRIC:decrease|increase` and repeatable `--guardrail
METRIC:increase|decrease` (an ADVERSE direction; giving any `--guardrail` replaces the defaults,
`--guardrail none` disables them).

`run` also accepts `--agents`, `--system-prompt-file`, `--cwd`, `--timeout`, `--run-id`,
`--pair-id`, and `--warmup`. Six measured runs per arm is the minimum `compare` needs for a paired
verdict; with fewer, every metric is `INCONCLUSIVE` by design.

### Zero-tolerance guardrails

Default zero-tolerance guardrails are deterministic policy checks. For completed cohorts with equal
measured run counts, any increase in the aggregate count of `permission_denials` or `tool_errors`
causes FAIL. They do not use the statistical significance test used for the primary metric. (With
unequal counts, possible only with `--allow-failed-runs` or unequal `--repeat`, the per-run mean is
compared instead.) This is a release policy, not an inference: it closes the gap where a real
increase at small n fails to reach significance. A guardrail on any other metric keeps the
statistical rule (its own `DECREASED`/`INCREASED` result).

`guardrails_detail` in the report lists, per guardrail: `metric`, `direction`, `rule`
(`zero_tolerance` or `statistical`), `fired`, `baseline_total`, `treatment_total`, `delta_total`,
`baseline_n`, `treatment_n`, and `basis` (`aggregate_count` or `per_run_mean`), so a FAIL can be
audited from the report alone.

## Failed runs and warmup failures

By default, if either cohort has one or more **measured** rows (warmup rows excluded) that did not
complete, `compare` refuses a verdict: exit code 4, with `validity: "INVALID_FAILED_RUNS"`,
`failed_baseline` and `failed_treatment` in the report. Silently dropping failed runs is a
selection bias - the cohort that kept fewer or easier runs looks artificially better.
`--allow-failed-runs` proceeds anyway with `validity: "DEGRADED_FAILED_RUNS"` and the same counts,
and caps the verdict at `INCONCLUSIVE`: exit 0 is impossible in this mode, whatever the remaining
completed runs show (`FAIL`, exit 6, is still possible). A clean comparison reports
`validity: "VALID"`.

A failed **warmup** now stops `run` itself immediately, before any measured run executes: an error
is printed and exit code 3 returned, leaving the manifest and the failed warmup row on disk.
Previously a failed warmup was silently ignored and all measured runs still ran with exit code 0.

## Warmup runs

`--warmup N` (default 1, must be >= 0) executes N runs before the measured ones: the first run in
a cohort pays a cold prompt-cache cost (a Haiku smoke run measured 67,026 cache-creation tokens on
the first run vs about 33,500 on every later one, about $0.152 vs $0.088), and whichever cohort
runs first absorbs that cost, biasing a comparison unless both pay it before being measured. A
failed warmup halts the run entirely (see above).

Warmup rows are written with `"warmup": true` and no `pair_id`; `N` is recorded in
`cli_options.warmup`. They are excluded from `run`'s summary, exit code, and every `compare`
calculation. **Recommendation:** even with warmups, interleave baseline and treatment runs in time
- a warmup removes the cache-cost bias but not other time-of-day or infrastructure drift.

## Cohort directory layout

Each `run` invocation writes to `results/runs/<run_id>/`:

- `manifest.json` - created with exclusive file creation before execution starts, so a duplicate
  `--run-id` fails immediately and never overwrites existing data.
- `metrics.jsonl` - one JSON row per warmup and measured run, appended as each completes.
- `warmup<N>.txt` / `run<N>.txt` - the raw result text for run N, if any (kept out of the metrics
  file, kept for manual inspection; the harness itself does not score output).

`run_id` defaults to a generated UUID4, overridable with `--run-id`, which must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` and resolve to a direct child of `results/runs/` - rejecting
path traversal (e.g. `../escaped`, `a/b`, `..`) before anything is created.

## Manifest fields

`schema_version`, `run_id`, `created_utc`, `model`, `variant`, `task_path` (resolved),
`task_sha256`, `agents_sha256` (or null), `system_prompt_sha256` (or null), `cwd` (resolved),
`cwd_tree_sha256` and `cwd_tree_policy` (see below), `claude_version` (`claude --version` output, or
`"unavailable:<reason>"` - never crashes the run), `runner_sha256` (SHA-256 of `run_eval.py`, so a
compare across a changed runner is rejected), `pair_id` (or null), and `cli_options`
(behaviour-affecting CLI values, including `warmup`).

`cwd_tree_sha256` makes the working tree's *content* part of cohort identity, not the `cwd` path
string: two checkouts of the same tree in different places compare as identical, the same path
with different contents does not. The contract:

- computed once, before the first (warmup) run;
- entries sorted by relative POSIX path (UTF-8 byte order); a regular file contributes
  `path \0 sha256(bytes) \n`; mtime, size and permissions never enter the hash;
- symlinks are never followed: a link contributes `path \0 symlink:<target> \n`, so a link cannot
  pull content from outside the tree;
- directories named `.git`, `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`,
  `.venv`, `venv`, `env`, `.tox`, `node_modules` and `results` are skipped at any depth; this list is
  versioned and written to the manifest as `cwd_tree_policy`, which is itself an identity field;
- special files (sockets, FIFOs, devices), unreadable files, or more than 5000 entries / 100 MB
  fail closed as `"unavailable:<reason>"`, rejected by `compare` unless `--allow-unverified-cwd`
  (then the two `cwd` paths must be equal).

Every metrics row also carries `run_id`, `manifest_file`, and `warmup` (true/false). Measured rows
also get a per-row `pair_id` of the form `<pair-id>:<run index>` when `--pair-id` was given;
warmup rows never do.

`--out` is only accepted when it equals the run's own generated `metrics.jsonl` path; any other
value is a validation error, to reject accidentally reintroducing the old shared flat-file
behaviour.

## Metrics

Tracked for both `run` summaries and `compare`: `tool_calls_total`, `subagents`, `num_turns`,
`output_tokens`, `cost_usd`, `duration_ms`, `permission_denials`, `tool_errors` - exactly what
`run_once()` extracts from the stream's final `result` record, its `tool_use` blocks, and its
`tool_result` blocks marked `is_error` (`tool_errors` counts failed tool calls, permission denials
included). Input and cache token counts are kept on every completed row (timed-out or result-less
rows have none) but are not compared.

A row is **completed** only if it has no truthy `error`, `is_error` is not true, `rc` is 0 (or
absent), and every metric above is a number; otherwise it is **failed**. Summaries
(`n_total`/`n_completed`/`n_failed` and per-metric `n`/`mean`/`median`/`stdev`/`cv`/`min`/`max`)
use completed rows only. `stdev` is sample stdev (ddof=1, `null` when n<2); `cv` is `stdev/mean`,
`null` (not `0`) when the mean is 0 or n<2.

## Decision rule

Per metric, `compare` reports a **direction-neutral** result: `DECREASED`, `INCREASED`, or
`INCONCLUSIVE`. A metric is always `INCONCLUSIVE` when either arm has fewer than `--min-n`
completed rows for it (default 6, at least 5). Above that floor, the two modes decide differently.

**Independent mode** uses the 95% bootstrap CI of the (treatment - baseline) **median** delta:
`INCONCLUSIVE` if the CI includes 0, `DECREASED`/`INCREASED` if wholly below/above - unchanged.

**Paired mode** decides differently. A Haiku A/A smoke run (two identical cohorts, same task,
`--pair-id`) found the percentile bootstrap CI anti-conservative for a false-positive ("A/A")
comparison, worse in paired mode. Measured false-PASS rate over 400 simulations on a cost
distribution calibrated to an early small Sonnet measurement (mean 0.613, CV 0.22), 1,000
bootstrap resamples:

| n per arm | bootstrap independent | bootstrap paired | exact sign-flip (paired) |
|---:|---:|---:|---:|
| 3  | 0.120 | 0.282 | 0.000 |
| 5  | 0.033 | 0.163 | 0.000 |
| 10 | 0.037 | 0.113 | 0.058 |
| 20 | 0.043 | 0.068 | 0.050 |

(Current code: the paired bootstrap resamples the mean paired difference. Reproduce with
`python docs/simulate_aa.py`; `docs/MEASUREMENTS.md` also gives the first, median-based figures.)

So **paired mode does not use the bootstrap CI to decide the per-metric result.** It uses
`signflip_pvalue()` instead: an exact two-sided sign-flip permutation test of the **mean** paired
difference (full enumeration of all 2**n sign patterns for n <= 16; seeded Monte Carlo with 20,000
permutations above that, counting the observed pattern so p is never 0). `DECREASED`/`INCREASED`
only when p < 0.05, direction from the sign of the mean difference; otherwise `INCONCLUSIVE`. The
bootstrap CI is still reported in paired mode but is descriptive only there.

The exact test cannot reach p < 0.05 below n = 6 (min achievable two-sided p = 2/2**n: 0.03125 at
n=6, 0.0625 at n=5 - never below 0.05). A paired metric with fewer than 6 pairs is therefore always
`INCONCLUSIVE`, regardless of `--min-n` or effect size. This is also why `--min-n` defaults to 6
with a hard floor of 5 (argument error below): 5 is where the independent bootstrap's own A/A rate
(0.033) first becomes acceptable (3 measured 0.120).

Every metric's result also carries an `estimand`: `"mean_paired_difference"` in paired mode,
`"median_difference"` independent. Paired `delta` is the mean of the paired diffs, and its
bootstrap CI now resamples that same mean (not the median, as before) - the CI and the point
estimate must describe the same quantity. The paired PASS/FAIL/INCONCLUSIVE decision still comes
from the exact sign-flip test above, never from the CI.

`--primary METRIC:decrease|increase` declares the one metric and direction that determines the
verdict. `--guardrail METRIC:increase|decrease` (repeatable) declares an adverse direction to fail
on. Non-primary, non-guardrail metrics are reported for information only and never affect the
verdict, so an always-zero or noisy metric can never silently block or force one.

Overall verdict (a metric's move is CI-supported in independent mode, p-value-supported at p <
0.05 in paired mode; see above):

- **PASS** - the primary metric's result matches the declared direction AND no guardrail fired
  (see Zero-tolerance guardrails; a non-zero-tolerance guardrail firing means it moved adversely).
- **FAIL** - the primary metric moved opposite to the declared direction, OR any guardrail fired.
- **INCONCLUSIVE** - everything else (including the primary being `INCONCLUSIVE`, too few
  completed/paired rows, or `--allow-failed-runs` capping a would-be PASS).

"Better" and "worse" are not built into any metric - they are a policy the user declares through
`--primary`/`--guardrail`. Fewer tool calls in a code review, for example, could mean less
thorough work, not a win.

## Bootstrap and reproducibility

The CI is a 95% percentile bootstrap, using `random.Random(seed)` (default seed 0) and a
configurable resample count (`--bootstrap`, default 10000, must be >= 1000). If every completed
row in both arms carries a non-null `pair_id`, unique within its arm and matching 1:1 across arms
(two cohorts run with the same `--pair-id` and `--repeat`), `compare` uses **paired** resampling of
the paired differences (of their mean - see `estimand` above); otherwise **independent** resampling
with replacement within each arm (of the median). Running `compare` twice with the same seed and
inputs produces byte-identical JSON.

`compare` also reports, per metric, the **baseline arm's CV** (`planning_cv`; pooling both arms
would count a real difference as noise) and a normal-approximation **planning** estimate of the
n-per-arm needed to detect `--effect` (default 0.20, must be > 0) at alpha=0.05/power=0.80:
`n_per_arm = ceil(2 * (z_(1-alpha/2)+z_power)^2 * CV^2 / effect^2)` - a planning heuristic, not a
power guarantee or a verdict input. `p_value` is the sign-flip result in paired mode, `null`
independent (CI alone decides).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | `run`: every row completed. `compare`: valid comparison, verdict PASS. |
| 1 | `compare`: valid comparison, verdict INCONCLUSIVE (also: `--allow-failed-runs` capped a would-be PASS to INCONCLUSIVE). |
| 2 | Argument/input/validation error (bad CLI args, bad task path, malformed `--primary`/`--guardrail` spec, `--bootstrap` < 1000, `--out` not the run's own path, duplicate `--run-id`, or `--run-id` failing its pattern/traversal check). |
| 3 | `run`: one or more measured rows failed, or a warmup run failed (which now stops before any measured run executes). |
| 4 | `compare`: cohort mismatch or invalid/incomplete manifest; also an unverified/mismatched `cwd_tree_sha256` (unless `--allow-unverified-cwd` with equal `cwd` paths), a failed measured row without `--allow-failed-runs`, or invalid pairing (`pairing_invalid`: a manifest `pair_id` on one side without a matching, fully-pairable counterpart on the other). |
| 5 | Internal/unexpected error. |
| 6 | `compare`: valid comparison, verdict FAIL (still possible even with `--allow-failed-runs`). |

## Identity validation before any statistics

Before computing anything, `compare` requires the two manifests to agree on `schema_version`,
`task_sha256`, `model`, `claude_version`, and `runner_sha256`, and requires `variant` to differ. It
also rejects duplicate `session_id`s within one arm and any row whose `run_id` does not match its
own manifest. Any mismatch is reported by field name, exit 4.

`cwd_tree_sha256` (see Manifest fields) is checked the same way, with one refinement: an
`"unavailable:..."` or missing value on either side is rejected unless `--allow-unverified-cwd` is
given, in which case the two manifests' `cwd` paths must be equal instead (an explicit, weaker
fallback, never a silent pass). **Pairing validity** is checked next: if either manifest has a
non-null `pair_id`, both must have the *same* `pair_id` and the measured completed rows must
satisfy full 1:1 pairing, or `compare` refuses with `"error": "pairing_invalid"`, exit 4, instead
of silently falling back to independent-mode analysis (cohorts with no `pair_id` on either side are
unaffected). **Failed measured runs** are checked last, before any metric is computed - see
"Failed runs and warmup failures" above.

## Limitations

- Noise estimates are small-sample. A Sonnet paired A/A pilot (6+6 measured runs, 1 warmup each,
  code-review task) measured CVs of about 7% for `cost_usd`, 10-13% for duration/output tokens,
  16-20% for turns, 22-27% for `tool_calls_total`. An earlier n=3 legacy cohort showed 22% for
  cost, likely inflated by an unrecognised cold-cache run. Treat these as planning figures, not
  validated thresholds; the calibration tests use the older, more conservative 22%. Full numbers
  in `docs/MEASUREMENTS.md`; one machine, one task, not a benchmark.
- No verified B arm in this repository; every real comparison still needs its own baseline and
  treatment runs.
- No quality judging (behavioural counts, not correctness) and no transcript analytics beyond the
  `result` record and `tool_use` blocks.
- No ICC or variance-component decomposition. The only inferential methods are the bootstrap CI
  (independent mode) and the sign-flip permutation test of the mean (paired mode); neither is a
  formal power guarantee.
- `compare` requires an exact match on Claude CLI version, runner source, and working-tree content
  between arms; any of those changing means a fresh baseline.
- Metric direction (decrease/increase is "good") is a user-declared policy via
  `--primary`/`--guardrail`, not an inherent property of the metric.

## Development

```
python -m unittest test_run_eval.py test_contract_blackbox.py
python docs/simulate_aa.py
```

The tests never call the Claude CLI: `run_once` is replaced by a stub in every test that runs
`run`, and `compare` tests use synthetic cohorts written to a temporary directory.
`test_contract_blackbox.py` was written from the written contract alone, independently of the
implementation. `docs/simulate_aa.py` recomputes the A/A false-positive table in
`docs/MEASUREMENTS.md` with a fixed seed and no model calls.

## Related work / how this differs

- **`vercel-labs/agent-eval`** (MIT) - A/B repeat-run evaluation for Claude Code with tool-call
  counting.
- **`davidcjw/agentmeter`** (MIT) - transcript analytics for agent runs, including
  permission-denial tracking.
- **`yussypu/deja`** (MIT) - variance-component decomposition, ICC, flip rate, and paired
  permutation tests for agent evals.

This tool makes no novelty claim over any of them. It differs mainly in being deliberately small:
Python standard library only, no quality judging, no ICC or variance decomposition - just isolated
run cohorts, a bootstrap or sign-flip comparison, and a primary/guardrail verdict rule.
