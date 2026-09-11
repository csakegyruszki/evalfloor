# Claude Code behaviour-eval harness

A small, stdlib-only (Python 3.9+) A/B harness for `claude -p --output-format
stream-json` behaviour: tool calls, subagents, tokens, duration, cost,
permission denials. No output-quality judging, no transcript parsing.

## Quick start

The `examples/` directory holds a small code-review task, a Python file with
planted defects, and two system prompts that differ only in their persona
line. Run the baseline variant into an isolated cohort:

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

Each `run` adds one warmup run first (`--warmup 1`), so one arm costs seven
runs.

Compare the two cohorts (paths are the printed `run dir:` lines from `run`):

```
python run_eval.py compare --baseline results/runs/<baseline-id> \
    --treatment results/runs/<treatment-id>
```

## Release policy (the defaults)

- **Primary: `cost_usd:decrease`.** Cost is a direct, additive operational
  outcome, and the most stable metric measured (about 7% CV in a Sonnet
  pilot), so a 6+6 paired design can resolve roughly 20% differences.
- **Guardrails: `permission_denials:increase`, `tool_errors:increase`.** A
  treatment that saves money by hitting more denied or failed tool calls
  fails. Add `--guardrail duration_ms:increase` if runtime matters.
- **Diagnostic only:** `tool_calls_total`, `num_turns`, `subagents`,
  `output_tokens`. They are reported, never decide. Fewer tool calls is not
  a quality signal: an agent can be better because it reads or verifies more.
  Making tool calls the primary metric is a separate study, needs roughly
  10-20 runs per arm (22-27% CV measured), and says nothing about quality.

Override with `--primary METRIC:decrease|increase` and repeatable
`--guardrail METRIC:increase|decrease` (an ADVERSE direction; giving any
`--guardrail` replaces the defaults, `--guardrail none` disables them).

`run` also accepts `--agents`, `--system-prompt-file`, `--cwd`, `--timeout`,
`--run-id`, `--pair-id`, and `--warmup`. Six measured runs per arm is the
minimum `compare` needs for a paired verdict; with fewer, every metric is
`INCONCLUSIVE` by design.

## Warmup runs

`--warmup N` (default 1, must be >= 0) executes N runs before the measured
ones, because the first run in a cohort pays a cold prompt-cache cost: a
Haiku smoke run measured 67,026 cache-creation tokens on the first run of a
cohort against about 33,500 on every later run (about $0.152 vs $0.088).
Whichever cohort runs first absorbs that cost, biasing a comparison unless
both cohorts pay it before being measured.

Warmup rows are written to `metrics.jsonl` with `"warmup": true` and no
`pair_id`, and `N` is recorded in `cli_options.warmup`. They are excluded
from `run`'s summary, from `run`'s exit code, and from every `compare`
calculation. **Recommendation:** even with warmups, interleave baseline and
treatment runs in time for anything that matters - a warmup removes the
cache-cost bias but not other time-of-day or infrastructure drift.

## Cohort directory layout

Each `run` invocation writes to `results/runs/<run_id>/`:

- `manifest.json` - created with exclusive file creation before execution
  starts, so a duplicate `--run-id` fails immediately and never overwrites
  existing data.
- `metrics.jsonl` - one JSON row per warmup and measured run, appended as
  each completes.
- `warmup<N>.txt` / `run<N>.txt` - the raw result text for run N, if any
  (kept out of the metrics file; kept for manual inspection; the harness itself does not score output).

`run_id` defaults to a generated UUID4, overridable with `--run-id`.

## Manifest fields

`schema_version`, `run_id`, `created_utc`, `model`, `variant`, `task_path`
(resolved), `task_sha256`, `agents_sha256` (or null), `system_prompt_sha256`
(or null), `cwd` (resolved), `claude_version` (`claude --version` output, or
`"unavailable:<reason>"` - never crashes the run), `runner_sha256` (SHA-256
of `run_eval.py`, so a compare across a changed runner is rejected),
`pair_id` (or null), and `cli_options` (behaviour-affecting CLI values,
including `warmup`).

Every metrics row also carries `run_id`, `manifest_file`, and `warmup`
(true/false). Measured rows also get a per-row `pair_id` of the form
`<pair-id>:<run index>` when `--pair-id` was given; warmup rows never do.

`--out` is only accepted when it equals the run's own generated
`metrics.jsonl` path; any other value is a validation error, to reject
accidentally reintroducing the old shared flat-file behaviour.

## Metrics

Tracked for both `run` summaries and `compare`: `tool_calls_total`,
`subagents`, `num_turns`, `output_tokens`, `cost_usd`, `duration_ms`,
`permission_denials`, `tool_errors` - exactly what `run_once()` extracts
from the stream's final `result` record, its `tool_use` blocks, and its
`tool_result` blocks marked `is_error` (`tool_errors` counts failed tool
calls, permission denials included). Input and cache token counts are kept
on every row but are not compared.

A row is **completed** only if it has no truthy `error`, `is_error` is not
true, `rc` is 0 (or absent), and every metric above is a number; otherwise
it is **failed**. Summaries (`n_total`/`n_completed`/`n_failed` and
per-metric `n`/`mean`/`median`/`stdev`/`cv`/`min`/`max`) use completed rows
only. `stdev` is sample stdev (ddof=1, `null` when n<2); `cv` is
`stdev/mean`, `null` (not `0`) when the mean is 0 or n<2.

## Decision rule

Per metric, `compare` reports a **direction-neutral** result:
`DECREASED`, `INCREASED`, or `INCONCLUSIVE`. A metric is always
`INCONCLUSIVE` when either arm has fewer than `--min-n` completed rows for
that metric (default 6, must be at least 5). Above that floor, the two
modes decide differently - see the next section.

**Independent mode** uses the 95% bootstrap CI of the (treatment - baseline)
**median** delta: `INCONCLUSIVE` if the CI includes 0, `DECREASED` if
wholly below 0, `INCREASED` if wholly above - unchanged from before.

**Paired mode** decides differently. A Haiku A/A smoke run (two identical
cohorts, same task, `--pair-id`) found the percentile bootstrap CI
anti-conservative for a false-positive ("A/A") comparison, worse in paired
mode than independent. Measured false-PASS rate over 400 simulations on a
cost distribution calibrated to an early small Sonnet measurement (mean 0.613,
CV 0.22), 1,000 bootstrap resamples:

| n per arm | bootstrap independent | bootstrap paired | exact sign-flip (paired) |
|---:|---:|---:|---:|
| 3  | 0.120 | 0.270 | 0.000 |
| 5  | 0.033 | 0.058 | 0.000 |
| 10 | 0.037 | 0.075 | 0.058 |
| 20 | 0.043 | 0.055 | not enumerated |

So **paired mode does not use the bootstrap CI to decide the per-metric
result.** It uses `signflip_pvalue()` instead: an exact two-sided sign-flip
permutation test of the **mean** paired difference (full enumeration of all
2**n sign patterns for n <= 16; seeded Monte Carlo with 20,000 permutations
above that, counting the observed pattern so p is never 0). A metric is
`DECREASED`/`INCREASED` only when p < 0.05, direction from the sign of the
mean difference; otherwise `INCONCLUSIVE`. The bootstrap CI is still
reported in paired mode but is descriptive only there.

The exact test cannot reach p < 0.05 below n = 6 (min achievable two-sided
p = 2/2**n: 0.03125 at n=6, 0.0625 at n=5 - never below 0.05). A paired
metric with fewer than 6 pairs is therefore always `INCONCLUSIVE`,
regardless of `--min-n` or effect size. This is also why `--min-n` defaults
to 6 with a hard floor of 5 (argument error below): 5 is where the
independent bootstrap's own A/A rate (0.033) first becomes acceptable (3
measured 0.120).

`--primary METRIC:decrease|increase` declares the one metric and direction
that determines the verdict. `--guardrail METRIC:increase|decrease`
(repeatable) declares an adverse direction to fail on. Non-primary,
non-guardrail metrics are reported for information only and never affect
the verdict, so an always-zero or noisy metric can never silently block or
force one.

Overall verdict (a metric's move is CI-supported in independent mode,
p-value-supported at p < 0.05 in paired mode; see above):

- **PASS** - the primary metric's result matches the declared direction
  AND no guardrail shows a move in its adverse direction.
- **FAIL** - the primary metric moved opposite to the declared direction,
  OR any guardrail shows an adverse move.
- **INCONCLUSIVE** - everything else (including the primary being
  `INCONCLUSIVE`, or too few completed/paired rows).

"Better" and "worse" are not built into any metric - they are a policy the
user declares through `--primary`/`--guardrail`. Fewer tool calls in a code
review, for example, could mean less thorough work, not a win.

## Bootstrap and reproducibility

The CI is a 95% percentile bootstrap of the median delta, using
`random.Random(seed)` (default seed 0) and a configurable resample count
(`--bootstrap`, default 10000, must be >= 1000). If every completed row in
both arms carries a non-null `pair_id`, unique within its arm and matching
1:1 across arms (two cohorts run with the same `--pair-id` and `--repeat`),
`compare` uses **paired** resampling of the paired differences; otherwise
**independent** resampling with replacement within each arm. Running
`compare` twice with the same seed and inputs produces byte-identical JSON.

`compare` also reports, per metric, the **baseline arm's CV**
(`planning_cv`; pooling both arms would count a real difference as noise)
and a normal-approximation **planning** estimate of the n-per-arm needed to
detect `--effect` (default 0.20, must be > 0) at alpha=0.05/power=0.80:
`n_per_arm = ceil(2 * (z_(1-alpha/2)+z_power)^2 * CV^2 / effect^2)` - a
planning heuristic, not a power guarantee, and not used for the verdict.
Every metric also reports `p_value`: the sign-flip result in paired mode,
`null` in independent mode (there the CI alone decides).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | `run`: every row completed. `compare`: valid comparison, verdict PASS. |
| 1 | `compare`: valid comparison, verdict INCONCLUSIVE. |
| 2 | Argument/input/validation error (bad CLI args, bad task path, malformed `--primary`/`--guardrail` spec, `--bootstrap` < 1000, `--out` not the run's own path, duplicate `--run-id`). |
| 3 | `run`: one or more rows failed. |
| 4 | `compare`: cohort mismatch or invalid/incomplete manifest. |
| 5 | Internal/unexpected error. |
| 6 | `compare`: valid comparison, verdict FAIL. |

## Identity validation before any statistics

Before computing anything, `compare` requires the two manifests to agree on
`schema_version`, `task_sha256`, `model`, `claude_version`, and
`runner_sha256`, and requires `variant` to differ. It also rejects
duplicate `session_id`s within one arm and any row whose `run_id` does not
match its own manifest. Any mismatch is reported by field name, exit 4.

## Limitations

- Noise estimates are small-sample. A Sonnet paired A/A pilot (6+6 measured
  runs, 1 warmup each, code-review task) measured CVs of about 7% for
  `cost_usd`, 10-13% for duration and output tokens, 16-20% for turns and
  22-27% for `tool_calls_total`. An earlier n=3 legacy cohort had shown 22%
  for cost, most likely inflated by an unrecognised cold-cache run. Treat all
  of these as planning figures, not validated thresholds; the calibration
  tests use the older, more conservative 22%. Setup, dates and the full
  numbers are in `docs/MEASUREMENTS.md`. They come from one machine and one
  task and are not a benchmark.
- No verified B arm in this repository; every real comparison still needs
  its own baseline and treatment runs.
- No quality judging: metrics are behavioural counts, not correctness.
- No transcript analytics beyond the stream's `result` record and
  `tool_use` blocks.
- No ICC or variance-component decomposition. The only inferential methods
  are the bootstrap CI (independent mode) and the sign-flip permutation
  test of the mean (paired mode); neither is a formal power guarantee.
- `compare` requires an exact match on Claude CLI version and runner
  source between arms; a CLI or harness update means a fresh baseline.
- Metric direction (decrease/increase is "good") is a user-declared policy
  via `--primary`/`--guardrail`, not an inherent property of the metric.

## Related work / how this differs

- **`vercel-labs/agent-eval`** (MIT) - A/B repeat-run evaluation for Claude
  Code with tool-call counting.
- **`davidcjw/agentmeter`** (MIT) - transcript analytics for agent runs,
  including permission-denial tracking.
- **`yussypu/deja`** (MIT) - variance-component decomposition, ICC, flip
  rate, and paired permutation tests for agent evals.

This tool makes no novelty claim over any of them. It differs mainly in
being deliberately small: Python standard library only, no quality
judging, no ICC or variance decomposition - just isolated run cohorts, a
bootstrap or sign-flip comparison, and a primary/guardrail verdict rule.
