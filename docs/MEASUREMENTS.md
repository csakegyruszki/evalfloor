# Measurements behind the README

These are the author's measurements that the README quotes. They come from one
Windows machine, Claude Code 2.1.269, and two small tasks. They document why the
defaults are what they are; they are not a benchmark and do not generalise to
other tasks or models. Raw cohort files are not published: they contain local
paths and model output.

## 1. Cold prompt-cache order effect (Haiku, 2026-09-12)

Task `examples/tasks/noise-floor.md`, cwd `examples/fixtures/noise`,
`claude-haiku-4-5-20251001`, two identical cohorts of 3 runs, run back to back,
no warmup.

| run | cache-creation tokens | cost (USD) |
|---|---:|---:|
| cohort A, run 1 | 67,026 | 0.1519 |
| cohort A, runs 2-3 | 37,117 / 33,539 | 0.0952 / 0.0881 |
| cohort B, runs 1-3 | 33,147 / 33,553 / 33,524 | 0.0880 / 0.0883 / 0.0877 |

The first run of the first cohort paid for building the prompt cache. With
`--min-n 3` (below today's floor) the identical cohorts were reported as a cost
`DECREASED` / `PASS`: a false result. This motivated `--warmup` and the floors.

## 2. False-positive rate of the decision methods (simulation)

A/A comparisons (both arms from the same distribution) on a cost distribution
with mean 0.613 and CV 0.22, 400 simulations, 1,000 bootstrap resamples. The
rate is the share of simulations in which the method declared a difference.

| n per arm | bootstrap, independent | bootstrap, paired | exact sign-flip, paired |
|---:|---:|---:|---:|
| 3 | 0.120 | 0.270 | 0.000 |
| 5 | 0.033 | 0.058 | 0.000 |
| 10 | 0.037 | 0.075 | 0.058 |
| 20 | 0.043 | 0.055 | not enumerated |

The paired bootstrap exceeds 0.05 at every n, hence the sign-flip test in
paired mode. The exact sign-flip test cannot reach p < 0.05 below n = 6
(minimum p = 2 / 2^n).

## 3. Haiku paired A/A smoke with warmup (2026-09-12)

Same task, two identical cohorts of 1 warmup + 6 measured runs, paired.
Verdict `INCONCLUSIVE`; every metric `INCONCLUSIVE` (cost p = 0.0625).

## 4. Sonnet paired A/A pilot (2026-09-12)

Task `examples/tasks/code-review.md`, system prompt
`examples/variants/code-reviewer-A-persona.txt`, cwd `examples/fixtures/review`,
`claude-sonnet-5`, two identical cohorts of 1 warmup + 6 measured runs, paired.
All 14 runs completed; total cost USD 5.35 including warmups. Verdict
`INCONCLUSIVE`; every metric `INCONCLUSIVE`.

| metric | CV cohort A | CV cohort B | sign-flip p | planning n per arm for a 20% effect |
|---|---:|---:|---:|---:|
| cost_usd | 6.8% | 7.7% | 0.44 | 2 (the paired floor of 6 applies) |
| duration_ms | 10.2% | 11.7% | 0.31 | 5 |
| output_tokens | 11.3% | 13.3% | 0.44 | 5 |
| num_turns | 15.6% | 19.6% | 0.75 | 10 |
| tool_calls_total | 21.9% | 26.6% | 0.75 | 19 |

The first warmup cost USD 0.519 against 0.34-0.39 for measured runs. An earlier
3-run cohort on the same task, without warmup, had shown a 22% cost CV; its most
expensive run shows the same ratio to the others, so that figure was most likely
inflated by a cold-cache run (inferred; those runs recorded no cache fields).

## 5. Calibration tests (in `test_run_eval.py`)

The test suite re-checks the decision rule on every run: A/A false-PASS rate at
most 0.05 at the default `--min-n`, in both modes; a known 30% cost decrease
detected at least 80% of the time with 20 per arm; a primary improvement with an
adverse guardrail never yields `PASS`.
