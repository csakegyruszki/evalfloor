#!/usr/bin/env python3
"""Small Claude Code behaviour-eval harness (stdlib only, Python 3.9+).

Measures BEHAVIOUR of `claude -p --output-format stream-json`: tool calls,
subagents, tokens, duration, cost, permission denials - all read from the
stream's final `result` record and its `tool_use` blocks (no transcript
parser needed).

Subcommands: `run` executes a task/variant N times into an ISOLATED cohort
at results/runs/<run_id>/ (manifest.json + metrics.jsonl + result text),
wrapping the original run_once() execution/parsing path. `compare` loads
two cohorts, checks they are actually comparable (same task bytes, model,
Claude CLI version, runner source), then reports a deterministic bootstrap
CI per metric and a PASS/FAIL/INCONCLUSIVE verdict from a declared primary
metric plus optional guardrails.

See README.md for the manifest schema, metric definitions, decision rule,
and exit codes.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, pathlib, random, re, shutil, statistics, subprocess, sys, time, uuid
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RUNS_DIR = RESULTS_DIR / "runs"
# Windows `claude` is an npm shim (claude.cmd/.ps1); resolve it explicitly.
CLAUDE_BIN = shutil.which("claude") or "claude"

SCHEMA_VERSION = 1
# Numeric metrics tracked by `run` summaries and `compare` (exactly what run_once() sets).
METRICS = ("tool_calls_total", "subagents", "num_turns", "output_tokens",
           "cost_usd", "duration_ms", "permission_denials", "tool_errors")
# Release policy (README "Release policy"): cost is the primary outcome; permission denials and failed
# tool calls are guardrails; tool calls, turns, subagents and tokens are diagnostic only.
DEFAULT_PRIMARY = "cost_usd:decrease"
DEFAULT_GUARDRAILS = ("permission_denials:increase", "tool_errors:increase")
DEFAULT_MIN_N = 6  # sanity floor, not a power claim (measured: bootstrap A/A rate 0.12 at n=3)
MIN_MIN_N = 5  # hard floor: bootstrap A/A rate is 0.033 at n=5, unacceptable below that
PAIRED_MIN_N = 6  # exact sign-flip cannot reach p<0.05 below n=6 (min p = 2/2**n = 2/64)
DEFAULT_WARMUP = 1  # first run pays a cold prompt-cache cost (measured: 67,026 vs ~33,500 cache-creation tokens)
DEFAULT_EFFECT = 0.20
DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 0
DEFAULT_MC_PERMS = 20000  # signflip_pvalue Monte Carlo sample count above n=16
Z_SUM = 2.801585  # z_(1-alpha/2)+z_power, alpha=0.05 two-sided, power=0.80

# Exit codes (README.md keeps this in sync): 0 run-ok/compare-PASS, 1 compare-INCONCLUSIVE,
# 2 arg/validation error, 3 run-had-failures, 4 compare-cohort-mismatch (also: failed measured
# runs without --allow-failed-runs, invalid pairing, unverified/mismatched cwd tree), 5 internal,
# 6 compare-FAIL.
EXIT_OK, EXIT_INCONCLUSIVE, EXIT_ARG_ERROR = 0, 1, 2
EXIT_RUN_FAILED, EXIT_COHORT_MISMATCH, EXIT_INTERNAL, EXIT_FAIL = 3, 4, 5, 6

# Guardrails on these metrics fire on ANY adverse mean move, regardless of statistical
# significance (P0-2): a treatment that trades cost for even a slight increase in denials or
# tool errors should never pass just because n is small or the effect isn't "significant".
ZERO_TOLERANCE_METRICS = ("permission_denials", "tool_errors")

# cwd_tree_sha256: directories skipped when walking the working tree, and size limits above
# which the hash is not computed at all (too slow / not meant for huge trees).
CWD_TREE_SKIP_DIRS = (".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                      ".venv", "venv", "env", ".tox", "node_modules", "results")
# Written to every manifest as `cwd_tree_policy`; two hashes are only comparable under the same policy.
CWD_TREE_POLICY = "v1:" + ",".join(CWD_TREE_SKIP_DIRS)
CWD_TREE_MAX_FILES = 5000
CWD_TREE_MAX_BYTES = 100 * 1024 * 1024

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

def run_once(prompt: str, model: str, cwd: pathlib.Path, agents_json: str | None,
             timeout_s: int, system_prompt_file: str | None = None) -> dict:
    """Run one Claude Code invocation; always returns a metrics dict (an `error` field
    marks an expected failure, this never raises for that)."""
    sid = str(uuid.uuid4())
    # Prompt goes over STDIN: the Windows claude.cmd shim truncates a multi-line argv.
    cmd = [CLAUDE_BIN, "-p", "--output-format", "stream-json", "--verbose",
           "--session-id", sid, "--model", model]
    if agents_json: cmd += ["--agents", agents_json]
    if system_prompt_file: cmd += ["--system-prompt-file", str(pathlib.Path(system_prompt_file).resolve())]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), input=prompt, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout_s)
        raw, rc, stderr = proc.stdout or "", proc.returncode, (proc.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        return {"session_id": sid, "error": f"timeout {timeout_s}s", "wall_s": round(time.monotonic() - t0, 1)}
    wall = round(time.monotonic() - t0, 1)
    tools: Counter = Counter()
    subagents, result_rec, nonjson, tool_errors = 0, None, 0, 0
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            nonjson += 1 if line else 0
            continue
        try:
            o = json.loads(line)
        except Exception:
            nonjson += 1; continue
        if o.get("type") == "result": result_rec = o
        content = (o.get("message") or {}).get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    name = b.get("name", "?"); tools[name] += 1
                    subagents += 1 if name in ("Agent", "Task") else 0
                elif isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                    tool_errors += 1  # failed tool calls, permission denials included
    row: dict = {"session_id": sid, "wall_s": wall, "rc": rc, "nonjson_lines": nonjson,
                 "tool_calls_total": sum(tools.values()), "tool_calls": dict(tools.most_common()),
                 "subagents": subagents, "tool_errors": tool_errors}
    if result_rec is None:
        row["error"] = "no `result` record in the stream"; row["stderr_tail"] = stderr
        return row
    u = result_rec.get("usage") or {}
    row.update({
        "is_error": bool(result_rec.get("is_error")), "stop_reason": result_rec.get("stop_reason"),
        "num_turns": result_rec.get("num_turns"), "duration_ms": result_rec.get("duration_ms"),
        "duration_api_ms": result_rec.get("duration_api_ms"), "ttft_ms": result_rec.get("ttft_ms"),
        "cost_usd": result_rec.get("total_cost_usd"), "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"), "cache_read": u.get("cache_read_input_tokens"),
        "cache_creation": u.get("cache_creation_input_tokens"),
        "permission_denials": len(result_rec.get("permission_denials") or []),
        "result_text_len": len(result_rec.get("result") or ""),
        # Raw text for scoring; popped into its own file by cmd_run(), never in metrics.jsonl.
        "_result_text": result_rec.get("result") or "",
    })
    return row

def is_completed_row(row: dict) -> bool:
    """Failed if truthy `error`/`is_error`, non-zero `rc`, or any tracked metric missing/non-numeric."""
    if row.get("error") or row.get("is_error"): return False
    rc = row.get("rc")
    if rc is not None and rc != 0: return False
    return all(isinstance(row.get(m), (int, float)) for m in METRICS)

def summarize(rows: list) -> dict:
    """n_total/n_completed/n_failed plus per-metric n/mean/median/stdev/cv/min/max (completed rows only; stdev is sample stdev ddof=1; cv null when mean==0 or n<2)."""
    completed = [r for r in rows if is_completed_row(r)]
    out: dict = {"n_total": len(rows), "n_completed": len(completed),
                 "n_failed": len(rows) - len(completed), "metrics": {}}
    for m in METRICS:
        vals = [r[m] for r in completed if isinstance(r.get(m), (int, float))]
        if not vals:
            out["metrics"][m] = None
            continue
        n, mean = len(vals), statistics.fmean(vals)
        sd = statistics.stdev(vals) if n >= 2 else None
        out["metrics"][m] = {"n": n, "mean": mean, "median": statistics.median(vals), "stdev": sd,
                              "cv": (sd / mean) if sd is not None and mean != 0 else None,
                              "min": min(vals), "max": max(vals)}
    return out

# --- manifest / cohort isolation ---

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def resolve_claude_version() -> str:
    """`claude --version` output, or "unavailable:<reason>" - never raises."""
    try:
        proc = subprocess.run([CLAUDE_BIN, "--version"], capture_output=True, text=True, timeout=10)
        out = (proc.stdout or proc.stderr or "").strip()
        return out if out else "unavailable:empty_output"
    except Exception as e:
        return f"unavailable:{e}"

def compute_cwd_tree_sha256(cwd) -> str:
    """Content identity of the working tree under CWD_TREE_POLICY, so the evaluated tree, not the
    `cwd` path string, is part of cohort identity. Entries are sorted by relative POSIX path (UTF-8
    byte order); a regular file contributes `path \\0 sha256(bytes) \\n`, a symlink
    `path \\0 symlink:<target> \\n` and is never followed; mtime and permissions never enter the
    hash. Directories named in CWD_TREE_SKIP_DIRS are skipped at any depth. Fails closed with
    "unavailable:<reason>" on special files (sockets, FIFOs, devices), unreadable files, or more
    than CWD_TREE_MAX_FILES entries / CWD_TREE_MAX_BYTES bytes. Never raises."""
    try:
        root = pathlib.Path(cwd)
        entries, total_size = [], 0
        for dirpath, dirs, files in os.walk(root, followlinks=False):
            descend = []
            for d in dirs:
                if d in CWD_TREE_SKIP_DIRS: continue
                (files if (pathlib.Path(dirpath) / d).is_symlink() else descend).append(d)
            dirs[:] = descend  # a symlinked directory is recorded as a link, never entered
            for name in files:
                path = pathlib.Path(dirpath) / name
                rel = path.relative_to(root).as_posix()
                if path.is_symlink():
                    entries.append((rel, "symlink:" + os.readlink(path)))
                elif path.is_file():
                    total_size += path.stat().st_size
                    if total_size > CWD_TREE_MAX_BYTES: return "unavailable:too_large"
                    entries.append((rel, sha256_file(path)))
                else:
                    return f"unavailable:special_file:{rel}"
                if len(entries) > CWD_TREE_MAX_FILES: return "unavailable:too_large"
        entries.sort(key=lambda e: e[0].encode("utf-8"))
        h = hashlib.sha256()
        for rel, value in entries:
            h.update(rel.encode("utf-8") + b"\0" + value.encode("utf-8") + b"\n")
        return h.hexdigest()
    except Exception as e:  # unreadable file, permission error, broken path: fail closed
        return f"unavailable:{type(e).__name__}"

def build_manifest(run_id, model, variant, task_path, agents_sha256, system_prompt_sha256,
                    cwd, pair_id, cli_options, cwd_tree_sha256) -> dict:
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model, "variant": variant, "task_path": str(task_path),
            "task_sha256": sha256_file(task_path), "agents_sha256": agents_sha256,
            "system_prompt_sha256": system_prompt_sha256, "cwd": str(cwd),
            "cwd_tree_sha256": cwd_tree_sha256, "cwd_tree_policy": CWD_TREE_POLICY,
            "claude_version": resolve_claude_version(),
            "runner_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "pair_id": pair_id, "cli_options": cli_options}

def write_json_atomic(path, obj) -> None:
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8"); tmp.replace(path)

class ArgError(Exception):
    """Raised for a CLI input problem inside cmd_run(); caught once at the top."""

def _resolve_optional(path_str, label):
    if not path_str: return None
    p = pathlib.Path(path_str).resolve()
    if not p.exists(): raise ArgError(f"{label} not found: {p}")
    return p

def cmd_run(args) -> int:
    task_path = pathlib.Path(args.task).resolve()
    if not task_path.exists():
        print(f"ERROR: task file not found: {task_path}", file=sys.stderr); return EXIT_ARG_ERROR
    prompt = task_path.read_text(encoding="utf-8").strip()
    if not prompt: print("ERROR: task file is empty", file=sys.stderr); return EXIT_ARG_ERROR
    if args.repeat < 1: print("ERROR: --repeat must be >= 1", file=sys.stderr); return EXIT_ARG_ERROR
    if args.warmup < 0: print("ERROR: --warmup must be >= 0", file=sys.stderr); return EXIT_ARG_ERROR
    try:
        ap_path = _resolve_optional(args.agents, "agents file")
        sp_path = _resolve_optional(args.system_prompt_file, "system prompt file")
    except ArgError as e:
        print(f"ERROR: {e}", file=sys.stderr); return EXIT_ARG_ERROR
    agents_json = ap_path.read_text(encoding="utf-8") if ap_path else None
    agents_sha = sha256_file(ap_path) if ap_path else None
    system_prompt_sha = sha256_file(sp_path) if sp_path else None
    cwd = pathlib.Path(args.cwd).resolve() if args.cwd else task_path.parent
    run_id = args.run_id or str(uuid.uuid4())
    if not RUN_ID_RE.match(run_id):
        print(f"ERROR: --run-id must match {RUN_ID_RE.pattern}: {run_id}", file=sys.stderr); return EXIT_ARG_ERROR
    run_dir = (RUNS_DIR / run_id).resolve()
    if run_dir.parent != RUNS_DIR.resolve():  # P1-c: reject path traversal (e.g. "../escaped", "a/b") before creating anything
        print(f"ERROR: --run-id must resolve to a direct child of {RUNS_DIR}: {run_id}", file=sys.stderr); return EXIT_ARG_ERROR
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"ERROR: run-id already exists, refusing to overwrite: {run_id}", file=sys.stderr); return EXIT_ARG_ERROR
    metrics_path, manifest_path = run_dir / "metrics.jsonl", run_dir / "manifest.json"
    if args.out is not None and pathlib.Path(args.out).resolve() != metrics_path.resolve():
        print(f"ERROR: --out must equal this run's own metrics.jsonl ({metrics_path})", file=sys.stderr)
        return EXIT_ARG_ERROR  # prevents silently reintroducing the old mixed-history append behaviour
    cli_options = {"model": args.model, "variant": args.variant, "task": str(task_path), "repeat": args.repeat,
                   "timeout": args.timeout, "agents": args.agents, "system_prompt_file": args.system_prompt_file,
                   "cwd": str(cwd), "warmup": args.warmup}
    cwd_tree_sha = compute_cwd_tree_sha256(cwd)  # computed before the first run (P1-a)
    manifest = build_manifest(run_id, args.model, args.variant, task_path, agents_sha, system_prompt_sha,
                               cwd, args.pair_id, cli_options, cwd_tree_sha)
    try:
        with manifest_path.open("x", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
    except FileExistsError:
        print(f"ERROR: manifest already exists for run-id: {run_id}", file=sys.stderr); return EXIT_ARG_ERROR

    def do_one(i, total, label, warmup, filename_prefix):
        print(f"[{label} {i}/{total}] {task_path.name} variant={args.variant} model={args.model} ...", flush=True)
        row = run_once(prompt, args.model, cwd, agents_json, args.timeout, args.system_prompt_file)
        row.update({"task": task_path.name, "variant": args.variant, "model": args.model,
                    "run_id": run_id, "manifest_file": "manifest.json", "warmup": warmup})
        if warmup:
            row["run_index"] = -i  # negative index keeps warmup rows distinguishable from measured run_index 1..repeat
        else:
            row["run_index"] = i
            # Per-row key: run i of a baseline cohort pairs with run i of a treatment cohort sharing --pair-id.
            if args.pair_id: row["pair_id"] = f"{args.pair_id}:{i}"
        text = row.pop("_result_text", "")
        if text:
            txt_path = run_dir / f"{filename_prefix}{i}.txt"; txt_path.write_text(text, encoding="utf-8")
            row["result_text_file"] = txt_path.name
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row.get("error"):
            print(f"    ERROR: {row['error']}", flush=True)
        else:
            print(f"    tool={row['tool_calls_total']} subagent={row['subagents']} turns={row['num_turns']} "
                  f"out_tok={row['output_tokens']} ${row['cost_usd']:.4f} {row['duration_ms']}ms", flush=True)
        return row

    for i in range(1, args.warmup + 1):
        warmup_row = do_one(i, args.warmup, "warmup", True, "warmup")
        if not is_completed_row(warmup_row):
            # P0-3: a failed warmup stops before any measured run - the manifest and the failed
            # warmup row stay on disk (already written above) for inspection.
            print(f"ERROR: warmup run {i}/{args.warmup} failed: {warmup_row.get('error', 'not completed')}",
                  file=sys.stderr)
            return EXIT_RUN_FAILED
    rows = [do_one(i, args.repeat, "run", False, "run") for i in range(1, args.repeat + 1)]
    summary = summarize(rows)
    print(f"\n=== summary ({args.variant}) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"\nrun dir: {run_dir}")
    return EXIT_OK if summary["n_failed"] == 0 else EXIT_RUN_FAILED

# --- compare subcommand ---

def load_cohort(run_dir) -> tuple:
    """(manifest, rows) for a run dir; raises ValueError on missing/malformed input."""
    run_dir = pathlib.Path(run_dir)
    manifest_path, metrics_path = run_dir / "manifest.json", run_dir / "metrics.jsonl"
    if not manifest_path.exists() or not metrics_path.exists():
        raise ValueError(f"missing manifest.json or metrics.jsonl under {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"malformed manifest.json in {run_dir}: {e}")
    rows = []
    for line_no, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line: continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"malformed metrics.jsonl line {line_no} in {run_dir}: {e}")
    return manifest, rows

def validate_comparable_cohorts(base_manifest, treat_manifest, base_rows, treat_rows,
                                 allow_unverified_cwd=False) -> list:
    """List of mismatched field names; empty means comparable."""
    identity = ("schema_version", "task_sha256", "model", "claude_version", "runner_sha256", "cwd_tree_policy")
    # A field missing from both manifests would compare equal (None == None), so absence is itself a mismatch.
    mismatches = [f for f in identity if base_manifest.get(f) is None or treat_manifest.get(f) is None
                  or base_manifest.get(f) != treat_manifest.get(f)]
    if any(str(m.get("claude_version", "")).startswith("unavailable:") for m in (base_manifest, treat_manifest)):
        mismatches.append("claude_version_unavailable")
    # P1-a: cwd_tree_sha256 is part of cohort identity (the `cwd` string alone is not - two
    # different checkouts of the same tree must compare as identical). An "unavailable:..."
    # value on either side is rejected unless --allow-unverified-cwd is given, in which case the
    # two manifests' `cwd` paths must be equal instead.
    base_tree, treat_tree = base_manifest.get("cwd_tree_sha256"), treat_manifest.get("cwd_tree_sha256")
    unverified = (base_tree is None or str(base_tree).startswith("unavailable:")
                  or treat_tree is None or str(treat_tree).startswith("unavailable:"))
    if unverified:
        if not allow_unverified_cwd or base_manifest.get("cwd") != treat_manifest.get("cwd"):
            mismatches.append("cwd_tree_sha256")
    elif base_tree != treat_tree:
        mismatches.append("cwd_tree_sha256")
    if base_manifest.get("variant") == treat_manifest.get("variant"): mismatches.append("variant_not_distinct")
    for label, manifest, rows in (("baseline", base_manifest, base_rows), ("treatment", treat_manifest, treat_rows)):
        ids = [r.get("session_id") for r in rows]
        if len(ids) != len(set(ids)): mismatches.append(f"{label}_duplicate_session_id")
        if any(r.get("run_id") != manifest.get("run_id") for r in rows):
            mismatches.append(f"{label}_row_run_id_mismatch")
    return mismatches

def cohort_is_paired(base_completed: list, treat_completed: list) -> bool:
    """True only if every completed row in both arms has a non-null pair_id, unique within its arm, with matching id sets across arms."""
    if not base_completed or not treat_completed: return False
    base_ids = [r.get("pair_id") for r in base_completed]
    treat_ids = [r.get("pair_id") for r in treat_completed]
    if any(pid is None for pid in base_ids + treat_ids): return False
    if len(set(base_ids)) != len(base_ids) or len(set(treat_ids)) != len(treat_ids): return False
    return set(base_ids) == set(treat_ids)

def pair_diffs(name: str, base_rows: list, treat_rows: list):
    """Paired (treatment-baseline) diffs in deterministic order, or None if this metric is not fully pairable (caller falls back to independent mode)."""
    base_map = {r.get("pair_id"): r.get(name) for r in base_rows if isinstance(r.get(name), (int, float))}
    treat_map = {r.get("pair_id"): r.get(name) for r in treat_rows if isinstance(r.get(name), (int, float))}
    common = set(base_map) & set(treat_map)
    if not common or len(common) < min(len(base_map), len(treat_map)): return None
    return [treat_map[k] - base_map[k] for k in sorted(common, key=str)]

def bootstrap_delta_ci(base_vals, treat_vals, seed: int, resamples: int, paired_diffs=None) -> tuple:
    """95% percentile bootstrap CI of the delta via random.Random(seed) - deterministic for a fixed
    seed. Paired mode resamples the MEAN of the paired diffs (P2: matches the reported point
    estimate, which is also the mean - a median-of-resamples CI was answering a different
    question than the delta it was attached to). Independent mode resamples each arm with
    replacement and reports the median delta, unchanged."""
    rnd = random.Random(seed)
    deltas = []
    if paired_diffs is not None:
        n = len(paired_diffs)
        for _ in range(resamples):
            deltas.append(statistics.fmean([paired_diffs[rnd.randrange(n)] for _ in range(n)]))
    else:
        nb, nt = len(base_vals), len(treat_vals)
        for _ in range(resamples):
            bs = [base_vals[rnd.randrange(nb)] for _ in range(nb)]
            ts = [treat_vals[rnd.randrange(nt)] for _ in range(nt)]
            deltas.append(statistics.median(ts) - statistics.median(bs))
    deltas.sort()
    def q(p):
        idx = p * (resamples - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        return deltas[lo] if lo == hi else deltas[lo] + (deltas[hi] - deltas[lo]) * (idx - lo)
    return q(0.025), q(0.975)

def signflip_pvalue(diffs: list, seed: int, mc_perms: int = DEFAULT_MC_PERMS) -> float:
    """Exact two-sided sign-flip permutation p-value of the MEAN paired difference. Full
    enumeration of all 2**n sign patterns for n<=16; seeded Monte Carlo (random.Random(seed))
    above that, counting the observed pattern itself so p is never 0. Zero diffs carry no sign
    information and are excluded, as in the standard sign test."""
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0: return 1.0
    observed = abs(statistics.fmean(diffs))
    tol = 1e-9
    if n <= 16:
        hits, total = 0, 1 << n
        for mask in range(total):
            s = sum(diffs[i] if (mask >> i) & 1 else -diffs[i] for i in range(n))
            if abs(s / n) >= observed - tol: hits += 1
        return hits / total
    rnd = random.Random(seed)
    hits = 1  # the observed pattern counts as one of the permutations
    for _ in range(mc_perms):
        s = sum(d if rnd.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= observed - tol: hits += 1
    return hits / (mc_perms + 1)

def compare_metric(name, base_rows, treat_rows, min_n, seed, resamples, effect, paired,
                    mc_perms=DEFAULT_MC_PERMS) -> dict:
    """Per-metric arm stats, bootstrap CI, and a direction-neutral result (DECREASED/INCREASED/INCONCLUSIVE),
    plus the baseline CV and a planning n-per-arm estimate for `effect` (not a power guarantee).
    Independent mode: result from the 95% bootstrap CI of the median delta (unchanged). Paired
    mode: result from an exact/Monte-Carlo sign-flip p-value of the MEAN paired difference
    (p<0.05), direction from its sign; the bootstrap CI is still reported but is descriptive
    only there (percentile bootstrap A/A rates were measured anti-conservative, worst in paired
    mode - see README). A metric with fewer than PAIRED_MIN_N=6 pairs is always INCONCLUSIVE:
    the exact sign-flip test cannot reach p<0.05 below n=6 (min p = 2/2**n)."""
    base_vals = [r[name] for r in base_rows if isinstance(r.get(name), (int, float))]
    treat_vals = [r[name] for r in treat_rows if isinstance(r.get(name), (int, float))]
    def arm(vals):
        if not vals: return {"n": 0, "mean": None, "median": None, "stdev": None, "cv": None}
        mean, sd = statistics.fmean(vals), (statistics.stdev(vals) if len(vals) >= 2 else None)
        return {"n": len(vals), "mean": mean, "median": statistics.median(vals), "stdev": sd,
                "cv": (sd / mean) if sd is not None and mean != 0 else None}
    base_desc, treat_desc = arm(base_vals), arm(treat_vals)
    # Planning estimate from the BASELINE arm's CV: pooling both arms would fold a real shift into the noise.
    pcv = base_desc["cv"]
    n_needed = None if pcv is None else max(2, math.ceil(2 * Z_SUM * Z_SUM * pcv * pcv / (effect * effect)))
    diffs = pair_diffs(name, base_rows, treat_rows) if paired else None
    # P2: the estimand is declared regardless of the min_n gate below, so a caller can always see
    # which quantity this metric's delta/CI answer for - mean of paired diffs, or median delta.
    estimand = "mean_paired_difference" if diffs is not None else "median_difference"
    result = {"baseline": base_desc, "treatment": treat_desc, "planning_cv": pcv, "n_per_arm_for_effect": n_needed,
              "p_value": None, "estimand": estimand}
    if base_desc["n"] < min_n or treat_desc["n"] < min_n:
        result.update({"delta": None, "ci_low": None, "ci_high": None, "result": "INCONCLUSIVE"}); return result
    if diffs is not None:
        lo, hi = bootstrap_delta_ci(None, None, seed, resamples, paired_diffs=diffs)
        delta = statistics.fmean(diffs)
        if len(diffs) < PAIRED_MIN_N:
            verdict, p = "INCONCLUSIVE", None
        else:
            p = signflip_pvalue(diffs, seed, mc_perms)
            verdict = ("DECREASED" if delta < 0 else "INCREASED") if p < 0.05 else "INCONCLUSIVE"
        result["p_value"] = p
    else:
        lo, hi = bootstrap_delta_ci(base_vals, treat_vals, seed, resamples)
        delta = treat_desc["median"] - base_desc["median"]
        verdict = "INCONCLUSIVE" if lo <= 0 <= hi else ("DECREASED" if hi < 0 else "INCREASED")
    result.update({"delta": delta, "ci_low": lo, "ci_high": hi, "result": verdict})
    return result

def parse_metric_direction(spec: str) -> tuple:
    if ":" not in spec: raise ValueError(f"bad metric:direction spec (want METRIC:decrease|increase): {spec}")
    metric, direction = spec.rsplit(":", 1)
    if metric not in METRICS: raise ValueError(f"unknown metric: {metric} (known: {', '.join(METRICS)})")
    if direction not in ("decrease", "increase"):
        raise ValueError(f"bad direction (want decrease|increase): {direction}")
    return metric, direction

_MOVED = {"decrease": "DECREASED", "increase": "INCREASED"}

def guardrail_fired(metric, direction, metrics_report) -> tuple:
    """(fired, rule) for one guardrail metric+direction. P0-2: a guardrail on a
    ZERO_TOLERANCE_METRICS metric fires on ANY adverse mean move, regardless of statistical
    significance - a treatment that trades cost for even a slight increase in denials or tool
    errors must never pass just because n is small or the move isn't "significant". Guardrails on
    every other metric keep the original statistical rule (the metric's own DECREASED/INCREASED
    result, which already required significance)."""
    res = metrics_report[metric]
    if metric in ZERO_TOLERANCE_METRICS:
        base_mean, treat_mean = res["baseline"]["mean"], res["treatment"]["mean"]
        if base_mean is None or treat_mean is None: return False, "zero_tolerance"
        fired = (treat_mean > base_mean) if direction == "increase" else (treat_mean < base_mean)
        return fired, "zero_tolerance"
    return res["result"] == _MOVED[direction], "statistical"

def overall_verdict(metrics_report, primary_metric, primary_dir, guardrails) -> str:
    """PASS: primary moved in its declared direction and no guardrail fired (see guardrail_fired).
    FAIL: primary moved the opposite way, or a guardrail fired. INCONCLUSIVE otherwise.
    Non-guardrail metrics never affect the verdict."""
    primary_res = metrics_report[primary_metric]["result"]
    opposite = _MOVED["increase" if primary_dir == "decrease" else "decrease"]
    if primary_res == opposite or any(guardrail_fired(m, d, metrics_report)[0] for m, d in guardrails):
        return "FAIL"
    return "PASS" if primary_res == _MOVED[primary_dir] else "INCONCLUSIVE"

def cmd_compare(args) -> int:
    try:
        primary_metric, primary_dir = parse_metric_direction(args.primary)
        specs = list(DEFAULT_GUARDRAILS) if not args.guardrail else args.guardrail
        if specs == ["none"]: specs = []  # explicit opt-out of the default guardrails
        guardrails = [parse_metric_direction(spec) for spec in specs]
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr); return EXIT_ARG_ERROR
    if args.bootstrap < 1000:
        print(json.dumps({"error": "--bootstrap must be >= 1000"}), file=sys.stderr); return EXIT_ARG_ERROR
    if args.min_n < MIN_MIN_N or not (math.isfinite(args.effect) and args.effect > 0):  # below n=5 the bootstrap A/A rate is unacceptable (0.12 at n=3)
        print(json.dumps({"error": f"--min-n must be >= {MIN_MIN_N} and --effect > 0"}), file=sys.stderr); return EXIT_ARG_ERROR
    try:
        base_manifest, base_rows = load_cohort(args.baseline)
        treat_manifest, treat_rows = load_cohort(args.treatment)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr); return EXIT_COHORT_MISMATCH
    mismatches = validate_comparable_cohorts(base_manifest, treat_manifest, base_rows, treat_rows,
                                              args.allow_unverified_cwd)
    if mismatches:
        report = {"error": "cohort_mismatch", "fields": mismatches}
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.report: write_json_atomic(args.report, report)
        return EXIT_COHORT_MISMATCH
    base_completed = [r for r in base_rows if is_completed_row(r) and not r.get("warmup")]
    treat_completed = [r for r in treat_rows if is_completed_row(r) and not r.get("warmup")]
    # P0-1: a failed MEASURED row (warmup rows excluded) in either cohort makes the comparison
    # invalid by default - silently dropping it would introduce selection bias (the cohort that
    # kept fewer/easier runs looks artificially better). --allow-failed-runs opts back in, but
    # then the verdict can never be PASS.
    base_failed = [r for r in base_rows if not r.get("warmup") and not is_completed_row(r)]
    treat_failed = [r for r in treat_rows if not r.get("warmup") and not is_completed_row(r)]
    if (base_failed or treat_failed) and not args.allow_failed_runs:
        report = {"error": "failed_runs", "validity": "INVALID_FAILED_RUNS",
                  "failed_baseline": len(base_failed), "failed_treatment": len(treat_failed)}
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.report: write_json_atomic(args.report, report)
        return EXIT_COHORT_MISMATCH
    # P1-b: a manifest-level pair_id on either side commits both cohorts to paired analysis - if
    # they don't actually agree (same pair_id, and every measured completed row pairs up 1:1),
    # that is a broken experiment design, not something to silently fall back to independent mode for.
    base_pair_id, treat_pair_id = base_manifest.get("pair_id"), treat_manifest.get("pair_id")
    if base_pair_id is not None or treat_pair_id is not None:
        if base_pair_id != treat_pair_id or not cohort_is_paired(base_completed, treat_completed):
            report = {"error": "pairing_invalid", "baseline_pair_id": base_pair_id, "treatment_pair_id": treat_pair_id}
            print(json.dumps(report, indent=2, sort_keys=True))
            if args.report: write_json_atomic(args.report, report)
            return EXIT_COHORT_MISMATCH
    paired = cohort_is_paired(base_completed, treat_completed)
    metrics_report = {m: compare_metric(m, base_completed, treat_completed, args.min_n, args.seed,
                                         args.bootstrap, args.effect, paired) for m in METRICS}
    verdict = overall_verdict(metrics_report, primary_metric, primary_dir, guardrails)
    if (base_failed or treat_failed) and verdict == "PASS":
        verdict = "INCONCLUSIVE"  # --allow-failed-runs: failed measured runs cap the verdict, PASS is never allowed
    def total(rows, metric):
        return sum(r[metric] for r in rows if isinstance(r.get(metric), (int, float)))
    # Zero-tolerance guardrails are deterministic policy checks, not inference: with equal measured run
    # counts, any increase of the aggregate count fails (equivalent to the per-run mean compared in
    # guardrail_fired); with unequal counts (only possible with --allow-failed-runs or unequal --repeat)
    # the per-run mean is the basis, and the report says so.
    guardrails_detail = []
    for m, d in guardrails:
        fired, rule = guardrail_fired(m, d, metrics_report)
        bt, tt = total(base_completed, m), total(treat_completed, m)
        guardrails_detail.append({
            "metric": m, "direction": d, "rule": rule, "fired": fired,
            "baseline_total": bt, "treatment_total": tt, "delta_total": tt - bt,
            "baseline_n": len(base_completed), "treatment_n": len(treat_completed),
            "basis": "aggregate_count" if len(base_completed) == len(treat_completed) else "per_run_mean"})
    report = {"baseline_run_id": base_manifest.get("run_id"), "treatment_run_id": treat_manifest.get("run_id"),
              "baseline_variant": base_manifest.get("variant"), "treatment_variant": treat_manifest.get("variant"),
              "paired": paired, "primary": f"{primary_metric}:{primary_dir}",
              "guardrails": [f"{m}:{d}" for m, d in guardrails], "guardrails_detail": guardrails_detail,
              "seed": args.seed, "bootstrap": args.bootstrap, "min_n": args.min_n, "effect": args.effect,
              "validity": "DEGRADED_FAILED_RUNS" if (base_failed or treat_failed) else "VALID",
              "failed_baseline": len(base_failed), "failed_treatment": len(treat_failed),
              "metrics": metrics_report, "verdict": verdict}
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report: write_json_atomic(args.report, report)
    return {"PASS": EXIT_OK, "FAIL": EXIT_FAIL}.get(verdict, EXIT_INCONCLUSIVE)

# --- CLI wiring ---

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Execute a task/variant and record an isolated run cohort.")
    run_p.add_argument("--task", required=True, help="task file (its content is the prompt)")
    run_p.add_argument("--variant", required=True, help="variant name (e.g. baseline, no-persona)")
    run_p.add_argument("--repeat", type=int, default=1); run_p.add_argument("--timeout", type=int, default=600)
    run_p.add_argument("--model", default="claude-haiku-4-5-20251001", help="PINNED for reproducibility")
    run_p.add_argument("--agents", default=None, help="agent-definition JSON file (passed to --agents)")
    run_p.add_argument("--system-prompt-file", default=None, help="A/B variant's only difference")
    run_p.add_argument("--cwd", default=None, help="working directory (default: the task file's directory)")
    run_p.add_argument("--out", default=None, help="must equal the run's own metrics.jsonl path if given")
    run_p.add_argument("--run-id", default=None, help="explicit run id (default: generated UUID4)")
    run_p.add_argument("--pair-id", default=None, help="pairing key stored in manifest and on every row")
    run_p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="runs executed before the measured ones, to pay the cold prompt-cache cost (default 1)")
    cmp_p = sub.add_parser("compare", help="Compare two run cohorts with a bootstrap CI and verdict.")
    cmp_p.add_argument("--baseline", required=True, help="baseline run directory (results/runs/<id>)")
    cmp_p.add_argument("--treatment", required=True, help="treatment run directory (results/runs/<id>)")
    cmp_p.add_argument("--primary", default=DEFAULT_PRIMARY,
                        help=f"METRIC:decrease|increase (default {DEFAULT_PRIMARY})")
    cmp_p.add_argument("--guardrail", action="append", default=None,
                        help="METRIC:increase|decrease, repeatable; replaces the defaults "
                             f"({', '.join(DEFAULT_GUARDRAILS)}); 'none' disables them")
    cmp_p.add_argument("--min-n", type=int, default=DEFAULT_MIN_N); cmp_p.add_argument("--effect", type=float, default=DEFAULT_EFFECT)
    cmp_p.add_argument("--seed", type=int, default=DEFAULT_SEED); cmp_p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    cmp_p.add_argument("--report", default=None, help="optional path to also write the JSON report")
    cmp_p.add_argument("--allow-failed-runs", action="store_true",
                        help="proceed despite failed measured rows in either cohort; caps the verdict at INCONCLUSIVE (never PASS)")
    cmp_p.add_argument("--allow-unverified-cwd", action="store_true",
                        help="accept an 'unavailable:...' cwd_tree_sha256 if both manifests' --cwd paths are equal")
    return ap

def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "run": return cmd_run(args)
        if args.command == "compare": return cmd_compare(args)
    except Exception as e:  # pragma: no cover - unexpected internal failure
        print(json.dumps({"error": f"internal error: {e}"}), file=sys.stderr)
        return EXIT_INTERNAL
    return EXIT_INTERNAL  # pragma: no cover - argparse enforces a valid command

if __name__ == "__main__":
    sys.exit(main())