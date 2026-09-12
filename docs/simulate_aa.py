"""Reproduce the A/A false-positive table in docs/MEASUREMENTS.md (section 2).

Both arms are drawn from the same cost distribution (mean 0.613, CV 0.22), so any
declared difference is a false positive. Uses the bootstrap and sign-flip functions
from run_eval.py, standard library only, fixed seed. Run from the repository root:

    python docs/simulate_aa.py
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import run_eval as ev  # noqa: E402

SIMS, RESAMPLES, SEED = 4000, 1000, 7  # 4000: Monte Carlo SE about 0.0035 at a 5% rate
MEAN, CV = 0.613, 0.22
MC_PERMS = 2000  # only used above n = 16, where the sign-flip test switches to Monte Carlo


def main() -> None:
    rng = random.Random(SEED)

    def sample(n):
        return [max(0.01, rng.gauss(MEAN, MEAN * CV)) for _ in range(n)]

    print("n    bootstrap-independent  bootstrap-paired  sign-flip-paired")
    for n in (3, 5, 10, 20):
        independent = paired = signflip = 0
        for s in range(SIMS):
            a, b = sample(n), sample(n)
            lo, hi = ev.bootstrap_delta_ci(a, b, s, RESAMPLES)
            independent += not (lo <= 0 <= hi)
            diffs = [y - x for x, y in zip(a, b)]
            lo, hi = ev.bootstrap_delta_ci(None, None, s, RESAMPLES, paired_diffs=diffs)
            paired += not (lo <= 0 <= hi)
            signflip += ev.signflip_pvalue(diffs, s, MC_PERMS) < 0.05
        print(f"{n:<4} {independent / SIMS:>21.3f} {paired / SIMS:>17.3f} {signflip / SIMS:>17.3f}")


if __name__ == "__main__":
    main()
