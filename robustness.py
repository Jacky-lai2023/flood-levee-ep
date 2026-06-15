"""Realization robustness: how much does the breach probability depend on the single
synthetic foundation field? Re-runs the k-field -> fragility -> breach-EP chain across
foundation seeds (the flood hazard is fixed real data, so hazard_gev.py is NOT re-run).

This backs the README's "conditional on one realization" disclosure with reproducible
numbers. Run:  uv run python robustness.py            # default 14 seeds
              uv run python robustness.py 30          # more seeds
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEEDS = [2026, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def run(mod, env=None):
    subprocess.run([sys.executable, str(HERE / mod)], check=True,
                   capture_output=True, text=True, env=env)


def pf_for_seed(seed):
    env = dict(os.environ, KFIELD_SEED=str(seed))
    run("kfield.py", env)
    run("fragility.py")
    out = subprocess.run([sys.executable, str(HERE / "breach_ep.py")],
                         check=True, capture_output=True, text=True).stdout
    return float(re.search(r"P_f = ([0-9.]+)", out).group(1))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(SEEDS)
    seeds = (SEEDS + list(range(14, 14 + n)))[:n]
    pf = np.array([pf_for_seed(s) for s in seeds])
    rp = 1.0 / pf
    print(f"\nRealization robustness over {len(seeds)} foundation seeds:")
    print(f"  P_f       median {np.median(pf):.4f} | range {pf.min():.4f}–{pf.max():.4f} "
          f"({pf.max()/pf.min():.1f}x spread)")
    print(f"  return    median 1-in-{1/np.median(pf):.0f} yr | range 1-in-{1/pf.max():.0f} "
          f"(worst) … 1-in-{1/pf.min():.0f} yr (best)")
    seed0 = pf[0]
    print(f"  seed 2026 P_f {seed0:.4f} (1-in-{1/seed0:.0f}) — "
          f"{np.mean(pf < seed0)*100:.0f}% of realizations are less pessimistic")
    # restore the canonical seed-2026 outputs
    run("kfield.py")
    run("fragility.py")
    run("breach_ep.py")
    print("  (canonical seed-2026 outputs restored)")


if __name__ == "__main__":
    main()
