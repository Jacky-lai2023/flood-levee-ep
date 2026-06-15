import numpy as np
from pathlib import Path
from hazard_gev import flow_to_head, gev_quantile

OUT_DIR = Path("outputs")
def curve_eval(h, h_grid, curve):
    return np.interp(h, h_grid, curve)

haz = np.load(OUT_DIR / "hazard.npz", allow_pickle=False)
fr = np.load(OUT_DIR / "fragility.npz")
h_grid = fr["h_grid"]
curves = fr["curves"]
mu, sigma, xi = haz["post_mu"], haz["post_sigma"], haz["post_xi"]

def run(n_years, seed=2026, n_pairs=1000):
    rng = np.random.default_rng(seed)
    Pf = np.empty(n_pairs)
    for p in range(n_pairs):
        j = rng.integers(len(mu))
        c = curves[rng.integers(len(curves))]
        u = rng.uniform(size=n_years)
        flow = gev_quantile(u, mu[j], sigma[j], xi[j])
        flow = flow[np.isfinite(flow) & (flow > 0)]
        head = flow_to_head(flow)
        Pf[p] = np.mean(curve_eval(head, h_grid, c))
    lo, hi = np.quantile(Pf, 0.025), np.quantile(Pf, 0.975)
    return lo, hi, hi-lo

# CrI width across n_years and across seeds
for ny in [4000, 40000, 400000]:
    widths = []
    los, his = [], []
    for s in [2026, 1, 7, 42, 99]:
        lo, hi, w = run(ny, seed=s)
        widths.append(w); los.append(lo); his.append(hi)
    print(f"n_years={ny:>7}: width mean={np.mean(widths):.5f} sd={np.std(widths):.5f}  "
          f"lo={np.mean(los):.5f} hi={np.mean(his):.5f}")
