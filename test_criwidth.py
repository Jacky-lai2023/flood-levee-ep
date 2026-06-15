import numpy as np
from pathlib import Path
from hazard_gev import flow_to_head, gev_quantile
OUT_DIR = Path("outputs")
def curve_eval(h, hg, c): return np.interp(h, hg, c)
haz = np.load(OUT_DIR/"hazard.npz", allow_pickle=False)
fr = np.load(OUT_DIR/"fragility.npz")
h_grid, curves = fr["h_grid"], fr["curves"]
mu, sigma, xi = haz["post_mu"], haz["post_sigma"], haz["post_xi"]
def run(ny, seed, n_pairs=1000):
    rng = np.random.default_rng(seed)
    Pf = np.empty(n_pairs)
    for p in range(n_pairs):
        j = rng.integers(len(mu)); c = curves[rng.integers(len(curves))]
        flow = gev_quantile(rng.uniform(size=ny), mu[j], sigma[j], xi[j])
        flow = flow[np.isfinite(flow)&(flow>0)]
        Pf[p] = np.mean(curve_eval(flow_to_head(flow), h_grid, c))
    lo,hi = np.quantile(Pf,0.025), np.quantile(Pf,0.975)
    return lo,hi,hi-lo
for ny in [4000, 40000]:
    res = [run(ny,s) for s in [2026,1,7,42,99]]
    w = [r[2] for r in res]; lo=[r[0] for r in res]; hi=[r[1] for r in res]
    print(f"n_years={ny:>6}: width {np.mean(w):.5f}+-{np.std(w):.5f}  lo={np.mean(lo):.5f} hi={np.mean(hi):.5f}")
print(f"seed2026 reported CrI (n=4000): {run(4000,2026)[0]:.4f}-{run(4000,2026)[1]:.4f}")
