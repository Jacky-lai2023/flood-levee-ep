import numpy as np
from pathlib import Path
from hazard_gev import flow_to_head, gev_quantile
from breach_ep import curve_eval

OUT = Path("/home/jacky/Desktop/personal/RMS_TC/flood-levee-ep/outputs")
haz = np.load(OUT/"hazard.npz", allow_pickle=False)
fr = np.load(OUT/"fragility.npz")
h_grid = fr["h_grid"]; curves = fr["curves"]
mu, sigma, xi = haz["post_mu"], haz["post_sigma"], haz["post_xi"]

def cri(n_years, seed=2026, n_pairs=1000):
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
    return lo, hi, hi-lo, np.std(Pf)

print("=== CrI width vs n_years (seed 2026, the production seed) ===")
for ny in [4000, 20000, 100000, 400000]:
    lo, hi, w, sd = cri(ny)
    print(f"n_years={ny:7d}  CrI [{lo:.5f},{hi:.5f}]  width {w:.5f}  sd {sd:.5f}  RP 1-in-{1/hi:.0f} to 1-in-{1/lo:.0f}")
