import numpy as np
from pathlib import Path
from hazard_gev import flow_to_head, gev_quantile

OUT_DIR = Path("outputs")
def curve_eval(h, h_grid, curve):
    return np.interp(h, h_grid, curve)

haz = np.load(OUT_DIR / "hazard.npz", allow_pickle=False)
fr = np.load(OUT_DIR / "fragility.npz")
h_grid = fr["h_grid"]; curves = fr["curves"]
mu, sigma, xi = haz["post_mu"], haz["post_sigma"], haz["post_xi"]

# Variance decomposition: for a SET of fixed (j, curve) epistemic draws,
# estimate the true conditional mean very precisely (n_years huge) vs n=4000.
# Var(Pf_pairs) = Var_epistemic(true means) + E[per-pair MC var].
# If MC leakage is material it inflates the SECOND term relative to the first.
rng = np.random.default_rng(2026)
n_pairs = 1000
js = rng.integers(len(mu), size=n_pairs)
ks = rng.integers(len(curves), size=n_pairs)

def pf_for(j, k, n_years, rng):
    u = rng.uniform(size=n_years)
    flow = gev_quantile(u, mu[j], sigma[j], xi[j])
    flow = flow[np.isfinite(flow) & (flow > 0)]
    return np.mean(curve_eval(flow_to_head(flow), h_grid, curves[k]))

# true conditional means at large n
rng2 = np.random.default_rng(123)
truemeans = np.array([pf_for(js[p], ks[p], 200000, rng2) for p in range(n_pairs)])
var_epistemic = np.var(truemeans, ddof=1)

# per-pair MC variance at n=4000, averaged
rng3 = np.random.default_rng(456)
mc_vars = []
for p in range(0, n_pairs, 10):  # subsample for speed
    reps = np.array([pf_for(js[p], ks[p], 4000, rng3) for _ in range(30)])
    mc_vars.append(np.var(reps, ddof=1))
mean_mc_var = np.mean(mc_vars)

print(f"Var_epistemic (true means)      = {var_epistemic:.3e}  sd={np.sqrt(var_epistemic):.5f}")
print(f"E[per-pair MC var] @ n=4000      = {mean_mc_var:.3e}  sd={np.sqrt(mean_mc_var):.5f}")
print(f"MC var fraction of total         = {mean_mc_var/(var_epistemic+mean_mc_var)*100:.2f}%")
# inflation of the sd of Pf_pairs due to MC leakage:
infl = np.sqrt(var_epistemic+mean_mc_var)/np.sqrt(var_epistemic)
print(f"sd inflation factor              = {infl:.4f}  ({(infl-1)*100:.2f}% wider)")
