import numpy as np
from pathlib import Path
from hazard_gev import flow_to_head, gev_quantile
from breach_ep import curve_eval

OUT = Path(__file__).parent / "outputs"
haz = np.load(OUT/"hazard.npz", allow_pickle=False)
fr = np.load(OUT/"fragility.npz")
h_grid = fr["h_grid"]; curves = fr["curves"]
mu, sigma, xi = haz["post_mu"], haz["post_sigma"], haz["post_xi"]

# Fix 1000 epistemic pairs (j, k-curve) exactly as production does.
rng = np.random.default_rng(2026)
n_pairs = 1000
js = np.empty(n_pairs, dtype=int); ks = np.empty(n_pairs, dtype=int)
for p in range(n_pairs):
    js[p] = rng.integers(len(mu))
    ks[p] = rng.integers(len(curves))
    rng.uniform(size=4000)  # consume same stream as production (head draw)

def pf_inner(j, kidx, n_years, rng):
    u = rng.uniform(size=n_years)
    flow = gev_quantile(u, mu[j], sigma[j], xi[j])
    flow = flow[np.isfinite(flow)&(flow>0)]
    head = flow_to_head(flow)
    return np.mean(curve_eval(head, h_grid, curves[kidx]))

# For each fixed pair, measure aleatory MC variance across R independent inner runs at n_years=4000.
R = 40
arng = np.random.default_rng(777)
per_pair_mc_var = np.empty(n_pairs)
pair_point = np.empty(n_pairs)  # high-accuracy "true" inner expectation
brng = np.random.default_rng(555)
for p in range(n_pairs):
    reps = np.array([pf_inner(js[p], ks[p], 4000, arng) for _ in range(R)])
    per_pair_mc_var[p] = reps.var(ddof=1)
    pair_point[p] = pf_inner(js[p], ks[p], 200000, brng)  # near-exact inner E

# Variance of the high-accuracy (near-exact) point estimates = PURE epistemic spread
var_epistemic = pair_point.var(ddof=1)
# Mean per-pair aleatory MC variance at n_years=4000 (this is what leaks in)
mean_aleatory_var = per_pair_mc_var.mean()
# Total variance observed at n_years=4000 should ~ var_epistemic + mean_aleatory_var
print(f"Pure epistemic Var (near-exact inner, n=200000):     {var_epistemic:.3e}  sd {np.sqrt(var_epistemic):.5f}")
print(f"Mean per-pair aleatory MC Var (n_years=4000):         {mean_aleatory_var:.3e}  sd {np.sqrt(mean_aleatory_var):.5f}")
print(f"Aleatory leak as fraction of total variance:          {mean_aleatory_var/(var_epistemic+mean_aleatory_var)*100:.2f}%")
print(f"Implied SD inflation: sqrt(1+leak/epi) = {np.sqrt(1+mean_aleatory_var/var_epistemic):.4f}x")

# What the 95% CrI width would be: pure-epistemic vs with-leak (Gaussian approx on quantiles is rough;
# better: directly compare quantile interval of pair_point (exact) vs noisy 4000-draws single estimate)
rng2 = np.random.default_rng(2026)
noisy = np.array([pf_inner(js[p], ks[p], 4000, rng2) for p in range(n_pairs)])
def w(a): 
    lo,hi=np.quantile(a,0.025),np.quantile(a,0.975); return lo,hi,hi-lo
elo,ehi,ew = w(pair_point)
nlo,nhi,nw = w(noisy)
print(f"\nCrI width, near-exact inner (n=200000): [{elo:.5f},{ehi:.5f}] width {ew:.5f}")
print(f"CrI width, production inner (n=4000):   [{nlo:.5f},{nhi:.5f}] width {nw:.5f}")
print(f"Width inflation from MC leak: {(nw-ew)/ew*100:+.1f}%")
