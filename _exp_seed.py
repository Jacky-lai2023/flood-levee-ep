"""Experiment: vary the true-field seed across the whole chain and recompute P_f / RP.

Replicates kfield -> fragility -> breach_ep in-memory, changing ONLY the seed that
generates the synthetic 'true' field and the soundings/Gibbs sampler that descend from it.
Hazard (Stage 1) is held fixed (its own posterior catalogue), as in the pipeline.
"""
import sys
import numpy as np

import kfield as K
import fragility as F

OUT = K.OUT_DIR

# Hazard catalogue is fixed (Stage 1 independent of k-field seed)
haz = np.load(OUT / "hazard.npz", allow_pickle=False)
pred_head = haz["pred_head"]


def keff_for_seed(seed):
    rng = np.random.default_rng(seed)
    x, y, xx, yy = K.grid()
    field = K.true_field(rng)
    col_ix = np.linspace(6, K.NX - 7, K.N_COLUMNS).astype(int)
    pts_all = np.c_[xx.ravel(), yy.ravel()]
    obs_mask = np.zeros((K.NY, K.NX), dtype=bool)
    obs_mask[:, col_ix] = True
    obs_pts = pts_all[obs_mask.ravel()]
    obs_val = field.ravel()[obs_mask.ravel()]
    obs_val = obs_val + 0.03 * rng.standard_normal(len(obs_val))
    centres = K.basis_centres()
    Phi_obs = K.rbf_basis(obs_pts, centres, lx=10.0, ly=3.5)
    Phi_all = K.rbf_basis(pts_all, centres, lx=10.0, ly=3.5)
    sampler = K.BayesianLasso(Phi_obs, obs_val, seed=seed)
    beta_s, mu_s = sampler.sample(n_samples=1500, burn=500)
    recon = mu_s[:, None] + beta_s @ Phi_all.T
    L_seep = 70.0
    cx0, cx1 = (K.W - L_seep) / 2, (K.W + L_seep) / 2
    zone = (yy <= 0.4 * K.D) & (xx >= cx0) & (xx <= cx1)
    zone_flat = zone.ravel()
    eff_log10k = recon[:, zone_flat].mean(1)
    keff = 10.0 ** eff_log10k
    true_eff = 10.0 ** field.ravel()[zone_flat].mean()
    return keff, true_eff


def frag_mean_for_keff(keff_post, seed=99):
    rng = np.random.default_rng(seed)
    kdraws = rng.choice(keff_post, size=F.N_KDRAWS, replace=True)
    d70 = F.lognormal(rng, F.D70_MED, F.D70_COV, F.N_ALEATORY)
    D = F.lognormal(rng, F.DAQ_MED, F.DAQ_COV, F.N_ALEATORY)
    L = F.lognormal(rng, F.LSEEP_MED, F.LSEEP_COV, F.N_ALEATORY)
    mp = F.lognormal(rng, F.MP_MED, F.MP_COV, F.N_ALEATORY)
    crest = F.lognormal(rng, F.CREST_MED, F.CREST_COV, F.N_ALEATORY)
    curves = np.empty((F.N_KDRAWS, len(F.H_GRID)))
    for i, k in enumerate(kdraws):
        Hc = F.sellmeijer_critical_head(np.full(F.N_ALEATORY, k), d70, D, L, mp)
        piping = F.H_GRID[:, None] >= Hc[None, :]
        overtop = F.H_GRID[:, None] >= crest[None, :]
        curves[i] = (piping | overtop).mean(1)
    return curves.mean(0)


def pf_for_seed(seed):
    keff, true_eff = keff_for_seed(seed)
    frag_mean = frag_mean_for_keff(keff)
    P_f = float(np.mean(np.interp(pred_head, F.H_GRID, frag_mean)))
    return P_f, true_eff, float(np.median(keff))


seeds = [2026, 1, 2, 7, 42, 99, 123, 777, 2025, 2027, 5000, 31337] + list(range(1, 41))
seeds = list(dict.fromkeys(seeds))  # dedupe, keep order

rows = []
for s in seeds:
    pf, te, km = pf_for_seed(s)
    rows.append((s, pf, te, km))
    print(f"seed {s:6d}  P_f {pf:.4f}  RP 1-in-{1/pf:6.0f}  true_eff {te:.2e}  keff_med {km:.2e}")

pfs = np.array([r[1] for r in rows])
tes = np.array([r[2] for r in rows])
rps = 1 / pfs
pf2026 = rows[0][1]
pctl = 100 * np.mean(pfs <= pf2026)
print("\n--- SUMMARY ---")
print(f"n seeds = {len(rows)}")
print(f"P_f min {pfs.min():.4f}  max {pfs.max():.4f}  ratio {pfs.max()/pfs.min():.1f}x")
print(f"RP median {np.median(rps):.0f} yr  range {rps.min():.0f}-{rps.max():.0f}")
print(f"true_eff min {tes.min():.2e} max {tes.max():.2e} median {np.median(tes):.2e}")
print(f"seed2026 P_f {pf2026:.4f} at percentile {pctl:.0f}% of seed distribution")
print(f"seed2026 RP 1-in-{1/pf2026:.0f} vs median 1-in-{np.median(rps):.0f}")
