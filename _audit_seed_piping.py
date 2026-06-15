"""Audit: does the seed-2026 frozen field bias the PIPING SHARE (80%) and basal-zone k?

Recomputes, per true-field seed:
  - basal-central zone mean log10k of the TRUE field (the finding's +0.67 sigma claim)
  - global field mean log10k
  - posterior keff median
  - P_f and the piping share (% of breach exposure involving piping)
across many seeds, to see whether seed 2026's headline (1-in-119, ~80% piping) is an
outlier or typical.
"""
import numpy as np
import kfield as K
import fragility as F

OUT = K.OUT_DIR
haz = np.load(OUT / "hazard.npz", allow_pickle=False)
pred_head = haz["pred_head"]


def basal_zone_mask():
    x, y, xx, yy = K.grid()
    L_seep = 70.0
    cx0, cx1 = (K.W - L_seep) / 2, (K.W + L_seep) / 2
    zone = (yy <= 0.4 * K.D) & (xx >= cx0) & (xx <= cx1)
    return zone.ravel(), xx, yy


ZONE_FLAT, XX, YY = basal_zone_mask()


def true_field_stats(seed):
    rng = np.random.default_rng(seed)
    field = K.true_field(rng)
    fz = field.ravel()
    return float(fz[ZONE_FLAT].mean()), float(fz.mean())


def keff_for_seed(seed):
    rng = np.random.default_rng(seed)
    x, y, xx, yy = K.grid()
    field = K.true_field(rng)
    col_ix = np.linspace(6, K.NX - 7, K.N_COLUMNS).astype(int)
    pts_all = np.c_[xx.ravel(), yy.ravel()]
    obs_mask = np.zeros((K.NY, K.NX), dtype=bool)
    obs_mask[:, col_ix] = True
    obs_pts = pts_all[obs_mask.ravel()]
    obs_val = field.ravel()[obs_mask.ravel()] + 0.03 * rng.standard_normal(obs_mask.sum())
    centres = K.basis_centres()
    Phi_obs = K.rbf_basis(obs_pts, centres, lx=10.0, ly=3.5)
    Phi_all = K.rbf_basis(pts_all, centres, lx=10.0, ly=3.5)
    sampler = K.BayesianLasso(Phi_obs, obs_val, seed=seed)
    beta_s, mu_s = sampler.sample(n_samples=1500, burn=500)
    recon = mu_s[:, None] + beta_s @ Phi_all.T
    eff_log10k = recon[:, ZONE_FLAT].mean(1)
    return 10.0 ** eff_log10k


def pf_and_piping(seed, fseed=99):
    keff_post = keff_for_seed(seed)
    rng = np.random.default_rng(fseed)
    kdraws = rng.choice(keff_post, size=F.N_KDRAWS, replace=True)
    d70 = F.lognormal(rng, F.D70_MED, F.D70_COV, F.N_ALEATORY)
    D = F.lognormal(rng, F.DAQ_MED, F.DAQ_COV, F.N_ALEATORY)
    L = F.lognormal(rng, F.LSEEP_MED, F.LSEEP_COV, F.N_ALEATORY)
    mp = F.lognormal(rng, F.MP_MED, F.MP_COV, F.N_ALEATORY)
    crest = F.lognormal(rng, F.CREST_MED, F.CREST_COV, F.N_ALEATORY)
    piponly = np.empty((F.N_KDRAWS, len(F.H_GRID)))
    ovtonly = np.empty_like(piponly)
    both = np.empty_like(piponly)
    union = np.empty_like(piponly)
    for i, k in enumerate(kdraws):
        Hc = F.sellmeijer_critical_head(np.full(F.N_ALEATORY, k), d70, D, L, mp)
        piping = F.H_GRID[:, None] >= Hc[None, :]
        overtop = F.H_GRID[:, None] >= crest[None, :]
        piponly[i] = (piping & ~overtop).mean(1)
        ovtonly[i] = (overtop & ~piping).mean(1)
        both[i] = (piping & overtop).mean(1)
        union[i] = (piping | overtop).mean(1)
    frag_mean = union.mean(0)
    P_f = float(np.mean(np.interp(pred_head, F.H_GRID, frag_mean)))
    P_pip_only = float(np.mean(np.interp(pred_head, F.H_GRID, piponly.mean(0))))
    P_both = float(np.mean(np.interp(pred_head, F.H_GRID, both.mean(0))))
    P_pip = P_pip_only + P_both
    share = P_pip / P_f * 100
    return P_f, share, float(np.median(keff_post))


seeds = [2026] + list(range(1, 31))
seeds = list(dict.fromkeys(seeds))

# True-field basal stats across a big ensemble (cheap, no Gibbs)
big = [true_field_stats(s) for s in range(200)]
bz = np.array([b[0] for b in big])
gm = np.array([b[1] for b in big])
z2026 = true_field_stats(2026)
print("=== TRUE-FIELD ENSEMBLE (200 seeds, no Gibbs) ===")
print(f"basal-zone mean log10k: {bz.mean():.3f} +/- {bz.std():.3f}")
print(f"global   mean log10k: {gm.mean():.3f} +/- {gm.std():.3f}")
print(f"seed2026 basal {z2026[0]:.3f}  => z = {(z2026[0]-bz.mean())/bz.std():+.2f} sigma "
      f"(percentile {100*np.mean(bz<=z2026[0]):.0f}%)")
print(f"seed2026 global {z2026[1]:.3f} => z = {(z2026[1]-gm.mean())/gm.std():+.2f} sigma")
print()

rows = []
print("=== FULL CHAIN per seed (Gibbs) ===")
for s in seeds:
    pf, share, km = pf_and_piping(s)
    rows.append((s, pf, share, km))
    print(f"seed {s:5d}  P_f {pf:.4f}  RP 1-in-{1/pf:5.0f}  piping {share:4.0f}%  keff_med {km:.2e}")

pfs = np.array([r[1] for r in rows])
shares = np.array([r[2] for r in rows])
pf2026, sh2026 = rows[0][1], rows[0][2]
print("\n--- SUMMARY ---")
print(f"P_f: median {np.median(pfs):.4f} range {pfs.min():.4f}-{pfs.max():.4f}")
print(f"RP : median 1-in-{1/np.median(pfs):.0f}  range 1-in-{1/pfs.max():.0f}..1-in-{1/pfs.min():.0f}")
print(f"piping share: median {np.median(shares):.0f}%  range {shares.min():.0f}-{shares.max():.0f}%")
print(f"seed2026: P_f {pf2026:.4f} (1-in-{1/pf2026:.0f}) percentile {100*np.mean(pfs<=pf2026):.0f}%; "
      f"piping {sh2026:.0f}% percentile {100*np.mean(shares<=sh2026):.0f}%")
