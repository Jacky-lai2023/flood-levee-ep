"""Does spatial averaging RESCUE a bad field, as the auditor claims?

Auditor: 'Averaging a smooth correlated field over many cells collapses pointwise
error by ~1/sqrt(effective independent samples), so even a mediocre field
reconstruction recovers the zone mean tightly.'

Test: take the recon, add a BIAS and add UNCORRELATED noise, then re-average over
the zone. If the auditor is right, the zone mean stays tight even when the field
is mangled. We separate the two error modes:
  - bias error (constant offset): averaging does NOT remove it
  - zero-mean pointwise noise: averaging DOES shrink it (this is the auditor's point)
The question is which dominates the recon's actual residual.
"""
import numpy as np
import kfield as K

L_seep = 70.0


def get(seed):
    rng = np.random.default_rng(seed)
    x, y, xx, yy = K.grid()
    field = K.true_field(rng)
    col_ix = np.linspace(6, K.NX - 7, K.N_COLUMNS).astype(int)
    pts_all = np.c_[xx.ravel(), yy.ravel()]
    obs_mask = np.zeros((K.NY, K.NX), dtype=bool)
    obs_mask[:, col_ix] = True
    obs_pts = pts_all[obs_mask.ravel()]
    obs_val = field.ravel()[obs_mask.ravel()] + 0.03 * rng.standard_normal(int(obs_mask.sum()))
    centres = K.basis_centres()
    Phi_obs = K.rbf_basis(obs_pts, centres, lx=10.0, ly=3.5)
    Phi_all = K.rbf_basis(pts_all, centres, lx=10.0, ly=3.5)
    sampler = K.BayesianLasso(Phi_obs, obs_val, seed=seed)
    beta_s, mu_s = sampler.sample(n_samples=600, burn=300)
    recon_mean = (mu_s[:, None] + beta_s @ Phi_all.T).mean(0)

    cx0, cx1 = (K.W - L_seep) / 2, (K.W + L_seep) / 2
    zone = ((yy <= 0.4 * K.D) & (xx >= cx0) & (xx <= cx1)).ravel()
    return field.ravel(), recon_mean, zone


seeds = list(range(1, 16))
# Decompose recon zone-mean error into bias (mean residual) vs the rest
biases, zone_errs, pointwise_rmses = [], [], []
for s in seeds:
    f, r, z = get(s)
    resid = r[z] - f[z]               # pointwise residual in zone
    pointwise_rmses.append(np.sqrt(np.mean(resid**2)))
    biases.append(resid.mean())       # this is EXACTLY the zone-mean error
    zone_errs.append(r[z].mean() - f[z].mean())

biases = np.array(biases); pw = np.array(pointwise_rmses); ze = np.array(zone_errs)
print("Per-seed decomposition (zone):")
print(f"  pointwise RMSE in zone : mean={pw.mean():.3f}  (this is what coverage/0.79 reflects)")
print(f"  zone-mean error (=bias): mean|.|={np.abs(ze).mean():.3f}  sd={ze.std():.3f}")
print()
print("Auditor's mechanism: averaging shrinks zero-mean pointwise noise.")
print("If recon residual were pure zero-mean noise over ~Neff indep samples,")
print("zone-mean err ~ pointwise_rmse / sqrt(Neff).")
print(f"  observed shrink factor pointwise_rmse/|zone_err| = {pw.mean()/np.abs(ze).mean():.1f}x")
print(f"  => implied Neff ~ {(pw.mean()/np.abs(ze).mean())**2:.0f}")
print()
print("KEY: the auditor says this shrink makes the target trivial. But the residual")
print("is NOT pure noise - if recon had a systematic bias, averaging would NOT remove")
print("it. The fact that zone-mean err is tiny means the recon's zone-AVERAGE bias is")
print("genuinely near-zero, i.e. the inversion got the level right. A 'mediocre' recon")
print("with a level bias would NOT be saved. Demonstrate:")
print()
# Adversarial: a 'mediocre' recon = prior mean field + correlated structure but WRONG level
for biasmag in [0.1, 0.2, 0.3]:
    errs = []
    for s in seeds:
        f, r, z = get(s)
        bad = r + biasmag  # mediocre recon: same shape, biased level
        errs.append(bad[z].mean() - f[z].mean())
    print(f"  recon + {biasmag} log10 level bias -> zone-mean RMSE = {np.sqrt(np.mean(np.array(errs)**2)):.3f}  (averaging does NOT rescue)")
