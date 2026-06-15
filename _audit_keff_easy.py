"""Audit experiment: is k_eff aggregate recovery 'easy / near-tautological'?

Compares, across many seeds, the recon's k_eff error vs several baselines:
  A) recon posterior-mean k_eff (the headline)
  B) naive in-zone sounding mean (auditor's stated baseline)
  C) PRIOR-ONLY: just return the global prior mean MEAN_LOG10K (ignores all data)
  D) GLOBAL sounding mean (all 5 soundings, full depth, not just in-zone)
Also measures: spread of true_eff across seeds (is the target moving?), and
the effective number of independent samples implied by the averaging.
"""
import numpy as np
import kfield as K

L_seep = 70.0


def setup(seed):
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
    beta_s, mu_s = sampler.sample(n_samples=800, burn=300)
    recon = mu_s[:, None] + beta_s @ Phi_all.T
    recon_mean = recon.mean(0)

    cx0, cx1 = (K.W - L_seep) / 2, (K.W + L_seep) / 2
    zone = (yy <= 0.4 * K.D) & (xx >= cx0) & (xx <= cx1)
    zf = zone.ravel()

    true_eff_log = field.ravel()[zf].mean()

    # A) recon
    recon_eff_log = recon_mean[zf].mean()

    # B) naive in-zone sounding mean: among observed columns, the cells that fall in zone
    in_zone_obs = obs_mask.ravel() & zf
    naive_inzone_log = field.ravel()[in_zone_obs].mean()  # use true (noiseless) sounding values
    naive_inzone_log_noisy = (field.ravel() + 0)  # placeholder

    # use the actual noisy soundings for the in-zone naive estimate
    # rebuild obs_val map over full grid
    obs_full = np.full(K.NX * K.NY, np.nan)
    obs_full[obs_mask.ravel()] = obs_val
    naive_inzone_noisy_log = np.nanmean(obs_full[in_zone_obs])

    # C) prior only
    prior_log = K.MEAN_LOG10K

    # D) global sounding mean (all observed cells)
    global_sound_log = np.nanmean(obs_full[obs_mask.ravel()])

    return dict(
        true=true_eff_log, recon=recon_eff_log,
        naive_inzone=naive_inzone_noisy_log,
        prior=prior_log, global_sound=global_sound_log,
        n_inzone_cells=int(zf.sum()), n_inzone_obs=int(in_zone_obs.sum()),
        field_zone_sd=float(field.ravel()[zf].std()),
    )


seeds = list(range(1, 31))
rows = [setup(s) for s in seeds]

import numpy as np
def rmse(key):
    return np.sqrt(np.mean([(r[key] - r["true"]) ** 2 for r in rows]))

trues = np.array([r["true"] for r in rows])
print(f"seeds={len(rows)}  in-zone cells={rows[0]['n_inzone_cells']}  in-zone observed cells={rows[0]['n_inzone_obs']}")
print(f"true_eff_log10k across seeds: mean={trues.mean():.3f} sd={trues.std():.3f} range=[{trues.min():.3f},{trues.max():.3f}]")
print(f"  (target moves by {trues.std():.3f} log10 across seeds; field pointwise SD={K.SD_LOG10K})")
print()
print(f"RMSE of k_eff (log10 units) vs true:")
print(f"  A recon posterior mean   : {rmse('recon'):.4f}")
print(f"  B naive IN-ZONE soundings: {rmse('naive_inzone'):.4f}")
print(f"  C prior-only (ignore data): {rmse('prior'):.4f}")
print(f"  D global sounding mean    : {rmse('global_sound'):.4f}")
print()
# How much does recon beat each baseline?
print(f"recon vs prior-only ratio : {rmse('prior')/rmse('recon'):.2f}x  (is target trivially the prior?)")
print(f"recon vs naive-inzone ratio: {rmse('naive_inzone')/rmse('recon'):.2f}x")
