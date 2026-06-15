"""Stage 2 - the signature move: a Bayesian posterior hydraulic-conductivity field.

Reconstructs the spatially variable log10(k) field of the levee foundation aquifer from
a handful of sparse vertical CPT-like soundings, using Bayesian compressed sensing:
a Gaussian radial-basis-function dictionary + a Park & Casella (2008) Bayesian-Lasso
Gibbs sampler. This is a slim, self-contained re-implementation of the engine built for
the author's MSc dissertation ("Reconstructing Hydraulic Conductivity in Landslide Dams
from Sparse CPT - Bayesian Compressed Sensing"), here repurposed to feed a flood-defence
piping fragility instead of a dam seepage model.

The deliverable for Stage 3 is the *posterior distribution of the effective seepage-path
hydraulic conductivity* k_eff (geometric mean of k along the piping path) - i.e. soil
uncertainty that flows from sparse data, not a generic textbook lognormal.

Run:  uv run python kfield.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"

# Aquifer cross-section (metres). Horizontal sandy foundation layer beneath the levee;
# piping develops along its top, over a seepage length L under the levee base.
W = 80.0          # horizontal extent
D = 10.0          # aquifer thickness
NX, NY = 80, 24   # grid resolution
MEAN_LOG10K = -4.3   # ~5e-5 m/s, fine sand (piping-susceptible, defended-standard levee)
SD_LOG10K = 0.5      # spatial variability -> k spans ~ 3e-5 .. 3e-4 m/s
LEN_X, LEN_Y = 26.0, 3.5  # anisotropic correlation: long horizontal, short vertical (layering)
NUG_FRAC = 0.10
N_COLUMNS = 5     # number of CPT-like soundings
# SEED fixes the synthetic "true" foundation field. The headline is conditional on this one
# realization; override via KFIELD_SEED to sweep realizations (see robustness.py / README).
SEED = int(os.environ.get("KFIELD_SEED", "2026"))


# ----------------------------------------------------------------------------
# Ground-truth field (anisotropic Gaussian random field via Cholesky).
# ----------------------------------------------------------------------------
def grid():
    x = np.linspace(0, W, NX)
    y = np.linspace(0, D, NY)
    xx, yy = np.meshgrid(x, y)  # (NY, NX)
    return x, y, xx, yy


def true_field(rng):
    x, y, xx, yy = grid()
    pts = np.c_[xx.ravel(), yy.ravel()]
    dx = pts[:, 0][:, None] - pts[:, 0][None, :]
    dy = pts[:, 1][:, None] - pts[:, 1][None, :]
    corr = np.exp(-0.5 * ((dx / LEN_X) ** 2 + (dy / LEN_Y) ** 2))
    cov = SD_LOG10K**2 * ((1 - NUG_FRAC) * corr + NUG_FRAC * np.eye(len(pts)))
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(pts)))
    field = MEAN_LOG10K + L @ rng.standard_normal(len(pts))
    return field.reshape(NY, NX)


# ----------------------------------------------------------------------------
# Gaussian RBF sparse dictionary (after the dissertation's SparseBasis).
# ----------------------------------------------------------------------------
def rbf_basis(pts, centres, lx, ly):
    dx = pts[:, 0][:, None] - centres[:, 0][None, :]
    dy = pts[:, 1][:, None] - centres[:, 1][None, :]
    return np.exp(-0.5 * ((dx / lx) ** 2 + (dy / ly) ** 2))


def basis_centres(spacing_x=8.0, spacing_y=3.0):
    cx = np.arange(0, W + 1e-9, spacing_x)
    cy = np.arange(0, D + 1e-9, spacing_y)
    cxx, cyy = np.meshgrid(cx, cy)
    return np.c_[cxx.ravel(), cyy.ravel()]


# ----------------------------------------------------------------------------
# Bayesian-Lasso Gibbs sampler (Park & Casella 2008), faithful slim version.
# Conditionals:
#   beta | .  ~ N(A^-1 X^T (y-mu), sigma^2 A^-1),  A = X^T X + diag(psi)
#   psi_j     ~ InverseGaussian(mean=sqrt(lam2 sigma2)/|beta_j|, shape=lam2)
#   lam2      ~ Gamma(p + r, 0.5*sum(1/psi) + delta)
#   sigma2    ~ InvGamma(a0 + (n+p)/2, b0 + 0.5*(||y-mu-Xbeta||^2 + sum beta^2 psi))
#   mu        ~ N(mean(y - X beta), sigma2/n)
# ----------------------------------------------------------------------------
class BayesianLasso:
    def __init__(self, X, y, a0=1e-2, b0=1e-2, r=1.0, delta=10.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.n, self.p = self.X.shape
        self.ymean = self.y.mean()
        self.yc = self.y - self.ymean
        self.XtX = self.X.T @ self.X
        self.Xty = self.X.T @ self.yc
        self.ones = self.X.sum(0)
        self.a0, self.b0, self.r, self.delta = a0, b0, r, delta

    def _chol(self, A):
        for j in range(6):
            try:
                return np.linalg.cholesky(A + (0 if j == 0 else 10.0 ** (j - 9)) * np.eye(self.p))
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError("Cholesky failed")

    def sample(self, n_samples=1500, burn=500, thin=1):
        beta = np.zeros(self.p)
        psi = np.ones(self.p)
        sigma2, mu, lam2 = 1.0, 0.0, 1.0
        keep_beta, keep_mu = [], []
        total = burn + n_samples
        for it in range(total):
            # mu (intercept on top of centred y)
            mu = self.rng.normal((self.yc - self.X @ beta).mean(), np.sqrt(sigma2 / self.n))
            # beta
            A = self.XtX.copy()
            A[np.diag_indices(self.p)] += psi
            L = self._chol(A)
            rhs = self.Xty - mu * self.ones
            mean = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            beta = mean + np.sqrt(sigma2) * np.linalg.solve(L.T, self.rng.standard_normal(self.p))
            # psi (inverse Gaussian)
            absb = np.maximum(np.abs(beta), 1e-12)
            mu_ig = np.minimum(np.sqrt(lam2 * sigma2) / absb, 1e12)
            psi = self.rng.wald(mu_ig, lam2)
            psi = np.where((psi > 0) & np.isfinite(psi), psi, 1e-8)
            # lambda^2
            lam2 = self.rng.gamma(self.p + self.r, 1.0 / (0.5 * np.sum(1.0 / psi) + self.delta))
            # sigma^2
            resid = self.yc - mu - self.X @ beta
            shape = self.a0 + (self.n + self.p) / 2
            scale = self.b0 + 0.5 * (resid @ resid + np.sum(beta**2 * psi))
            sigma2 = 1.0 / self.rng.gamma(shape, 1.0 / scale)
            if it >= burn and (it - burn) % thin == 0:
                keep_beta.append(beta.copy())
                keep_mu.append(mu)
        return np.array(keep_beta), np.array(keep_mu) + self.ymean


def main():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    x, y, xx, yy = grid()
    field = true_field(rng)

    # Sparse CPT-like vertical soundings at N_COLUMNS evenly spaced x-locations.
    col_ix = np.linspace(6, NX - 7, N_COLUMNS).astype(int)
    pts_all = np.c_[xx.ravel(), yy.ravel()]
    obs_mask = np.zeros((NY, NX), dtype=bool)
    obs_mask[:, col_ix] = True
    obs_pts = pts_all[obs_mask.ravel()]
    obs_val = field.ravel()[obs_mask.ravel()]
    # small measurement noise on the soundings
    obs_val = obs_val + 0.03 * rng.standard_normal(len(obs_val))

    centres = basis_centres()
    Phi_obs = rbf_basis(obs_pts, centres, lx=10.0, ly=3.5)
    Phi_all = rbf_basis(pts_all, centres, lx=10.0, ly=3.5)

    print(f"K-FIELD — grid {NX}x{NY}, {len(centres)} RBF centres, "
          f"{N_COLUMNS} soundings -> {len(obs_val)} observations")
    sampler = BayesianLasso(Phi_obs, obs_val, seed=SEED)
    beta_s, mu_s = sampler.sample(n_samples=1500, burn=500)
    print(f"  Gibbs: kept {len(beta_s)} draws (p={beta_s.shape[1]})")

    # Posterior field draws: mu + Phi_all @ beta  (shape: n_draws x n_cells)
    recon = mu_s[:, None] + beta_s @ Phi_all.T
    recon_mean = recon.mean(0).reshape(NY, NX)
    recon_std = recon.std(0).reshape(NY, NX)

    # Reconstruction fidelity vs ground truth
    rmse = float(np.sqrt(np.mean((recon_mean - field) ** 2)))
    # 95% credible-interval coverage of the true field
    lo = np.quantile(recon, 0.025, 0).reshape(NY, NX)
    hi = np.quantile(recon, 0.975, 0).reshape(NY, NX)
    coverage = float(np.mean((field >= lo) & (field <= hi)))
    print(f"  reconstruction RMSE {rmse:.3f} log10-units | 95% CrI coverage {coverage:.3f}")

    # Effective seepage-path k for the Sellmeijer piping path. Backward-erosion piping is
    # driven by ~1D Darcy flow ALONG a near-horizontal path under the levee, so the correct
    # bulk reduction is directional series/parallel, NOT a geometric mean (which is the
    # heuristic for areal 2D flow): arithmetic-mean k over depth within each x-column
    # (transverse / parallel to the path), then harmonic-mean those columns along x (series
    # along the flow direction). NB this is still a BULK reduction — piping *initiates* at the
    # most-permeable connected path (a high quantile of the field), so the scalar k_eff is a
    # deliberate simplification of this limit state (see README limitations).
    L_seep = 70.0
    cx0, cx1 = (W - L_seep) / 2, (W + L_seep) / 2
    row_mask = y <= 0.4 * D
    col_mask = (x >= cx0) & (x <= cx1)

    def series_parallel_keff(k_grid):
        # k_grid: (..., NY, NX) hydraulic conductivity. Returns (...,) effective k.
        sub = k_grid[..., row_mask, :][..., :, col_mask]      # (..., n_rows, n_cols)
        k_col = sub.mean(axis=-2)                              # depth arithmetic mean -> (..., n_cols)
        return k_col.shape[-1] / (1.0 / k_col).sum(axis=-1)    # harmonic mean along flow

    keff = series_parallel_keff(10.0 ** recon.reshape(-1, NY, NX))   # (n_draws,)
    eff_log10k = np.log10(keff)
    true_eff = float(series_parallel_keff(10.0 ** field))
    em, elo, ehi = np.median(keff), np.quantile(keff, 0.025), np.quantile(keff, 0.975)
    print(f"  effective seepage-path k (series/parallel): posterior median {em:.2e} m/s "
          f"[{elo:.2e}, {ehi:.2e}] (true {true_eff:.2e})")

    np.savez(
        OUT_DIR / "kfield.npz",
        keff=keff, eff_log10k=eff_log10k, true_eff=true_eff,
        field=field, recon_mean=recon_mean, recon_std=recon_std,
        x=x, y=y, col_x=x[col_ix], rmse=rmse, coverage=coverage,
        L_seep=L_seep, mean_log10k=MEAN_LOG10K,
    )
    plot_fields(x, y, field, recon_mean, recon_std, x[col_ix], keff, true_eff)
    print("  saved -> outputs/kfield.npz, figures/kfield_reconstruction.png, figures/kfield_keff.png")


def plot_fields(x, y, field, recon_mean, recon_std, col_x, keff, true_eff):
    import matplotlib.pyplot as plt

    ext = [x.min(), x.max(), y.min(), y.max()]
    vmin, vmax = field.min(), field.max()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.2))
    for a, data, title in zip(
        ax, [field, recon_mean, recon_std],
        ["True log$_{10}$k field", "Posterior mean (from 5 soundings)", "Posterior std"],
    ):
        vm = (vmin, vmax) if "std" not in title.lower() else (0, recon_std.max())
        cmap = "viridis" if "std" not in title.lower() else "magma"
        im = a.imshow(data, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=vm[0], vmax=vm[1])
        for cx in col_x:
            a.axvline(cx, color="white", lw=0.8, ls=":", alpha=0.8)
        a.set_title(title, fontsize=10)
        a.set_xlabel("x (m)")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    ax[0].set_ylabel("depth (m)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kfield_reconstruction.png", dpi=140)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(6.4, 4.0))
    a.hist(keff, bins=50, density=True, color="#5B8FB9", edgecolor="white", alpha=0.85)
    a.axvline(np.median(keff), color="#1F3A5F", lw=2, label=f"posterior median {np.median(keff):.2e}")
    a.axvline(true_eff, color="#C44536", lw=2, ls="--", label=f"true {true_eff:.2e}")
    a.set_xlabel("Effective seepage-path hydraulic conductivity k (m/s)")
    a.set_ylabel("posterior density")
    a.set_title("Reconstructed k driving the piping fragility")
    a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kfield_keff.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
