"""Stage 3 - VULNERABILITY: a levee breach fragility curve.

P(breach | river head H) for an idealised levee, combining two mechanisms as a series
system (breach if either occurs):

  * Backward-erosion piping  - Sellmeijer (2011) critical-head rule. The driving random
    variable is the foundation hydraulic conductivity k, supplied as the Bayesian POSTERIOR
    reconstructed in Stage 2 (kfield.py), not a generic literature lognormal. Other soil /
    geometry parameters (d70, aquifer thickness D, seepage length L, model factor m_p) use
    lognormal distributions from the levee-piping reliability literature
    (Sellmeijer et al. 2011; Schweckendiek 2014; D'Oria et al. 2019).
  * Overtopping - breach when H exceeds the (uncertain) crest height.

We compute one fragility curve per posterior k-draw, so the spread of curves is the
uncertainty in vulnerability that flows directly from the sparse-data k reconstruction.

Run:  uv run python fragility.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"

# ---- Sellmeijer (2011) constants -------------------------------------------
ETA = 0.25                 # White's drag-force coefficient
THETA_DEG = 37.0           # bedding angle of the sand grains
RHO_S, RHO_W, G = 2650.0, 1000.0, 9.81
GAMMA_SP = (RHO_S - RHO_W) * G   # submerged unit weight of particles (N/m3)
GAMMA_W = RHO_W * G
D70M = 2.08e-4             # reference grain size (m)
NU = 1.33e-6              # kinematic viscosity of water ~10 C (m2/s)

# ---- aleatory parameter distributions (lognormal: median, coefficient of var) ----
D70_MED, D70_COV = 2.0e-4, 0.12     # representative grain size (m)
DAQ_MED, DAQ_COV = 10.0, 0.10       # aquifer thickness (m)
LSEEP_MED, LSEEP_COV = 70.0, 0.10   # seepage length under the levee (m)
MP_MED, MP_COV = 1.0, 0.12          # Sellmeijer model-uncertainty factor
CREST_MED, CREST_COV = 6.5, 0.05    # levee crest height above landside ground (m)

H_GRID = np.linspace(0.0, 8.0, 81)
N_ALEATORY = 600
N_KDRAWS = 500
SEED = 99


def lognormal(rng, median, cov, size):
    sigma = np.sqrt(np.log(1 + cov**2))
    return median * np.exp(sigma * rng.standard_normal(size))


def sellmeijer_critical_head(k, d70, D, L, mp):
    """Sellmeijer 2011 critical head Hc = L * F_R * F_S * F_G  (all inputs arrays)."""
    F_R = ETA * (GAMMA_SP / GAMMA_W) * np.tan(np.deg2rad(THETA_DEG))
    kappa = k * NU / G                       # intrinsic permeability (m2)
    F_S = (D70M / np.cbrt(kappa * L)) * (d70 / D70M) ** 0.4
    DL = D / L
    F_G = 0.91 * DL ** (0.28 / (DL**2.8 - 1.0) + 0.04)
    return mp * L * F_R * F_S * F_G


def main():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    kf = np.load(OUT_DIR / "kfield.npz")
    keff_post = kf["keff"]
    kdraws = rng.choice(keff_post, size=N_KDRAWS, replace=True)

    # Aleatory draws shared across k-draws (common random numbers -> smooth curves).
    d70 = lognormal(rng, D70_MED, D70_COV, N_ALEATORY)
    D = lognormal(rng, DAQ_MED, DAQ_COV, N_ALEATORY)
    L = lognormal(rng, LSEEP_MED, LSEEP_COV, N_ALEATORY)
    mp = lognormal(rng, MP_MED, MP_COV, N_ALEATORY)
    crest = lognormal(rng, CREST_MED, CREST_COV, N_ALEATORY)

    # Per k-draw fragility curve.
    curves = np.empty((N_KDRAWS, len(H_GRID)))
    pip_curves = np.empty_like(curves)
    ovt_curves = np.empty_like(curves)
    for i, k in enumerate(kdraws):
        Hc = sellmeijer_critical_head(np.full(N_ALEATORY, k), d70, D, L, mp)  # (N_ALEATORY,)
        # breach indicators over H grid: (len_H, N_ALEATORY)
        piping = H_GRID[:, None] >= Hc[None, :]
        overtop = H_GRID[:, None] >= crest[None, :]
        breach = piping | overtop
        curves[i] = breach.mean(1)
        pip_curves[i] = piping.mean(1)
        ovt_curves[i] = overtop.mean(1)

    frag_med = np.median(curves, 0)
    frag_lo = np.quantile(curves, 0.025, 0)
    frag_hi = np.quantile(curves, 0.975, 0)
    pip_med = pip_curves.mean(0)
    ovt_med = ovt_curves.mean(0)

    # Half-breach head (P=0.5) for the median curve.
    h50 = float(np.interp(0.5, frag_med, H_GRID))
    print(f"FRAGILITY — {N_KDRAWS} k-draws x {N_ALEATORY} aleatory draws")
    print(f"  Sellmeijer F_R = {ETA*(GAMMA_SP/GAMMA_W)*np.tan(np.deg2rad(THETA_DEG)):.3f}")
    print(f"  half-breach head P(breach)=0.5 at H = {h50:.2f} m (median curve)")
    for h in (3.0, 4.0, 5.0, 5.74):
        j = int(np.argmin(np.abs(H_GRID - h)))
        print(f"  P(breach | H={h:.2f} m) = {frag_med[j]:.3f}  [{frag_lo[j]:.3f}, {frag_hi[j]:.3f}]")

    np.savez(
        OUT_DIR / "fragility.npz",
        h_grid=H_GRID, curves=curves, frag_med=frag_med, frag_lo=frag_lo, frag_hi=frag_hi,
        pip_med=pip_med, ovt_med=ovt_med, h50=h50,
    )
    plot_fragility(H_GRID, frag_med, frag_lo, frag_hi, pip_med, ovt_med, curves)
    print("  saved -> outputs/fragility.npz, figures/fragility_curve.png")


def plot_fragility(h, med, lo, hi, pip, ovt, curves):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    # thin sample of individual k-draw curves
    for c in curves[:: max(1, len(curves) // 40)]:
        ax.plot(h, c, color="#9DB4C0", lw=0.5, alpha=0.4)
    ax.fill_between(h, lo, hi, color="#5B8FB9", alpha=0.30, label="95% CrI (from reconstructed k)")
    ax.plot(h, med, color="#1F3A5F", lw=2.4, label="fragility — combined")
    ax.plot(h, pip, color="#C44536", lw=1.6, ls="--", label="piping only (Sellmeijer)")
    ax.plot(h, ovt, color="#E08E45", lw=1.6, ls=":", label="overtopping only")
    ax.set_xlabel("River head at levee H (m)")
    ax.set_ylabel("P(breach | H)")
    ax.set_title("Levee breach fragility")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fragility_curve.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
