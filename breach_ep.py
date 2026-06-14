"""Stage 4 - FINANCIAL/EP layer: annual breach probability and the breach EP curve.

Convolves the flood hazard (Stage 1 posterior-predictive annual-maximum head) with the
levee breach fragility (Stage 3):

    P_f = E_{H ~ hazard}[ P(breach | H) ]      (annual breach probability)

The headline number is P_f and its return period 1/P_f, with a 95% credible interval that
propagates BOTH hazard uncertainty (GEV posterior) AND foundation-k uncertainty (the
Stage-2 Bayesian reconstruction, carried as the family of fragility curves). A synthetic
event catalogue and a breach exceedance-probability curve complete the chain.

Run:  uv run python breach_ep.py   (after hazard_gev.py, kfield.py, fragility.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hazard_gev import flow_to_head, gev_quantile

HERE = Path(__file__).parent
OUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"


def curve_eval(h, h_grid, curve):
    return np.interp(h, h_grid, curve)


def main():
    # allow_pickle=False: we read only numeric arrays here (string metadata in the
    # npz is never accessed), so no object unpickling occurs.
    haz = np.load(OUT_DIR / "hazard.npz", allow_pickle=False)
    fr = np.load(OUT_DIR / "fragility.npz")
    h_grid = fr["h_grid"]
    frag_med, pip_med, ovt_med = fr["frag_med"], fr["pip_med"], fr["ovt_med"]
    curves = fr["curves"]                # (n_kdraws, len_h) epistemic-from-k
    pred_head = haz["pred_head"]         # posterior-predictive annual-max head catalogue
    mu, sigma, xi = haz["post_mu"], haz["post_sigma"], haz["post_xi"]

    # ---- point estimate: marginal fragility x posterior-predictive head ----
    P_f = float(np.mean(curve_eval(pred_head, h_grid, frag_med)))
    P_pip = float(np.mean(curve_eval(pred_head, h_grid, pip_med)))
    P_ovt = float(np.mean(curve_eval(pred_head, h_grid, ovt_med)))
    print(f"BREACH EP — convolving hazard x fragility")
    print(f"  annual breach probability  P_f = {P_f:.4f}  ->  return period {1/P_f:,.0f} yr")
    print(f"  by mechanism (annual): piping {P_pip:.4f} (RP {1/P_pip:,.0f} yr) | "
          f"overtopping {P_ovt:.5f} (RP {1/P_ovt:,.0f} yr)")
    print(f"  -> piping dominates: {P_pip/(P_pip+P_ovt)*100:.0f}% of breach exposure")

    # ---- credible interval: pair hazard-posterior draws with k-fragility curves ----
    rng = np.random.default_rng(2026)
    n_pairs, n_years = 1000, 4000
    Pf_pairs = np.empty(n_pairs)
    for p in range(n_pairs):
        j = rng.integers(len(mu))
        c = curves[rng.integers(len(curves))]
        u = rng.uniform(size=n_years)
        flow = gev_quantile(u, mu[j], sigma[j], xi[j])
        flow = flow[np.isfinite(flow) & (flow > 0)]
        head = flow_to_head(flow)
        Pf_pairs[p] = np.mean(curve_eval(head, h_grid, c))
    pf_med = float(np.median(Pf_pairs))
    pf_lo, pf_hi = float(np.quantile(Pf_pairs, 0.025)), float(np.quantile(Pf_pairs, 0.975))
    print(f"  P_f 95% credible interval: {pf_lo:.4f} - {pf_hi:.4f}  "
          f"(return period {1/pf_hi:,.0f} - {1/pf_lo:,.0f} yr; median {1/pf_med:,.0f} yr)")

    # ---- synthetic breach event catalogue ----
    n_cat = 50000
    sim_head = rng.choice(pred_head, size=n_cat, replace=True)
    breach = rng.uniform(size=n_cat) < curve_eval(sim_head, h_grid, frag_med)
    n_breach = int(breach.sum())
    print(f"  synthetic catalogue: {n_breach} breaches in {n_cat:,} simulated years "
          f"(rate {n_breach/n_cat:.4f}); mean head at breach {sim_head[breach].mean():.2f} m")

    # ---- breach exceedance-probability curve ----
    h_levels = np.linspace(0, 8, 161)
    hazard_ep = np.array([np.mean(pred_head >= h) for h in h_levels])          # P(annual max head >= h)
    breach_ep = np.array([np.mean((sim_head >= h) & breach) for h in h_levels])  # annual P(breach AND head >= h)

    np.savez(
        OUT_DIR / "breach_ep.npz",
        P_f=P_f, P_pip=P_pip, P_ovt=P_ovt, pf_med=pf_med, pf_lo=pf_lo, pf_hi=pf_hi,
        h_levels=h_levels, hazard_ep=hazard_ep, breach_ep=breach_ep,
        n_breach=n_breach, n_cat=n_cat, mean_head_breach=float(sim_head[breach].mean()),
    )
    plot_ep(h_levels, hazard_ep, breach_ep, P_f, pf_lo, pf_hi)
    print("  saved -> outputs/breach_ep.npz, figures/breach_ep_curve.png")

    # one-line headline for the README
    print("\nHEADLINE:")
    print(f"  Levee annual breach probability {P_f:.3f} (1-in-{1/P_f:.0f} yr), "
          f"95% CrI 1-in-{1/pf_hi:.0f} to 1-in-{1/pf_lo:.0f} yr; piping-dominated.")


def plot_ep(h, hazard_ep, breach_ep, P_f, pf_lo, pf_hi):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(h, hazard_ep, color="#1F3A5F", lw=2.0, label="flood hazard: P(annual-max head ≥ h)")
    ax.plot(h, breach_ep, color="#C44536", lw=2.0, label="breach EP: P(breach ∩ head ≥ h)")
    ax.fill_between(h, 0, breach_ep, color="#C44536", alpha=0.15)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.1)
    ax.set_xlabel("River head at levee h (m)")
    ax.set_ylabel("Annual exceedance probability")
    ax.set_title("Hazard vs breach exceedance-probability curve")
    ax.axhline(P_f, color="#555", lw=1.0, ls="--")
    ax.text(0.1, P_f * 1.15, f"annual breach prob P_f = {P_f:.3f}  (1-in-{1/P_f:.0f} yr)", fontsize=8, color="#555")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "breach_ep_curve.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
