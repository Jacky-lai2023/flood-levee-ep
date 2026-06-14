"""Stage 1 - Flood HAZARD.

Fit a GEV to the real NRFA annual-maximum discharge series three independent ways
(MLE, L-moments, Bayesian NUTS) and triangulate the return levels, then map discharge
to the river head driving the levee via a transparent idealised rating curve.

Outputs (outputs/hazard.npz):
    method point estimates + Bayesian posterior of (mu, sigma, xi);
    return-level table (flow) per method with credible intervals;
    a posterior-predictive catalogue of annual-maximum *head* at the levee, which
    Stage 4 (breach_ep) convolves with the fragility curve.

Why three methods? Agreement across MLE / L-moments / Bayesian is evidence the tail
is data-constrained, not an artefact of one estimator. The Bayesian posterior is what
we propagate downstream so hazard uncertainty reaches the final breach probability.

Run:  uv run python hazard_gev.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # bit-identical NUTS on any machine

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"
STATION = "39001"

RETURN_PERIODS = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)

# ----------------------------------------------------------------------------
# Idealised discharge -> river-head rating curve.
#
# H = max(0, C * Q**EXP - Z0). The EXP = 0.6 exponent is the wide-rectangular-
# channel Manning normal-depth relation (h ~ Q^(3/5)); C and Z0 are calibrated so
# the median annual flood gives ~2 m of head at the levee and a ~100-yr flood ~5.5 m
# -- the band over which the Sellmeijer piping fragility is sensitive. These two
# constants are an explicit modelling choice for an idealised cross-section, NOT a
# surveyed rating for the real Kingston reach.
# ----------------------------------------------------------------------------
RATING_C = 0.195
RATING_EXP = 0.6
RATING_Z0 = 4.0


def flow_to_head(q: np.ndarray) -> np.ndarray:
    """Map discharge (m3/s) to river head above landside ground (m), floored at 0."""
    return np.maximum(0.0, RATING_C * np.asarray(q, float) ** RATING_EXP - RATING_Z0)


# ----------------------------------------------------------------------------
# GEV machinery (EVT convention: xi>0 Frechet/heavy tail, xi<0 Weibull/bounded).
# ----------------------------------------------------------------------------
def gev_quantile(p, mu, sigma, xi):
    """Return level at non-exceedance probability p. Gumbel branch for |xi|<1e-6."""
    p = np.asarray(p, float)
    safe_xi = np.where(np.abs(xi) > 1e-6, xi, 1e-6)
    gev = mu + sigma * ((-np.log(p)) ** (-safe_xi) - 1.0) / safe_xi
    gumbel = mu - sigma * np.log(-np.log(p))
    return np.where(np.abs(xi) > 1e-6, gev, gumbel)


def return_level(T, mu, sigma, xi):
    return gev_quantile(1.0 - 1.0 / np.asarray(T, float), mu, sigma, xi)


def fit_mle(x: np.ndarray):
    """scipy genextreme uses shape c = -xi_EVT."""
    c, loc, scale = stats.genextreme.fit(x)
    return {"mu": loc, "sigma": scale, "xi": -c}


def fit_lmoments(x: np.ndarray):
    """Hosking & Wallis GEV L-moment estimators via probability-weighted moments."""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    i = np.arange(1, n + 1)
    b0 = xs.mean()
    b1 = np.sum((i - 1) / (n - 1) * xs) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * xs) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    t3 = l3 / l2
    c = 2.0 / (3.0 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * c + 2.9554 * c**2  # Hosking shape (xi_EVT = -k)
    from scipy.special import gamma

    g = gamma(1 + k)
    alpha = l2 * k / ((1 - 2 ** (-k)) * g)
    xi_loc = l1 - alpha * (1 - g) / k
    return {"mu": xi_loc, "sigma": alpha, "xi": -k}


def gev_log_prob(y, mu, sigma, xi):
    z = (y - mu) / sigma
    arg = 1 + xi * z
    valid = arg > 1e-9
    safe_arg = jnp.where(valid, arg, 1.0)
    safe_xi = jnp.where(jnp.abs(xi) > 1e-6, xi, 1e-6)
    gev_ld = -jnp.log(sigma) - (1.0 + 1.0 / safe_xi) * jnp.log(safe_arg) - safe_arg ** (-1.0 / safe_xi)
    gumbel_ld = -jnp.log(sigma) - z - jnp.exp(-z)
    ld = jnp.where(jnp.abs(xi) > 1e-6, gev_ld, gumbel_ld)
    return jnp.where(valid, ld, -jnp.inf)


def make_model(sample_mean, sample_sd):
    def gev_model(y):
        mu = numpyro.sample("mu", dist.Normal(sample_mean, 2 * sample_sd))
        sigma = numpyro.sample("sigma", dist.HalfNormal(2 * sample_sd))
        # River-flood tails are typically near-Gumbel to mildly heavy; weakly
        # informative prior centred at 0 lets the 142-year record drive xi.
        xi = numpyro.sample("xi", dist.Normal(0.0, 0.2))
        numpyro.factor("gev_lik", gev_log_prob(y, mu, sigma, xi).sum())

    return gev_model


def run_nuts(x, seed=20260614):
    mcmc = MCMC(
        NUTS(make_model(x.mean(), x.std(ddof=1)), target_accept_prob=0.99),
        num_warmup=1000, num_samples=2000, num_chains=4,
        progress_bar=False, chain_method="sequential",
    )
    mcmc.run(jax.random.PRNGKey(seed), y=jnp.asarray(x, dtype=jnp.float32),
             extra_fields=("diverging",))
    return mcmc


def summarise(a, alpha=0.05):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)), float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    rec = json.loads((DATA_DIR / f"amax_{STATION}.json").read_text())
    x = np.asarray(rec["flow"], float)
    name = rec["station_name"]
    print(f"HAZARD — {name} (NRFA {STATION}), n={len(x)} AMAX, {min(rec['years'])}-{max(rec['years'])}")

    mle = fit_mle(x)
    lmo = fit_lmoments(x)
    print(f"  MLE       mu={mle['mu']:.1f} sigma={mle['sigma']:.1f} xi={mle['xi']:+.3f}")
    print(f"  L-moments mu={lmo['mu']:.1f} sigma={lmo['sigma']:.1f} xi={lmo['xi']:+.3f}")

    mcmc = run_nuts(x)
    post = mcmc.get_samples()
    post = {k: np.asarray(v) for k, v in post.items()}
    n_div = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    # R-hat via numpyro print_summary-equivalent
    import numpyro.diagnostics as diag
    grouped = {k: np.asarray(mcmc.get_samples(group_by_chain=True)[k]) for k in ["mu", "sigma", "xi"]}
    rhat = {k: float(diag.gelman_rubin(v)) for k, v in grouped.items()}
    ess = {k: float(diag.effective_sample_size(v)) for k, v in grouped.items()}
    pm = {k: float(np.median(post[k])) for k in ["mu", "sigma", "xi"]}
    print(f"  Bayesian  mu={pm['mu']:.1f} sigma={pm['sigma']:.1f} xi={pm['xi']:+.3f} "
          f"| R-hat {max(rhat.values()):.3f} | min ESS {min(ess.values()):.0f} | {n_div} divergences")

    # Return-level table (flow) per method
    rl_table = {}
    for label, prm in [("mle", mle), ("lmoments", lmo)]:
        rl_table[label] = return_level(RETURN_PERIODS, prm["mu"], prm["sigma"], prm["xi"])
    bayes_rl = np.array([return_level(T, post["mu"], post["sigma"], post["xi"]) for T in RETURN_PERIODS])
    rl_table["bayes_median"] = np.median(bayes_rl, axis=1)
    rl_table["bayes_lo"] = np.quantile(bayes_rl, 0.025, axis=1)
    rl_table["bayes_hi"] = np.quantile(bayes_rl, 0.975, axis=1)

    j100 = int(np.where(RETURN_PERIODS == 100)[0][0])
    rl100_med, rl100_lo, rl100_hi = bayes_rl[j100].mean(), rl_table["bayes_lo"][j100], rl_table["bayes_hi"][j100]
    print(f"  100-yr return level (flow):  MLE {rl_table['mle'][j100]:.0f}  "
          f"L-mom {rl_table['lmoments'][j100]:.0f}  "
          f"Bayes {rl_table['bayes_median'][j100]:.0f} m3/s "
          f"[{rl100_lo:.0f}, {rl100_hi:.0f}]")
    print(f"  -> as head at levee: 100-yr {flow_to_head(rl_table['bayes_median'][j100]):.2f} m; "
          f"median-AMAX {flow_to_head(np.median(x)):.2f} m")

    # Posterior-predictive catalogue of annual-maximum HEAD (for Stage 4).
    rng = np.random.default_rng(7)
    n_pred = 60000
    idx = rng.integers(0, len(post["mu"]), n_pred)
    u = rng.uniform(size=n_pred)
    pred_flow = gev_quantile(u, post["mu"][idx], post["sigma"][idx], post["xi"][idx])
    pred_flow = pred_flow[np.isfinite(pred_flow) & (pred_flow > 0)]
    pred_head = flow_to_head(pred_flow)

    np.savez(
        OUT_DIR / "hazard.npz",
        station=STATION, station_name=name, flow=x, years=np.asarray(rec["years"]),
        post_mu=post["mu"], post_sigma=post["sigma"], post_xi=post["xi"],
        return_periods=RETURN_PERIODS,
        rl_mle=rl_table["mle"], rl_lmoments=rl_table["lmoments"],
        rl_bayes_median=rl_table["bayes_median"], rl_bayes_lo=rl_table["bayes_lo"], rl_bayes_hi=rl_table["bayes_hi"],
        rhat_max=max(rhat.values()), ess_min=min(ess.values()), n_div=n_div,
        rating_c=RATING_C, rating_exp=RATING_EXP, rating_z0=RATING_Z0,
        pred_head=pred_head, pred_flow=pred_flow,
    )
    plot_returnlevels(x, rl_table, name)
    print(f"  saved -> outputs/hazard.npz, figures/hazard_returnlevels.png")


def plot_returnlevels(x, rl, name):
    import matplotlib.pyplot as plt

    xs = np.sort(x)
    n = len(xs)
    # Gringorten plotting positions -> empirical return period
    pp = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    emp_T = 1.0 / (1.0 - pp)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(emp_T, xs, s=18, color="#333", alpha=0.7, zorder=3, label=f"observed AMAX (n={n})")
    ax.plot(RETURN_PERIODS, rl["mle"], color="#C44536", lw=1.6, label="GEV — MLE")
    ax.plot(RETURN_PERIODS, rl["lmoments"], color="#E08E45", lw=1.6, ls="--", label="GEV — L-moments")
    ax.plot(RETURN_PERIODS, rl["bayes_median"], color="#1F3A5F", lw=2.0, label="GEV — Bayesian median")
    ax.fill_between(RETURN_PERIODS, rl["bayes_lo"], rl["bayes_hi"], color="#5B8FB9", alpha=0.25, label="Bayesian 95% CrI")
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Annual-maximum discharge (m$^3$/s)")
    ax.set_title(f"Flood hazard — {name} (NRFA 39001)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hazard_returnlevels.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
