# Flood-Defence Breach Risk — a Hazard → Vulnerability → EP chain

A self-contained catastrophe-modelling demonstrator for a river levee. It walks the full
chain — **flood hazard → geotechnical breach fragility → annual breach probability and an
exceedance-probability curve** — on *real* UK flood data, and drives the vulnerability layer
with a **Bayesian posterior hydraulic-conductivity field reconstructed from sparse soundings**
(the engine from my MSc dissertation, repurposed from dam seepage to levee piping).

The point is the **intersection**: flood defences fail by *geotechnical* mechanisms
(backward-erosion piping is seepage-driven), so this is where flood hazard and ground-and-water
modelling genuinely meet.

```
 Stage 1 HAZARD            Stage 2 K-FIELD              Stage 3 FRAGILITY          Stage 4 EP
 GEV on real NRFA   →   Bayesian-Lasso recon of   →   Sellmeijer piping +   →   convolve hazard×
 annual-max flow        sparse-sounding log-k         overtopping, MC over      fragility → P_f,
 (MLE/L-mom/Bayes)      → posterior k_eff             k-posterior + lit.        catalogue, EP curve
```

## Headline result

> **Idealised Thames-at-Kingston levee: annual breach probability P_f = 0.008, i.e. a
> 1-in-122-year breach (95% CrI 1-in-49 to 1-in-475 yr), piping-dominated (62% of exposure).**
> Uncertainty propagates *both* the flood-hazard GEV posterior *and* the reconstructed-k
> posterior into that interval.

| Quantity | Value |
|---|---|
| Hazard: 100-yr discharge (Bayesian median) | **678 m³/s** [610, 800] — MLE 666 / L-moments 660 |
| Hazard: GEV shape ξ (three methods) | **−0.048 / −0.056 / −0.063** (near-Gumbel, agreement) |
| Hazard NUTS diagnostics | R-hat 1.000, min ESS 4088, 16/8000 divergences |
| k-field: effective seepage-path k (posterior) | **9.9×10⁻⁵ m/s** [8.7, 11.3]×10⁻⁵ (true 9.4×10⁻⁵) |
| k-field: reconstruction from 5 soundings | RMSE 0.18 log₁₀-units |
| Fragility: half-breach head | **6.26 m**; P(breach \| 100-yr head) = 0.17 |
| **Breach: annual probability P_f** | **0.0082 → 1-in-122 yr** (CrI 1-in-49 … 1-in-475) |
| Breach: dominant mechanism | **piping 1-in-149 yr (62%)** vs overtopping 1-in-242 yr |

![hazard](figures/hazard_returnlevels.png)
![kfield](figures/kfield_reconstruction.png)
![fragility](figures/fragility_curve.png)
![breach EP](figures/breach_ep_curve.png)

## The four stages

**1 — Flood hazard (`hazard_gev.py`).** GEV fit to 142 annual-maximum discharges (Thames at
Kingston, NRFA 39001, 1883–2024) three independent ways — MLE, L-moments (Hosking), and
Bayesian NUTS (numpyro). Agreement across estimators is evidence the tail is data-constrained.
Discharge is mapped to river head at the levee via a transparent idealised rating
`H = max(0, C·Q^0.6 − Z0)` (wide-channel Manning exponent; constants calibrated, not surveyed).
The Bayesian posterior is carried forward as a posterior-predictive catalogue of annual-maximum head.

**2 — k-field (`kfield.py`) — the signature move.** The foundation aquifer's spatially variable
log₁₀(k) field is reconstructed from **5 sparse vertical CPT-like soundings** by Bayesian
compressed sensing: a Gaussian RBF dictionary + a Park & Casella (2008) Bayesian-Lasso Gibbs
sampler (a slim, self-contained re-implementation of my dissertation engine). The output is the
**posterior distribution of the effective seepage-path conductivity k_eff** — soil uncertainty
that flows from data, not a textbook lognormal. (The pointwise field CrI coverage is ~0.79,
slightly over-confident — the same "denser arrays over-confident" behaviour I documented in the
dissertation; the aggregate k_eff is well-calibrated, with the truth inside its 95% interval.)

**3 — Fragility (`fragility.py`).** Sellmeijer (2011) backward-erosion piping critical head
`Hc = L·F_R·F_S·F_G`, with k drawn from the Stage-2 posterior and d70, aquifer thickness,
seepage length and a model-uncertainty factor from lognormal literature distributions; plus an
overtopping mechanism (uncertain crest). Series system (breach if either). One fragility curve
per k-posterior draw, so the curve's spread *is* the propagated sparse-data k uncertainty.

**4 — Breach EP (`breach_ep.py`).** Convolve hazard × fragility:
`P_f = E_H[ P(breach | H) ]`. The 95% credible interval pairs GEV-posterior draws with
k-fragility curves. A 50,000-year synthetic catalogue and a breach exceedance-probability curve
(hazard head-exceedance vs P(breach ∩ head ≥ h)) complete the chain.

## Data-driven vs literature-driven (stated, not hidden)

- **Data-driven:** the flood hazard — real NRFA annual-maximum discharge, longest available record.
- **Literature-driven:** the idealised levee geometry and the soil/geometry parameter
  distributions (Sellmeijer constants; d70, D, L, model factor, crest), from the levee-piping
  reliability literature. The *k* distribution is replaced by my own Bayesian reconstruction.
- **Idealised:** a single levee cross-section. This is a **methodological demonstrator**, not a
  real-asset assessment of any specific Thames levee.

## Reproduce

```bash
uv sync
uv run python download.py      # NRFA AMAX (station 39001; pass another id to change)
uv run python hazard_gev.py    # Stage 1  -> outputs/hazard.npz, figures/
uv run python kfield.py        # Stage 2  -> outputs/kfield.npz, figures/
uv run python fragility.py     # Stage 3  -> outputs/fragility.npz, figures/
uv run python breach_ep.py     # Stage 4  -> outputs/breach_ep.npz, figures/
```
JAX is pinned to CPU for bit-identical NUTS on any machine; all seeds are fixed.

## Limitations

- Single idealised cross-section; no spatial reach of multiple levee segments, no 2D/3D FE seepage
  (the dissertation's full field solver is reused only to *produce* the k posterior, not re-solved
  per Monte-Carlo draw).
- Stationary flood frequency — no non-stationarity / climate trend in the GEV (the AMAX record
  shows the well-known recent clustering but it is not modelled here).
- The discharge→head rating and the levee geometry are calibrated illustrative values, so the
  *absolute* 1-in-122-yr figure is demonstrative; the **method** (uncertainty-propagating
  hazard→fragility→EP with a data-reconstructed k) is the deliverable.
- Piping and overtopping only; slope instability and micro-instability are out of scope.

## Development note — AI-paired build

This repository is the output of an **AI-paired build loop** (Claude Code as implementation
collaborator) under my direction.

- **Mine:** the project framing (flood × geotechnical intersection; full hazard→vulnerability→EP
  chain), the methodology (three-estimator GEV triangulation; reusing my dissertation's Bayesian
  compressed-sensing engine to supply a *data-driven* k for the fragility; Sellmeijer as the
  limit state; series-system fragility; joint hazard+k uncertainty propagation), the levee/soil
  parameterisation and calibration targets, every accept/reject decision, and the disclosure
  standard (what goes in *Limitations* and the data/literature split).
- **AI-assisted:** the Python implementation, the slim re-implementation of the Bayesian-Lasso
  sampler and RBF dictionary from my dissertation library, draft README/figure code, and
  numerical-stability patterns (Cholesky jitter, JAX Gumbel-limit branch).
- **Cross-checks I require:** every numeric claim here reconciles to a code path; the Sellmeijer
  formulation and constants are checked against the source literature; GEV diagnostics (R-hat,
  ESS, divergences) and the reconstruction's calibration (coverage, RMSE) are surfaced, not hidden.

This is the workflow I would bring to a catastrophe-modelling R&D team: AI as an accelerator on
the implementation axis, candidate judgment on methodology, scope, verification and disclosure.

## References

- Sellmeijer, J.B. et al. (2011). *Fine-tuning of the backward erosion piping model through
  small-scale, medium-scale and IJkdijk experiments.* European J. Environmental & Civil Eng.
- Schweckendiek, T. (2014). *On reducing piping uncertainties — a Bayesian decision approach.* TU Delft.
- D'Oria, M., Mignosa, P., Tanda, M.G. (2019). *Probabilistic assessment of flood hazard due to
  levee breaches using fragility functions.* Water Resources Research 55.
- Park, T. & Casella, G. (2008). *The Bayesian Lasso.* JASA 103(482).
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values.* Springer.
- NRFA Peak Flow Dataset (v12), UK Centre for Ecology & Hydrology — https://nrfa.ceh.ac.uk
