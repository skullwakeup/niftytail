# NIFTYTAIL — Day-6 N_min study and A3 timing pilot

**Specification:** decision D32 (`docs/decisions.md`); methodology §7 Step 0 and §8; configs `[nmin_study]`.
**Rule result (D32, unchanged):** **RULE FAILURE: no N_u in {50, 75, 100, 150} qualifies** (rule n_min = null), and the grid was not extended.
**Adopted (D33, owner-delegated override):** **N_min = 100, q_max = 0.90.** This is recorded as `[threshold] n_min = 100` with `[nmin_study] override_decision = "D33"`, and in the JSON `adopted` block. The override is a judgement, not a rule outcome (§6).
**Source of every number:** `reports/threshold_selection/nmin_study.json`, written by `uv run --locked python -m niftytail nmin-study configs\base.toml`.
- Day 6B: two runs gave sha256 `86CB7B59…13BBD3`.
- Day 6C: the regeneration with the D33 config and the new descriptive column gave `5A8349A7…64FC97` in two runs, at about 39 s each.
- Every value from the 6B file is unchanged except `config_sha256`, because the config changed. Only the two per-cell keys and the `adopted` block were added.
**Number format:** every non-integer is `format(x, ".4g")`; integers are printed as is.

Synthetic data only. No real series was loaded or fitted.

## 1. Design and definitions (D32)

| Item | Value |
|---|---|
| Cells | true ξ ∈ {0.1, 0.25, 0.4} × N_u ∈ {50, 75, 100, 150} (12 cells), β = 1 |
| Replications | 2000 per cell |
| Seeds | cell k = 4·i_ξ + i_N uses `Generator(PCG64(SeedSequence(20261111).spawn(12)[k]))`; replications are drawn sequentially; execution is serial |
| Excesses | i.i.d. GPD by inversion, y = (β/ξ)·expm1(−ξ·log1p(−U)) |
| Fit | `fit_mle(y, min_excesses=2, …)` (D30/D31); "converged" means no EstimationError |
| Quantity | VaR99 excess quantile x_p = `tail_var(u=0, ξ, β, n=W=1000, n_u=N_u, p=1/100)`, from the true and the fitted (ξ, β) |
| Relative error | (x̂ − x_p)/x_p over converged fits (scale-free, because u cancels) |
| Rule | N_min = smallest N_u such that, at every ξ, convergence ≥ 0.99, RMSE(ξ̂) ≤ 0.15 and relative RMSE(x_p) ≤ 0.15; comparisons are exact and ties pass |

## 2. Per-cell results

All statistics are over converged fits.

| ξ | N_u | R | converged | conv. rate | EstimationError | bias(ξ̂) | RMSE(ξ̂) | rel. RMSE(x_p) | P(ξ̂ ≥ 0.5) | P(ξ̂ ≥ 1) | P(irregular) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 50 | 2000 | 2000 | 1 | 0 | -0.05517 | 0.1898 | 0.1565 | 0.0045 | 0 | 0.0045 |
| 0.1 | 75 | 2000 | 2000 | 1 | 0 | -0.03243 | 0.1394 | 0.1263 | 0.0015 | 0 | 0 |
| 0.1 | 100 | 2000 | 2000 | 1 | 0 | -0.02688 | 0.1278 | 0.1122 | 0 | 0 | 0 |
| 0.1 | 150 | 2000 | 2000 | 1 | 0 | -0.01618 | 0.1003 | 0.1052 | 0 | 0 | 0 |
| 0.25 | 50 | 2000 | 2000 | 1 | 0 | -0.04179 | 0.2002 | 0.1782 | 0.066 | 0 | 0.001 |
| 0.25 | 75 | 2000 | 2000 | 1 | 0 | -0.03298 | 0.1582 | 0.1477 | 0.0345 | 0 | 0 |
| 0.25 | 100 | 2000 | 2000 | 1 | 0 | -0.01986 | 0.1309 | 0.1355 | 0.015 | 0 | 0 |
| 0.25 | 150 | 2000 | 2000 | 1 | 0 | -0.01571 | 0.1053 | 0.1237 | 0.006 | 0 | 0 |
| 0.4 | 50 | 2000 | 2000 | 1 | 0 | -0.04116 | 0.211 | 0.1934 | 0.253 | 0.001 | 0.0005 |
| 0.4 | 75 | 2000 | 2000 | 1 | 0 | -0.02022 | 0.1747 | 0.1727 | 0.247 | 0 | 0 |
| 0.4 | 100 | 2000 | 2000 | 1 | 0 | -0.0277 | 0.1459 | 0.1558 | 0.1925 | 0 | 0 |
| 0.4 | 150 | 2000 | 2000 | 1 | 0 | -0.01465 | 0.1163 | 0.1507 | 0.156 | 0 | 0 |

## 3. Rule table

| N_u | ξ | convergence ≥ 0.99 | RMSE(ξ̂) ≤ 0.15 | rel. RMSE(x_p) ≤ 0.15 | N_u qualifies |
|---|---|---|---|---|---|
| 50 | 0.1 | pass | FAIL | FAIL | no |
| 50 | 0.25 | pass | FAIL | FAIL | no |
| 50 | 0.4 | pass | FAIL | FAIL | no |
| 75 | 0.1 | pass | pass | pass | no |
| 75 | 0.25 | pass | FAIL | pass | no |
| 75 | 0.4 | pass | FAIL | FAIL | no |
| 100 | 0.1 | pass | pass | pass | no |
| 100 | 0.25 | pass | pass | pass | no |
| 100 | 0.4 | pass | pass | FAIL | no |
| 150 | 0.1 | pass | pass | pass | no |
| 150 | 0.25 | pass | pass | pass | no |
| 150 | 0.4 | pass | pass | FAIL | no |

**Outcome:** N_min = null (rule failure). The binding criterion is the relative RMSE of x_p at ξ = 0.4. It is above 0.15 at every N_u (0.1934, 0.1727, 0.1558, 0.1507), and is 0.1507 at the largest grid value, N_u = 150. Convergence was 2000/2000 in every cell. No EstimationError occurred, and at N_u ≥ 75 no fit was irregular.

## 3a. Secondary descriptive metric (D33; never part of the rule)

`rel_rmse_var_level_ref` is the relative RMSE on a VaR *level*. Each error (x̂ − x_p) is divided by (u_ref/β_ref + x_p), with:
- u_ref = Φ⁻¹(1 − N_u/W), a normal reference threshold;
- β_ref = (1 − Φ(u_ref))/φ(u_ref), the Mills ratio (normal GPD scale at u_ref);
- u_ref/β_ref, the reference threshold in GPD-scale units (β = 1 in the study).

It is computed from the same seeded draws and fits; no new random streams are used. Algebraically it equals rel. RMSE(x_p) · x_p/(u_ref/β_ref + x_p).

| ξ | N_u | rel. RMSE(x_p) (rule) | u_ref/β_ref | rel. RMSE VaR level (reference) |
|---|---|---|---|---|
| 0.1 | 50 | 0.1565 | 3.393 | 0.05317 |
| 0.1 | 75 | 0.1263 | 2.717 | 0.05697 |
| 0.1 | 100 | 0.1122 | 2.249 | 0.06004 |
| 0.1 | 150 | 0.1052 | 1.611 | 0.06931 |
| 0.25 | 50 | 0.1782 | 3.393 | 0.06569 |
| 0.25 | 75 | 0.1477 | 2.717 | 0.07249 |
| 0.25 | 100 | 0.1355 | 2.249 | 0.07867 |
| 0.25 | 150 | 0.1237 | 1.611 | 0.08734 |
| 0.4 | 50 | 0.1934 | 3.393 | 0.0773 |
| 0.4 | 75 | 0.1727 | 2.717 | 0.09201 |
| 0.4 | 100 | 0.1558 | 2.249 | 0.09768 |
| 0.4 | 150 | 0.1507 | 1.611 | 0.1133 |

## 3b. Adopted N_min (D33)

| Item | Value |
|---|---|
| Rule result (D32) | failure, n_min = null |
| Adopted | N_min = 100, q_max = 1 − 100/1000 = 9/10 = 0.90 (source "D33") |
| Why N_u = 100 | RMSE(ξ̂) ≤ 0.15 and 2000/2000 converged at every ξ; the only failing criterion (0.1558 at ξ = 0.4) is the conservative excess-quantile one; the VaR-level reference is 0.09768 there; consistent with the plan's N_min ≈ 75–100 expectation |
| Why not 75 | RMSE(ξ̂) fails at ξ = 0.25 (0.1582) and ξ = 0.4 (0.1747) |
| Why not 150 | it restricts q to 0.80–0.85 |

## 4. Limitations

- **The adopted N_min = 100 is an override (D33), a judgement and not a rule outcome.** The pre-declared rule failed.
- **The VaR-level reference is descriptive only.** It depends on a normal reference threshold, which is an assumption and not the data.

- **Asymptotic, idealised design:** the excesses are exact GPD draws above a known threshold. There is no GPD-approximation bias and no threshold-selection effect, so real-data errors are likely larger.
- **Relative-RMSE definition:** the criterion uses the VaR99 *excess* quantile x_p (u = 0). It is a conservative, scale-free reading of "relative RMSE(VaR99)". On a VaR level u + x_p the relative error would be smaller, because u > 0.
- **Coarse grid:** the conclusion holds only for these four N_u values and these three ξ. Per D32, the grid is not extended.
- **Monte Carlo error:** with 2000 replications per cell, a value near a threshold (for example 0.1507 against 0.15) carries Monte Carlo error. The rule uses the point estimates as pre-declared.

## 5. A3 timing pilot (D32 (6); A6 Day 6)

Run: `uv run --locked python scripts\timing_pilot.py configs\base.toml` (synthetic data, exit 0, about 41 s in total).

**Hardware and software:** Windows-11-10.0.26300-SP0; `os.cpu_count()` = 32; W = 31 workers; Python 3.13.17; scipy 1.18.1 (`goodness_of_fit` seed argument: `rng`); numpy 2.5.4.

Median of 3 repeats, single process, seeded GPD(ξ = 0.25, β = 1) excesses:

| N_u | t_gof (B = 199) s | t_gof (B = 999) s | t_fit_mle s |
|---|---|---|---|
| 100 | 0.6143 | 3.291 | 0.00238 |
| 250 | 0.6418 | 3.322 | 0.001896 |
| 400 | 0.6712 | 3.19 | 0.00173 |

| Item | Value |
|---|---|
| S4 reference simulation (10⁷ observations, burn-in 1000, chunked Python recursion), one run | 1.304 s |
| 31 concurrent copies of t_gof (B = 199, N_u = 250): median | 1.451 s |
| Parallel efficiency e = t_single / median t_parallel | 0.4423 |
| S = Σ over the 16 candidates of t_gof199(2000(1 − q_k)), piecewise-linear | 10.28 s per replication |
| T_total(200) = 200·10·S/(W·e) + t_S4 | 0.4167 h |
| T_total(100) | 0.2086 h |
| **Pre-declared rule (A3)** | **T_total(200) ≤ 2 h → R = 200** |
| Day-7 serial estimate of one real selection (16 × mean t_gof999) | 52.29 s |

The pilot does not change `[simstudy]`; R = 200 is already the configured value. The low efficiency e is consistent with 31 processes sharing logical CPUs (SMT); this was not investigated further.
