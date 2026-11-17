# NIFTYTAIL — Methodology Specification

**Status:** Day 1 complete (closeout 2026-11-11). Corrections approved by the owner.

**Source of truth:** [`docs/plan_v2.2.md`](plan_v2.2.md). The V2.2 amendment A0–A6 supersedes conflicting V2.1 text. Decisions D1–D20 are listed in [`docs/decisions.md`](decisions.md).

**Reference values:** every number marked *(R#)* is reproduced by [`docs/reference_values.py`](reference_values.py), which uses only the standard library:

```
py -3.13 -I docs\reference_values.py
```

Its output is in Appendix A.

**Literature tags:** citations such as [L08, F] refer to [`docs/literature.md`](literature.md) and carry an evidence status:
- **M**: metadata verified.
- **C1**: claim checked against the full text, read directly.
- **C1-sections**: the specific statements cited were checked directly against the relevant sections of the original text; the paper was not read in full.
- **C2**: claim checked against an abstract or a summary of the full text.
- **F**: full text still needs reading.
- **U**: unverified.

An F tag means the claim is *attributed but not yet confirmed*.

**Discrepancies:** places where this document found the plan inconsistent are marked **⚠ DX-n**. They are listed in §17 and are *not* resolved silently.

---

## 1. Notation (A0)

| Symbol | Meaning |
|---|---|
| c | confidence level, c ∈ {0.95, 0.99} primary and {0.995, 0.999} descriptive |
| p = 1 − c | upper-tail loss probability |
| r_t | log return on trading day t |
| L_t = −r_t | loss, positive meaning a loss |
| W | rolling estimation window length (1000) |
| T0 | training length (2000) |
| VaR_c, ES_c | loss-space risk measures, positive numbers |
| v, e | return-space VaR and ES used **only** in FZ0 (§11): v = −VaR_c, e = −ES_c |

There are no bare α subscripts on VaR or ES. In FZ0, α ≡ p.

**Exact arithmetic rule.** Confidence levels are read from configuration *strings* and converted with `fractions.Fraction`. Tail counts m = W·p and order-statistic indices ⌈W·c⌉ are computed exactly.

Floating-point products are not safe for this (R1):

| c | float `ceil(1000*(1-c))` | exact m |
|---|---|---|
| 0.95 | **51** | 50 |
| 0.99 | **11** | 10 |
| 0.995 | **6** | 5 |
| 0.999 | **2** | 1 |

---

## 2. Data, returns and losses

**Series.** The primary series is the Nifty 50 Price Return index, field Close (D2). The TRI Gross series is a sensitivity run only. The two are never mixed within a run.

**Index adjustments.** The index level already absorbs corporate actions through divisor adjustment, as described by NSE methodology documents (confirmed on Day 3). No stock-style split, bonus or dividend adjustment is applied.

**Sample.**
- The sample starts at the official base date, 3-Nov-1995 (D1). The first return is the first trading day after that date.
- Values shown for Jul-1990 to Nov-1995 are back-calculated and are excluded.
- The sample ends at the last close available at retrieval.
- Coverage is confirmed from the downloaded files on Day 3, subject to D11.

**Returns and losses.** r_t = ln(P_t / P_{t−1}) between consecutive *available* trading days, and L_t = −r_t.
- Missing observations are never filled or interpolated.
- A return spanning a holiday gap is one observation.

**Units.** All modelling, forecasting and backtesting is in log-loss units.
- For display, VaR can be converted to a simple loss as VaR_simple = 1 − exp(−VaR_c). This is exact because the transformation is monotone.
- ES is **not** converted, because the transformation does not commute with expectation.

---

## 3. Sample design and leakage rules (D3)

| Period | Definition | Use |
|---|---|---|
| Training | returns 1…T0, T0 = 2000 | EDA decisions, dependence diagnostics, **threshold selection** |
| Rolling window | L[t−W+1 … t], W = 1000 | the fit behind the forecast for day t+1 |
| Out-of-sample (OOS) | target days T0+1 … N (≈ 5,650) | all backtests and model comparisons |

**Leakage rules:**
1. The forecast for t+1 uses only L[t−W+1 … t].
2. No full-sample statistic enters any forecast.
3. Threshold selection receives only the training slice.
4. q\* is frozen and committed before the first OOS run (the Day-7 gate).

**Rolling indexing (D41; code `rolling.py`).** The code is 0-based over the loss series L[0..n−1], and training is L[0:T0]. The forecast for target j, for j = T0 … n−1, uses the window L[j−W:j]. The AR filter specs also use pre_lag = L[j−W−1] (D39 (d)). If j−W−1 < 0, the pre-lag is missing and the day is flagged "no_pre_window_lag". Under W = 2000 this happens at the first target only. Under W = 1000 the first target has window L[1000:2000] and pre-lag L[999]. This is the 1-based rule above with t = j: the window ends at the last observed day. Each target gets one filter fit and one GPD fit per q. No state carries over from one target to the next. The per-target code receives only L[:j]. The tests check future-perturbation invariance, past-beyond-window invariance and the exact window passed to each model (`tests/test_rolling.py`).

Expected breaches over T = 5,650 OOS days:
- c = 0.95: 282.5;
- c = 0.99: 56.5;
- c = 0.999: 5.65, which is why 0.999 is descriptive only.

---

## 4. Historical Simulation (A1)

**Definition.** Sort the window losses ascending: L_(1) ≤ … ≤ L_(W). The empirical CDF is F_W(x) = #{L_i ≤ x}/W, and m = W·p is computed exactly.

- **VaR** is the lower inverse of F_W (numpy's `inverted_cdf`):

  `VaR_c = inf{x : F_W(x) ≥ c} = L_(⌈W·c⌉)`
- **ES** is the Acerbi–Tasche integral form ES_c = (1/p) ∫_c^1 VaR_u du, evaluated exactly on F_W:

  `ES_c = [ (m − ⌊m⌋)·L_(W−⌊m⌋) + Σ_{j=W−⌊m⌋+1}^{W} L_(j) ] / m`

  The ES-as-integral form is attributed to [L21, F].

**Interpretation.**
- If m is an integer, ES is the mean of the m largest losses, and VaR = L_(W−m) is the next order statistic below them. Exactly m window losses exceed VaR when there are no ties.
- For W = 1000, c = 0.95 uses the worst **50** losses and c = 0.99 the worst **10**.
- If m is not an integer, the order statistic L_(W−⌊m⌋), which is VaR in that case, gets the fractional weight m − ⌊m⌋.
- ES ≥ VaR always.

**Assumptions.** The window is treated as a sample from the current loss distribution: unconditional and stationary within the window. There is no extrapolation beyond the largest window loss.

**Numerical limits.** All index arithmetic is exact. Ties are handled deterministically by the sort.

**Worked examples.** Losses are 1, 2, …, W (R2, exact, cross-checked by exact piecewise integration of VaR_u):

| W | c | m | VaR | ES |
|---|---|---|---|---|
| 1000 | 0.99 | 10 | 990 | 1991/2 = 995.5 |
| 1000 | 0.95 | 50 | 950 | 1951/2 = 975.5 |
| 250 | 0.99 | 5/2 | 248 | (½·248 + 249 + 250)/2.5 = 1246/5 = 249.2 |
| 250 | 0.95 | 25/2 | 238 | (½·238 + Σ239…250)/12.5 = 6106/25 = 244.24 |
| 20 | 0.95 | 1 | 19 | 20 |

**Tail-selection check (R2):** replacing 100 losses by large gains leaves VaR and ES unchanged. Exactly 10 losses exceed VaR at W = 1000, c = 0.99.

---

## 5. Gaussian benchmark

**Definition.** Fit μ̂ and σ̂ (ddof = 1) to the window. With z_c = Φ⁻¹(c):

- `VaR_c = μ̂ + σ̂·z_c`
- `ES_c = μ̂ + σ̂·φ(z_c)/p`

**Assumption.** Normal losses within the window. This is a known misspecification for equity returns and is the reason the benchmark is included.

**Worked example** (μ = 0, σ = 1; R3; ES checked against ∫_z^∞ xφ(x)dx / p):

| c | VaR | ES |
|---|---|---|
| 0.95 | 1.644854 | 2.062713 |
| 0.99 | 2.326348 | 2.665214 |

---

## 6. POT-GPD

**Parameterisation.** The excess over threshold u is y = L − u ≥ 0.

- `G(y; ξ, β) = 1 − (1 + ξy/β)^(−1/ξ)` for ξ ≠ 0
- `G(y; ξ, β) = 1 − exp(−y/β)` for ξ = 0

Here β > 0. If ξ < 0, the support ends at −β/ξ. [L01, L02, L05: F]

**Implementation notes.**
- SciPy's `genpareto(c=ξ, loc=0, scale=β)` uses the same sign for ξ. **Trap:** `genextreme` uses c = −ξ.
- Location is always fixed at 0.

**Estimation.**
- MLE minimises our own negative log-likelihood over (ξ, log β). It starts from Probability-Weighted Moments (PWM) estimates [L06, F].
- `scipy.stats.genpareto.fit(floc=0)` is used in tests only, as an independent cross-check.
- MLE regularity requires ξ > −½ [L04, F].
- Asymptotic SE(ξ̂) ≈ (1+ξ)/√N_u [L04/L05, F]. At ξ = 0.2 (R4):

| N_u | 400 | 200 | 144 | 100 | 50 |
|---|---|---|---|---|---|
| SE(ξ̂) | 0.060 | 0.085 | 0.100 | 0.120 | 0.170 |

**Tail estimator.** With n = window length and N_u = number of exceedances, for x > u:

`F̄(x) ≈ (N_u/n)(1 + ξ(x−u)/β)^(−1/ξ)` [L15, F]

**VaR and ES:**
- `VaR_c = u + (β/ξ)·[((n/N_u)·p)^(−ξ) − 1]`. Valid only if p < N_u/n; this is guarded in code.
- `ES_c = (VaR_c + β − ξu)/(1 − ξ)`. ES is finite **iff ξ < 1**. Finite variance additionally needs ξ < ½.
- At ξ = 0: `VaR_c = u − β·ln((n/N_u)·p)` and `ES_c = VaR_c + β`.
- ES − VaR = (β + ξ(VaR − u))/(1 − ξ) > 0 for all admissible ξ < 1.

**Numerical limits.**
- The ξ = 0 branch is used when |ξ| < 1e-8. Elsewhere `expm1`/`log1p` are used.
- Evidence for this choice (R4): at ξ = ±1e-9, the naive formula differs from the limit by about 4e-8 because of cancellation.
- If ξ̂ ≥ 1, ES is reported as ∞ and flagged, never clipped.
- Too few exceedances or a non-converging fit raise `EstimationError`. The forecast becomes NaN and is flagged.

**Worked example.** n = 1000, N_u = 100, u = 2, ξ = 0.25, β = 0.5 (R4). VaR is cross-checked by bisection on F̄ and ES by integrating F̄ beyond VaR:

| c | VaR | ES |
|---|---|---|
| 0.99 | 3.556559 | 4.742078 |
| 0.95 | 2.378414 | 3.171219 |

For c = 0.99, (n/N_u)·p = 0.1 and 0.1^(−0.25) = 1.778279, so VaR = 2 + 2·0.778279. Because β − ξu = 0, ES = VaR/0.75.

---

## 7. Threshold selection (V2.1 §3; training data only)

**Why low thresholds are preferred.**
- Threshold choice "involves balancing bias and variance". Too low a threshold gives GPD-approximation bias; too high a threshold gives estimation variance [L10, C1].
- The prescription is the lowest threshold that is adequate and stable [L08, F; L13, C2 (abstract)].
- "Lowest" is applied **only after** the adequacy and stability checks.

**Procedure.** All constants are pre-registered in `configs/base.toml` (D4a–D4d).

- **Step 0: N_min.** Before seeing data, a simulation sets N_min.
  - Design: true ξ ∈ {0.1, 0.25, 0.4}, β = 1, N_u ∈ {50, 75, 100, 150}, 2,000 replications, seed 20261111.
  - Rule: N_min is the smallest N_u with MLE convergence ≥ 99%, RMSE(ξ̂) ≤ 0.15 and relative RMSE(VaR99) ≤ 15% at every ξ.
  - This gives q_max = 1 − N_min/W.
- **Step 1: grid.** q_k = 0.80, 0.81, …, 0.95 (16 candidates, ascending). Threshold u_k is the training q_k-quantile.
- **Step 2: Anderson–Darling (AD) p-value per candidate.** Parametric bootstrap via `scipy.stats.goodness_of_fit`: GPD with loc = 0, refit per replicate, 999 reps, seed 20261111. Day 7 checks that the replicate refit keeps loc = 0.
- **Step 3: ForwardStop** [L14, C1; L13, C1-sections]:

  `k̂ = max{k ∈ 1…m : −(1/k) Σ_{i≤k} ln(1 − p_i) ≤ α_FS}`

  with k̂ = 0 if the set is empty. The first k̂ candidates are rejected, q_FS = q_{k̂+1}, and α_FS = 0.10 (D4b).
- **Step 4: stable region.** q\* is the smallest q ≥ q_FS with q ≤ q_max such that every candidate in [q, q + 0.03] (4 grid points) satisfies, relative to q:
  - |Δξ̂| ≤ 0.10;
  - |ΔVaR99|/VaR99 ≤ 5%;
  - |ΔES99|/ES99 ≤ 10%;
  - the candidate is not rejected by ForwardStop.

  These are practical tolerances, not hypothesis tests (D4c).
- **Step 5: split-half check.** Steps 2–4 are rerun on returns 1–1000 and 1001–2000. A difference of more than 0.03 from q\* is flagged to you.
- **Fallback.** If no q qualifies, this is recorded as a rule failure and you decide. There is no automatic choice.

**Guarantee status.**
- ForwardStop controls the false discovery rate (FDR) only for **independent U(0,1) null p-values** [L14, C1, Theorem 2]. Nested thresholds violate this assumption.
- Bader et al. note "the p-values of the sequential tests here are dependent". They support reasonable error control **by simulation, not proof** [L13, C1-sections: arXiv v1 method section and concluding discussion].
- **No guarantee is claimed.** Behaviour is measured in §8.

**Fixed vs recalibrated threshold (D5).** The quantile level q\* is fixed, and u_t is recomputed as the q\*-quantile of each window, so N_u = (1 − q\*)·W is constant.

**Reporting.** `reports/threshold_selection/` holds one row per candidate:
> q, u, N_u (training and window), ξ̂, SE, profile CI, β̂, β\* = β̂ − ξ̂u, AD statistic, AD p, ForwardStop statistic, rejected?, VaR95/ES95/VaR99/ES99 with CIs, convergence share, flags

It also holds the MRL, ξ̂/β\* stability, VaR/ES-vs-q and ForwardStop plots.

**Known risk** [L10, C1]: Scarrott & MacDonald show a rainfall example in which two thresholds both consistent with MRL guidance give very different shape estimates (0.21 vs 0.003), and in which several stability plateaus exist. A "first stable region" rule can therefore land on the first of several plateaus. Selection uncertainty is reported in three ways:
- the spread of VaR and ES across the stable region;
- the split-half q values;
- OOS sensitivity runs at q\* − 0.02, q\* + 0.02 and q\* + 0.03.

**DX-2 (approved, D22).** Bader et al. compute AD p-values from a fast approximation based on Choulakian–Stephens results, not by bootstrap. The arXiv v1 sections were checked directly [L13, C1-sections]. They state that "bootstrap based tests that put a lower bound on the p-values (1 divided by the bootstrap sample size) may lead to premature stopping", and that "parametric bootstrap puts bounds on the approximate p-values which would invalidate the stopping rules".

**Decision:** the V2.2 parametric bootstrap is **retained**, with B = 999 in the real selection and B = 199 in the A3 study.

**Documented finite-replication limitation:**
- The smallest attainable p-value is 1/(B+1) under the (1 + #)/(B + 1) convention, about 0.001 at B = 999 and 0.005 at B = 199.
- ForwardStop's transformed values −ln(1 − p) are therefore bounded below, which can weaken the stopping rule.
- The behaviour of the rule, including any premature stopping, is measured by the A3 study, not assumed.

---

## 8. Threshold-procedure simulation study (A3; Day 7, before the real selection)

All results are labelled **"simulation evidence (DGPs S1–S4)"**, never guarantees.

**Settings.**
- Sample size n = 2000. The last 1000 points are also refit at q\*.
- R = 200 replications per cell. A pre-declared rule drops R to 100 if the Day-6 timing pilot projects more than 2 hours.
- AD bootstrap uses 199 reps in the study, against 999 in the real selection.
- Seeds come from `SeedSequence(20261111).spawn`.
- The study runs in parallel with `ProcessPoolExecutor`, and results are independent of worker count.

| ID | Data-generating process (DGP) | Known truth | Cells |
|---|---|---|---|
| S1 | i.i.d. spliced: CDF Φ below u0 = Φ⁻¹(q0); above it, q0 + (1−q0)·G(x−u0; ξ, β) with β = (1−q0)/φ(u0) (continuous density) | exact marginal and boundary q0 | q0 ∈ {0.85, 0.90} × ξ ∈ {0.10, 0.25, 0.40} |
| S2 | i.i.d. Student-t, ν = 4 | exact VaR and ES; asymptotic ξ = 0.25 | 1 |
| S3 | ARMAX Fréchet X_t = max(aX_{t−1}, (1−a)Z_t), mapped through U_t = exp(−1/X_t) and L_t = F_S1⁻¹(U_t) | S1 marginal (q0 = 0.90, ξ = 0.25); extremal index 1 − a [F] | a ∈ {0.3, 0.6} |
| S4 | GARCH(1,1), a1 = 0.08, b1 = 0.90, unit variance, standardised t(6) innovations, burn-in 1000; **simulated, never fitted** | reference VaR and ES from a seeded 10⁷-point simulation | 1 |

**Metrics per cell:**
1. Distribution of the selected q\*, including the fallback rate.
2. Frequency of false selection, P(q\* < q0) (S1, S3).
3. Empirical FDR, the mean of V/max(1, R_rej) (S1, S3).
4. Bias and RMSE of ξ̂.
5. Relative bias and RMSE of VaR95, ES95, VaR99 and ES99, at n = 2000 and for the 1000-point window refit.
6. Rates of non-convergence, ξ̂ ≥ 0.5 and ξ̂ ≥ 1.

**Review triggers (D13).** In either S3 cell, any of the following is reported to the owner and **never acts automatically**:
- false selection > 0.20;
- empirical FDR > 2·α_FS;
- rule failure > 0.10.

A fired trigger is documented as a limitation, and the owner chooses one of:
- (i) proceed as specified;
- (ii) raise α_FS;
- (iii) apply the GoF test to declustered cluster maxima.

**Any change to the procedure requires the full A3 study to be re-run before the real training data is touched.** Until that re-run is complete, the real selection (§7) is not executed.

**Required outputs.** Committed to `reports/threshold_selection/simstudy/` and labelled "simulation evidence (DGPs S1–S4)".
- **Metrics table:** one row per cell, with all six metric groups above, the replication count R and the seeds used.
- **Plots (one per metric family):**
  - (a) histogram of the selected q\* per cell, with q0 marked and fallback counts annotated;
  - (b) empirical FDR per cell against the nominal α_FS and the 2·α_FS trigger line;
  - (c) distribution of ξ̂(q\*) − ξ per cell;
  - (d) relative errors of VaR95, ES95, VaR99 and ES99 per cell, at n = 2000 and for the 1000-point refit;
  - (e) failure rates per cell (rule failure, non-convergence, ξ̂ ≥ 0.5, ξ̂ ≥ 1).
- **Trigger evaluation:** values against each trigger for both S3 cells, reported to the owner before Step 2 of §7 runs on real data.

---

## 9. Dependence diagnostics (V2.1 §2.4)

Decisions use training data. Full-sample versions are reported and labelled "descriptive".

| Target | Diagnostic |
|---|---|
| r_t | ACF to lag 20; Ljung–Box Q(10), Q(20). LB on raw returns is size-distorted under conditional heteroskedasticity, so it is interpreted cautiously. |
| \|r_t\|, r_t² | ACF; Ljung–Box; ARCH-LM [L18, F] |
| clustering of extremes | exceedance timeline; QQ plot of inter-exceedance times against Exp(1); ACF of the exceedance indicator |
| dependence among exceedances | extremal index θ̂ by the Ferro–Segers intervals estimator, automatic declustering [L11, C2]; mean cluster size ≈ 1/θ̂ |

**Consequences.**
- Using all exceedances still targets the marginal tail.
- i.i.d. likelihood CIs are too narrow; block-bootstrap CIs (§13) are reported alongside them.
- The declustered fit is used **only** to compare ξ̂. It is never used for VaR.
- No GARCH model in v1 (D9).

### 9.1 Exact constants (D29)
The exact constants and conventions of the Day-4 battery are pinned in D29 (`decisions.md`) and `configs/base.toml` `[dependence]`. They cover the return definition, the threshold convention u = L_(⌈nq⌉), the ACF, Ljung–Box and ARCH-LM lags, the exceedance levels and the θ̂ formula. Results are in `eda.md`.

---

## 10. VaR backtests

**Hit.** `I_{t+1} = 1{L_{t+1} > VaR_{c,t}}` (strict inequality). Each model is evaluated on **its own admissible dates** (A2).

### 10.1 Kupiec POF, likelihood-ratio form [L22, F]

With x breaches in T days, p̂ = x/T and 0·ln 0 = 0:

`LR_POF = −2·[(T−x)·ln(1−p0) + x·ln p0 − (T−x)·ln(1−p̂) − x·ln p̂]`

- H0: p = p0 against H1: p ≠ p0 (two-sided).
- Reject at the 5% level if LR > χ²₁(0.95) = z²₀.₉₇₅ = 3.841459 (R6).
- Check values: LR(250, 5, 0.01) = 1.956810, and LR(250, 0, 0.01) = −2T·ln(1−p0) = 5.025168 (R6).
- **LR acceptance regions** (R6, R6b; each checked by two algebraically different LR formulas):
  - T = 5,650: [251, 315] at p0 = 0.05 and [43, 71] at p0 = 0.01.
  - T = 250: [1, 6] at p0 = 0.01 and [7, 19] at p0 = 0.05. Zero breaches at T = 250, p0 = 0.01 are **rejected** (LR = 5.025168 > 3.841459). DX-1, approved D21.

**The two tests have different acceptance regions; never interchange them** (R6, R6b, R7):

| T | p0 | Kupiec LR region (§10.1) | Equal-tailed exact binomial region (§10.2, D19) |
|---|---|---|---|
| 5,650 | 0.05 | [251, 315] | [251, 315] |
| 5,650 | 0.01 | **[43, 71]** | **[42, 72]** |
| 250 | 0.05 | **[7, 19]** | **[6, 20]** |
| 250 | 0.01 | **[1, 6]** | **[0, 6]** |

### 10.2 Equal-tailed exact binomial test (D19)

With X ~ Bin(T, p0):

`p_bin = min{1, 2·min[P(X ≤ x), P(X ≥ x)]}`

- Implemented with `scipy.stats.binom.cdf`/`sf`, **not** `binomtest`.
- The test does not reject (p_bin > 0.05) exactly when x lies in the equal-tailed region, where each tail probability exceeds 2.5%.
- **Equal-tailed regions** (R7, exact big-integer arithmetic). These are kept separate from the LR regions:

| T | p0 = 0.05 | p0 = 0.01 | p0 = 0.001 |
|---|---|---|---|
| 5650 | [251, 315] | **[42, 72]** | [2, 11] |
| 6650 | [298, 368] | [51, 83] | [2, 12] |
| 4770 | [209, 268] | [35, 62] | [1, 9] |
| 250 | [6, 20] | [0, 6] | [0, 2] |

- Example (R7, exact): T = 250, x = 5, p0 = 0.01 gives p_bin = 0.215625.

### 10.3 Size and power, independent hits (R8)

Exact values come from enumeration. The Monte Carlo check uses seed 20261111 and R = 100,000.

| p0 | true p | exact | MC (se) |
|---|---|---|---|
| 0.01 | 0.0100 | 0.0523 | 0.0516 (0.0007) |
| 0.01 | 0.0125 | 0.4508 | 0.4497 (0.0016) |
| 0.01 | 0.0150 | 0.9295 | 0.9295 (0.0008) |
| 0.01 | 0.0200 | 1.0000 | 1.0000 |
| 0.05 | 0.0500 | 0.0472 | 0.0478 (0.0007) |
| 0.05 | 0.0550 | 0.3879 | 0.3877 (0.0015) |
| 0.05 | 0.0600 | 0.9070 | 0.9060 (0.0009) |
| 0.05 | 0.0700 | 1.0000 | 1.0000 |

### 10.4 Dependence-aware uncertainty (R9)

Hits follow a two-state Markov chain with the **correct** rate p0 and persistence π11:
- π01 = p0(1 − π11)/(1 − p0);
- ρ = π11 − π01;
- Var(x) ≈ T·p0(1−p0)·(1+ρ)/(1−ρ).

The true size of the nominal 5% Kupiec test, computed by exact dynamic programming over (count, state):

| p0 | π11 | ρ | variance inflation | true size |
|---|---|---|---|---|
| 0.01 | 0.01 | 0.000 | 1.000 | 0.0523 |
| 0.01 | 0.05 | 0.040 | 1.084 | 0.0623 |
| 0.01 | 0.10 | 0.091 | 1.200 | 0.0764 |
| 0.01 | 0.20 | 0.192 | 1.475 | 0.1099 |
| 0.05 | 0.05 | 0.000 | 1.000 | 0.0472 |
| 0.05 | 0.10 | 0.053 | 1.111 | 0.0598 |
| 0.05 | 0.20 | 0.158 | 1.375 | 0.0906 |
| 0.05 | 0.30 | 0.263 | 1.714 | 0.1296 |

MC check: π11 = 0.10 with R = 2,000 gives 0.0795 (se 0.0060).

**Applied to the real results** (D8): estimate π̂11 from each model's hit sequence, then report the adjusted size and a block-bootstrap CI for the breach rate (§13) next to the binomial Clopper–Pearson CI.

### 10.5 Christoffersen independence and conditional coverage [L23, F] (D20)

**Transition counts.** n_ij counts transitions from state i to state j in the hit sequence. π̂01 = n01/(n00 + n01), π̂11 = n11/(n10 + n11), and π̂ = (n01 + n11)/(n00 + n01 + n10 + n11).

- `LR_ind = −2·[ln L(π̂) − ln L(π̂01, π̂11)]`, where L(π) = (1−π)^(n00+n10) π^(n01+n11) and L(π01, π11) = (1−π01)^n00 π01^n01 (1−π11)^n10 π11^n11.
- `LR_cc = LR_POF + LR_ind`.
- The convention 0·ln 0 = 0 handles zero transition counts.

**Monte Carlo p-values.** p = (1 + #{LR\* ≥ LR_obs})/(R + 1), with R = 9,999 and seed 20261111.
- The IND null simulates i.i.d. Bernoulli(**p̂**).
- The CC null simulates i.i.d. Bernoulli(**p0**).
- This corrects V2.1 §4, which used p̂ for both.

**Why Monte Carlo.** Under independence E[n11] = T·p² ≈ 0.57 at c = 0.99, so the χ² approximation is unreliable.

**Blind spot.** The test captures first-order Markov dependence only. Inter-breach durations are shown descriptively; the duration test [L24, F] is not implemented in v1.

### 10.6 Basel traffic light (A4; D17)

- VaR at c = 0.99 only.
- Non-overlapping consecutive 250-observation blocks starting at the first OOS day: 22 full blocks for T ≈ 5,650. The final partial block (≈ 150 days) gets a breach count with **no zone**.
- Zones: green 0–4, yellow 5–9, red ≥ 10 [L25, F; to be confirmed against BCBS 22].
- **No verdict on the full sample. Labelled as a descriptive regulatory reference.**
- **Required outputs, per model:**
  - (a) a **table** with one row per block: block index, start and end dates, n_obs (250, or about 150 for the partial block), breach count and zone (empty for the partial block);
  - (b) a **strip plot** with blocks on the x-axis and one row per model, coloured by zone; the partial block is shown uncoloured;
  - (c) **zone counts** per model: the number of full blocks in green, yellow and red.
- **Block-construction tests:**
  - T = 5,650 gives exactly 22 full blocks plus one partial block of 150 observations, with no zone;
  - blocks are contiguous and non-overlapping, starting at OOS day 1;
  - zone boundaries fall at 4/5 and 9/10 breaches.
- Limitations:
  - it is a capital device, not a statistical test;
  - it is calibrated to 250 observations under independence;
  - blocks are not independent under clustering;
  - block alignment is arbitrary;
  - per-block power is very low: the Kupiec LR acceptance region at T = 250, p0 = 0.01 is [1, 6], and the equal-tailed exact binomial region is [0, 6] (DX-1, D21; R6b, R7).

### 10.7 ES exceedance-residual test (A5a; D14) [L15, F for the original form]

On each model's breach days H, define z_{t+1} = (L_{t+1} − ES_{c,t}) / ES_{c,t}.

- H0: E[z | breach] = 0. H1: E[z | breach] > 0, i.e. ES is under-forecast. The test is one-sided.
- Statistic: z̄. Null-centred i.i.d. bootstrap: z̃ = z − z̄, B = 9,999, seed 20261111, p = (1 + #{mean(z̃\*) ≥ z̄})/(B + 1).
- If n_breach < 20, the result is descriptive only.
- Limitations:
  - dependent residuals, which the i.i.d. bootstrap understates;
  - conditioning on each model's own VaR;
  - low power (about 56 breaches at c = 0.99).
- **Required checks (A5a):**
  - (1) residuals constructed with mean 0 give p-values that are approximately uniform over repeated draws (slow test);
  - (2) an ES under-forecast of +20% gives a small p-value;
  - (3) a hand-computed z̄ on a small constructed sample is reproduced exactly;
  - (4) n_breach < 20 returns "descriptive only" with no p-value.

---

## 11. FZ0 joint VaR–ES scoring (V2.1 §2, approach A; A2; D7)

**Background.** ES alone is not elicitable [L26, F], but the pair (VaR, ES) is jointly elicitable [L27, C2]. Comparative backtests require elicitable functionals [L29, C2].

**Source.** Patton, Ziegel & Chen (2019), eq. (6) in arXiv 1707.05108 [L28, C1; the journal equation number is F].

**Return-space convention** (used internally, the only place FZ0 is computed):
- Y = realised log return;
- α = p (0.05 or 0.01);
- v = −VaR_c and e = −ES_c;
- admissibility **e ≤ v < 0**, from Proposition 1 ("VaR and ES are both strictly negative") and footnote 1 ("requires imposing that e ≤ v").

`FZ0(Y, v, e; α) = −(1/(α·e))·1{Y ≤ v}·(v − Y) + v/e + ln(−e) − 1`

Lower is better.

**Conversion.** One function maps loss-space forecasts to return space. `fz0` raises `ValueError` if v ≥ 0, e > v, or any value is NaN. Nothing is ever repaired.

**Tie.** At Y = v the indicator is 1 but (v − Y) = 0, so the score is continuous. The strict hit convention (L > VaR) cannot change any score.

**Analytic identity.** This is derived here, not taken from the paper. At the true (v, e) of a continuous Y, E[(v−Y)·1{Y≤v}] = α(v − e), so **E[FZ0] = ln(−e)**. Under N(0,1), checked by numerical integration (R5):
- α = 0.05: 0.724022;
- α = 0.01: 0.980284.

Zero-degree homogeneity of score differences was also checked (R5).

**Worked examples** (R5):

| α | v | e | Y | FZ0 |
|---|---|---|---|---|
| 0.05 | −1.644854 | −2.062713 | −1 | 0.521445 |
| 0.05 | −1.644854 | −2.062713 | −2 | 3.964933 |
| 0.05 | −1.644854 | −2.062713 | −3 | 13.660902 |
| 0.05 | −1.644854 | −2.062713 | Y = v | 0.521445 |
| 0.01 | −2.326348 | −2.665214 | −2 | 0.853140 |
| 0.01 | −2.326348 | −2.665214 | −3 | 26.128862 |

Conversion check: VaR_c = 1.644854, ES_c = 2.062713 and L = 3 give Y = −3 and FZ0 = 13.660902 (R5).

**Common evaluation mask (A2).**
- admissible(i, t) means VaR and ES are finite with e ≤ v < 0. Inadmissible cases include NaN or fit failure, ES = ∞ (ξ̂ ≥ 1), VaR_c ≤ 0, and ES < VaR.
- The mask is C_t = ∧_{i ∈ {HS, Gaussian, POT}} admissible(i, t), computed per level.
- Mean FZ0 and all DM tests use **only C**, the same dates for every model and pair.
- Reported per level:
  - total OOS dates;
  - admissible forecasts per model;
  - inadmissible forecasts per model, by reason;
  - |C|;
  - mean FZ0 per model on C;
  - DM statistics on C.
- If |C|/total < 0.99, the report shows a prominent flag and lists the excluded dates.

**Six-model mask and families (D42).** This note supersedes the three-model mask above, as D38 (4) requires; `src/niftytail/backtest.py` implements it.
- C_t is computed per primary level and per pass, over the six primary keys gaussian, hs, pot_gpd (q\*), garch_n, fhs and garch_evt (q_c).
- A sensitivity key `<name>@<q>` is descriptive. It is scored on C ∧ admissible(key), so it never shrinks C. The W = 2000 pass has its own C.
- Each inadmissible forecast gets exactly one reason, taken in this order: missing (sub-counted by flag: garch_error, no_pre_window_lag, gpd_error, var_not_in_tail, other), es_infinite, var_nonpositive, es_below_var.
- DM families, reported separately:
  - (i) the plan's gaussian/hs/pot_gpd pairs, unadjusted;
  - (ii) the D38 (5) conditional family, Holm at 5%;
  - (iii) every other pair, including each sensitivity against its base key, descriptive and without a verdict.
- FZ0 and DM are computed at the primary levels only. Coverage tests (§10) run per key on its own admissible dates, compressed. At 0.995 and 0.999 only the breach count, p̂ and the equal-tailed binomial p are reported, as descriptive.
- Per-date listings (excluded dates, traffic-light block dates, hit strings) are written only under `reports/runs/`.
- Seeds and the remaining conventions are those pinned in D42 (e) and (f).

---

## 12. Diebold–Mariano test (A5b; D15) [L30, F]

- d_t = FZ0_A,t − FZ0_B,t on C, with T_c = |C|.
- `DM = d̄ / √(Ω̂/T_c)`, where Ω̂ is the Newey–West long-run variance with Bartlett weights 1 − j/(ℓ+1) [L31, F].
- Lag **ℓ = ⌊4(T_c/100)^{2/9}⌋**, which is 9 at T_c = 5,650 (R10, exact). Attributing this rule to [L31] is F.
- Two-sided test against N(0,1). Sensitivity: ℓ doubled.
- Masking produces gaps; HAC is applied to the compressed series (limitation noted).
- **Independent HAC comparison (A5b).** In tests, d̄ and Ω̂ must match an independent computation: OLS of d on a constant via `statsmodels` with `cov_type="HAC"`, the same `maxlags = ℓ`, Bartlett kernel and no small-sample correction (`use_correction=False`). The DM statistic equals that regression's t-statistic.
  - Tolerance: relative 1e-10.
  - Covered cases: ℓ = 9 and ℓ = 18.

---

## 13. Moving-block bootstrap (A5c; D16)

- Block length b = ⌈n^{1/3}⌉: 13 at n = 2000 and 18 at n = 5650 (R10, exact). The n^{1/3} rate is F; the constant is a pre-declared choice.
- B = 1,999 resamples, percentile intervals, no circular wrapping, seed 20261111.
- Sensitivity on the key CIs only: b/2 and 2b.

---

## 14. Parameter uncertainty

- Profile-likelihood and delta-method CIs for ξ, VaR and ES [L08, F].
- Block-bootstrap CIs (§13) alongside them.
- Rolling ξ̂ ± CI path over OOS.

---

## 15. Reproducibility (A5e)

**Interpreter and dependencies:**
1. `requires-python = ">=3.13,<3.14"` in `pyproject.toml`, and `.python-version = 3.13`. Python 3.14 is never used.
2. Direct dependencies are listed in `pyproject.toml` with lower bounds. **Exact versions come only from the committed `uv.lock`.**
3. Every run and test uses `uv sync --locked` / `uv run --locked`, which fails if `uv.lock` and `pyproject.toml` disagree.

**Manifest** (`manifest.json` per run):
4. Python version; every installed package with its version (`importlib.metadata`); platform; the numpy BLAS configuration; git commit hash plus a dirty-tree flag; sha256 of the config file; sha256 of every input data file.

**Determinism:**
5. BLAS and OpenMP threads are pinned to 1. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are set at process start in `niftytail.__main__` before numpy is imported, and in `tests/conftest.py` for test runs.
6. Each task gets one `numpy.random.Generator(PCG64(seed))`. Parallel tasks get child streams via `SeedSequence(seed).spawn(k)`, so results do not depend on worker count or scheduling. The global `np.random` is never used.
7. Optimiser method, start values, tolerances and iteration limits come from config.
8. Output order is sorted (index, then model, then level) and never depends on dict or set iteration order.
9. Floats are written with `repr` (round-trip precision, up to 17 significant digits).
10. No timestamps, hostnames or absolute paths inside artifacts that must be byte-identical. Timestamps appear only in run-folder names and logs.

**Acceptance:**
11. Same machine and same lockfile: `forecasts.csv` and `backtest.csv` are byte-identical across two runs.
12. A different machine: agreement to relative tolerance 1e-10. Bytes are not required, because libm and BLAS can differ.

---

## 16. Limitations (V2.1 §8)

1. At c = 0.99 with W = 1000, VaR is the 10th-largest-loss region of the window, so EVT should look close to HS. EVT's advantage lies at c ≥ 0.995, which cannot be backtested.
2. Under clustering, Kupiec over-rejects (R9) and the Christoffersen independence test is expected to reject for the unconditional models only (Gaussian, HS, POT-GPD). Amended by D38: the plan's risk 2 said "all unconditional models" when no conditional model existed; for the conditional models of §18 no such rejection is expected in advance.
3. Low power: a true breach rate of 1.25% against a nominal 1% is detected 45% of the time.
4. ForwardStop has no guarantee under nested dependence; evidence comes from simulation only.
5. Tolerances and the N_min criteria are practical constants.
6. Tail shape in the 1995–2003 training period may not represent later periods. This is monitored post hoc and never retuned.
7. Licensing is unresolved (D11).
8. Circuit-breaker censoring and the holiday-versus-missing ambiguity are open until the Day-3 audit.
9. Several foundational claims are F-status (see `literature.md`).

---

## 17. Discrepancies found while cross-checking against `plan_v2.2.md`

DX-1 and DX-2 were approved by the owner on 2026-11-11 (D21, D22). The plan text itself stays unchanged.

| ID | Location in plan | Finding | Handling (status) |
|---|---|---|---|
| **DX-1** | A4 Limitations: "Kupiec acceptance region at 250 obs is [0, 6]" | [0, 6] is the **equal-tailed binomial** region. The **Kupiec LR** region at T = 250, p0 = 0.01 is **[1, 6]**, because x = 0 gives LR = 5.025 > 3.841 (R6b). The V2 §2.2 table's 250-row "[0, 6]" is correct only as an equal-tailed binomial region. | Every region is labelled by its test (§10.1, §10.2, §10.6). **Approved (D21).** |
| **DX-2** | V2.1 §3 Step 2 (AD p-values by parametric bootstrap) | Bader et al. [L13] use a fast table-based AD approximation and state that bootstrap p-value lower bounds may cause premature stopping and weaken the stopping rules. | V2.2 bootstrap retained, with its finite-replication limitation documented in §7. **Approved (D22).** |
| **DX-3** | V2.1 §4: Christoffersen MC "Bernoulli(p̂)" for both tests | Corrected by D20 (CC simulated at p0). | Already approved as D20. Listed for traceability. |
| **DX-4** | V2.1 §3 / V2 text: "Kupiec power ≈ 0.40 / 0.91" (V2) | Already corrected in V2.1 to 0.4508 / 0.9295 (R8). The V2 numbers survive only inside the historical V2 sections of the plan copy. | No action. Listed for traceability. |
| **DX-5** | V2.1 §2.3 "Valid only if 1−α < N_u/n" and other α = c usages | This is the pre-A0 notation. In c/p notation the condition is p < N_u/n. | Notation only; A0 governs. |

---

## 18. Conditional models (D38, D39)

Added on Day 8A by decision D38 (owner-approved 2026-11-12). `plan_v2.2.md` is frozen and unchanged; D38 supersedes D9's "no GARCH in v1" and the plan's three-model list. Amended on Day 8A-2 by D39 (owner-approved 2026-11-13): the filter specification is chosen by training BIC, the AR specs use one pre-window observation, and q_c = 0.88 is fixed by design.

**Design (2×3).**

| | Normal | Empirical | EVT |
|---|---|---|---|
| Unconditional, on the raw window losses (§4–§6, unchanged) | Gaussian | HS | POT-GPD |
| Conditional, on the standardized residuals of the D39 filter fitted in the same window | GARCH-N | FHS [L35, F] | GARCH-EVT [L15, F] |

**Filter family (loss space; D39 (c)).** Four candidate specifications, one of which is used for all three conditional models:

| | constant mean | AR(1) mean |
|---|---|---|
| GARCH(1,1) | F1 (the D38 filter) | F2 |
| GJR(1,1) [L39, F] | F3 | F4 |

- mean: `μ_t = φ0` (F1, F3) or `μ_t = φ0 + φ1·L_{t−1}` with \|φ1\| < 1 (F2, F4)
- `ε_t = L_t − μ_t = σ_t·z_t`
- `σ²_t = ω + (α + γ·1{ε_{t−1} > 0})·ε²_{t−1} + β·σ²_{t−1}`, with γ = 0 in F1 and F2. ε > 0 is a loss above its conditional mean (bad news).
- backcast `σ²_1 = mean(ε_t²)` over the window (it depends on the mean parameters)
- constraints ω > 0, α ≥ 0, γ ≥ 0, β ≥ 0, α + γ/2 + β < 1. A negative γ is excluded (a recorded restriction).

**Choice of the specification (D39 (b), (c); training data only).** D38 (2) justified a constant mean by a full-sample lag-1 ACF of r of about 0.05. That was the wrong series: on the training sample the D38 filter leaves Ljung–Box(10) on the level of z at Q = 42.7 (p = 5.5e-6), worse than on the raw losses (Q = 26.1). The specification is therefore chosen by a rule fixed before any fit:

- all four candidates are fitted on the same conditional sample t = 2…2000 of the training losses (1999 likelihood terms; L_1 is only the lag of the AR specs)
- `BIC = 2·NLL + k·ln(1999)` [L40, F], with k = 4, 5, 5, 6
- the lowest BIC is THE filter; ties go to the smaller k; no other criterion overrides it
- result: **F4, AR(1) with GJR(1,1)** (`reports/conditional/filter_selection.json`; table in `docs/conditional.md` §10)

**Information set of an AR spec in rolling use (D39 (d)).** The window L[t−W+1 … t] plus ONE pre-window observation L[t−W], used only as the lag of the first residual. Every window therefore has W residuals. No future data is used. This is a recorded deviation from "window only" (§3) for the conditional models. If L[t−W] does not exist (the first origin of the W = 2000 sensitivity), the conditional forecasts of that day are missing and counted as "no_pre_window_lag", separately from `GarchError`.

**Estimation.** Gaussian quasi-maximum likelihood [L36, F] over the W window observations:

`NLL = ½ Σ_t (ln 2π + ln σ²_t + ε²_t/σ²_t)`

- Reparameterisation: with m and v the window mean and variance (ddof = 0; the pre-window lag is excluded), θ = (t, a, b1, b2[, g][, p]) gives φ1 = tanh(p), φ0 = (m + √v·t)·(1 − φ1), ω = v·exp(a), (α, γ/2, β, 1 − α − γ/2 − β) = softmax(b1, g, b2, 0). The constraints hold by construction. For F1 this is the D38 map, and the F1 code path is numerically unchanged. The m, v scaling only makes the optimiser's path location- and scale-free.
- Optimiser: scipy Nelder–Mead on θ; `[garch]` xatol 1e-8, fatol 1e-10, maxiter 5000, maxfev 10000; initial simplex = the start and the start + 0.1 in each coordinate.
- Starts: t = 0 (the window mean as unconditional mean). Variance starts: GARCH (α, β) = (0.05, 0.90) and (0.10, 0.85); GJR (α, γ, β) = (0.03, 0.08, 0.88) and (0.05, 0.10, 0.83); ω = v·(1 − α − γ/2 − β). AR specs cross these with two φ1 starts, 0 and the OLS slope of L_t on L_{t−1} (4 starts; a slope outside (−1, 1) drops its two starts). The usable start (scipy success, finite NLL) with the lowest NLL is kept.
- Failure (non-finite or constant input, non-finite pre-window lag, no usable start, non-finite σ path or forecast) raises `GarchError`. Nothing is clipped; there is no fallback.
- Flag "near_boundary" if α + γ/2 + β > 0.999 or α < 1e-6 (reported only).

**Residuals and forecast.** z_t = ε_t/σ_t for t = 1…W: W residuals, so the threshold, n_min, q_max and A1 conventions apply unchanged. One-step forecast: `μ_{t+1} = φ0 + φ1·L_t` and `σ_{t+1} = √(ω + (α + γ·1{ε_t > 0})·ε_t² + β·σ_t²)`.

**Conditional risk (implemented on Day 8B).**

- `VaR_c = μ_{t+1} + σ_{t+1}·VaR_z` and `ES_c = μ_{t+1} + σ_{t+1}·ES_z`
- GARCH-N: VaR_z = Φ⁻¹(c), ES_z = φ(Φ⁻¹(c))/(1 − c) (the §5 values at μ = 0, σ = 1)
- FHS: the §4 (A1) HS conventions applied to the W residuals
- GARCH-EVT: §6 on the residuals: `fit_mle` on the excesses over z_(⌈W·q_c⌉), then `tail_var`/`tail_es`

**Residual threshold level q_c (D39 (a)).** `[conditional] q_c = "0.88"`, fixed by design: it equals q_selected (D37), so that GARCH-EVT and POT-GPD differ only in conditioning. McNeil–Frey also use a fixed level [L15, F]. q_c is not data-selected. The Day-8A threshold procedure on the D38 residuals failed (`all_rejected`; `docs/conditional.md` §6) and is not re-run to choose q_c; the later run on the F4 residuals is descriptive only (`docs/conditional.md` §10.4).

**OOS failure policy (Day 9+).** A `GarchError` or a missing pre-window lag on a forecast day makes all three conditional forecasts missing that day; the two causes are counted separately. The common mask (§11) runs over all six models, so the day is excluded from FZ0 and DM for every model. Backtests of the conditional models use their valid days; counts are reported. If `GarchError` failures exceed 1% of the OOS days, the run stops and is reported to the owner.

**Pre-declared conditional comparison family.** FZ0 with DM (§12, D15 lags) at both primary levels: GARCH-EVT vs POT-GPD, GARCH-EVT vs GARCH-N, GARCH-EVT vs FHS. Holm's step-down procedure [L37, F] over these 6 tests at 5%. The plan's existing comparisons are unchanged; all other pairs are descriptive.

**Sensitivity.** GARCH-EVT is run at q_c − 0.02, q_c + 0.02 and q_c + 0.03, i.e. 0.86, 0.90 and 0.91 (the D36/D37 rules for below-grid and below-n_min levels). The W = 2000 window sensitivity (D3) applies to all six models.

**Limitations.**

1. QML with a Gaussian likelihood is consistent for the GARCH parameters but not efficient [L36, F].
2. Forecasts ignore parameter-estimation error.
3. The specification is the BIC choice among four candidates on the training sample and may still be wrong. It is fixed for the whole out-of-sample period and never re-selected. γ is restricted to γ ≥ 0.
4. The A3 evidence for the threshold procedure (§8) applies to residuals only through the i.i.d. cells (S1, S2): estimated residuals are approximately, not exactly, i.i.d.
5. The residual threshold evidence (`docs/conditional.md`) uses one filter fit on the training losses; out of sample the filter is refitted in every 1000-day window.
6. The chosen filter F4 still rejects Ljung–Box on z² at lag 20 at the 1% level on the training sample (`docs/conditional.md` §10.2). D39 (e): this is reported, and no further specification is added.
7. q_c is not data-selected, and the GPD fit to the residual tails is weak on the training sample (D39 (a)). The out-of-sample backtest is the judge.

**`models.py` interface (D40).** Loss space; a forecast function sees only its arguments (a window, the pre-window lag, the config, a q level), never a date or data beyond the window.

- `Forecast` (frozen): `model`, `var` and `es` per level of `[levels]` primary + descriptive (Fraction keys; NaN when missing; ES = +inf only if ξ̂ ≥ 1), sorted `flags`, `diagnostics`.
- `gaussian(window, cfg)` §5; `hs(window, cfg)` §4 (A1); `pot_gpd(window, cfg, q)` §6 with u = L_(⌈Wq⌉). A level with p ≥ N_u/W is NaN ("var_not_in_tail"); a failed GPD fit is NaN everywhere ("gpd_error"); an infinite ES is flagged "es_infinite".
- `conditional(window, pre_lag, cfg, q_c)`: one fit of the `[conditional] filter` spec (F4), then GARCH-N, FHS = μ_{t+1} + σ_{t+1}·`hs(z)` and GARCH-EVT = μ_{t+1} + σ_{t+1}·`pot_gpd(z, q_c)`. A tuple of q levels gives one GARCH-EVT forecast per level from the same fit. Failures: "garch_error" and "no_pre_window_lag" (all three NaN), "gpd_error" (GARCH-EVT only).
- `forecast_all(window, pre_lag, cfg, q_star, q_c)` returns the six forecasts under the stable names of `MODELS`: gaussian, hs, pot_gpd, garch_n, fhs, garch_evt.
- No randomness and no warm start from a previous day.

**Checks (tests/test_garch.py).** NLL against an independent loop for every spec (relative 1e-12); hand-computed 3-observation recursions and forecasts (GARCH, GJR, AR(1)); the constraint maps; scale and shift equivariance (relative 1e-6); seeded recovery studies with tolerances pinned in D38 (8) and D39 (f) before they were run. Results: `docs/conditional.md`.

---

## Appendix A — Output of `py -3.13 -I docs\reference_values.py` (run 2026-11-11, Python 3.13.17, exit status 0)

Every line marked `check` or `raises` is **enforced**: a mismatch raises `CheckFailed` and the script exits with status 1. The failure path was verified by deliberately injecting a wrong value, which gave exit status 1.

```
NIFTYTAIL reference values - stdlib only - Python 3.13.17

==============================================================================
R1  Notation A0: c = confidence, p = 1 - c; m = W*p must be exact
==============================================================================
  c=0.95: float ceil(W*(1-c)) =  51   float W*(1-c) = 50.00000000000004       exact m = 50 [exact]
  c=0.99: float ceil(W*(1-c)) =  11   float W*(1-c) = 10.000000000000009      exact m = 10 [exact]
  c=0.995: float ceil(W*(1-c)) =   6   float W*(1-c) = 5.000000000000004       exact m = 5 [exact]
  c=0.999: float ceil(W*(1-c)) =   2   float W*(1-c) = 1.0000000000000009      exact m = 1 [exact]
  -> float tail counts are unsafe (11 instead of 10, 51 instead of 50); parse c as Fraction(str).

==============================================================================
R2  Historical Simulation (A1), losses = 1..W  [exact]
==============================================================================
      W      c      m    VaR     ES (exact)   ES float
   1000   0.99     10    990         1991/2   995.5000
   1000   0.95     50    950         1951/2   975.5000
    250   0.99    5/2    248         1246/5   249.2000
    250   0.95   25/2    238        6106/25   244.2400
     20   0.95      1     19             20    20.0000
  tail-selection (100 losses -> large gains): VaR 990->990, ES 1991/2->1991/2
  exceedances strictly above VaR (W=1000, c=0.99, no ties): 10

==============================================================================
R3  Gaussian (mu=0, sigma=1)  [float]
==============================================================================
  c=0.95: VaR = 1.644854   ES = 2.062713
    check VaR c=0.95 vs table: 1.644853627 vs 1.644854  |diff|=3.73e-07  tol=5e-07
    check ES c=0.95 vs table: 2.062712808 vs 2.062713  |diff|=1.92e-07  tol=5e-07
    check ES c=0.95 closed form vs integral: 2.062712808 vs 2.062712808  |diff|=3.29e-14  tol=1e-08
  c=0.99: VaR = 2.326348   ES = 2.665214
    check VaR c=0.99 vs table: 2.326347874 vs 2.326348  |diff|=1.26e-07  tol=5e-07
    check ES c=0.99 vs table: 2.66521422 vs 2.665214  |diff|=2.20e-07  tol=5e-07
    check ES c=0.99 closed form vs integral: 2.66521422 vs 2.66521422  |diff|=4.97e-14  tol=1e-08

==============================================================================
R4  POT-GPD: n=1000, N_u=100, u=2, xi=0.25, beta=0.5  [float]
==============================================================================
  c=0.99: VaR = 3.556559   ES = 4.742078
    check VaR c=0.99 vs table: 3.55655882 vs 3.556559  |diff|=1.80e-07  tol=5e-07
    check ES  c=0.99 vs table: 4.742078427 vs 4.742078  |diff|=4.27e-07  tol=5e-07
    check VaR c=0.99 closed form vs bisection: 3.55655882 vs 3.55655882  |diff|=0.00e+00  tol=1e-09
    check ES  c=0.99 closed form vs tail integral: 4.742078427 vs 4.742078427  |diff|=6.22e-15  tol=1e-06
  c=0.95: VaR = 2.378414   ES = 3.171219
    check VaR c=0.95 vs table: 2.37841423 vs 2.378414  |diff|=2.30e-07  tol=5e-07
    check ES  c=0.95 vs table: 3.171218973 vs 3.171219  |diff|=2.67e-08  tol=5e-07
    check VaR c=0.95 closed form vs bisection: 2.37841423 vs 2.37841423  |diff|=0.00e+00  tol=1e-09
    check ES  c=0.95 closed form vs tail integral: 3.171218973 vs 3.171218973  |diff|=5.77e-15  tol=1e-06
  xi=+1e-09: naive VaR = 3.151292506   xi=0 limit VaR = 3.151292546 (ES limit = VaR + beta = 3.651292546)
    check xi=+1e-09 naive formula near the xi=0 limit: 3.151292506 vs 3.151292546  |diff|=4.07e-08  tol=1e-06
  xi=-1e-09: naive VaR = 3.151292561   xi=0 limit VaR = 3.151292546 (ES limit = VaR + beta = 3.651292546)
    check xi=-1e-09 naive formula near the xi=0 limit: 3.151292561 vs 3.151292546  |diff|=1.48e-08  tol=1e-06
  SE(xi_hat) ~ (1+xi)/sqrt(N_u) at xi=0.2: N_u=400: 0.060, N_u=200: 0.085, N_u=144: 0.100, N_u=100: 0.120, N_u=50: 0.170
    check SE heuristic N_u=400: 0.06 vs 0.06  |diff|=0.00e+00  tol=1e-12
    check SE heuristic N_u=200: 0.085 vs 0.085  |diff|=0.00e+00  tol=1e-12
    check SE heuristic N_u=144: 0.1 vs 0.1  |diff|=0.00e+00  tol=1e-12
    check SE heuristic N_u=100: 0.12 vs 0.12  |diff|=0.00e+00  tol=1e-12
    check SE heuristic N_u=50: 0.17 vs 0.17  |diff|=0.00e+00  tol=1e-12

==============================================================================
R5  FZ0 (return space; alpha = tail probability p)  [float]
==============================================================================
  alpha=0.05: v = -1.644854  e = -2.062713
    FZ0(Y = -1      ) = 0.521445
    check FZ0 alpha=0.05 Y = -1: 0.5214445246 vs 0.521445  |diff|=4.75e-07  tol=5e-07
    FZ0(Y = -2      ) = 3.964933
    check FZ0 alpha=0.05 Y = -2: 3.964932845 vs 3.964933  |diff|=1.55e-07  tol=5e-07
    FZ0(Y = -3      ) = 13.660902
    check FZ0 alpha=0.05 Y = -3: 13.66090212 vs 13.660902  |diff|=1.18e-07  tol=5e-07
    FZ0(Y = v (tie) ) = 0.521445
    check FZ0 alpha=0.05 Y = v (tie): 0.5214445246 vs 0.521445  |diff|=4.75e-07  tol=5e-07
    check E[FZ0] alpha=0.05 numeric vs ln(-e): 0.7240220133 vs 0.7240220133  |diff|=7.77e-16  tol=1e-09
  alpha=0.01: v = -2.326348  e = -2.665214
    FZ0(Y = -2      ) = 0.853140
    check FZ0 alpha=0.01 Y = -2: 0.8531403061 vs 0.85314  |diff|=3.06e-07  tol=5e-07
    FZ0(Y = -3      ) = 26.128862
    check FZ0 alpha=0.01 Y = -3: 26.12886189 vs 26.128862  |diff|=1.10e-07  tol=5e-07
    check E[FZ0] alpha=0.01 numeric vs ln(-e): 0.9802844373 vs 0.9802844373  |diff|=2.33e-15  tol=1e-09
  conversion: VaR_L=1.644854, ES_L=2.062713, L=3 -> Y=-3
    check loss-space conversion example: 13.66090212 vs 13.660902  |diff|=1.18e-07  tol=5e-07
    check homogeneity of degree 0 (score differences): 3.408135274 vs 3.408135274  |diff|=1.78e-15  tol=1e-12
  invalid inputs must raise ValueError:
    raises ValueError: y = NaN
    raises ValueError: v = NaN
    raises ValueError: e = NaN
    raises ValueError: alpha = NaN
    raises ValueError: y = -inf
    raises ValueError: alpha = 0
    raises ValueError: alpha = 1
    raises ValueError: v = 0
    raises ValueError: v > 0
    raises ValueError: e > v
    check boundary e == v accepted: 0.4054651081 vs 0.4054651081  |diff|=0.00e+00  tol=1e-12

==============================================================================
R6  Kupiec LR (two-sided, 5%): reject if LR > chi2_1(0.95)
==============================================================================
  chi2_1(0.95) = z_0.975^2 = 3.841458821  [float, derived]
    check chi2_1(0.95) critical value: 3.841458821 vs 3.841459  |diff|=1.79e-07  tol=5e-07
    check LR(T=250, x=5, p0=0.01): 1.956809788 vs 1.9568  |diff|=9.79e-06  tol=5e-05
    check LR(T=250, x=5) two independent formulas: 1.956809788 vs 1.956809788  |diff|=3.02e-14  tol=1e-10
    check LR(T=250, x=0, p0=0.01) = -2T ln(1-p0): 5.025167927 vs 5.025167927  |diff|=0.00e+00  tol=1e-12
  LR acceptance region T=5650 p0=0.05: [251, 315]  (alt formula [251, 315]; min |LR-crit| at boundaries = 6.214e-03)
  LR acceptance region T=5650 p0=0.01: [43, 71]  (alt formula [43, 71]; min |LR-crit| at boundaries = 1.108e-01)

==============================================================================
R6b Kupiec LR acceptance regions at T=250 (traffic-light block size; DX-1)
==============================================================================
  LR acceptance region T=250 p0=0.01: [1, 6]  (alt formula [1, 6]; min |LR-crit| at boundaries = 2.861e-01)
  LR acceptance region T=250 p0=0.05: [7, 19]  (alt formula [7, 19]; min |LR-crit| at boundaries = 1.981e-01)
  zero breaches, T=250, p0=0.01: LR = 5.025168 > 3.841459 -> rejected

==============================================================================
R7  Equal-tailed exact binomial (D19), separate from LR  [exact big-int]
==============================================================================
      T     p0     E[x]  region (each tail > 2.5%)
   5650   0.05   282.50  [251, 315]
   5650   0.01    56.50  [42, 72]
   5650  0.001     5.65  [2, 11]
   6650   0.05   332.50  [298, 368]
   6650   0.01    66.50  [51, 83]
   6650  0.001     6.65  [2, 12]
   4770   0.05   238.50  [209, 268]
   4770   0.01    47.70  [35, 62]
   4770  0.001     4.77  [1, 9]
    250   0.05    12.50  [6, 20]
    250   0.01     2.50  [0, 6]
    250  0.001     0.25  [0, 2]
  D19 example: T=250, x=5, p0=0.01 -> equal-tailed p = 0.215625  [exact rational]
    check D19 example p-value: 0.2156247462 vs 0.215625  |diff|=2.54e-07  tol=5e-07
  D19 p-value > 0.05 exactly on [42, 72] at T=5650, p0=0.01 (checked at x=41,42,72,73)
  NOTE: LR and equal-tailed regions coincide at p0=0.05 ([251,315]) but differ at p0=0.01 (LR [43,71] vs equal-tailed [42,72]); at T=250, p0=0.01: LR [1,6] vs [0,6].

==============================================================================
R8  Kupiec LR size/power, iid hits, T=5650: exact [float sum] + MC [seed=20261111, R=100000]
==============================================================================
  p0=0.01  true p=0.01    exact=0.0523   MC=0.0516 (se 0.0007)
    check exact power p0=0.01 p=0.01 (4 dp): 0.0523 vs 0.0523  |diff|=0.00e+00  tol=1e-12
  p0=0.01  true p=0.0125  exact=0.4508   MC=0.4497 (se 0.0016)
    check exact power p0=0.01 p=0.0125 (4 dp): 0.4508 vs 0.4508  |diff|=0.00e+00  tol=1e-12
  p0=0.01  true p=0.015   exact=0.9295   MC=0.9295 (se 0.0008)
    check exact power p0=0.01 p=0.015 (4 dp): 0.9295 vs 0.9295  |diff|=0.00e+00  tol=1e-12
  p0=0.01  true p=0.02    exact=1.0000   MC=1.0000 (se 0.0000)
    check exact power p0=0.01 p=0.02 (4 dp): 1 vs 1  |diff|=0.00e+00  tol=1e-12
  p0=0.05  true p=0.05    exact=0.0472   MC=0.0478 (se 0.0007)
    check exact power p0=0.05 p=0.05 (4 dp): 0.0472 vs 0.0472  |diff|=0.00e+00  tol=1e-12
  p0=0.05  true p=0.055   exact=0.3879   MC=0.3877 (se 0.0015)
    check exact power p0=0.05 p=0.055 (4 dp): 0.3879 vs 0.3879  |diff|=0.00e+00  tol=1e-12
  p0=0.05  true p=0.06    exact=0.9070   MC=0.9060 (se 0.0009)
    check exact power p0=0.05 p=0.06 (4 dp): 0.907 vs 0.907  |diff|=0.00e+00  tol=1e-12
  p0=0.05  true p=0.07    exact=1.0000   MC=1.0000 (se 0.0000)
    check exact power p0=0.05 p=0.07 (4 dp): 1 vs 1  |diff|=0.00e+00  tol=1e-12

==============================================================================
R9  Kupiec true size, two-state Markov hits with correct rate p0, T=5650  [exact DP, float]
==============================================================================
  p0=0.01  pi11=0.01  rho=0.000  var-inflation=1.000  true size of nominal 5% test = 0.0523
    check Markov size p0=0.01 pi11=0.01 (4 dp): 0.0523 vs 0.0523  |diff|=0.00e+00  tol=1e-12
  p0=0.01  pi11=0.05  rho=0.040  var-inflation=1.084  true size of nominal 5% test = 0.0623
    check Markov size p0=0.01 pi11=0.05 (4 dp): 0.0623 vs 0.0623  |diff|=0.00e+00  tol=1e-12
  p0=0.01  pi11=0.1   rho=0.091  var-inflation=1.200  true size of nominal 5% test = 0.0764
    check Markov size p0=0.01 pi11=0.1 (4 dp): 0.0764 vs 0.0764  |diff|=0.00e+00  tol=1e-12
  p0=0.01  pi11=0.2   rho=0.192  var-inflation=1.475  true size of nominal 5% test = 0.1099
    check Markov size p0=0.01 pi11=0.2 (4 dp): 0.1099 vs 0.1099  |diff|=0.00e+00  tol=1e-12
  p0=0.05  pi11=0.05  rho=0.000  var-inflation=1.000  true size of nominal 5% test = 0.0472
    check Markov size p0=0.05 pi11=0.05 (4 dp): 0.0472 vs 0.0472  |diff|=0.00e+00  tol=1e-12
  p0=0.05  pi11=0.1   rho=0.053  var-inflation=1.111  true size of nominal 5% test = 0.0598
    check Markov size p0=0.05 pi11=0.1 (4 dp): 0.0598 vs 0.0598  |diff|=0.00e+00  tol=1e-12
  p0=0.05  pi11=0.2   rho=0.158  var-inflation=1.375  true size of nominal 5% test = 0.0906
    check Markov size p0=0.05 pi11=0.2 (4 dp): 0.0906 vs 0.0906  |diff|=0.00e+00  tol=1e-12
  p0=0.05  pi11=0.3   rho=0.263  var-inflation=1.714  true size of nominal 5% test = 0.1296
    check Markov size p0=0.05 pi11=0.3 (4 dp): 0.1296 vs 0.1296  |diff|=0.00e+00  tol=1e-12
  MC check p0=0.01 pi11=0.10 [seed=20261111, R=2000]: size = 0.0795 (se 0.0059)

==============================================================================
R10 Integer rules  [exact]
==============================================================================
  DM Newey-West lag floor(4*(T/100)^(2/9)) at T=5650: 9 (float: 9)
  block length ceil(n^(1/3)) at n=2000: 13   (12^3=1728 < 2000 <= 13^3=2197)
  block length ceil(n^(1/3)) at n=5650: 18   (17^3=4913 < 5650 <= 18^3=5832)

ALL 1386 CHECKS PASSED
```
