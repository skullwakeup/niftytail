# NIFTYTAIL — Conditional models: GARCH filter, residual threshold evidence and filter specification (Day 8A, D38; Day 8A-2, D39)

**Status:** §1–§9 are the Day-8A record (D38, the constant-mean GARCH(1,1) filter): evidence only, and no q_c was chosen there. §10 is Day 8A-2 (D39, owner-approved 2026-11-13): **q_c = 0.88 is fixed by design** and **the filter is chosen by training BIC: F4, AR(1) with GJR(1,1)**. No out-of-sample return has been computed.

**Source of every number in §4–§6:** `reports/threshold_selection/real_residuals/selection.json` (file sha256 `97249cef7ced190edae6d07cdf7e406cdea500ac7768805c204d0f9acb98ac44`, `body_sha256` `b6d3b05cc689a302fc27e6bcb499ac3f1d94e3b64be4717678fdcbe3f975dd7e`). The tables were written by a script that reads that file; values are shown to 4 significant digits. The recovery table in §2 was written by the same script from `tests/test_garch.py`.

**Sample:** the first 2000 log returns (training slice), dated 1995-11-06 to 2003-11-04; the same slice as Day 7C. Losses are L = −r in log-return units (0.01 = 1%). Standardized residuals z are unit-free.

## 1. What D38 changes

- The model set becomes a 2×3 design. Unconditional on raw losses (unchanged): Gaussian, HS, POT-GPD. Conditional on GARCH(1,1) standardized residuals of the same rolling window: GARCH-N, FHS, GARCH-EVT.
- D38 supersedes D9's "no GARCH in v1". `plan_v2.2.md` is frozen and unchanged.
- Filter: L_t = μ + ε_t, ε_t = σ_t z_t, σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}; constant mean; Gaussian QML; backcast σ²_1 = mean(ε²). A window of W losses gives W residuals.
- Conditional risk (Day 8B): VaR = μ + σ_{t+1}·VaR_z and ES = μ + σ_{t+1}·ES_z, where (VaR_z, ES_z) come from the normal formulas, HS on the residuals, or a GPD fit to the residual excesses above the q_c-quantile.
- A failed GARCH fit raises `GarchError`; out of sample that day is missing for all three conditional models, and the run stops if failures exceed 1% of the days.
- Pre-declared comparison family: GARCH-EVT vs POT-GPD, vs GARCH-N and vs FHS at both primary levels (6 FZ0/DM tests, Holm at 5%).
- Full text: `docs/decisions.md` D38 and `docs/methodology.md` §18.

## 2. Checks of the filter (synthetic data only)

### 2.1 Recovery study (D38 (8); tolerances pinned before the study was run)

GARCH(1,1) with (α, β) = (0.08, 0.90), ω = 0.02, μ = 0; 50 simulated series per cell; burn-in 1000. A cell passes if all 50 fits converge, \|mean(α̂) − 0.08\| ≤ 0.02, \|mean(β̂) − 0.90\| ≤ 0.04, \|mean of mean(z)\| ≤ 0.02 and \|mean of var(z) − 1\| ≤ 0.02.

| cell | fits converged | bias α̂ | bias β̂ | mean of mean(z) | mean of var(z) | result |
|---|---|---|---|---|---|---|
| normal, n = 1000 | 50 / 50 | 0.003194 | -0.01877 | 8.322e-05 | 0.9994 | pass |
| normal, n = 2000 | 50 / 50 | 0.000495 | -0.005199 | 0.0004548 | 0.9999 | pass |
| standardized t(6), n = 1000 | 50 / 50 | 0.002856 | -0.01705 | 0.002679 | 0.9995 | pass |
| standardized t(6), n = 2000 | 50 / 50 | 0.004337 | -0.01413 | -0.003187 | 0.9996 | pass |

The bias of β̂ is negative in all four cells. It is inside the tolerance, and at n = 1000 (the rolling-window length) it is about four times the n = 2000 value in the normal cells.

### 2.2 Other tests (`tests/test_garch.py`)

- NLL against an independent Python loop: relative 1e-12.
- A hand-computed 3-observation recursion and one-step forecast.
- The constraint map: round trip; ω > 0, α > 0, β > 0, α + β < 1 for 2000 random parameter vectors.
- Equivariance: L·c gives μ·c, ω·c², the same α and β, and σ and the forecast ·c; L + a gives μ + a. Relative 1e-6.
- The two-start rule keeps the lower NLL (also with a deliberately poor start in either position).
- `GarchError` on a constant series, on a non-finite input, when no start converges and on a non-finite forecast.
- Determinism: two fits give identical results.

### 2.3 Cross-check against the `arch` package (outside the repository, not a test)

`arch` 9.0.0 installed in a scratch virtual environment under Smart App Control and ran (it reported running its recursion without JIT compilation). One simulated series (n = 1000, the test fixture, seed 20261111, true μ = 0.5):

| | μ̂ | ω̂ | α̂ | β̂ | NLL |
|---|---|---|---|---|---|
| `niftytail.garch` | 0.490502 | 0.040539 | 0.111485 | 0.847437 | 1342.840552 |
| `arch`, default backcast (a different likelihood) | 0.490482 | 0.040566 | 0.111466 | 0.847422 | 1342.823324 |
| `arch`, backcast fixed at mean(ε²) | 0.490506 | 0.040541 | 0.111446 | 0.847446 | 1342.841697 |

- Differences from the second `arch` row: 4e-6 in μ̂, 2e-6 in ω̂, 4e-5 in α̂, 9e-6 in β̂.
- `arch`'s default backcast is an exponentially weighted average, so its first row maximises a slightly different likelihood. In its third row the backcast is a fixed number, whereas ours moves with μ; `arch` also applies variance bounds. The NLL values are therefore not expected to agree exactly.
- Our NLL evaluated at the two `arch` estimates is 1342.840555 and 1342.840559, both above our minimum of 1342.840552. The estimates agree to the precision the likelihood surface allows.

## 3. Method of the residual threshold evidence

1. One GARCH(1,1) fit on the 2000 training losses; z_t = ε_t/σ_t, t = 1…2000.
2. The unchanged Day-7 procedure (D34) on z: grid q = 0.80…0.95, GPD fit per candidate, AD p-value by parametric bootstrap (B = 999), ForwardStop (α_FS = 0.10), stable region, split-half check.
3. Seed node SeedSequence(20261111).spawn(2)[1]; no random generator is shared with Day 7C.
4. Diagnostics with no influence on the result: serial-dependence tests on z against the raw losses, extremal index and declustered-ξ comparison, ξ̂ side by side at q_c, 0.90 and 0.95.
5. **Stated limitation.** The "conv. share" column and the split halves refit the GPD on residuals of this single training GARCH fit. Out of sample, the residuals come from a GARCH refit in every 1000-day window.

## 4. Training GARCH fit

| Quantity | Value |
|---|---|
| observations | 2000 |
| μ̂ (loss units) | -0.0006021 |
| ω̂ | 1.175e-05 |
| α̂ | 0.1062 |
| β̂ | 0.8551 |
| α̂ + β̂ | 0.9613 |
| implied unconditional SD, √(ω̂/(1 − α̂ − β̂)) | 0.01742 |
| NLL at the optimum | -5488 |
| start kept (0: (0.05, 0.90); 1: (0.10, 0.85)) | 0 |
| flags | none |

Moments of the 2000 standardized residuals:

| mean(z) | var(z) (ddof = 0) | skewness | excess kurtosis |
|---|---|---|---|
| 0.02621 | 0.9993 | 0.06671 | 2.988 |

## 5. Residual diagnostics against the raw losses

Ljung–Box Q(m) with df = m on the series and on its squares; Engle ARCH-LM with df = lags (D29 conventions). "raw L" is the training loss series; "residual z" is the standardized residual series.

| Test | lag / df | raw L: statistic | raw L: p | residual z: statistic | residual z: p |
|---|---|---|---|---|---|
| Ljung–Box, level | 10 | 26.08 | 0.00364 | 42.74 | 5.531e-06 |
| Ljung–Box, level | 20 | 40.38 | 0.004465 | 53.36 | 7.2e-05 |
| Ljung–Box, squares | 10 | 161.6 | 1.481e-29 | 9.76 | 0.4618 |
| Ljung–Box, squares | 20 | 214.1 | 1.805e-34 | 44.84 | 0.001158 |
| ARCH-LM | 5 | 99.24 | 7.645e-20 | 6.225 | 0.2849 |
| ARCH-LM | 10 | 109 | 8.373e-19 | 9.609 | 0.4755 |

Cross-check: the raw-loss statistics equal the Day-4 values for r and r² in `reports/eda/eda_summary.json` (relative 1e-9): yes.

## 6. Residual threshold selection

### 6.1 Result

| Quantity | Value |
|---|---|
| k̂ (candidates rejected by ForwardStop) | 16 |
| q_FS | n/a |
| **q_c** (the rule's q\* on the residuals) | **n/a** |
| rule failure | all_rejected |
| q_star_at_grid_floor | n/a |
| split-half q, residuals 1–1000 | 0.83 |
| split-half q, residuals 1001–2000 | 0.80 |
| split-half flag | no |
| flags | none |

### 6.2 Candidate table: fit, goodness of fit, ForwardStop, stable region

"FS stat" is the running ForwardStop statistic. "N_u window" is (1 − q)·W for W = 1000. "conv. share" is the share of the 1,001 rolling 1000-point windows of the training residuals in which the GPD fit converges (see the limitation in §3).

| q | u (z units) | N_u | N_u window | ξ̂ | SE | 95% profile CI | β̂ | β\* | AD stat | AD p | FS stat | rejected | conv. share | first failing stable condition | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.7891 | 400 | 200 | 0.05482 | 0.05274 | [-0.02241, 0.1548] | 0.5639 | 0.5206 | 1.134 | 0.023 | 0.02327 | yes | 1 | rejected | none |
| 0.81 | 0.8342 | 380 | 190 | 0.07961 | 0.05538 | [-0.005089, 0.1894] | 0.5351 | 0.4687 | 0.5568 | 0.237 | 0.1469 | yes | 1 | rejected | none |
| 0.82 | 0.8584 | 360 | 180 | 0.07232 | 0.05652 | [-0.01211, 0.1823] | 0.5461 | 0.4841 | 0.7488 | 0.105 | 0.1349 | yes | 1 | rejected | none |
| 0.83 | 0.8996 | 340 | 170 | 0.08835 | 0.05902 | [-0.001783, 0.2062] | 0.5295 | 0.45 | 0.6618 | 0.156 | 0.1436 | yes | 1 | rejected | none |
| 0.84 | 0.9223 | 320 | 160 | 0.07361 | 0.06002 | [-0.01464, 0.1896] | 0.5501 | 0.4822 | 0.9579 | 0.05 | 0.1251 | yes | 1 | rejected | none |
| 0.85 | 0.9739 | 300 | 150 | 0.09956 | 0.06348 | [0.002732, 0.2273] | 0.5221 | 0.4251 | 0.6074 | 0.173 | 0.1359 | yes | 1 | rejected | none |
| 0.86 | 1.003 | 280 | 140 | 0.08909 | 0.06509 | [-0.007356, 0.2171] | 0.5381 | 0.4487 | 0.8392 | 0.08 | 0.1284 | yes | 1 | rejected | none |
| 0.87 | 1.053 | 260 | 130 | 0.1068 | 0.06864 | [0.003247, 0.2451] | 0.5214 | 0.4088 | 0.6865 | 0.154 | 0.1333 | yes | 1 | rejected | none |
| 0.88 | 1.096 | 240 | 120 | 0.1079 | 0.07152 | [0.001172, 0.2513] | 0.5247 | 0.4065 | 0.8094 | 0.071 | 0.1266 | yes | 1 | rejected | none |
| 0.89 | 1.138 | 220 | 110 | 0.1028 | 0.07435 | [-0.005771, 0.2495] | 0.5358 | 0.4188 | 1.073 | 0.031 | 0.1171 | yes | 1 | rejected | none |
| 0.90 | 1.21 | 200 | 100 | 0.1389 | 0.08053 | [0.01711, 0.3046] | 0.5007 | 0.3326 | 0.7475 | 0.098 | 0.1159 | yes | 1 | rejected | none |
| 0.91 | 1.262 | 180 | 90 | 0.1356 | 0.08465 | [0.01035, 0.3072] | 0.5112 | 0.34 | 0.9562 | 0.044 | 0.11 | yes | 1 | rejected | below_n_min_window |
| 0.92 | 1.343 | 160 | 80 | 0.1779 | 0.09312 | [0.0355, 0.3751] | 0.4764 | 0.2375 | 0.7673 | 0.078 | 0.1077 | yes | 1 | rejected | below_n_min_window |
| 0.93 | 1.392 | 140 | 70 | 0.149 | 0.0971 | [0.007396, 0.346] | 0.5186 | 0.3113 | 1.321 | 0.015 | 0.1011 | yes | 1 | rejected | below_n_min_window |
| 0.94 | 1.51 | 120 | 60 | 0.2344 | 0.1127 | [0.06067, 0.4802] | 0.4475 | 0.09363 | 0.6895 | 0.118 | 0.1028 | yes | 1 | rejected | below_n_min_window |
| 0.95 | 1.6 | 100 | 50 | 0.2491 | 0.1249 | [0.05886, 0.5208] | 0.4535 | 0.05503 | 0.8707 | 0.047 | 0.09934 | yes | 1 | rejected | below_n_min_window, var95_not_in_tail |

### 6.3 Candidate table: VaR_z and ES_z (residual scale, n = 2000 fit, delta-method 95% CIs)

These are quantiles of z, not of losses. A loss-space forecast is μ + σ_{t+1} times these numbers.

| q | VaR95_z | 95% CI | ES95_z | 95% CI | VaR99_z | 95% CI | ES99_z | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 1.601 | [1.515, 1.688] | 2.245 | [2.075, 2.415] | 2.625 | [2.405, 2.845] | 3.328 | [2.9, 3.756] |
| 0.81 | 1.588 | [1.504, 1.672] | 2.234 | [2.061, 2.408] | 2.61 | [2.385, 2.834] | 3.345 | [2.89, 3.799] |
| 0.82 | 1.591 | [1.507, 1.676] | 2.237 | [2.066, 2.409] | 2.614 | [2.391, 2.837] | 3.34 | [2.892, 3.787] |
| 0.83 | 1.584 | [1.502, 1.666] | 2.231 | [2.057, 2.405] | 2.604 | [2.379, 2.829] | 3.35 | [2.885, 3.816] |
| 0.84 | 1.59 | [1.507, 1.673] | 2.237 | [2.066, 2.408] | 2.614 | [2.392, 2.836] | 3.342 | [2.892, 3.793] |
| 0.85 | 1.58 | [1.5, 1.66] | 2.227 | [2.053, 2.401] | 2.597 | [2.371, 2.822] | 3.356 | [2.877, 3.835] |
| 0.86 | 1.583 | [1.504, 1.663] | 2.231 | [2.06, 2.402] | 2.604 | [2.38, 2.828] | 3.351 | [2.883, 3.82] |
| 0.87 | 1.578 | [1.502, 1.654] | 2.224 | [2.052, 2.397] | 2.592 | [2.367, 2.817] | 3.36 | [2.872, 3.848] |
| 0.88 | 1.578 | [1.504, 1.652] | 2.224 | [2.052, 2.396] | 2.591 | [2.367, 2.816] | 3.36 | [2.87, 3.85] |
| 0.89 | 1.578 | [1.507, 1.65] | 2.226 | [2.056, 2.395] | 2.595 | [2.372, 2.819] | 3.359 | [2.874, 3.845] |
| 0.90 | 1.574 | [1.51, 1.639] | 2.215 | [2.043, 2.386] | 2.569 | [2.344, 2.793] | 3.369 | [2.846, 3.892] |
| 0.91 | 1.575 | [1.515, 1.635] | 2.215 | [2.047, 2.384] | 2.571 | [2.347, 2.794] | 3.368 | [2.848, 3.887] |
| 0.92 | 1.577 | [1.527, 1.626] | 2.207 | [2.036, 2.378] | 2.542 | [2.319, 2.765] | 3.381 | [2.815, 3.947] |
| 0.93 | 1.571 | [1.529, 1.612] | 2.211 | [2.047, 2.375] | 2.562 | [2.339, 2.786] | 3.377 | [2.842, 3.912] |
| 0.94 | 1.593 | [1.571, 1.615] | 2.203 | [2.034, 2.372] | 2.506 | [2.288, 2.725] | 3.395 | [2.763, 4.028] |
| 0.95 | n/a | n/a | n/a | n/a | 2.497 | [2.282, 2.713] | 3.399 | [2.75, 4.049] |

### 6.4 Shape estimate at q_c, 0.90 and 0.95, side by side

| Level | q | N_u | ξ̂ | 95% profile CI | lower end open |
|---|---|---|---|---|---|
| q_c | n/a | n/a | n/a | n/a | n/a |
| 0.90 | 0.90 | 200 | 0.1389 | [0.01711, 0.3046] | no |
| 0.95 | 0.95 | 100 | 0.2491 | [0.05886, 0.5208] | no |

### 6.5 Split-half check

| Half | k̂ | q_FS | q | rule failure | \|q − q_c\| |
|---|---|---|---|---|---|
| residuals 1–1000 | 3 | 0.83 | 0.83 | none | n/a |
| residuals 1001–2000 | 0 | 0.80 | 0.80 | none | n/a |

Flag limit: a difference of more than 0.03, or a half without a q. Each half holds residuals from the one full-training GARCH fit.

### 6.6 Dependence in the residual exceedances (diagnostic only)

θ̂ is the Ferro–Segers intervals estimator; clusters come from `decluster_fs` (D36 (c)). The declustered fit is never used for VaR or ES.

| q | N_u | θ̂ | branch | 1/θ̂ | clusters C | run length r | N_u / C | ξ̂ all | 95% CI | ξ̂ declustered | 95% CI | difference | all in decl. CI | decl. in all CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 400 | 0.8393 | T>2 | 1.191 | 298 | 1 | 1.342 | 0.05482 | [-0.02241, 0.1548] | 0.0337 | [-0.052, 0.1477] | -0.02112 | yes | yes |
| 0.81 | 380 | 0.8301 | T>2 | 1.205 | 287 | 1 | 1.324 | 0.07961 | [-0.005089, 0.1894] | 0.05642 | [-0.03598, 0.1795] | -0.02319 | yes | yes |
| 0.82 | 360 | 0.7935 | T>2 | 1.26 | 275 | 1 | 1.309 | 0.07232 | [-0.01211, 0.1823] | 0.05219 | [-0.04037, 0.1758] | -0.02013 | yes | yes |
| 0.83 | 340 | 0.8163 | T>2 | 1.225 | 265 | 1 | 1.283 | 0.08835 | [-0.001783, 0.2062] | 0.06793 | [-0.02949, 0.1982] | -0.02042 | yes | yes |
| 0.84 | 320 | 0.7987 | T>2 | 1.252 | 255 | 1 | 1.255 | 0.07361 | [-0.01464, 0.1896] | 0.06478 | [-0.03296, 0.1958] | -0.008831 | yes | yes |
| 0.85 | 300 | 0.835 | T>2 | 1.198 | 241 | 1 | 1.245 | 0.09956 | [0.002732, 0.2273] | 0.08997 | [-0.01631, 0.233] | -0.009597 | yes | yes |
| 0.86 | 280 | 0.8625 | T>2 | 1.159 | 229 | 1 | 1.223 | 0.08909 | [-0.007356, 0.2171] | 0.08693 | [-0.0203, 0.2317] | -0.002159 | yes | yes |
| 0.87 | 260 | 0.8805 | T>2 | 1.136 | 216 | 1 | 1.204 | 0.1068 | [0.003247, 0.2451] | 0.1135 | [-0.003277, 0.272] | 0.006652 | yes | yes |
| 0.88 | 240 | 0.8483 | T>2 | 1.179 | 200 | 1 | 1.2 | 0.1079 | [0.001172, 0.2513] | 0.1133 | [-0.006796, 0.2774] | 0.005365 | yes | yes |
| 0.89 | 220 | 0.8811 | T>2 | 1.135 | 184 | 1 | 1.196 | 0.1028 | [-0.005771, 0.2495] | 0.1059 | [-0.01581, 0.2731] | 0.003052 | yes | yes |
| 0.90 | 200 | 0.8831 | T>2 | 1.132 | 166 | 1 | 1.205 | 0.1389 | [0.01711, 0.3046] | 0.1265 | [-0.005419, 0.3091] | -0.01241 | yes | yes |
| 0.91 | 180 | 0.9101 | T>2 | 1.099 | 150 | 1 | 1.2 | 0.1356 | [0.01035, 0.3072] | 0.1176 | [-0.01667, 0.3042] | -0.01802 | yes | yes |
| 0.92 | 160 | 0.9607 | T>2 | 1.041 | 138 | 1 | 1.159 | 0.1779 | [0.0355, 0.3751] | 0.178 | [0.0227, 0.3962] | 0.00012 | yes | yes |
| 0.93 | 140 | 1 | T>2 | 1 | 140 | 0 | 1 | 0.149 | [0.007396, 0.346] | 0.149 | [0.007396, 0.346] | 0 | yes | yes |
| 0.94 | 120 | 0.8906 | T>2 | 1.123 | 104 | 1 | 1.154 | 0.2344 | [0.06067, 0.4802] | 0.219 | [0.03365, 0.4853] | -0.01538 | yes | yes |
| 0.95 | 100 | 0.8161 | T>2 | 1.225 | 80 | 3 | 1.25 | 0.2491 | [0.05886, 0.5208] | 0.2215 | [0.01533, 0.5226] | -0.02759 | yes | yes |

Extremal index of the raw losses (Day 7C, `real/selection.json`) next to the residuals:

| q | θ̂ raw losses (7C) | N_u / C raw | θ̂ residuals | N_u / C residuals |
|---|---|---|---|---|
| 0.80 | 0.7066 | 1.695 | 0.8393 | 1.342 |
| 0.81 | 0.7226 | 1.652 | 0.8301 | 1.324 |
| 0.82 | 0.7202 | 1.607 | 0.7935 | 1.309 |
| 0.83 | 0.7167 | 1.596 | 0.8163 | 1.283 |
| 0.84 | 0.7328 | 1.561 | 0.7987 | 1.255 |
| 0.85 | 0.6579 | 1.546 | 0.835 | 1.245 |
| 0.86 | 0.6535 | 1.918 | 0.8625 | 1.223 |
| 0.87 | 0.6162 | 1.844 | 0.8805 | 1.204 |
| 0.88 | 0.6324 | 1.778 | 0.8483 | 1.2 |
| 0.89 | 0.6023 | 1.692 | 0.8811 | 1.196 |
| 0.90 | 0.4988 | 2.128 | 0.8831 | 1.205 |
| 0.91 | 0.5448 | 2 | 0.9101 | 1.2 |
| 0.92 | 0.5423 | 1.928 | 0.9607 | 1.159 |
| 0.93 | 0.5802 | 1.842 | 1 | 1 |
| 0.94 | 0.6189 | 1.69 | 0.8906 | 1.154 |
| 0.95 | 0.6995 | 1.493 | 0.8161 | 1.25 |

## 7. What the numbers show (no decision)

Every number below is copied from the tables above.

**The rule fails on the residuals.**

- ForwardStop rejects all 16 candidates (k̂ = 16), so q_FS and q_c are undefined (`all_rejected`). The command exited with status 1 and chose nothing. Under the plan's fallback (§3) the choice goes to the owner.
- The margin is small. The running statistic is at or below α_FS = 0.10 at two points only: k = 1 (0.02327) and k = 16 (0.09934). The rule takes the largest such k (D34 (5)), so the last value alone carries the rejection of all candidates.
- No single AD p-value is large: the smallest is 0.015 (q = 0.93) and the largest 0.237 (q = 0.81). On the raw losses (Day 7C) the p-values at q ≥ 0.88 were between 0.21 and 0.654; on the residuals they are between 0.015 and 0.118 at the same candidates.
- The p-values are bootstrap estimates with B = 999, so the statistic has Monte Carlo error; its size relative to the 0.0007 margin was not measured (no other seed was run).
- The Day-7C selection on raw losses (q_selected = 0.88) is not affected.

**The GARCH fit.**

- α̂ = 0.1062, β̂ = 0.8551, α̂ + β̂ = 0.9613; no flag; the first start was kept. The implied unconditional SD is 0.01742.
- Residual moments: mean 0.02621, variance 0.9993, skewness 0.06671, excess kurtosis 2.988. The residuals are close to symmetric and heavier-tailed than a normal distribution.

**What the filter removes and what it leaves.**

- ARCH effects: the ARCH-LM p-value goes from 7.645e-20 (raw) to 0.2849 (residuals) at 5 lags and from 8.373e-19 to 0.4755 at 10 lags. Ljung–Box on the squares at lag 10 goes from 1.481e-29 to 0.4618.
- Ljung–Box on the squares at lag 20 is still small on the residuals: p = 0.001158 (statistic 44.84, against 214.1 on the raw losses).
- Serial correlation in the level is not removed. Ljung–Box on z gives p = 5.531e-06 at lag 10 and 7.2e-05 at lag 20; on the raw losses the p-values are 0.00364 and 0.004465. The statistics are larger after standardization (42.74 against 26.08 at lag 10). The filter has a constant mean (D38 (2)), so this is expected in direction; its size is recorded here for the owner.
- Extremal clustering is weaker: θ̂ is 0.7935 to 1 on the residuals against 0.4988 to 0.7328 on the raw losses, and exceedances per cluster are 1 to 1.342 against 1.493 to 2.128.

**Tail shape and residual-scale risk.**

- ξ̂ rises with the threshold, from 0.05482 at q = 0.80 to 0.2491 at q = 0.95. The 95% profile CI contains 0 at seven candidates (0.80–0.84, 0.86, 0.89) and excludes it at the other nine.
- The declustered ξ̂ lies inside the all-exceedance CI at every q, and the reverse also holds; the differences are between −0.02759 and 0.006652.
- VaR99_z is between 2.497 and 2.625 across the grid and ES99_z between 3.328 and 3.399. The N(0,1) values used by GARCH-N are 2.326348 and 2.665214 (methodology §5).
- VaR95_z is between 1.571 and 1.601 and ES95_z between 2.203 and 2.245. The N(0,1) values are 1.644854 and 2.062713.
- The GPD fit converged in every rolling 1000-point window of the residuals at every q (share 1).

**Split halves.**

- Residuals 1–1000: k̂ = 3, q = 0.83. Residuals 1001–2000: k̂ = 0, q = 0.80. Both halves give a q while the full sample gives none.
- The split-half flag is "no" only because it is not evaluated when the full-sample q is undefined (D34 (7)).

**Not produced.** The three per-day residual figures are drawn at q_c only (as Day 7C at q\*); with no q_c, none was written.

**Open, for the owner (D39).** The rule gives no q_c. No value is proposed in this document.

## 8. Limitations of this evidence

1. One GARCH fit on 2000 training losses stands in for the 1000-day rolling refits used out of sample (§3, item 5).
2. QML with a Gaussian likelihood is consistent but not efficient; forecasts will ignore parameter-estimation error; GARCH(1,1) has no leverage term (D38 (7)).
3. Estimated residuals are approximately, not exactly, i.i.d. The A3 simulation evidence for the threshold procedure applies to them only through the i.i.d. cells S1 and S2, and those cells showed the D36 limitation: false selection of 0.88–0.945 and a modal q\* at the grid floor.
4. AD p-values come from B = 999 bootstrap replications; the procedure carries no error-rate guarantee (methodology §7).
5. The recovery study shows a negative bias of β̂ at n = 1000 (§2.1); it is inside the pinned tolerance and is not corrected.

## 9. Reproduce

```powershell
uv run --locked python -m niftytail threshold-select-residuals configs\base.toml
```

About 5 minutes, serial. Two consecutive runs gave byte-identical tracked files. Config sha256 recorded in the JSON: `1e1e1cd4c80df1c3c1ada9881a0787d757503a95fecd4c492e6f4015cb48fe33`.

## 10. Filter specification (D39, Day 8A-2)

**Decision D39 (owner-approved 2026-11-13).**

- **q_c = 0.88, fixed by design.** It equals q_selected, so that GARCH-EVT and POT-GPD differ only in conditioning. It is not data-selected. The rule failure of §6 is recorded as a finding: the GPD fit to the residual tails is weak, and the out-of-sample backtest is the judge. Config: `[conditional] q_c = "0.88"`.
- **Correction to D38 (2).** The reason given for a constant mean (a full-sample lag-1 ACF of r of about 0.05) used the wrong series. §5 shows Ljung–Box(10) on the level of z at 42.74 (p = 5.531e-06), worse than on the raw losses (26.08).
- **The filter is chosen by a rule fixed before any fit** (§10.1). The result is **F4: AR(1) mean with GJR(1,1) variance**. It is the filter of GARCH-N, FHS and GARCH-EVT.
- **AR specs use one pre-window observation** as the lag of the first residual, so a window of W losses still gives W residuals.

**Source of every number in §10.1, §10.2 and §10.4:** `reports/conditional/filter_selection.json` (sha256 `84ee40f40e85e2df4eb0013bbb75482b97f7ed34edbdd44463d979eee3102a12`, `body_sha256` `13bad3ae236e62242860676f09d3dfed6c24e196078863a48b5b2bcec1e5054e`) and `reports/threshold_selection/real_residuals/selection_filter.json` (sha256 `ebe0309b3ad02c1c4ef8616767a332d5ce670e385a1aa7dde8c1662fe5915c84`, `body_sha256` `f0365cadac0846a334141ac827e0af029408c2d931e673dda92b8396b85af57c`). The tables were written by a script that reads those files; values are shown to 4 significant digits (NLL and BIC to 3 decimals). Config sha256 recorded in both: `b5d6bfe39010a42753c2431065418d439d88808339e89becb1b36a8f14c7e3c6`.

### 10.1 The rule and the BIC table

- Candidates: F1 constant mean with GARCH(1,1) (the D38 filter); F2 AR(1) with GARCH(1,1); F3 constant mean with GJR(1,1); F4 AR(1) with GJR(1,1).
- Model: μ_t = φ0 or φ0 + φ1·L_{t−1}; σ²_t = ω + (α + γ·1{ε_{t−1} > 0})·ε²_{t−1} + β·σ²_{t−1}. ε > 0 is a loss above its conditional mean. γ ≥ 0.
- All four are fitted by Gaussian QML on the same conditional sample t = 2…2000 of the training losses: 1999 likelihood terms. L_1 is only the lag of the AR specs.
- BIC = 2·NLL + k·ln(1999). The lowest BIC wins; ties go to the smaller k. No other criterion overrides it.

| spec | mean | variance | k | NLL | BIC | BIC − min | φ0 | φ1 | ω | α | γ | β | α + γ/2 + β | start kept | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | constant | GARCH(1,1) | 4 | -5484.956 | -10939.510 | 17.792 | -0.0006078 | n/a | 1.176e-05 | 0.1063 | n/a | 0.855 | 0.9613 | 1 | none |
| F2 | AR(1) | GARCH(1,1) | 5 | -5493.272 | -10948.542 | 8.761 | -0.0005805 | 0.1006 | 1.214e-05 | 0.1102 | n/a | 0.8497 | 0.9599 | 0 | none |
| F3 | constant | GJR(1,1) | 5 | -5492.374 | -10946.747 | 10.556 | -0.0003966 | n/a | 1.473e-05 | 0.06957 | 0.08899 | 0.836 | 0.95 | 0 | none |
| F4 | AR(1) | GJR(1,1) | 6 | -5501.452 | -10957.302 | 0.000 | -0.0002999 | 0.1053 | 1.513e-05 | 0.06888 | 0.0997 | 0.8304 | 0.9491 | 1 | none |

- F4 has the lowest BIC. The next candidate, F2, is 8.761 higher; F1 is 17.792 higher.
- No candidate has a flag. All four fits converged.
- φ̂1 is 0.1006 (F2) and 0.1053 (F4). γ̂ is 0.08899 (F3) and 0.0997 (F4).
- The F1 row differs slightly from §4 (α̂ 0.1063 against 0.1062) because §4 used all 2000 losses and this table uses the 1999 of the conditional sample.

### 10.2 Diagnostics of the winner, with F1 for contrast

Aggregates only. None of these can override the BIC rule (D39 (e)). Ljung–Box with df = lag; ARCH-LM with df = lags; D29 conventions. Sign-bias tests after Engle and Ng [L38, F], in the convention pinned in D39 (e): z_t² is regressed on a constant and S⁻_{t−1}, S⁻_{t−1}·e_{t−1} or S⁺_{t−1}·e_{t−1}, where e = −ε is the return shock and S⁻ marks a negative return shock; the single tests are t tests (the "lag / df" column gives their residual df) and the joint test is LM = rows·R² with 3 df; 1998 rows.

| Test | lag / df | F4 (winner): statistic | F4: p | F1: statistic | F1: p |
|---|---|---|---|---|---|
| Ljung–Box, level of z | 10 | 21.27 | 0.01927 | 42.46 | 6.221e-06 |
| Ljung–Box, level of z | 20 | 32.29 | 0.04026 | 53.16 | 7.702e-05 |
| Ljung–Box, z² | 10 | 10.78 | 0.3746 | 9.801 | 0.4581 |
| Ljung–Box, z² | 20 | 60.89 | 5.173e-06 | 44.92 | 0.001131 |
| ARCH-LM | 5 | 7.77 | 0.1693 | 6.258 | 0.2819 |
| ARCH-LM | 10 | 10.38 | 0.4078 | 9.658 | 0.471 |
| Sign bias (t) | 1996 | 1.272 | 0.2036 | 2.156 | 0.03122 |
| Negative size bias (t) | 1996 | -0.6909 | 0.4897 | -2.297 | 0.02172 |
| Positive size bias (t) | 1996 | -1.211 | 0.2262 | -2.104 | 0.03549 |
| Sign bias, joint (LM) | 3 | 1.933 | 0.5865 | 7.183 | 0.06628 |

| filter | ACF lag 1 | lag 2 | lag 3 | lag 4 | lag 5 | mean(z) | var(z) | skewness | excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|
| F4 | 0.01371 | -0.004478 | 0.02924 | 0.04696 | 0.03232 | 0.00719 | 0.9998 | 0.02933 | 3.156 |
| F1 | 0.1036 | 0.009215 | 0.03199 | 0.05151 | 0.03088 | 0.02618 | 0.9995 | 0.06741 | 2.989 |

What the numbers show:

- **Level.** The lag-1 ACF of z falls from 0.1036 (F1) to 0.01371 (F4). Ljung–Box on the level goes from p = 6.221e-06 to 0.01927 at lag 10 and from 7.702e-05 to 0.04026 at lag 20. F4 does not reject at 1%; it rejects at 5%.
- **Sign bias.** F1 rejects all three single tests at 5% (p = 0.03122, 0.02172, 0.03549); its joint test has p = 0.06628. F4 rejects none (p = 0.2036, 0.4897, 0.2262; joint 0.5865).
- **Squares, lag 10 and ARCH-LM.** Neither filter rejects.
- **Limitation (D39 (e)): squares, lag 20.** F4 still rejects Ljung–Box on z² at lag 20 at the 1% level: Q = 60.89, p = 5.173e-06. F1 rejects too (44.92, p = 0.001131), and the statistic is larger under F4. This is reported as a limitation. No further specification is added.
- **Moments.** Under F4 the mean of z is 0.00719 (F1: 0.02618), the variance 0.9998, the skewness 0.02933 and the excess kurtosis 3.156 (F1: 2.989). The residuals stay heavy-tailed.

### 10.3 Checks of the new specifications (synthetic data only)

**Recovery study (D39 (f); tolerances pinned before the study was run).** n = 2000, normal innovations, 50 series per spec, true (φ1, α, γ, β) = (0.10, 0.03, 0.08, 0.88) as applicable, ω = 1 − α − γ/2 − β, burn-in 1000. A cell passes if all 50 fits converge and the bias of the mean estimate is within 0.03 (φ1), 0.02 (α), 0.04 (γ) and 0.04 (β). Numbers from `tests/test_garch.py` (`recovery_cell_spec`).

| cell | fits converged | bias φ̂1 | bias α̂ | bias γ̂ | bias β̂ | result |
|---|---|---|---|---|---|---|
| F2, AR(1)-GARCH | 50 / 50 | -0.001982 | -0.0002658 | n/a | -0.01787 | pass |
| F3, constant-GJR | 50 / 50 | n/a | -0.002231 | 0.003188 | -0.001442 | pass |
| F4, AR(1)-GJR | 50 / 50 | 0.001701 | 0.0009242 | 0.004618 | -0.009002 | pass |

**Other new tests (`tests/test_garch.py`).**

- NLL against an independent Python loop for each of the four specs: relative 1e-12.
- Hand-computed 3-observation recursions and one-step forecasts for GJR and for AR(1).
- Constraint maps for every spec: round trip; ω > 0, α > 0, γ > 0, β > 0, α + γ/2 + β < 1 and \|φ1\| < 1 for 2000 random parameter vectors.
- Equivariance for F2, F3, F4: L·c scales φ0, ε and σ by c and leaves φ1, α, γ, β unchanged; L + a gives φ0 + a·(1 − φ1). Relative 1e-6.
- The pinned starts map to their values; an OLS slope outside (−1, 1) drops its two starts.
- The BIC rule: all four fitted on the same conditional sample; ties go to the smaller k.
- **The F1 code path is unchanged:** all Day-8A tests pass unmodified, and the generalised code reproduces the `garch` block, the residual moments and the serial tests of the Day-8A `selection.json` exactly.

**Cross-check against the `arch` package (outside the repository, not a test).** `arch` 9.0.0 in a scratch virtual environment; one simulated series per spec (n = 2000, true (φ1, α, γ, β) = (0.30, 0.08, 0.12, 0.78) as applicable). `arch` is given returns r = −L, mean AR with 1 lag where applicable, and o = 1 for GJR; its indicator 1{e < 0} on the return shock is our 1{ε > 0}. "max diff" is the largest absolute difference over (φ0, φ1, ω, α, γ, β).

| spec | fit | φ0 | φ1 | ω | α | γ | β | NLL | max diff from ours | our NLL at this estimate |
|---|---|---|---|---|---|---|---|---|---|---|
| F2 | `niftytail.garch` | -0.023629 | 0.330366 | 0.136078 | 0.070579 | n/a | 0.795138 | 2830.706726 | | |
| F2 | `arch`, default backcast | -0.023609 | 0.330416 | 0.135957 | 0.070555 | n/a | 0.795260 | 2830.711726 | 1.22e-04 | 2830.706738 |
| F2 | `arch`, backcast fixed at mean(ε²) | -0.023629 | 0.330367 | 0.136076 | 0.070577 | n/a | 0.795140 | 2830.706751 | 2.38e-06 | 2830.706726 |
| F3 | `niftytail.garch` | 0.010952 | n/a | 0.074834 | 0.065680 | 0.112417 | 0.796998 | 2676.461569 | | |
| F3 | `arch`, default backcast | 0.010387 | n/a | 0.074123 | 0.066774 | 0.112072 | 0.796900 | 2672.766336 | 1.09e-03 | 2676.465125 |
| F3 | `arch`, backcast fixed at mean(ε²) | 0.011235 | n/a | 0.074843 | 0.065909 | 0.113014 | 0.796868 | 2676.452327 | 5.97e-04 | 2676.463105 |
| F4 | `niftytail.garch` | 0.019201 | 0.265934 | 0.069473 | 0.050882 | 0.099702 | 0.831683 | 2768.395609 | | |
| F4 | `arch`, default backcast | 0.019434 | 0.265906 | 0.069199 | 0.050565 | 0.099683 | 0.832293 | 2768.142835 | 6.10e-04 | 2768.395884 |
| F4 | `arch`, backcast fixed at mean(ε²) | 0.019154 | 0.265946 | 0.069475 | 0.050869 | 0.099620 | 0.831688 | 2768.397184 | 8.25e-05 | 2768.395638 |

- `arch` maximises a slightly different likelihood in every row: its default backcast is an exponentially weighted average, and with a fixed backcast its first variance is ω + (α + γ/2 + β)·backcast, whereas ours is the backcast itself. The NLL columns are therefore not expected to agree exactly; the GJR rows differ most.
- Our NLL evaluated at each `arch` estimate is above our minimum in all six cases.

### 10.4 Descriptive re-run of the threshold procedure on the F4 residuals

**DESCRIPTIVE ONLY. No decision may cite this run; q_c is fixed by design.** Command `threshold-select-residuals --selected-filter`; output `selection_filter.json` (labelled "DESCRIPTIVE ONLY (D39): residuals of the BIC-selected filter F4, training") and five figures under `real_residuals/selection_filter/`. The Day-8A `selection.json` is unchanged.

- Input: the 1999 standardized residuals of the F4 fit of §10.1 (φ̂1 = 0.1053, γ̂ = 0.0997).
- The procedure accepts exactly `[design] train_size` values, so this run passes a configuration copy with train_size = 1999 (split halves of 999 and 1000 residuals; 1,000 rolling windows in "conv. share"). `threshold.py` is unchanged. This handling was pinned by the implementer in D39 (g) and is flagged there for the owner.
- Seed node as in §3; B = 999.

| Quantity | Value |
|---|---|
| k̂ (candidates rejected by ForwardStop) | 16 |
| q_FS | n/a |
| q from the rule | n/a |
| rule failure | all_rejected |
| final ForwardStop statistic (k = 16) | 0.08588 |
| split-half q, residuals 1–999 | 0.83 (k̂ = 3) |
| split-half q, residuals 1000–1999 | 0.80 (k̂ = 0) |

| q | u (z units) | N_u | ξ̂ | 95% profile CI | AD p | FS stat | rejected | conv. share | first failing stable condition | VaR99_z | ES99_z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.7679 | 399 | 0.0413 | [-0.02998, 0.1351] | 0.006 | 0.006018 | yes | 1 | rejected | 2.585 | 3.258 |
| 0.81 | 0.8029 | 379 | 0.04816 | [-0.0259, 0.1458] | 0.004 | 0.005013 | yes | 1 | rejected | 2.578 | 3.258 |
| 0.82 | 0.8322 | 359 | 0.04704 | [-0.0279, 0.1462] | 0.001 | 0.003676 | yes | 1 | rejected | 2.58 | 3.259 |
| 0.83 | 0.884 | 339 | 0.07179 | [-0.01066, 0.1807] | 0.019 | 0.007552 | yes | 1 | rejected | 2.557 | 3.261 |
| 0.84 | 0.9266 | 319 | 0.08691 | [-0.00102, 0.2033] | 0.061 | 0.01863 | yes | 1 | rejected | 2.546 | 3.266 |
| 0.85 | 0.9638 | 299 | 0.09279 | [0.001547, 0.2141] | 0.033 | 0.02112 | yes | 1 | rejected | 2.541 | 3.268 |
| 0.86 | 1.009 | 279 | 0.11 | [0.01208, 0.2406] | 0.101 | 0.03331 | yes | 1 | rejected | 2.529 | 3.275 |
| 0.87 | 1.043 | 259 | 0.1047 | [0.005875, 0.2373] | 0.025 | 0.03231 | yes | 1 | rejected | 2.533 | 3.274 |
| 0.88 | 1.103 | 239 | 0.1414 | [0.02969, 0.2921] | 0.213 | 0.05534 | yes | 1 | rejected | 2.508 | 3.291 |
| 0.89 | 1.141 | 219 | 0.1345 | [0.02154, 0.288] | 0.112 | 0.06168 | yes | 1 | rejected | 2.513 | 3.287 |
| 0.90 | 1.199 | 199 | 0.1551 | [0.03275, 0.3227] | 0.129 | 0.06863 | yes | 1 | rejected | 2.499 | 3.295 |
| 0.91 | 1.261 | 179 | 0.1808 | [0.04653, 0.3664] | 0.147 | 0.07616 | yes | 1 | rejected | 2.482 | 3.306 |
| 0.92 | 1.318 | 159 | 0.1863 | [0.04474, 0.3842] | 0.117 | 0.07987 | yes | 1 | rejected | 2.479 | 3.309 |
| 0.93 | 1.374 | 139 | 0.1717 | [0.02809, 0.374] | 0.019 | 0.07554 | yes | 1 | rejected | 2.489 | 3.305 |
| 0.94 | 1.47 | 119 | 0.2196 | [0.05368, 0.4557] | 0.05 | 0.07392 | yes | 1 | rejected | 2.457 | 3.314 |
| 0.95 | 1.58 | 99 | 0.3008 | [0.09749, 0.5956] | 0.233 | 0.08588 | yes | 1 | rejected | 2.412 | 3.349 |

- The rule fails again: all 16 candidates are rejected (`all_rejected`). The final ForwardStop statistic is 0.08588 against α_FS = 0.10 (§6: 0.09934).
- AD p-values are between 0.001 and 0.233 (§6: 0.015 to 0.237). The three lowest candidates have p ≤ 0.006.
- ξ̂ rises with the threshold, from 0.0413 at q = 0.80 to 0.3008 at q = 0.95 (§6: 0.05482 to 0.2491).
- At q_c = 0.88: ξ̂ = 0.1414 with 95% profile CI [0.02969, 0.2921], AD p = 0.213, VaR99_z = 2.508, ES99_z = 3.291.
- Across the grid VaR99_z is between 2.412 and 2.585 and ES99_z between 3.258 and 3.349.
- The GPD fit converged in every rolling 1000-point window at every q.
- Changing the filter from F1 to F4 does not change the finding of §7: the GPD fit to the residual tails is weak on the training sample.

### 10.5 Limitations added by D39

1. q_c = 0.88 is not data-selected. The threshold rule rejects every candidate on the residuals of both F1 (§6) and F4 (§10.4).
2. F4 still rejects Ljung–Box on z² at lag 20 at 1% (§10.2). No further specification is added.
3. The specification is chosen once, on the training sample, among four candidates. It is not re-selected out of sample. γ is restricted to γ ≥ 0.
4. BIC compares Gaussian quasi-likelihoods; the residuals are heavy-tailed (excess kurtosis 3.156).
5. The sign-bias convention and the GJR, BIC and Engle–Ng citations are F-status [L38–L40].
6. The descriptive run of §10.4 uses 1999 residuals and a configuration copy with train_size = 1999; it is not comparable observation-for-observation with §6 (2000 residuals).

### 10.6 Reproduce

```powershell
uv run --locked python -m niftytail filter-select configs\base.toml
uv run --locked python -m niftytail threshold-select-residuals configs\base.toml --selected-filter
```

`filter-select` takes under a second; the descriptive run about 4 minutes, serial, and exits with status 1 because of the rule failure. Two consecutive runs of each gave byte-identical tracked files. The §9 command without the flag still runs the Day-8A procedure, but it would now record the new config sha256, so the Day-8A `selection.json` is kept as committed and not regenerated.

## 11. Day 8B: limitation and finding recorded in D40

Numbers copied from `reports/conditional/filter_selection.json`, D39 (a) and §10.4; nothing was recomputed.

**(c) Limitation.** F4 rejects Ljung–Box on z² at lag 20 (p = 5.173e-06) although its persistence α̂ + γ̂/2 + β̂ = 0.9491 is below F1's (0.9613). Interpreted as long-memory volatility beyond GARCH(1,1)-type dynamics. No specification is added.

**(d) Finding.** The GPD fit to the standardized residual tails is poor under both filters: on the D38 (F1) residuals the AD p-values are 0.015–0.24 and ξ̂ rises from 0.055 to 0.249 across the grid; on the F4 residuals (descriptive run, §10.4) the AD p-values are 0.001–0.233 and ξ̂ rises from 0.0413 to 0.3008. It is not a filter artefact. The OOS ES tests judge whether it matters at the primary levels. The descriptive levels 0.995 and 0.999 are flagged as highly q_c-sensitive.

**Interface.** `src/niftytail/models.py` (methodology §18, "`models.py` interface (D40)"). Tests: `tests/test_models.py`, synthetic data only. The test results and the runtime projection are in D40 (i).
