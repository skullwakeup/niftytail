# NIFTYTAIL — A3 threshold-procedure simulation study (Day 7B, D35)

**Every number in this document is simulation evidence (DGPs S1-S4).** It describes how the threshold procedure of D34 behaves on four synthetic data-generating processes. It is not a guarantee, and it says nothing about the Nifty 50 data, which this study never loads.

- **Decision:** D35 (definitions), D34 (the procedure under study), D13 (design and review triggers).
- **Artifact:** `reports/threshold_selection/simstudy/metrics.json` (tracked). Config sha256 `b602725a28e6a3983867ebc83229e0dbcfd20e574f54f417032424eef5cf0e82`; records sha256 `386659d4468c7e19476847687c0a70faaacb766b66e7308186f90aa1f4782d60`.
- **Figures** (tracked SVG, same folder): `fig_a_q_star.svg`, `fig_b_fdr.svg`, `fig_c_xi_error.svg`, `fig_d_risk_error.svg`, `fig_e_failures.svg`.
- **Command:** `uv run --locked python -m niftytail threshold-study configs\base.toml [--workers N]`.
- **The review triggers in §4 are reported, not acted on.** The owner decides.

## 1. Design (D35 summary)

- **Per replication:** one sample of n = 2000 losses; `select_threshold` with B = 199 bootstrap samples per candidate, α_FS = 0.10, q_max = 0.90, without split halves, intervals or the convergence column (D35 (7): these switches change no rule-relevant output). If q\* is defined, the estimates at q\* are recorded, and the model is refitted at q\* on the last 1000 losses.
- **R = 200 replications per cell, 10 cells, 2000 replications in total.**
- **Seeds:** `SeedSequence(20261111).spawn(11)`: children 0–9 are the cells and child 10 is the S4 reference. Replication r of a cell is `cell.spawn(200)[r]`; its two children are the data stream and the GoF seed.
- **Truth:** exact for S1 and S3 (the S3 marginal is the S1 distribution with q0 = 0.90 and ξ = 0.25); closed form for S2, whose ξ = 0.25 is asymptotic only; a simulated reference for S4, which has no exact ξ and no boundary.

| c | DGP | q0 | ξ | a |
|---|---|---|---|---|
| 0 | S1 q0=0.85 xi=0.1 | 0.85 | 0.1 | n/a |
| 1 | S1 q0=0.85 xi=0.25 | 0.85 | 0.25 | n/a |
| 2 | S1 q0=0.85 xi=0.4 | 0.85 | 0.4 | n/a |
| 3 | S1 q0=0.9 xi=0.1 | 0.9 | 0.1 | n/a |
| 4 | S1 q0=0.9 xi=0.25 | 0.9 | 0.25 | n/a |
| 5 | S1 q0=0.9 xi=0.4 | 0.9 | 0.4 | n/a |
| 6 | S2 t(4) | n/a | 0.25 | n/a |
| 7 | S3 a=0.3 | 0.9 | 0.25 | 0.3 |
| 8 | S3 a=0.6 | 0.9 | 0.25 | 0.6 |
| 9 | S4 GARCH t(6) | n/a | n/a | n/a |

q0 is the boundary of S1 and S3: thresholds q ≥ q0 are exactly GPD, thresholds q < q0 are not. "False selection" means q\* < q0.

**S4 reference** (one simulation of 10⁷ losses after a burn-in of 1000; Monte Carlo error from 10 batches of 10⁶):

| Quantity | Value | Batch-means SE | Relative SE |
|---|---|---|---|
| variance | 0.9911 | 0.005692 | 0.005743 |
| var95 | 1.503 | 0.00209 | 0.00139 |
| es95 | 2.277 | 0.005049 | 0.002217 |
| var99 | 2.677 | 0.005505 | 0.002057 |
| es99 | 3.724 | 0.01511 | 0.004057 |

The sample variance is within 3 batch-means SEs of its theoretical value 1.

## 2. S4 generator check and the Day-6 timing figure

D35 required a check because the Day-6 pilot reported 1.3 s for the 10⁷-step S4 simulation, which looked too fast for a Python loop.

- **The Day-6 pilot was correct.** `scripts/timing_pilot.py` runs the exact GARCH(1,1) recursion in a plain Python loop over blocks of 10⁶ innovations. The same loop measured here takes about 100 to 160 ns per step on this machine, so 10⁷ steps take 1 to 1.6 s.
- **The study's generator** (`garch_returns`, used for both the replications and the reference) is the same kind of plain loop. A test compares it with an independent array-indexed loop on the same innovations over 10⁵ steps (relative tolerance 1e-10), and checks that continuing the path in blocks changes nothing.
- **Reference variance:** see the table above.

## 3. Results per cell

### M1. Distribution of q\* and rule failures

Counts of the selected q\* per grid value (out of R = 200), the two kinds of rule failure, and the failure rate (failures / R).

| c | DGP | 0.8 | 0.81 | 0.82 | 0.83 | 0.84 | 0.85 | 0.86 | 0.87 | 0.88 | 0.89 | 0.9 | 0.91 | 0.92 | 0.93 | 0.94 | 0.95 | q* defined | all_rejected | no_stable_region | failure rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | S1 q0=0.85 xi=0.1 | 159 | 11 | 7 | 8 | 1 | 0 | 1 | 3 | 2 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 195 | 1 | 4 | 0.025 |
| 1 | S1 q0=0.85 xi=0.25 | 157 | 11 | 2 | 3 | 9 | 3 | 1 | 2 | 1 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 196 | 0 | 4 | 0.02 |
| 2 | S1 q0=0.85 xi=0.4 | 141 | 16 | 10 | 6 | 7 | 2 | 3 | 0 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 191 | 2 | 7 | 0.045 |
| 3 | S1 q0=0.9 xi=0.1 | 111 | 13 | 10 | 12 | 8 | 9 | 7 | 8 | 6 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 193 | 1 | 6 | 0.035 |
| 4 | S1 q0=0.9 xi=0.25 | 72 | 20 | 22 | 15 | 12 | 14 | 10 | 9 | 10 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 191 | 1 | 8 | 0.045 |
| 5 | S1 q0=0.9 xi=0.4 | 51 | 16 | 11 | 13 | 23 | 13 | 15 | 15 | 12 | 7 | 9 | 0 | 0 | 0 | 0 | 0 | 185 | 0 | 15 | 0.075 |
| 6 | S2 t(4) | 151 | 12 | 3 | 5 | 2 | 6 | 4 | 1 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 190 | 5 | 5 | 0.05 |
| 7 | S3 a=0.3 | 99 | 18 | 22 | 12 | 7 | 6 | 8 | 5 | 3 | 3 | 4 | 0 | 0 | 0 | 0 | 0 | 187 | 1 | 12 | 0.065 |
| 8 | S3 a=0.6 | 105 | 24 | 12 | 11 | 10 | 10 | 10 | 8 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 194 | 1 | 5 | 0.03 |
| 9 | S4 GARCH t(6) | 163 | 5 | 3 | 3 | 5 | 2 | 1 | 5 | 3 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 195 | 3 | 2 | 0.025 |

### M2. False selection (S1 and S3)

- **Primary:** #{q\* defined and q\* < q0} / R. A rule failure counts as "not false".
- **Conditional:** the same count over the replications with q\* defined.

| c | DGP | q* < q0 (count) | primary (/R) | Wilson 95% | q* defined | conditional | Wilson 95% |
|---|---|---|---|---|---|---|---|
| 0 | S1 q0=0.85 xi=0.1 | 186 | 0.93 | [0.8859, 0.9578] | 195 | 0.9538 | [0.9146, 0.9755] |
| 1 | S1 q0=0.85 xi=0.25 | 182 | 0.91 | [0.8622, 0.9423] | 196 | 0.9286 | [0.8837, 0.957] |
| 2 | S1 q0=0.85 xi=0.4 | 180 | 0.9 | [0.8506, 0.9343] | 191 | 0.9424 | [0.8998, 0.9675] |
| 3 | S1 q0=0.9 xi=0.1 | 189 | 0.945 | [0.9042, 0.969] | 193 | 0.9793 | [0.9479, 0.9919] |
| 4 | S1 q0=0.9 xi=0.25 | 189 | 0.945 | [0.9042, 0.969] | 191 | 0.9895 | [0.9626, 0.9971] |
| 5 | S1 q0=0.9 xi=0.4 | 176 | 0.88 | [0.8277, 0.918] | 185 | 0.9514 | [0.9101, 0.9742] |
| 7 | S3 a=0.3 | 183 | 0.915 | [0.8681, 0.9463] | 187 | 0.9786 | [0.9463, 0.9917] |
| 8 | S3 a=0.6 | 193 | 0.965 | [0.9295, 0.9829] | 194 | 0.9948 | [0.9714, 0.9991] |

### M3. Empirical FDR of the ForwardStop stage (S1 and S3)

Mean over all R replications of V / max(1, k̂), where k̂ is the number of rejected candidates and V the number of them with q_k ≥ q0 (true nulls). SD uses ddof = 1.

| c | DGP | FDR mean | SD | mean ± 1.96·SD/√R | > α_FS (0.10)? | > 2α_FS (0.20)? |
|---|---|---|---|---|---|---|
| 0 | S1 q0=0.85 xi=0.1 | 0.03188 | 0.1239 | [0.01471, 0.04905] | no | no |
| 1 | S1 q0=0.85 xi=0.25 | 0.03345 | 0.1225 | [0.01647, 0.05043] | no | no |
| 2 | S1 q0=0.85 xi=0.4 | 0.04208 | 0.1461 | [0.02183, 0.06232] | no | no |
| 3 | S1 q0=0.9 xi=0.1 | 0.007903 | 0.04579 | [0.001556, 0.01425] | no | no |
| 4 | S1 q0=0.9 xi=0.25 | 0.01096 | 0.05489 | [0.003354, 0.01857] | no | no |
| 5 | S1 q0=0.9 xi=0.4 | 0.01244 | 0.05024 | [0.005474, 0.0194] | no | no |
| 7 | S3 a=0.3 | 0.01564 | 0.06268 | [0.006948, 0.02432] | no | no |
| 8 | S3 a=0.6 | 0.00483 | 0.03772 | [-0.000398, 0.01006] | no | no |

### M4. Shape estimate at q\*

Over the replications with q\* defined. For S2 the truth is the asymptotic ξ = 0.25, so "bias" there mixes estimation error with the finite-threshold difference. S4 has no truth.

| c | DGP | n used | truth ξ | kind | mean ξ̂ | SD | bias | RMSE |
|---|---|---|---|---|---|---|---|---|
| 0 | S1 q0=0.85 xi=0.1 | 195 | 0.1 | exact | 0.08223 | 0.06597 | -0.01777 | 0.06816 |
| 1 | S1 q0=0.85 xi=0.25 | 196 | 0.25 | exact | 0.2152 | 0.07415 | -0.0348 | 0.08174 |
| 2 | S1 q0=0.85 xi=0.4 | 191 | 0.4 | exact | 0.3702 | 0.07308 | -0.02985 | 0.07877 |
| 3 | S1 q0=0.9 xi=0.1 | 193 | 0.1 | exact | 0.06084 | 0.06901 | -0.03916 | 0.07919 |
| 4 | S1 q0=0.9 xi=0.25 | 191 | 0.25 | exact | 0.1976 | 0.08719 | -0.05244 | 0.1015 |
| 5 | S1 q0=0.9 xi=0.4 | 185 | 0.4 | exact | 0.3446 | 0.0995 | -0.05535 | 0.1136 |
| 6 | S2 t(4) | 190 | 0.25 | asymptotic | 0.1196 | 0.06938 | -0.1304 | 0.1476 |
| 7 | S3 a=0.3 | 187 | 0.25 | exact | 0.1732 | 0.1099 | -0.07678 | 0.1339 |
| 8 | S3 a=0.6 | 194 | 0.25 | exact | 0.1636 | 0.1382 | -0.08642 | 0.1627 |
| 9 | S4 GARCH t(6) | 195 | n/a | none | 0.09719 | 0.09815 | n/a | n/a |

### M5. VaR and ES at q\*

Relative error = (estimate − truth) / truth. "n=2000" is the fit on the whole sample; "window" is the refit on the last 1000 losses. "used / excluded" counts the replications with q\* defined that have, or lack, a finite estimate (infinite ES at ξ̂ ≥ 1, or a failed refit). For S4 the truth is the reference, and its relative SE is shown.

| c | DGP | quantity | truth | truth rel. SE | n=2000 rel. bias | n=2000 rel. RMSE | used / excluded | window rel. bias | window rel. RMSE | used / excluded |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | S1 q0=0.85 xi=0.1 | var95 | 1.783 | n/a | 0.002566 | 0.03154 | 195 / 0 | 0.006496 | 0.04644 | 195 / 0 |
| 0 | S1 q0=0.85 xi=0.1 | es95 | 2.581 | n/a | -0.001883 | 0.04489 | 195 / 0 | -0.0009497 | 0.06287 | 195 / 0 |
| 0 | S1 q0=0.85 xi=0.1 | var99 | 3.037 | n/a | -0.002802 | 0.04882 | 195 / 0 | -0.002862 | 0.06799 | 195 / 0 |
| 0 | S1 q0=0.85 xi=0.1 | es99 | 3.974 | n/a | -0.008133 | 0.0788 | 195 / 0 | -0.0105 | 0.109 | 195 / 0 |
| 1 | S1 q0=0.85 xi=0.25 | var95 | 1.85 | n/a | 0.008904 | 0.03973 | 196 / 0 | 0.00563 | 0.05418 | 196 / 0 |
| 1 | S1 q0=0.85 xi=0.25 | es95 | 2.979 | n/a | -0.004115 | 0.06472 | 196 / 0 | -0.007156 | 0.08433 | 196 / 0 |
| 1 | S1 q0=0.85 xi=0.25 | var99 | 3.527 | n/a | -0.00361 | 0.06511 | 196 / 0 | -0.008252 | 0.08409 | 196 / 0 |
| 1 | S1 q0=0.85 xi=0.25 | es99 | 5.216 | n/a | -0.02133 | 0.1182 | 196 / 0 | -0.02273 | 0.156 | 196 / 0 |
| 2 | S1 q0=0.85 xi=0.4 | var95 | 1.924 | n/a | 0.007852 | 0.04642 | 191 / 0 | 0.008812 | 0.06049 | 191 / 0 |
| 2 | S1 q0=0.85 xi=0.4 | es95 | 3.588 | n/a | -0.005568 | 0.09733 | 191 / 0 | -0.00512 | 0.1368 | 191 / 0 |
| 2 | S1 q0=0.85 xi=0.4 | var99 | 4.179 | n/a | -0.003551 | 0.08338 | 191 / 0 | -0.005304 | 0.118 | 191 / 0 |
| 2 | S1 q0=0.85 xi=0.4 | es99 | 7.347 | n/a | -0.02001 | 0.1738 | 191 / 0 | -0.01901 | 0.2418 | 191 / 0 |
| 3 | S1 q0=0.9 xi=0.1 | var95 | 1.691 | n/a | 0.007454 | 0.03544 | 193 / 0 | 0.00939 | 0.04582 | 193 / 0 |
| 3 | S1 q0=0.9 xi=0.1 | es95 | 2.369 | n/a | 0.002616 | 0.04335 | 193 / 0 | 0.001643 | 0.0611 | 193 / 0 |
| 3 | S1 q0=0.9 xi=0.1 | var99 | 2.757 | n/a | 0.003658 | 0.04787 | 193 / 0 | 0.002117 | 0.06776 | 193 / 0 |
| 3 | S1 q0=0.9 xi=0.1 | es99 | 3.554 | n/a | -0.009943 | 0.07097 | 193 / 0 | -0.01543 | 0.1002 | 193 / 0 |
| 4 | S1 q0=0.9 xi=0.25 | var95 | 1.713 | n/a | 0.01327 | 0.03891 | 191 / 0 | 0.01389 | 0.04884 | 191 / 0 |
| 4 | S1 q0=0.9 xi=0.25 | es95 | 2.616 | n/a | 0.003709 | 0.05685 | 191 / 0 | -0.002998 | 0.07154 | 191 / 0 |
| 4 | S1 q0=0.9 xi=0.25 | var99 | 3.055 | n/a | 0.007094 | 0.05735 | 191 / 0 | -0.0004036 | 0.07248 | 191 / 0 |
| 4 | S1 q0=0.9 xi=0.25 | es99 | 4.406 | n/a | -0.01645 | 0.1125 | 191 / 0 | -0.03213 | 0.1424 | 191 / 0 |
| 5 | S1 q0=0.9 xi=0.4 | var95 | 1.737 | n/a | 0.01314 | 0.04362 | 185 / 0 | 0.01422 | 0.05165 | 185 / 0 |
| 5 | S1 q0=0.9 xi=0.4 | es95 | 2.99 | n/a | -0.005511 | 0.08278 | 185 / 0 | 0.009844 | 0.2184 | 185 / 0 |
| 5 | S1 q0=0.9 xi=0.4 | var99 | 3.435 | n/a | 0.0002291 | 0.06837 | 185 / 0 | 0.001223 | 0.1001 | 185 / 0 |
| 5 | S1 q0=0.9 xi=0.4 | es99 | 5.821 | n/a | -0.03187 | 0.1725 | 185 / 0 | 0.005929 | 0.5173 | 185 / 0 |
| 6 | S2 t(4) | var95 | 2.132 | n/a | 0.005012 | 0.03465 | 190 / 0 | 0.007538 | 0.05275 | 190 / 0 |
| 6 | S2 t(4) | es95 | 3.203 | n/a | 0.00178 | 0.05185 | 190 / 0 | 0.0002427 | 0.07681 | 190 / 0 |
| 6 | S2 t(4) | var99 | 3.747 | n/a | 0.01436 | 0.05764 | 190 / 0 | 0.01146 | 0.08314 | 190 / 0 |
| 6 | S2 t(4) | es99 | 5.221 | n/a | -0.02011 | 0.09331 | 190 / 0 | -0.02591 | 0.1345 | 190 / 0 |
| 7 | S3 a=0.3 | var95 | 1.713 | n/a | 0.01363 | 0.05833 | 187 / 0 | 0.01401 | 0.069 | 187 / 0 |
| 7 | S3 a=0.3 | es95 | 2.616 | n/a | 3.424e-05 | 0.1118 | 187 / 0 | -0.004395 | 0.1583 | 187 / 0 |
| 7 | S3 a=0.3 | var99 | 3.055 | n/a | 0.002595 | 0.1111 | 187 / 0 | -0.006447 | 0.1417 | 187 / 0 |
| 7 | S3 a=0.3 | es99 | 4.406 | n/a | -0.02532 | 0.2021 | 187 / 0 | -0.0319 | 0.3176 | 187 / 0 |
| 8 | S3 a=0.6 | var95 | 1.713 | n/a | 0.02328 | 0.07636 | 194 / 0 | 0.02303 | 0.1082 | 194 / 0 |
| 8 | S3 a=0.6 | es95 | 2.616 | n/a | 0.01443 | 0.1453 | 194 / 0 | 0.0008855 | 0.1904 | 194 / 0 |
| 8 | S3 a=0.6 | var99 | 3.055 | n/a | 0.0134 | 0.1319 | 194 / 0 | -0.000939 | 0.1832 | 194 / 0 |
| 8 | S3 a=0.6 | es99 | 4.406 | n/a | -0.002834 | 0.2953 | 194 / 0 | -0.03309 | 0.3463 | 194 / 0 |
| 9 | S4 GARCH t(6) | var95 | 1.503 | 0.00139 | 0.01756 | 0.1045 | 195 / 0 | 0.01604 | 0.1552 | 195 / 0 |
| 9 | S4 GARCH t(6) | es95 | 2.277 | 0.002217 | 0.006316 | 0.1542 | 195 / 0 | -0.005312 | 0.2156 | 195 / 0 |
| 9 | S4 GARCH t(6) | var99 | 2.677 | 0.002057 | 0.0135 | 0.1599 | 195 / 0 | -0.0004538 | 0.2227 | 195 / 0 |
| 9 | S4 GARCH t(6) | es99 | 3.724 | 0.004057 | -0.01887 | 0.2328 | 195 / 0 | -0.04156 | 0.3108 | 195 / 0 |

### M6. Failures

- **fit_failed pairs:** (candidate, replication) pairs in which `fit_mle` or scipy's GoF call raised, out of 16 × R.
- **Non-convergence at q\*:** of the n = 2000 fit (cannot occur when q\* is defined) and of the window refit.
- **P(ξ̂ ≥ 0.5), P(ξ̂ ≥ 1):** over the replications with q\* defined.

| c | DGP | fit_failed pairs | share | n=2000 non-conv. at q* | window non-conv. | rate | P(ξ̂ ≥ 0.5) n=2000 | P(ξ̂ ≥ 1) n=2000 | P(ξ̂ ≥ 0.5) window | P(ξ̂ ≥ 1) window |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | S1 q0=0.85 xi=0.1 | 0 / 3200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | S1 q0=0.85 xi=0.25 | 0 / 3200 | 0 | 0 | 0 | 0 | 0 | 0 | 0.005102 | 0 |
| 2 | S1 q0=0.85 xi=0.4 | 0 / 3200 | 0 | 0 | 0 | 0 | 0.04188 | 0 | 0.07853 | 0 |
| 3 | S1 q0=0.9 xi=0.1 | 0 / 3200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | S1 q0=0.9 xi=0.25 | 0 / 3200 | 0 | 0 | 0 | 0 | 0 | 0 | 0.005236 | 0 |
| 5 | S1 q0=0.9 xi=0.4 | 0 / 3200 | 0 | 0 | 0 | 0 | 0.08108 | 0 | 0.0973 | 0 |
| 6 | S2 t(4) | 0 / 3200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | S3 a=0.3 | 0 / 3200 | 0 | 0 | 0 | 0 | 0.005348 | 0 | 0.02139 | 0 |
| 8 | S3 a=0.6 | 0 / 3200 | 0 | 0 | 0 | 0 | 0.01031 | 0 | 0.02577 | 0 |
| 9 | S4 GARCH t(6) | 0 / 3200 | 0 | 0 | 0 | 0 | 0 | 0 | 0.005128 | 0 |

## 4. Review triggers (D13; D35 (5))

Evaluated for both S3 cells with strict ">" in exact rational arithmetic. **They are reported here and never act automatically.** "Near miss" is set only when the trigger does not fire: for false selection, when the conditional value is above the line; for FDR, when the upper end of the interval is above the line.

| Cell | Trigger | Value | 95% interval | Fires if | Result | Near miss |
|---|---|---|---|---|---|---|
| S3 a=0.3 | false selection, primary P(q* < q0) | 0.915 | Wilson [0.8681, 0.9463] | > 0.2 | **FIRED** | no |
| S3 a=0.3 | false selection, conditional on q* defined (not a trigger) | 0.9786 | Wilson [0.9463, 0.9917] | n/a | n/a | n/a |
| S3 a=0.3 | empirical FDR (mean) | 0.01564 | MC [0.006948, 0.02432] | > 0.2 | NOT FIRED | no |
| S3 a=0.3 | rule-failure rate | 0.065 | n/a | > 0.1 | NOT FIRED | n/a |
| S3 a=0.6 | false selection, primary P(q* < q0) | 0.965 | Wilson [0.9295, 0.9829] | > 0.2 | **FIRED** | no |
| S3 a=0.6 | false selection, conditional on q* defined (not a trigger) | 0.9948 | Wilson [0.9714, 0.9991] | n/a | n/a | n/a |
| S3 a=0.6 | empirical FDR (mean) | 0.00483 | MC [-0.000398, 0.01006] | > 0.2 | NOT FIRED | no |
| S3 a=0.6 | rule-failure rate | 0.03 | n/a | > 0.1 | NOT FIRED | n/a |

**Outcome.**
- **False selection: FIRED in both S3 cells** (0.915 for a = 0.3 and 0.965 for a = 0.6, against the line 0.20). The whole Wilson interval lies far above the line in both cells.
- **Empirical FDR: NOT FIRED** in either cell (0.01564 and 0.00483, against 0.20). No near miss.
- **Rule-failure rate: NOT FIRED** in either cell (0.065 and 0.03, against 0.10).

**What follows (plan A3; D13).** A fired trigger is documented as a limitation, and the owner chooses among: (i) proceed as specified; (ii) raise α_FS; (iii) apply the GoF test to declustered cluster maxima. Any change to the procedure re-runs this study before the real training data is touched. **No choice is made here, and Day 7C has not started.**

**Decision recorded later (D36, 2026-11-12).** The sentence above is the state at the Day-7B commit. On 2026-11-12 the owner chose **option (i), proceed as specified**: α_FS stays 0.10, the procedure is unchanged and this study is not re-run. Options (ii) and (iii) were not adopted; the reasons and the limitation that goes with the decision are in `decisions.md` D36 and `docs/threshold_selection.md`.

## 5. What the numbers show

These are readings of the tables above. They are not decisions.

1. **False selection is very high in every cell that has a boundary, not only under dependence.** The primary rate is between 0.88 and 0.965 in the six i.i.d. S1 cells and the two S3 cells. The S1 cell with the same marginal as S3 (c = 4) has 0.945, against 0.915 and 0.965 in S3. On this evidence the result is a property of the procedure on this kind of marginal, and dependence changes it little.
2. **The lowest grid value is the most frequent choice in every cell.** q\* = 0.80 is selected in 51 to 163 of 200 replications. In the q0 = 0.90 cells the selected q\* spreads over 0.80 to 0.90 but rarely reaches the boundary (1 to 9 of 200 at q\* = 0.90).
3. **The ForwardStop stage rejects few candidates.** The empirical FDR is far below α_FS = 0.10 in every cell (0.0048 to 0.042), so true nulls are almost never rejected. For q\* ≥ q0, however, ForwardStop would have to reject every candidate below q0 (5 of them for q0 = 0.85, 10 for q0 = 0.90), and it rarely does.
4. **Rule failures are infrequent:** 2% to 7.5% per cell, and below the 10% line in both S3 cells. Over all cells, 68 of the 83 failures are "no_stable_region" and 15 are "all_rejected".
5. **ξ̂(q\*) is biased downwards in every cell with a known ξ:** −0.018 to −0.055 in S1 and −0.077 and −0.086 in S3, with RMSE 0.068 to 0.114 in S1 and 0.134 and 0.163 in S3. In S2 the mean ξ̂ is 0.12 against the asymptotic 0.25.
6. **VaR and ES are nearly unbiased despite the false selection.** For the n = 2000 fit the relative bias is within ±2.4% for every VaR and within ±3.2% for every ES, in every cell. The relative RMSE grows with the level, with ξ and with dependence: for VaR99 it is 4.8% to 8.3% in S1, 5.8% in S2, 11.1% and 13.2% in S3 and 16.0% in S4; for ES99 it is 7.1% to 17.4% in S1, 9.3% in S2, 20.2% and 29.5% in S3 and 23.3% in S4.
7. **The 1000-point window refit is noisier, as expected.** Its relative RMSE is larger than the n = 2000 value in every row. The largest values are for ES99: 0.517 in S1 (q0 = 0.90, ξ = 0.40), 0.318 and 0.346 in S3, and 0.311 in S4.
8. **Dependence roughly doubles the estimation error at the same marginal.** Against the S1 cell c = 4, the VaR99 relative RMSE rises from 0.057 to 0.111 (a = 0.3) and 0.132 (a = 0.6), and the ES99 relative RMSE from 0.113 to 0.202 and 0.295.
9. **No estimation failures occurred.** `fit_failed` is 0 of 32,000 (candidate, replication) pairs, every window refit converged, and no ξ̂ reached 1. P(ξ̂ ≥ 0.5) is at most 8.1% (n = 2000) and 9.7% (window), both in the ξ = 0.40 cells.

**Side check (not part of `metrics.json`; 40 samples per cell with the study's seeds, B = 199).** The share of samples with an AD p-value ≤ 0.10 at a threshold below the boundary was 0.60 at q = 0.80 and 0.20 at q = 0.85 in the S1 cell c = 4, and 0.45 and 0.125 in the S3 cell a = 0.3. The AD test therefore has moderate power at the lowest candidate and little power closer to the boundary, which is consistent with point 3. This check is small and was not pre-declared; it is given only to show that the false-selection rates are not an implementation artefact.

## 6. Limitations

- **Monte Carlo error at R = 200.** A proportion near 0.2 has a 95% interval of about ±0.055. Cells are compared with the trigger lines by their point estimates, as pre-declared; the intervals are shown so that close calls are visible.
- **B = 199 in the study, B = 999 in the real selection (D22).** The smallest attainable p-value is 1/200 = 0.005 here and 0.001 there. ForwardStop sees slightly different p-values in the real selection; the study does not measure B = 999.
- **Dependence is tested only through S3 and S4.** S3 has clustered extremes with an exact marginal; S4 has volatility clustering. Other forms of dependence, and non-stationarity, are untested.
- **S4 has no exact ξ and no boundary.** False selection and FDR are undefined there. Its VaR/ES truth is a simulated reference with the Monte Carlo error shown in §1.
- **Truth is the marginal distribution.** For S3 and S4 the VaR and ES targets are unconditional; the study says nothing about conditional risk.
- **S2's ξ = 0.25 is asymptotic.** At q between 0.80 and 0.90 the excesses of a t(4) are not exactly GPD, so a non-zero "bias" there is expected.
- **The window refit uses the q\* selected on the full sample** and is not a rolling out-of-sample exercise.
- **No theoretical guarantee** is claimed for ForwardStop with nested thresholds (plan A3). These are measurements on four processes.

## 7. Run record

**Hardware and software:** Windows-11-10.0.26300-SP0; AMD64 (family 26, model 68); `os.cpu_count()` = 32; Python 3.13.17; numpy 2.5.4; scipy 1.18.1; matplotlib 3.11.2. BLAS/OpenMP threads pinned to 1.

| Run | Workers | Pilot (2 replications per cell) | Projection = pilot / 2 × 200 | Guard (2 h) | Total wall-clock, pilot included | Exit |
|---|---|---|---|---|---|---|
| 1 | 31 (default) | 26.03 s | 2,603 s = 0.72 h | pass | 1,820.7 s = 30.3 min | 0 |
| 2 | 16 | 46.97 s | 4,697 s = 1.30 h | pass | 3,040.3 s = 50.7 min | 0 |

**Byte identity across the two runs** (different worker counts): all six files are identical.

| File | sha256 (both runs) |
|---|---|
| `metrics.json` | `cfa424e0fa1049eca51fc98d37c00fa66aca285d699d4673ff38274b003212b5` |
| `fig_a_q_star.svg` | `43f67664537d755187adc18882aafa930b3cc9d7f668514c273a3a9bbb43ad8f` |
| `fig_b_fdr.svg` | `b5280855c490ecf5d21c0069e1a43979412adc77d32d93561027b5f0f4bff793` |
| `fig_c_xi_error.svg` | `c4f5bc0fd07739928f6c3172d0c7fd590ebccb3396a22694b092f14d9caa3423` |
| `fig_d_risk_error.svg` | `7af8f34c1a2d6f0bb9f5ddc31e85c89923a70461c51c00b95432b547ba057fe2` |
| `fig_e_failures.svg` | `21edc0a93b8ff8f9533eb3c4ad0f2dc738b7801f578e04385489317aef646273` |

**Notes.**
- The projection overestimates: the 20 pilot tasks run at once, so the pilot measures one task's duration, not the throughput of a full pool. Both projections stayed under the guard.
- A short side check (§5) ran on the same machine during the first minutes of run 2, so run 2's time is slightly inflated.
- **Figures are tracked as SVG.** `tests/test_repo_hygiene.py` and `.gitignore` forbid tracked market data and tabular files, not images, and `reports/threshold_selection/` is declared committable. The figures show aggregates of synthetic data only. They are written without a date, with a fixed `svg.hashsalt` and LF line endings. No PNG is tracked.
- The per-replication records are not stored; `records_sha256` in `metrics.json` identifies them.
