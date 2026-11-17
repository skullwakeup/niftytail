# NIFTYTAIL — Real threshold selection on the training slice (Day 7C, D36)

**Status:** the owner signed off on 2026-11-12 (D37, §12): `q_selected = "0.88"` in `configs/base.toml`. Sections 1–11 are the Day-7C-1 evidence, unchanged.

**Source of every number below:** `reports/threshold_selection/real/selection.json` (file sha256 `7929c129a03509cb31b56cea1804cbfd4559e99c09b960d86b658b86e86741f9`, `body_sha256` `8fbba84e4422fab8c48542858c605e1b9b60f545cb1f284f10c01ce1583fa613`). The tables were written by a script that reads that file; values are shown to 4 significant digits. Losses are L = −r in log-return units (0.01 = 1%).

**Sample:** the first 2000 log returns (training slice), dated 1995-11-06 to 2003-11-04. No later return was computed.

## 1. Method recap

1. Grid q = 0.80, 0.81, …, 0.95 (16 candidates); u_k = L_(⌈n·q_k⌉) of the training losses; N_u = number of losses > u_k.
2. Per candidate: GPD fit by maximum likelihood (`fit_mle`), profile CI for ξ, VaR and ES at 95% and 99% with delta-method CIs.
3. Per candidate: Anderson–Darling p-value by parametric bootstrap (`scipy.stats.goodness_of_fit`, loc = 0, B = 999).
4. ForwardStop with α_FS = 0.10: k̂ = max{k : −(1/k)·Σ_(i≤k) ln(1 − p_i) ≤ α_FS}; the first k̂ candidates are rejected; q_FS = q_(k̂+1).
5. Stable region: q\* = the smallest q ≥ q_FS with q ≤ q_max = 0.90 whose window [q, q + 0.03] (4 grid points) keeps \|Δξ̂\| ≤ 0.10, \|ΔVaR99\| ≤ 5% and \|ΔES99\| ≤ 10% relative to q.
6. Split-half: steps 1–5 on returns 1–1000 and 1001–2000; flag if a half has no q or differs from q\* by more than 0.03.
7. No q qualifies → rule failure; nothing is chosen automatically.
8. Seeds: SeedSequence(20261111), children as in D34 (4). BLAS threads = 1. Definitions: D34; this run: D36.
9. Diagnostics with no influence on q\*: extremal index θ̂ (Ferro–Segers intervals estimator), intervals declustering (`decluster_fs`, D36 (c)), declustered-ξ comparison, mean residual life.
10. The procedure carries no error-rate guarantee (methodology §7); its measured behaviour and the limitation that follows are in §7 below.

## 2. Result

| Quantity | Value |
|---|---|
| k̂ (candidates rejected by ForwardStop) | 8 |
| q_FS | 0.88 |
| **q\*** | **0.88** |
| rule failure | none |
| q_star_at_grid_floor | no |
| split-half q, returns 1–1000 | n/a (rule failure: no_stable_region) |
| split-half q, returns 1001–2000 | 0.80 |
| split-half flag | yes |
| flags | split_half_flag |

### 2.1 Candidate table: fit, goodness of fit, ForwardStop, stable region

"FS stat" is the running ForwardStop statistic. "N_u window" is (1 − q)·W for W = 1000. "conv. share" is the share of the 1,001 rolling windows inside the training slice in which the fit converges.

| q | u | N_u | N_u window | ξ̂ | SE | 95% profile CI | β̂ | β\* | AD stat | AD p | FS stat | rejected | conv. share | first failing stable condition | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.01138 | 400 | 200 | 0.07681 | 0.05384 | [-0.01553, 0.1914] | 0.009648 | 0.008774 | 0.6131 | 0.199 | 0.2219 | yes | 1 | rejected | none |
| 0.81 | 0.0118 | 380 | 190 | 0.06876 | 0.05483 | [-0.02396, 0.1842] | 0.00984 | 0.009028 | 0.8644 | 0.064 | 0.144 | yes | 1 | rejected | none |
| 0.82 | 0.01252 | 360 | 180 | 0.09021 | 0.05746 | [-0.009386, 0.2147] | 0.009474 | 0.008345 | 0.7571 | 0.116 | 0.1371 | yes | 1 | rejected | none |
| 0.83 | 0.01322 | 340 | 170 | 0.1115 | 0.06028 | [0.00443, 0.246] | 0.009154 | 0.007679 | 1.122 | 0.029 | 0.1102 | yes | 1 | rejected | none |
| 0.84 | 0.01364 | 320 | 160 | 0.09353 | 0.06113 | [-0.01259, 0.2275] | 0.009533 | 0.008257 | 1.137 | 0.025 | 0.09322 | yes | 1 | rejected | none |
| 0.85 | 0.01401 | 300 | 150 | 0.06627 | 0.06156 | [-0.03683, 0.197] | 0.01011 | 0.009181 | 1.018 | 0.029 | 0.08258 | yes | 1 | rejected | none |
| 0.86 | 0.01466 | 280 | 140 | 0.06029 | 0.06336 | [-0.04453, 0.1936] | 0.01027 | 0.009389 | 1.123 | 0.025 | 0.0744 | yes | 1 | rejected | none |
| 0.87 | 0.01571 | 260 | 130 | 0.0884 | 0.0675 | [-0.02594, 0.234] | 0.009754 | 0.008365 | 0.7616 | 0.115 | 0.08037 | yes | 1 | rejected | none |
| 0.88 | 0.01671 | 240 | 120 | 0.1157 | 0.07202 | [-0.009205, 0.2753] | 0.009331 | 0.007398 | 0.5253 | 0.282 | 0.1083 | no | 1 | none (window passes) | none |
| 0.89 | 0.01759 | 220 | 110 | 0.1252 | 0.07586 | [-0.007259, 0.2952] | 0.009262 | 0.00706 | 0.5763 | 0.21 | 0.121 | no | 1 | tol_xi at q = 0.92 | none |
| 0.90 | 0.01859 | 200 | 100 | 0.1423 | 0.08077 | [-0.0007109, 0.3268] | 0.009093 | 0.006447 | 0.5596 | 0.233 | 0.1341 | no | 1 | tol_xi at q = 0.92 | none |
| 0.91 | 0.01972 | 180 | 90 | 0.1695 | 0.08717 | [0.01164, 0.3743] | 0.008805 | 0.005462 | 0.5381 | 0.246 | 0.1465 | no | 1 | above_q_max | below_n_min_window |
| 0.92 | 0.02124 | 160 | 80 | 0.2684 | 0.1003 | [0.07273, 0.5252] | 0.007659 | 0.001959 | 0.3326 | 0.596 | 0.2049 | no | 1 | above_q_max | below_n_min_window |
| 0.93 | 0.02217 | 140 | 70 | 0.2357 | 0.1044 | [0.03209, 0.5083] | 0.008335 | 0.00311 | 0.3228 | 0.607 | 0.257 | no | 1 | above_q_max | below_n_min_window |
| 0.94 | 0.02314 | 120 | 60 | 0.1652 | 0.1064 | [-0.03779, 0.4432] | 0.00964 | 0.005817 | 0.3026 | 0.654 | 0.3106 | no | 1 | above_q_max | below_n_min_window |
| 0.95 | 0.02483 | 100 | 50 | 0.1547 | 0.1155 | [-0.07103, 0.4747] | 0.01012 | 0.00628 | 0.3454 | 0.567 | 0.3435 | no | 1 | above_q_max | below_n_min_window, var95_not_in_tail |

### 2.2 Candidate table: VaR and ES (n = 2000 fit, delta-method 95% CIs)

At q = 0.95 the 95% level is not inside the tail (p ≥ N_u/n), so VaR95 and ES95 are not defined there (flag `var95_not_in_tail`).

| q | VaR95 | 95% CI | ES95 | 95% CI | VaR99 | 95% CI | ES99 | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.02549 | [0.02397, 0.02701] | 0.03711 | [0.03399, 0.04024] | 0.04387 | [0.03984, 0.04791] | 0.05703 | [0.04893, 0.06513] |
| 0.81 | 0.02556 | [0.02403, 0.02708] | 0.03714 | [0.03406, 0.04022] | 0.04391 | [0.03992, 0.04791] | 0.05685 | [0.04891, 0.0648] |
| 0.82 | 0.02539 | [0.02389, 0.02688] | 0.03707 | [0.03392, 0.04023] | 0.04381 | [0.03972, 0.04789] | 0.05732 | [0.0489, 0.06575] |
| 0.83 | 0.02523 | [0.02376, 0.02669] | 0.03704 | [0.0338, 0.04028] | 0.04372 | [0.03955, 0.0479] | 0.05786 | [0.04892, 0.0668] |
| 0.84 | 0.02535 | [0.02388, 0.02682] | 0.03707 | [0.03392, 0.04022] | 0.04381 | [0.03973, 0.04789] | 0.05744 | [0.04889, 0.06598] |
| 0.85 | 0.02553 | [0.02405, 0.02702] | 0.03718 | [0.03414, 0.04021] | 0.044 | [0.04003, 0.04797] | 0.05696 | [0.04896, 0.06495] |
| 0.86 | 0.02557 | [0.02411, 0.02704] | 0.03721 | [0.03421, 0.04021] | 0.04405 | [0.04011, 0.04799] | 0.05687 | [0.04898, 0.06475] |
| 0.87 | 0.02543 | [0.02403, 0.02683] | 0.03707 | [0.03402, 0.04013] | 0.04379 | [0.03978, 0.04779] | 0.05721 | [0.04879, 0.06563] |
| 0.88 | 0.0253 | [0.02398, 0.02663] | 0.03698 | [0.03386, 0.0401] | 0.04357 | [0.0395, 0.04764] | 0.05763 | [0.04863, 0.06664] |
| 0.89 | 0.02527 | [0.02401, 0.02653] | 0.03695 | [0.03383, 0.04008] | 0.0435 | [0.03941, 0.04758] | 0.05779 | [0.04856, 0.06702] |
| 0.90 | 0.02522 | [0.02404, 0.02639] | 0.03692 | [0.03377, 0.04006] | 0.04337 | [0.03925, 0.04748] | 0.05808 | [0.04845, 0.06771] |
| 0.91 | 0.02517 | [0.02411, 0.02622] | 0.03688 | [0.03368, 0.04008] | 0.04316 | [0.03901, 0.04732] | 0.05855 | [0.04825, 0.06885] |
| 0.92 | 0.02508 | [0.02423, 0.02592] | 0.03695 | [0.03331, 0.04059] | 0.04257 | [0.03822, 0.04691] | 0.06086 | [0.04747, 0.07424] |
| 0.93 | 0.02508 | [0.02438, 0.02578] | 0.03689 | [0.03351, 0.04027] | 0.04274 | [0.03849, 0.047] | 0.06 | [0.04772, 0.07227] |
| 0.94 | 0.02493 | [0.02446, 0.02539] | 0.03683 | [0.03383, 0.03983] | 0.04324 | [0.03908, 0.04741] | 0.05877 | [0.04845, 0.06909] |
| 0.95 | n/a | n/a | n/a | n/a | 0.04333 | [0.03918, 0.04748] | 0.05868 | [0.04864, 0.06873] |

## 3. Shape estimate at q\*, 0.90 and 0.95, side by side

ξ̂(q\*) is not to be read alone. From the candidate table; no new fits.

| Level | q | N_u | ξ̂ | 95% profile CI | lower end open |
|---|---|---|---|---|---|
| q\* | 0.88 | 240 | 0.1157 | [-0.009205, 0.2753] | no |
| 0.90 | 0.90 | 200 | 0.1423 | [-0.0007109, 0.3268] | no |
| 0.95 | 0.95 | 100 | 0.1547 | [-0.07103, 0.4747] | no |

## 4. Split-half check

| Half | k̂ | q_FS | q | rule failure | \|q − q\*\| |
|---|---|---|---|---|---|
| returns 1–1000 | 11 | 0.91 | n/a | no_stable_region | n/a |
| returns 1001–2000 | 0 | 0.80 | 0.80 | none | 0.08 |

Flag limit: a difference of more than 0.03, or a half without a q. **Split-half flag: yes.**

## 5. Dependence: extremal index and declustered comparison (diagnostic only)

θ̂ is the Ferro–Segers intervals estimator; 1/θ̂ is the implied mean cluster size. Clusters come from `decluster_fs` (D36 (c)): C clusters separated by the inter-exceedance times longer than r. "ξ̂ declustered" is a GPD fit to the cluster maxima's excesses over the same u_k, attempted only with at least 30 cluster maxima (a descriptive cut). **The declustered fit is never used for VaR or ES** (methodology §9). There is no pass/fail rule.

| q | N_u | θ̂ | branch | 1/θ̂ | clusters C | run length r | N_u / C | ξ̂ all | 95% CI | ξ̂ declustered | 95% CI | difference | all in decl. CI | decl. in all CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 400 | 0.7066 | T>2 | 1.415 | 236 | 2 | 1.695 | 0.07681 | [-0.01553, 0.1914] | 0.03232 | [-0.08375, 0.1842] | -0.04449 | yes | yes |
| 0.81 | 380 | 0.7226 | T>2 | 1.384 | 230 | 2 | 1.652 | 0.06876 | [-0.02396, 0.1842] | 0.03152 | [-0.08518, 0.1842] | -0.03724 | yes | yes |
| 0.82 | 360 | 0.7202 | T>2 | 1.389 | 224 | 2 | 1.607 | 0.09021 | [-0.009386, 0.2147] | 0.04325 | [-0.07754, 0.202] | -0.04696 | yes | yes |
| 0.83 | 340 | 0.7167 | T>2 | 1.395 | 213 | 2 | 1.596 | 0.1115 | [0.00443, 0.246] | 0.05326 | [-0.07309, 0.2201] | -0.05827 | yes | yes |
| 0.84 | 320 | 0.7328 | T>2 | 1.365 | 205 | 2 | 1.561 | 0.09353 | [-0.01259, 0.2275] | 0.03911 | [-0.08547, 0.2038] | -0.05442 | yes | yes |
| 0.85 | 300 | 0.6579 | T>2 | 1.52 | 194 | 2 | 1.546 | 0.06627 | [-0.03683, 0.197] | 0.01383 | [-0.1078, 0.1748] | -0.05244 | yes | yes |
| 0.86 | 280 | 0.6535 | T>2 | 1.53 | 146 | 3 | 1.918 | 0.06029 | [-0.04453, 0.1936] | 0.02904 | [-0.1145, 0.2212] | -0.03124 | yes | yes |
| 0.87 | 260 | 0.6162 | T>2 | 1.623 | 141 | 3 | 1.844 | 0.0884 | [-0.02594, 0.234] | 0.06733 | [-0.09067, 0.2794] | -0.02108 | yes | yes |
| 0.88 | 240 | 0.6324 | T>2 | 1.581 | 135 | 3 | 1.778 | 0.1157 | [-0.009205, 0.2753] | 0.0872 | [-0.07925, 0.3117] | -0.02849 | yes | yes |
| 0.89 | 220 | 0.6023 | T>2 | 1.66 | 130 | 3 | 1.692 | 0.1252 | [-0.007259, 0.2952] | 0.129 | [-0.05131, 0.3731] | 0.003781 | yes | yes |
| 0.90 | 200 | 0.4988 | T>2 | 2.005 | 94 | 5 | 2.128 | 0.1423 | [-0.0007109, 0.3268] | 0.09733 | [-0.1042, 0.3772] | -0.04497 | yes | yes |
| 0.91 | 180 | 0.5448 | T>2 | 1.836 | 90 | 5 | 2 | 0.1695 | [0.01164, 0.3743] | 0.1247 | [-0.09004, 0.4252] | -0.04481 | yes | yes |
| 0.92 | 160 | 0.5423 | T>2 | 1.844 | 83 | 5 | 1.928 | 0.2684 | [0.07273, 0.5252] | 0.1725 | [-0.07368, 0.5233] | -0.0959 | yes | yes |
| 0.93 | 140 | 0.5802 | T>2 | 1.724 | 76 | 5 | 1.842 | 0.2357 | [0.03209, 0.5083] | 0.1516 | [-0.1021, 0.5139] | -0.08411 | yes | yes |
| 0.94 | 120 | 0.6189 | T>2 | 1.616 | 71 | 5 | 1.69 | 0.1652 | [-0.03779, 0.4432] | 0.1385 | [-0.1202, 0.5057] | -0.02668 | yes | yes |
| 0.95 | 100 | 0.6995 | T>2 | 1.43 | 67 | 4 | 1.493 | 0.1547 | [-0.07103, 0.4747] | 0.2188 | [-0.07173, 0.6365] | 0.06411 | yes | yes |

At q\* = 0.88: θ̂ = 0.6324, 1/θ̂ = 1.581, 135 clusters from 240 exceedances (1.778 per cluster, r = 3); ξ̂ all = 0.1157 [-0.009205, 0.2753], ξ̂ declustered = 0.0872 [-0.07925, 0.3117], difference -0.02849.

## 6. Mean residual life over the grid thresholds

Mean excess e(u_k) = mean(L − u_k \| L > u_k). The band uses the i.i.d. formula and is descriptive only.

| q | u | N_u | mean excess | ±1.96 SE band |
|---|---|---|---|---|
| 0.80 | 0.01138 | 400 | 0.01045 | [0.009336, 0.01157] |
| 0.81 | 0.0118 | 380 | 0.01057 | [0.009419, 0.01172] |
| 0.82 | 0.01252 | 360 | 0.01041 | [0.009228, 0.0116] |
| 0.83 | 0.01322 | 340 | 0.0103 | [0.009071, 0.01152] |
| 0.84 | 0.01364 | 320 | 0.01051 | [0.009243, 0.01179] |
| 0.85 | 0.01401 | 300 | 0.01083 | [0.009508, 0.01215] |
| 0.86 | 0.01466 | 280 | 0.01093 | [0.009561, 0.01231] |
| 0.87 | 0.01571 | 260 | 0.0107 | [0.009268, 0.01213] |
| 0.88 | 0.01671 | 240 | 0.01055 | [0.009044, 0.01205] |
| 0.89 | 0.01759 | 220 | 0.01058 | [0.008995, 0.01216] |
| 0.90 | 0.01859 | 200 | 0.01058 | [0.008904, 0.01226] |
| 0.91 | 0.01972 | 180 | 0.01056 | [0.008769, 0.01236] |
| 0.92 | 0.02124 | 160 | 0.01028 | [0.008339, 0.01221] |
| 0.93 | 0.02217 | 140 | 0.01076 | [0.008645, 0.01287] |
| 0.94 | 0.02314 | 120 | 0.01148 | [0.009156, 0.01381] |
| 0.95 | 0.02483 | 100 | 0.01191 | [0.009304, 0.01451] |

## 7. Limitation that goes with this selection (D36)

On 2026-11-12 the owner chose option (i) for the fired A3 false-selection trigger: proceed as specified. α_FS stays 0.10, the procedure is unchanged and the A3 study was not re-run. The following is recorded with that decision (numbers from `reports/threshold_selection/simstudy/metrics.json`; details in `docs/threshold_study.md`):

- **False selection.** P(q\* < q0) is 0.88–0.965 in every S1 and S3 cell of the study, i.i.d. cells included. The FDR and rule-failure triggers did not fire. The cause is not established by the pinned metrics.
- **Grid floor.** q = 0.80 is the most frequent q\* in every cell of the study (51–163 of 200 replications), including S2 and S4, which have no boundary. A real q\* equal to the floor would be reported as "no evidence against the floor", not as a located threshold. **Here `q_star_at_grid_floor` is no: q\* = 0.88 is not the floor, so that wording does not apply.**
- **One-sided error.** In the study the mean ξ̂ bias at q\* is negative in all 9 cells with a truth (−0.018 to −0.130) and the ES99 relative bias is negative in all 10 cells (−0.003 to −0.032). It is small relative to the RMSE at the 99% level but lies in the risk-understating direction. Behaviour beyond 99% was not measured; the descriptive levels 0.995 and 0.999 are flagged as likely understated when q\* is low.
- **Scope.** The simulated worlds with a boundary have an exact GPD tail above q0. This is evidence for those worlds only, not for the Nifty 50.
- **Options not adopted.** (iii) GoF on declustered cluster maxima: the i.i.d. S1 cells show the same false-selection rate as S3, so dependence is not the cause. (ii) raising α_FS: reasoning only, not tested (D36 (a)).

## 8. Pinned out-of-sample sensitivity set (D36 (b); nothing is run in 7C)

Offsets from `[threshold]` sensitivity_offsets applied to q\* = 0.88, plus the pre-declared extra run at q = q_max = 0.90:

| Run | q | on the grid | window N_u = (1 − q)·W | ≥ n_min = 100 |
|---|---|---|---|---|
| q\* − 0.02 | 0.86 | yes | 140 | yes |
| q\* + 0.02 | 0.90 | yes | 100 | yes |
| q\* + 0.03 | 0.91 | yes | 90 | no |
| extra, q = q_max | 0.90 | yes | 100 | yes |

- The extra run at q = 0.90 coincides with q\* + 0.02, so the set has three distinct levels: 0.86, 0.90, 0.91.
- **Open point for the owner:** q\* + 0.03 = 0.91 has window N_u = 90, below n_min = 100 (the candidate table flags it `below_n_min_window`). D36 (b) (1) says the window N_u must be ≥ n_min. Whether this run is made, replaced or dropped is not decided here.
- The config key for the q = 0.90 run is added at the OOS stage, not now.

## 9. What the numbers show

This section reads the tables. It decides nothing.

- **ForwardStop.** The first 8 candidates (q = 0.80 to 0.87) are rejected. Their AD p-values range from 0.025 to 0.199; the p-values below 0.05 are at q = 0.83, 0.84, 0.85, 0.86. The running statistic is at or below α_FS at q = 0.84, 0.85, 0.86, 0.87; the largest such k gives k̂ = 8. From q = 0.88 upwards the AD p-values are 0.21 or higher.
- **Stable region.** q\* equals q_FS: the first candidate that is not rejected also has a passing window. Inside the window q = 0.88–0.91, ξ̂ runs from 0.1157 to 0.1695 (largest \|Δξ̂\| relative to q\*: 0.05378, tolerance 0.10), the largest relative VaR99 change is 0.009279 (tolerance 0.05) and the largest relative ES99 change is 0.01587 (tolerance 0.10). The window member q = 0.91 lies above q_max, which D34 (6) allows.
- **Later candidates.** q = 0.89: tol_xi at q = 0.92; q = 0.90: tol_xi at q = 0.92; q = 0.91: above_q_max; q = 0.92: above_q_max; q = 0.93: above_q_max; q = 0.94: above_q_max; q = 0.95: above_q_max. ξ̂ rises to 0.2684 at q = 0.92.
- **Risk measures across the grid.** VaR99 lies between 0.04257 and 0.04405 and ES99 between 0.05685 and 0.06086 over all 16 candidates. At q\*: VaR99 = 0.04357 [0.0395, 0.04764], ES99 = 0.05763 [0.04863, 0.06664]. At q = 0.90: VaR99 = 0.04337 [0.03925, 0.04748], ES99 = 0.05808 [0.04845, 0.06771].
- **Shape.** ξ̂ lies between 0.06029 and 0.2684 over the grid. The three side-by-side estimates are 0.1157 [-0.009205, 0.2753] at q\*, 0.1423 [-0.0007109, 0.3268] at 0.90 and 0.1547 [-0.07103, 0.4747] at 0.95. The 95% profile CI contains 0 at 12 of 16 candidates, q\* included. The point estimate at 0.90 and at 0.95 is higher than at q\*; all three intervals overlap.
- **Split halves.** The two halves do not agree with each other or with the full sample. Returns 1–1000: k̂ = 11, q_FS = 0.91, which is above q_max, so no q qualifies (no_stable_region). Returns 1001–2000: k̂ = 0 and q = 0.80, the grid floor. The flag is raised on both counts. This is the pre-declared signal that the selected level depends on the sub-period.
- **Dependence.** θ̂ is below 1 at every candidate (0.4988 to 0.7328), so exceedances cluster. Likelihood-based CIs that treat exceedances as independent are therefore too narrow (methodology §9); block-bootstrap CIs come later (plan Day 10).
- **Declustered comparison.** The declustered ξ̂ is lower than the all-exceedance ξ̂ at 14 of 16 candidates (differences from -0.0959 to 0.06411). At 16 of 16 candidates each estimate lies inside the other's 95% CI.
- **Numerics.** No candidate fit failed, no `xi_gap` flag was raised, and the rolling-window convergence share is 1 at every candidate.
- **Reading against the limitation in §7.** The study measured where q\* falls relative to a known boundary; the real data has no known boundary, so the study cannot say whether 0.88 is too low. It does say that, in those worlds, the procedure's q\* was usually below the boundary and ξ̂ and ES99 at q\* were biased downwards. The sensitivity run at q = 0.90 is the pre-declared check on the high-threshold side.

## 10. Not verified

- The Ferro–Segers JRSS-B paper was not read; the declustering convention was read in the first author's thesis (D36 (c)). The R package extRemes appears to index the cut differently (⌈θ̂·N⌉ instead of 1 + ⌊θ̂·(N − 1)⌋); this was not resolved.
- β\* has no confidence interval in figure (b): no formula for it is pinned.
- The CIs in this document treat exceedances as independent; θ̂ < 1 says they are not.
- The SVG figures were checked as PNG previews rendered from the same code; the SVG files themselves were only byte-compared.
- Nothing out-of-sample was run or looked at.

## 11. Run record

- Command: `uv run --locked python -m niftytail threshold-select configs\base.toml`, run twice on 2026-11-12 (268.2 s and 265.2 s, serial, BLAS threads 1). Exit status 0 both times.
- **All six tracked output files are byte-identical across the two runs.**
- Inputs recorded in `selection.json`: config sha256 `b602725a28e6a3983867ebc83229e0dbcfd20e574f54f417032424eef5cf0e82`; manifest sha256 `87dca720be73fd81ddc8a76598bcf9d8533a8daddc2d1a7d57978585f78e0c04`; plan sha256 `44f42b13a9daf43d1857cfb9f258f6b46d9f3fc9cace9fdca58f82416465d66d`; `nmin_study.json` sha256 `5a8349a7773fcb38dc2fe264ccf6fa32d5bb5493df5c17a555c35aee6964fc97`; `simstudy/metrics.json` sha256 `cfa424e0fa1049eca51fc98d37c00fa66aca285d699d4673ff38274b003212b5`.

| Tracked file | sha256 |
|---|---|
| `selection.json` | `7929c129a03509cb31b56cea1804cbfd4559e99c09b960d86b658b86e86741f9` |
| `fig_a_mrl.svg` | `0b080c2f8158e951786453bd673b7cfa5346e1e9061195ec2968e25fdab1aaa3` |
| `fig_b_xi_beta_star.svg` | `397081d6a1031c1b01084de5641f723cd9c7132ee73511da2bb427502312a070` |
| `fig_c_var_es.svg` | `db19e0b919e140bce168fc8188fff70df4060d1df7a727c423173010105ae2e5` |
| `fig_d_ad_forwardstop.svg` | `0be28685244d8b0d51dac50338a8a500745a76ca457a1f70b19e0b8a9c47c153` |
| `fig_e_theta_declustered.svg` | `7449d2e12d6b49b618bc5c9bd4da6aabfde5b48e217cc533ec0c3735cf8f4162` |

Figures (tracked, aggregates only): (a) `fig_a_mrl.svg` mean residual life; (b) `fig_b_xi_beta_star.svg` ξ̂ with profile CI and β\* against q; (c) `fig_c_var_es.svg` VaR and ES at 95% and 99% with CIs; (d) `fig_d_ad_forwardstop.svg` AD p-values, the ForwardStop statistic against α_FS, q_FS; (e) `fig_e_theta_declustered.svg` θ̂ and ξ̂ from all exceedances against the declustered ξ̂.

Not tracked (derived from per-day data; `reports/threshold_selection/real/figures/`, gitignored): the exceedance timeline at q\*, the QQ plot of inter-exceedance times against Exp(1) and the ACF of the exceedance indicator at q\*.

## 12. Owner sign-off (D37, 2026-11-12)

- (a) q\* = 0.88 from `reports/threshold_selection/real/selection.json` (sha256 7929c129a03509cb31b56cea1804cbfd4559e99c09b960d86b658b86e86741f9) is adopted as `[threshold] q_selected = "0.88"`.
- (b) The split-half flag is accepted and not acted on. Half 1 (returns 1–1000): no_stable_region, k̂ = 11, q_FS = 0.91 > q_max. Half 2 (returns 1001–2000): q = 0.80, k̂ = 0. Reasons: each half has half the exceedances, and A3 showed low AD power even at full size; across the full-sample grid 0.80–0.90 the risk measures barely move: VaR99 0.04337 to 0.04405 (max/min − 1 = 0.0157), ES99 0.05685 to 0.05808 (max/min − 1 = 0.0216) (from selection.json). Limitation kept: possible tail non-stationarity within the training period; the real data rejects more candidates (k̂ = 8) than the A3 cells typically did, so the A3 evidence transfers only partly.
- (c) OOS sensitivity set: 0.86 (q\* − 0.02); 0.90 (q\* + 0.02, which is also the D36 (b)(2) q_max run, counted once); 0.91 (q\* + 0.03). 0.91 is run as plan §3 specifies and labelled "window N_u = 90 < n_min = 100, descriptive"; the D36 (b)(1) N_u condition was written for below-grid offsets and does not remove a plan-mandated run. No level is added.
- (d) Reporting: ForwardStop rejects candidates 0.80–0.87 jointly (maximum rule); q = 0.80's own AD p-value is 0.199, so it is not described as individually rejected. The 95% profile CIs of ξ̂ include 0 at q\* [-0.009205, 0.2753], 0.90 [-0.0007109, 0.3268] and 0.95 [-0.07103, 0.4747]; the training data does not establish ξ > 0 at 95%, so "heavy tail" is not headlined.

This supersedes the open points in §8: 0.91 is run and labelled descriptive, and the set has three distinct levels (0.86, 0.90, 0.91).
