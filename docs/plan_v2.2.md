# NIFTYTAIL — Plan V2.2 (= V2.1 + correction amendment A1–A6)

> **Reading order:** the V2.2 amendment (A1–A6) comes first and **supersedes** any conflicting V2.1 text below. All other V2.1 decisions are preserved.

> **Status (2026-11-11):**
> - V2.2 approved in principle.
> - **D1–D10 and D12–D17 approved as recommended.**
> - **D11 (data licensing) deferred.** It must be decided before Day 3; no data is touched until then.
> - Complete master implementation prompt received and reviewed.
> - Implementation **not started**. Day-1 proposal below awaits approval.

## DAY-1 IMPLEMENTATION PROPOSAL (awaiting approval)

### Current state (inspected 2026-11-11)

| Item | State |
|---|---|
| `C:\Users\Sambi\Desktop\Niftytail` | exists, **empty** |
| Git | **not initialised**, here or in any parent folder |
| Python 3.13 | available, 3.13.17 via `py -3.13`. The default `python` is 3.14.7 and must not be used. |
| uv | 0.12.15 available |
| V2.2 plan | exists only as this plan file, outside the project |

### Clarifications resolved this session

These become new decisions:

| # | Decision | Resolution |
|---|---|---|
| D18 | Location of documents | `docs/methodology.md`, `docs/literature.md`, `docs/decisions.md`, `docs/status.md`, plus `docs/plan_v2.2.md` as a verbatim, read-only copy of this approved plan (the in-project source of truth). `README.md` stays at the root on Day 2. |
| D19 | Kupiec exact binomial p-value | **Equal-tailed:** p = min{1, 2·min[P(X ≤ x), P(X ≥ x)]}, with X ~ Bin(T, p0). This is implemented with `scipy.stats.binom.cdf` and `binom.sf`, **not** `binomtest`, whose default two-sided method differs. p ≤ 0.05 is equivalent to x falling outside the equal-tailed regions computed in V2: [251, 315] at p0 = 0.05 and [42, 72] at p0 = 0.01, for T = 5,650. Those regions become test reference values. |
| D20 | Christoffersen MC nulls | **Correction to V2.1 §4.** The IND test simulates i.i.d. Bernoulli(p̂); the CC test simulates i.i.d. Bernoulli(**p0**). MC p-value = (1 + #{LR\* ≥ LR_obs})/(R + 1), with R = 9,999 and seed 20261111. |

Remaining open item: **D11, data licensing.** It must be decided before Day 3, and it does not block Days 1–2.

### Day-1 scope: files to create after approval

Nothing else is created. Git, uv, packages and data are untouched.

1. **`docs/plan_v2.2.md`**: a verbatim copy of this plan.
2. **`docs/methodology.md`**: the single mathematical specification. Its sections:
   1. Notation: c and p = 1 − c (A0), with exact rational parsing and the float-trap example.
   2. Data and returns (series, base date, log returns, loss sign, unit conventions).
   3. Sample split (T0, W, OOS; leakage rules).
   4. HS: VaR and integrated fractional ES (A1) with the five-row hand table.
   5. Gaussian VaR and ES.
   6. GPD: parameterisation, MLE and PWM, the ξ = 0 branch, tail VaR and ES, ξ < 1 and ξ < ½ conditions, SE(ξ̂).
   7. Threshold procedure: Steps 0–5, ForwardStop eq. (6) with its Theorem-2 assumption stated, stable region, fallback, and fixing q\* across windows.
   8. The A3 simulation study design, labelled "simulation evidence".
   9. Dependence diagnostics.
   10. Backtests: Kupiec LR plus D19; Christoffersen plus D20; traffic light (A4); ES residual test (A5a).
   11. FZ0 in return space: equation, conditions, conversion, tie note, analytic identity, six-row hand table, common mask (A2).
   12. Diebold–Mariano and Newey–West lag (A5b).
   13. Moving-block bootstrap (A5c).
   14. Reproducibility (A5e).
   15. Limitations (V2.1 §8).

   Each formula states its definition, assumptions, parameterisation, numerical limits and a worked example. Each cites its literature entry with a status code.
3. **`docs/reference_values.py`**: a **stdlib-only** script, run with `py -3.13 -I docs\reference_values.py` (no install, no uv). It independently re-derives every hand value in `methodology.md`:
   - GPD VaR and ES at 95% and 99%;
   - Gaussian values;
   - the HS table;
   - the FZ0 table, plus the identity E[FZ0] = ln(−e) by stdlib Simpson integration;
   - Kupiec LR = 1.9568;
   - the Kupiec LR regions [251, 315] and [43, 71];
   - the equal-tailed binomial regions [251, 315] and [42, 72];
   - the exact power table and the Markov size table;
   - the float-trap demonstration;
   - DM lag 9 and block lengths 13 and 18;
   - an equal-tailed binomial p-value example at T = 250, x = 5, p0 = 0.01.

   Its printed output is pasted into an appendix of `methodology.md`, so every number traces to code. From Day 5, the package tests assert the same values. Purpose: reproducibility of the specification itself. It is not application code.
4. **`docs/literature.md`**: the full evidence matrix, 31 entries. Columns: title, authors, year, venue, DOI or URL, metadata status, claim relied on, claim status (M/C/F/U), methodology, findings, limitations, relevance.
   - Findings and limitations are filled **only for texts actually read**. Every other cell says "not read".
   - Day-1 reading attempts are limited to open-access texts: Scarrott & MacDonald (REVSTAT OA), Bader et al. (arXiv version if available), the remainder of Patton–Ziegel–Chen (already fetched) and G'Sell et al. (already fetched).
   - Coles (2001) and paywalled papers stay F unless you can supply access.
   - Indian-market search: web searches for Nifty or Indian EVT, POT, VaR and ES studies. Only papers whose metadata resolves via Crossref or the publisher are added, as M. **No novelty claim** is made.
   - External requests carry only paper titles and DOIs, never project data or personal details.
5. **`docs/decisions.md`**: D1–D20, each with date, status, chosen option, rationale and source section.
   - Approved 2026-11-11: D1–D10 and D12–D20.
   - **Open:** D11, deadline before Day 3.
6. **`docs/status.md`**, kept concise:
   - Milestone: Day 1.
   - Done: the work above.
   - Pending: Day 2.
   - Tests: none yet, because there is no package.
   - Known issues: D11, plus the F-status claims.
   - Next action: Day-2 setup, pending your approval.

### Day-1 acceptance criteria

The V2.1 Day-1 criteria and the A6 criteria are both used:
- Every formula in `methodology.md` has a worked example produced by `reference_values.py`. The script runs without error under `py -3.13 -I`, and I report its actual output.
- Only the c/p notation is used, so there are no bare α subscripts on VaR or ES; FZ0's α is defined as p.
- Every literature claim relied on is marked C or explicitly F. Nothing is labelled verified unless it was checked this session.
- `decisions.md` contains D1–D20, with D11 open.
- A cross-check of `methodology.md` against `docs/plan_v2.2.md`, where any discrepancy is reported to you rather than resolved silently.
- Your review and sign-off.

### Out of scope for Day 1

`git init`, `uv init`, `uv add` and `uv sync`; any package code; tests of package code; data download.

### Then STOP

I report the files created, the script output and the outstanding items, and wait for your approval before Day 2.

## V2.2 AMENDMENT

### A0. Notation (removes the V2.1 ambiguity behind point 1)

Single convention used everywhere:
- **c** is the confidence level (0.95 or 0.99).
- **p = 1 − c** is the upper-tail loss probability (0.05 or 0.01).

V2.1 wrote VaR_α with α = c in the GPD and HS formulas, but α = p in the FZ0 formula. That mixed usage is the real defect. `methodology.md` uses only c and p, and FZ0's α is p.

Under V2.1's own notation, ⌈W(1−α)⌉ with α = c already gave 50 and 10. However, **both** that form and the form k = ⌈W·p⌉ are numerically unsafe in floating point. Measured this session: `ceil(1000*(1-0.99)) = 11` and `ceil(1000*(1-0.95)) = 51`. So:
- Confidence levels are parsed from config strings into `fractions.Fraction`.
- m = W·p is computed exactly.
- No tail count is ever derived from a float product.

### A1. Historical Simulation (HS): exact conventions

Window losses are sorted ascending: L_(1) ≤ … ≤ L_(W). The empirical CDF is F_W(x) = #{L_i ≤ x}/W, and m = W·p (an exact rational).

**VaR (lower inverse of F_W, i.e. numpy `inverted_cdf`):**

VaR_c = inf{x : F_W(x) ≥ c} = L_(⌈W·c⌉)

**ES (Acerbi–Tasche form, ES_c = (1/p)∫_c^1 VaR_u du, evaluated exactly on F_W):**

ES_c = [ (m − ⌊m⌋)·L_(W−⌊m⌋) + Σ_{j=W−⌊m⌋+1}^{W} L_(j) ] / m

- **Integer m** (our design: W=1000 or 2000, c ∈ {0.95, 0.99, 0.995, 0.999}). ES is the mean of the m largest losses, VaR = L_(W−m) is the next order statistic below them, and exactly m window losses exceed VaR when there are no ties. For W = 1000: **c = 0.95 → worst 50; c = 0.99 → worst 10.**
- **Non-integer m.** The order statistic L_(W−⌊m⌋), which equals VaR in this case, gets the fractional weight m − ⌊m⌋.
- In both cases ES ≥ VaR.

**Hand examples** (losses 1, 2, …, W, computed this session; the fractional case was cross-checked by numerically integrating VaR_u):

| W | c | m | VaR | ES |
|---|---|---|---|---|
| 1000 | 0.99 | 10 | 990 | 995.5 (mean of 991…1000) |
| 1000 | 0.95 | 50 | 950 | 975.5 (mean of 951…1000) |
| 250 | 0.99 | 2.5 | 248 | (0.5·248 + 249 + 250)/2.5 = **249.2** |
| 250 | 0.95 | 12.5 | 238 | (0.5·238 + Σ239…250)/12.5 = **244.24** |
| 20 | 0.95 | 1 | 19 | 20 |

**Tests:**
1. All five hand examples above.
2. The same results on randomly permuted input, so the code sorts internally.
3. **Float-trap test:** c given as "0.99" and W = 1000 yields m == 10 exactly, VaR = L_(990), and an ES over exactly 10 values.
4. **Tail selection:** with exactly m strictly larger values than VaR (no ties), adding large *gains* (negative losses) changes neither VaR nor ES. This proves the code takes the loss tail, not the return tail.
5. Ties at the boundary give a deterministic result.
6. Agreement with `numpy.quantile(method="inverted_cdf")` for VaR, an independent implementation.
7. ES equals a `quad` integral of the empirical VaR_u for the fractional cases.

### A2. FZ0 model comparison on a common evaluation sample

**Admissibility.** Forecast (i, t) is *admissible* iff VaR and ES are finite and, in return space, e ≤ v < 0. This means a non-NaN VaR and ES, a non-infinite ES (ξ̂ < 1 for POT), VaR_L > 0 and ES_L ≥ VaR_L.

**Common mask.** C_t = ∧_{i ∈ M} admissible(i, t) over the compared model set M = {HS, Gaussian, POT}. It is computed separately for each level c.

**Primary comparison.**
- Mean FZ0 per model, and all pairwise DM tests, use **only dates with C_t = true**, the same dates for every model and every pair.
- Inadmissible forecasts are **never repaired, clipped or substituted**. They only shrink the common sample.

**Reported per level:**

| total OOS dates | admissible forecasts per model | inadmissible per model, with reason counts (NaN / fit failure, ES infinite, VaR ≤ 0, ES < VaR) | common dates \|C\| | mean FZ0 per model on C | DM statistics on C |

- If |C| / total < 0.99, the report shows a prominent flag, and the excluded dates are listed with their causes.
- **Coverage statistics are kept separate.** Kupiec, Christoffersen, traffic light and the ES residual test are computed per model on **that model's own admissible dates**, and each model's admissible-date count is reported.
- **Masking and DM:** the loss-difference series on C has gaps where dates were removed. HAC is applied to the compressed series, which is noted as a limitation; it is negligible if |C| is close to the total.

**Tests:**
- On a constructed panel where one model has inadmissible forecasts on known dates, the mask, the counts and the mean FZ0 on C are checked by hand.
- No model's score uses a date outside C.
- Inadmissible values are passed to `fz0` and still raise.

### A3. Threshold procedure: dependence-aware simulation study (Day 7, before the real selection)

Status labels:
- The FDR guarantee of ForwardStop (G'Sell et al. 2016, Thm 2) assumes independent U(0,1) null p-values. Nested thresholds violate that, so **no theoretical guarantee is claimed for our use**.
- **V2.1's claim that "dependence makes AD p-values too small, a conservative direction" is withdrawn.** The direction is unknown and is measured here.
- All results are labelled **"simulation evidence (DGPs S1–S4)"**.

**Data-generating processes.** Each sample has n = 2,000, the training size. For the rolling check, the last 1,000 points are refit at q\*.

| ID | DGP | Known truth | Cells |
|---|---|---|---|
| S1 | i.i.d. **spliced**: below u0 = Φ⁻¹(q0) the CDF is Φ; above it, the CDF is q0 + (1−q0)·G(x−u0; ξ, β) with β = (1−q0)/φ(u0) so the density is continuous | exact marginal, ξ and boundary q0: thresholds q ≥ q0 are exactly GPD by threshold stability, q < q0 are inadequate | q0 ∈ {0.85, 0.90} × ξ ∈ {0.10, 0.25, 0.40} → 6 |
| S2 | i.i.d. Student-t, ν = 4 | exact VaR and ES (closed form); asymptotic ξ = 0.25; no exact boundary | 1 |
| S3 | **clustered extremes with exact marginal**: max-autoregressive (ARMAX) Fréchet process X_t = max(aX_{t−1}, (1−a)Z_t), marginal unit Fréchet; U_t = exp(−1/X_t); L_t = F_S1⁻¹(U_t) | the S1 marginal (q0 = 0.90, ξ = 0.25), boundary known; extremal index 1−a (standard property, F-status; also checked by our θ̂ test) | a ∈ {0.3, 0.6} → 2 |
| S4 | **volatility-clustered returns**: GARCH(1,1), a1 = 0.08, b1 = 0.90, ω set for unit variance, standardised t(ν=6) innovations, burn-in 1,000. **Simulated only, no GARCH fitting, so no scope change.** | reference marginal VaR and ES from one seeded 10⁷-observation simulation, with MC error reported; no exact ξ or boundary | 1 |

**Settings.**
- R = 200 replications per cell (10 cells).
- AD bootstrap uses 199 reps per candidate in the study, versus 999 in the real selection; this difference is stated.
- Seeds come from `SeedSequence(20261111).spawn` per cell and replication.
- The study runs in parallel with stdlib `ProcessPoolExecutor`. Results do not depend on worker count.
- A Day-6 timing pilot runs first. If the projected runtime exceeds 2 hours, R drops to 100, a pre-declared rule.

**Metrics per cell:**
1. Distribution of the selected q\*, including the rule-failure (fallback) rate.
2. **False selection frequency:** P(q\* < q0), for S1 and S3.
3. **Empirical FDR of the ForwardStop stage:** the mean of V / max(1, R_rej), where R_rej is the number of rejected candidates and V the number of rejected candidates with q_k ≥ q0 (true nulls). Compared with the nominal α_FS, for S1 and S3.
4. **Shape:** bias and RMSE of ξ̂(q\*) against the true ξ (S1, S3), or against the asymptotic 0.25 for S2, labelled as asymptotic.
5. **VaR and ES:** relative bias and relative RMSE of VaR95, ES95, VaR99 and ES99 at q\*, for n = 2,000 and for the last-1,000 window refit, against the true or reference values.
6. **Failures:** MLE non-convergence rate, P(ξ̂ ≥ 0.5), P(ξ̂ ≥ 1).

S3 has the **same marginal** as the S1 cell (q0 = 0.90, ξ = 0.25). The difference between the two isolates the effect of dependence.

**Pre-declared review triggers.** These lead to a decision by you, never to an automatic change. In either S3 cell, the triggers are:
- false selection > 0.20;
- empirical FDR > 2·α_FS;
- rule failure > 0.10.

If any trigger fires, it is documented as a limitation and you choose among:
- (i) proceed as specified;
- (ii) raise α_FS;
- (iii) apply the GoF test to declustered cluster maxima.

Any change re-runs the study **before** the real training data is touched.

### A4. Basel traffic light: correct use

- It applies to **VaR99 only** and to **non-overlapping consecutive 250-observation blocks**, starting at the first OOS day.
  - With T ≈ 5,650 there are 22 full blocks.
  - The final partial block (≈ 150 days) shows its breach count with **no zone**.
- Zone boundaries are green 0–4, yellow 5–9, red ≥ 10, per BCBS 22 (1996). That document is F-status, and its zone definitions will be confirmed against the original text.
- The output is a table and a strip plot of zone by block, plus the count of blocks per zone per model.
- **No traffic-light verdict is ever issued on the full 5,650-day sample.** The standard thresholds are not rescaled to it.
- Limitations stated in the report:
  - the framework is a regulatory capital device, not a statistical test;
  - its boundaries are calibrated to 250 observations under independent breaches;
  - blocks are not independent under volatility clustering;
  - the alignment of block starts is arbitrary (fixed, not optimised);
  - per-block power is very low (the Kupiec acceptance region at 250 obs is [0, 6]).
- It is labelled **"descriptive regulatory reference"**.

### A5. Remaining specifications

**(a) ES exceedance-residual test** (in the spirit of McNeil & Frey 2000; their exact form is F-status, and this is our specification).
- On each model's own breach days H = {t : L_{t+1} > VaR_t}, define the scale-free residual z_{t+1} = (L_{t+1} − ES_t)/ES_t, where ES_t is known at t.
- H0: E[z | breach] = 0, i.e. ES correctly forecasts the mean loss beyond VaR. H1: E[z | breach] > 0, i.e. ES under-forecasts. The test is **one-sided**.
- Statistic: z̄. The p-value comes from a null-centred i.i.d. bootstrap: z̃ = z − z̄, B = 9,999, seed 20261111, and p = (1 + #{mean(z̃\*) ≥ z̄}) / (B + 1).
- Reported: n_breach, z̄ and p. If n_breach < 20, the result is descriptive only and no p-value is given.
- Limitations:
  - clustered breaches make the residuals dependent, so the i.i.d. bootstrap understates variance;
  - the test conditions on each model's own VaR, so models are not compared on the same days;
  - power is low (about 56 breaches at 99%).
- Tests:
  - residuals constructed with mean 0 give p ≈ uniform under repeated draws (slow test);
  - a +20% ES under-forecast gives a small p;
  - the hand-computed z̄ is matched.

**(b) Diebold–Mariano.**
- d_t = FZ0_A,t − FZ0_B,t on the common mask C, with T_c = |C|.
- DM = d̄ / √(Ω̂/T_c), where Ω̂ is the Newey–West (Bartlett kernel) long-run variance with **lag ℓ = ⌊4·(T_c/100)^{2/9}⌋** (ℓ = 9 at T_c = 5,650).
- This is a fixed, pre-declared deterministic rule. Newey & West (1994), *Automatic lag selection in covariance matrix estimation*, Rev. Econ. Stud. 61(4):631–653, DOI 10.2307/2297912, has M status. Attributing this exact rule-of-thumb form to that paper is F-status.
- Two-sided test against an N(0,1) reference. One sensitivity is reported: ℓ doubled.
- Test: d̄ and Ω̂ match an independent computation, OLS of d on a constant with `statsmodels` `cov_type="HAC"` and the same maxlags and kernel.

**(c) Moving-block bootstrap.**
- Block length b = ⌈n^{1/3}⌉: b = 13 for n = 2,000 (training), b = 18 for n = 5,650 (hit sequences and OOS). The n^{1/3} *rate* is the known order for block-bootstrap variance estimation (F-status); the constant 1 is a pre-declared choice.
- Sensitivity on the key CIs only: b/2 and 2b.
- Settings: B = 1,999 resamples, percentile intervals, circular wrapping off, seed 20261111.
- Automatic block-length selection is not adopted.

**(d) Quantile and ES conventions** are as in A1 for HS. GPD and Gaussian VaR and ES are analytic (V2 §2.3, §2.5). Hits use the strict inequality L > VaR.

**(e) Version pinning and deterministic execution.**
- `requires-python = ">=3.13,<3.14"` and `.python-version = 3.13`.
- Direct dependencies are listed in pyproject with lower bounds. **Exact versions come only from the committed `uv.lock`.** Every run and test uses `uv sync --locked` / `uv run --locked`.
- `manifest.json` records Python and package versions (`importlib.metadata`), platform, the numpy BLAS configuration, the git commit with a dirty flag, the config hash and the data sha256.
- Determinism rules:
  - BLAS and OpenMP threads are pinned to 1 (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`), set at process start in `__main__`.
  - Each task gets one `numpy.random.Generator(PCG64)`, and parallel tasks get child seeds via `SeedSequence.spawn`. There is no global `np.random`.
  - Optimiser method, start values, tolerances and iteration limits come from config.
  - Output order is sorted and does not depend on dict or set ordering.
  - Floats are written with `repr`, 17 significant digits.
  - No timestamps appear inside compared artifacts.
- **Reproducibility criteria:**
  - Same machine and same lockfile: byte-identical `forecasts.csv` and `backtest.csv`.
  - A different machine: agreement to relative tolerance 1e-10, not bytes, because libm and BLAS can differ.

### A6. Updated acceptance criteria (changes only; other V2.1 rows stand)

| Day | Added or changed acceptance criteria |
|---|---|
| 1 | `methodology.md` uses only the c/p notation (A0); includes the A1 HS formulas and hand table, and the A5 test definitions |
| 2 | `uv sync --locked` succeeds from `uv.lock`; `requires-python` pinned; thread-pinning in place; manifest fields present |
| 6 | N_min study done (V2.1 §3, Step 0); **timing pilot** for A3 done and R confirmed |
| 7 | **A3 study complete for all 10 cells**; metrics table and plots committed as "simulation evidence"; review triggers evaluated and reported to you **before** the real training selection; then the real selection; q\* plus evidence committed before any OOS run |
| 8 | All A1 HS tests pass, including the float-trap and tail-selection tests |
| 10 | A2 common-mask tests; ES residual test (A5a); DM matches the statsmodels HAC computation (A5b); traffic-light block construction (22 full blocks, partial block unzoned) (A4); block-bootstrap reproducibility (A5c) |
| 11 | The report includes the full A2 count table per level; traffic light shown per block only |
| 14 | Byte-identical rerun on this machine; cross-machine tolerance documented |

### V2.2 decisions added (D1–D12 preserved)

| # | Decision | Recommendation |
|---|---|---|
| D13 | Threshold simulation design and review triggers | A3 as specified |
| D14 | ES residual test | relative residual, one-sided, i.i.d. bootstrap, n_breach ≥ 20 |
| D15 | DM lag | ⌊4(T_c/100)^{2/9}⌋, with a doubled-lag sensitivity |
| D16 | Block length | ⌈n^{1/3}⌉, with ½× and 2× sensitivity |
| D17 | Traffic light | VaR99, 22 non-overlapping 250-day blocks, descriptive only |
| D7 (amended) | FZ0 | return space (approach A) **plus the common evaluation mask** |

---

# V2.1 (retained; superseded where A1–A6 conflict)

## Context

The project is new and lives in `c:\Users\Sambi\Desktop\Niftytail`. The folder is empty and has no git repository. Python 3.14.7 is the default and 3.13 is installed. uv 0.12.15 is available.

V2 was reviewed. V2.1 resolves five points:
- the FZ0 sign convention;
- threshold-selection safeguards;
- a reproducible Kupiec power specification;
- the justification for the training period;
- the licensing distinctions.

New evidence gathered this session:
- The ForwardStop definition and its assumptions were read in the G'Sell et al. full text.
- Bader et al. (2018) was verified as the established ordered-GoF threshold method.
- The niftyindices Terms of Use and Disclaimer were read, and are quoted verbatim in §5.
- Kupiec size and power were computed exactly and confirmed by seeded simulation.
- An analytic identity for expected FZ0 at the true values was derived and confirmed by simulation.

Nothing outside this plan file was created or modified.

### Corrections to V2 (stated explicitly)
1. **The V2 "Kupiec power" figures (0.40 / 0.91) were wrong.** They were the power of an equal-tailed exact binomial test, not Kupiec's LR test. The correct Kupiec LR figures are in §4: 0.451 and 0.930.
2. **The V2 99% acceptance region [42, 72] was also from the equal-tailed binomial test.** The Kupiec LR acceptance region is **[43, 71]**.
3. **The V2 threshold rule (lowest q with AD p ≥ 0.10 plus CI overlap) had no control for multiple testing.** It used an inappropriate CI-overlap criterion on nested samples. It is replaced in §3.
4. **The V2 FZ0 section presented a loss-space formula as primary.** V2.1 uses approach A: the published return-space formula internally (§2).

---

## 1. Updated methodology (summary of the final specification)

| Element | Specification |
|---|---|
| Series | Nifty 50 Price Return index, Close (primary); TRI Gross (sensitivity); never mixed within a run |
| Sample | from the official base date 3-Nov-1995 to the last close at retrieval. The Jul-1990 to Nov-1995 back-calculated values are excluded. |
| Returns | `r_t = ln(P_t/P_{t-1})`. Models produce loss-space numbers, with loss `L_t = −r_t`. |
| Split | training = returns 1…T0 (T0 = 2000); rolling window W = 1000; OOS = targets T0+1…N (≈ 5,650) |
| Horizon | 1 day; forecast from origin t uses only `L[t−W+1…t]` |
| Models | Historical Simulation (HS) (**see A1**: VaR = L_(⌈Wc⌉); exact-rational fractional ES), Gaussian (window μ̂, σ̂), POT-GPD |
| GPD | `G(y)=1−(1+ξy/β)^(−1/ξ)`, β > 0. VaR and ES formulas as in V2 §2.3; ES finite iff ξ < 1; ξ = 0 limit branch. MLE uses our own NLL, started from Probability-Weighted Moments (PWM) estimates, with `scipy.genpareto.fit(floc=0)` as an independent cross-check. |
| Threshold | fixed quantile level q\* from the §3 procedure on training only; u_t recomputed per window, so N_u = (1−q\*)·W is constant |
| Levels | primary pairs (VaR95, ES95) and (VaR99, ES99); 99.5% and 99.9% descriptive only |
| Backtests | Kupiec LR plus exact binomial p; Christoffersen IND/CC with Monte Carlo p-values; Basel traffic light (250-day blocks, descriptive, **A4**); ES exceedance-residual test (**A5a**); FZ0 (§2) on the **common mask (A2)** with Diebold–Mariano (DM) (**A5b**) |
| Dependence | ACF and Ljung–Box on r, \|r\| and r²; ARCH-LM; extremal index (Ferro–Segers); inter-exceedance times; indicator ACF; declustered-ξ comparison; block-bootstrap CIs. No GARCH in v1. |

V2 hand checks carried forward:
- **GPD** (n=1000, N_u=100, u=2, ξ=0.25, β=0.5): VaR99 = 3.556559, ES99 = 4.742078, VaR95 = 2.378414, ES95 = 3.171219.
- **Gaussian:** VaR95 = 1.644854, ES95 = 2.062713, VaR99 = 2.326348, ES99 = 2.665214.

---

## 2. FZ0: correct mathematics (approach A, as you preferred)

**Source.** Patton, Ziegel & Chen (2019), *J. Econometrics* 211(2):388–413, DOI 10.1016/j.jeconom.2018.10.008. The formula is equation (6) of the arXiv version 1707.05108, which I read this session. The equation number in the journal version still needs checking.
- The conditions come from the same text. Proposition 1 assumes "VaR and ES are both strictly negative", and footnote 1 says consistency "requires imposing that e ≤ v".
- Their Figure 1 uses (v, e) = (−1.64, −2.06) as "the α = 0.05 VaR and ES from a standard Normal".

**Convention used internally (return space).**
- Y_t is the realised log return.
- α is the tail probability: 0.05 for the 95% level, 0.01 for 99%.
- v_t and e_t are the return-space VaR and ES, both negative numbers.
- Admissibility: **e ≤ v < 0**.

**Formula.**

`FZ0(Y, v, e; α) = −(1/(α·e)) · 1{Y ≤ v} · (v − Y) + v/e + ln(−e) − 1`

Lower is better.

**Boundary conversion.** This is the only place signs change. Models output loss-space numbers (VaR_L > 0, ES_L ≥ VaR_L). A single function `to_return_space` maps them as `v = −VaR_L`, `e = −ES_L` and `Y = r_t`. FZ0 is computed only in return space. Reports show the model, level and mean FZ0 score; the score is a scalar with no sign convention to present.

**Validity handling.**
- `fz0()` raises `ValueError` if any v ≥ 0, any e > v, or any value is NaN.
- **Superseded by A2.** Comparison uses the common evaluation mask; nothing is excluded model by model and nothing is repaired.
- ES ≥ VaR holds by construction for all three models (V2 §2.3, §2.5). The only realistic violation is VaR95_L ≤ 0 in some window, which gives v ≥ 0.

**Tie note.** At Y = v the indicator is 1 but (v − Y) = 0. The score is therefore continuous at the boundary, and the hit convention (strict L > VaR, i.e. Y < v) cannot change any score.

**Hand-calculated reference values** (computed this session):

| α | v | e | Y | FZ0 |
|---|---|---|---|---|
| 0.05 | −1.644854 | −2.062713 | −1 (no hit) | 0.521445 |
| 0.05 | −1.644854 | −2.062713 | −2 (hit) | 3.964933 |
| 0.05 | −1.644854 | −2.062713 | −3 (hit) | 13.660902 |
| 0.05 | −1.644854 | −2.062713 | Y = v (tie) | 0.521445 |
| 0.01 | −2.326348 | −2.665214 | −2 (no hit) | 0.853140 |
| 0.01 | −2.326348 | −2.665214 | −3 (hit) | 26.128862 |

Worked example for the α=0.05, Y=−3 row: −1/(0.05·(−2.062713)) = 9.695975; times (v − Y) = 1.355146 gives 13.139457; adding v/e + ln(−e) − 1 = 0.521445 gives 13.660902.

**Analytic identity (my derivation, not taken from the paper).** If Y is continuous and (v, e) are the true return-space VaR and ES, then E[(v−Y)·1{Y≤v}] = α(v − e). Substituting gives **E[FZ0] = ln(−e)**.
- Seeded check (seed 20261111, 400,000 N(0,1) draws):
  - α = 0.05: mean 0.72567 (se 0.0019) vs ln(2.062713) = 0.72402.
  - α = 0.01: mean 0.98371 (se 0.0028) vs 0.98028.
- Scaling both forecasts by 0.9 raised the mean score (0.7477 and 1.0276), as consistency requires.

**Tests.**
1. Hand values from the table above, to tolerance 1e-6.
2. Tie continuity.
3. The identity E[FZ0] = ln(−e) by `scipy.integrate.quad` over the N(0,1) density. This is deterministic, uses tolerance 1e-8, and is independent of the scoring code path.
4. Consistency: the expected score (by quadrature) on a (v, e) grid is minimised at the true values for α ∈ {0.05, 0.01}.
5. Zero-degree homogeneity of loss differences, the paper's defining property: `FZ0(cY,cv₁,ce₁) − FZ0(cY,cv₂,ce₂) = FZ0(Y,v₁,e₁) − FZ0(Y,v₂,e₂)` for c > 0.
6. Conversion layer: loss-space inputs (VaR_L=1.644854, ES_L=2.062713, L=3) pass through `to_return_space` and give 13.660902.
7. Invalid inputs raise: v = 0, v > 0, e > v, NaN. e = v is accepted, as the boundary allowed by e ≤ v.
8. The wrapper's count of inadmissible forecasts is correct on a constructed series.

Approach B (a native loss-space formula) is **not** implemented. Test 6 already proves the conversion is exact.

---

## 3. Threshold selection: safeguarded procedure

**Why low thresholds are preferred (established reasoning).** The GPD is an asymptotic approximation for excesses.
- Too low a threshold gives bias, because the approximation fails.
- Too high a threshold gives variance, because N_u is small.

Bader, Yan & Zhang (2018) state this trade-off in their abstract (verified), and Scarrott & MacDonald (2012) review it. The standard prescription is the **lowest threshold above which the GPD is adequate and estimates are stable** (Coles 2001, ch. 4; full text to be checked). Once bias is negligible, raising the threshold further only discards data. "Lowest" is therefore correct only *conditional on* an adequacy test and a stability check. It is never applied as "lowest passing" on its own.

**Procedure.** It runs on training data only and its constants are pre-registered.

- **Step 0: minimum exceedances.** N_min is fixed *before* seeing data, by a Day-6 simulation study.
  - Design: true ξ ∈ {0.1, 0.25, 0.4}, β = 1, N_u ∈ {50, 75, 100, 150}, 2,000 replications, seed 20261111.
  - Rule: N_min is the smallest N_u with MLE convergence ≥ 99%, RMSE(ξ̂) ≤ 0.15 and relative RMSE(VaR99) ≤ 15%, at every true ξ.
  - This gives q_max = 1 − N_min/W for the rolling window.
  - Expectation, not yet a result: N_min ≈ 75–100, so q_max ≈ 0.90–0.925.
- **Step 1: candidate grid.** q_k = 0.80, 0.81, …, 0.95 (k = 1…16, ascending, so H_1 is the lowest threshold). The training thresholds are u_k = the training q_k-quantile. Training N_u ranges from 400 down to 100.
- **Step 2: Anderson–Darling (AD) GoF p-value per candidate.**
  - The null H_k is "excesses over u_k follow a GPD".
  - p_k is a parametric-bootstrap p-value from `scipy.stats.goodness_of_fit` (GPD, loc fixed at 0, refit in each replicate, 999 reps, seed 20261111). Day 7 checks that the replicate refit fixes loc = 0.
- **Step 3: multiplicity control with ForwardStop** (G'Sell et al. 2016, eq. 6, read in full text), as used for threshold selection by Bader et al. (2018).
  - `k̂ = max{k ∈ 1…m : −(1/k) Σ_{i≤k} ln(1 − p_i) ≤ α_FS}`, with k̂ = 0 if the set is empty.
  - The first k̂ candidates are rejected; q_FS = q_{k̂+1}.
  - α_FS = 0.10 is recommended (decision D4b). Because AD power is low at a few hundred exceedances, a higher α_FS gives better protection against accepting a biased low threshold.
- **Step 4: stable region.** Starting at q_FS, q\* is the **smallest** q ≥ q_FS, q ≤ q_max, such that every candidate in [q, q+0.03] (4 grid points) satisfies all of these, relative to q:
  - |ξ̂(q′) − ξ̂(q)| ≤ 0.10;
  - |VaR99(q′)/VaR99(q) − 1| ≤ 5%;
  - |ES99(q′)/ES99(q) − 1| ≤ 10%;
  - p_{q′} is not rejected by ForwardStop.
- **Step 5: training-split consistency.** Steps 2–4 are rerun on each training half (returns 1–1000 and 1001–2000). The resulting q values are reported. A difference of more than 0.03 from q\* is flagged and brought to you.
- **Fallback.** If no q qualifies, nothing is chosen automatically. It is recorded as a rule failure and you decide, with the full diagnostic table in front of you.

**Safeguards.**
- **Multiple testing.** ForwardStop controls FDR across the ordered candidates.
  - Limitation: its guarantee (Theorem 2) assumes the null p-values are independent and uniform. Nested thresholds share exceedances, so their p-values are positively dependent and the formal guarantee does not hold here.
  - Bader et al. assess the procedure by simulation, so we treat it as an established, automated, multiplicity-aware rule rather than an exact FDR guarantee.
  - Its empirical behaviour under independence and under dependence is measured by the A3 simulation study and reported as simulation evidence.
- **Interpreting GoF p-values.**
  - A large p-value means only "no evidence against the GPD at this N_u". It does not confirm the GPD.
  - AD power falls as N_u falls, so p-values naturally rise with q. That is why GoF alone is never used to select and is always paired with stability.
  - **Withdrawn in V2.2:** the earlier claim of a known direction for the effect of dependence on AD p-values. The direction is unknown and is measured by the A3 simulation study.
- **Stability tolerances.**
  - The ξ tolerance 0.10 is about one asymptotic standard error at the training N_u in use. SE(ξ̂) ≈ (1+ξ)/√N_u (standard GPD MLE asymptotics, Smith 1985 / Davison & Smith 1990; full text to be checked). At ξ = 0.2 this is 0.085 for N_u = 200 and 0.10 for N_u = 144.
  - Differences below one SE are treated as practically indistinguishable.
  - Nested estimates are positively correlated, so their differences vary less than the individual SEs suggest. The tolerance is therefore lenient, and it is a **practical criterion, not a hypothesis test**.
  - The VaR (5%) and ES (10%) tolerances express economic materiality. ES gets the wider tolerance because it is more sensitive to ξ.
  - All constants are pre-registered in `configs/base.toml`.
- **Minimum exceedances.** These are set by simulation (Step 0), not convention, and are enforced at the rolling-window level, where N_u is half the training N_u.
- **VaR/ES sensitivity.** VaR and ES at 95% and 99%, with CIs, are reported for every candidate.
- **Selection uncertainty.** This is quantified three ways:
  - (i) the spread of VaR and ES across the stable region [q\*, q\*+0.03];
  - (ii) the training-half q values from Step 5;
  - (iii) declared OOS sensitivity runs at q\* − 0.02, q\* + 0.02 and q\* + 0.03.
  - A nested bootstrap of the whole procedure (outer block bootstrap times inner GoF bootstrap) is **deferred**. It would cost about 200 × 16 × 999 refits and adds no decision value over (i)–(iii). It can be added if the stable region turns out to be narrow.

**Full candidate diagnostic report.** Written to `reports/threshold_selection/` and committed with q\* before any OOS run. It contains one row per candidate:

> q, u, N_u (training), N_u (window), ξ̂, SE, profile CI, β̂, β\* = β̂ − ξ̂u, AD statistic, AD p, ForwardStop running statistic, rejected?, VaR95/ES95/VaR99/ES99 with CIs, MLE convergence share across training windows, flags

It also includes these plots:
- mean residual life (MRL);
- ξ̂ and β\* stability;
- VaR and ES against q;
- AD p-values with the ForwardStop boundary.

**Fixed vs recalibrated threshold.** The V2 decision is unchanged: fix the quantile level q\* and recompute u_t in each window.
- The bias of the GPD approximation depends on how deep into the tail the threshold sits, which is the quantile level. Variance depends on N_u.
- Transferring q, not u, from the 2,000-point training set to 1,000-point windows therefore keeps the bias property, while variance is controlled separately by N_min at window size.

---

## 4. Backtesting simulation specification (reproducible)

**Kupiec proportion-of-failures (POF) test.** For T OOS days, x breaches and nominal breach probability p0, with p̂ = x/T and the convention 0·ln 0 = 0:

`LR_POF = −2·[ (T−x)·ln(1−p0) + x·ln(p0) − (T−x)·ln(1−p̂) − x·ln(p̂) ]`

- Hypotheses: H0: p = p0 against H1: p ≠ p0. The test is **two-sided**, because the LR penalises excess and deficit breaches alike.
- Significance 5%: reject if LR > χ²₁(0.95) = **3.841459**.
- Reported alongside it: the exact two-sided binomial p-value and the observed breach rate.

**Exact acceptance regions** at T = 5,650 (enumerated over x = 0…T):
- p0 = 0.05: x ∈ [251, 315];
- p0 = 0.01: x ∈ [43, 71].

**Size and power under i.i.d. hits.** Exact values are computed by summing the binomial pmf over the rejection region. A Monte Carlo (MC) confirmation uses `random.binomialvariate`, R = 100,000, seed **20261111** (computed this session):

| p0 | true p | exact | MC (se) |
|---|---|---|---|
| 0.01 | 0.0100 (size) | 0.0523 | 0.0516 (0.0007) |
| 0.01 | 0.0125 | **0.4508** | 0.4497 (0.0016) |
| 0.01 | 0.0150 | **0.9295** | 0.9295 (0.0008) |
| 0.01 | 0.0200 | 1.0000 | 1.0000 |
| 0.05 | 0.0500 (size) | 0.0472 | 0.0478 (0.0007) |
| 0.05 | 0.0550 | **0.3879** | 0.3877 (0.0015) |
| 0.05 | 0.0600 | **0.9070** | 0.9060 (0.0009) |
| 0.05 | 0.0700 | 1.0000 | 1.0000 |

Interpretation: a 25% under-forecast of breach frequency (1.25% vs 1%) is detected less than half the time.

**Two kinds of uncertainty, kept separate.**
- *Conventional binomial uncertainty* assumes independent hits: Var(x) = T·p0·(1−p0), a Clopper–Pearson CI for the breach rate, and the table above.
- *Dependence-aware uncertainty* models hits as a two-state Markov chain with the **correct** unconditional rate p0 and persistence π11 = P(breach | breach yesterday). Then π01 = p0(1−π11)/(1−p0), the lag-1 autocorrelation is ρ = π11 − π01, and asymptotically Var(x) ≈ T·p0(1−p0)·(1+ρ)/(1−ρ).
- The **true size of Kupiec's nominal 5% test** under this model was computed exactly, by dynamic programming over (count, state):

| p0 | π11 | ρ | variance inflation | true size of nominal 5% Kupiec |
|---|---|---|---|---|
| 0.01 | 0.01 (i.i.d.) | 0.000 | 1.000 | 0.052 |
| 0.01 | 0.05 | 0.040 | 1.084 | 0.062 |
| 0.01 | 0.10 | 0.091 | 1.200 | 0.076 |
| 0.01 | 0.20 | 0.192 | 1.475 | 0.110 |
| 0.05 | 0.10 | 0.053 | 1.111 | 0.060 |
| 0.05 | 0.20 | 0.158 | 1.375 | 0.091 |
| 0.05 | 0.30 | 0.263 | 1.714 | 0.130 |

MC check (p0 = 0.01, π11 = 0.10, R = 2,000, seed 20261111): 0.0795 (se 0.006), which agrees with the exact 0.0764.

This quantifies V2's claim: clustered breaches make Kupiec over-reject even when the unconditional rate is right.

**Applied to the real results.**
- We estimate π̂11 from the realised hit sequence of each model.
- We report the dependence-adjusted Kupiec size at that π̂11.
- We report a moving-block-bootstrap CI for the breach rate, with block length fixed by rule and seed fixed, next to the binomial CI.

**Christoffersen tests.** MC p-values: 9,999 i.i.d. Bernoulli(p̂) sequences of length T, seed 20261111. The first-order-Markov blind spot is documented, and inter-breach durations are shown descriptively.

**Implementation.**
- `backtest.py` provides `kupiec_lr`, `kupiec_power_exact(T, p0, p1)` and `kupiec_size_markov(T, p0, π11)` (the dynamic program). Seeds and replication counts come from `[backtest]` in the config.
- Tests reproduce every number in the two tables above.
- Kupiec check from V2: LR(T=250, x=5, p0=0.01) = 1.9568.

---

## 5. Training period justification, and data licensing

### 5a. Why T0 = 2000

- **Different jobs.**
  - The training set chooses a design constant (q\*) and supports diagnostics, which needs diagnostic power and a resolvable stability check.
  - The rolling window (W = 1000) estimates parameters for each forecast, which trades adaptivity against variance.
  - The two needs differ, so the sizes differ.
- **Sufficiency, quantified.**
  - The stability tolerance (Δξ ≤ 0.10) is only meaningful if SE(ξ̂) ≲ 0.10, which needs N_u ≳ 144 at ξ ≈ 0.2.
  - With T0 = 2000 that holds for q ≤ 0.928: N_u = 400 → SE ≈ 0.06 at q = 0.80, and N_u = 100 → SE ≈ 0.12 at q = 0.95.
  - With T0 = 1000, N_u = 200…50 and SE ≈ 0.085…0.17, so over most of the grid the tolerance would be smaller than the noise.
  - 2000 is therefore the smallest round size that makes the stability step informative. The cost is about 1,000 OOS days, roughly 10 fewer expected 99% breaches.
- **Regime representativeness.**
  - The training span is about Nov-1995 to late 2003. Context from memory, to be confirmed against the data: it includes the 1997–98 Asian crisis and the 2000–01 dot-com and domestic market turmoil. It excludes 2008 and 2020, which fall in OOS.
  - The selected *level* q\* adapts to volatility, because u_t rescales in each window. It does not adapt to changes in tail *shape*.
- **Defensibility through OOS.** This is checked without using OOS to choose anything.
  - (i) The training-half consistency check (§3, Step 5).
  - (ii) A post-hoc, clearly labelled rerun of the selection procedure on the non-overlapping OOS blocks of 2,000 returns, reporting the q each block would have chosen. If the blocks drift away from q\*, that is reported as a limitation. The primary specification is never changed; any change becomes a new, separately reported configuration.
  - (iii) The OOS sensitivity runs at the stable-region endpoints.
  - (iv) The rolling path of ξ̂ with CIs.

### 5b. Data licensing

These are statements of what the text says, not legal conclusions. The pages were read this session.

*Terms of Use* (niftyindices.com/terms-of-use):
- "The Site may be used only for lawful purposes by individuals for personal, non-commercial use only."
- "No material from the Site may be copied, modified, reproduced, republished, uploaded, transmitted, posted or distributed in any form without prior written permission from the Company."

*Disclaimer* (niftyindices.com/disclaimer):
- "No part of this Site or information contained herein may be reproduced, stored in a retrieval system or transmitted in any form or by any means … without prior written permission of the Company or its applicable group company."
- "Use or distribution of the Company's … index data and the use of their index data to create financial products require a license from NSE Indices Limited and/or its third party licensors."

| Use | What the text says | Status |
|---|---|---|
| Local use (download and analyse on your PC) | Personal, non-commercial site use is permitted. The Disclaimer's "stored in a retrieval system" and "Use … of index data … require a license" wording is broad, and the text does not say whether it covers personal analysis. | **Unclear from the text** |
| Academic research | Neither page mentions academic or research use. | **Not addressed** |
| Redistribution of data | Requires prior written permission ("copied … distributed"). | **Not permitted without permission** → never done |
| Publishing derived results | Not addressed explicitly. "Modified … republished" and "use … of index data" could plausibly reach derived series that track the index (returns, daily VaR paths). The text is silent on aggregate statistics. | **Unclear from the text** |

Consequences built into the plan:
- Raw and processed data are gitignored from day 1, and a test asserts that `git ls-files data/` is empty.
- No data, and no derived per-day series, are uploaded anywhere.
- Before any publication: seek written permission from NSE Indices, or obtain the data through a licensed channel. The site lists "Authorised Data Vendors", and your institution may hold a subscription.
- Whether to download for local research now, given the unclear status, is **your decision** (D11).

---

## 6. Updated architecture

```
src/niftytail/
  config.py       tomllib → frozen dataclasses; validation of all pre-registered constants
  data.py         load_raw, validate → ValidationReport, manifest sha256 check          (DataValidationError)
  losses.py       log_returns, log_losses
  gpd.py          nll, fit_mle (PWM start), fit_pwm, tail_var, tail_es, profile_ci, delta_ci, se_xi   (EstimationError)
  threshold.py    candidate_table, gof_ad_bootstrap, forward_stop, stable_region, select_threshold(train_only) → ThresholdDecision
  simstudy.py     (V2.2) DGPs S1–S4 + N_min study + A3 threshold study runner (ProcessPoolExecutor, SeedSequence.spawn)
  diagnostics.py  acf/ljung_box/arch_lm (statsmodels), extremal_index_fs, decluster_fs, block_bootstrap (A5c)
  models.py       historical, gaussian, pot_gpd : (window, cfg) → Forecast (loss space); MODELS dict
  rolling.py      run_rolling(losses, model_fn, W, T0, cfg) → DataFrame (index = target date); enforces split
  backtest.py     hits, kupiec_lr, kupiec_power_exact, kupiec_size_markov, christoffersen(+MC), traffic_light,
                  es_residual_test, to_return_space, fz0, common_mask (A2), score_models, diebold_mariano, block_ci_rate
  plots.py        Figure builders
  experiment.py   run(config) → reports/runs/<ts>_<name>/ {config.toml, manifest.json, forecasts.csv, backtest.csv, figures/, run.log}
  __main__.py     python -m niftytail {validate | select-threshold | run} <config>
app/dashboard.py  read-only Streamlit over reports/runs/
reports/threshold_selection/   committed pre-registration evidence (no raw data)
```

**Invariants enforced in code and tests:**
- `threshold.py` raises if given data beyond T0.
- `fz0` operates only in return space and raises on inadmissible inputs.
- `rolling.py` future-perturbation invariance.
- No raw data tracked by git.

`configs/base.toml` holds every constant named in §3–§4: grid, N_min (after Day 6), α_FS, tolerances, gof_reps, mc_reps, seeds, T0, W and levels.

---

## 7. Updated two-week roadmap

| Day | Track | Deliverable | Acceptance |
|---|---|---|---|
| 1 | Research | Read the full or open texts: Scarrott–MacDonald, Bader et al., the rest of PZC, Coles ch. 4–5 if accessible. Indian-market search. `methodology.md` (§1–§4 formulas plus hand checks), `literature.md`, `decisions.md` | every claim we rely on is C or explicitly marked F; your review |
| 2 | Setup | §9 sequence; config + logging; tooling green | verification commands pass; `git ls-files data/` empty; first local commit |
| 3 | Data | you download (subject to D11); loader, validation, manifest; spot-checks; methodology PDF; licence quotes in `data.md` | flags explained or listed; coverage confirmed |
| 4 | Dependence | `losses.py`; EDA; dependence battery on training | figures from one command; findings written |
| 5 | Statistics | `gpd.py` core | hand cases, scipy cross-check, ES-vs-quadrature pass |
| 6 | Statistics | uncertainty, edge cases, MC recovery; **N_min simulation study** | N_min fixed by the pre-declared criterion and committed |
| 7 | **Gate** | `threshold.py`; full candidate table; ForwardStop; stable region; split-half check; extremal index and declustering | **q\* plus evidence committed before any OOS run; your sign-off** |
| 8 | Statistics | `models.py` | baseline and equivariance tests |
| 9 | Evaluation | `rolling.py` | leakage and alignment tests |
| 10 | Evaluation | `backtest.py` incl. FZ0 (§2 tests), Kupiec exact/Markov (§4 tables reproduced), Christoffersen MC | all reference values reproduced |
| 11 | Evaluation | primary run + declared sensitivities + post-hoc OOS-block threshold check | results inspected with you |
| 12 | Visualization | figures and tables | traceable to run_id |
| 13 | Dashboard | read-only Streamlit | loads all runs; no estimation imports |
| 14 | Docs | README reproduction, report draft, limitations, `status.md` | clean rerun gives byte-identical `forecasts.csv` |

---

## 8. Outstanding risks

1. **EVT ≈ HS at 99% by construction** (VaR99 is the 10th-largest loss in a 1,000-day window). The far-tail advantage is not backtestable.
2. **Kupiec over-rejects under clustering.** Under the two-state Markov hit model, true size is 0.076–0.13 for the persistence values examined. Christoffersen independence rejection is expected for all unconditional models.
3. **Low power.** Kupiec detects a 1.25% true breach rate only 45% of the time.
4. **ForwardStop's FDR guarantee is not formally valid** for nested thresholds. How dependence affects AD p-values is unknown. Behaviour is assessed only as simulation evidence (A3), never as a guarantee.
5. **Stability tolerances and N_min criteria are practical constants.** They are pre-registered and justified, but other reasonable values exist; sensitivity runs address this.
6. **Training regime (1995–2003) may not represent later tail shape.** This is monitored post hoc and never retuned.
7. **Licensing is unclear** for local use, research use and derived publications. This could block publication.
8. **Circuit-breaker censoring and holiday/missing ambiguity** remain to be verified in the Day-3 audit.
9. **Full-text verification pending** for several foundational claims (status F in §10).

---

## 9. Exact local setup sequence (Day 2, after approval)

This sequence was already given in V2. It runs in `c:\Users\Sambi\Desktop\Niftytail`. It creates no GitHub repository and no remote, and pushes nothing.

```
git init -b main
uv init --lib --name niftytail --python 3.13 --vcs none .
uv add numpy pandas scipy statsmodels matplotlib
uv add --group dashboard streamlit
uv add --dev pytest pytest-cov ruff mypy pandas-stubs
uv sync --all-groups
# write .gitignore (data/raw/*, data/processed/*, reports/runs/*, .venv/, __pycache__/, .coverage)
# write tool config in pyproject.toml; configs/base.toml; config.py; logging; tests/test_config.py; tests/test_repo_hygiene.py
uv run python -c "import sys; print(sys.version)"      # 3.13.x
uv run ruff check . ; uv run ruff format --check . ; uv run mypy src ; uv run pytest
git add -A ; git commit -m "Project scaffold"
git remote -v                                          # must be empty
git ls-files data/                                     # must be empty
```

---

## 10. Literature evidence matrix (V2 matrix plus additions)

Status codes:
- **M** = metadata verified via Crossref this session.
- **C** = claim checked against an abstract or full text this session.
- **F** = full text needed.
- **U** = unverified.

**New in V2.1:**
- Bader, Yan, Zhang (2018), *Automated threshold selection for extreme value analysis via ordered goodness-of-fit tests with adjustment for false discovery rate*, Ann. Appl. Stat. 12(1):310–329. DOI 10.1214/17-AOAS1092. Status **M, C (abstract)**: the bias–variance framing, the AD test plus an ordered FDR stopping rule, and the observation that earlier methods ignored ordered multiple testing.
- G'Sell, Wager, Chouldechova, Tibshirani (2016), *Sequential selection procedures and false discovery rate control*, JRSS-B 78(2):423–444. DOI 10.1111/rssb.12122. Status **M, C (full text, arXiv 1309.5352)**: the ForwardStop eq. (6), and Theorem 2's assumption of independent U(0,1) null p-values.

**Carried from V2:**

| Status | Entries |
|---|---|
| M + C | Hill 1975 (abstract); Davison & Smith 1990 (POT, abstract); Ferro & Segers 2003 (abstract); Fissler & Ziegel 2016 (abstract); Nolde & Ziegel 2017 (abstract); **Patton, Ziegel & Chen 2019 (full text)**; Karmakar 2013 (keywords) |
| M + F | Balkema & de Haan 1974; Pickands 1975; Smith 1985; Hosking & Wallis 1987; Grimshaw 1993; Coles 2001; Embrechts, Klüppelberg & Mikosch 1997; Choulakian & Stephens 2001; McNeil & Frey 2000; McNeil 1997; Gilli & Këllezi 2006; Engle 1982; Bollerslev 1986; Artzner et al. 1999; Acerbi & Tasche 2002; Kupiec 1995; Christoffersen 1998; Christoffersen & Pelletier 2004; Gneiting 2011; Diebold & Mariano 1995 |
| M via journal page | Scarrott & MacDonald 2012 (DOI 10.57805/revstat.v10i1.110), C via a citing summary |
| URL via search | BCBS 22 (1996) |
| U | Acerbi & Székely 2014 (not cited) |

The full per-entry table is in V2 §4 and moves to `docs/literature.md` on Day 1.

---

## 11. Final decisions requiring approval

| # | Decision | Recommendation |
|---|---|---|
| D1 | Sample start | 3-Nov-1995 base date; exclude the back-calculated 1990–95 values |
| D2 | Series | PR Close primary; TRI Gross sensitivity |
| D3 | Split | T0 = 2000, W = 1000; W = 2000 sensitivity |
| D4a | Threshold procedure | §3 as specified (grid 0.80–0.95, step 0.01; AD bootstrap; ForwardStop; stable region of 4 points; split-half check) |
| D4b | ForwardStop level | α_FS = 0.10 |
| D4c | Tolerances | Δξ ≤ 0.10, ΔVaR99 ≤ 5%, ΔES99 ≤ 10% |
| D4d | N_min criterion | Day-6 simulation rule as in §3, Step 0 |
| D5 | Threshold across windows | fixed q\*, u recomputed per window |
| D6 | Levels | (VaR95, ES95), (VaR99, ES99) primary; 99.5% and 99.9% descriptive |
| D7 | FZ0 | approach A (return space internally); approach B not implemented |
| D8 | Kupiec reporting | LR + exact binomial p + dependence-adjusted size at π̂11 + block-bootstrap rate CI |
| D9 | Dependence | diagnostics + declustered-ξ comparison + block bootstrap; no GARCH in v1 |
| D10 | Tooling | §9 as written |
| D11 | Data acquisition under unclear licence | Your call. Options: (a) download from niftyindices for local, non-published research now and seek NSE permission before any publication; (b) seek permission first; (c) use a licensed vendor or institutional source. |
| D12 | Data in git | gitignored plus a hygiene test |

## Verification of this plan's numbers

All figures in §2 and §4 were computed this session with stdlib Python 3.13:
- exact enumeration and dynamic programming;
- seeded Monte Carlo (seed 20261111; R as stated).

The DOIs were resolved via Crossref, and the licence quotes were taken from the live pages on 2026-11-11. Day 1 re-derives everything independently in `methodology.md`, and Day 10 tests reproduce it.
