# NIFTYTAIL — Literature Evidence Matrix

**Status:** Day-1, 2026-11-11.

**Evidence codes:**
- **M**: metadata (authors, title, year, venue, volume and pages) verified via the Crossref API on 2026-11-11, unless another source is named.
- **C1**: the claim we rely on was checked against the **full text, read directly** in this work package.
- **C1-sections**: the specific statements cited were checked directly against the relevant sections of the original text; **not a full-paper read**.
- **C2**: the claim was checked against an **abstract**, or an automated **summary** of the full text.
- **F**: the claim requires full-text reading; it is attributed but not confirmed.
- **U**: unverified.

**Rules:**
- *Findings* and *limitations* are filled in only for texts actually read (C1). Every other cell says "not read".
- Established literature is kept separate from NIFTYTAIL's own results; no project results exist yet.
- No novelty claim is made.

## A. Matrix

### Extreme value theory foundations

| ID | Reference | DOI / URL | Meta | Claim relied on | Claim | Methodology / findings / limitations | Relevance |
|---|---|---|---|---|---|---|---|
| L01 | Balkema, A.A., de Haan, L. (1974). Residual life time at great age. *Ann. Probab.* 2(5):792–804 | 10.1214/aop/1176996548 | M | GPD limit for excesses over high thresholds | F (abstract mentions limit laws for residual life) | not read | theoretical basis of POT |
| L02 | Pickands, J. III (1975). Statistical inference using extreme order statistics. *Ann. Statist.* 3(1):119–131 | 10.1214/aos/1176343003 | M | GPD tail approximation | F | not read | theoretical basis of POT |
| L03 | Hill, B.M. (1975). A simple general approach to inference about the tail of a distribution. *Ann. Statist.* 3(5):1163–1174 | 10.1214/aos/1176343247 | M | tail inference conditional on the extreme order statistics (used only for EDA) | C2 (abstract) | not read | exploratory tail-index estimate |
| L04 | Smith, R.L. (1985). Maximum likelihood estimation in a class of nonregular cases. *Biometrika* 72(1):67–90 | 10.1093/biomet/72.1.67 | M | MLE regular for ξ > −½; asymptotic variance | F | not read | validity of GPD MLE and SEs |
| L05 | Davison, A.C., Smith, R.L. (1990). Models for exceedances over high thresholds. *JRSS-B* 52(3):393–425 | 10.1111/j.2517-6161.1990.tb01796.x | M | POT modelling of sizes and occurrence of exceedances; MRL plot | C2 (POT, abstract) / F (MRL) | not read | core POT methodology |
| L06 | Hosking, J.R.M., Wallis, J.R. (1987). Parameter and quantile estimation for the GPD. *Technometrics* 29(3):339–349 | 10.1080/00401706.1987.10488243 | M | PWM estimator | F | not read | MLE starting values; cross-check |
| L07 | Grimshaw, S.D. (1993). Computing maximum likelihood estimates for the GPD. *Technometrics* 35(2):185–191 | 10.1080/00401706.1993.10485040 | M | MLE numerics (optional) | F | not read | numerical robustness |
| L08 | Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer | 10.1007/978-1-4471-3675-0 | M | threshold diagnostics; profile likelihood; dependence (ch. 5); lowest adequate threshold | F (book not accessible in this work package) | not read | primary textbook |
| L09 | Embrechts, P., Klüppelberg, C., Mikosch, T. (1997). *Modelling Extremal Events*. Springer | 10.1007/978-3-642-33483-2 | M | theory reference | F | not read | background |

### Threshold selection

| ID | Reference | DOI / URL | Meta | Claim relied on | Claim | Methodology / findings / limitations | Relevance |
|---|---|---|---|---|---|---|---|
| L10 | Scarrott, C., MacDonald, A. (2012). A review of extreme value threshold estimation and uncertainty quantification. *REVSTAT* 10(1):33–60 | 10.57805/revstat.v10i1.110; open access at ine.pt/revstat/pdf/rs120102.pdf | M (journal page + PDF header; not Crossref) | threshold choice balances bias (asymptotic approximation) against variance (sparse excesses) | **C1** (full text, pdftotext, 2026-11-11) | *Method:* review of fixed-threshold diagnostics (MRL, stability plots), rules of thumb, GoF-based, mixture and Bayesian approaches. *Finding:* in their rainfall example, thresholds u = 0.395 and u = 1.2 both satisfy MRL guidance but give ξ̂ = 0.21 vs 0.003, and stability plots show several plateaus. *Limitation:* a review; graphical diagnostics are subjective. | justifies the bias–variance rationale; warns of multiple plateaus (methodology §7 risk) |
| L11 | Ferro, C.A.T., Segers, J. (2003). Inference for clusters of extreme values. *JRSS-B* 65(2):545–556 | 10.1111/1467-9868.00401 | M | extremal index estimable before declustering; automatic declustering; bootstrap for cluster characteristics | C2 (abstract) | not read | dependence diagnostics |
| L11a | Ferro, C.A.T. (2003). *Statistical Methods for Clusters of Extreme Values*. PhD thesis, Lancaster University | empslocal.ex.ac.uk/people/staff/ferro/Publications/Thesis/ (chapter4.pdf) | M (title page of the thesis file; not Crossref) | intervals estimator θ̂, formulas (4.5), (4.8), (4.9); intervals declustering: the C − 1 = ⌊θ̂(N − 1)⌋ largest inter-exceedance times are inter-cluster times, T_(C) := 0 if C = N, ties decrease C, equal to runs declustering with r = T_(C) | **C1-sections** (the estimator formulas and §4.1.5, text extracted with pypdf, 2026-11-12); the JRSS-B paper L11 itself was not read | not read in full | `decluster_fs` convention (D36 (c)); agrees with the θ̂ formula of D29 (7) |
| L12 | Choulakian, V., Stephens, M.A. (2001). Goodness-of-fit tests for the GPD. *Technometrics* 43(4):478–484 | 10.1198/00401700152672573 | M | AD and CvM GoF for the GPD with estimated parameters | F | not read | GoF background (DX-2) |
| L13 | Bader, B., Yan, J., Zhang, X. (2018). Automated threshold selection for extreme value analysis via ordered goodness-of-fit tests with adjustment for false discovery rate. *Ann. Appl. Stat.* 12(1):310–329. Preprint arXiv 1604.02024v1 (2016) | 10.1214/17-AOAS1092 | M | (a) bias–variance framing; (b) AD test plus an ordered stopping rule (ForwardStop/StrongStop) for automated selection; (c) "the p-values of the sequential tests here are dependent"; error control under dependence is supported by simulation, not proof; (d) "bootstrap based tests that put a lower bound on the p-values (1 divided by the bootstrap sample size) may lead to premature stopping"; "parametric bootstrap puts bounds on the approximate p-values which would invalidate the stopping rules"; they use a fast table-based approximation instead | (a, b) C2 (Crossref abstract); **(c, d) C1-sections**: checked directly against the original arXiv v1 text (pdftotext, 2026-11-11), in the introduction, method section, GoF-test section and concluding discussion. **Not a full-paper read**; the published version was not checked. | Only the cited statements were read; the full findings and limitations are not summarised | basis of Steps 2–3; DX-2 (D22) |
| L14 | G'Sell, M.G., Wager, S., Chouldechova, A., Tibshirani, R. (2016). Sequential selection procedures and false discovery rate control. *JRSS-B* 78(2):423–444. Preprint arXiv 1309.5352 | 10.1111/rssb.12122 | M | ForwardStop: k̂ = max{k : −(1/k)Σ_{i≤k} ln(1−p_i) ≤ α}; FDR control when null p-values are independent U(0,1) (Theorem 2) | **C1** (arXiv full text, pdftotext) | *Method:* Rényi-representation transform of ordered p-values plus BH. *Finding:* ForwardStop and StrongStop control FDR (StrongStop also FWER) under independence. *Limitation:* requires independent uniform nulls. | Step 3 of threshold selection |

### EVT in financial risk

| ID | Reference | DOI / URL | Meta | Claim relied on | Claim | Methodology / findings / limitations | Relevance |
|---|---|---|---|---|---|---|---|
| L15 | McNeil, A.J., Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. *J. Empirical Finance* 7(3–4):271–300 | 10.1016/S0927-5398(00)00012-8 | M | GPD tail VaR/ES formulas; conditional vs unconditional EVT; exceedance-residual ES test; rolling n/k design | F (preprint fetched; text not extractable) | not read | central reference; GARCH-EVT as phase-2 remedy |
| L16 | McNeil, A.J. (1997). Estimating the tails of loss severity distributions using extreme value theory. *ASTIN Bull.* 27(1):117–137 | 10.2143/AST.27.1.563210 | M | optional published reference case | F | not read | possible validation case |
| L17 | Gilli, M., Këllezi, E. (2006). An application of extreme value theory for measuring financial risk. *Comput. Econ.* 27(2–3):207–228 | 10.1007/s10614-006-9025-7 | M | applied POT template | F | not read | applied example |
| L18 | Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica* 50(4):987– | 10.2307/1912773 | M | ARCH-LM test | F | not read | dependence diagnostics |
| L19 | Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *J. Econometrics* 31(3):307–327 | 10.1016/0304-4076(86)90063-1 | M | GARCH (background; S4 simulation DGP) | F | not read | simulation DGP background |

### Conditional models (D38; added on Day 8A, 2026-11-13)

Metadata in this block is as given in the Day-8A task. It was not fetched or checked against Crossref (owner instruction), so Meta is U. No text was read.

| ID | Reference | DOI / URL | Meta | Claim relied on | Claim | Methodology / findings / limitations | Relevance |
|---|---|---|---|---|---|---|---|
| L15 | McNeil & Frey (2000), as in the row above | as above | M (row above) | additional claim for D38: two-stage GARCH-EVT (GARCH filter by QML, GPD on the standardized residuals, VaR/ES rescaled by the one-step μ and σ forecast); their mean equation is AR(1) (we use a constant mean, D38 (2)) | F | not read | GARCH-EVT model definition |
| L35 | Barone-Adesi, G., Giannopoulos, K., Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *J. Futures Markets* 19(5):583–602 | not recorded | U (as given; title from memory, not verified) | filtered historical simulation (HS on GARCH standardized residuals) | F | not read | FHS model definition |
| L36 | Bollerslev, T., Wooldridge, J.M. (1992). Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. *Econometric Reviews* 11(2):143–172 | not recorded | U (as given; title from memory, not verified) | Gaussian QML is consistent for correctly specified conditional mean and variance parameters under non-normal innovations; not efficient | F | not read | GARCH estimation, limitation 1 of methodology §18 |
| L37 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian J. Statistics* 6(2):65–70 | not recorded | U (as given; title from memory, not verified) | step-down procedure controlling the family-wise error rate without independence assumptions | F | not read | conditional comparison family, D38 (5) |
| L38 | Engle, R.F., Ng, V.K. (1993). Measuring and testing the impact of news on volatility. *J. Finance* 48(5):1749–1778 | not recorded | U (as given; title from memory, not verified) | sign-bias, negative-size-bias and positive-size-bias tests and their joint LM test on standardized residuals; the regression convention used here is pinned in D39 (e) and was not checked against the paper | F | not read | filter diagnostics, D39 (e) |
| L39 | Glosten, L.R., Jagannathan, R., Runkle, D.E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *J. Finance* 48(5):1779–1801 | not recorded | U (as given; title from memory, not verified) | GJR-GARCH: an extra ARCH coefficient after negative return shocks | F | not read | filter candidates F3 and F4, D39 (c) |
| L40 | Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics* 6(2):461–464 | not recorded | U (as given; title from memory, not verified) | BIC = −2 ln L + k ln n as a model-selection criterion | F | not read | filter selection rule, D39 (c) |

### Risk measures, elicitability and backtesting

| ID | Reference | DOI / URL | Meta | Claim relied on | Claim | Methodology / findings / limitations | Relevance |
|---|---|---|---|---|---|---|---|
| L20 | Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). Coherent measures of risk. *Math. Finance* 9(3):203–228 | 10.1111/1467-9965.00068 | M | coherence axioms; VaR not subadditive | F | not read | motivation for ES |
| L21 | Acerbi, C., Tasche, D. (2002). On the coherence of expected shortfall. *J. Banking & Finance* 26(7):1487–1503 | 10.1016/S0378-4266(02)00283-2 | M | ES coherent; ES as an integral of VaR | F | not read | ES definition (HS ES, §4) |
| L22 | Kupiec, P.H. (1995). Techniques for verifying the accuracy of risk measurement models. *J. Derivatives* 3(2):73–84 | 10.3905/jod.1995.407942 | M | POF LR test | F | not read | §10.1 |
| L23 | Christoffersen, P.F. (1998). Evaluating interval forecasts. *Int. Econ. Rev.* 39(4):841– | 10.2307/2527341 | M | independence and conditional-coverage LR tests | F | not read | §10.5 |
| L24 | Christoffersen, P., Pelletier, D. (2004). Backtesting value-at-risk: a duration-based approach. *J. Fin. Econometrics* 2(1):84–108 | 10.1093/jjfinec/nbh004 | M | duration test (cited as a limitation only) | F | not read | §10.5 |
| L25 | Basel Committee on Banking Supervision (1996). Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements (BCBS 22) | https://www.bis.org/publ/bcbs22.pdf | URL (web search, 2026-11-11; not Crossref) | traffic-light zones for 250 observations | F | not read | §10.6 |
| L26 | Gneiting, T. (2011). Making and evaluating point forecasts. *JASA* 106(494):746–762 | 10.1198/jasa.2011.r10138 | M | ES is not elicitable | F | not read | §11 rationale |
| L27 | Fissler, T., Ziegel, J.F. (2016). Higher order elicitability and Osband's principle. *Ann. Statist.* 44(4):1680–1707 | 10.1214/16-AOS1439 | M | (VaR, ES) jointly elicitable under mild conditions | C2 (abstract) | not read | §11 |
| L28 | Patton, A.J., Ziegel, J.F., Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and Value-at-Risk). *J. Econometrics* 211(2):388–413. Preprint arXiv 1707.05108 | 10.1016/j.jeconom.2018.10.008 | M | FZ0 eq. (6); VaR and ES strictly negative; e ≤ v (fn. 1); FZ0 the unique zero-homogeneous FZ loss (Prop. 1); DM tests on out-of-sample FZ0 losses | **C1** (arXiv full text, pdftotext; the journal equation number is F) | *Method:* FZ-loss M-estimation of dynamic VaR–ES models. *Finding (their data):* GAS/hybrid ES–VaR models outperform GARCH and rolling-window forecasts on four equity indices; a 500-day rolling window was the worst model. *Limitation:* zero-homogeneity relies on negative VaR and ES. | FZ0 definition (§11) |
| L29 | Nolde, N., Ziegel, J.F. (2017). Elicitability and backtesting: perspectives for banking regulation. *Ann. Appl. Stat.* 11(4):1833–1874 | 10.1214/17-AOAS1041 | M | comparative backtests need elicitable measures; traditional backtests are unsuited to method comparison | C2 (abstract) | not read | §11 rationale |
| L30 | Diebold, F.X., Mariano, R.S. (1995). Comparing predictive accuracy. *JBES* 13(3):253–263 | 10.1080/07350015.1995.10524599 | M | DM test | F | not read | §12 |
| L31 | Newey, W.K., West, K.D. (1994). Automatic lag selection in covariance matrix estimation. *Rev. Econ. Stud.* 61(4):631–653 | 10.2307/2297912 | M | HAC lag selection; attributing the rule ⌊4(T/100)^{2/9}⌋ is F | F | not read | §12 |

### Indian-market prior work

| ID | Reference | DOI / URL | Meta | Claim relied on | Claim | Methodology / findings / limitations | Relevance |
|---|---|---|---|---|---|---|---|
| L32 | Karmakar, M. (2013). Estimation of tail-related risk measures in the Indian stock market: an extreme value approach. *Rev. Financial Econ.* 22(3):79–85 | 10.1016/j.rfe.2013.05.001 | M | EVT, POT, GARCH, VaR and ES applied to the Indian market | C2 (RePEc keywords and a search summary only) | not read | closest prior Indian study; must be read before any comparison or novelty statement |
| L33 | Paul, S., Sharma, P. (2021). Forecasting gains by using extreme value theory with realised GARCH filter. *IIMB Management Review* 33(1):64–70 | 10.1016/j.iimb.2021.03.011 | M | VaR/ES forecasting for the Nifty with realised-GARCH-EVT | F (content seen only in a web-search summary) | not read | conditional EVT on Nifty; phase-2 context |
| L34 | Das, S. (2025). Predicting stock market crash with Bayesian generalised Pareto regression. arXiv 2506.17549 (preprint, no journal reference listed) | arXiv:2506.17549 | M (arXiv listing, not Crossref) | Bayesian GPD regression of Nifty 50 extreme losses on volatility covariates | C2 (arXiv abstract) | not read; **not peer-reviewed** | Nifty tail-risk context |

### Not cited

| ID | Reference | Status | Note |
|---|---|---|---|
| — | Acerbi, C., Székely, B. (2014). Backtesting expected shortfall. *Risk* (December) | U | no DOI; deferred; not cited |

## B. Leads from the Indian-market search (2026-11-11), not yet verified (U)

These came from one web search. They are **not** cited until their metadata is verified:
- Paul, S. and Karmakar, M. (2017), *Multinational Finance Journal* 21(4), component GARCH-EVT for intraday VaR and ES (multi-market).
- An *International Business & Economics Research Journal* (Clute) BRICS GPD VaR/ES study (2015); authors not recorded.
- Umme Hani (2025), *Reliability: Theory & Applications* (gnedenko.net), Nifty VaR/ES with realised GARCH-EVT.
- Jitender (2021), *Acta Universitatis Sapientiae, Economics and Business*: Nifty VaR by HS and bootstrap HS (not EVT).

## C. Separation of evidence

- **Established methodology** (attributed above with status codes): GPD/POT theory, VaR/ES definitions, backtests, FZ0 scoring, ForwardStop.
- **Our hypotheses** (V2, untested):
  - H1: 0 < ξ̂ < ½ across windows.
  - H2: POT reaches Kupiec-acceptable 99% coverage where the Gaussian does not.
  - H3: all unconditional models reject Christoffersen independence.
  - H4: POT has a lower mean FZ0 than HS at 99%.
- **Our results:** none yet.

## D. Day-1 reading outcome

| Item | Result |
|---|---|
| Read in full (C1) | Scarrott & MacDonald (L10), G'Sell et al. (L14), Patton, Ziegel & Chen (L28) |
| Bader et al. (L13) | Day-1 closeout: the dependence and bootstrap statements were checked directly in the arXiv v1 text (C1-sections). A full reading is still recommended before Day 7. |
| Not accessible in this work package | Coles (2001) (L08, book) and McNeil & Frey (2000) (L15, the preprint PDF uses fonts that cannot be extracted). Both stay F until the full text is available. |
