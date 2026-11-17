# NIFTYTAIL — Day 11 results review

**Owner-approved interpretation of Day 11 (D46); numbers copied, not recomputed.**

Every number below was checked against the tracked JSON (6 significant digits). A p-value written "<= 1e-4" is the Monte Carlo floor 1/(B+1) with B = 9999.

Sources (sha256 of the file read):
- `reports/oos/w1000/backtest.json`: `bc79fd419d16ae5ac2cf03800ef6b505088bd9ea27fc559078d78cdb35b7f20a`
- `reports/oos/w2000/backtest.json`: `f900783ca58644aae3fba27fe51f6b936107959ed01366178ee64e29b44b5f12`
- `reports/threshold_selection/oos_blocks/selection.json`: `5a64bb708a7714f13fdedc7846857c7d8c7b24e6f4c9a68a755b94e75d3cd49c`

Expected breaches: 285.8 at 95%, 57.16 at 99%, 5.716 at 99.9%. Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.*.*.n`; 5716 × (0.05, 0.01, 0.001).

## Findings (a)

**H.** Only fhs and garch_evt are not rejected at 5% by any pre-declared coverage test (Kupiec, Christoffersen IND and CC, ES residual) at both primary levels in both windows.

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.{0.95,0.99}.<key>.kupiec.reject`, `.christoffersen.p_ind`, `.christoffersen.p_cc`, `.es_residual.p`.

**F1.** Conditioning dominates. gaussian, hs and pot_gpd reject IND at both levels and both windows (p <= 1e-4, except pot_gpd W2000 0.99 p = 0.0025); garch_n, fhs and garch_evt do not (min p = 0.1491). garch_evt beats pot_gpd on FZ0 in all four level x window cells (DM family (ii), Holm p <= 0.0109754).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.{0.95,0.99}.<key>.christoffersen.p_ind`; `dm.conditional[a=garch_evt, b=pot_gpd].p_holm`, `.d_bar`.

**F2.** Given conditioning, the tail shape matters. At 0.99, garch_n rejects Kupiec (98 and 86 breaches; p 8.32291e-07 and 0.000358255) and the ES test (p <= 3e-4) in both windows; garch_evt rejects neither. On FZ0, garch_evt beats garch_n at W1000 (Holm 0.0464471 at 0.95, 0.0243432 at 0.99) but not at W2000 (Holm 0.060491 at both levels).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.0.99.{garch_n,garch_evt}.breaches`, `.kupiec.p`, `.kupiec.reject`, `.es_residual.p`; `dm.conditional[a=garch_evt, b=garch_n].p_holm`, `.reject`.

**F3.** garch_evt vs fhs: not rejected at 0.99 (Holm 0.544465, 0.920528). At 0.95 rejected at W2000 only (Holm 0.0122724, d_bar -0.00159999). Descriptive 0.999: garch_evt 10 / 4 breaches, fhs 15 / 8 (W1000 / W2000).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `dm.conditional[a=garch_evt, b=fhs].p_holm`, `.reject`, `.d_bar`; `coverage.0.999.{garch_evt,fhs}.breaches`.

**F4.** pot_gpd vs hs is not rejected in any of the four cells (DM family (i)).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `dm.unconditional[a=hs, b=pot_gpd].reject`.

**F5.** Threshold robustness. garch_evt 0.99 breaches: 62 at every q_c key (W1000); 54-55 (W2000). All 12 garch_evt sensitivity DM tests (family (iii)) are not significant at 5%. D45: risk_material false in all six block entries.

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.0.99.garch_evt*.breaches`; `dm.descriptive[a=garch_evt@q, b=garch_evt].p`; `reports/threshold_selection/oos_blocks/selection.json`: `{raw,residuals}[*].check.risk_material`.

**F6.** Window. At W2000, hs and pot_gpd reject Kupiec at 0.95 (238 breaches, p 0.00284974); garch_n, fhs and garch_evt do not.

Source: `reports/oos/w2000/backtest.json`: `coverage.0.95.{hs,pot_gpd,garch_n,fhs,garch_evt}.breaches`, `.kupiec.p`, `.kupiec.reject`.

**F7.** gaussian shows the fat-tail pattern: 251 / 201 breaches at 0.95, 109 / 90 at 0.99, 46 / 38 at 0.999 (W1000 / W2000).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.{0.95,0.99,0.999}.gaussian.breaches`.

## Not claimed (b)

**C1.** garch_evt > garch_n on FZ0 is not claimed independent of W (F2).

Source: F2 (same sources).

**C2.** garch_evt > fhs at the primary levels is not claimed (F3); the 0.999 comparison is descriptive.

Source: F3 (same sources).

**C3.** The four pot_gpd sensitivity DM rejections (W1000 0.95: @0.9 p 0.000742831, @0.91 p 0.000471212; W1000 0.99: @0.91 p 0.00527062; W2000 0.95: @0.86 p 0.00258428) all favour the higher threshold: d_bar < 0 for @0.9 and @0.91, d_bar > 0 for @0.86. Their |d_bar| <= 0.0052, against |mean FZ0| >= 2.88532 for pot_gpd (all four level x window cells). Economically negligible; q_selected is unchanged (plan §5a (ii)). Across all 6 (level, key) pairs the sign agrees between windows in 4 of 6; this is reported, not interpreted.

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `dm.descriptive[a=pot_gpd@q, b=pot_gpd].p`, `.d_bar`; `fz0.{0.95,0.99}.pot_gpd.mean`.

## Limitations (c)

**L1.** ES is mildly understated: z_bar > 0 for every key at both levels in both windows. garch_evt W1000 0.99 ES p = 0.0582 (one-sided, D14); garch_evt@0.86 p = 0.0465. This direction was predicted before the OOS data by D36 (a)(3).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.{0.95,0.99}.<all 12 keys>.es_residual.z_bar`; `reports/oos/w1000/backtest.json`: `coverage.0.99.{garch_evt,garch_evt@0.86}.es_residual.p`.

**L2.** pot_gpd at 0.999 (W1000): 12 breaches, binomial p 0.0287412 (hs identical); W2000: 9, p 0.249519. Consistent with D36 (a)(3).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`: `coverage.0.999.{pot_gpd,hs}.breaches`, `.p_binomial`.

**L3.** Breach clustering in the unconditional models was expected (D38 (7)).

Source: docs/decisions.md D38 (7) (no number).

**L4.** The C3 direction is consistent with two independent pieces of evidence: the pre-registered A3 finding that the procedure tends to select below the true threshold with a risk-understating bias at q* (D36 (a)(1), (a)(3)), and the D45 raw blocks selecting q below q_selected (0.8, 0.81, 0.85). Recorded as a limitation of the threshold procedure for the unconditional model only: no garch_evt sensitivity DM test is significant (F5). No change to q_selected or q_c.

Source: docs/decisions.md D36 (a)(1), (a)(3); `reports/threshold_selection/oos_blocks/selection.json`: `raw[*].selection.q_star`; F5.

## Carried limitations

Carried, by reference only: D36 (a), D37 (b), D39 (a) and (h); D45 (B1 at the grid floor; residuals B2 near_boundary with alpha at zero; B3 overlaps B2 by 284 returns).

## Reporting convention (d)

A p-value printed as 0.0001 by docs/oos.md is the MC floor 1/(B+1) and is written "<= 1e-4" in all prose. docs/oos.md and its generator are NOT changed (D44 (h)).

## Addendum (D49): TRI

**TRI sensitivity addendum (D49); D46 above is unchanged.** Passes: PR W1000, PR W2000, TRI W1000. Every number was checked against the tracked JSON (6 significant digits); T2 was computed from the JSON.

Sources (sha256 of the file read):
- `reports/oos/w1000/backtest.json`: `bc79fd419d16ae5ac2cf03800ef6b505088bd9ea27fc559078d78cdb35b7f20a`
- `reports/oos/w2000/backtest.json`: `f900783ca58644aae3fba27fe51f6b936107959ed01366178ee64e29b44b5f12`
- `reports/oos/tri_w1000/backtest.json`: `425aba7b13494c4af47a9b689d624966084870a948c999947daec266ac4006ad`

**T1.** Holding in all three passes:
- F1 (IND rejected for gaussian, hs and pot_gpd at both levels; not rejected for garch_n, fhs and garch_evt; garch_evt beats pot_gpd in DM family (ii) at both levels);
- F4 (pot_gpd vs hs never rejected);
- the F7 gaussian pattern;
- garch_n rejects Kupiec and the ES test at 0.99.

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`, `reports/oos/tri_w1000/backtest.json`: `coverage.{0.95,0.99}.<key>.christoffersen.p_ind`; `dm.conditional[a=garch_evt, b=pot_gpd].reject`, `.d_bar`; `dm.unconditional[a=hs, b=pot_gpd].reject`; `coverage.{0.95,0.99,0.999}.gaussian.breaches`; `coverage.0.99.garch_n.kupiec.reject`, `.es_residual.p`.

**T2.** TRI vs PR on the identical 5716 days (computed): |breaches_TRI - breaches_PR| <= 2 for every key and primary level; traffic-light zone counts identical for all six keys; DM family (ii) verdicts identical in 5 of 6.

Source: `reports/oos/w1000/backtest.json`, `reports/oos/tri_w1000/backtest.json`: `n_rows`; `coverage.{0.95,0.99}.<key>.breaches`; `traffic_light.<key>.zones`; `dm.conditional[*].reject`.

**T3.** Scope of H. H holds on PR (W1000 and W2000). On TRI, fhs is not rejected by any coverage test at either primary level, but garch_evt is rejected by the 0.99 ES residual test (p 0.0438; PR 0.0582). Its sensitivity keys also reject at 0.99 at @0.86 (0.0367) and @0.9 (0.0483), not at @0.91 (0.0541). L1 is extended: garch_evt's 99% ES is borderline-understated and crosses 5% on TRI, in the direction predicted by D36 (a)(3).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`, `reports/oos/tri_w1000/backtest.json`: `coverage.{0.95,0.99}.{fhs,garch_evt}.kupiec.reject`, `.christoffersen.p_ind`, `.christoffersen.p_cc`, `.es_residual.p`; `reports/oos/tri_w1000/backtest.json`: `coverage.0.99.garch_evt@{0.86,0.9,0.91}.es_residual.p`, `.es_residual.z_bar`; PR value from `reports/oos/w1000/backtest.json`: `coverage.0.99.garch_evt.es_residual.p`.

**T4.** garch_evt vs fhs at 0.95 is rejected in favour of garch_evt on TRI W1000 (Holm 0.0463013, d_bar -0.00208936) and PR W2000 (Holm 0.0122724, d_bar -0.00159999), and not on PR W1000 (Holm 0.114964). At 0.99 it is not rejected in any pass (Holm 0.544465, 0.920528, 0.382477).

Source: `reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`, `reports/oos/tri_w1000/backtest.json`: `dm.conditional[a=garch_evt, b=fhs].p_holm`, `.reject`, `.d_bar`.

**T5.** Scope of F5. On TRI, 1 of 6 garch_evt sensitivity DM tests is significant (0.99 @0.91, p 0.0482307, d_bar +0.00162994, which favours q_c). The pot_gpd sensitivity rejections on TRI (0.95 @0.9 p 0.000808242, 0.95 @0.91 p 0.000160991, 0.99 @0.91 p 0.0365789) all favour the higher q, as in C3 / L4.

Source: `reports/oos/tri_w1000/backtest.json`: `dm.descriptive[a=garch_evt@q, b=garch_evt].p`, `.d_bar`; `dm.descriptive[a=pot_gpd@q, b=pot_gpd].p`, `.d_bar`.

**T6.** Claim policy for the Day-14 report:
- headline claims are restricted to statements that hold in all three passes (T1);
- any other statement names the passes in which it holds and is labelled "pass-dependent";
- a statement decided by a p-value in [0.01, 0.10] is labelled "borderline".

Source: docs/decisions.md D49 (no number).
