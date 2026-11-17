# NIFTYTAIL — Status

**Updated:** 2026-11-17.

## Current state
- Last commit before this one: `b0568c9` "15C-8: cross-platform verification (D66) and wording"; this commit is "15C-9: non-finite values in the D66 comparison".
- **Day 15C-9 (2026-11-17; bug fix in a D66 implementation pin, owner-approved; no methodology change, no new decision, every tracked JSON, figure and protected file unchanged).** Fixes the non-finite case of the D66 comparison, found by the reviewer on Linux: `demo.max_rel_diff` returned nan for inf vs inf and for nan vs any value, and max() could hide it (a nan in the first list position against a number passed). Now a pair of floats with a non-finite member gives 0.0 if both are nan or both are the same infinity, else inf; finite values and the Windows behaviour (bytes) are unchanged. Tests: `tests/test_demo.py` (inf vs inf, -inf vs inf, nan vs nan, nan vs 1.0 in either list position, `compare` on JSON bytes with NaN and Infinity off Windows). A temporary update mirror of this commit on top of the public repository (clean clone, outside the repository; deleted afterwards): 9 of 9 scans pass.
- **Day 15C-8 (2026-11-17; owner-approved; no methodology change, every tracked JSON, figure and protected file unchanged).** D66 written: a run on Linux reported by the reviewer (one demo byte test failed; forecast.json 46 values differ, max relative 1.24e-6; series.json 836 values, max 1.34e-7) shows that byte identity holds on Windows only; the cross-machine tolerance for VaR, ES, mu and sigma becomes relative 1e-5, counts, flags and verdicts stay exact. `demo.py`: `compare` and `max_rel_diff`; `demo-data --verify` is unchanged on Windows, elsewhere it passes on values within relative 1e-5 (absolute floor 1e-12) with its own message. `tests/test_repo_hygiene.py`: the demo test uses the same check; `tests/test_demo.py`: synthetic tests of the comparison. `docs/reproducibility.md` "Across machines": the tolerance and the reported Linux result. `dashboard_data.VERIFIED` and the README: the cross-platform clause. Wording: README "(research report)", "(D1-D66)", the Deploy sentence on `uv.lock` removed; `docs/report.md` header "NIFTYTAIL report". A temporary update mirror of this commit on top of the public repository (clean clone, outside the repository; deleted afterwards): 9 of 9 scans pass.
- **Day 15C-7 (2026-11-17; owner note only, no new decision; no code, every tracked JSON unchanged).** `docs/publication.md`, Owner steps item 6: after `git push`, reboot the public app, because the host reloads `app/` on a push but keeps the already-imported package from `src/` in memory, so a new `dashboard.py` can meet an old `dashboard_data` (seen 2026-11-17: AttributeError on `REPLAY_HEADING` until the reboot).
- **Day 15C-6 (2026-11-17; display and wording only, owner-approved, no new decision; every tracked JSON unchanged).** `app/dashboard.py`: under the Proof heading "Commits", the caption "Commit IDs refer to the development repository; the public copy is published as one commit per release (D65 (n))." in both modes; in demo mode the replay heading ends in "(synthetic index)", as the other demo titles (local mode unchanged). `dashboard_data.py`: tooltip dates of the market chart (line and candles), the worst-days triangles and the drawdown chart come from a new string field `date_t` (the stored ISO date, unchanged text), because the temporal x field "date" was shown as epoch milliseconds; the recent-losses chart (ordinal x) is unchanged; the date is not formatted by the chart library, so no time-zone shift. `README.md`: "(D1-D65)". Tests: `tests/test_dashboard_data.py` (no tooltip field is a temporal channel of the same layer, every spec, both palettes, every range; the tooltip date equals the ISO string), `app/tests/test_app.py` (the Proof caption in both modes; the demo-only replay heading). A temporary update mirror of this commit on top of the public repository (clean clone, outside the repository; deleted afterwards): 9 of 9 scans pass.
- **Day 15C-5 (2026-11-17; wording only, owner-approved, no new decision).** `README.md`: the line on the development process and tooling now says it is described in the internship report submitted to IDEAS – ISI Kolkata, which is not part of this repository. A temporary squash mirror of this commit (clean clone, outside the repository; deleted afterwards): 9 of 9 scans pass and the new README line is unchanged by the wording rules.
- **Day 15C-4 (2026-11-16).** D65 (n): the public copy is one commit per release; commit IDs recorded in manifests and docs refer to the development repository.
- **Day 15C-3 (2026-11-16; docs).** D65 pins (14)-(15). `docs/decisions.md`: the Roles note under the header. `README.md`: the line on the development process and tooling.
- **Day 15C-2 (2026-11-16).** D65 pins (9)-(13). `dashboard_data.py`: PASSED_NOTE "passed every coverage test on the price index (D46 H); pass-dependent, borderline"; `T1_BORDERLINE` (the README line, shown under T1 in Track record); `REPRODUCE` holds only commands that pass from the public copy; `VERIFIED`. `app/dashboard.py`: one "Reproduce it yourself" block with that sentence; in demo mode the card reads "Last session (synthetic demo)" and the NSE link button is not drawn. `README.md`: the same sentence. `docs/data.md`: large-return and identical-close dates as counts by decade; the halts table without its two flag columns. Tests: tests/test_data.py, tests/test_dashboard_data.py, tests/test_report_claims.py, app/tests/test_app.py.
- **Day 15C-1 (2026-11-16; docs and one test file).** D65 written (the reviewer's (a)-(q); the implementer's pins (1)-(8)). `docs/report.md`: title without "pre-registered"; §1, §3 and §7 state the amendments and departures with their decisions (D9, D38, D39, D33, D36, D37 (b), D39 (a)); T1 is the headline, followed by the borderline line (four cells, values copied from the three backtest.json); "Headline 2" is now "Main finding on the price index (pass-dependent, borderline)"; §2 says who ran the spot-check comparison; §7 gains the limitations of the tests and of the application; Kupiec power "at T = 5,650"; reference list cleaned. `README.md`: the same borderline line, `uv sync --locked`, Disclaimer, the coverage-run note, the live demo URL. `docs/data.md`: §8 heading. New `tests/test_report_claims.py` (8 tests: the report and the README against the tracked JSON).
- **Day 15B-3 (2026-11-16; docs only).** `docs/publication.md`: LICENSE done (MIT), supervisor approval given (2026-11-16), scrub done (D63, D64); the remaining owner steps are `scripts\publish.cmd <mirror dir> --review <file>`, the review of that file, creating the GitHub repository, the push and the deploy (Python 3.13, `app/dashboard.py`, secret `NIFTYTAIL_PUBLIC = "1"`). README "Deploy" step 1 names `--review`. Nothing is pushed or deployed.
- **Day 15B-2 (2026-11-16).** D64 written (the reviewer's (a)-(c); the implementer's pins (1)-(11)). `docs/results_review.md`: the two provenance sentences now read "checked against the tracked JSON"; sha256 `f12f46b5…` -> `b2d86d4f810de6c741744155dee8bdbd2fb0052d37c307978a1355559a8259b3` (docs/protected_hashes.md as a D64 amendment, `dashboard_data.py`); the H, T1 and T3 paragraphs are byte-identical (their sha256, taken before the change, is pinned in `tests/test_dashboard_data.py`). `publish.py`: in the same rewrite, the path drops and the wording rules of new `scripts/publish_rules.toml` (never in the mirror); a STOP before any rewrite when a protected file holds a match that no rule covers; three new scans (wording, protected hash, rewrite beyond the rules); `--review <file>` (`scripts/publish.cmd` and `__main__.py` pass it on). `docs/publication.md`: what the rewrite changes and what is scanned. Tests: `tests/test_publish.py` (24 cases, synthetic repositories). Simulation on a scratch clone of this repository, so that the working repository gets no commit: 8 of 8 scans pass; in a fresh clone of that mirror `uv sync`, pytest, mypy, ruff and the app tests pass; the mirror was deleted afterwards.
- **Day 15B-1 (2026-11-16).** D63. The protected-hash list (33 entries, unchanged; 33 of 33 recomputed hashes match) is in new `docs/protected_hashes.md`; the tests, tools and docs that read or cite the list use it. New `LICENSE` (MIT, "Copyright (c) 2026 Sambit Sinha"), README "License", pyproject.toml `license = "MIT"` (uv.lock unchanged). `docs/publication.md`: approval given (2026-11-16) and LICENSE done.
- **Day 15A-4b (2026-11-16).** D62 pin (21): the publish scan treats the one synthetic test token of the 15A-3 history as a placeholder; `tests/test_publish.py` builds its planted token at runtime. Found by the first simulation run (api_token FAIL on that fixture only; every other scan passed; mirror deleted).
- **Day 15A-4 (2026-11-16).** D62 pins (17)-(20). New `app/requirements.txt` for Streamlit Community Cloud (streamlit and pyarrow at the uv.lock versions, plus the package; uv.lock and pyproject.toml unchanged). README "Live demo" (placeholder) and "Deploy". Test: `tests/test_repo_hygiene.py` (pins equal uv.lock). The local Cloud simulation runs right after this commit through `scripts\publish.cmd` (which adds the "Publish: commit map" commit); nothing is pushed or deployed.
- **Day 15A-3 (2026-11-16).** D62 pins (10)-(16). New `src/niftytail/publish.py` (command `publish`) and `scripts/publish.cmd` (never pushes): refuses a dirty tree, rewrites a fresh clone with git-filter-repo (personal e-mail -> noreply in metadata, files and messages), commits `docs/commit-map.txt`, rewrites again into the mirror and scans it (e-mail, forbidden paths, secrets file, key, api_token, files over 5 MB), printing counts only. New `docs/publication.md` (D23 checklist: supervisor approval pending, e-mail handled by the mirror, LICENSE pending, history scan automated). Tests: `tests/test_publish.py` (synthetic repos, no network). `__main__.py` gains the command (additive).
- **Day 15A-2 (2026-11-16).** D62 pins (5)-(9). New `src/niftytail/demo.py` and command `demo-data configs/base.toml [--verify]` (synthetic index through the `forecast` path) writing the tracked `app/demo/forecast.json` and `app/demo/series.json`; `demo-data --verify` exit 0 ("verify: bytes match"). Public mode (environment or st.secrets) shows "Today (demo)" first with the demo banner and "synthetic index" chart titles; local mode is unchanged and never shows the demo. Changed: `__main__.py` (additive), `dashboard_data.py`, `app/dashboard.py`, tests, app tests.
- **Day 15A-1 (2026-11-16).** D62 written (the reviewer's (a)-(e); the implementer's pins (1)-(4)). README: "Headline result (holds in all three passes)" with T1 only, H under "Main finding on the price index (pass-dependent, borderline)", "Market data is never included in this repository."; dashboard heading renamed to the same main-finding text (claim text unchanged); pyproject.toml authors without e-mail (uv.lock unchanged, `uv lock --check` passes). Tests: `tests/test_repo_hygiene.py`, `app/tests/test_app.py`.
- **Day 14 done (2026-11-16).** D61 pin (8). New `docs/report.md` (4,178 words; Abstract, 1 Question and contribution, 2 Data, 3 Methodology, 4 Results, 5 Robustness, 6 Application, 7 Limitations, 8 Reproducibility, References) under the D49 T6 claim policy. Its headlines are T1 and H only, H labelled pass-dependent and borderline. Its tables are copied from the tracked backtest.json files and checked by script. Its figures are the tracked SVGs. Day 14 in all: 14-1 leftovers (`c1d506b`), 14-2 reproducibility record (`5b37709`, every published artefact byte-identical at its recorded commit), 14-3 README (`9e1fcb1`), 14-4 report.
- **Day 14-3 (2026-11-16).** D61 pin (7). README restructured for a public reader: what NIFTYTAIL is, headline results (T1 and H verbatim, H with its T6 labels), quick start, one ordered "Reproduce the study" path with the 14-2 durations, Using the model, Automatic daily update, Dashboard, Data policy (unchanged), repository map, where decisions and status live. The per-step command blocks moved unchanged to new `docs/reproduce.md`. No LICENSE yet (Day 15).
- **Day 14-2 (2026-11-16).** D61 pins (4)-(6). New `docs/reproducibility.md`: from `c1d506b` on a clean tree, every published artefact verified at its recorded commit (git worktree, gitignored inputs linked read only): `run --verify` PR W1000 (89 s), PR W2000 (94 s), TRI W1000 (85 s), all four files each byte-identical, including `forecasts.csv` (the plan's acceptance); `tl-blocks --verify`, `figures --verify` (exit 0 with the committed outputs in place; the first bare-checkout attempts exited 1 with the recomputed hashes equal to the protected-hash list), `figures-local --verify`, `forecast --verify` and `forecast --check-run` (20/20 identical); 33 of 33 protected hashes match; the machine and the cross-machine statement. No per-day value or date was written or printed.
- **Day 14-1 (2026-11-16).** D61 written (the reviewer's (a)-(i); the implementer's pins (1)-(3) for 14-1). `scripts/update.cmd`: validate-live before fetch-api; Today's source KPI as a small wrapping card; M1 and M2 time axes labelled with the year; the All-range change KPI "Change since the first session". Changed: `scripts/update.cmd`, `dashboard_data.py`, `app/dashboard.py`, tests, app tests.
- **Day 13F (2026-11-16).** D60 written (the reviewer's (a)-(h); the implementer's pins (1)-(10)). New module `src/niftytail/eodhd.py` (command `fetch-api`; key from NIFTYTAIL_EODHD_KEY only, never shown); `live.py`: data/api reader and merge after data/live (`load_series_full`), the D25 row rules, the api manifest sha256 in the forecast key and provenance; `__main__.py` gains the command (additive); `scripts/update.cmd` gains fetch-api and prints a failing command's error lines; `dashboard_data.py` accepts the "api" source; README "Automatic daily update". Tests: `tests/test_eodhd.py` (fake urlopen, no network), `tests/test_repo_hygiene.py` (data/api ignored). No real-data run: the owner runs the first fetch.
- **Day 13E-4 (2026-11-16).** D59 pins (30)-(44), display only (forecast.json and series.json unchanged): M1 y domain fitted with 3% padding, one axis per chart and no vertical gridlines, window shading only in "All", capped return periods and the xi < 0 upper-limit text, C3 floor 1e-6 and 1% ticks, captions (replay, stress, volatility, history count), % of position under the rupee cards, source labels, D49 T3 in a collapsed expander, C2 dates on x with "next", short Track record labels, LaTeX powers of ten, `[logger] hideWelcomeMessage = true` (hides Streamlit's welcome popup). Changed: `dashboard_data.py`, `app/dashboard.py`, `.streamlit/config.toml`, tests, app tests.
- **Day 13E-3 (2026-11-16).** D59 pins (22)-(29). `app/dashboard.py`: four sections (Today local only, Track record, How it works, Proof) as nested st.tabs; Today = freshness, market (KPIs, M1, M2, last-session card, NSE link), position and budget, risk cards, C2 with the health table, crash calculator, stress and replay, the full and far-tail tables; How it works = the numbered pipeline with today's values, the self-check and C3; Proof = provenance, the forecast provenance and check.json. `.streamlit/config.toml`: dark theme (dark only, D59 pin (23)), static serving, Inter and IBM Plex Mono from `app/static/fonts/` with their OFL licences. README "Dashboard" and "Using the model". Tests: `app/tests/test_app.py` (AppTest, synthetic forecast and series).
- **Day 13E-2 (2026-11-16).** D59 pins (11)-(21). `dashboard_data.py`: % of position (2 decimals), mirrored tail VaR/ES and the self-check, exceedance probabilities and return periods, return levels, the history count, the volatility scenario, the replay, market range statistics, freshness, the schema-2 check with series.json and check.json (`live_bundle`, the disagreement warning), and the Vega-Lite specs M1, M2, C2, C3 with the (s) palette per mode. Tests: `tests/test_dashboard_data.py` (synthetic only; POT-GPD and GARCH-EVT values built with `niftytail.gpd`).
- **Day 13E-1 (2026-11-16).** D59 written (the reviewer's (a)-(s); the implementer's pins (1)-(10) for 13E-1). `live.py`: trailing-weekday guard in the shared live check; forecast.json schema_version 2 (per-model `tail` block copied from the diagnostics, `config.xi_zero_tol`); `series.json` beside it (history with OHLC and source, window, recent, tail_sample, worst), covered by --verify; an existing forecast.json now acts as --verify (amends D58 (d)); `--check-run` writes `check.json`. New `scripts/update.cmd` (validate-live if data\live exists, forecast, check-run; step names and exit codes only); `.gitattributes` keeps it CRLF. README "Using the model" step 1 gains the To-date rule. Tests: `tests/test_live.py` (guard, schema v2, series.json, unchanged/verify, check.json), `tests/test_repo_hygiene.py` (series.json and check.json ignored). `__main__.py` unchanged (no new flag needed).
- **Day 13D-2 (2026-11-16).** `dashboard_data.py`: rupee view (`rupee_loss`, `budget_position`, `inr`), `live_forecast` with the schema check, table rows for the tab (D58 (e), (f); pins (11)-(14)). `app/dashboard.py`: tab "Risk forecast" (local mode only). README section "Using the model" (PowerShell and cmd). Tests: `tests/test_dashboard_data.py` (formulas, grouping, schema, public-mode exclusion), `app/tests/test_app.py` (AppTest with a synthetic forecast.json; 10 passed in `.venv-dashboard`).
- **Day 13D real-data runs (2026-11-16)**, from commit `087d7c6` on a clean tree, `data/raw` only (no `data/live`): `forecast` exit 0; `forecast --verify` exit 0; `forecast --check-run reports/runs/e65780dc4f81b1cad5a3bdd88b0fd18bb69f9ba3585655814af338174c0b28c9` (PR W1000) exit 0, "20/20 identical". The forecast.json is under the gitignored `reports/live/`; no date or value reached a log, the terminal or a tracked file.
- **Day 13D-1 (2026-11-15).** D58: `rolling._target` promoted to `rolling.forecast_target` (body unchanged; `run_rolling` byte-identical: the existing rolling tests pass unchanged and every protected hash is unchanged). New module `src/niftytail/live.py` (commands `validate-live` and `forecast`, with `--verify` and `--check-run`); `__main__.py` gains the two commands (additive). `.gitignore` gains `/reports/live/` (`data/live/` is covered by `/data/`). Tests: `tests/test_live.py` (synthetic only), `tests/test_repo_hygiene.py` (data/live and reports/live ignored). The real-data runs follow this commit.
- **Day 13C done (2026-11-15).** D57, presentation only: plain-language Overview subheaders with source captions (claim text verbatim), Provenance inputs as a sha256sum-format code block, `toolbarMode = "viewer"` in `.streamlit/config.toml`, three new app tests. Changed: `app/dashboard.py`, `app/tests/test_app.py`, `.streamlit/config.toml`. In `.venv-dashboard`: app tests 8 passed; public-mode health check "ok". `dashboard_data.py`, uv.lock and pyproject.toml unchanged.
- **ENV-1 dashboard / Day 13B done (2026-11-15).** D56: the `dashboard` group pins `pyarrow>=24.0.0,<25` (Smart App Control blocks pyarrow 26.0.1); uv.lock changes pyarrow only. Day 13B local run and verify done natively on Windows in `.venv-dashboard` (gitignored), public mode: `app/tests` 5 passed, `/_stcore/health` "ok". Supersedes the WSL2 run of D55 (a). README "Dashboard" and the two app docstrings now give the PowerShell commands. The project `.venv` is unchanged (default + dev, D43).
- **Day 13A-2 (2026-11-15).** `app/dashboard.py` (thin UI: imports only streamlit and niftytail.dashboard_data; eight tabs, a ninth "Local only" tab in local mode), `app/tests/test_app.py` (streamlit.testing AppTest; not collected on Windows, run in WSL in Day 13B), `.streamlit/config.toml` (D55 (f) theme; usage stats off), README "Dashboard" section. pyproject.toml unchanged (testpaths and mypy files already exclude app/). The app and its tests have not been run yet: streamlit is not in the Windows `.venv`.
- **Day 13A-1 done (2026-11-15).** D55 written (reviewer's (a)-(h) plus the implementer's (1)-(15)). New module `src/niftytail/dashboard_data.py` (standard library only; 18 tracked inputs hash-checked against the protected-hash list; rows for the dashboard tabs; local mode only with the Day-12C-2 hashes). Tests: `tests/test_dashboard_data.py` (100% coverage; includes the AST import test). `.gitignore` gains `.venv-wsl/`. Streamlit is not installed in the Windows `.venv` (D43).
- **Day 13-0b done (2026-11-15).** From commit `435fdca7e6a63ef60a6c606265f39b33739e4ad2` on a clean tree: `figures --replace` (exit 0) and `figures --verify` (exit 0, all 9 hashes match). Only F5's drawing changed, but every SVG embeds the generating commit, so all 8 SVGs and docs/figures.md have new bytes; the 9 protected hashes were replaced (full sha256 from sha256sum):
  - `reports/figures/coverage.svg` `69051fb301613ecb5d4874540a8252a206541474106e11d8f2de8512343d7d76`
  - `reports/figures/tests_heatmap.svg` `82ac0ad2e5e83e728417fca98ae634b03a4b68b623af8dcc2dd2748da717ec51`
  - `reports/figures/dm.svg` `304d6b3db31908051d08e3ad6812b90c573e73a3410d10538ff0d08d58270cc9`
  - `reports/figures/traffic_light.svg` `b5fd5825ba172b4199ea5c526996d063455a2a6a198a4e8f46e24fa35ed3c23a`
  - `reports/figures/sensitivity.svg` `2a5979cf240dacaedd3a96777b113ac2ef2f6a3f9cc1702f91ff05ebe4b7f382`
  - `reports/figures/oos_blocks.svg` `e4d96237146ded2da29e150565bd810a9b0ee638d99fd21a3770198cf2d23394`
  - `reports/figures/xi_blocks.svg` `21fd325d4bee413ac6f9b78da484a3f5c90b473b586ae4541143a54c56fd805f`
  - `reports/figures/descriptive_levels.svg` `bde61ac2c9d26986b2fcc41896848bfcb957310192896fe1d6916f24cb2b6faf`
  - `docs/figures.md` `c82969e3d64acd6d387e95380707342a82abe9a0d461cb5510a1fbd91142e223`
  - Checked: no ISO date and no month-name date in any SVG or in the doc.
- **Day 13-0a done (2026-11-15; D52 follow-up).** F5: y labels on the left column only ("VaR <level>" over "ΔFZ0 vs q = 0.88" / "breach rate (%)"), a figure-level note "ΔFZ0 < 0: better than q = 0.88". New test: no label, drawn tick label or title of one F5 panel overlaps another panel's frame or texts, on the fixture and on the published inputs (it fails on the previous layout). Changed: `src/niftytail/figures.py`, `tests/test_figures.py`. The SVG is re-rendered in Day 13-0b.
- **Day 12C-2 done (2026-11-15).** From commit `ab5c135148271f7d03205772851dfd613c702ad8` on a clean tree: `figures-local` (exit 0) and `figures-local --verify` (exit 0, all 5 hashes match). Every D53 identity check passed (11A manifest commit 04572a3 and clean flag; 11A forecasts.csv = the xi_path provenance hash; 12A xi_path.csv = `4c90b70f…`; the three tracked inputs at their protected hashes; per-date traffic-light counts = tracked w1000). G3: the mean FZ0 over C of all six primary keys at 0.95 and 0.99 is repr-equal to the tracked w1000 value (12 of 12). Local output (gitignored, `git check-ignore` confirmed; nothing committed): `reports/runs/53f989e13b86b036d05b0ba0a12f6a09cea33c6b33687784bdbc66194ea764bb/`, sha256:
  - `figures/var_es_paths.svg` `8f3e34a5fb80e0edd085dd6c3d6e0a501497eccbe56b392d63fc1af8631329b8`
  - `figures/xi_path.svg` `e9b2e0bd0578772e91677d255522c3ce603874b13c5b847dab1fc3cb0a4245b3`
  - `figures/cum_fz0.svg` `82829741dd75ce3247ddcf7f561e02691e9e20b755c6b7e9376d1870a8cc15c6`
  - `figures/traffic_light_calendar.svg` `e9fc54cf319cb21a06b8d34cbce1a7f6c60999334397dead36e88e20f49f7e18`
  - `index.md` `001451effbdd16bc7bc0e91354b435c6ff191190813b487584d2afad4c086364`
  - No date or per-day value reached a log, the terminal or a tracked file; the tree stayed clean.
- **Day 12C-1 done (2026-11-15).** D53 written (reviewer's (a)-(e) plus the implementer's (1)-(9)). New module `src/niftytail/figures_local.py` (command `figures-local`; G1-G4 to a gitignored run dir); `__main__.py` gains the command (additive). Tests: `tests/test_figures_local.py`, synthetic only.
- **Day 12B-6 done (2026-11-15).** From commit `941f603e3d853c5be1bbe5e28c4f0a35478f9d35` on a clean tree: `figures --replace` (exit 0) and `figures --verify` (exit 0, all 9 hashes match). The 9 protected hashes were replaced (full sha256 from sha256sum):
  - `reports/figures/coverage.svg` `33fadb3340666f573bc3348b449ce841bf82dcfc8a8c41e4ecf00e022e381a8a`
  - `reports/figures/tests_heatmap.svg` `36deeb8c74c8ae981e31bb18269d6e104bced97508d357b484835cd936ca6b8b`
  - `reports/figures/dm.svg` `50419b8d99a17990ec8a863d7ef9652d80b16fe394c52db45f253627a4a8def2`
  - `reports/figures/traffic_light.svg` `6c24faa74612125f332509e3abfc496077219b014174a12d2c465bed22d0cc13`
  - `reports/figures/sensitivity.svg` `9d46afa9f089586f377b5dc6118fc2a2aab667d6b65385b7785d89a07d1a1ccb`
  - `reports/figures/oos_blocks.svg` `45afaa41e5f7b2285b56141cf7fc2e2fe085e301ee788614189c681751aee3a1`
  - `reports/figures/xi_blocks.svg` `5c7c05f7e034f29dc9786321707ab1afa934d94581afa6d950d9eb74473fb41e`
  - `reports/figures/descriptive_levels.svg` `7c253b25382c6ab8324c8f4e4b730669fd1a972a38d2f502c44f4db780c5f308`
  - `docs/figures.md` `eec2919909a891b62092e5fb1578f0b86c4504ef6df88b767a413e22ffb066df`
  - Checked: no ISO date and no month-name date in any SVG or in the doc.
- **Day 12B-5a done (2026-11-15).** D54 (owner): `figures --replace` overwrites only the 8 SVGs and docs/figures.md, only from a clean tree; without it the command still refuses an existing output. Changed: `src/niftytail/figures.py`, `src/niftytail/__main__.py` (one flag), `tests/test_figures.py`.
- **Day 12B-5 done (2026-11-15).** D52: display names, the F1 legend below the panels, F3 reject fills for both families and the new x label, F5 FZ0 rows as differences from q = 0.88, the F6 pass label, ≤ for the MC floor. Changed: `src/niftytail/figures.py`, `tests/test_figures.py`. The tracked SVGs and docs/figures.md are not yet re-rendered (Day 12B-6).
- **Day 12B-4 done (2026-11-15).** The protected "keep their sha256" list gains traffic_light_blocks.json, the 8 SVGs and docs/figures.md (full sha256 from sha256sum, cross-checked against the committed blobs).
- **Day 12B-3 done (2026-11-15).** From commit `014de6bfb41743d4a291608e5e635c87198cfb39` on a clean tree: `figures` (exit 0) and `figures --verify` (exit 0, all 9 hashes match). Tracked, sha256 from `sha256sum`:
  - `reports/figures/coverage.svg` `3495637daab2594f7de57d992fcaa303704a8b80caae06b77bab12db3320d7cb`
  - `reports/figures/tests_heatmap.svg` `a359ec2f9540523aad26dee101c32c0a2242df2b513cebfb51e91cb02e7eae69`
  - `reports/figures/dm.svg` `bc19dd0fb2e8c4382737633337eb36c974eceb9c90ba71124ad5a5eb09eef786`
  - `reports/figures/traffic_light.svg` `c5c65798ec3604e552c6a510d563950d78940b75818db7402e40bce8e2ba1aae`
  - `reports/figures/sensitivity.svg` `d17d8c60c2521edc12d9e98ede2b6b1a30f1c7abe5a37ed9965e9ca56ebc3f3f`
  - `reports/figures/oos_blocks.svg` `e62698e219a7f452e26c71498718eabb66df824a68af6cfda2e383562e0c52f9`
  - `reports/figures/xi_blocks.svg` `a86d5578cd79be7e8448d6efed0dca6ce0f5cbb4ed506d30d6813d47a54766e7`
  - `reports/figures/descriptive_levels.svg` `6fce9ea21f1edfda60842513c1236ed15c2d420feec4a9aa0c9464006adc14d5`
  - `docs/figures.md` (generated) `98f79d97b5515d40406f04790e0506787e36a96e2daa7f3688ac607750cea813`
  - Checked: no ISO date and no month-name date in any SVG or in the doc; inputs only the D51 (a) files (hash-checked).
- **Day 12B-2 done (2026-11-15).** From commit `65697f44e3cb5b192e960bdfe309ea4fd9c83434` on a clean tree: `tl-blocks` (exit 0) and `tl-blocks --verify` (exit 0, hash matches). Tracked `reports/oos/traffic_light_blocks.json`, sha256 `5d736d95d0121937c107873d3d05f3d57953df5f1239b2b37edd964e670af1f0`. Every D51 (3) check passed for the three run dirs (PR W1000 `e65780dc…0b28c9`, PR W2000 `dadebdda…1270a7`, TRI W1000 `316df0ba…c35c67d`); for all 12 keys x 3 passes the recomputed blocks equal the stored per-date blocks and the zone counts, full_blocks, flagged_blocks and partial_block equal the tracked backtest.json traffic_light (36 of 36). Checked: no ISO date, no date key, no list with 100 or more elements.
- **Day 12B-1 done (2026-11-15).** D51 written (the reviewer's (a)-(j), amended before code, plus the implementer's (1)-(13)). New modules `src/niftytail/tl_blocks.py` (command `tl-blocks`, D51 (j)) and `src/niftytail/figures.py` (command `figures`, the eight builders); `__main__.py` gains the two commands (additive only). `plots.py` is unchanged (D51 (1)). Tests: `tests/test_tl_blocks.py`, `tests/test_figures.py`, synthetic only.
- **Day 11C-3 done (2026-11-15).** TRI spot check done: 10 of 10 dates match niftyindices.com (owner, 2026-11-15); recorded in the gitignored sheet and in `docs/data.md` §9.
- **Day 12B-0 done (2026-11-15; no code change).** D50: the protected "keep their sha256" list now includes the 16 published artefacts (full sha256 from sha256sum, cross-checked against the committed blobs), and two hard rules were added (generated docs are never edited by hand; verify first).
- **Day 11T done (2026-11-15; no code change).** D49 records the TRI addendum to D46 (T1-T6, including the Day-14 claim policy); `docs/results_review.md` gained the section "Addendum (D49): TRI", the earlier text unchanged. D46 is not edited.
- Unchanged: every other code file, configs/base.toml, the plan, every tracked JSON and generated doc; uv.lock and pyproject.toml change only for the D56 pyarrow pin.
- ENV-1 (D43): the project `.venv` contains the default and dev groups only.

## Next recommended action
Released 2026-11-17: public repository https://github.com/skullwakeup/niftytail and app https://niftytail.streamlit.app. No open task.
- W = 2000 is not run on TRI (D46 (e): insufficient history), a recorded limitation of D2.

## Day 11T: what was done (2026-11-15; no code change)
- Every T1-T5 number and claim was checked against `reports/oos/{w1000,w2000,tri_w1000}/backtest.json` before writing; T2 was computed in code from the JSON (max |breach difference| = 2 over all 12 keys at 0.95 and 0.99; traffic-light zones identical for 12 of 12 keys; DM family (ii) verdicts identical in 5 of 6, the difference being garch_evt vs fhs at 0.95). No mismatch.
- D49 written, the addendum appended to `docs/results_review.md`, this section.

## Day 11C: what was done (2026-11-15; TRI, W = 1000)
- Commands, from commit `4c68343` on a clean tree, in order (all exit 0): `validate-tri`, `tri-run` (31 workers), `tri-run --verify --workers 8`, `tri-publish`.
- `validate-tri` wrote `data/raw_tri/MANIFEST.toml` (sha256 `74c3babf553b88ed320a6208c997efea47e894f28439f318164ea7ca132e5490`). It found 0 hard errors.
  - 28 files, 6808 rows, Net "-" = 129.
  - Flags: calendar_gap 12, identical_close_pair 3, large_return 65, stale_close_run 0, weekend_date 32.
  - PR-vs-TRI date differences over the whole TRI span: pr_only 1, tri_only 0; over the OOS span: 0 and 0.
  - The manifest and validation_report.json are byte-identical to the dry run of D48 (14).
- Alignment: t0_tri = 1091; n_targets = 5716; garch_error 0; no_pre_window_lag 0; stop_for_owner false. \|C\| = 5716 (share 1.0) at 0.95 and at 0.99; no below-min-share flag.
- Run dir (gitignored, kept): `reports/runs/316df0ba02cc927721383ee4c7f7e768af8d3bfa6d4a6041421f5b82ac35c67d`. sha256, identical in the run and the verify:
  - forecasts.csv `f34c95d9d54f4707ecde5f37ed3c1f9d8f17b8daf7967c46b673ca65d663890d`
  - summary.json `0d9174f16e2514b7634bb1ddc8b6fc6d4f0d25964ac0cd2b5769e7cb3e246ec6`
  - backtest.json `425aba7b13494c4af47a9b689d624966084870a948c999947daec266ac4006ad`
  - backtest_per_date.json `3c88059faf2f980a493c91d9a9d41176b58414f1ea358f1a79c38c81d4ac9656`
- Seconds: rolling 72.8 and evaluate 14.8 (31 workers); verify rolling 225.9 and evaluate 14.7 (8 workers).
- Tracked (aggregates only): `reports/oos/tri_w1000/` with the full sha256 of each file:
  - backtest.json `425aba7b13494c4af47a9b689d624966084870a948c999947daec266ac4006ad`
  - summary.json `0d9174f16e2514b7634bb1ddc8b6fc6d4f0d25964ac0cd2b5769e7cb3e246ec6`
  - manifest.json `5b0c5833fd74b1846e60c6dcc150977c070d58929b7ae54a174ec9869be8cd0a`
  - `docs/oos_tri.md` (generated) `2a2ac2bac696ccde76bc4a5bb5ae042b664c10fa6779d96828f6cb134bf3438a`
  - Checked: no ISO date, no date key, no list with 100 or more elements; the doc equals a fresh render.
- PR vs TRI on the identical 5716 days (copied by script from `reports/oos/w1000/backtest.json` and `reports/oos/tri_w1000/backtest.json`; primary keys; the sensitivity keys are in docs/oos_tri.md):

| 0.95 key | breaches PR / TRI | p_hat PR / TRI | Kupiec p PR / TRI | p_ind PR / TRI | p_cc PR / TRI | ES p PR / TRI | mean FZ0 PR / TRI |
|---|---|---|---|---|---|---|---|
| gaussian | 251 / 251 | 0.0439118 / 0.0439118 | 0.0311768 / 0.0311768 | 0.0001 / 0.0001 | 0.0001 / 0.0001 | 0.0001 / 0.0001 | -3.4628 / -3.46488 |
| hs | 271 / 271 | 0.0474108 / 0.0474108 | 0.365094 / 0.365094 | 0.0001 / 0.0001 | 0.0001 / 0.0002 | 0.0944 / 0.0996 | -3.49352 / -3.49525 |
| pot_gpd | 270 / 272 | 0.0472358 / 0.0475857 | 0.333318 / 0.398657 | 0.0001 / 0.0001 | 0.0001 / 0.0001 | 0.1288 / 0.1469 | -3.49285 / -3.49489 |
| garch_n | 311 / 311 | 0.0544087 / 0.0544087 | 0.131345 / 0.131345 | 0.3026 / 0.3026 | 0.1915 / 0.1915 | 0.0001 / 0.0001 | -3.71809 / -3.72049 |
| fhs | 294 / 293 | 0.0514346 / 0.0512596 | 0.620304 / 0.66339 | 0.3707 / 0.3833 | 0.601 / 0.6321 | 0.246 / 0.2414 | -3.72837 / -3.73059 |
| garch_evt | 293 / 291 | 0.0512596 / 0.0509097 | 0.66339 / 0.753004 | 0.1491 / 0.0887 | 0.3171 / 0.2251 | 0.28 / 0.2405 | -3.73016 / -3.73268 |

| 0.99 key | breaches PR / TRI | p_hat PR / TRI | Kupiec p PR / TRI | p_ind PR / TRI | p_cc PR / TRI | ES p PR / TRI | mean FZ0 PR / TRI |
|---|---|---|---|---|---|---|---|
| gaussian | 109 / 109 | 0.0190693 / 0.0190693 | 9.07617e-10 / 9.07617e-10 | 0.0001 / 0.0001 | 0.0001 / 0.0001 | 0.0003 / 0.0003 | -2.67887 / -2.68013 |
| hs | 69 / 68 | 0.0120714 / 0.0118964 | 0.127426 / 0.161767 | 0.0001 / 0.0001 | 0.0001 / 0.0001 | 0.2311 / 0.2008 | -2.90593 / -2.90796 |
| pot_gpd | 62 / 62 | 0.0108467 / 0.0108467 | 0.525613 / 0.525613 | 0.0001 / 0.0001 | 0.0001 / 0.0001 | 0.1067 / 0.1085 | -2.91339 / -2.91647 |
| garch_n | 98 / 96 | 0.0171449 / 0.016795 | 8.32291e-07 / 2.53736e-06 | 0.5792 / 0.6065 | 0.0001 / 0.0001 | 0.0001 / 0.0001 | -3.2051 / -3.20725 |
| fhs | 64 / 64 | 0.0111966 / 0.0111966 | 0.372307 / 0.372307 | 0.3198 / 0.3198 | 0.3115 / 0.3115 | 0.125 / 0.122 | -3.28945 / -3.28994 |
| garch_evt | 62 / 61 | 0.0108467 / 0.0106718 | 0.525613 / 0.613585 | 0.3284 / 0.3381 | 0.447 / 0.5037 | 0.0582 / 0.0438 | -3.29256 / -3.29433 |

| DM (ii) level | A vs B | Holm p PR / TRI | verdict PR / TRI |
|---|---|---|---|
| 0.95 | garch_evt vs pot_gpd | 4.36092e-05 / 4.05588e-05 | reject / reject |
| 0.95 | garch_evt vs garch_n | 0.0464471 / 0.0420636 | reject / reject |
| 0.95 | garch_evt vs fhs | 0.114964 / 0.0463013 | not rejected / reject |
| 0.99 | garch_evt vs pot_gpd | 0.0109754 / 0.0105477 | reject / reject |
| 0.99 | garch_evt vs garch_n | 0.0243432 / 0.0240089 | reject / reject |
| 0.99 | garch_evt vs fhs | 0.544465 / 0.382477 | not rejected / not rejected |

| traffic light key | green/yellow/red PR | green/yellow/red TRI |
|---|---|---|
| gaussian | 17/2/3 | 17/2/3 |
| hs | 19/1/2 | 19/1/2 |
| pot_gpd | 18/2/2 | 18/2/2 |
| garch_n | 11/10/1 | 11/10/1 |
| fhs | 19/2/1 | 19/2/1 |
| garch_evt | 19/2/1 | 19/2/1 |

## Day 12A: what was done (2026-11-15; OOS path, descriptive)
- Commands, from commit `c70202f3ac1a703b4f2a7831a247794f50ccb14e` on a clean tree (all exit 0): `xi-path` (31 workers, 133.8 s), `xi-path --verify --workers 8` (335.8 s), `xi-path-publish`.
- D47 (c): 91456 cells compared against the 11A forecasts.csv (5716 targets × 2 keys × 4 levels × VaR and ES), with 0 mismatches. The refit ξ, β and n_u equal the diagnostics at every target. 11A run dir `e65780dc4f81b1cad5a3bdd88b0fd18bb69f9ba3585655814af338174c0b28c9`; its forecasts.csv sha256 is `714523a4fb354cd06cdb1764ed2fc79978ec8e14b1a4f52899dc433423c54cd9`.
- Run dir (gitignored, kept): `reports/runs/897974801445bf4d2d543a325ce9f9eee5ff9c4047fc22d38295ca84b42efff1`. The xi_path.csv sha256 is `4c90b70fbe5269ff55fcc6a34a30571664a839cd07f5e3c8491339aab9377c0d` and the summary.json sha256 is `e2ac9882bd7cd826ff80f46ee9b79a28ab45f2025f0f887c2bc2b300af92c30f`; both are identical in the run and the verify. The tracked summary is byte-identical to the run file.
- Numbers copied by script from `reports/oos/xi_path/summary.json`. Descriptive (plan §5a (iv)); overlapping windows make the path strongly autocorrelated, so shares are not independent evidence.

| key | n targets | n with xi | n with CI | n lower open | no_xi | no_upper_crossing | estimate_outside_range | estimation_error |
|---|---|---|---|---|---|---|---|---|
| garch_evt | 5716 | 5716 | 5716 | 0 | 0 | 0 | 0 | 0 |
| pot_gpd | 5716 | 5716 | 5716 | 0 | 0 | 0 | 0 | 0 |

| key | xi q0 | q0.05 | q0.25 | q0.5 | q0.75 | q0.95 | q1 |
|---|---|---|---|---|---|---|---|
| garch_evt | -0.253663 | -0.15053 | -0.0676809 | 0.00895425 | 0.0688431 | 0.136016 | 0.224453 |
| pot_gpd | -0.134216 | -0.0857173 | 0.00129286 | 0.0978785 | 0.220132 | 0.43988 | 0.569052 |

| key | share ci_lo > 0 | ci_hi < 0 | CI contains 0 | xi < 0 |
|---|---|---|---|---|
| garch_evt | 0.00927222 | 0.00682295 | 0.983905 | 0.479881 |
| pot_gpd | 0.338523 | 0 | 0.661477 | 0.245801 |

- pot_gpd training reference (q = 22/25): ξ̂ 0.115689, 95% profile CI [-0.00920515, 0.275268], from `reports/threshold_selection/real/selection.json` (sha256 `7929c129a03509cb31b56cea1804cbfd4559e99c09b960d86b658b86e86741f9`).
- Blocks: 23 per key (22 full plus the partial block of 216), in the tracked JSON and docs/xi_path.md.
- Tracked, aggregates only: `reports/oos/xi_path/summary.json` and `docs/xi_path.md` (generated; sha256 `50e15bec883d72d3d5c1e783d7ed5f601b6efe821c42a522e5b227979772b75b`). Checked: no ISO date, no date key, no list with 100 or more elements; the doc equals a fresh render.

## Day 11R: what was done (2026-11-15; no code change)
- First pass: the reviewer's draft C3 ("direction not consistent across windows") did not match `dm.descriptive[pot_gpd@q, pot_gpd].d_bar`. It was reported and nothing was written. The owner replaced C3 and added L4. Every statement of the final text was then verified (6 significant digits).
- D46 written, `docs/results_review.md` written (statements, numbers copied, one source path per statement), this section.

## Day 11B: what was done (2026-11-15; OOS blocks, post-hoc)
- Commands, from commit `c677aa6b53ad958d7063981e0dcc659b3618773a` on a clean tree (all exit 0): `oos-blocks` (31 workers), `oos-blocks --verify --workers 3`, `oos-blocks-publish`.
- Run dir (gitignored, kept): `reports/runs/73b18a23e09cf7ed770e9fc51b9006b7e5b1db8d4686e1839abdb0d158836b27`. The sha256 of oos_blocks.json is `5a64bb708a7714f13fdedc7846857c7d8c7b24e6f4c9a68a755b94e75d3cd49c`, identical in the run and the verify; the tracked selection.json is byte-identical to it.
- Layout: n_oos = 5716 (equals w1000 n_targets), remainder = 1716; blocks B1 offset 0, B2 offset 2000, B3 offset 3716 (overlapping, 284 returns with B2, descriptive).
- garch_error entries: 0.
- Results (copied by script from `reports/threshold_selection/oos_blocks/selection.json`). Raw rows: POST-HOC (plan §5a (ii)), q_selected is not changed. Residual rows: DESCRIPTIVE ONLY; no decision may cite them.

| series | block | q* | q_FS | k_hat | rule_failure | at grid floor | half 1 | half 2 | q_drift (reason) | var99_rel | es99_rel | risk_material (reason) | F4 near_boundary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | B1 | 0.8 | 0.8 | 0 | — | yes | 0.8 | 0.8 | yes (gap) | 0.00102535 | 0.000720564 | no (—) | — |
| raw | B2 | 0.81 | 0.81 | 1 | — | no | 0.8 | 0.82 | yes (gap) | 0.000794252 | -0.00361341 | no (—) | — |
| raw | B3 | 0.85 | 0.85 | 5 | — | no | 0.8 | 0.8 | no (—) | 0.00922343 | -0.0430862 | no (—) | — |
| residuals | B1 | 0.8 | 0.8 | 0 | — | yes | 0.8 | 0.8 | yes (gap) | 0.000197336 | -0.0048128 | no (—) | no |
| residuals | B2 | 0.82 | 0.82 | 2 | — | no | 0.8 | 0.82 | yes (gap) | 0.0166689 | 0.00118733 | no (—) | yes |
| residuals | B3 | 0.8 | 0.8 | 0 | — | yes | 0.8 | 0.8 | yes (gap) | 0.0071382 | -0.00435864 | no (—) | no |

- Seconds per task (run, 31 workers): raw B1 415.2, B2 427.5, B3 427.6; residuals B1 349.7, B2 360.3, B3 349.7. Verify (3 workers): 413.8, 241.7, 429.5, 184.7, 247.0, 237.2.
- Tracked, aggregates only (D23): `reports/threshold_selection/oos_blocks/selection.json` and `docs/oos_blocks.md` (generated). Checked: no ISO date, no date key, no list with 100 or more elements; the doc equals a fresh render. docs/oos_blocks.md sha256 `869a03323e7fa7a7717a4ab7fe2722cff34e93056527f6e75edf6d36b5522db5`.

## Day 11A: what was done (2026-11-15; first use of the OOS data)
- Commands, from commit `04572a3` (all exit 0): `run --window 1000`, `run --window 1000 --verify --workers 8`, `run --window 2000`, `run --window 2000 --verify --workers 8`, `oos-publish`.
- Per pass (numbers copied from the run-dir `summary.json`, `timing.json` and `backtest.json`):

| | W = 1000 | W = 2000 |
|---|---|---|
| n_targets | 5716 | 5716 |
| garch_error / share | 0 / 0.0 | 0 / 0.0 |
| no_pre_window_lag | 0 | 1 |
| stop_for_owner | false | false |
| \|C\| (share), 0.95 and 0.99 | 5716 (1.0) | 5715 (0.9998250524842547) |
| below common_mask_min_share | no | no |
| verify (8 workers): forecasts.csv, summary.json, backtest.json, backtest_per_date.json | all match | all match |
| rolling / evaluate seconds (31 workers) | 65.1 / 12.9 | 68.5 / 12.9 |

- Run dirs (gitignored, kept): `reports/runs/e65780dc…0b28c9` (W = 1000), `reports/runs/dadebdda…1270a7` (W = 2000).
- Tracked (aggregates only; D23): `reports/oos/w1000/` and `reports/oos/w2000/` (backtest.json, summary.json, manifest.json); `docs/oos.md`, generated by `oos-publish`. Checked: no ISO date, no date key, no list with 100 or more elements; the doc equals a fresh render.
- sha256 (full, from `sha256sum`; corrected on Day 11B-1, the earlier shortened w2000 backtest.json and summary.json suffixes were wrong):
  - `reports/oos/w1000/backtest.json`: `bc79fd419d16ae5ac2cf03800ef6b505088bd9ea27fc559078d78cdb35b7f20a`
  - `reports/oos/w1000/summary.json`: `fb6a87261a1303858afe2d8545b2dfdf6ca956a052f30864a01cfce884b91cb4`
  - `reports/oos/w1000/manifest.json`: `289f603029705c284b0e1f3996a36fd9ae38b44a6944b883ccce544bed9781b5`
  - `reports/oos/w2000/backtest.json`: `f900783ca58644aae3fba27fe51f6b936107959ed01366178ee64e29b44b5f12`
  - `reports/oos/w2000/summary.json`: `b2c88d60dba06eff292ae380f6004cba0a5ca4645f476dcfd9c5f98ada88785c`
  - `reports/oos/w2000/manifest.json`: `289f603029705c284b0e1f3996a36fd9ae38b44a6944b883ccce544bed9781b5`
  - `docs/oos.md`: `22825783dd86fc127316661d9a4e0cf0a2ccbba325f74b359e0413be26ea674e`
  - The two manifest.json files are byte-identical: the manifest does not hold the window.

## Day 10: what was done (2026-11-15; synthetic data only)
- `docs/decisions.md` working copy rewritten with LF line endings (the committed blob was unchanged: same hash). D42 was then added before any code.
- **Tests (`tests/test_backtest.py`, 85 cases, 19.4 s).** Every reference value below is reproduced with the Appendix A tolerances:

| Reference | Value(s) | Result |
|---|---|---|
| R5 FZ0 worked examples (6) and the tie | 0.521445, 3.964933, 13.660902, 0.521445 (tie), 0.853140, 26.128862 | pass (5e-7) |
| R5 E[FZ0] = ln(−e), quad over fz0 itself | 0.724022, 0.980284 | pass (1e-9) |
| R5 degree-0 homogeneity; conversion via `to_return_space` | 13.660902 | pass (1e-12; 5e-7) |
| R5 invalid inputs (10 raises); boundary e == v accepted | — | pass |
| R6 critical value, LR(250,5,0.01), LR(250,0,0.01) | 3.841459, 1.956810, 5.025168 | pass |
| R6/R6b LR acceptance regions (4) | [251,315], [43,71], [1,6], [7,19] | pass |
| R7 equal-tailed regions (12); p_bin(250,5,0.01) | all 12 cells; 0.215625 | pass |
| "Never interchange" rows | equal at 5650/0.05 only | pass |
| R8 exact size/power (8 rows, 4 dp) | 0.0523 … 1.0000 | pass |
| R9 Markov true size (8 rows, 4 dp); variance inflation (3 dp) | 0.0523 … 0.1296; 1.000 … 1.714 | pass |
| R10 DM lag, block length | 9 at T = 5650 and 5716; b = 13 (2000), 18 (5650), 18 (5716) | pass |
| DM vs statsmodels HAC (use_correction=False), ℓ = 9 and 18 | d̄, Ω̂, DM = t | pass (rel 1e-10) |

- Other tests:
  - Christoffersen: hand transition counts and LRs, 0·ln 0 = 0, x = 0 and x = T, the MC formula against an independent recomputation, the IND null at p̂ vs the CC null at p0.
  - Traffic light: 22 + 150 at T = 5650 and 22 + 216 at T = 5716, contiguity, zone boundaries 4/5 and 9/10, the inadmissible-row flag.
  - ES residual A5a (1)–(4): uniformity (slow, in the gate) gives KS p = 0.485 and a 5% rejection rate of 0.02; the +20% under-forecast gives p = 0.0001.
  - Constructed-panel mask: reasons, |C| = 5 of 10, mean FZ0 on C by hand, scores unaffected by non-C dates, a sensitivity NaN leaves C unchanged, the W = 2000 mask is its own, inadmissible values raise in fz0.
  - A one-row alignment shift changes the hits from 10 to 0.
  - Holm with ties and a NaN.
  - MC and bootstrap results are identical forward, reversed and in a 2-worker process pool.
  - `evaluate` end-to-end on a rolling.py table (n = 300, W = 100): aggregates contain no date, and two runs are byte-identical.
- Day-9 fixup: `test_injected_garch_error` asserts NaN at all four levels.
- **Methodology §11** gains the "six-model mask and families (D42)" note.

## Check results (actually executed, ENV-1 fix D43, project `.venv`)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **757 passed**, 0 skipped (incl. the `test_environment.py` canary); total coverage 99% (3,583 statements, 23 missed); 544.3 s |
| `uv run --locked mypy` | pass: "Success: no issues found in 20 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| `uv.lock`, `pyproject.toml` | unchanged (sha256) |

## Check results (actually executed, Day 10)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` (scratch venv, same lock) | **757 passed**, 0 skipped; total coverage 99% (3,583 statements, 23 missed; `backtest.py` 100%); 458.9 s |
| `uv run --locked mypy` (project `.venv`) | pass: "Success: no issues found in 20 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| sha256, unchanged | `docs/plan_v2.2.md` and all six tracked JSONs of the protected-hash list; `uv.lock`, `pyproject.toml` and `configs/base.toml` unchanged |

## Day 9: what was done (2026-11-13; synthetic data only)
- **D41** was pinned before code. The implementation details that the implementer pinned are listed in D41.
- **Tests (`tests/test_rolling.py`, 19 cases).** The small design is n = 300, W = 100, T0 = 150, with every model real:
  - HS VaR equals the known order statistic of L[j−W:j].
  - A step at k = 200 leaves targets up to k bit-identical and changes target k+1.
  - Future perturbation at m = 150, 220, 299 leaves every forecast (all models and sensitivities) for targets j ≤ m bit-identical.
  - Changing L[:80] leaves targets j ≥ 181 bit-identical. At j = 180 only the conditional models move, through pre_lag L[79].
  - A spy confirms the window and pre_lag of every call: at the small design (W = 100 and W = T0) and at the real sizes (n = 7716, W = 1000 and 2000).
  - Two runs are byte-identical, and workers 1 and 3 are byte-identical.
  - An injected GarchError gives NaN and the flag for the 6 conditional keys only, and the error is counted. 1 error in 100 targets does not stop the pass; 2 errors set stop_for_owner. no_pre_window_lag is counted separately.
  - Run-directory hash, exact CSV round-trip, nothing written outside `reports/runs/`, and the gitignore check.
- **Methodology §3** gains the "rolling indexing (D41)" note.

## Check results (actually executed, Day 9)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **672 passed**, 0 skipped; total coverage 99% (3,192 statements, 23 missed; `rolling.py` 100%); 636.6 s |
| `uv run --locked mypy` | pass: "Success: no issues found in 19 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| sha256, unchanged | `docs/plan_v2.2.md` and all six tracked JSONs of the protected-hash list; `uv.lock`, `pyproject.toml` and `configs/base.toml` unchanged |

## Day 8B: what was done (2026-11-13; synthetic data only)
- **D40** was written before any test was run; its tolerances (f) were pinned first. The shift tolerance failed for the four models with a Nelder–Mead fit (worst 1.9e-7 × std(window)); the implementer stopped and reported, and the owner amended it in D40 (g).
- **Config:** `[conditional] filter = "F4"`, strictly parsed (one of F1–F4); a test asserts it equals the winner in `filter_selection.json`.
- **`models.py`:** `Forecast`, `MODELS`, `levels`, `gaussian`, `hs`, `pot_gpd`, `conditional` (single q or a tuple of q levels; `evt_key`), `forecast_all`. Methodology §18 gains the interface note; `docs/conditional.md` §11 records D40 (c) and (d).
- **Tests (`tests/test_models.py`, 26 cases):** the §4 table exact, §5 and the §6 worked example (full `pot_gpd` path with the fit pinned) within 1e-6; R2 tail selection; ES ≥ VaR; scale and shift equivariance for all six models, plus an exact-shift pot_gpd check at 1e-12 × std; permutation invariance of the unconditional models; exact conditional consistency; the multi-q test (one `fit_garch` call, exact equality with single-q calls); baselines; every failure path; determinism; the signature contract; the config key.
- **Baselines** (D40 (f); i.i.d. N(0,1), W = 1000, R = 20): mean gaussian VaR99 2.3158 (tolerance 0.05 around 2.326348), mean hs VaR99 2.3036 (0.15), mean |garch_n − gaussian| 0.0568 (0.10). All pass.
- **Deviations:** the sorted-window detail and the "no excess" guard in `pot_gpd` (D40 (e)); the "mean absolute difference" reading of the garch_n baseline (D40 (f)); the shift tolerance, amended by the owner (D40 (g)).

## Check results (actually executed, Day 8B)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **653 passed**, 0 skipped; total coverage 99% (3,074 statements, 23 missed; `models.py` 100%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 18 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| sha256, unchanged | `docs/plan_v2.2.md` `44f42b13…5d66d`; `nmin_study.json` `5a8349a7…64fc97`; `simstudy/metrics.json` `cfa424e0…3212b5`; `real/selection.json` `7929c129…741f9`; `real_residuals/selection.json` `97249cef…98ac44`; `filter_selection.json` `84ee40f4…102a12`; `selection_filter.json` `ebe0309b…915c84`. `uv.lock` and `pyproject.toml` unchanged. `configs/base.toml`: `[conditional] filter` added (D40) |

## Day 8A-2: what was done (2026-11-13; training data only)
- **D39** was written before any new fit was run: q_c by design; the correction to D38 (2); the BIC rule over F1–F4 with its reparameterisation, starts and guards; the AR information set; the diagnostics and the sign-bias convention; the recovery tolerances. Its part (h) records the outcome. Methodology §18 is rewritten for D39. Literature: L38–L40, all F; their metadata was not fetched (owner instruction).
- **Config:** new section `[conditional]` with `q_c = "0.88"`, strictly parsed (a grid member, ≤ q_max). `q_selected` is unchanged.
- **`garch.py`:** `fit_garch(losses, cfg, spec, lag=...)` for F1–F4; `garch_filter`, `garch_nll`, `forecast_sigma` extended; new `forecast_mean`, `n_params`, `bic`, `select_filter`. `GarchFit` gains `phi1`, `gamma`, `spec` (defaults keep F1 as before). No new dependency; `uv.lock` unchanged.
- **`diagnostics.py`:** `sign_bias` (Engle–Ng tests, D39 (e) convention), tested against statsmodels OLS.
- **Recovery study** (D39 (f); 3 cells × 50 series, n = 2000, normal; bias of the mean estimate): F2: φ̂1 −0.001982, α̂ −0.0002658, β̂ −0.01787; F3: α̂ −0.002231, γ̂ 0.003188, β̂ −0.001442; F4: φ̂1 0.001701, α̂ 0.0009242, γ̂ 0.004618, β̂ −0.009002. All pass (tolerances 0.03, 0.02, 0.04, 0.04); all 150 fits converged.
- **`arch` cross-check** (scratch venv outside the repo, not a test): `arch` 9.0.0. Largest parameter difference with the backcast fixed at mean(ε²): 2.38e-06 (F2), 5.97e-04 (F3), 8.25e-05 (F4). Our NLL at every `arch` estimate is above our minimum. Details: `docs/conditional.md` §10.3.
- **Command `filter-select`** (new module `filter_selection.py`): the four fits on the conditional sample t = 2…2000 (1999 likelihood terms). Run twice: exit 0, the tracked JSON byte-identical.
- **BIC** (from the JSON): F1 −10939.510, F2 −10948.542, F3 −10946.747, **F4 −10957.302**. F4: φ̂0 = −0.0002999, φ̂1 = 0.1053, ω̂ = 1.513e-05, α̂ = 0.06888, γ̂ = 0.0997, β̂ = 0.8304, α̂ + γ̂/2 + β̂ = 0.9491; no flag on any candidate.
- **Diagnostics, F4 against F1** (p-values, F1 → F4): Ljung–Box level lag 10: 6.221e-06 → 0.01927; lag 20: 7.702e-05 → 0.04026. Ljung–Box squares lag 10: 0.4581 → 0.3746; lag 20: 0.001131 → 5.173e-06. ARCH-LM 5 lags: 0.2819 → 0.1693; 10 lags: 0.471 → 0.4078. Sign-bias joint: 0.06628 → 0.5865. Lag-1 ACF of z: 0.1036 → 0.01371.
- **Limitation (D39 (e)):** F4 still rejects Ljung–Box on z² at lag 20 at 1%. It is reported; no further specification is added.
- **Descriptive re-run** (`threshold-select-residuals --selected-filter`; DESCRIPTIVE ONLY, no decision may cite it): on the 1999 F4 residuals the rule fails again, `all_rejected`, k̂ = 16, final ForwardStop statistic 0.08588; AD p-values 0.001 to 0.233; ξ̂ from 0.0413 (q = 0.80) to 0.3008 (q = 0.95); split halves 0.83 (k̂ = 3) and 0.80 (k̂ = 0). At q_c = 0.88: ξ̂ = 0.1414 [0.02969, 0.2921]. Output: `real_residuals/selection_filter.json` and five figures under `real_residuals/selection_filter/`.
- **Deviations from the task, all recorded in D39:** (1) the descriptive run passes a configuration copy with train_size = 1999, because the winning fit has 1999 residuals and `select_threshold` accepts exactly train_size values; this handling is the implementer's and is flagged for the owner. (2) "crossed" starts were read as 2 × 2 = 4 starts for the AR specs. (3) One synthetic small-design test (`test_threshold.py`, W = 100, n_min = 50) now builds its configuration as a copy of the live one instead of re-parsing it, because no grid value of q_c is valid for that design; no assertion changed. (4) `test_threshold_residuals.py`: the tracked-file set of `real_residuals/` now includes `selection_filter.json`, and the Day-8A sha256 is pinned in the test.

## Day 8A: what was done (2026-11-13; training data only)
- **D38** was written before any GARCH code was run: the 2×3 design, the filter and its numerical conventions, conditional risk formulas (for Day 8B), the OOS failure policy, the conditional comparison family (6 tests, Holm at 5%), sensitivity runs, limitations, the recovery-study tolerances and the residual-evidence specification. Methodology §18 is new; §16 item 2 is amended. Literature: L15 (extra claim), L35–L37, all F; their metadata was not fetched (owner instruction).
- **Config:** new `[garch]` section (xatol 1e-8, fatol 1e-10, maxiter 5000, maxfev 10000, simplex_step 0.1), strictly parsed. `q_selected` is unchanged and no q_c key exists.
- **`garch.py`:** `fit_garch`, `garch_filter`, `forecast_sigma`, `garch_nll`, `constrain`/`unconstrain`, `GarchError`. No new dependency; `uv.lock` unchanged.
- **Recovery study** (D38 (8); 4 cells × 50 series; bias of α̂ / bias of β̂): normal n = 1000: 0.003194 / −0.01877; normal n = 2000: 0.000495 / −0.005199; t(6) n = 1000: 0.002856 / −0.01705; t(6) n = 2000: 0.004337 / −0.01413. All pass (tolerances 0.02 and 0.04); all 200 fits converged.
- **`arch` cross-check** (scratch venv outside the repo, not a test): `arch` 9.0.0 installed and ran. On one simulated series the estimates differ by 4e-6 (μ), 2e-6 (ω), 4e-5 (α), 9e-6 (β) with the backcast fixed at mean(ε²); details in `docs/conditional.md` §2.3.
- **Command `threshold-select-residuals`** (new module `threshold_residuals.py`; `threshold_real.py`, `threshold.py` and the 7C outputs are untouched): one GARCH fit on the 2000 training losses, then the D34 procedure on the residuals with seed node SeedSequence(20261111).spawn(2)[1]. Run twice: exit status 1 both times (rule failure), 6 tracked files byte-identical.
- **Training GARCH fit** (from the JSON): μ̂ = −0.0006021, ω̂ = 1.175e-05, α̂ = 0.1062, β̂ = 0.8551, α̂ + β̂ = 0.9613; no flag; start 0.
- **Residuals vs raw losses** (p-values, raw → residual): Ljung–Box level lag 10: 0.00364 → 5.531e-06; lag 20: 0.004465 → 7.2e-05. Ljung–Box squares lag 10: 1.481e-29 → 0.4618; lag 20: 1.805e-34 → 0.001158. ARCH-LM 5 lags: 7.645e-20 → 0.2849; 10 lags: 8.373e-19 → 0.4755. Serial correlation in the level is not removed by the constant-mean filter.
- **Selection on residuals:** k̂ = 16, q_FS undefined, q_c undefined, rule failure `all_rejected`. The ForwardStop statistic is ≤ α_FS = 0.10 only at k = 1 (0.02327) and k = 16 (0.09934). AD p-values are between 0.015 and 0.237. Split halves: 0.83 (k̂ = 3) and 0.80 (k̂ = 0); the split-half flag is not evaluated when the full-sample q is undefined. ξ̂ at 0.90: 0.1389 [0.01711, 0.3046]; at 0.95: 0.2491 [0.05886, 0.5208].
- With no q_c, the three per-day residual figures (drawn at q_c only) were not written; `real_residuals/figures/` is gitignored for when they are.

## Day 7C-1: what was done (2026-11-12; first use of the real training data)
- **D36** was written before the real data was loaded: owner decision (i) on the fired false-selection trigger (α_FS = 0.10, procedure unchanged, study not re-run), the limitation that goes with it, the pinned OOS sensitivity set, the `decluster_fs` convention and the 7C diagnostics.
- **`decluster_fs`** (`diagnostics.py`): Ferro–Segers intervals declustering. C − 1 = ⌊θ̂·(N − 1)⌋ in exact integers, r = T_(C) (0 if C = N), clusters separated by the times > r; ties stay inside a cluster. Read in Ferro's thesis §4.1.5 (literature L11a, C1-sections); the JRSS-B paper stays unread (L11, F). `extremal_index_fs` now shares an exact-rational helper with it; its results are unchanged.
- **`src/niftytail/threshold_real.py`** (new) and the command **`threshold-select`**: manifest-verified load, first 2001 prices only, `select_threshold` with B = 999, then the diagnostics (extremal index, declustered-ξ comparison, `xi_comparison`, grid-floor flag, mean residual life). Outputs: `selection.json` and five SVG figures (tracked, aggregates only); three per-day figures (gitignored).
- **`plots.py`**: the three exceedance figures moved into `write_exceedance_figures`, reused at q\*; the EDA figures are unchanged.
- **Real run, twice:** 268.2 s and 265.2 s, exit 0; all six tracked files byte-identical. `selection.json` sha256 `7929c129…e86741f9`.
- **Result (details and all tables: `docs/threshold_selection.md`):**
  - ForwardStop rejects q = 0.80–0.87 (k̂ = 8); q_FS = 0.88; the window 0.88–0.91 passes, so q\* = 0.88.
  - ξ̂ side by side: 0.1157 [−0.0092, 0.2753] at q\*, 0.1423 [−0.0007, 0.3268] at 0.90, 0.1547 [−0.0710, 0.4747] at 0.95.
  - Split-half: returns 1–1000 give no stable region (q_FS = 0.91 > q_max); returns 1001–2000 give 0.80. Flag raised.
  - θ̂ = 0.6324 at q\* (135 clusters from 240 exceedances). Declustered ξ̂ = 0.0872 [−0.0792, 0.3117]; each estimate lies inside the other's CI.
- **Tests:** `tests/test_threshold_real.py` (15 functions) and 9 new cases in `tests/test_diagnostics.py`; synthetic data only; no existing test was changed, removed or renamed.
- **Deviations from the task text (reported):**
  1. β\* is drawn without a CI in figure (b): no CI formula for β\* is pinned.
  2. The CLI tests use a copy of the config with B = 19 (speed); the rule-failure test forces the failure with a tolerance of 1e-9 and stubs the convergence column.
  3. The JSON stores the split halves as q, q_FS, k̂ and rule failure, not their full candidate tables.
  4. `selection.json` records lower-case hex for all hashes (the protected-hash list writes the `nmin_study.json` hash in upper case).
  5. A test reads the committed `selection.json` (aggregates only) to check its body hash, the pinned hashes and the absence of per-day data.
- **Not verified:** see `docs/threshold_selection.md` §10.

**Current milestone:** **Day 7B is COMPLETE** (D35: the A3 threshold-procedure simulation study), committed locally.
- **A review trigger FIRED.** False selection, P(q\* < q0), is 0.915 (S3, a = 0.3) and 0.965 (S3, a = 0.6) against the pre-declared line 0.20. The FDR and rule-failure triggers did not fire. Details: `docs/threshold_study.md` §4.
- **Nothing was decided or changed in response.** Per plan A3 the owner chooses among (i) proceed as specified, (ii) raise α_FS, (iii) apply the GoF test to declustered cluster maxima. Any change re-runs the study before the real training data is touched.
- **Day 7C (the real selection) has not started** and does not start before the owner's decision on the trigger and an explicit go-ahead.
- Synthetic data only: no real series was loaded, and no threshold was selected.
- ENV-1 is mitigated by D24; mypy passes (strict, 13 source files, no new override).

## Day 7B: what was done (2026-11-12; synthetic data only)
- **D35** pins what the plan leaves open for A3: sampling and truth formulas, denominators, seed layout, aggregation, trigger arithmetic, the runtime guard and the output layout.
- **`src/niftytail/threshold_study.py`** (new; `simstudy.py` and `nmin_study.json` are untouched, sha256 `5A8349A7…64FC97`): DGPs S1–S4 and their truth, the S4 reference with batch-means error, one-replication records, metrics M1–M6, triggers, the `ProcessPoolExecutor` runner and five SVG figures.
- **`select_threshold`** has two new keyword switches, `halves` and `intervals` (both default True, so existing behaviour and tests are unchanged). A test shows that every rule-relevant output is identical with the switches off.
- **Command `threshold-study`:** pilot of 2 replications per cell, runtime guard, then the full study. It writes `reports/threshold_selection/simstudy/metrics.json` and `fig_a` to `fig_e` (`.svg`), all tracked.
- **Runs (R = 200, B = 199, 10 cells):**
  - 31 workers: pilot 26.0 s, projection 0.72 h, total 1,820.7 s (30.3 min).
  - 16 workers: pilot 47.0 s, projection 1.30 h, total 3,040.3 s (50.7 min).
  - **All six output files are byte-identical across the two runs.** `metrics.json` sha256 `CFA424E0…3212B5`.
- **Headline results (simulation evidence, DGPs S1-S4; full tables in `docs/threshold_study.md`):**
  - False selection is 0.88 to 0.965 in every cell with a boundary, i.i.d. (S1) and dependent (S3) alike. q\* = 0.80 is the most frequent choice in every cell.
  - Empirical FDR is 0.005 to 0.042, far below α_FS = 0.10, in every cell.
  - Rule-failure rate is 2% to 7.5%.
  - ξ̂(q\*) is biased downwards (−0.018 to −0.086 where ξ is known).
  - VaR and ES at q\* are nearly unbiased for the n = 2000 fit (within ±2.4% and ±3.2%). The relative RMSE of VaR99 is 5% to 8% in S1, 11% and 13% in S3 and 16% in S4.
  - No fit failed: 0 of 32,000 (candidate, replication) pairs; every window refit converged.
- **S4 check (D35 (2)):** the Day-6 pilot figure of 1.3 s for 10⁷ steps was correct. The pilot and the study both run the exact recursion in a plain Python loop, which takes about 100 to 160 ns per step here. The generator equals an independent loop on the same innovations (rel 1e-10, 10⁵ steps). The reference variance is 0.9911 with batch-means SE 0.0057, within 3 SEs of 1.
- **Figures:** tracked as SVG. The hygiene test forbids tracked market data and tabular files, not images; the figures show aggregates of synthetic data. Byte-identical across runs. No PNG is tracked.
- **Tests (`tests/test_threshold_study.py`, 81 cases in 39 functions, about 40 s under coverage).** No existing test was changed, removed or renamed.
- **Bug found and fixed before the run:** the S1 quantile function returned NaN at v = 1 for q0 = 0.85 (a rounding of (1 − q0) made w exceed 1). It is fixed and tested. v = 1 is never drawn, so no result depended on it.
- **Implementation choices not fixed by D35 (reported to the owner, not separately approved):**
  1. The runtime guard is built into the command: every run starts with the pilot and stops, writing nothing, if the projection exceeds the limit. The pilot time includes starting the worker processes and excludes the S4 reference.
  2. Seeds are addressed by spawn key, which equals the pinned `spawn` layout (tested against real spawning) and makes a record independent of R and of the worker count.
  3. S3 raises if a uniform draw is exactly 0 or 1. Otherwise the running maximum would absorb the invalid innovation and no non-finite loss would appear.
  4. The S4 path is simulated in blocks of 10⁶ innovations to limit memory; the block size does not change the path (tested).
  5. FDR: SD with ddof = 1; the trigger compares the exact rational mean with the limit.
  6. In M6 the window probabilities P(ξ̂ ≥ 0.5) and P(ξ̂ ≥ 1) use the replications with q\* defined as the denominator; a failed refit counts as "not above".
  7. `workers = 1` runs in the calling process, without a pool.
  8. The command's log line also carries the worker count and the pilot and projected times, besides file names and wall-clock.
  9. A side check outside `metrics.json` (40 samples in two cells) measured the AD rejection rate below the boundary. It is reported in `docs/threshold_study.md` §5 and labelled as not pre-declared.
- **Not verified:**
  - B = 999 (the real selection's setting) was not run.
  - Behaviour on processes other than S1–S4.
  - Rendering of the SVG files in a viewer was not inspected; only their bytes were compared.
  - The cause of the high false selection is not established by the pinned metrics; the side check is small.

## Day 7A: what was done (2026-11-12; synthetic data only)
- **D34** pins the definitions the plan leaves open, items (1)–(9), including the correction in item (6): stable-window members above q_max are allowed; only q\* is bounded by q_max.
- **`src/niftytail/threshold.py`:**
  - `candidate_table` (Steps 1–2: one row per grid candidate, with flags);
  - `gof_ad_bootstrap` (scipy `goodness_of_fit`, GPD, loc = 0, AD);
  - `forward_stop` (the maximum rule, +∞ at p = 1);
  - `stable_region` (Step 4, with the first failing condition per candidate);
  - `split_half` and `split_half_flag` (Step 5);
  - `window_convergence_share`;
  - `select_threshold` → `ThresholdDecision` (frozen dataclass, deterministic `to_json`).
  - No CLI command, no config change and no dependency change.
- **Tests (`tests/test_threshold.py`, 59 cases in 39 test functions; 26 s alone, about 37 s under coverage):**
  - ForwardStop: hand values, the non-monotone maximum case, k̂ = 0, k̂ = m, p = 1, and an independent numpy computation.
  - Stable region on hand-built tables: smallest q, inclusive boundaries, window existence, q_max binding q\* but not the window, bad members, and both rule failures.
  - GoF: determinism, p ∈ [1/(B+1), 1], rough uniformity under the null (KS statistic below 0.30 over 40 draws) and p = 0.01 (the minimum at B = 99) for a lognormal sample.
  - Candidate table: N_u and u conventions (with ties), β\*, VaR95 null at q = 0.95, equality with direct `gpd` calls, the seed layout, and every flag.
  - Split-half slices and the flag rule; end-to-end structure and byte-identical JSON; length guard; no import of the data layer; convergence share.
  - No existing test was changed, removed or renamed.
- **scipy `goodness_of_fit` (1.18.1), as observed:**
  - It makes B + 2 calls to `genpareto.fit`: the null fit to the data, a refit of the data for the observed statistic, and one refit per Monte Carlo sample. **Every call passes `floc=0` and nothing else** (proved by a recording wrapper in the tests).
  - It accepts `rng=` (a Generator); `random_state=` is the older spelling.
  - p = (1 + #{AD\* ≥ AD})/(B + 1). scipy's ξ̂ and ours agree to about 1e-5 on the test samples.
- **Implementation choices not fixed by D34 (reported to the owner, not separately approved):**
  1. `select_threshold` has an extra keyword `convergence_share` (default True, the D34 behaviour). The column costs about 16,000 `fit_mle` calls (about 30 s), so the A3 study will need False. Day 7B decides.
  2. `SeedSequence.spawn` is stateful, so children are built from the spawn key. This equals `ss.spawn(3)[i]` for a fresh `ss` and does not advance `ss`; an `ss` that has already spawned gives the same output as a fresh one.
  3. `fit_mle` is called with `min_excesses = 2` (the structural minimum, as in D32). N_min acts only through q_max.
  4. "raises" in D34 (3) is read literally: any exception from `fit_mle` or scipy's call gives `fit_failed`. The exception type name (never its message) is recorded in `fit_error`.
  5. Not covered by D34: for ξ̂ ≤ −½ the SE and delta-method CIs are undefined, and a profile CI can fail. These are null with flag "ci_unavailable"; the rule is unaffected.
  6. The VaR guard p < N_u/n is applied at both levels. "var99_not_in_tail" can occur only with heavy ties.
  7. "below_n_min_window" is set on every candidate with (1 − q)·W < n_min, that is, every q > q_max.
  8. Tolerances are compared exactly (float against Fraction, as in D32), so the tests use dyadic tolerances for the equality cases.
  9. A zero VaR99 or ES99 at the window base makes the relative difference undefined; that window fails as "not_finite".
  10. JSON: Fractions as "22/25"; a ForwardStop statistic of +∞ is written as the string "inf" (strict JSON has no infinity).
  11. The halves keep their full candidate tables. Their `convergence_share` is null (a half holds a single 1000-point window).
  12. `split_half` splits at n // 2 (1000 at the enforced length of 2000).
- **Not verified:** nothing was run on real data, by design. The ForwardStop and stable-region behaviour under dependence is the subject of Day 7B. B = 999 was not run.

## Day 6C: what was done (2026-11-12; synthetic data only)
- **D33:**
  - `[threshold] n_min = 100` and `[nmin_study] override_decision = "D33"`.
  - The `nmin-study` guard accepts a config n_min after a rule failure only with the override key. When the rule gives a result, the config n_min must equal it; unset or different is an error. A failing guard writes nothing.
  - The JSON gets an `adopted` block: n_min 100, source "D33", rule_n_min null, q_max "9/10".
- **New descriptive per-cell metric** `rel_rmse_var_level_ref` (plus `u_over_beta_ref`): the relative RMSE on a VaR level with a normal reference threshold. It is never part of the rule. At ξ = 0.4, N_u = 100 it is 0.09768, against the rule metric's 0.1558.
- **Regenerated `nmin_study.json`:** two runs gave sha256 `5A8349A7…64FC97`. Every 6B value is unchanged except `config_sha256`, which changed because the config did.
- **Test changed (intended):** the base-constants test now expects `n_min == 100` and `override_decision == "D33"`.

## Day 6B: what was done (2026-11-12; synthetic data only)
- **D32**, `src/niftytail/simstudy.py` and the `nmin-study` command write `reports/threshold_selection/nmin_study.json`. It is tracked and deterministic; two runs gave the same sha256, `86CB7B59…13BBD3`, at about 40 s each.
- **N_min: RULE FAILURE.**
  - Convergence was 2000/2000 in all 12 cells.
  - The relative RMSE of the VaR99 excess quantile at ξ = 0.4 is above 0.15 at every N_u: 0.1934, 0.1727, 0.1558 and 0.1507.
  - RMSE(ξ̂) passes at N_u ≥ 100 for every ξ.
  - Per D32 (4), `n_min` was not written and the grid was not extended. The full table is in `docs/nmin_study.md`.
- **Timing pilot** (`scripts/timing_pilot.py`):
  - Hardware: cpu_count 32, W = 31, e = 0.4423.
  - S = 10.28 s; T_total(200) = 0.4167 h and T_total(100) = 0.2086 h, so the rule gives **R = 200**.
  - Day-7 serial estimate of one real selection: 52.29 s.
  - `[simstudy]` is unchanged.
- **Config:** new `[nmin_study] var_level = "0.99"`. `[threshold] n_min` is validated as a member of `nmin_study.n_u` when present.

## Day 6A: what was done (2026-11-12; synthetic data only)
- **D31**, `gpd.py`:
  - The support check now runs in every NLL branch.
  - Boundary rule: ξ̂ ≤ −1 + 1e-6 raises. Irregular fits (ξ̂ ≤ −0.5) and fits with ξ̂ ≥ 1 are returned.
  - New functions: `se_xi`; `profile_ci_xi` (`ProfileCI`, with `lower_open`); `expected_covariance` (negative off-diagonal); `risk_gradient`; and `delta_ci` (`DeltaCI`).
  - Six new `[gpd]` keys.
- **Day-5 test changed (intended, D31 (2)):** the constant-y fit now raises at the ξ = −1 boundary instead of being returned.
- **Seeded outcomes:**
  - Uniform n = 30 hits the boundary rule.
  - GPD(ξ = −0.6, n = 30) is returned with `regular` False and `lower_open` True.
  - Profile-CI bounds reproduce χ²₁(0.95) within 1e-6.
  - MC recovery of ξ (bias and RMSE) passes at N_u = 400 for ξ ∈ {−0.25, 0.1, 0.4}.
  - Profile-CI coverage lies within [0.92, 0.98].
  - Delta SE / empirical SD(VaR99̂) = 0.994; the empirical N_u·cov off-diagonal is −1.39.
  - Descriptive delta-CI coverage: VaR99 0.925, ES99 0.917.
- **Correction during the review:** the delta-method covariance was pinned with +β off-diagonal; that sign was wrong. With +β the SE ratio was 1.86 and coverage 0.997. Corrected to −β in D31.

## Day 5: what was done (2026-11-12; synthetic data only, no real series loaded or fitted)
- **D30** pins the GPD estimator and its numerical conventions before any real fit. The config gets a new `[gpd]` section (Nelder-Mead, xatol "1e-10", fatol "1e-12", maxiter 5000, maxfev 10000) with strict parsing and tests.
- **`src/niftytail/gpd.py`** contains:
  - `EstimationError`;
  - `excesses` (strict L > u);
  - `nll` (+∞ outside the domain; ξ = 0 branch; log1p form);
  - `fit_pwm` (Hosking–Wallis, loc 0);
  - `fit_mle` (Nelder–Mead over (ξ, ln β), PWM start with exponential fallback, never clipped, `GpdFit` with a `regular` flag);
  - `tail_var` (exact rational r, p < n_u/n guard, expm1);
  - `tail_es` (∞ if ξ ≥ 1) and `tail_risk`.
- **Tests (`tests/test_gpd.py`, 86 cases):**
  - Hand cases.
  - The nll agrees with scipy `genpareto.logpdf` at rel 1e-12.
  - The MLE agrees with `genpareto.fit(floc=0)` within 1e-3, with an nll no worse than scipy's. The gradient at the optimum is below 1e-4, and no perturbation gives a lower nll.
  - VaR agrees with bisection and `isf` within 1e-10, and ES agrees with `quad` at rel 1e-8.
  - The worked example is reproduced at rel 1e-6.
  - Branch-switch checks: first-order expansion at ξ = ±1.1e-8; at ±0.9e-8, bit-equality with the zero branch plus the truncation bound.
  - Determinism, the exponential-fallback start (constant y, and a monkeypatched PWM failure on a GPD sample, where both starts reach the same optimum) and value-free error messages.
- **Not in Day 5:** profile and delta CIs, SE(ξ), the MC recovery and N_min studies, and the edge-case policy. These are Day 6.

## Day 4: what was done (2026-11-12)
- **D29** pins the battery constants and conventions before any data were seen. The config gets a new `[dependence]` section with strict parsing and tests.
- **Code:**
  - `losses.py`: returns, losses, training slice, the exact-ceiling tail threshold and the strict exceedance mask.
  - `diagnostics.py`: ACF, Ljung–Box, ARCH-LM, inter-exceedance times and the Ferro–Segers θ̂.
  - `eda.py`: the battery and a deterministic summary.
  - `plots.py`: six training figures (Agg backend, fixed dpi, no PNG metadata).
  - CLI command `eda`.
- **Tests:**
  - Hand cases.
  - statsmodels references (`acf`, `acorr_ljungbox`, `het_arch`) agree at rel 1e-10.
  - Seeded checks: iid ARCH-LM p > 0.01; ARCH(1) p < 0.001; iid θ̂ = 1; ARMAX θ̂ within 0.05 of 1 − a for a = 0.3 and 0.6. All pass with the pinned tolerances.
  - Log returns: absolute tolerance 1e-14 against `np.diff(np.log(p))` and against a 50-digit Decimal reference. A relative tolerance is meaningless for small returns (owner decision, Day-4A review).
  - Synthetic `eda` determinism, key whitelist, lists of 20 elements or fewer, and a docs/eda.md number-pattern check.
- **Real data:** `eda` exited 0 twice, and the `eda_summary.json` sha256 was identical in both runs. Figures are gitignored (`reports/eda/figures/`); the summary is tracked. Findings are in `docs/eda.md` §8.
- **Deferred (by D29):** declustering and the declustered-ξ comparison (Day 7), and the block bootstrap (Day 10).

## Day 1: complete (closeout 2026-11-11)
- `docs/plan_v2.2.md`: verbatim copy of the approved plan; sha256 `44f42b13…5d66d`, unchanged.
- `docs/reference_values.py`: every check is enforced (`CheckFailed`, exit status 1 on mismatch; verified by deliberately injecting a wrong value).
  - Includes R6b (Kupiec LR regions at T = 250) and NaN/inf/alpha validation for FZ0.
  - Run with `py -3.13 -I docs\reference_values.py`: **exit 0, "ALL 1386 CHECKS PASSED"**. The output is byte-identical to the Appendix A record.
- `docs/methodology.md`: A3 review choices and re-run requirement, A3 outputs, A4 outputs, A5a checks, A5b HAC comparison, the full A5e list, and a separate table of LR vs equal-tailed regions.
- `docs/decisions.md`: D21 (DX-1) and D22 (DX-2) approved.
- `docs/literature.md`: Bader et al. marked C1-sections (statements checked in the arXiv v1 text; not a full read).

## Day 2: what was done
- `git init -b main` with a repo-local author (Sambit Sinha). **No remote.**
- `uv init --lib`: `requires-python = ">=3.13,<3.14"` and `.python-version = 3.13`.
- **Environment-specific setting (kept by owner decision, 2026-11-11):** `pyproject.toml` contains `[tool.uv] python-preference = "only-system"`.
  - **Reason:** on this machine, Windows Smart App Control (state On) blocks uv's *managed* Python interpreter (`AppData\Roaming\uv\python\cpython-3.13-...\python.exe`, os error 4551).
  - The setting makes uv use the installed system Python 3.13.17 (`AppData\Local\Python\pythoncore-3.13-64`).
  - It changes no dependency or version. On machines without this restriction it only stops uv from downloading its own interpreter, so a system Python 3.13 must be installed there.
- `uv add` of the authorized dependencies, then `uv sync --all-groups`: initially 61 packages locked in `uv.lock`; the venv runs Python 3.13.17.
- **ENV-1 version change (D24, 2026-11-11):** pandas 3.1.1 → 2.3.3, statsmodels 0.15.0 → 0.14.6, mypy 2.4.1 → 1.20.2 and pandas-stubs 3.1.0.261029 → 2.3.3.260113. The newer compiled wheels were blocked by Smart App Control; the older wheels load.
  - Dependency fallout: pytz and types-pytz were added; ast-serialize, formulaic, interface-meta and wrapt were removed. 59 packages are now locked.
  - numpy, scipy, matplotlib, ruff, pytest, pytest-cov and streamlit are unchanged.
  - `tests/test_environment.py` is an ENV-1 canary (pandas CSV round trip; statsmodels HAC OLS). It is not skippable.
- `.gitignore` covers market data and tabular files, `reports/runs/`, the venv, caches and coverage output.
- Frozen-plan protection: `docs/plan_v2.2.md` is excluded from ruff (`extend-exclude` in `pyproject.toml`; ruff's formatter also formats Markdown). Its SHA-256 is pinned by `test_frozen_plan_sha256` in `tests/test_repo_hygiene.py`.
- `.gitattributes`: `* text=auto eol=lf` keeps LF line endings in the working tree on every platform, preserving the approved sha256 of `docs/plan_v2.2.md` and byte-identical artifacts (A5e).
- Tool configuration for ruff, mypy (strict, `src`), pytest and coverage.
- Code:
  - `config.py`: exact Fraction parsing and strict validation.
  - `__main__.py`: BLAS/OpenMP thread pinning, JSON-lines logging, `check-config`.
  - `experiment.py`: **run-manifest metadata only** (A5e item 4; A6 Day 2). It records:
    - Python version and all installed packages with their versions;
    - platform (no hostname);
    - numpy BLAS/LAPACK build configuration plus the thread-pinning variables;
    - git commit and dirty flag;
    - config sha256 and the sha256 of every input data file.

    Paths are repository-relative, there are no timestamps, and the JSON is written deterministically. There is no experiment execution, data loading, model fitting or forecasting.
  - `configs/base.toml`.
- Tests: `test_config.py`, `test_main.py`, `test_experiment.py`, `test_repo_hygiene.py`, `test_environment.py` and `conftest.py`.

## Day 3A: what was done (synthetic tests only; the real data was not validated yet)
- **Format inspection** (read-only, structure only, no values; D23) of the owner's 32 NSE files in `data/raw/`:
  - 7,717 rows. Windows are contiguous and non-overlapping, from 1995-11-03 to 2026-11-11 (the last window is partial).
  - UTF-8 without BOM, CRLF line endings, no trailing newline, every field quoted.
  - Identical header in all files: `"Index Name","Date","Open","High","Low","Close"`. Index Name is always `NIFTY 50`.
  - Date pattern `dd Mon yyyy`, rows newest first.
  - Close is always a plain decimal. Open/High/Low hold 3 `-` placeholders in the first file.
  - First date 1995-11-03, 0 out-of-window dates, 0 duplicate dates within files, 42 weekend dates in total.
  - No STOP condition was met.
- **Config:** `[data]` raw_dir, manifest, validation_dir (relative paths, no `..`); new `[validation]` section (large_return_flag "0.05", gap_flag_days 5, stale_run_min 3, spot_check_n 10).
- **`src/niftytail/data.py`:** strict parser matching the inspected format; manifest writer (deterministic TOML, never overwrites) and reader; `load_raw` (manifest verified first); `validate` (all D25 hard errors collected, report-only flags, seeded spot-check dates, informational base-value check); `write_outputs` (JSON, CSV and text with dates, counts and hashes only).
- **CLI:** `validate <config> [--init-manifest --retrieved YYYY-MM-DD --source-url URL]`, with `niftytail.data` imported lazily. Outputs go to `data/validation/` (gitignored).
- **Tests:** `tests/test_data.py` uses synthetic files in the inspected format and covers every hard error and flag, the manifest round trip and refusal to overwrite, CLI exit codes, byte-identical reruns, value-free outputs, the seeded spot check and git-ignoring of outputs. `tests/test_config.py` is extended.

## Day 3B: what was done (first contact with the real data)
- **Code (D26):** absent-weekday audit in `src/niftytail/data.py` (report only). It adds `YearAudit`, `AbsentWeekday`, `ValidationReport.yearly` and `.absent_weekdays`. `validation_report.json` gains a `yearly` key, and a new output `absent_weekdays.csv` has the columns date, weekday and run_length. Existing outputs and keys are unchanged, and the CLI "outputs" list includes the new file automatically. Nine synthetic tests were added; no existing test was changed.
- **Real run:** `validate configs\base.toml --init-manifest --retrieved 2026-11-11 --source-url https://www.niftyindices.com/reports/historical-data` exited 0, wrote `data/raw/MANIFEST.toml` and reported 0 hard errors.
  - A rerun produced byte-identical outputs and an identical manifest (SHA-256 compared).
  - A second `--init-manifest` was refused ("manifest already exists; refusing to overwrite").
  - All 32 raw CSVs are byte-identical to the pre-run snapshot and match the manifest (32/32).
- **Actual counts:** 32 files; 7,717 rows; 1995-11-03 to 2026-11-11.
  - Flags: weekend_date 42, missing Open/High/Low 1/1/1 (3 cells, one row), calendar_gap 15, large_return 80, stale_close_run 0, ohlc_inconsistent 0.
  - 419 absent weekdays (375 single days, 20 runs of ≥ 2); first_close_equals_base_value true.
  - Every figure from the Day-3A inspection matches, so no STOP condition was met.
- **Audits (`docs/data.md` §5):**
  - Weekend dates: 29 EXPLAINED, 13 UNEXPLAINED.
  - Absent-weekday runs: 7 EXPLAINED, 13 UNEXPLAINED. The 375 single-day absences are UNVERIFIED.
  - Market-wide halts: 7 public halt/suspension events were matched by date only.
  - All sources were identified via web-search summaries; the cited pages were not opened (fetches timed out).
- **Sample size:** n_returns 7,716 and OOS 5,716 (T0 = 2000). The plan and methodology use "≈ 5,650"; this is listed for owner decision and nothing was edited.
- **NSE methodology PDF** (`data/docs/`, owner-supplied, September 2026): base date Nov 03, 1995 and base value 1000 (p. 284). It does not mention market-wide circuit breakers or halts.
- **Site terms:** not re-verified (fetch timed out). The plan §5b wording is quoted in `docs/data.md`.
- **Pending:**
  - The owner's spot check of 10 dates (`docs/data.md` §8, "PENDING: owner results").
  - The owner decisions listed in `docs/data.md` §10.
  - The TRI download (not yet needed).

## Day 3C: what was done (closeout, 2026-11-12)
- **Spot check:** 7 of 10 dates were compared with public reference closes. The reviewer supplied them from secondary sources; the owner did not perform the comparison.
  - All 7 are MATCH: 5 exact, 1 within 0.05 points (RBI rounding) and 1 within 1.00 points.
  - The comparison ran via `load_raw` (manifest verified first) in a scratch script outside the repo, and only MATCH/MISMATCH was printed.
  - The three 1997–1998 dates are unchecked (no independent source). Sources are listed in `docs/data.md` §8.
- **D27 numbers** (scratch computation, D19 definition): the equal-tailed regions at T = 5,650 reproduce [251, 315], [42, 72] and [2, 11]. At T = 5,716 they are [254, 319], [43, 72] and [2, 11]. Also at T = 5,716: NW lag 9 (exact and float), block length 18, divmod(5716, 250) = (22, 216).
- **Code (D28):** report-only flag `identical_close_pair` in `src/niftytail/data.py`: exactly two identical consecutive closes, dated at the second row and disjoint from `stale_close_run`. It is included in `FLAG_CATEGORIES` and the counts. Seven synthetic tests were added; no existing test was changed.
- **Real-data rerun:** `validate` exited 0 with 0 errors.
  - Unchanged counts: n_rows 7,717, weekend_date 42, calendar_gap 15, large_return 80, stale_close_run 0, ohlc_inconsistent 0, absent weekdays 419.
  - New count: **identical_close_pair 8**, all on weekdays. None involves a weekend-dated row or an UNEXPLAINED weekend date.
  - A second run was byte-identical, and the manifest sha256 is unchanged (`87DCA720…E0C04`).
- **Decisions:** D27 (audit dispositions: all rows kept, labels documentation only, halt days kept with a limitation, sample-size numbers at the actual T = 5,716, TRI before Day 11) and D28 (identical-close pair flag).
- **Still open (`docs/data.md` §10):** terms of use not re-verified; EXPLAINED labels rest on search-result summaries; 3 spot-check dates unchecked.

## Day 7C-2 (2026-11-12)
- D37: q_selected = 0.88; config line `q_selected = "0.88"  # D37`. No code change. No real-data or OOS run.
- Two tests that pinned the pre-gate state "q_selected unset" now pin the D37 value (`test_config.py` line 86, `test_main.py` line 74). One synthetic small-design test (`test_threshold.py`, W = 100, n_min = 50) built from the live config now drops the copied `q_selected` key, which is invalid for that design; no assertion changed.

## Check results (actually executed, Day 8A-2)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **627 passed**, 0 skipped; total coverage 99% (2,947 statements, 23 missed; `garch.py`, `filter_selection.py`, `diagnostics.py`, `threshold_residuals.py`, `threshold_real.py`, `threshold.py` 100%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 17 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| sha256, unchanged | `docs/plan_v2.2.md` `44f42b13…5d66d`; `nmin_study.json` `5a8349a7…64fc97`; `simstudy/metrics.json` `cfa424e0…3212b5`; `real/selection.json` `7929c129…741f9`; `real_residuals/selection.json` `97249cef…98ac44`. `uv.lock` and `pyproject.toml` unchanged. `configs/base.toml`: `[conditional] q_c` added (D39); `q_selected` unchanged |
| F1 path unchanged | fits on four synthetic series byte-identical before and after the change; the generalised code reproduces the `garch` block, moments and serial tests of the Day-8A `selection.json` exactly |
| `filter-select`, twice | exit 0 twice; under a second; `reports/conditional/filter_selection.json` byte-identical, sha256 `84ee40f40e85e2df4eb0013bbb75482b97f7ed34edbdd44463d979eee3102a12` |
| `threshold-select-residuals --selected-filter`, twice | exit 1 twice (rule failure `all_rejected`; descriptive only); 219 s and 248 s; six tracked output files byte-identical; `selection_filter.json` sha256 `ebe0309b3ad02c1c4ef8616767a332d5ce670e385a1aa7dde8c1662fe5915c84` |

## Check results (actually executed, Day 8A)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **586 passed**, 0 skipped; total coverage 99% (2,750 statements, 23 missed; `garch.py`, `threshold_residuals.py`, `threshold_real.py`, `threshold.py` 100%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 16 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| sha256 | `docs/plan_v2.2.md` `44f42b13…5d66d`; `nmin_study.json` `5a8349a7…64fc97`; `simstudy/metrics.json` `cfa424e0…3212b5`; `real/selection.json` `7929c129…741f9`; all unchanged. `uv.lock` and `pyproject.toml` unchanged. `configs/base.toml`: `[garch]` added (D38); `q_selected` unchanged |
| `threshold-select-residuals`, twice | exit 1 twice (rule failure `all_rejected`); 248 s and 262 s; six tracked output files byte-identical; `real_residuals/selection.json` sha256 `97249cef7ced190edae6d07cdf7e406cdea500ac7768805c204d0f9acb98ac44` |

## Check results (actually executed, Day 7C-1)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **557 passed**, 0 skipped; total coverage 99% (2,542 statements, 23 missed; `threshold_real.py`, `diagnostics.py`, `threshold.py` 100%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 14 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| sha256 | `docs/plan_v2.2.md` `44f42b13…5d66d`; `nmin_study.json` `5a8349a7…64fc97`; `simstudy/metrics.json` `cfa424e0…3212b5`; all unchanged. `uv.lock`, `pyproject.toml` and `configs/base.toml` unchanged |
| `threshold-select`, twice | exit 0 twice; 268.2 s and 265.2 s; six tracked output files byte-identical |

## Check results (actually executed, Day 7B)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **533 passed**, 0 skipped; total coverage 99% (2,274 statements, 23 missed; `threshold.py` and `threshold_study.py` 100%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 13 source files"; `pyproject.toml` unchanged |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| sha256 | `docs/plan_v2.2.md` `44F42B13…5D66D`; `uv.lock`, `pyproject.toml` and `configs/base.toml` unchanged; `nmin_study.json` `5A8349A7…64FC97` |
| `threshold-study`, 31 then 16 workers | exit 0 twice; 1,820.7 s and 3,040.3 s; six output files byte-identical |

## Check results (actually executed, Day 7A)

| Check | Result |
|---|---|
| `uv run --locked pytest -q -rs --cov` | **452 passed**, 0 skipped; total coverage 98% (1,864 statements, 23 missed; `threshold.py` 100%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 12 source files"; `pyproject.toml` unchanged |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass |
| `docs/plan_v2.2.md` sha256 | `44F42B13…5D66D` (unchanged); `uv.lock` unchanged |

## Check results (actually executed, Day 6)

| Check | Result |
|---|---|
| 6A `uv run --locked pytest -q` | **368 passed**; slow MC tests 6.54 s (profile coverage), 1.86 s (delta), 0.99 / 0.91 / 0.90 s (recovery) |
| 6B `uv run --locked pytest -q -rs --cov` | **379 passed**, 0 skipped; total coverage 98% (1,637 statements, 23 missed; `simstudy.py` 96%, `gpd.py` 97%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 11 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass (34 files, `scripts/` included) |
| `nmin-study`, twice | exit 0; sha256 `86CB7B59…13BBD3` in both runs; 38.8 s and 44.7 s |
| `scripts/timing_pilot.py` | exit 0; 41.2 s; rule gives R = 200 |

## Check results (actually executed, Day 5)

| Check | Result |
|---|---|
| `uv run --locked pytest -q --cov` | **330 passed**, 0 failed; total coverage 98% (1,439 statements, 18 missed; `gpd.py` 97%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 10 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass (29 files) |

## Check results (actually executed, Day 4B)

| Check | Result |
|---|---|
| `uv run --locked pytest -q --cov` | **234 passed**, 0 failed; total coverage 98% (1,293 statements, 16 missed) |
| `uv run --locked mypy` | pass: "Success: no issues found in 9 source files" |
| `uv run --locked ruff check .` / `ruff format --check .` | pass / pass (27 files) |
| `eda` on the real data, twice | exit 0 both times; `eda_summary.json` sha256 `CF2CF0F0…6053D5` in both runs |
| `git check-ignore -v` | figures ignored (`.gitignore:14`); `eda_summary.json` not ignored |
| Spot-check value grep (scratch, counts only) | 0 matches in 37 tracked files |

## Check results (actually executed, Day 3C)

| Check | Result |
|---|---|
| `uv run --locked ruff check .` | pass (exit 0) |
| `uv run --locked ruff format --check .` | pass (exit 0; 19 files) |
| `uv run --locked pytest --cov -rs` | **177 passed**, 0 skipped, 0 failed; total coverage 97% (948 statements, 15 missed; `data.py` 99%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 5 source files" |
| `uv run --locked python -m niftytail check-config configs\base.toml` | exit 0, "configuration valid" |
| `py -3.13 -I docs\reference_values.py` | exit 0, "ALL 1386 CHECKS PASSED" |
| `Get-FileHash docs\plan_v2.2.md` | `44F42B13…5D66D` (unchanged) |
| `validate` on the real data | exit 0; 0 hard errors; deterministic rerun; manifest unchanged |

## Check results (actually executed, Day 3B)

| Check | Result |
|---|---|
| `uv run --locked ruff check .` | pass (exit 0) |
| `uv run --locked ruff format --check .` | pass (exit 0; 19 files, now including `docs/data.md`) |
| `uv run --locked pytest --cov -rs` | **170 passed**, 0 skipped, 0 failed; total coverage 97% (946 statements, 15 missed; `data.py` 99%) |
| `uv run --locked mypy` | pass: "Success: no issues found in 5 source files" |
| `uv run --locked python -m niftytail check-config configs\base.toml` | exit 0, "configuration valid" |
| `py -3.13 -I docs\reference_values.py` | exit 0, "ALL 1386 CHECKS PASSED" |
| `Get-FileHash docs\plan_v2.2.md` | `44F42B13…5D66D` (unchanged) |
| `validate` on the real data | exit 0; 0 hard errors; deterministic rerun |

## Check results (actually executed, Day 3A)

| Check | Result |
|---|---|
| `uv run --locked ruff check .` | pass (exit 0) |
| `uv run --locked ruff format --check .` | pass (exit 0; 18 files: the 13 `.py` files plus 5 Markdown files; `docs/plan_v2.2.md` excluded) |
| `uv run --locked pytest --cov` | **161 passed**, 0 skipped, 0 failed; total coverage 97% (909 statements, 15 missed; `data.py` 99%) |
| `uv run --locked python -m niftytail check-config configs\base.toml` | exit 0, "configuration valid" |
| `py -3.13 -I docs\reference_values.py` | exit 0, 1386 checks passed |
| `uv run --locked mypy` | **pass** (mypy 1.20.2, strict): exit 0, "Success: no issues found in 5 source files" |

## Open blockers (owner decisions required before Day 3)
1. **ENV-1 mitigated by D24.**
   - Windows Smart App Control (state On) blocked the compiled wheels of pandas 3.1.1, statsmodels 0.15.0 and mypy 2.4.1.
   - The older wheels pinned by D24 load: verified 3/3 in a scratch probe, and in the repo by the canary tests and a real mypy run.
   - Smart App Control stays on, and WSL2 is not used.
   - Blocking is reputation-based and not guaranteed. If it returns, the canary tests fail and ENV-1 re-opens (fallbacks are in D24).
2. **D11, data licensing:** resolved by D23; the owner downloads the data manually (Day 3).

## Other known issues
- **D39 (open):** q_c is undefined after the Day-8A rule failure; GARCH-EVT cannot be implemented until the owner decides.
- `real_residuals/selection.json` stores the config_sha256 of the Day-8A config (with `[garch]`); the older artefacts store earlier config hashes (below).
- `threshold_residuals.write_figures` swaps `threshold_real.LABEL` while drawing, because the 7C module was not to be edited on Day 8A; a label argument on `threshold_real.write_figures` would be cleaner once that file may be touched.
- The gitignored per-day residual figures reuse `plots.write_exceedance_figures`, whose titles say "training losses"; the files under `real_residuals/figures/` show residuals.
- `selection.json`, `simstudy/metrics.json` and `nmin_study.json` store config_sha256 `b602725a…cf0e82`, the pre-D37 config. D37 changed only `q_selected`; the artefacts were not regenerated (as `eda_summary.json` above).
- `reports/eda/eda_summary.json` records the config_sha256 of the Day-4 config. It is stale relative to the current `configs/base.toml` and was not regenerated (out of scope).
- GPD (D30):
  - ES is finite only for ξ < 1; at ξ ≥ 1 it is returned as ∞.
  - The MLE domain is ξ > −1, and `regular` flags ξ > −0.5.
  - Since D31, a fit at the ξ = −1 boundary (ξ̂ ≤ −1 + 1e-6) raises EstimationError. Fits with −1 + 1e-6 < ξ̂ ≤ −0.5 are returned with `regular = False`, and their profile CI may be open on the left.
  - Delta-method CIs ignore the uncertainty of u and n_u/n, and need ξ̂ > −½. ES CIs need ξ̂ < 1.
  - Profile CIs raise if there is no upper crossing by ξ = 10.
  - Inside |ξ| < xi_zero_tol the zero branch drops the first-order term. The truncation error is about |ξ|·|f′(0)|: below 1e-8 relative for VaR at c = 0.99 on the worked example, and below about 4e-7 absolute for an n = 200 NLL.
- scipy is untyped to mypy: a narrow `[[tool.mypy.overrides]]` (`scipy`, `scipy.*` only) sets `ignore_missing_imports`; scipy-stubs was not added, to keep `uv.lock` unchanged. Revisit scipy-stubs at Day 5.
- Many foundational literature claims are still F-status (`literature.md`).
- The `pyproject.toml` `authors` field contains the owner's email (inserted by `uv init` from the git identity). It is harmless locally; review it before any publication.
- Before any publication (D23): internship organisation's approval; author e-mail decision (pyproject.toml and commit history); LICENSE file; history scan for data and secrets. No remote exists until the owner instructs it.

## Git status
Local commits on `main`, in order:
1. "Project scaffold"
2. "Correct stale ENV-1 commit note"
3. "Exclude frozen plan from ruff; pin its SHA-256"
4. "Record D23: data acquisition option (a), resolves D11"
5. "Move to older pandas/statsmodels/mypy wheels (D24, ENV-1 mitigation)"
6. "Fix mypy literal-required finding in blas_info"
7. "Day 3A: data loader, validator, manifest and validate command (D25)"
8. "Day 3B: real-data validation, absent-weekday audit and docs/data.md (D26)"
9. "Day 3 closeout: spot check, identical-close flag, audit dispositions (D27, D28)"
10. "Day 4A: losses and dependence diagnostics (D29)"
11. "Day 4B: EDA command, dependence findings (D29)"
12. "Day 5: GPD core (D30)"
13. "Day 6A: GPD uncertainty and edge-case policy (D31)"
14. "Day 6B: N_min study and A3 timing pilot (D32)"
15. "Day 6C: N_min override D33 (N_min = 100, q_max = 0.90)"
16. "Day 7A: threshold procedure core (D34)"
17. "Day 7B: A3 threshold-procedure study (D35)"
18. "Add the project rules"
19. "Day 7C-1: real threshold selection evidence (D36)"
20. "Day 7C-2: q_selected = 0.88 (D37)"
21. "Day 8A: conditional-model amendment, GARCH filter, residual threshold evidence (D38)"
22. "Day 8A-2: filter specification by training BIC, q_c = 0.88 (D39)"
23. "Day 8B: six-model forecasts (D40)"
24. "Day 9: rolling engine, leakage and alignment tests (D41)"
25. "Day 10: backtests, reference values reproduced (D42)"
26. "ENV-1: project venv without the dashboard group (D43)"
27. "Day 11A-1: OOS run command and provenance (D44)"
28. "Day 11A-2: W=1000 and W=2000 OOS results (D44)"
29. "Day 11B-1: OOS block threshold check command (D45)"
30. "Day 11B-2: OOS block threshold results (D45)"
31. "Day 11R: results review and TRI design (D46)"
32. "Day 12A-1: rolling xi path command (D47)"
33. "Day 12A-2: rolling xi path results (D47)"
34. "Day 11C-1: TRI data layer and run command (D48)"
35. "Day 11C-2: TRI W=1000 results (D48)"

There is still no remote. Market data, the manifest and validation outputs are gitignored and untracked. ENV-1: the project `.venv` has no dashboard group (D43).

