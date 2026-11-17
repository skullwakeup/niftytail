# NIFTYTAIL: per-step commands (moved from the README, Day 14-3)

The Day-2 to Day-8A command blocks below are moved unchanged from the README (D61 (g)). The ordered path that uses them is the README section "Reproduce the study"; the record of the byte-identical reruns is [reproducibility.md](reproducibility.md).

## Command line

```powershell
uv run --locked python -m niftytail check-config configs\base.toml
```

On success, it prints one JSON log line ending in `"message": "configuration valid"` and exits with status 0.

```powershell
# first run only: record the owner's download in data/raw/MANIFEST.toml (refuses to overwrite)
uv run --locked python -m niftytail validate configs\base.toml --init-manifest --retrieved YYYY-MM-DD --source-url URL
# every later run
uv run --locked python -m niftytail validate configs\base.toml
```

`validate` checks the raw files against the manifest and the D25 rules. It exits with status 0 only if there are no hard errors. Flags are reported, never applied.

Data layout (all gitignored; nothing under `data/` is ever committed):

| Path | Contents |
|---|---|
| `data/raw/` | the owner's unedited NSE CSV downloads plus `MANIFEST.toml` (names, sizes, sha256, row counts, date ranges) |
| `data/validation/` | `validation_report.json`, `flags.csv`, `spot_check_sheet.txt`, `absent_weekdays.csv`. These contain dates, counts and hashes only, with no index levels. |

```powershell
# Day-4 dependence battery (D29): verified data -> summary + figures
uv run --locked python -m niftytail eda configs\base.toml
```

`eda` loads the data through the manifest-verified loader and runs the D29 battery: on the training returns, plus a descriptive full-sample version. It writes `reports/eda/eda_summary.json`, which is tracked and holds aggregates and dates only; reruns produce identical bytes. It also writes `reports/eda/figures/*.png`, which are gitignored because they show the daily series. It logs only file names and `OK`. Findings are in [`docs/eda.md`](eda.md).

```powershell
# Day-6 N_min simulation study (D32): synthetic GPD data only, about 40 s
uv run --locked python -m niftytail nmin-study configs\base.toml
# Day-6 A3 timing pilot (D32 (6)): synthetic data only, prints timings, about 1 minute here
uv run --locked python scripts\timing_pilot.py configs\base.toml
```

`nmin-study` writes `reports/threshold_selection/nmin_study.json`, which is tracked, holds aggregates only, and has identical bytes on every rerun. It logs the file name and the adopted N_min. It exits with status 1, writing nothing, in two cases: the rule gives a result and `[threshold] n_min` is unset or different, or the rule fails and the config sets n_min without `[nmin_study] override_decision` (D33). The JSON `adopted` block records the N_min in use and its source. Results are in [`docs/nmin_study.md`](nmin_study.md).

```powershell
# Day-7B A3 threshold-procedure study (D35): synthetic data only, about 30 minutes here with 31 workers
uv run --locked python -m niftytail threshold-study configs\base.toml
uv run --locked python -m niftytail threshold-study configs\base.toml --workers 16
```

`threshold-study` runs the threshold procedure on simulated samples from four data-generating processes (S1–S4, 10 cells, 200 replications each). It loads no market data. It first runs a pilot of 2 replications per cell; if the projected time exceeds `[simstudy] max_runtime_hours`, it exits with status 1 and writes nothing. Otherwise it writes `reports/threshold_selection/simstudy/metrics.json` and five SVG figures, all tracked, all labelled "simulation evidence (DGPs S1-S4)", and all with identical bytes on every rerun and for every worker count. It logs file names and wall-clock times only. Results and the review triggers are in [`docs/threshold_study.md`](threshold_study.md).

```powershell
# Day-7C real threshold selection on the training slice (D36): about 5 minutes, serial
uv run --locked python -m niftytail threshold-select configs\base.toml
```

`threshold-select` is the only command that feeds real data to the threshold procedure. It loads the data through the manifest-verified loader, turns only the first `[design] train_size` + 1 prices into returns, and runs Steps 1–5 with B = 999. It writes `reports/threshold_selection/real/selection.json` and five SVG figures (tracked, aggregates only, identical bytes on every rerun), plus three figures derived from per-day data under `reports/threshold_selection/real/figures/` (gitignored). It logs aggregates only. If no q qualifies, it writes the evidence, exits with status 1 and chooses nothing. It never sets `q_selected`. Results are in [`docs/threshold_selection.md`](threshold_selection.md).

```powershell
# Day-8A threshold evidence on GARCH residuals of the training slice (D38): about 5 minutes, serial
uv run --locked python -m niftytail threshold-select-residuals configs\base.toml
```

`threshold-select-residuals` loads the same training slice as `threshold-select`, fits one constant-mean GARCH(1,1) by Gaussian QML (`src/niftytail/garch.py`, D38) and runs the unchanged Steps 1–5 on the standardized residuals, with B = 999 and a seed node separate from Day 7C. It writes `reports/threshold_selection/real_residuals/selection.json` (the `threshold-select` schema plus a `garch` block and `residual_diagnostics`) and five SVG figures (tracked, aggregates only, identical bytes on every rerun), plus three per-day figures under `reports/threshold_selection/real_residuals/figures/` (gitignored). No residual, volatility or loss series is written or logged. It exits with status 1 if the GARCH fit fails (nothing written) or if no q qualifies (evidence written, nothing chosen). It sets no config value. Results are in [`docs/conditional.md`](conditional.md).

```powershell
# Day-8A-2 filter specification by training BIC (D39): under a second
uv run --locked python -m niftytail filter-select configs\base.toml
# Day-8A-2 DESCRIPTIVE threshold run on the residuals of the selected filter (D39): about 5 minutes
uv run --locked python -m niftytail threshold-select-residuals configs\base.toml --selected-filter
```

`filter-select` loads the same training slice and fits four filters by Gaussian QML on the conditional sample t = 2…2000: constant or AR(1) mean, with GARCH(1,1) or GJR(1,1) variance (`src/niftytail/garch.py`, D39). The lowest BIC is the filter of all three conditional models. It writes `reports/conditional/filter_selection.json` (tracked, aggregates only, identical bytes on every rerun): NLL, k, BIC, parameters and flags of the four candidates, the winner, and diagnostics of the winner and of the constant-mean GARCH(1,1). It exits with status 1, writing nothing, if any fit fails. With `--selected-filter`, `threshold-select-residuals` runs the threshold procedure on the 1999 residuals of the winning fit and writes `reports/threshold_selection/real_residuals/selection_filter.json` and five figures under `real_residuals/selection_filter/`. That output is labelled DESCRIPTIVE ONLY: `[conditional] q_c` is fixed by design (D39) and no decision may cite the run. The Day-8A `selection.json` is not rewritten by it.

Further commands are added on their scheduled days.

