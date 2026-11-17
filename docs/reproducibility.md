# NIFTYTAIL: reproducibility record (Day 14-2, D61 (e), (f))

**Plan §7 row 14 and A6 row 14: a clean rerun gives byte-identical `forecasts.csv` on this machine; the cross-machine tolerance is documented below.** Every value here is copied from the command output (exit codes, wall-clock seconds, sha256). No per-day value or date appears.

Recorded on 2026-11-16 from commit `c1d506b` ("Day 14-1") on a clean tree.

## Machine

| item | value |
|---|---|
| OS | Microsoft Windows 11 Pro, version 10.0.26300 |
| CPU | AMD Ryzen 9 9950X3D 16-Core Processor, 32 logical processors |
| Memory | 61.6 GB |
| Python | 3.13.17 (system interpreter; `python-preference = "only-system"`) |
| numpy / scipy | 2.5.4 / 1.18.1 (from `uv.lock`) |
| BLAS | scipy-openblas 0.3.34.106.0; `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1, set by `niftytail.__main__.pin_threads` before numpy is imported (A5e) |
| uv / git | 0.12.15 / 2.55.0.windows.5 |

## How each artefact was verified

Every published artefact embeds or is keyed by the commit that generated it: the run directories `reports/runs/<sha256(config, manifest, W, commit)>` (D44 (b), D48 (d)), the commit in each SVG and in `docs/figures.md` (D51), the D53 run directory and the forecast key (D58, D60). A `--verify` at a later commit therefore looks for different bytes by construction. Each published artefact was verified at its recorded generating commit instead:

- a detached `git worktree` of that commit, outside the repository;
- the gitignored `data/` and `reports/runs/` linked in read-only (directory junctions), nothing written;
- the commit's own code (`PYTHONPATH=<worktree>/src`) run in the project `.venv`. The default and dev dependency groups are identical at all five commits; `uv.lock` differs only in the dashboard group's pyarrow pin (D56), which the project `.venv` does not contain.

`figures` and `tl-blocks` were generated at their commit and committed in the next one, so their outputs are not in that commit's checkout: the first `--verify` found the previous SVGs on disk (figures) or no file (tl-blocks) and exited 1. Their recomputed sha256 already equalled the committed artefacts (docs/protected_hashes.md hashes). The original protocol (status.md Day 12B-2 and 13-0b) verifies with the fresh outputs in the working tree, which `--verify` allows; with the committed outputs copied into the worktree, both exited 0 (the second rows below). The local forecast was verified at `c1d506b` itself after `forecast` had written it.

## Commands and results

| command | commit | exit | wall clock (s) | result |
|---|---|---|---|---|
| `forecast configs/base.toml` | `c1d506b` | 0 | 6 | OK; forecast.json sha256 `7255abe4226ebf43a94a575d65fe879c0e8ca95da50cf9f2e203c68096b325d2` |
| `forecast configs/base.toml --verify` | `c1d506b` | 0 | 7 | "verify: bytes match" (forecast.json and series.json) |
| `forecast configs/base.toml --check-run reports/runs/e65780dc…0b28c9` | `c1d506b` | 0 | 3 | "20/20 identical" |
| `figures configs/base.toml --verify` (first attempt) | `435fdca` | 1 | 3 | MISMATCH against the pre-commit SVGs on disk; computed = the 9 protected hashes (docs/protected_hashes.md) |
| `figures configs/base.toml --verify` | `435fdca` | 0 | 5 | "verify: all hashes match" (8 SVGs and docs/figures.md) |
| `tl-blocks configs/base.toml --verify` (first attempt) | `65697f4` | 1 | 1 | no file on disk; computed `5d736d95d0121937c107873d3d05f3d57953df5f1239b2b37edd964e670af1f0` (= docs/protected_hashes.md) |
| `tl-blocks configs/base.toml --verify` | `65697f4` | 0 | 1 | "verify: hash matches" |
| `figures-local configs/base.toml --verify` | `ab5c135` | 0 | 2 | "verify: all hashes match" (G1-G4 and index.md, the status.md Day 12C-2 values) |
| `run configs/base.toml --window 1000 --verify --workers 31` | `04572a3` | 0 | 89 | "verify: all hashes match": forecasts.csv, summary.json, backtest.json, backtest_per_date.json; 5716 targets |
| `run configs/base.toml --window 2000 --verify --workers 31` | `04572a3` | 0 | 94 | "verify: all hashes match" (the same four files); 5716 targets |
| `tri-run configs/base.toml --verify --workers 31` | `4c68343` | 0 | 85 | "verify: all hashes match" (the same four files); 5716 targets |

Byte-identical `forecasts.csv` (the plan's acceptance), sha256 recomputed and on disk:

| pass | run dir | forecasts.csv sha256 |
|---|---|---|
| PR W1000 | `e65780dc4f81b1cad5a3bdd88b0fd18bb69f9ba3585655814af338174c0b28c9` | `714523a4fb354cd06cdb1764ed2fc79978ec8e14b1a4f52899dc433423c54cd9` |
| PR W2000 | `dadebddafa91…` (status.md Day 11A) | `5ad7a9fae98c8fd906b2e3cbdd66a0214f85e7f4c75a4a15583aa2de91a44755` |
| TRI W1000 | `316df0ba02cc…` (status.md Day 11C) | `f34c95d9d54f4707ecde5f37ed3c1f9d8f17b8daf7967c46b673ca65d663890d` |

The PR W1000 value equals the one recorded in `docs/xi_path.md` (provenance of the D47 run).

## Protected hashes

All 33 sha256 values in docs/protected_hashes.md (the frozen plan, the threshold, filter and simulation evidence, the 16 published artefacts of D50 and the 10 figure files of D51) were recomputed at `c1d506b`: 33 of 33 match.

## Across machines

Byte identity is established on Windows (the development platform) only (D66). On another operating system, floating-point differences are possible (BLAS, libm and CPU instruction paths differ), and a single last-bit difference changes a sha256. To compare a rerun made elsewhere, compare `forecasts.csv` (for the demo, the two files under `app/demo/`):

- VaR, ES, mu and sigma: relative tolerance 1e-5 (D66; it was 1e-9, which the Linux run below disproved);
- counts, flags columns, breach counts and every test verdict (Kupiec, Christoffersen IND and CC, ES residual, traffic-light zones, Diebold-Mariano rejections): exact equality.

Off Windows, `demo-data --verify` and the demo test of `tests/test_repo_hygiene.py` apply this rule: bytes first; if they differ, the parsed JSON must have the same structure and the same non-float values, and every float must agree within relative 1e-5 (absolute floor 1e-12). The command then prints "verify: values match within relative 1e-5 (not byte-identical; docs/reproducibility.md)" and exits 0; otherwise it prints MISMATCH with the maximum relative difference and exits 1.

Second machine, as reported by the reviewer: Linux, Python 3.13.18, the same `uv.lock`, a fresh clone of the public repository at `644621b`, before D66. One test failed, `tests/test_repo_hygiene.py::test_demo_dir_holds_exactly_the_generated_files`, and `demo-data configs/base.toml --verify` printed MISMATCH for `forecast.json` and `series.json`. Regenerated against tracked: `forecast.json`, 46 values differ, maximum relative difference 1.24e-6 (GARCH-based values, e.g. `models.fhs.es.0.95`); `series.json`, 836 values differ, maximum 1.34e-7. All other 1,060 tests, mypy, ruff, the 23 app tests and `reference_values.py` passed. Both differences are within the D66 tolerance.
