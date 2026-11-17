"""Rolling xi-hat path with profile CIs (D47; plan §5a (iv)). Descriptive only.

For every OOS target the window and pre-window lag are rebuilt as in ``rolling.run_rolling``
(D41) and the forecast functions are called again (``models.pot_gpd``, ``models.conditional``);
xi, beta, u and n_u come from their diagnostics. The excesses are recomputed, refitted (the
refit must reproduce the diagnostics exactly) and given a profile CI. Every VaR and ES is
compared, repr for repr, with the 11A forecasts.csv; any difference stops the command.

The per-target path (xi_path.csv) stays in the gitignored run dir; the tracked summary holds
aggregates only (D23).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from niftytail import backtest, models, oos, rolling
from niftytail import data as dl
from niftytail import experiment as ex
from niftytail import losses as ls
from niftytail.config import Config, config_sha256
from niftytail.garch import fit_garch, garch_filter
from niftytail.gpd import EstimationError, excesses, fit_mle, profile_ci_xi
from niftytail.threshold import MIN_EXCESSES

__all__ = [
    "CSV_NAME",
    "DOC",
    "KEYS",
    "QUANTILES",
    "REASONS",
    "RUN_11A_COMMIT",
    "TRACKED",
    "XiPathMismatch",
    "blocks",
    "check_forecasts",
    "compute",
    "key_summary",
    "publish",
    "render_doc",
    "run",
    "run_dir",
]

RUN_11A_COMMIT = "04572a36048b57fe710d87268025369f4b4bf741"  # D47 (c), (1)
KEYS = ("pot_gpd", "garch_evt")
FIELDS = ("xi", "beta", "u", "n_u", "ci_lo", "ci_hi", "ci_reason")
REASONS = ("no_xi", "no_upper_crossing", "estimate_outside_range", "estimation_error")
QUANTILES = ("0", "0.05", "0.25", "0.5", "0.75", "0.95", "1")
CSV_NAME = "xi_path.csv"
SUMMARY = "summary.json"
TRACKED = oos.OOS_DIR / "xi_path" / SUMMARY
DOC = Path("docs/xi_path.md")
TRAINING = Path("reports/threshold_selection/real/selection.json")
LABEL = (
    "Descriptive (plan §5a (iv)); overlapping windows make the path strongly autocorrelated, "
    "so shares are not independent evidence"
)
FloatArray = ls.FloatArray
Row = dict[str, Any]


class XiPathMismatch(Exception):
    """A D47 (c) consistency check failed. Messages hold target indices only, no values."""


def _dec(q: Fraction) -> str:
    return str(Decimal(q.numerator) / Decimal(q.denominator))


def _reason(exc: EstimationError) -> str:
    msg = str(exc)
    if "no upper crossing" in msg:
        return "no_upper_crossing"
    return "estimate_outside_range" if "outside" in msg else "estimation_error"


def _xi_entry(j: int, f: models.Forecast, sample: FloatArray, u_key: str, cfg: Config) -> Row:
    """xi, beta, u, n_u from the diagnostics; refit check; profile CI (D47 (3), (4))."""
    d = f.diagnostics
    nan = math.nan
    n_u = d.get("n_u", nan)
    out: Row = {"xi": d.get("xi", nan), "beta": d.get("beta", nan), "u": d.get(u_key, nan)}
    out.update(n_u=None if math.isnan(n_u) else int(n_u))
    out.update(ci_lo=nan, ci_hi=nan, ci_reason=None, lower_open=False)
    if math.isnan(out["xi"]):
        out["ci_reason"] = "no_xi"
        return out
    tol = float(cfg.threshold.xi_zero_tol)
    y = excesses(np.sort(sample), d[u_key])
    fit = fit_mle(y, min_excesses=MIN_EXCESSES, xi_zero_tol=tol, settings=cfg.gpd)
    if (fit.xi, fit.beta, int(y.size)) != (d["xi"], d["beta"], int(d["n_u"])):
        raise XiPathMismatch(f"target {j} ({f.model}): refit differs from the forecast fit")
    try:
        ci = profile_ci_xi(y, fit, xi_zero_tol=tol, settings=cfg.gpd)
    except EstimationError as exc:
        out["ci_reason"] = _reason(exc)
        return out
    out.update(ci_lo=ci.lower, ci_hi=ci.upper, lower_open=ci.lower_open)
    return out


def _target(past: FloatArray, j: int, window: int, cfg: Config) -> dict[str, Any]:
    """Both keys for target ``j``; ``past`` is ``L[:j]`` (D41 (a))."""
    x = past[-window:]
    pre = float(past[-window - 1]) if past.size > window else None
    q_star, q_c = cfg.threshold.q_selected, cfg.conditional.q_c
    assert q_star is not None
    pot = models.pot_gpd(x, cfg, q_star)
    evt = models.conditional(x, pre, cfg, q_c)["garch_evt"]
    if math.isnan(evt.diagnostics.get("xi", math.nan)):
        z = x  # no residuals: the entry has no xi and needs none
    else:
        fit = fit_garch(x, cfg.garch, cfg.conditional.filter, lag=pre)
        z = garch_filter(x, fit, pre)[2]
    return {
        "pot_gpd": _xi_entry(j, pot, x, "u", cfg),
        "garch_evt": _xi_entry(j, evt, z, "u_z", cfg),
        "forecasts": {k: (f.var, f.es) for k, f in (("pot_gpd", pot), ("garch_evt", evt))},
    }


def _chunk(task: tuple[FloatArray, list[int], int, Config]) -> list[dict[str, Any]]:
    past, js, window, cfg = task
    return [_target(past[:j], j, window, cfg) for j in js]


def compute(
    losses: FloatArray, cfg: Config, window: int, workers: int
) -> tuple[list[int], list[dict[str, Any]]]:
    """Targets ``T0 .. n-1`` and one result per target; independent of ``workers``."""
    t0 = cfg.design.train_size
    js = list(range(t0, losses.size))
    parts = [c.tolist() for c in np.array_split(np.asarray(js), min(workers, len(js)))]
    tasks = [(losses[: c[-1]].copy(), c, window, cfg) for c in parts]
    if workers == 1:
        out = [_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            out = list(pool.map(_chunk, tasks))
    return js, [r for part in out for r in part]


def check_forecasts(
    js: list[int], dates: list[str], results: list[dict[str, Any]], table: pd.DataFrame, cfg: Config
) -> int:
    """D47 (c): (j, date) index and every VaR/ES equal to the 11A table; returns cells compared."""
    if [int(j) for j in table.index.get_level_values("j")] != js or [
        str(d) for d in table.index.get_level_values("date")
    ] != dates:
        raise XiPathMismatch("the 11A table index differs from the targets")
    n = 0
    for c in models.levels(cfg):
        for key in KEYS:
            for i, kind in enumerate(("var", "es")):
                ref = table[f"{key}:{kind}:{_dec(c)}"].to_numpy(dtype=np.float64)
                for row, (j, r) in enumerate(zip(js, results, strict=True)):
                    if repr(float(r["forecasts"][key][i][c])) != repr(float(ref[row])):
                        raise XiPathMismatch(f"target {j}: {key} {kind} {_dec(c)} differs from 11A")
                    n += 1
    return n


def _csv(js: list[int], dates: list[str], results: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["j", "date", *(f"{k}:{f}" for k in KEYS for f in FIELDS)])
    for j, d, r in zip(js, dates, results, strict=True):
        cells: list[Any] = [j, d]
        for k in KEYS:
            e = r[k]
            cells += [repr(float(e[f])) for f in ("xi", "beta", "u")]
            cells.append("" if e["n_u"] is None else e["n_u"])
            cells += [repr(float(e["ci_lo"])), repr(float(e["ci_hi"])), e["ci_reason"] or ""]
        w.writerow(cells)
    return buf.getvalue().encode("utf-8")


def blocks(n: int, cfg: Config) -> list[dict[str, Any]]:
    """The D42 (7) layout over ``n`` target rows: block, start/end row, n_obs, partial."""
    out = backtest.traffic_light(np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool), cfg.backtest)
    return [
        {
            "block": b["block"],
            "start_row": b["start_row"],
            "end_row": b["end_row"],
            "n_obs": b["n_obs"],
            "partial": b["zone"] is None,
        }
        for b in out
    ]


def _q(x: FloatArray, q: str) -> float | None:
    return float(np.quantile(x, float(q), method="inverted_cdf")) if x.size else None


def _share(mask: Any, n: int) -> float | None:
    return float(np.sum(mask)) / n if n else None


def key_summary(entries: list[Row], cfg: Config) -> dict[str, Any]:
    """D47 (e), (7): counts, quantiles, shares and the 250-target blocks for one key."""
    xi = np.array([e["xi"] for e in entries], dtype=np.float64)
    lo = np.array([e["ci_lo"] for e in entries], dtype=np.float64)
    hi = np.array([e["ci_hi"] for e in entries], dtype=np.float64)
    has_xi, has_ci = ~np.isnan(xi), ~np.isnan(lo)
    n_ci = int(has_ci.sum())
    reasons = [e["ci_reason"] for e in entries]
    out: dict[str, Any] = {
        "n_targets": len(entries),
        "n_with_xi": int(has_xi.sum()),
        "n_with_ci": n_ci,
        "n_lower_open": sum(bool(e["lower_open"]) for e in entries),
        "ci_failures": {r: reasons.count(r) for r in REASONS},
        "xi_quantiles": {q: _q(xi[has_xi], q) for q in QUANTILES},
        "shares_over_targets_with_ci": {
            "ci_lo_gt_0": _share(lo[has_ci] > 0, n_ci),
            "ci_hi_lt_0": _share(hi[has_ci] < 0, n_ci),
            "ci_contains_0": _share((lo[has_ci] <= 0) & (hi[has_ci] >= 0), n_ci),
            "xi_lt_0": _share(xi[has_ci] < 0, n_ci),
        },
        "blocks": [],
    }
    for b in blocks(len(entries), cfg):
        s = slice(b["start_row"], b["end_row"] + 1)
        bx, bl = xi[s][has_xi[s]], lo[s][has_ci[s]]
        out["blocks"].append(
            {
                "block": b["block"],
                "n_obs": b["n_obs"],
                "partial": b["partial"],
                "xi_median": _q(bx, "0.5"),
                "share_ci_lo_gt_0": _share(bl > 0, int(bl.size)),
            }
        )
    return out


def run_dir(root: Path, config_sha: str, manifest_sha: str, window: int, commit: str) -> Path:
    """D47 (e): ``reports/runs/<sha256("xi-path", config, data manifest, W, commit)>``."""
    key = f"xi-path\n{config_sha}\n{manifest_sha}\n{window}\n{commit}\n".encode()
    return root / oos.RUNS_DIR / hashlib.sha256(key).hexdigest()


def _where(cfg: Config, root: Path, config_path: Path) -> tuple[str, str, str, Path]:
    commit = oos.clean_commit(root)
    cfg_sha = config_sha256(config_path)
    man_sha = ex.file_sha256(root / cfg.data.manifest)
    return commit, cfg_sha, man_sha, run_dir(root, cfg_sha, man_sha, cfg.design.window, commit)


def _training_reference(root: Path) -> dict[str, Any]:
    path = root / TRAINING
    x = json.loads(path.read_text(encoding="utf-8"))["xi_comparison"]["q_star"]
    return {
        "q": x["q"],
        "xi_hat": x["xi_hat"],
        "xi_ci": x["xi_ci"],
        "source": TRAINING.as_posix(),
        "source_sha256": ex.file_sha256(path),
    }


def run(
    cfg: Config,
    config_path: str | Path,
    workers: int,
    verify: bool,
    root: str | Path,
    log: logging.Logger,
) -> int:
    """D47 (g): exit 0 done/verified; 1 refused, mismatch or missing 11A run dir.

    Raises:
        OosError: dirty tree or no commit.
        XiPathMismatch: a D47 (c) check failed (nothing is written).
        DataValidationError: from the manifest-verified loader.
    """
    base = Path(root)
    w = cfg.design.window
    commit, cfg_sha, man_sha, rd = _where(cfg, base, base / config_path)
    if not verify and (rd / CSV_NAME).exists():
        log.error("run dir already holds xi_path.csv; refusing", extra={"run_dir": rd.name})
        return 1
    rd11 = oos.run_dir(base, cfg_sha, man_sha, w, RUN_11A_COMMIT)
    f11 = rd11 / rolling.FORECASTS_CSV
    if not f11.is_file():
        log.error("no 11A run dir for this config and data", extra={"run_11a_dir": rd11.name})
        return 1
    df = dl.load_raw(base / cfg.data.raw_dir, base / cfg.data.manifest, cfg.data.start_date)
    losses = ls.log_losses(df["close"].to_numpy())
    all_dates = [d.date().isoformat() for d in ls.return_dates(pd.DatetimeIndex(df.index))]
    start = time.perf_counter()
    js, results = compute(losses, cfg, w, workers)
    seconds = time.perf_counter() - start
    dates = [all_dates[j] for j in js]
    cells = check_forecasts(js, dates, results, rolling.read_table(f11), cfg)
    summary: dict[str, Any] = {
        "decision": "D47",
        "label": LABEL,
        "window": w,
        "provenance": {
            "commit": commit,
            "config_sha256": cfg_sha,
            "data_manifest_sha256": man_sha,
            "plan_sha256": ex.file_sha256(base / "docs" / "plan_v2.2.md"),
            "run_11a_dir": rd11.name,
            "run_11a_forecasts_sha256": ex.file_sha256(f11),
        },
        "keys": {k: key_summary([r[k] for r in results], cfg) for k in KEYS},
    }
    summary["keys"]["pot_gpd"]["training_reference"] = _training_reference(base)
    files = {CSV_NAME: _csv(js, dates, results), SUMMARY: backtest.to_json(summary).encode()}
    for k in KEYS:
        s = summary["keys"][k]
        log.info(
            "key",
            extra={
                "run_dir": rd.name,
                "key": k,
                "n_targets": s["n_targets"],
                "n_with_xi": s["n_with_xi"],
                "n_with_ci": s["n_with_ci"],
                "ci_failures": s["ci_failures"],
            },
        )
    info = {"run_dir": rd.name, "cells_compared": cells, "seconds": seconds, "workers": workers}
    if verify:
        hashes = {
            n: {
                "computed": hashlib.sha256(b).hexdigest(),
                "on_disk": hashlib.sha256((rd / n).read_bytes()).hexdigest()
                if (rd / n).is_file()
                else None,
            }
            for n, b in files.items()
        }
        ok = all(h["computed"] == h["on_disk"] for h in hashes.values())
        log.log(
            logging.INFO if ok else logging.ERROR,
            "verify: all hashes match" if ok else "verify: MISMATCH",
            extra={**info, "hashes": hashes},
        )
        return 0 if ok else 1
    rd.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (rd / n).write_bytes(b)
    hashes_out = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    log.info("OK", extra={**info, "sha256": hashes_out})
    return 0


def publish(cfg: Config, config_path: str | Path, root: str | Path) -> list[Path]:
    """D47 (g), (8): copy the current commit's summary.json and write docs/xi_path.md.

    Raises:
        OosError: dirty tree, no complete run for the current commit, or a tracked summary
            from another commit.
    """
    base = Path(root)
    commit, _, _, rd = _where(cfg, base, base / config_path)
    if not all((rd / n).is_file() for n in (CSV_NAME, SUMMARY)):
        raise oos.OosError("no complete xi-path run dir for the current commit")
    out = base / TRACKED
    if out.is_file():
        old = json.loads(out.read_text(encoding="utf-8"))["provenance"]["commit"]
        if old != commit:
            raise oos.OosError("xi_path summary already published from another commit")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((rd / SUMMARY).read_bytes())
    doc = base / DOC
    doc.write_text(render_doc(base), encoding="utf-8", newline="\n")
    return [out, doc]


def render_doc(root: str | Path) -> str:
    """docs/xi_path.md from the tracked summary (deterministic; numbers only)."""
    base = Path(root)
    path = base / TRACKED
    s = json.loads(path.read_text(encoding="utf-8"))
    p, v = s["provenance"], oos._v
    out = [
        "# NIFTYTAIL — Rolling xi-hat path (generated)",
        "",
        f"**{s['label']}.**",
        "",
        "Generated by `python -m niftytail xi-path-publish` from "
        f"`{TRACKED.as_posix()}` (D47 (f)). Do not edit by hand.",
        "",
        f"- Window: {s['window']}",
        f"- Code commit: `{p['commit']}`",
        f"- Config sha256: `{p['config_sha256']}`",
        f"- Data manifest sha256: `{p['data_manifest_sha256']}`",
        f"- Plan sha256: `{p['plan_sha256']}`",
        f"- 11A run dir: `{p['run_11a_dir']}`; its forecasts.csv sha256: "
        f"`{p['run_11a_forecasts_sha256']}`",
        f"- {TRACKED.as_posix()} sha256: `{ex.file_sha256(path)}`",
        "",
        "## Counts and CI failures",
        "",
    ]
    keys = list(s["keys"])
    out += oos._table(
        ["key", "n targets", "n with xi", "n with CI", "n lower open", *REASONS],
        [
            [
                k,
                *(
                    v(s["keys"][k][c])
                    for c in ("n_targets", "n_with_xi", "n_with_ci", "n_lower_open")
                ),
            ]
            + [v(s["keys"][k]["ci_failures"][r]) for r in REASONS]
            for k in keys
        ],
    )
    out += ["## xi quantiles (inverted_cdf, targets with xi)", ""]
    out += oos._table(
        ["key", *QUANTILES],
        [[k, *(v(s["keys"][k]["xi_quantiles"][q]) for q in QUANTILES)] for k in keys],
    )
    shares = ("ci_lo_gt_0", "ci_hi_lt_0", "ci_contains_0", "xi_lt_0")
    out += ["## Shares over targets with a CI", ""]
    out += oos._table(
        ["key", "ci_lo > 0", "ci_hi < 0", "CI contains 0", "xi < 0"],
        [[k, *(v(s["keys"][k]["shares_over_targets_with_ci"][c]) for c in shares)] for k in keys],
    )
    t = s["keys"]["pot_gpd"]["training_reference"]
    out += ["## pot_gpd training reference (q_selected, training sample)", ""]
    out += oos._table(
        ["q", "xi_hat", "95% profile CI", "source", "source sha256"],
        [
            [
                _dec(Fraction(t["q"])),
                v(t["xi_hat"]),
                oos._ci(t["xi_ci"]),
                f"`{t['source']}`",
                f"`{t['source_sha256']}`",
            ]
        ],
    )
    out += ["## 250-target blocks (D42 (7) layout)", ""]
    rows = []
    for i, b in enumerate(s["keys"][keys[0]]["blocks"]):
        cells = [v(b["block"]), v(b["n_obs"]), v(b["partial"])]
        for k in keys:
            kb = s["keys"][k]["blocks"][i]
            cells += [v(kb["xi_median"]), v(kb["share_ci_lo_gt_0"])]
        rows.append(cells)
    out += oos._table(
        ["block", "n_obs", "partial"]
        + [f"{k} {c}" for k in keys for c in ("median xi", "share ci_lo > 0")],
        rows,
    )
    return "\n".join(out).rstrip("\n") + "\n"
