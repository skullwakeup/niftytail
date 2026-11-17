"""Next-day risk forecast: the backtested models applied to the newest data (D58).

Nothing is estimated or selected here beyond what a rolling target does: the same config
(q_selected, q_c, filter, W, the four levels), horizon 1, and the ONE per-target code path
:func:`niftytail.rolling.forecast_target`. Only the six primary keys are carried (no ``@q``
sensitivity keys); they are computed exactly as in :func:`niftytail.rolling.run_rolling`.

Live data (``data/live/``, gitignored, owner-downloaded; D23) only continues ``data/raw/``,
which stays untouched so every published run remains reproducible. ``validate-live`` applies
the D25 hard-error rules to the live files plus continuity with the raw snapshot and writes
``data/live/MANIFEST.toml``; flags are counted, never applied. ``forecast`` writes
``reports/live/<key>/forecast.json`` (gitignored); its as-of date is the only date in it.
Beside it ``series.json`` (local only, dates allowed; D59 (c)) holds the loaded sessions and
the per-day values the dashboard draws, and ``--check-run`` writes ``check.json`` (D59 (d)).
Rows fetched from EODHD (``fetch-api``, D60) live in ``data/api/`` (gitignored) and are loaded
after data/live: each later source adds only dates after the earlier ones end, and an overlap
must have the same session dates and closes to 2 decimals.
Logs and the terminal never receive a date or a per-day value.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import re
import tomllib
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from niftytail import backtest, oos, rolling
from niftytail import data as dl
from niftytail import experiment as ex
from niftytail import losses as ls
from niftytail.config import Config, config_sha256
from niftytail.models import MODELS, Forecast

__all__ = [
    "API_DIR",
    "API_MANIFEST",
    "CHECK_JSON",
    "CHECK_N",
    "FORECAST_JSON",
    "LIVE_DIR",
    "LIVE_MANIFEST",
    "LIVE_REPORTS",
    "SERIES_JSON",
    "build",
    "check_run",
    "forecast_dir",
    "load_series",
    "load_series_full",
    "parse_api_csv",
    "row_errors",
    "run",
    "validate_command",
]

LIVE_DIR = Path("data/live")
LIVE_MANIFEST = LIVE_DIR / "MANIFEST.toml"
LIVE_REPORTS = Path("reports/live")
API_DIR = Path("data/api")  # D60: EODHD rows, gitignored (covered by /data/)
API_MANIFEST = API_DIR / "MANIFEST.toml"
API_HEADER = "date,open,high,low,close"
API_SOURCE = "EODHD NSEI.INDX"
API_STALE = "data/api manifest does not match its files: run fetch-api"
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
FORECAST_JSON = "forecast.json"
SERIES_JSON = "series.json"
CHECK_JSON = "check.json"
CHECK_N = 20
WORST_N = 5
SCHEMA_VERSION = 2
TRAILING_WEEKDAY = (
    "a data/live file ends on a weekday after its last row: "
    "set the download's To date to the last trading day"
)
CONDITIONAL = ("garch_n", "fhs", "garch_evt")
GPD = ("pot_gpd", "garch_evt")
FloatArray = np.ndarray[Any, np.dtype[np.float64]]


# ---------------------------------------------------------------------------
# Live data: D25 hard errors plus continuity with data/raw
# ---------------------------------------------------------------------------
def _live_check(cfg: Config, base: Path) -> tuple[list[str], list[Any], list[Any]]:
    """``(errors, raw parsed, live parsed)``; data/raw is manifest-verified first (raises)."""
    raw = base / cfg.data.raw_dir
    dl.load_raw(raw, base / cfg.data.manifest, start_date=cfg.data.start_date)
    raw_parsed = [dl._parse_file(p) for p in dl._csv_files(raw)]
    live_parsed = [dl._parse_file(p) for p in dl._csv_files(base / LIVE_DIR)]
    errors = [e for p in live_parsed for e in p.errors]
    if not live_parsed:
        errors.append("no CSV files found in data/live")
    errors += dl._structure_errors(raw_parsed + live_parsed)
    starts = [p.window[0] for p in live_parsed if p.window is not None]
    raw_end = max(p.window[1] for p in raw_parsed if p.window is not None)
    if starts and min(starts) != raw_end + timedelta(days=1):
        errors.append("data/live does not start the day after the last data/raw window")
    if any(_trailing_weekday(p) for p in live_parsed):
        errors.append(TRAILING_WEEKDAY)
    return errors, raw_parsed, live_parsed


def _trailing_weekday(p: Any) -> bool:
    """A weekday in (last row date, window end]: a session whose close may not be published yet
    (D59 (a)). A trailing weekend passes; a trailing NSE holiday is rejected on purpose."""
    if p.window is None or not p.dates:
        return False
    d, end = max(p.dates), p.window[1]
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            return True
    return False


def row_errors(rows: list[tuple[date, float, float, float, float]]) -> list[str]:
    """The D25 row rules for (date, open, high, low, close): finite positive prices,
    ascending unique dates. Messages carry no date or value."""
    errors = []
    if any(not (math.isfinite(v) and v > 0) for r in rows for v in r[1:]):
        errors.append("a price is not finite and positive")
    if any(b[0] <= a[0] for a, b in itertools.pairwise(rows)):
        errors.append("dates are not ascending and unique")
    return errors


def parse_api_csv(blob: bytes) -> list[tuple[date, float, float, float, float]]:
    """A data/api file (fixed header, ISO dates, repr floats).

    Raises:
        DataValidationError: a wrong header, a malformed row or a D25 row-rule error.
    """
    lines = blob.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0] != API_HEADER:
        raise dl.DataValidationError(["data/api: a file has a wrong header"])
    rows = []
    for line in lines[1:]:
        cells = line.split(",")
        try:
            if len(cells) != 5 or not _ISO.fullmatch(cells[0]):
                raise ValueError
            rows.append((date.fromisoformat(cells[0]), *(float(c) for c in cells[1:])))
        except ValueError:
            raise dl.DataValidationError(["data/api: a file has a malformed row"]) from None
    errors = row_errors(rows)  # type: ignore[arg-type]
    if not rows or errors:
        raise dl.DataValidationError([f"data/api: {e}" for e in errors or ["a file is empty"]])
    return rows  # type: ignore[return-value]


def _read_api(base: Path) -> tuple[list[tuple[date, float, float, float, float]], str]:
    """Every data/api row, after the manifest check, and the manifest sha256."""
    try:
        listed = {
            f["name"]: f
            for f in tomllib.loads((base / API_MANIFEST).read_text(encoding="utf-8"))["file"]
        }
    except (OSError, ValueError, KeyError, TypeError):
        raise dl.DataValidationError([API_STALE]) from None
    names = sorted(p.name for p in (base / API_DIR).glob("*.csv"))
    if names != sorted(listed):
        raise dl.DataValidationError([API_STALE])
    rows = []
    for name in names:
        blob = (base / API_DIR / name).read_bytes()
        part = parse_api_csv(blob)
        f = listed[name]
        meta = (hashlib.sha256(blob).hexdigest(), len(part), part[0][0], part[-1][0])
        if meta != (
            f.get("sha256"),
            f.get("rows"),
            *map(_iso_date, (f.get("first"), f.get("last"))),
        ):
            raise dl.DataValidationError([API_STALE])
        rows += part
    rows.sort()
    errors = row_errors(rows)
    if errors:
        raise dl.DataValidationError([f"data/api: {e}" for e in errors])
    return rows, ex.file_sha256(base / API_MANIFEST)


def _iso_date(v: Any) -> date | None:
    return date.fromisoformat(v) if isinstance(v, str) and _ISO.fullmatch(v) else None


def _merge_api(
    df: pd.DataFrame, sources: list[str], rows: list[tuple[date, float, float, float, float]]
) -> tuple[pd.DataFrame, list[str]]:
    """Append the api rows after the earlier data; an overlap must agree (D60 (e))."""
    end = df.index[-1].date()
    over = {r[0]: r[4] for r in rows if r[0] <= end}
    if over:
        lo = min(over)
        earlier = {
            d.date(): float(c)
            for d, c in zip(df.index, df["close"].to_numpy(), strict=True)
            if lo <= d.date() <= end
        }
        if set(over) != set(earlier):
            raise dl.DataValidationError(["data/api overlaps the earlier data with other dates"])
        if any(round(over[d], 2) != round(earlier[d], 2) for d in over):
            raise dl.DataValidationError(["data/api overlap: a close differs from earlier data"])
    new = [r for r in rows if r[0] > end]
    if not new:
        return df, sources
    add = pd.DataFrame(
        [r[1:] for r in new],
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex([pd.Timestamp(r[0]) for r in new], name="date"),
    )
    return pd.concat([df, add]), [*sources, *(["api"] * len(new))]


def load_series_full(
    cfg: Config, root: str | Path
) -> tuple[pd.DataFrame, list[str], str | None, str | None]:
    """data/raw, then data/live (D58), then data/api (D60) as one ascending frame.

    Returns the frame, each session's source ("raw" | "live" | "api"), and the live and api
    manifest sha256 (None without the directory).

    Raises:
        DataValidationError: any hard error; a live or api manifest that does not match its
            files; an api overlap with other dates or another close (2 decimals).
    """
    base = Path(root)
    df, live_sha = _raw_live(cfg, base)
    raw_last = max(f.last_date for f in dl.read_manifest(base / cfg.data.manifest).files)
    sources = ["live" if d.date() > raw_last else "raw" for d in df.index]
    if not (base / API_DIR).is_dir():
        return df, sources, live_sha, None
    rows, api_sha = _read_api(base)
    df, sources = _merge_api(df, sources, rows)
    return df, sources, live_sha, api_sha


def load_series(cfg: Config, root: str | Path) -> tuple[pd.DataFrame, str | None]:
    """:func:`load_series_full` without the sources and the api manifest sha256."""
    df, _, live_sha, _ = load_series_full(cfg, root)
    return df, live_sha


def _raw_live(cfg: Config, base: Path) -> tuple[pd.DataFrame, str | None]:
    """data/raw (its manifest) plus data/live (its manifest); the live manifest sha256 or
    None without a data/live directory.

    Raises:
        DataValidationError: on any hard error in either directory, or a live manifest
            that does not match the live files (run ``validate-live``).
    """
    if not (base / LIVE_DIR).is_dir():
        df = dl.load_raw(
            base / cfg.data.raw_dir, base / cfg.data.manifest, start_date=cfg.data.start_date
        )
        return df, None
    errors, raw_parsed, live_parsed = _live_check(cfg, base)
    man = base / LIVE_MANIFEST
    errors += dl._manifest_errors(man, base / LIVE_DIR)
    infos = [dl._info(p) for p in live_parsed if p.window is not None]
    errors += dl._manifest_content_errors(man, infos)
    if errors:
        raise dl.DataValidationError(errors)
    return dl._frame(dl._collect(raw_parsed + live_parsed)), ex.file_sha256(man)


def validate_command(cfg: Config, root: str | Path, retrieved: date, log: logging.Logger) -> int:
    """``validate-live`` (D58 (c)). Exit 0 iff no hard error; the manifest is (re)written only
    when the live files changed. Logs counts only."""
    base = Path(root)
    errors, raw_parsed, live_parsed = _live_check(cfg, base)
    if errors:
        for e in errors:
            log.error("data error", extra={"error": e})
        log.error("validate-live failed: nothing written", extra={"n_errors": len(errors)})
        return 1
    infos = [dl._info(p) for p in live_parsed]
    man = base / LIVE_MANIFEST
    now = sorted((f.name, f.bytes, f.sha256, f.rows, f.first_date, f.last_date) for f in infos)
    try:
        listed = dl.read_manifest(man).files
        same = now == sorted(
            (e.name, e.bytes, e.sha256, e.rows, e.first_date, e.last_date) for e in listed
        )
    except dl.DataValidationError:
        same = False
    if not same:
        man.unlink(missing_ok=True)
        source = dl.read_manifest(base / cfg.data.manifest).source_url
        dl.write_manifest(man, infos, source_url=source, retrieved=retrieved)
    rows = dl._collect(raw_parsed + live_parsed)
    v = cfg.validation
    flags, _ = dl._flags(
        rows,
        large_return_flag=v.large_return_flag,
        gap_flag_days=v.gap_flag_days,
        stale_run_min=v.stale_run_min,
    )
    raw_last = max(d for p in raw_parsed for d in p.dates)
    live_flags = [f.category for f in flags if f.date > raw_last]
    log.info(
        "validate-live passed",
        extra={
            "n_files": len(infos),
            "n_rows": sum(f.rows for f in infos),
            "manifest": "unchanged" if same else "written",
            "flag_counts": {c: live_flags.count(c) for c in dl.FLAG_CATEGORIES},
        },
    )
    return 0


# ---------------------------------------------------------------------------
# Forecasts through the one rolling code path
# ---------------------------------------------------------------------------
def _primary_qs(cfg: Config) -> tuple[tuple[Fraction], tuple[Fraction]]:
    q = cfg.threshold.q_selected
    if q is None:
        raise ValueError("[threshold] q_selected: must be set")
    return (q,), (cfg.conditional.q_c,)


def _chunk(task: tuple[FloatArray, list[int], int, Config]) -> list[dict[str, Forecast]]:
    past, js, window, cfg = task
    pot, evt = _primary_qs(cfg)
    return [rolling.forecast_target(past[:j], window, cfg, pot, evt) for j in js]


def _forecasts(
    x: FloatArray, js: list[int], window: int, cfg: Config, workers: int
) -> list[dict[str, Forecast]]:
    """The primary forecasts for the targets ``js`` (each sees only ``x[:j]``)."""
    chunks = [c.tolist() for c in np.array_split(np.asarray(js), min(workers, len(js)))]
    tasks = [(x[: c[-1]].copy(), c, window, cfg) for c in chunks]
    if workers == 1:
        parts = [_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_chunk, tasks))
    return [f for part in parts for f in part]


def _num(v: float) -> float | str | None:
    """JSON value: finite float, "inf" for +inf (unbounded ES), None for NaN (missing)."""
    if math.isnan(v):
        return None
    return "inf" if math.isinf(v) else v


TAIL_KEYS = {"pot_gpd": ("u", "n_u"), "garch_evt": ("u_z", "n_u")}  # D59 (b)


def _model(key: str, f: Forecast) -> dict[str, Any]:
    out: dict[str, Any] = {
        "var": {rolling._dec(c): _num(f.var[c]) for c in f.var},
        "es": {rolling._dec(c): _num(f.es[c]) for c in f.es},
        "flags": list(f.flags),
    }
    names = (("mu_next", "sigma_next") if key in CONDITIONAL else ()) + (
        ("xi", "beta") if key in GPD else ()
    )
    for n in names:
        out[n] = _num(f.diagnostics.get(n, math.nan))
    if key in TAIL_KEYS:  # copied from the diagnostics, never recomputed
        out["tail"] = {n: _num(f.diagnostics.get(n, math.nan)) for n in TAIL_KEYS[key]}
    return out


def _model_hits(
    loss: FloatArray, fcs: list[dict[str, Forecast]], key: str, c: Fraction
) -> tuple[FloatArray, Any, Any]:
    """``(VaR, admissible hits, admissible)`` of one model at level ``c`` over the targets."""
    var = np.array([fc[key].var[c] for fc in fcs])
    es = np.array([fc[key].es[c] for fc in fcs])
    flags = [";".join(fc[key].flags) for fc in fcs]
    ok = np.array([r == "" for r in backtest.admissibility(var, es, flags)])
    return var, backtest.hits(loss, var) * ok, ok


def _health(
    x: FloatArray, fcs: list[dict[str, Forecast]], js: list[int], cfg: Config
) -> dict[str, Any]:
    """Breaches at the traffic-light level over the last block of targets, and the zone."""
    c = cfg.backtest.traffic_light_level
    out = {}
    for k in MODELS:
        _, h, ok = _model_hits(x[js], fcs, k, c)
        (block,) = backtest.traffic_light(h, ok, cfg.backtest)
        out[k] = {f: block[f] for f in ("breaches", "inadmissible_rows", "n_obs", "zone")}
    return {"level": rolling._dec(c), "targets": len(js), "models": out}


def _targets(
    x: FloatArray, cfg: Config, workers: int
) -> tuple[list[int], list[dict[str, Forecast]]]:
    """The health targets ``n - block .. n - 1`` then the next day ``n``, and their forecasts.

    Raises:
        ValueError: if fewer than ``W + traffic_light_block + 1`` losses are available.
    """
    n, w, block = x.size, cfg.design.window, cfg.backtest.traffic_light_block
    if n < w + block + 1:
        raise ValueError(f"losses: need at least W + block + 1 = {w + block + 1} values")
    js = [*range(n - block, n), n]
    return js, _forecasts(x, js, w, cfg, workers)


def build(
    losses: FloatArray, as_of: str, cfg: Config, workers: int, provenance: dict[str, Any]
) -> dict[str, Any]:
    """The forecast document for the day after ``as_of`` (the last loss date).

    Raises:
        ValueError: if fewer than ``W + traffic_light_block + 1`` losses are available.
    """
    x = np.asarray(losses, dtype=np.float64)
    return _document(x, as_of, cfg, *_targets(x, cfg, workers), provenance)


def _document(
    x: FloatArray,
    as_of: str,
    cfg: Config,
    js: list[int],
    fcs: list[dict[str, Forecast]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    w = cfg.design.window
    nxt = fcs[-1]
    sigma = nxt["garch_evt"].diagnostics.get("sigma_next", math.nan)
    std = nxt["gaussian"].diagnostics["sigma"]
    pot, evt = _primary_qs(cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of,
        "horizon": 1,
        "config": {
            "window": w,
            "q_selected": rolling._dec(pot[0]),
            "q_c": rolling._dec(evt[0]),
            "filter": cfg.conditional.filter,
            "levels": [rolling._dec(c) for c in cfg.levels.primary + cfg.levels.descriptive],
            "primary_levels": [rolling._dec(c) for c in cfg.levels.primary],
            "xi_zero_tol": float(cfg.threshold.xi_zero_tol),
        },
        "models": {k: _model(k, nxt[k]) for k in MODELS},
        "volatility": {
            "sigma_next": _num(sigma),
            "window_std": _num(std),
            "sigma_ratio": _num(sigma / std),
        },
        "health": _health(x, fcs[:-1], js[:-1], cfg),
        "provenance": provenance,
    }


def series_document(
    frame: pd.DataFrame,
    sources: list[str],
    x: FloatArray,
    dates: list[str],
    cfg: Config,
    js: list[int],
    fcs: list[dict[str, Forecast]],
) -> dict[str, Any]:
    """``series.json`` (D59 (c)): local only, dates allowed; from the data and forecasts that
    ``forecast`` already holds (no further forecast)."""
    w, c = cfg.design.window, cfg.backtest.traffic_light_level
    cols = {k: frame[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close")}
    history = [
        {"date": d.date().isoformat(), "source": s}
        | {k: _num(float(v[i])) for k, v in cols.items()}
        for i, (d, s) in enumerate(zip(frame.index, sources, strict=True))
    ]
    hjs, hfcs = js[:-1], fcs[:-1]
    per = {k: _model_hits(x[hjs], hfcs, k, c) for k in MODELS}
    recent = [
        {"date": dates[j], "loss": float(x[j])}
        | {k: {"var": _num(float(per[k][0][i])), "hit": int(per[k][1][i])} for k in MODELS}
        for i, j in enumerate(hjs)
    ]
    window = x[-w:]
    u = fcs[-1]["pot_gpd"].diagnostics["u"]
    order = sorted(range(x.size), key=lambda i: (-x[i], dates[i]))[:WORST_N]
    return {
        "schema_version": SCHEMA_VERSION,
        "level": rolling._dec(c),
        "history": history,
        "window": {"first": dates[-w], "last": dates[-1]},
        "recent": recent,
        "tail_sample": sorted((float(v) for v in window[window > u]), reverse=True),
        "worst": [{"date": dates[i], "loss": float(x[i])} for i in order],
    }


def forecast_dir(
    root: Path,
    cfg_sha: str,
    raw_sha: str,
    live_sha: str | None,
    commit: str,
    api_sha: str | None = None,
) -> Path:
    """``reports/live/<sha256(config, raw, live and api manifests, commit)>`` (D60 (e))."""
    key = f"{cfg_sha}\n{raw_sha}\n{live_sha or '-'}\n{api_sha or '-'}\n{commit}\n".encode()
    return root / LIVE_REPORTS / hashlib.sha256(key).hexdigest()


def _loaded(
    cfg: Config, base: Path
) -> tuple[pd.DataFrame, list[str], FloatArray, list[str], str | None, str | None]:
    """The frame, each session's source, the losses, their dates, the live and api sha."""
    df, sources, live_sha, api_sha = load_series_full(cfg, base)
    x = ls.log_losses(df["close"].to_numpy())
    dates = [d.date().isoformat() for d in ls.return_dates(pd.DatetimeIndex(df.index))]
    return df, sources, np.asarray(x, dtype=np.float64), dates, live_sha, api_sha


def _series(cfg: Config, base: Path) -> tuple[FloatArray, list[str], str | None]:
    _, _, x, dates, live_sha, _ = _loaded(cfg, base)
    return x, dates, live_sha


def run(
    cfg: Config,
    config_path: str | Path,
    workers: int,
    verify: bool,
    root: str | Path,
    log: logging.Logger,
) -> int:
    """``forecast`` (D58 (d), D59 (b)): writes forecast.json and series.json; with
    ``verify``, or when forecast.json already exists, compares both and writes nothing
    (exit 0 "verify: bytes match" / "unchanged", 1 on a mismatch).

    Raises:
        OosError: dirty tree or no commit.
        DataValidationError: from the loaders.
    """
    base = Path(root)
    commit = oos.clean_commit(base)
    cfg_sha = config_sha256(base / config_path)
    raw_sha = ex.file_sha256(base / cfg.data.manifest)
    frame, sources, x, dates, live_sha, api_sha = _loaded(cfg, base)
    out = forecast_dir(base, cfg_sha, raw_sha, live_sha, commit, api_sha)
    path = out / FORECAST_JSON
    existing = not verify and path.exists()
    prov = {
        "commit": commit,
        "config_sha256": cfg_sha,
        "raw_manifest_sha256": raw_sha,
        "live_manifest_sha256": live_sha,
        "api_manifest_sha256": api_sha,
        "n_losses": int(x.size),
    }
    js, fcs = _targets(x, cfg, workers)
    blobs = {
        path: backtest.to_json(_document(x, dates[-1], cfg, js, fcs, prov)).encode(),
        out / SERIES_JSON: backtest.to_json(
            series_document(frame, sources, x, dates, cfg, js, fcs)
        ).encode(),
    }
    if verify or existing:
        ok = all(p.is_file() and p.read_bytes() == b for p, b in blobs.items())
        word = "unchanged" if existing else "verify: bytes match"
        log.log(
            logging.INFO if ok else logging.ERROR,
            word if ok else "verify: MISMATCH",
            extra={"key": out.name},
        )
        return 0 if ok else 1
    out.mkdir(parents=True, exist_ok=True)
    for p, b in blobs.items():
        p.write_bytes(b)
    log.info("OK", extra={"key": out.name, "sha256": hashlib.sha256(blobs[path]).hexdigest()})
    return 0


def check_run(
    cfg: Config, config_path: str | Path, run_dir: str | Path, workers: int, root: str | Path
) -> int:
    """``forecast --check-run`` (D58 (d)): recompute CHECK_N targets of a run; print only
    "k/N identical". Exit 0 iff all are identical (loss, VaR, ES and flags of the primary keys).
    Also writes ``check.json`` {run, identical, n, commit} beside the forecast (D59 (d)).

    Raises:
        OosError: a dirty tree, a run made with another config, or no forecasts.csv.
    """
    base = Path(root)
    commit = oos.clean_commit(base)
    rd = base / run_dir
    man = json.loads((rd / oos.MANIFEST).read_text(encoding="utf-8"))
    if man["config"]["sha256"] != config_sha256(base / config_path):
        raise oos.OosError("the run was made with another config")
    window = int(json.loads((rd / oos.SUMMARY).read_text(encoding="utf-8"))["window"])
    table = rolling.read_table(rd / rolling.FORECASTS_CSV)
    span = [int(j) for j in table.index.get_level_values("j")]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(cfg.runtime.seed)))
    n = min(CHECK_N, len(span))
    js = sorted(span[int(i)] for i in rng.choice(len(span), size=n, replace=False))
    _, _, x, _, live_sha, api_sha = _loaded(cfg, base)
    fcs = _forecasts(x, js, window, cfg, workers)
    rows = table.droplevel("date")
    same = 0
    for j, fc in zip(js, fcs, strict=True):
        row = rows.loc[j]
        cells = [repr(float(x[j])) == repr(float(row["loss"]))]
        for k in MODELS:
            cells.append(";".join(fc[k].flags) == row[f"{k}:flags"])
            for c in fc[k].var:
                lvl = rolling._dec(c)
                cells.append(repr(fc[k].var[c]) == repr(float(row[f"{k}:var:{lvl}"])))
                cells.append(repr(fc[k].es[c]) == repr(float(row[f"{k}:es:{lvl}"])))
        same += all(cells)
    print(f"{same}/{n} identical")
    raw_sha = ex.file_sha256(base / cfg.data.manifest)
    out = forecast_dir(base, man["config"]["sha256"], raw_sha, live_sha, commit, api_sha)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"run": rd.name, "identical": same, "n": n, "commit": commit}
    (out / CHECK_JSON).write_bytes(backtest.to_json(doc).encode())
    return 0 if same == n else 1
