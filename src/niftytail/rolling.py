"""Rolling one-day-ahead forecasts over the out-of-sample targets (D41; plan §6, methodology §3).

0-based indexing over the loss series ``L[0..n-1]``; training is ``L[0:T0]``. The forecast
for target ``j`` (``T0 <= j < n``) uses the window ``L[j-W:j]`` and, for the AR filter
specs, ``pre_lag = L[j-W-1]`` (None when ``j-W-1 < 0``: "no_pre_window_lag"). Each target
gets ONE filter fit and one GPD fit per q; nothing carries over from one target to the
next. The per-target code only ever receives ``L[:j]``, so no future value can enter it.

One pass computes the six primary models (POT-GPD at q_selected, GARCH-EVT at q_c) and
the declared sensitivities ``pot_gpd@q`` and ``garch_evt@q`` at q + ``[threshold]
sensitivity_offsets`` (0.86, 0.90, 0.91). A level with ``(1 - q) W < n_min`` is flagged
"below_n_min_window". Contiguous target chunks run in a ``ProcessPoolExecutor``; the
result does not depend on the worker count.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from niftytail import models
from niftytail.config import Config
from niftytail.models import Forecast

__all__ = [
    "FORECASTS_CSV",
    "forecast_target",
    "read_table",
    "run_dir",
    "run_rolling",
    "write_table",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
FORECASTS_CSV = "forecasts.csv"
_STOP_SHARE = Fraction(1, 100)  # D38 (4): GarchError on more than 1% of the targets
_Qs = tuple[Fraction, ...]


def _dec(q: Fraction) -> str:
    return str(Decimal(q.numerator) / Decimal(q.denominator))


def _levels(q: Fraction, cfg: Config) -> _Qs:
    """``q`` then ``q + offset`` for each ``[threshold] sensitivity_offsets`` (D36/D37/D39)."""
    return (q, *(q + off for off in cfg.threshold.sensitivity_offsets))


def _keyed(name: str, qs: _Qs) -> list[tuple[str, Fraction]]:
    """``(key, q)``: the D40 name at the primary level, ``name@q`` for the sensitivities."""
    return [(name if i == 0 else f"{name}@{_dec(q)}", q) for i, q in enumerate(qs)]


def forecast_target(
    past: FloatArray, window: int, cfg: Config, pot_qs: _Qs, evt_qs: _Qs
) -> dict[str, Forecast]:
    """Every forecast for the target that follows ``past`` (= ``L[:j]``); also live (D58 (b))."""
    x = past[-window:]
    pre = float(past[-window - 1]) if past.size > window else None
    out = {"gaussian": models.gaussian(x, cfg), "hs": models.hs(x, cfg)}
    out.update({k: models.pot_gpd(x, cfg, q) for k, q in _keyed("pot_gpd", pot_qs)})
    cond = models.conditional(x, pre, cfg, evt_qs)
    cond["garch_evt"] = cond.pop(models.evt_key(evt_qs[0]))
    out.update(cond)
    n_min = cfg.threshold.n_min
    for k, q in _keyed("pot_gpd", pot_qs) + _keyed("garch_evt", evt_qs):
        if n_min is not None and (1 - q) * window < n_min:
            f = out[k]
            out[k] = dataclasses.replace(f, flags=tuple(sorted({*f.flags, "below_n_min_window"})))
    return out


def _chunk(
    task: tuple[FloatArray, int, list[int], int, Config, _Qs, _Qs],
) -> list[dict[str, Forecast]]:
    """Forecasts for the targets ``js``; ``past`` is ``L[:max(js)]``, never beyond."""
    past, t0, js, window, cfg, pot_qs, evt_qs = task
    out = []
    for j in js:
        if not t0 <= j <= past.size:
            raise ValueError(f"target {j}: outside T0 <= j < n")
        out.append(forecast_target(past[:j], window, cfg, pot_qs, evt_qs))
    return out


def _flag_str(f: Forecast) -> str:
    return ";".join(f.flags)


def run_rolling(
    losses: npt.ArrayLike, dates: Sequence[Any], cfg: Config, window: int, workers: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One rolling pass over the targets ``T0 .. n-1`` (D41).

    Returns the table (index ``(j, date)``; sorted columns ``loss``, ``<model>:var:<c>``,
    ``<model>:es:<c>`` and ``<model>:flags``) and a summary of aggregates only: flag counts
    per model, targets with a missing VaR per model, GarchError and no_pre_window_lag
    counts, the GarchError share, ``stop_for_owner`` (share > 1%) and the runtime.

    Raises:
        ValueError: on a non-finite or non-1-D series, a date count that differs from the
            series length, a series shorter than ``T0 + 1``, ``window`` outside
            ``2 .. T0``, ``workers < 1`` or an unset q_selected.
    """
    start = time.perf_counter()
    x = np.asarray(losses, dtype=np.float64)
    n, t0 = x.size, cfg.design.train_size
    q_star = cfg.threshold.q_selected
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("losses: must be one-dimensional and finite")
    if len(dates) != n:
        raise ValueError("dates: must have one entry per loss")
    if n < t0 + 1:
        raise ValueError(f"losses: need at least T0 + 1 = {t0 + 1} values, got {n}")
    if not 2 <= window <= t0:
        raise ValueError(f"window: must lie in 2..T0 = {t0}")
    if workers < 1:
        raise ValueError("workers: must be >= 1")
    if q_star is None:
        raise ValueError("[threshold] q_selected: must be set before a rolling run")
    pot_qs, evt_qs = _levels(q_star, cfg), _levels(cfg.conditional.q_c, cfg)
    js = list(range(t0, n))
    chunks = [c.tolist() for c in np.array_split(np.asarray(js), min(workers, len(js)))]
    tasks = [(x[: c[-1]].copy(), t0, c, window, cfg, pot_qs, evt_qs) for c in chunks]
    if workers == 1:
        parts = [_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_chunk, tasks))
    fcs = [f for part in parts for f in part]

    rows: list[dict[str, Any]] = []
    for j, fc in zip(js, fcs, strict=True):
        row: dict[str, Any] = {"loss": float(x[j])}
        for key, f in fc.items():
            for c in models.levels(cfg):
                row[f"{key}:var:{_dec(c)}"] = f.var[c]
                row[f"{key}:es:{_dec(c)}"] = f.es[c]
            row[f"{key}:flags"] = _flag_str(f)
        rows.append(row)
    index = pd.MultiIndex.from_arrays([js, [str(dates[j]) for j in js]], names=["j", "date"])
    table = pd.DataFrame(rows, index=index)
    table = table[sorted(table.columns)]

    flags = {
        k: dict(sorted(Counter(fl for fc in fcs for fl in fc[k].flags).items())) for k in fcs[0]
    }
    missing = {k: sum(any(math.isnan(v) for v in fc[k].var.values()) for fc in fcs) for k in fcs[0]}
    garch_error = flags["garch_n"].get("garch_error", 0)
    summary = {
        "window": window,
        "first_target": t0,
        "n_targets": len(js),
        "flag_counts": dict(sorted(flags.items())),
        "missing_var": dict(sorted(missing.items())),
        "garch_error": garch_error,
        "no_pre_window_lag": flags["garch_n"].get("no_pre_window_lag", 0),
        "garch_error_share": garch_error / len(js),
        "stop_for_owner": Fraction(garch_error, len(js)) > _STOP_SHARE,
        "workers": workers,
        "runtime_seconds": time.perf_counter() - start,
    }
    return table, summary


def run_dir(repo_root: str | Path, config_sha256: str, manifest_sha256: str, window: int) -> Path:
    """``reports/runs/<sha256(config sha256, manifest sha256, window)>``: content, not time."""
    key = f"{config_sha256}\n{manifest_sha256}\n{window}\n".encode()
    return Path(repo_root) / "reports" / "runs" / hashlib.sha256(key).hexdigest()


def _cell(v: Any) -> str:
    return v if isinstance(v, str) else repr(float(v))


def write_table(table: pd.DataFrame, directory: str | Path) -> Path:
    """Write ``table`` as ``<directory>/forecasts.csv`` (repr floats, LF, no timestamp)."""
    path = Path(directory) / FORECASTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([*table.index.names, *table.columns])
        for idx, vals in zip(table.index, table.itertuples(index=False), strict=True):
            w.writerow([idx[0], idx[1], *map(_cell, vals)])
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a :func:`write_table` CSV back exactly (flags columns as strings)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        header, *body = list(csv.reader(fh))
    cols = header[2:]
    rows = [
        {c: (v if c.endswith(":flags") else float(v)) for c, v in zip(cols, r[2:], strict=True)}
        for r in body
    ]
    index = pd.MultiIndex.from_arrays(
        [[int(r[0]) for r in body], [r[1] for r in body]], names=header[:2]
    )
    return pd.DataFrame(rows, index=index, columns=cols)
