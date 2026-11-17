"""Public demo on synthetic data (D62 (b)): the ``demo-data`` command.

A synthetic daily index, never the Nifty 50: AR(1)-GJR(1,1) returns with standardized
Student-t innovations, 4,500 sessions on synthetic business-day dates and deterministic OHLC
around the close, seeded from ``SeedSequence([runtime] seed)`` (child 62). It goes through
the same code path as ``forecast`` (``live._targets`` -> ``rolling.forecast_target``,
``live._document`` with the health block, ``live.series_document``) and is written to the
tracked ``app/demo/forecast.json`` and ``app/demo/series.json``, which the public dashboard
shows as "Today (demo)". ``--verify`` regenerates both in memory and compares bytes.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from niftytail import backtest, live
from niftytail import losses as ls
from niftytail.config import Config, config_sha256

__all__ = ["DEMO_DIR", "FILES", "SOURCE", "documents", "run", "synthetic_index"]

DEMO_DIR = Path("app/demo")
FILES = (live.FORECAST_JSON, live.SERIES_JSON)
SOURCE = "demo"  # shown as "synthetic demo"
SESSIONS = 4500
FIRST_DATE = date(2000, 1, 3)  # a Monday; every later session is the next weekday
SEED_CHILD = 62  # SeedSequence(seed).spawn(63)[62], as D62
# AR(1)-GJR(1,1) on returns: r = mu + phi r_{t-1} + e,
# s2 = omega + (alpha + gamma 1{e < 0}) e2 + beta s2 (falls raise the variance)
MU, PHI, OMEGA, ALPHA, GAMMA, BETA, DF = 0.0004, 0.03, 2e-6, 0.03, 0.10, 0.90, 5
START_LEVEL = 10_000.0


def _dates(n: int) -> list[date]:
    out: list[date] = []
    d = FIRST_DATE
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def synthetic_index(cfg: Config, n: int = SESSIONS) -> pd.DataFrame:
    """Daily OHLC of the synthetic index (2 decimals), DatetimeIndex named ``date``."""
    seq = np.random.SeedSequence(cfg.runtime.seed).spawn(SEED_CHILD + 1)[SEED_CHILD]
    rng = np.random.Generator(np.random.PCG64(seq))
    z = rng.standard_t(DF, n) / math.sqrt(DF / (DF - 2))  # unit variance
    u = rng.uniform(0.0, 1.0, (n, 2))
    r = np.empty(n)
    s2 = OMEGA / (1 - ALPHA - GAMMA / 2 - BETA)
    prev_r, prev_e = 0.0, 0.0
    for t in range(n):
        s2 = OMEGA + (ALPHA + GAMMA * (prev_e < 0)) * prev_e**2 + BETA * s2 if t else s2
        e = math.sqrt(s2) * z[t]
        r[t] = MU + PHI * prev_r + e
        prev_r, prev_e = r[t], e
    close = START_LEVEL * np.exp(np.cumsum(r))
    prev = np.concatenate([[START_LEVEL], close[:-1]])
    open_ = prev * np.exp(0.3 * r)
    spread = np.abs(r) + 0.002
    high = np.maximum(open_, close) * np.exp(spread * u[:, 0])
    low = np.minimum(open_, close) * np.exp(-spread * u[:, 1])
    frame = pd.DataFrame(
        {
            k: np.round(v, 2)
            for k, v in (("open", open_), ("high", high), ("low", low), ("close", close))
        },
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in _dates(n)], name="date"),
    )
    return frame


def documents(cfg: Config, config_path: Path, workers: int) -> dict[str, bytes]:
    """The two demo files, through the ``forecast`` code path (no commit embedded)."""
    frame = synthetic_index(cfg)
    x = np.asarray(ls.log_losses(frame["close"].to_numpy()), dtype=np.float64)
    dates = [d.date().isoformat() for d in frame.index[1:]]
    js, fcs = live._targets(x, cfg, workers)
    prov: dict[str, Any] = {
        "commit": "synthetic-demo",
        "config_sha256": config_sha256(config_path),
        "raw_manifest_sha256": None,
        "live_manifest_sha256": None,
        "api_manifest_sha256": None,
        "n_losses": int(x.size),
        "demo": f"synthetic index, SeedSequence({cfg.runtime.seed}) child {SEED_CHILD}",
    }
    doc = live._document(x, dates[-1], cfg, js, fcs, prov)
    ser = live.series_document(frame, [SOURCE] * len(frame), x, dates, cfg, js, fcs)
    return {
        live.FORECAST_JSON: backtest.to_json(doc).encode(),
        live.SERIES_JSON: backtest.to_json(ser).encode(),
    }


def run(
    cfg: Config, config_path: str | Path, workers: int, verify: bool, root: str | Path, log: Any
) -> int:
    """``demo-data`` (D62 (b)): write the two files, or with ``verify`` compare bytes."""
    base = Path(root)
    blobs = documents(cfg, base / config_path, workers)
    out = base / DEMO_DIR
    if verify:
        ok = all((out / n).is_file() and (out / n).read_bytes() == b for n, b in blobs.items())
        log.log(
            logging.INFO if ok else logging.ERROR,
            "verify: bytes match" if ok else "verify: MISMATCH",
            extra={"files": sorted(blobs)},
        )
        return 0 if ok else 1
    out.mkdir(parents=True, exist_ok=True)
    for n, b in blobs.items():
        (out / n).write_bytes(b)
    log.info("demo-data written", extra={"files": sorted(blobs)})
    return 0
