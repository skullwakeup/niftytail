"""Day-4 EDA: the dependence battery and the aggregate-only ``eda_summary.json`` (D29).

The summary holds aggregates only (counts, moments, test statistics, p-values, quantile
thresholds, theta) and dates; never index levels or per-day returns (D23). Training
results are decision-relevant; full-sample results are labelled "descriptive".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from niftytail import diagnostics as dg
from niftytail import losses as ls
from niftytail.config import Config, config_sha256
from niftytail.data import load_raw

__all__ = ["SUMMARY_NAME", "battery", "level_key", "run_eda", "summary_json"]

SUMMARY_NAME = "eda_summary.json"


def level_key(q: Fraction) -> str:
    """JSON key of an exceedance level, e.g. ``Fraction(4, 5) -> "0.8"``."""
    return repr(float(q))


def _lb(x: ls.FloatArray, lags: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in lags:
        res = dg.ljung_box(x, m)
        out[str(m)] = {"q": res.q, "df": res.df, "p": res.p}
    return out


def battery(
    r: ls.FloatArray,
    *,
    acf_max_lag: int,
    ljung_box_lags: Sequence[int],
    arch_lm_lags: Sequence[int],
    levels: Sequence[Fraction],
    band_z: Fraction,
) -> dict[str, Any]:
    """Descriptive statistics, ACF/Ljung-Box, ARCH-LM and exceedance diagnostics of ``r``."""
    n = r.size
    jb = stats.jarque_bera(r)
    series: dict[str, Any] = {}
    for name, x in (("r", r), ("abs_r", np.abs(r)), ("r2", r**2)):
        series[name] = {
            "acf": dg.acf(x, acf_max_lag).tolist(),
            "ljung_box": _lb(x, ljung_box_lags),
        }
    arch: dict[str, Any] = {}
    for lags in arch_lm_lags:
        a = dg.arch_lm(r, lags)
        arch[str(lags)] = {"lm": a.lm, "df": a.df, "p": a.p, "nobs": a.nobs}
    loss = -r
    exc: dict[str, Any] = {}
    for q in levels:
        u = ls.tail_threshold(loss, q)
        mask = ls.exceedance_mask(loss, u)
        ind = mask.astype(np.float64)
        ei = dg.extremal_index_fs(dg.inter_exceedance_times(mask))
        exc[level_key(q)] = {
            "q": level_key(q),
            "u": u,
            "n_exceedances": int(mask.sum()),
            "theta": ei.theta,
            "branch": ei.branch,
            "mean_cluster_size": ei.mean_cluster_size,
            "indicator_acf": dg.acf(ind, acf_max_lag).tolist(),
            "indicator_ljung_box": _lb(ind, ljung_box_lags),
        }
    return {
        "descriptive_stats": {
            "n": n,
            "mean": float(np.mean(r)),
            "std": float(np.std(r, ddof=1)),
            "skewness": float(stats.skew(r)),
            "excess_kurtosis": float(stats.kurtosis(r)),
            "jarque_bera": {"stat": float(jb.statistic), "p": float(jb.pvalue)},
        },
        "acf_band": float(band_z) / float(np.sqrt(n)),
        "series": series,
        "arch_lm": arch,
        "exceedances": exc,
    }


def summary_json(summary: dict[str, Any]) -> str:
    """Deterministic serialisation: sorted keys, indent 2, LF, trailing newline, no NaN."""
    return json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"


def run_eda(cfg: Config, config_path: str | Path) -> list[Path]:
    """Load the verified raw data, run the battery, write the summary and figures.

    Returns the written paths. Raises ``DataValidationError`` or ``ValueError`` (messages
    contain no data values).
    """
    from niftytail import plots  # matplotlib only when figures are drawn

    dep = cfg.dependence
    df = load_raw(cfg.data.raw_dir, cfg.data.manifest, start_date=cfg.data.start_date)
    r = ls.log_returns(df["close"].to_numpy())
    dates = ls.return_dates(pd.DatetimeIndex(df.index))
    t0 = cfg.design.train_size
    r_tr = ls.training_slice(r, t0)
    d_tr = dates[:t0]
    kw: dict[str, Any] = {
        "acf_max_lag": dep.acf_max_lag,
        "ljung_box_lags": dep.ljung_box_lags,
        "arch_lm_lags": dep.arch_lm_lags,
        "levels": dep.exceedance_levels,
        "band_z": dep.acf_band_z,
    }

    def sample(x: ls.FloatArray, d: pd.DatetimeIndex, label: str) -> dict[str, Any]:
        return {
            "label": label,
            "n_returns": int(x.size),
            "first_return_date": d[0].date().isoformat(),
            "last_return_date": d[-1].date().isoformat(),
            **battery(x, **kw),
        }

    summary = {
        "decision": "D29",
        "config_sha256": config_sha256(config_path),
        "manifest_sha256": hashlib.sha256(Path(cfg.data.manifest).read_bytes()).hexdigest(),
        "train_size": t0,
        "training": sample(r_tr, d_tr, "decision-relevant"),
        "full_sample": sample(r, dates, "descriptive"),
    }
    out = Path(dep.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SUMMARY_NAME
    path.write_bytes(summary_json(summary).encode("utf-8"))
    figs = plots.write_figures(
        out / "figures",
        d_tr,
        r_tr,
        levels=dep.exceedance_levels,
        acf_max_lag=dep.acf_max_lag,
        band_z=dep.acf_band_z,
    )
    return [path, *figs]
