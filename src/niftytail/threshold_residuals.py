"""Day-8A: threshold evidence on GARCH standardized residuals of the training slice (D38 (9)).

One ``fit_garch`` on the ``design.train_size`` training losses; ``z`` = its standardized
residuals; then the unchanged D34 procedure (``select_threshold``) on ``z`` with a seed node
separate from Day 7C. The Day-7C machinery (``threshold_real``) is reused as it is: the
summary has the 7C schema plus a ``"garch"`` block and ``"residual_diagnostics"``.

Stated limitation: the convergence-share column and the split halves refit the GPD on
residuals of this single training GARCH fit; out of sample the residuals come from a refit
in every rolling window.

Day 8A-2 (D39 (g)): with ``selected_filter=True`` the same procedure runs on the residuals of
the filter chosen by training BIC (``garch.select_filter``), as a DESCRIPTIVE run only: no
decision may cite it (``q_c`` is fixed by design, D39 (a)). That fit has ``train_size - 1``
residuals, so the procedure gets a configuration copy with that ``train_size``. It writes
``selection_filter.json`` and never touches the Day-8A ``selection.json``.

Tracked outputs hold aggregates only (D23). No residual, sigma or loss series is written or
logged. A rule failure is written as evidence; nothing is chosen.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from niftytail import diagnostics as dg
from niftytail import losses as ls
from niftytail import threshold_real as tr
from niftytail.config import Config, config_sha256
from niftytail.data import load_raw
from niftytail.garch import SPECS, GarchFit, fit_garch, garch_filter, n_params, select_filter
from niftytail.threshold import ThresholdDecision, select_threshold

__all__ = [
    "DESCRIPTIVE_LABEL",
    "DESCRIPTIVE_NAME",
    "LABEL",
    "RESIDUAL_DIR",
    "build_summary",
    "filter_block",
    "garch_block",
    "load_training",
    "moments",
    "residual_seed",
    "run_selection",
    "serial_tests",
    "write_figures",
]

LABEL = "GARCH(1,1) standardized residuals, training (aggregates only)"
DESCRIPTIVE_LABEL = "DESCRIPTIVE ONLY (D39): residuals of the BIC-selected filter {}, training"
DESCRIPTIVE_NAME = "selection_filter"  # selection_filter.json; figures in selection_filter/
RESIDUAL_DIR = Path("reports/threshold_selection/real_residuals")


def residual_seed(seed: int) -> np.random.SeedSequence:
    """``SeedSequence(seed).spawn(2)[1]``: spawn key ``(1,)``, separate from Day 7C (D38 (9))."""
    return np.random.SeedSequence(seed).spawn(2)[1]


def garch_block(fit: GarchFit, n_obs: int) -> dict[str, Any]:
    """Estimates and flags of the training fit (aggregates only)."""
    return {
        "n_obs": n_obs,
        "mu": fit.mu,
        "omega": fit.omega,
        "alpha": fit.alpha,
        "beta": fit.beta,
        "persistence": fit.alpha + fit.gamma / 2 + fit.beta,
        "unconditional_sd": float(np.sqrt(fit.omega / (1 - fit.alpha - fit.gamma / 2 - fit.beta))),
        "nll": fit.nll,
        "start": fit.start,
        "flags": list(fit.flags),
    }


def filter_block(fit: GarchFit, n_residuals: int) -> dict[str, Any]:
    """Which filter a descriptive run used: spec, k and the terms ``garch_block`` lacks."""
    ar, gjr = SPECS[fit.spec]
    return {
        "spec": fit.spec,
        "k": n_params(fit.spec),
        "phi1": fit.phi1 if ar else None,
        "gamma": fit.gamma if gjr else None,
        "n_residuals": n_residuals,
    }


def moments(z: ls.FloatArray) -> dict[str, float]:
    """Mean, variance (ddof = 0), skewness and excess kurtosis of ``z``."""
    return {
        "mean": float(np.mean(z)),
        "variance": float(np.var(z)),
        "skewness": float(stats.skew(z)),
        "excess_kurtosis": float(stats.kurtosis(z)),
    }


def serial_tests(
    x: ls.FloatArray, ljung_box_lags: Sequence[int], arch_lm_lags: Sequence[int]
) -> dict[str, Any]:
    """Ljung-Box on ``x`` and ``x**2`` and ARCH-LM on ``x`` (D29 conventions)."""

    def lb(series: ls.FloatArray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for m in ljung_box_lags:
            res = dg.ljung_box(series, m)
            out[str(m)] = {"q": res.q, "df": res.df, "p": res.p}
        return out

    arch: dict[str, Any] = {}
    for lags in arch_lm_lags:
        a = dg.arch_lm(x, lags)
        arch[str(lags)] = {"lm": a.lm, "df": a.df, "p": a.p, "nobs": a.nobs}
    return {"ljung_box": {"level": lb(x), "squared": lb(x**2)}, "arch_lm": arch}


def build_summary(
    losses: ls.FloatArray,
    z: ls.FloatArray,
    fit: GarchFit,
    decision: ThresholdDecision,
    cfg: Config,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """The 7C summary computed on ``z``, plus the GARCH block and the residual diagnostics."""
    dep = cfg.dependence
    summary = tr.build_summary(z, decision, cfg, provenance)
    summary["decision"] = "D38"
    summary["label"] = LABEL
    summary["design"]["seed_spawn_key"] = list(residual_seed(cfg.runtime.seed).spawn_key)
    summary["garch"] = garch_block(fit, int(z.size))
    summary["residual_diagnostics"] = {
        "moments_z": moments(z),
        "residuals": serial_tests(z, dep.ljung_box_lags, dep.arch_lm_lags),
        "raw_losses": serial_tests(losses, dep.ljung_box_lags, dep.arch_lm_lags),
    }
    summary["body_sha256"] = tr.body_sha256(summary)
    return summary


def write_figures(summary: dict[str, Any], fig_dir: Path) -> list[Path]:
    """The five 7C figures, drawn from the residual summary and titled with ``LABEL``."""
    # ponytail: swaps tr.LABEL because the 7C module is not to be edited on Day 8A (single
    # process, serial); give tr.write_figures a label argument once 7C may be touched.
    old = tr.LABEL
    tr.LABEL = summary["label"]
    try:
        return tr.write_figures(summary, fig_dir)
    finally:
        tr.LABEL = old


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training(
    cfg: Config, config_path: str | Path
) -> tuple[ls.FloatArray, pd.DatetimeIndex, dict[str, Any]]:
    """The training losses, their dates and the provenance block, loaded as on Day 7C.

    Raises:
        DataValidationError: from the manifest-verified loader.
        ValueError: if the series has fewer than ``train_size + 1`` prices.
        OSError: if a pinned artefact or the manifest cannot be read.
    """
    t0 = cfg.design.train_size
    provenance: dict[str, Any] = {key: _sha256(path) for key, path in tr.PINNED_FILES.items()}
    provenance["config_sha256"] = config_sha256(config_path)
    provenance["manifest_sha256"] = _sha256(Path(cfg.data.manifest))
    df = load_raw(cfg.data.raw_dir, cfg.data.manifest, start_date=cfg.data.start_date)
    if len(df) < t0 + 1:
        raise ValueError(f"series too short: design.train_size = {t0} returns need {t0 + 1} prices")
    train = df.iloc[: t0 + 1]  # later prices are never turned into returns
    losses = -ls.log_returns(train["close"].to_numpy())
    dates = ls.return_dates(pd.DatetimeIndex(train.index))
    provenance.update(
        n_train=int(losses.size),
        first_train_return_date=dates[0].date().isoformat(),
        last_train_return_date=dates[-1].date().isoformat(),
    )
    return losses, dates, provenance


def run_selection(
    cfg: Config,
    config_path: str | Path,
    out_dir: Path = RESIDUAL_DIR,
    *,
    selected_filter: bool = False,
) -> tuple[dict[str, Any], list[Path]]:
    """Load the training slice as Day 7C, filter it, run Steps 1-5 on ``z``, write the evidence.

    ``selected_filter=False`` is the Day-8A run (D38 (9)): the F1 filter on all training
    losses, written to ``selection.json``. ``selected_filter=True`` is the descriptive D39 (g)
    run on the residuals of the BIC-selected fit, written to ``selection_filter.json``.

    Raises:
        DataValidationError: from the manifest-verified loader.
        GarchError: if a training GARCH fit fails (nothing is written).
        ValueError: if the series has fewer than ``train_size + 1`` prices (nothing is
            written), or from the procedure (messages contain no data values).
        OSError: if a pinned artefact or the manifest cannot be read.
    """
    from niftytail import plots  # matplotlib only when figures are drawn

    losses, dates, provenance = load_training(cfg, config_path)
    name, fig_dir, private = tr.SELECTION_NAME, out_dir, out_dir / tr.PRIVATE_DIR
    if selected_filter:
        fits, winner = select_filter(losses, cfg.garch)
        fit = fits[winner]
        z = garch_filter(losses[1:], fit, float(losses[0]))[2]
        dates = dates[1:]
        # D39 (g): the procedure accepts exactly train_size values; this fit has one fewer.
        cfg = dataclasses.replace(cfg, design=dataclasses.replace(cfg.design, train_size=z.size))
        name, fig_dir = f"{DESCRIPTIVE_NAME}.json", out_dir / DESCRIPTIVE_NAME
        private = private / DESCRIPTIVE_NAME
    else:
        fit = fit_garch(losses, cfg.garch)
        z = garch_filter(losses, fit)[2]
    decision = select_threshold(
        z,
        cfg,
        ss=residual_seed(cfg.runtime.seed),
        gof_reps=cfg.threshold.gof_bootstrap_reps,
        convergence_share=True,
        halves=True,
        intervals=True,
    )
    summary = build_summary(losses, z, fit, decision, cfg, provenance)
    if selected_filter:
        summary.update(
            decision="D39",
            label=DESCRIPTIVE_LABEL.format(fit.spec),
            descriptive_only=True,
            filter=filter_block(fit, int(z.size)),
        )
        summary["body_sha256"] = tr.body_sha256(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes(tr.summary_json(summary).encode("utf-8"))
    outputs = [path, *write_figures(summary, fig_dir)]
    if decision.q_star is not None:
        outputs += plots.write_exceedance_figures(
            private,
            dates,
            z,
            levels=[decision.q_star],
            acf_max_lag=cfg.dependence.acf_max_lag,
            band_z=cfg.dependence.acf_band_z,
        )
    return summary, outputs
