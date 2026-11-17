"""Day-7C: the real threshold selection on the training slice (D36; plan 3 Steps 1-5).

The only place where real data reaches ``select_threshold``. Only the first
``design.train_size + 1`` prices are turned into returns; nothing later is touched.

Tracked outputs hold aggregates only (D23): ``selection.json`` (candidate table, ForwardStop,
stable region, split-half, extremal index, declustered-xi comparison, provenance hashes) and
five SVG figures drawn from that JSON alone. Three figures that show per-day information go
to the gitignored ``figures/`` directory. Everything that is not the rule itself (the grid-
floor flag, ``xi_comparison``, the dependence table, the MRL points) is diagnostic and has no
influence on ``q*``. A rule failure is written as evidence; nothing is chosen.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from niftytail import diagnostics as dg
from niftytail import losses as ls
from niftytail.config import Config, config_sha256
from niftytail.data import load_raw
from niftytail.gpd import EstimationError, excesses, fit_mle, profile_ci_xi
from niftytail.threshold import LEVELS, MIN_EXCESSES, ThresholdDecision, select_threshold

__all__ = [
    "FIGURE_NAMES",
    "MIN_CLUSTER_MAXIMA",
    "PINNED_FILES",
    "REAL_DIR",
    "SELECTION_NAME",
    "body_sha256",
    "build_summary",
    "dependence_table",
    "floor_flag",
    "mrl_table",
    "run_selection",
    "summary_json",
    "write_figures",
    "xi_comparison",
]

LABEL = "real training data (aggregates only)"
REAL_DIR = Path("reports/threshold_selection/real")
SELECTION_NAME = "selection.json"
FIGURE_NAMES = (
    "fig_a_mrl.svg",
    "fig_b_xi_beta_star.svg",
    "fig_c_var_es.svg",
    "fig_d_ad_forwardstop.svg",
    "fig_e_theta_declustered.svg",
)
PRIVATE_DIR = "figures"  # gitignored: exceedance timeline, inter-exceedance QQ, indicator ACF
MIN_CLUSTER_MAXIMA = 30  # D36 (d): a descriptive cut for the declustered fit, not a constant
MRL_Z = 1.96  # D36 (d): descriptive band of the mean excess (i.i.d. formula)
XI_COMPARISON_LEVELS = (Fraction(90, 100), Fraction(95, 100))
SVG_HASHSALT = "niftytail-D36"  # fixed, so that matplotlib's SVG ids are reproducible
PINNED_FILES = {  # docs/protected_hashes.md: artefacts whose sha256 is recorded with the selection
    "plan_sha256": Path("docs/plan_v2.2.md"),
    "nmin_study_sha256": Path("reports/threshold_selection/nmin_study.json"),
    "simstudy_metrics_sha256": Path("reports/threshold_selection/simstudy/metrics.json"),
}

Row = dict[str, Any]


def floor_flag(q_star: Fraction | None, grid: Sequence[Fraction]) -> bool | None:
    """True iff ``q*`` is the lowest grid point; ``None`` on a rule failure (D36 (d))."""
    return None if q_star is None else q_star == grid[0]


def _xi_entry(row: Row) -> Row:
    return {k: row[k] for k in ("q", "n_u", "xi_hat", "xi_ci", "xi_ci_lower_open")}


def xi_comparison(rows: Sequence[Row], q_star: str | None) -> dict[str, Row | None]:
    """Shape estimate and profile CI at ``q*``, 0.90 and 0.95, side by side (no new fits).

    ``rows`` are the JSON candidate rows (``q`` as the string of the Fraction).
    """
    by_q = {r["q"]: r for r in rows}
    out: dict[str, Row | None] = {"q_star": None if q_star is None else _xi_entry(by_q[q_star])}
    for q in XI_COMPARISON_LEVELS:
        out[f"q_{float(q):.2f}"] = _xi_entry(by_q[str(q)]) if str(q) in by_q else None
    return out


def mrl_table(losses: ls.FloatArray, rows: Sequence[Row]) -> list[Row]:
    """Mean-residual-life points: mean excess over each ``u_k`` with a +-1.96 SE band."""
    out: list[Row] = []
    for r in rows:
        y = excesses(losses, r["u"])
        mean = float(y.mean())
        se = float(y.std(ddof=1)) / math.sqrt(y.size)
        out.append(
            {
                "q": r["q"],
                "u": r["u"],
                "n_u": int(y.size),
                "mean_excess": mean,
                "band": [mean - MRL_Z * se, mean + MRL_Z * se],
            }
        )
    return out


def _inside(x: float | None, ci: list[float] | None) -> bool | None:
    return None if x is None or ci is None else ci[0] <= x <= ci[1]


def dependence_table(losses: ls.FloatArray, rows: Sequence[Row], cfg: Config) -> list[Row]:
    """Per grid candidate: extremal index, intervals declustering, declustered-xi comparison.

    The declustered fit (``fit_mle`` and the profile CI on the cluster-maximum excesses over
    the same ``u_k``) is attempted only with at least ``MIN_CLUSTER_MAXIMA`` clusters;
    otherwise ``declustered_status`` is ``"not estimated"``. Diagnostic only (D36 (c), (d)).
    """
    tol = float(cfg.threshold.xi_zero_tol)
    out: list[Row] = []
    for r in rows:
        u = r["u"]
        ei = dg.extremal_index_fs(dg.inter_exceedance_times(ls.exceedance_mask(losses, u)))
        dc = dg.decluster_fs(losses, u)
        row: Row = {
            "q": r["q"],
            "n_exceedances": ei.n_exceedances,
            "theta": ei.theta,
            "branch": ei.branch,
            "mean_cluster_size": ei.mean_cluster_size,
            "n_clusters": dc.n_clusters,
            "run_length": dc.run_length,
            "exceedances_per_cluster": ei.n_exceedances / dc.n_clusters,
            "xi_all": r["xi_hat"],
            "xi_all_ci": r["xi_ci"],
            "declustered_status": "not estimated",
            "xi_declustered": None,
            "xi_declustered_ci": None,
            "xi_difference": None,
            "xi_all_in_declustered_ci": None,
            "xi_declustered_in_all_ci": None,
        }
        out.append(row)
        if dc.n_clusters < MIN_CLUSTER_MAXIMA:
            continue
        y = dc.cluster_maxima - u
        try:
            fit = fit_mle(y, min_excesses=MIN_EXCESSES, xi_zero_tol=tol, settings=cfg.gpd)
        except EstimationError:
            row["declustered_status"] = "fit_failed"
            continue
        row.update(declustered_status="estimated", xi_declustered=fit.xi)
        try:
            ci = profile_ci_xi(y, fit, xi_zero_tol=tol, settings=cfg.gpd)
        except EstimationError:
            pass
        else:
            row["xi_declustered_ci"] = [ci.lower, ci.upper]
        if r["xi_hat"] is not None:
            row["xi_difference"] = fit.xi - r["xi_hat"]
        row["xi_all_in_declustered_ci"] = _inside(r["xi_hat"], row["xi_declustered_ci"])
        row["xi_declustered_in_all_ci"] = _inside(fit.xi, r["xi_ci"])
    return out


def summary_json(summary: dict[str, Any]) -> str:
    """Deterministic serialisation: sorted keys, indent 2, LF, trailing newline, no NaN."""
    return json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"


def body_sha256(summary: dict[str, Any]) -> str:
    """SHA-256 of :func:`summary_json` applied to ``summary`` without ``"body_sha256"``."""
    body = {k: v for k, v in summary.items() if k != "body_sha256"}
    return hashlib.sha256(summary_json(body).encode("utf-8")).hexdigest()


def build_summary(
    losses: ls.FloatArray, decision: ThresholdDecision, cfg: Config, provenance: dict[str, Any]
) -> dict[str, Any]:
    """The content of ``selection.json`` (aggregates only; no per-day value, no array of n)."""
    th = cfg.threshold
    dec = json.loads(decision.to_json())  # floats by repr, Fractions as strings, "inf"
    rows: list[Row] = dec["candidates"]
    dependence = dependence_table(losses, rows, cfg)
    halves = decision.halves or ()
    summary: dict[str, Any] = {
        "decision": "D36",
        "label": LABEL,
        "provenance": provenance,
        "design": {
            "grid": [str(q) for q in th.grid],
            "levels": [str(c) for _, c in LEVELS],
            "alpha_fs": str(th.forwardstop_alpha),
            "gof_bootstrap_reps": th.gof_bootstrap_reps,
            "n_min": th.n_min,
            "q_max": str(1 - Fraction(th.n_min or 0, cfg.design.window)),
            "window": cfg.design.window,
            "seed": cfg.runtime.seed,
            "ci_level": str(cfg.gpd.ci_level),
            "min_cluster_maxima": MIN_CLUSTER_MAXIMA,
        },
        "selection": {
            "q_star": dec["q_star"],
            "q_fs": dec["q_fs"],
            "k_hat": dec["k_hat"],
            "rule_failure": dec["rule_failure"],
            "q_star_at_grid_floor": floor_flag(decision.q_star, th.grid),
            "flags": dec["flags"],
        },
        "split_half": {
            "flag": "split_half_flag" in decision.flags,
            "limit": str(th.split_half_flag),
            "halves": [
                {
                    "q_star": None if h.q_star is None else str(h.q_star),
                    "q_fs": None if h.q_fs is None else str(h.q_fs),
                    "k_hat": h.k_hat,
                    "rule_failure": h.rule_failure,
                }
                for h in halves
            ],
        },
        "candidates": rows,
        "xi_comparison": xi_comparison(rows, dec["q_star"]),
        "dependence": dependence,
        "dependence_at_q_star": next((d for d in dependence if d["q"] == dec["q_star"]), None),
        "mrl": mrl_table(losses, rows),
    }
    summary["body_sha256"] = body_sha256(summary)
    return summary


# ---------------------------------------------------------------------------
# Figures (tracked SVG; drawn from the summary alone, so aggregates only)
# ---------------------------------------------------------------------------
def _num(x: Any) -> float:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) else math.nan


def _col(rows: Sequence[Row], key: str) -> list[float]:
    return [_num(r[key]) for r in rows]


def _ci(rows: Sequence[Row], key: str) -> tuple[list[float], list[float]]:
    pairs = [r[key] if r[key] is not None else (math.nan, math.nan) for r in rows]
    return [_num(p[0]) for p in pairs], [_num(p[1]) for p in pairs]


def write_figures(summary: dict[str, Any], fig_dir: Path) -> list[Path]:
    """Write the five SVG figures from ``summary`` (deterministic bytes, as the D35 figures).

    (a) mean residual life over the grid thresholds; (b) ``xi_hat`` with its profile CI and
    ``beta*`` against q; (c) VaR and ES at 95% and 99% with delta-method CIs; (d) AD p-value
    and the ForwardStop statistic against ``alpha_FS``, ``q_FS`` marked; (e) ``theta_hat``
    and ``xi`` from all exceedances against the declustered ``xi``. ``q*`` is marked if defined.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    rows, dep, mrl = summary["candidates"], summary["dependence"], summary["mrl"]
    sel = summary["selection"]
    q = [float(Fraction(r["q"])) for r in rows]
    q_star = None if sel["q_star"] is None else float(Fraction(sel["q_star"]))
    q_fs = None if sel["q_fs"] is None else float(Fraction(sel["q_fs"]))
    alpha = float(Fraction(summary["design"]["alpha_fs"]))
    out: list[Path] = []

    def mark(ax: Axes, x: float | None = q_star, label: str = "q*") -> None:
        if x is not None:
            ax.axvline(x, color="tab:red", linestyle="--", linewidth=0.9, label=label)

    def series(
        ax: Axes,
        x: Sequence[float],
        y: Sequence[float],
        ci: tuple[list[float], list[float]] | None,
        colour: str,
        label: str,
    ) -> None:
        ax.plot(x, y, "o-", color=colour, markersize=3, linewidth=0.9, label=label)
        if ci is not None:
            ax.vlines(x, ci[0], ci[1], color=colour, linewidth=0.8, alpha=0.6)

    def save(fig: Figure, name: str) -> None:
        fig.suptitle(LABEL, fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
        plt.close(fig)
        (fig_dir / name).write_bytes(buf.getvalue().replace(b"\r\n", b"\n"))
        out.append(fig_dir / name)

    fig_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT, "svg.fonttype": "path"}):
        fig, ax = plt.subplots(figsize=(7, 4))
        band = ([m["band"][0] for m in mrl], [m["band"][1] for m in mrl])
        series(ax, _col(mrl, "u"), _col(mrl, "mean_excess"), band, "tab:blue", "mean excess")
        if q_star is not None:
            mark(ax, next(m["u"] for m in mrl if m["q"] == sel["q_star"]), "u(q*)")
        ax.set_xlabel("threshold u_k (grid q = 0.80 ... 0.95)")
        ax.set_ylabel("mean excess, ±1.96 SE (i.i.d. formula)")
        ax.set_title("Mean residual life over the grid thresholds", fontsize=9)
        ax.legend(fontsize=8)
        save(fig, FIGURE_NAMES[0])

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        series(a1, q, _col(rows, "xi_hat"), _ci(rows, "xi_ci"), "tab:blue", "xi_hat, profile CI")
        series(a2, q, _col(rows, "beta_star"), None, "tab:green", "beta* = beta - xi u")
        for ax in (a1, a2):
            mark(ax)
            ax.set_xlabel("q")
            ax.legend(fontsize=8)
        save(fig, FIGURE_NAMES[1])

        fig, axes = plt.subplots(1, len(LEVELS), figsize=(10, 4))
        for ax, (label, _) in zip(axes, LEVELS, strict=True):
            for m, colour in (("var", "tab:blue"), ("es", "tab:orange")):
                key = f"{m}{label}"
                series(ax, q, _col(rows, key), _ci(rows, key + "_ci"), colour, key)
            mark(ax)
            ax.set_xlabel("q")
            ax.set_title(f"VaR and ES at {label}%, delta-method CI", fontsize=9)
            ax.legend(fontsize=8)
        save(fig, FIGURE_NAMES[2])

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        series(a1, q, _col(rows, "ad_p"), None, "tab:blue", "AD p-value")
        rejected = [(x, r["ad_p"]) for x, r in zip(q, rows, strict=True) if r["rejected"]]
        if rejected:
            a1.plot(*zip(*rejected, strict=True), "x", color="tab:red", label="rejected")
        a1.set_ylim(-0.02, 1.02)
        series(a2, q, _col(rows, "fs_stat"), None, "tab:blue", "ForwardStop statistic")
        a2.axhline(alpha, color="black", linewidth=0.8, label="alpha_FS")
        a2.set_yscale("log")
        for ax in (a1, a2):
            mark(ax, q_fs, "q_FS")
            ax.set_xlabel("q")
            ax.legend(fontsize=8)
        save(fig, FIGURE_NAMES[3])

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        series(a1, q, _col(dep, "theta"), None, "tab:blue", "theta_hat (intervals)")
        a1.set_ylim(0, 1.05)
        lo = [x - 0.0015 for x in q]
        hi = [x + 0.0015 for x in q]
        series(a2, lo, _col(dep, "xi_all"), _ci(dep, "xi_all_ci"), "tab:blue", "xi, all")
        series(
            a2,
            hi,
            _col(dep, "xi_declustered"),
            _ci(dep, "xi_declustered_ci"),
            "tab:orange",
            "xi, cluster maxima",
        )
        for ax in (a1, a2):
            mark(ax)
            ax.set_xlabel("q")
            ax.legend(fontsize=8)
        save(fig, FIGURE_NAMES[4])
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_selection(
    cfg: Config, config_path: str | Path, out_dir: Path = REAL_DIR
) -> tuple[dict[str, Any], list[Path]]:
    """Load the verified raw data, run Steps 1-5 on the training slice, write the evidence.

    Returns ``(summary, written paths)``; the caller inspects ``summary["selection"]``.

    Raises:
        DataValidationError: from the manifest-verified loader.
        ValueError: if the series has fewer than ``train_size + 1`` prices (nothing is
            written), or from the procedure (messages contain no data values).
        OSError: if a pinned artefact or the manifest cannot be read.
    """
    from niftytail import plots  # matplotlib only when figures are drawn

    t0 = cfg.design.train_size
    provenance: dict[str, Any] = {key: _sha256(path) for key, path in PINNED_FILES.items()}
    provenance["config_sha256"] = config_sha256(config_path)
    provenance["manifest_sha256"] = _sha256(Path(cfg.data.manifest))
    df = load_raw(cfg.data.raw_dir, cfg.data.manifest, start_date=cfg.data.start_date)
    if len(df) < t0 + 1:
        raise ValueError(f"series too short: design.train_size = {t0} returns need {t0 + 1} prices")
    train = df.iloc[: t0 + 1]  # later prices are never turned into returns
    r = ls.log_returns(train["close"].to_numpy())
    dates = ls.return_dates(pd.DatetimeIndex(train.index))
    losses = -r
    provenance.update(
        n_train=int(losses.size),
        first_train_return_date=dates[0].date().isoformat(),
        last_train_return_date=dates[-1].date().isoformat(),
    )
    decision = select_threshold(
        losses,
        cfg,
        ss=np.random.SeedSequence(cfg.runtime.seed),
        gof_reps=cfg.threshold.gof_bootstrap_reps,
        convergence_share=True,
        halves=True,
        intervals=True,
    )
    summary = build_summary(losses, decision, cfg, provenance)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SELECTION_NAME
    path.write_bytes(summary_json(summary).encode("utf-8"))
    outputs = [path, *write_figures(summary, out_dir)]
    if decision.q_star is not None:
        outputs += plots.write_exceedance_figures(
            out_dir / PRIVATE_DIR,
            dates,
            losses,
            levels=[decision.q_star],
            acf_max_lag=cfg.dependence.acf_max_lag,
            band_z=cfg.dependence.acf_band_z,
        )
    return summary, outputs
