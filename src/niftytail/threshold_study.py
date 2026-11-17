"""A3 threshold-procedure simulation study (D35; plan A3; methodology 8). Synthetic data only.

Every result of this module is **simulation evidence (DGPs S1-S4)**, never a guarantee.

Cells (fixed order): S1 spliced normal/GPD for ``s1_q0`` (outer) x ``s1_xi`` (inner); S2
i.i.d. Student-t; S3 max-autoregressive Frechet process with the S1 marginal, one cell per
``s3_armax_a``; S4 GARCH(1,1) with standardised t innovations (simulated, never fitted).

Seeds (D35 (1)): with ``root = SeedSequence(runtime.seed)`` and ``m`` cells, cell ``c`` is
``root.spawn(m + 1)[c]`` and the S4 reference simulation ``root.spawn(m + 1)[m]``;
replication ``r`` is ``cell.spawn(R)[r]``; its data stream is ``PCG64(rep.spawn(2)[0])``
and ``rep.spawn(2)[1]`` goes to :func:`niftytail.threshold.select_threshold`. The children
are addressed by spawn key, so a record depends only on ``(config, cell, replication)``:
results do not depend on the worker count or on ``R``.

Per replication (D35 (3)): the selection runs without split halves, intervals or the
convergence column; if ``q*`` is defined, the estimates at ``q*`` and a refit on the last
``window_refit`` losses are recorded. Records are aggregated into the metrics M1-M6 and the
D13 review triggers, which are reported and never acted on. Nothing here reads a file.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import special, stats

from niftytail.config import Config, SimStudyConfig
from niftytail.gpd import EstimationError, excesses, fit_mle, tail_risk
from niftytail.losses import tail_threshold
from niftytail.threshold import LEVELS, MIN_EXCESSES, select_threshold

__all__ = [
    "FIGURE_NAMES",
    "LABEL",
    "METRICS_NAME",
    "S4_REFERENCE_BATCHES",
    "STUDY_DIR",
    "Cell",
    "aggregate_cell",
    "armax_frechet",
    "cell_truth",
    "cells",
    "check_design",
    "default_workers",
    "garch_returns",
    "project_seconds",
    "records_sha256",
    "reference_stats",
    "run_records",
    "run_replication",
    "run_study",
    "s1_ppf",
    "s1_truth",
    "s2_truth",
    "s3_losses",
    "s4_losses",
    "s4_reference",
    "sample",
    "summarise",
    "summary_json",
    "triggers",
    "wilson",
    "window_refit",
    "write_figures",
    "write_outputs",
]

LABEL = "simulation evidence (DGPs S1-S4)"
STUDY_DIR = Path("reports/threshold_selection/simstudy")
METRICS_NAME = "metrics.json"
FIGURE_NAMES = (
    "fig_a_q_star.svg",
    "fig_b_fdr.svg",
    "fig_c_xi_error.svg",
    "fig_d_risk_error.svg",
    "fig_e_failures.svg",
)
S4_REFERENCE_BATCHES = 10  # D35 (2): batch means for the Monte Carlo error of the reference
S4_CHUNK = 1_000_000  # innovations drawn per block of the GARCH recursion (memory only)
PILOT_REPLICATIONS = 2  # D35 (8): replications per cell in the runtime-guard pilot
MAX_WORKERS = 31  # D35 (6): default workers = min(cpu_count - 1, 31)
FDR_Z = 1.96  # D35 (4) M3: mean +/- 1.96 SD / sqrt(R)
WILSON_Z = float(stats.norm.ppf(0.975))  # D35 (4) M2: Wilson 95% interval
SVG_HASHSALT = "niftytail-D35"  # fixed, so that matplotlib's SVG ids are reproducible
RISK_KEYS = tuple(f"{m}{label}" for label, _ in LEVELS for m in ("var", "es"))

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
Record = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Cell:
    """One study cell. ``xi`` is the true shape (S1, S3) or the asymptotic one (S2)."""

    index: int
    dgp: Literal["S1", "S2", "S3", "S4"]
    label: str
    q0: Fraction | None = None
    xi: Fraction | None = None
    a: Fraction | None = None


def cells(cfg: Config) -> tuple[Cell, ...]:
    """The cells in their fixed order (D35 (1)): S1 (q0 outer, xi inner), S2, S3 (a), S4."""
    ss = cfg.simstudy
    out = [
        Cell(0, "S1", f"S1 q0={float(q0)} xi={float(xi)}", q0, xi)
        for q0 in ss.s1_q0
        for xi in ss.s1_xi
    ]
    out.append(Cell(0, "S2", f"S2 t({ss.s2_t_df})", xi=Fraction(1, ss.s2_t_df)))
    out += [Cell(0, "S3", f"S3 a={float(a)}", ss.s3_q0, ss.s3_xi, a) for a in ss.s3_armax_a]
    out.append(Cell(0, "S4", f"S4 GARCH t({ss.s4_t_df})"))
    return tuple(Cell(i, c.dgp, c.label, c.q0, c.xi, c.a) for i, c in enumerate(out))


# ---------------------------------------------------------------------------
# Data-generating processes and their truth (D35 (2))
# ---------------------------------------------------------------------------
def _s1_params(q0: Fraction, xi: Fraction) -> tuple[float, float, float, float]:
    if not (0 < q0 < 1 and 0 < xi < 1):
        raise ValueError("S1: needs 0 < q0 < 1 and 0 < xi < 1")
    u0 = float(special.ndtri(float(q0)))
    return float(q0), float(xi), u0, float(1 - q0) / float(stats.norm.pdf(u0))


def s1_ppf(v: npt.ArrayLike, q0: Fraction, xi: Fraction) -> FloatArray:
    """Quantile function of the spliced distribution S1.

    ``Phi^-1(v)`` for ``v <= q0``; ``u0 + beta0 ((1 - w)^-xi - 1) / xi`` with
    ``w = (v - q0) / (1 - q0)`` above, where ``u0 = Phi^-1(q0)`` and
    ``beta0 = (1 - q0) / phi(u0)`` (continuous density at ``u0``).
    """
    q, x, u0, beta0 = _s1_params(q0, xi)
    p = np.asarray(v, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore"):
        w = (np.maximum(p, q) - q) / (1 - q)  # same float 1 - q as at v = 1: w <= 1
        tail = u0 + beta0 * np.expm1(-x * np.log1p(-w)) / x
        body = special.ndtri(np.minimum(p, q))
    return np.asarray(np.where(p <= q, body, tail), dtype=np.float64)


def s1_truth(c: Fraction, q0: Fraction, xi: Fraction) -> tuple[float, float]:
    """Exact ``(VaR_c, ES_c)`` of S1: ``ES = VaR + (beta0 + xi (VaR - u0)) / (1 - xi)``.

    Raises:
        ValueError: unless ``q0 <= c < 1`` (the ES formula needs the VaR inside the GPD part).
    """
    if not q0 <= c < 1:
        raise ValueError("S1 truth: needs q0 <= c < 1")
    _, x, u0, beta0 = _s1_params(q0, xi)
    var = float(s1_ppf(float(c), q0, xi))
    return var, var + (beta0 + x * (var - u0)) / (1 - x)


def s2_truth(c: Fraction, nu: int) -> tuple[float, float]:
    """Exact ``(VaR_c, ES_c)`` of a standard Student-t with ``nu > 1`` degrees of freedom."""
    var = float(stats.t.ppf(float(c), nu))
    es = float(stats.t.pdf(var, nu)) / float(1 - c) * (nu + var * var) / (nu - 1)
    return var, es


def armax_frechet(u: npt.ArrayLike, a: float) -> FloatArray:
    """Max-autoregressive Frechet path from uniforms: ``X_0 = Z_0``,
    ``X_t = max(a X_{t-1}, (1 - a) Z_t)`` with ``Z_t = -1 / ln(U_t)`` (unit Frechet margins).

    Raises:
        ValueError: unless every ``U_t`` lies strictly inside (0, 1) (the maximum would
            otherwise absorb a zero or non-finite innovation silently).
    """
    with np.errstate(divide="ignore"):
        innovations = -1.0 / np.log(np.asarray(u, dtype=np.float64))
    if not np.all(np.isfinite(innovations) & (innovations > 0)):
        raise ValueError("S3: non-finite or zero Frechet innovation")
    z = innovations.tolist()
    x = z[:1]
    for zt in z[1:]:
        x.append(max(a * x[-1], (1 - a) * zt))
    return np.array(x, dtype=np.float64)


def s3_losses(u: npt.ArrayLike, a: Fraction, q0: Fraction, xi: Fraction) -> FloatArray:
    """S3: ``L_t = F_S1^-1(exp(-1 / X_t))`` for the ARMAX path ``X`` built from ``u``.

    Raises:
        ValueError: if any loss is not finite (nothing is clipped).
    """
    loss = s1_ppf(np.exp(-1.0 / armax_frechet(u, float(a))), q0, xi)
    if not np.all(np.isfinite(loss)):
        raise ValueError("S3: non-finite loss")
    return loss


def garch_returns(
    z: npt.ArrayLike, omega: float, a1: float, b1: float, s2: float = 1.0
) -> tuple[FloatArray, float]:
    """GARCH(1,1) returns ``r_t = sqrt(s2_t) z_t``, ``s2_{t+1} = omega + a1 r_t^2 + b1 s2_t``.

    ``s2`` is the conditional variance of the first return; the second element is the
    conditional variance that follows the last one (to continue the path in blocks).
    """
    out: list[float] = []
    for zt in np.asarray(z, dtype=np.float64).tolist():
        r = math.sqrt(s2) * zt
        out.append(r)
        s2 = omega + a1 * r * r + b1 * s2
    return np.array(out, dtype=np.float64), s2


def s4_losses(g: np.random.Generator, n: int, ss: SimStudyConfig) -> FloatArray:
    """S4: ``n`` losses ``L = -r`` after ``s4_burn_in`` discarded steps, ``sigma^2_0 = 1``.

    ``omega = 1 - a1 - b1`` (unit variance); innovations ``t(nu) sqrt((nu - 2) / nu)``.
    """
    nu = ss.s4_t_df
    scale = math.sqrt((nu - 2) / nu)
    omega = float(1 - ss.s4_garch_a1 - ss.s4_garch_b1)
    a1, b1 = float(ss.s4_garch_a1), float(ss.s4_garch_b1)
    total = ss.s4_burn_in + n
    parts: list[FloatArray] = []
    s2 = 1.0
    for start in range(0, total, S4_CHUNK):
        z = g.standard_t(nu, size=min(S4_CHUNK, total - start)) * scale
        r, s2 = garch_returns(z, omega, a1, b1, s2)
        parts.append(r)
    return -np.concatenate(parts)[ss.s4_burn_in :]


def _empirical(x: FloatArray) -> dict[str, float]:
    xs = np.sort(x)
    out = {"variance": float(np.var(x, ddof=1))}
    for label, c in LEVELS:
        rank = -((-xs.size * c.numerator) // c.denominator)  # ceil(c N), integer arithmetic
        var = float(xs[rank - 1])
        out[f"var{label}"] = var
        out[f"es{label}"] = float(np.mean(x[x > var]))
    return out


def reference_stats(losses: npt.ArrayLike, batches: int) -> dict[str, Any]:
    """Empirical reference ``VaR_c = L_(ceil(c N))``, ``ES_c = mean(L | L > VaR_c)`` and the
    sample variance, each with a batch-means standard error (D35 (2)).

    The sample is cut into ``batches`` contiguous blocks; ``se`` is the SD (ddof = 1) of
    the block estimates divided by ``sqrt(batches)``; ``rel_se = se / |value|``.

    Raises:
        ValueError: if the sample does not split into at least two equal blocks.
    """
    x = np.asarray(losses, dtype=np.float64)
    if batches < 2 or x.size % batches:
        raise ValueError("reference: the sample must split into >= 2 equal batches")
    full = _empirical(x)
    per = [_empirical(b) for b in np.split(x, batches)]
    out: dict[str, Any] = {"n": int(x.size), "batches": batches}
    for key, value in full.items():
        se = float(np.std([b[key] for b in per], ddof=1)) / math.sqrt(batches)
        out[key] = {"value": value, "se": se, "rel_se": se / abs(value)}
    out["variance_within_3_se"] = abs(full["variance"] - 1.0) <= 3 * out["variance"]["se"]
    return out


def _stream(cfg: Config, *key: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(cfg.runtime.seed, spawn_key=key)


def s4_reference(cfg: Config) -> dict[str, Any]:
    """The S4 reference: one simulation of ``s4_reference_size`` kept losses (D35 (2))."""
    g = np.random.Generator(np.random.PCG64(_stream(cfg, len(cells(cfg)))))
    losses = s4_losses(g, cfg.simstudy.s4_reference_size, cfg.simstudy)
    return reference_stats(losses, S4_REFERENCE_BATCHES)


def cell_truth(cell: Cell, cfg: Config, reference: dict[str, Any]) -> dict[str, Any]:
    """True shape and VaR/ES of a cell: exact (S1, S2, S3) or the S4 reference."""
    kind = {"S1": "exact", "S2": "asymptotic", "S3": "exact", "S4": None}[cell.dgp]
    out: dict[str, Any] = {"xi": None if cell.xi is None else float(cell.xi), "xi_kind": kind}
    for label, c in LEVELS:
        if cell.dgp == "S4":
            for m in ("var", "es"):
                ref = reference[f"{m}{label}"]
                out[f"{m}{label}"] = {"value": ref["value"], "rel_se": ref["rel_se"]}
            continue
        if cell.q0 is not None and cell.xi is not None:
            var, es = s1_truth(c, cell.q0, cell.xi)
        else:
            var, es = s2_truth(c, cfg.simstudy.s2_t_df)
        out[f"var{label}"] = {"value": var, "rel_se": None}
        out[f"es{label}"] = {"value": es, "rel_se": None}
    return out


def sample(cell: Cell, g: np.random.Generator, cfg: Config) -> FloatArray:
    """One sample of ``simstudy.sample_size`` losses from the cell's DGP."""
    ss = cfg.simstudy
    n = ss.sample_size
    if cell.dgp == "S2":
        return np.asarray(g.standard_t(ss.s2_t_df, size=n), dtype=np.float64)
    if cell.dgp == "S4":
        return s4_losses(g, n, ss)
    assert cell.q0 is not None and cell.xi is not None
    if cell.a is None:
        return s1_ppf(g.uniform(size=n), cell.q0, cell.xi)
    return s3_losses(g.uniform(size=n), cell.a, cell.q0, cell.xi)


# ---------------------------------------------------------------------------
# One replication (D35 (3))
# ---------------------------------------------------------------------------
def window_refit(x: FloatArray, q_star: Fraction, cfg: Config) -> dict[str, Any]:
    """Refit at ``q*`` on the last ``window_refit`` losses (threshold recomputed, D5).

    VaR/ES use ``n = window_refit`` and the window's actual ``N_u``. A failed fit gives
    ``converged = False`` and no estimates; ES is ``None`` when ``xi_hat >= 1`` and a level
    with ``p >= N_u / n`` has neither VaR nor ES.
    """
    w = x[-cfg.simstudy.window_refit :]
    u = tail_threshold(w, q_star)
    y = excesses(w, u)
    tol = float(cfg.threshold.xi_zero_tol)
    out: dict[str, Any] = {"converged": False, "n_u": int(y.size), "xi": None}
    out.update(dict.fromkeys(RISK_KEYS))
    try:
        fit = fit_mle(y, min_excesses=MIN_EXCESSES, xi_zero_tol=tol, settings=cfg.gpd)
    except (EstimationError, ValueError):
        return out
    out.update(converged=True, xi=fit.xi)
    for label, c in LEVELS:
        p = 1 - c
        if p < Fraction(int(y.size), int(w.size)):
            var, es = tail_risk(u, fit.xi, fit.beta, int(w.size), int(y.size), p, tol)
            out[f"var{label}"] = var
            out[f"es{label}"] = es if math.isfinite(es) else None
    return out


def run_replication(cfg: Config, cell: Cell, rep: int) -> Record:
    """The record of replication ``rep`` of ``cell``: a pure function of its arguments."""
    data_ss, gof_ss = _stream(cfg, cell.index, rep).spawn(2)
    x = sample(cell, np.random.Generator(np.random.PCG64(data_ss)), cfg)
    d = select_threshold(
        x,
        cfg,
        ss=gof_ss,
        gof_reps=cfg.simstudy.gof_bootstrap_reps,
        convergence_share=False,
        halves=False,
        intervals=False,
    )
    rows = d.candidates
    record: Record = {
        "cell": cell.index,
        "rep": rep,
        "k_hat": d.k_hat,
        "v": None if cell.q0 is None else sum(r["q"] >= cell.q0 for r in rows[: d.k_hat]),
        "q_star": None if d.q_star is None else str(d.q_star),
        "rule_failure": d.rule_failure,
        "n_fit_failed": sum("fit_failed" in r["flags"] for r in rows),
        "full": None,
        "refit": None,
    }
    if d.q_star is not None:
        row = next(r for r in rows if r["q"] == d.q_star)
        record["full"] = {"xi": row["xi_hat"], **{k: row[k] for k in RISK_KEYS}}
        record["refit"] = window_refit(x, d.q_star, cfg)
    return record


# ---------------------------------------------------------------------------
# Aggregation (D35 (4)) and review triggers (D35 (5))
# ---------------------------------------------------------------------------
def wilson(k: int, n: int) -> list[float]:
    """Wilson 95% score interval ``[lower, upper]`` for ``k`` successes in ``n`` trials."""
    z2 = WILSON_Z * WILSON_Z
    p = k / n
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [centre - half, centre + half]


def _moments(values: Sequence[float], truth: float | None) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    n = int(v.size)
    out: dict[str, Any] = {
        "n": n,
        "mean": float(v.mean()) if n else None,
        "sd": float(v.std(ddof=1)) if n > 1 else None,
        "bias": None,
        "rmse": None,
    }
    if n and truth is not None:
        out["bias"] = float(v.mean() - truth)
        out["rmse"] = math.sqrt(float(np.mean((v - truth) ** 2)))
    return out


def aggregate_cell(
    cell: Cell, records: Sequence[Record], truth: dict[str, Any], cfg: Config
) -> dict[str, Any]:
    """Metrics M1-M6 of one cell from its replication records (D35 (4)).

    * M1: ``q*`` counts per grid value, rule failures by type, failure rate over ``R``.
    * M2 (cells with a boundary ``q0``): primary false selection
      ``#{q* defined and q* < q0} / R`` and the conditional version over defined ``q*``.
    * M3 (same cells): mean over all ``R`` of ``V / max(1, k_hat)``, exact and as a float,
      with SD (ddof = 1) and ``mean +/- 1.96 SD / sqrt(R)``.
    * M4: mean, SD, bias and RMSE of ``xi_hat(q*)`` over defined ``q*``.
    * M5: relative bias and RMSE of VaR/ES at ``q*`` (full sample and window refit) over
      the replications with an estimate; the rest of the defined ones are ``n_excluded``.
    * M6: ``fit_failed`` share over all (candidate, replication) pairs and, over defined
      ``q*``, refit non-convergence and ``P(xi_hat >= 0.5)``, ``P(xi_hat >= 1)``.
    """
    th = cfg.threshold
    n_rep = len(records)
    defined = [r for r in records if r["q_star"] is not None]
    nd = len(defined)
    counts = {str(q): 0 for q in th.grid}
    for r in defined:
        counts[r["q_star"]] += 1
    out: dict[str, Any] = {
        "cell": cell.index,
        "label": cell.label,
        "replications": n_rep,
        "truth": truth,
        "m1_q_star": {
            "counts": counts,
            "n_defined": nd,
            "all_rejected": sum(r["rule_failure"] == "all_rejected" for r in records),
            "no_stable_region": sum(r["rule_failure"] == "no_stable_region" for r in records),
            "rule_failures": n_rep - nd,
            "rule_failure_rate": (n_rep - nd) / n_rep,
        },
        "m2_false_selection": None,
        "m3_fdr": None,
    }
    if cell.q0 is not None:
        k = sum(Fraction(r["q_star"]) < cell.q0 for r in defined)
        out["m2_false_selection"] = {
            "count": k,
            "primary": k / n_rep,
            "primary_wilson95": wilson(k, n_rep),
            "n_defined": nd,
            "conditional": k / nd if nd else None,
            "conditional_wilson95": wilson(k, nd) if nd else None,
        }
        fdp = [Fraction(r["v"], max(1, r["k_hat"])) for r in records]
        mean = sum(fdp, Fraction(0)) / n_rep
        sd = float(np.std([float(f) for f in fdp], ddof=1))
        half = FDR_Z * sd / math.sqrt(n_rep)
        alpha = th.forwardstop_alpha
        out["m3_fdr"] = {
            "mean": float(mean),
            "mean_exact": str(mean),
            "sd": sd,
            "ci95": [float(mean) - half, float(mean) + half],
            "alpha": str(alpha),
            "exceeds_alpha": mean > alpha,
            "exceeds_2alpha": mean > 2 * alpha,
        }
    m4 = _moments([r["full"]["xi"] for r in defined], truth["xi"])
    out["m4_xi"] = {**m4, "truth": truth["xi"], "truth_kind": truth["xi_kind"]}
    m5: dict[str, Any] = {}
    for key in RISK_KEYS:
        t = truth[key]["value"]
        m5[key] = {"truth": t, "truth_rel_se": truth[key]["rel_se"]}
        for part, name in (("full", "full"), ("refit", "window")):
            rel = [(r[part][key] - t) / t for r in defined if r[part][key] is not None]
            m = _moments(rel, 0.0)
            m5[key][name] = {
                "rel_bias": m["bias"],
                "rel_rmse": m["rmse"],
                "n_used": m["n"],
                "n_excluded": nd - m["n"],
            }
    out["m5_risk"] = m5

    def share(part: str, level: float) -> float | None:
        hits = sum(r[part]["xi"] is not None and r[part]["xi"] >= level for r in defined)
        return hits / nd if nd else None

    failed = sum(r["n_fit_failed"] for r in records)
    not_converged = sum(not r["refit"]["converged"] for r in defined)
    out["m6_failures"] = {
        "pairs": n_rep * len(th.grid),
        "fit_failed_pairs": failed,
        "fit_failed_share": failed / (n_rep * len(th.grid)),
        "full_nonconverged_at_q_star": sum(r["full"]["xi"] is None for r in defined),
        "window_nonconverged": not_converged,
        "window_nonconvergence_rate": not_converged / nd if nd else None,
        "full_p_xi_ge_0_5": share("full", 0.5),
        "full_p_xi_ge_1": share("full", 1.0),
        "window_p_xi_ge_0_5": share("refit", 0.5),
        "window_p_xi_ge_1": share("refit", 1.0),
    }
    return out


def triggers(
    study_cells: Sequence[Cell], metrics: Sequence[dict[str, Any]], cfg: Config
) -> list[dict[str, Any]]:
    """The D13 review triggers for every S3 cell (D35 (5)). Reported, never acted on.

    Fired iff, strictly and in exact rational arithmetic: primary false selection
    ``> trigger_false_selection``; FDR mean ``> trigger_fdr_multiple * alpha_FS``; rule
    failure rate ``> trigger_rule_failure``. ``near_miss`` (only when not fired): the
    conditional false selection, or the upper end of the FDR interval, is above the limit.
    """
    ss = cfg.simstudy
    fdr_limit = ss.trigger_fdr_multiple * cfg.threshold.forwardstop_alpha
    out: list[dict[str, Any]] = []
    for cell, m in zip(study_cells, metrics, strict=True):
        if cell.dgp != "S3":
            continue
        n_rep = m["replications"]
        m1, m2, m3 = m["m1_q_star"], m["m2_false_selection"], m["m3_fdr"]
        fs_fired = Fraction(m2["count"], n_rep) > ss.trigger_false_selection
        nd = m2["n_defined"]
        conditional_over = nd > 0 and Fraction(m2["count"], nd) > ss.trigger_false_selection
        fdr_fired = Fraction(m3["mean_exact"]) > fdr_limit
        rf_fired = Fraction(m1["rule_failures"], n_rep) > ss.trigger_rule_failure
        out.append(
            {
                "cell": cell.index,
                "label": cell.label,
                "false_selection": {
                    "value": m2["primary"],
                    "wilson95": m2["primary_wilson95"],
                    "conditional": m2["conditional"],
                    "conditional_wilson95": m2["conditional_wilson95"],
                    "limit": str(ss.trigger_false_selection),
                    "fired": fs_fired,
                    "near_miss": not fs_fired and conditional_over,
                },
                "fdr": {
                    "value": m3["mean"],
                    "ci95": m3["ci95"],
                    "limit": str(fdr_limit),
                    "fired": fdr_fired,
                    "near_miss": not fdr_fired and m3["ci95"][1] > fdr_limit,
                },
                "rule_failure": {
                    "value": m1["rule_failure_rate"],
                    "limit": str(ss.trigger_rule_failure),
                    "fired": rf_fired,
                },
                "any_fired": fs_fired or fdr_fired or rf_fired,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Runner (D35 (6), (8))
# ---------------------------------------------------------------------------
def check_design(cfg: Config) -> None:
    """Raise ``ValueError`` unless the study can run with this configuration."""
    ss = cfg.simstudy
    if ss.sample_size != cfg.design.train_size:
        raise ValueError("simstudy.sample_size must equal design.train_size (selection guard)")
    if ss.replications < 2:
        raise ValueError("simstudy.replications must be >= 2")
    if ss.s4_reference_size % S4_REFERENCE_BATCHES:
        raise ValueError("simstudy.s4_reference_size must split into the reference batches")
    if cfg.threshold.n_min is None:
        raise ValueError("threshold.n_min is not set (Day-6 gate)")


def default_workers() -> int:
    """``min(cpu_count - 1, 31)``, at least 1."""
    return max(1, min((os.cpu_count() or 2) - 1, MAX_WORKERS))


def project_seconds(pilot_seconds: float, pilot_replications: int, replications: int) -> float:
    """Runtime-guard projection (D35 (8)): ``pilot_seconds / pilot_replications * R``."""
    if pilot_replications < 1 or replications < 1 or not pilot_seconds >= 0:
        raise ValueError("projection: needs positive replication counts and a time >= 0")
    return pilot_seconds / pilot_replications * replications


def _task(task: tuple[Config, Cell, int]) -> Record:
    return run_replication(*task)


def run_records(cfg: Config, replications: int, workers: int) -> list[Record]:
    """Records of replications ``0..replications-1`` of every cell, sorted by (cell, rep).

    One task per (cell, replication); ``workers == 1`` runs in this process, otherwise in
    a ``ProcessPoolExecutor``. The result does not depend on ``workers``.
    """
    if workers < 1:
        raise ValueError("workers: must be >= 1")
    tasks = [(cfg, cell, rep) for cell in cells(cfg) for rep in range(replications)]
    if workers == 1:
        records = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_task, tasks))
    return sorted(records, key=lambda r: (r["cell"], r["rep"]))


def records_sha256(records: Sequence[Record]) -> str:
    """SHA-256 of the canonical records: one sorted-key JSON line each, LF, in order."""
    lines = (json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def summarise(
    cfg: Config, records: Sequence[Record], reference: dict[str, Any], *, config_sha256: str
) -> dict[str, Any]:
    """The study summary (no timestamps, runtime or hardware): design, cells, metrics, triggers."""
    ss, th = cfg.simstudy, cfg.threshold
    study_cells = cells(cfg)
    metrics = [
        aggregate_cell(
            cell,
            [r for r in records if r["cell"] == cell.index],
            cell_truth(cell, cfg, reference),
            cfg,
        )
        for cell in study_cells
    ]
    return {
        "label": LABEL,
        "decision": "D35",
        "config_sha256": config_sha256,
        "design": {
            "replications": ss.replications,
            "gof_bootstrap_reps": ss.gof_bootstrap_reps,
            "sample_size": ss.sample_size,
            "window_refit": ss.window_refit,
            "grid": [str(q) for q in th.grid],
            "levels": [str(c) for _, c in LEVELS],
            "alpha_fs": str(th.forwardstop_alpha),
            "q_max": str(1 - Fraction(th.n_min or 0, cfg.design.window)),
        },
        "seeds": {
            "seed": cfg.runtime.seed,
            "children": len(study_cells) + 1,
            "cell": "SeedSequence(seed).spawn(children)[cell]",
            "s4_reference": f"SeedSequence(seed).spawn(children)[{len(study_cells)}]",
            "replication": "cell.spawn(R)[rep]",
            "data": "PCG64(replication.spawn(2)[0])",
            "gof": "replication.spawn(2)[1], passed to select_threshold",
        },
        "cells": [
            {
                "index": c.index,
                "dgp": c.dgp,
                "label": c.label,
                "q0": None if c.q0 is None else str(c.q0),
                "xi": None if c.xi is None else str(c.xi),
                "a": None if c.a is None else str(c.a),
            }
            for c in study_cells
        ],
        "s4_reference": reference,
        "metrics": metrics,
        "triggers": triggers(study_cells, metrics, cfg),
        "records_sha256": records_sha256(records),
    }


def run_study(
    cfg: Config, *, config_sha256: str, workers: int
) -> tuple[dict[str, Any], list[Record]]:
    """Run the full study: ``(summary, records)``. The records are not stored anywhere."""
    check_design(cfg)
    reference = s4_reference(cfg)
    records = run_records(cfg, cfg.simstudy.replications, workers)
    return summarise(cfg, records, reference, config_sha256=config_sha256), records


def summary_json(summary: dict[str, Any]) -> str:
    """Deterministic serialisation: sorted keys, indent 2, LF, trailing newline, no NaN."""
    return json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"


# ---------------------------------------------------------------------------
# Figures and outputs (D35 (9))
# ---------------------------------------------------------------------------
def write_figures(summary: dict[str, Any], records: Sequence[Record], fig_dir: Path) -> list[Path]:
    """Write the five study figures as deterministic SVG files (aggregates of synthetic data).

    (a) ``q*`` histogram per cell with ``q0`` and the rule-failure count; (b) empirical FDR
    against ``alpha_FS`` and ``2 alpha_FS``; (c) ``xi_hat(q*) - xi``; (d) relative VaR/ES
    errors, full sample and window refit; (e) failure rates. No date in the metadata, a
    fixed hash salt and LF line endings make reruns byte-identical.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    info, metrics = summary["cells"], summary["metrics"]
    names = [c["label"] for c in info]
    grid = [float(Fraction(q)) for q in summary["design"]["grid"]]
    alpha = float(Fraction(summary["design"]["alpha_fs"]))
    by_cell = [[r for r in records if r["cell"] == c["index"]] for c in info]
    out: list[Path] = []

    def save(fig: Figure, name: str) -> None:
        fig.suptitle(LABEL, fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
        plt.close(fig)
        (fig_dir / name).write_bytes(buf.getvalue().replace(b"\r\n", b"\n"))
        out.append(fig_dir / name)

    def boxes(ax: Any, groups: list[list[float]], offset: float, colour: str) -> None:
        for i, values in enumerate(groups):
            if values:
                ax.boxplot(
                    [values],
                    positions=[i + offset],
                    widths=0.3,
                    flierprops={"markersize": 2},
                    medianprops={"color": colour},
                    manage_ticks=False,
                )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)

    fig_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT, "svg.fonttype": "path"}):
        cols = -(-len(info) // 2)
        fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6), sharey=True, squeeze=False)
        for ax, c, m in zip(axes.flat, info, metrics, strict=False):
            m1 = m["m1_q_star"]
            ax.bar(grid, list(m1["counts"].values()), width=0.008)
            if c["q0"] is not None:
                ax.axvline(float(Fraction(c["q0"])), color="tab:red", linestyle="--")
            ax.set_title(
                f"{c['label']}\nrule failures {m1['rule_failures']}/{m['replications']}",
                fontsize=8,
            )
            ax.set_xlabel("q*", fontsize=8)
        save(fig, FIGURE_NAMES[0])

        fig, ax = plt.subplots(figsize=(8, 4))
        fdr = [(n, m["m3_fdr"]) for n, m in zip(names, metrics, strict=True) if m["m3_fdr"]]
        ax.errorbar(
            range(len(fdr)),
            [f["mean"] for _, f in fdr],
            yerr=[f["ci95"][1] - f["mean"] for _, f in fdr],
            fmt="o",
            capsize=3,
        )
        ax.axhline(alpha, color="tab:orange", linestyle="--", label="alpha_FS")
        ax.axhline(2 * alpha, color="tab:red", linestyle="--", label="2 alpha_FS")
        ax.set_xticks(range(len(fdr)))
        ax.set_xticklabels([n for n, _ in fdr], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("empirical FDR (mean, 1.96 SD/sqrt(R))")
        ax.legend(fontsize=8)
        save(fig, FIGURE_NAMES[1])

        fig, ax = plt.subplots(figsize=(8, 4))
        boxes(
            ax,
            [
                [r["full"]["xi"] - m["truth"]["xi"] for r in rs if r["full"] is not None]
                if m["truth"]["xi"] is not None
                else []
                for rs, m in zip(by_cell, metrics, strict=True)
            ],
            0.0,
            "tab:blue",
        )
        ax.set_ylabel("xi_hat(q*) - xi (S2: asymptotic xi; S4: no truth)")
        save(fig, FIGURE_NAMES[2])

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
        for ax, key in zip(axes.flat, RISK_KEYS, strict=True):
            for part, offset, colour in (("full", -0.18, "tab:blue"), ("refit", 0.18, "tab:red")):
                boxes(
                    ax,
                    [
                        [
                            (r[part][key] - m["truth"][key]["value"]) / m["truth"][key]["value"]
                            for r in rs
                            if r[part] is not None and r[part][key] is not None
                        ]
                        for rs, m in zip(by_cell, metrics, strict=True)
                    ],
                    offset,
                    colour,
                )
            ax.set_title(
                f"{key}: relative error (blue: full sample, red: window refit)", fontsize=9
            )
        save(fig, FIGURE_NAMES[3])

        fig, ax = plt.subplots(figsize=(10, 4))
        series = (
            ("rule failure", [m["m1_q_star"]["rule_failure_rate"] for m in metrics]),
            (
                "refit not converged",
                [m["m6_failures"]["window_nonconvergence_rate"] for m in metrics],
            ),
            ("xi_hat >= 0.5 (full)", [m["m6_failures"]["full_p_xi_ge_0_5"] for m in metrics]),
            ("xi_hat >= 1 (full)", [m["m6_failures"]["full_p_xi_ge_1"] for m in metrics]),
        )
        for j, (name, values) in enumerate(series):
            ax.bar(
                [i + 0.2 * (j - 1.5) for i in range(len(names))],
                [v or 0.0 for v in values],
                width=0.2,
                label=name,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("rate")
        ax.legend(fontsize=8)
        save(fig, FIGURE_NAMES[4])
    return out


def write_outputs(summary: dict[str, Any], records: Sequence[Record], out_dir: Path) -> list[Path]:
    """Write ``metrics.json`` and the figures into ``out_dir``; returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / METRICS_NAME
    path.write_bytes(summary_json(summary).encode("utf-8"))
    return [path, *write_figures(summary, records, out_dir)]
