"""Day-6 N_min simulation study (D32; methodology 7 Step 0; D4d). Synthetic data only.

Design (configs ``[nmin_study]``): true xi x N_u cells, beta = 1, R replications per cell.
Cell ``k = len(n_u) * i_xi + i_nu`` draws all its replications sequentially from
``Generator(PCG64(SeedSequence(seed).spawn(n_cells)[k]))``; execution is serial, so the
results cannot depend on a worker count.

Quantity: the VaR excess quantile ``x_p = tail_var(u=0, xi, beta, n=W, n_u=N_u, p)`` with
``p = 1 - var_level`` exactly; relative error ``(x_hat - x_p) / x_p`` over converged fits.

Rule (pre-declared): N_min is the smallest N_u such that at every xi the convergence rate is
>= min_convergence, RMSE(xi_hat) <= max_rmse_xi and the relative RMSE of x_p is
<= max_rel_rmse_var99. No qualifying N_u is a rule failure (``n_min`` = null).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from niftytail.config import Config, GpdConfig
from niftytail.gpd import EstimationError, fit_mle, tail_var

__all__ = [
    "NMIN_OUTPUT",
    "adopted_block",
    "apply_rule",
    "nmin_cell",
    "run_nmin_study",
    "sample_gpd_excesses",
    "summary_json",
]

NMIN_OUTPUT = Path("reports/threshold_selection/nmin_study.json")
FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _key(xi: float | Fraction) -> str:
    return repr(float(xi))


def sample_gpd_excesses(g: np.random.Generator, xi: float, beta: float, n: int) -> FloatArray:
    """``n`` i.i.d. GPD excesses by inversion ``(beta/xi) expm1(-xi log1p(-U))``."""
    u = g.uniform(size=n)
    if xi == 0:
        return -beta * np.log1p(-u)
    return (beta / xi) * np.expm1(-xi * np.log1p(-u))


def nmin_cell(
    xi: float,
    n_u: int,
    replications: int,
    g: np.random.Generator,
    *,
    beta: float,
    window: int,
    p: Fraction,
    xi_zero_tol: float,
    settings: GpdConfig,
) -> dict[str, Any]:
    """Aggregates of one (xi, N_u) cell; statistics are over converged fits only.

    Secondary, descriptive only (D33; never enters the rule): the relative error on a
    VaR *level* ``(x_hat - x_p) / (s beta + x_p)``, where ``s = u_ref / beta_ref`` is the
    normal reference threshold in GPD-scale units: ``u_ref = Phi^-1(1 - N_u/W)`` and
    ``beta_ref = (1 - Phi(u_ref)) / phi(u_ref)`` (Mills ratio).
    """
    x_p = tail_var(0.0, xi, beta, window, n_u, p, xi_zero_tol)
    u_ref = float(stats.norm.ppf(float(1 - Fraction(n_u, window))))
    s_ref = u_ref / (float(stats.norm.sf(u_ref)) / float(stats.norm.pdf(u_ref)))
    xi_hat: list[float] = []
    rel_err: list[float] = []
    level_err: list[float] = []
    irregular = errors = 0
    for _ in range(replications):
        y = sample_gpd_excesses(g, xi, beta, n_u)
        try:
            f = fit_mle(y, min_excesses=2, xi_zero_tol=xi_zero_tol, settings=settings)
        except EstimationError:
            errors += 1
            continue
        x_hat = tail_var(0.0, f.xi, f.beta, window, n_u, p, xi_zero_tol)
        xi_hat.append(f.xi)
        rel_err.append((x_hat - x_p) / x_p)
        level_err.append((x_hat - x_p) / (s_ref * beta + x_p))
        irregular += not f.regular
    conv = len(xi_hat)
    est, rel, lev = np.array(xi_hat), np.array(rel_err), np.array(level_err)
    return {
        "xi": float(xi),
        "n_u": n_u,
        "replications": replications,
        "converged": conv,
        "convergence_rate": conv / replications,
        "n_estimation_error": errors,
        "bias_xi": float(est.mean() - xi) if conv else None,
        "rmse_xi": math.sqrt(float(np.mean((est - xi) ** 2))) if conv else None,
        "rel_rmse_var": math.sqrt(float(np.mean(rel**2))) if conv else None,
        "p_xi_ge_0_5": float(np.mean(est >= 0.5)) if conv else None,
        "p_xi_ge_1": float(np.mean(est >= 1.0)) if conv else None,
        "p_irregular": irregular / conv if conv else None,
        "u_over_beta_ref": s_ref,
        "rel_rmse_var_level_ref": math.sqrt(float(np.mean(lev**2))) if conv else None,
    }


def apply_rule(
    cells: Sequence[dict[str, Any]],
    *,
    n_u_grid: Sequence[int],
    xi_grid: Sequence[Fraction],
    min_convergence: Fraction,
    max_rmse_xi: Fraction,
    max_rel_rmse: Fraction,
) -> tuple[dict[str, Any], int | None]:
    """Per-N_u criterion table and N_min (smallest qualifying N_u, or None = rule failure).

    Comparisons are exact (floats against Fractions); ties pass (``>=`` / ``<=``).
    """
    lookup = {(c["xi"], c["n_u"]): c for c in cells}
    table: dict[str, Any] = {}
    n_min: int | None = None
    for n_u in n_u_grid:
        per: dict[str, Any] = {}
        for xi in xi_grid:
            c = lookup[(float(xi), n_u)]
            conv = Fraction(c["converged"], c["replications"]) >= min_convergence
            rm = c["rmse_xi"] is not None and c["rmse_xi"] <= max_rmse_xi
            rv = c["rel_rmse_var"] is not None and c["rel_rmse_var"] <= max_rel_rmse
            per[_key(xi)] = {"convergence": conv, "rmse_xi": rm, "rel_rmse_var": rv}
        ok = all(all(v.values()) for v in per.values())
        table[str(n_u)] = {"per_xi": per, "qualifies": ok}
        if ok and n_min is None:
            n_min = n_u
    return table, n_min


def run_nmin_study(cfg: Config, *, config_sha256: str) -> dict[str, Any]:
    """Run every cell serially and apply the rule; returns the aggregate-only summary."""
    ns = cfg.nmin_study
    p = 1 - ns.var_level
    window = cfg.design.window
    tol = float(cfg.threshold.xi_zero_tol)
    children = np.random.SeedSequence(cfg.runtime.seed).spawn(len(ns.xi_true) * len(ns.n_u))
    cells = [
        nmin_cell(
            float(xi),
            n_u,
            ns.replications,
            np.random.Generator(np.random.PCG64(children[len(ns.n_u) * i + j])),
            beta=float(ns.beta),
            window=window,
            p=p,
            xi_zero_tol=tol,
            settings=cfg.gpd,
        )
        for i, xi in enumerate(ns.xi_true)
        for j, n_u in enumerate(ns.n_u)
    ]
    table, n_min = apply_rule(
        cells,
        n_u_grid=ns.n_u,
        xi_grid=ns.xi_true,
        min_convergence=ns.min_convergence,
        max_rmse_xi=ns.max_rmse_xi,
        max_rel_rmse=ns.max_rel_rmse_var99,
    )
    q_max = None if n_min is None else 1 - Fraction(n_min, window)
    return {
        "decision": "D32",
        "config_sha256": config_sha256,
        "design": {
            "xi": [_key(x) for x in ns.xi_true],
            "beta": float(ns.beta),
            "n_u": list(ns.n_u),
            "replications": ns.replications,
            "seed": cfg.runtime.seed,
            "window": window,
            "var_level": str(ns.var_level),
        },
        "cells": cells,
        "rule": {
            "criteria": {
                "min_convergence": str(ns.min_convergence),
                "max_rmse_xi": str(ns.max_rmse_xi),
                "max_rel_rmse_var": str(ns.max_rel_rmse_var99),
            },
            "by_n_u": table,
            "n_min": n_min,
            "q_max": None if q_max is None else float(q_max),
            "q_max_exact": None if q_max is None else str(q_max),
        },
    }


def adopted_block(
    rule_n_min: int | None, config_n_min: int | None, override: str | None, window: int
) -> dict[str, Any] | None:
    """The N_min actually adopted (D33), or None if neither the rule nor the config sets one.

    Raises:
        ValueError: if a rule result exists and the config n_min is unset or different, or
            the config sets n_min after a rule failure without an ``override_decision``.
    """
    if rule_n_min is not None:
        if config_n_min != rule_n_min:
            raise ValueError(
                "threshold.n_min in the config disagrees with the study rule (unset or different)"
            )
        n_min, source = rule_n_min, "rule"
    elif config_n_min is None:
        return None
    elif override is None:
        raise ValueError("rule failure: threshold.n_min needs nmin_study.override_decision")
    else:
        n_min, source = config_n_min, override
    return {
        "n_min": n_min,
        "source": source,
        "rule_n_min": rule_n_min,
        "q_max": str(1 - Fraction(n_min, window)),
    }


def summary_json(summary: dict[str, Any]) -> str:
    """Deterministic serialisation: sorted keys, indent 2, LF, trailing newline, no NaN."""
    return json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
