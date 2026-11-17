"""One-day-ahead VaR/ES forecasts of the six models (D40; methodology 4-6, 18). Loss space.

Unconditional, on the raw window losses: ``gaussian`` (§5), ``hs`` (§4, A1), ``pot_gpd``
(§6). Conditional, on the standardized residuals of ONE filter fit per window (spec from
``[conditional] filter``, D39): GARCH-N, FHS and GARCH-EVT, all ``mu_next + sigma_next x``
the residual-scale risk, with FHS = :func:`hs` and GARCH-EVT = :func:`pot_gpd` applied to
the residuals (the same code paths). One filter fit can serve several GARCH-EVT levels.

Forecast functions see only their arguments: a window, the pre-window lag, the config and
a q level. Never a date or a series beyond the window. No randomness, no warm starts.
Missing values are NaN with a flag; ES is +inf only when the GPD shape is >= 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt

from niftytail.config import Config
from niftytail.garch import (
    SPECS,
    GarchError,
    fit_garch,
    forecast_mean,
    forecast_sigma,
    garch_filter,
)
from niftytail.gpd import EstimationError, excesses, fit_mle, tail_es, tail_var
from niftytail.losses import tail_threshold
from niftytail.threshold import MIN_EXCESSES

__all__ = [
    "MODELS",
    "Forecast",
    "conditional",
    "evt_key",
    "forecast_all",
    "gaussian",
    "hs",
    "levels",
    "pot_gpd",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
MODELS: dict[str, str] = {  # stable name -> display label (D40 (e))
    "gaussian": "Gaussian",
    "hs": "HS",
    "pot_gpd": "POT-GPD",
    "garch_n": "GARCH-N",
    "fhs": "FHS",
    "garch_evt": "GARCH-EVT",
}
_NAN = math.nan


@dataclass(frozen=True, slots=True)
class Forecast:
    """VaR and ES per confidence level (Fraction keys), sorted flags and fit diagnostics."""

    model: str
    var: dict[Fraction, float]
    es: dict[Fraction, float]
    flags: tuple[str, ...] = ()
    diagnostics: dict[str, float] = field(default_factory=dict)


def levels(cfg: Config) -> tuple[Fraction, ...]:
    """The forecast levels: ``[levels]`` primary then descriptive."""
    return cfg.levels.primary + cfg.levels.descriptive


def _missing(model: str, cfg: Config, flags: set[str], diag: dict[str, float]) -> Forecast:
    nan = dict.fromkeys(levels(cfg), _NAN)
    return Forecast(model, nan, dict(nan), tuple(sorted(flags)), diag)


def _sorted(window: npt.ArrayLike) -> FloatArray:
    """The window sorted ascending (the unconditional models depend on order statistics only)."""
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("window: must be one-dimensional, finite, with at least 2 values")
    return np.sort(x)


def hs(window: npt.ArrayLike, cfg: Config) -> Forecast:
    """Historical simulation (A1): ``VaR = L_(ceil(W c))``; ES with exact ``m = W p``.

    ``ES = [(m - floor(m)) L_(W - floor(m)) + sum of the floor(m) largest] / m``.
    """
    x = _sorted(window)
    w = x.size
    var: dict[Fraction, float] = {}
    es: dict[Fraction, float] = {}
    for c in levels(cfg):
        m = w * (1 - c)
        k = math.floor(m)
        rank = -((-w * c.numerator) // c.denominator)  # ceil(W c), integer arithmetic
        var[c] = float(x[rank - 1])
        es[c] = math.fsum([float(m - k) * x[w - k - 1], *x[w - k :]]) / float(m)
    return Forecast("hs", var, es)


def _normal(c: Fraction) -> tuple[float, float]:
    """Standard-normal ``(VaR_z, ES_z) = (z_c, phi(z_c) / p)`` (§5 at mu = 0, sigma = 1)."""
    z = NormalDist().inv_cdf(float(c))
    return z, NormalDist().pdf(z) / float(1 - c)


def gaussian(window: npt.ArrayLike, cfg: Config) -> Forecast:
    """Gaussian benchmark (§5): ``mu + sigma z_c`` and ``mu + sigma phi(z_c) / p``, ddof = 1."""
    x = _sorted(window)
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    var: dict[Fraction, float] = {}
    es: dict[Fraction, float] = {}
    for c in levels(cfg):
        z, e = _normal(c)
        var[c], es[c] = mu + sigma * z, mu + sigma * e
    return Forecast("gaussian", var, es, (), {"mu": mu, "sigma": sigma})


def pot_gpd(window: npt.ArrayLike, cfg: Config, q: Fraction) -> Forecast:
    """POT-GPD (§6): GPD MLE on the excesses over ``u = L_(ceil(W q))`` (D29), tail VaR/ES.

    A level with ``p >= N_u / W`` is NaN with "var_not_in_tail"; a failed fit (or no
    excess) is NaN everywhere with "gpd_error"; an infinite ES is flagged "es_infinite".
    """
    x = _sorted(window)
    w = x.size
    u = tail_threshold(x, q)
    y = excesses(x, u)  # ascending, because x is sorted
    n_u = int(y.size)
    tol = float(cfg.threshold.xi_zero_tol)
    diag = {"u": u, "n_u": float(n_u), "xi": _NAN, "beta": _NAN}
    try:
        if n_u == 0:
            raise EstimationError("no excess over the threshold")
        fit = fit_mle(y, min_excesses=MIN_EXCESSES, xi_zero_tol=tol, settings=cfg.gpd)
    except EstimationError:
        return _missing("pot_gpd", cfg, {"gpd_error"}, diag)
    diag.update(xi=fit.xi, beta=fit.beta)
    flags: set[str] = set()
    var: dict[Fraction, float] = {}
    es: dict[Fraction, float] = {}
    for c in levels(cfg):
        p = 1 - c
        if not p < Fraction(n_u, w):
            var[c] = es[c] = _NAN
            flags.add("var_not_in_tail")
            continue
        var[c] = tail_var(u, fit.xi, fit.beta, w, n_u, p, tol)
        es[c] = tail_es(var[c], u, fit.xi, fit.beta)
        if math.isinf(es[c]):
            flags.add("es_infinite")
    return Forecast("pot_gpd", var, es, tuple(sorted(flags)), diag)


def _scaled(
    model: str, z: Forecast, mu: float, sigma: float, flags: set[str], diag: dict[str, float]
) -> Forecast:
    """``mu + sigma x`` the residual-scale forecast ``z``; NaN and +inf pass through."""
    return Forecast(
        model,
        {c: mu + sigma * v for c, v in z.var.items()},
        {c: mu + sigma * v for c, v in z.es.items()},
        tuple(sorted(flags | set(z.flags))),
        diag,
    )


def evt_key(q: Fraction) -> str:
    """Key of the GARCH-EVT forecast at ``q`` in a multi-q :func:`conditional` result."""
    return f"garch_evt@{Decimal(q.numerator) / Decimal(q.denominator)}"


def conditional(
    window: npt.ArrayLike,
    pre_lag: float | None,
    cfg: Config,
    q_c: Fraction | tuple[Fraction, ...],
) -> dict[str, Forecast]:
    """GARCH-N, FHS and GARCH-EVT from ONE fit of the ``[conditional] filter`` spec (§18).

    ``pre_lag`` is the observation just before the window, the lag of the first residual of
    an AR spec (D39 (d)); it is ignored by a constant-mean spec. A single ``q_c`` gives the
    keys "garch_n", "fhs", "garch_evt". A tuple of levels gives "garch_n", "fhs" and one
    GARCH-EVT forecast per level under :func:`evt_key`, all from the same filter fit.

    Raises:
        ValueError: if a tuple ``q_c`` is empty or has a repeated level.
    """
    qs = (q_c,) if isinstance(q_c, Fraction) else q_c
    if not qs or len(set(qs)) != len(qs):
        raise ValueError("q_c: needs at least one level and no repeated level")
    keys = ["garch_evt"] if isinstance(q_c, Fraction) else [evt_key(q) for q in qs]
    spec = cfg.conditional.filter
    x = np.asarray(window, dtype=np.float64)

    def missing(flag: str) -> dict[str, Forecast]:
        names = [("garch_n", "garch_n"), ("fhs", "fhs"), *(("garch_evt", k) for k in keys)]
        return {k: _missing(n, cfg, {flag}, {}) for n, k in names}

    if SPECS[spec][0] and pre_lag is None:
        return missing("no_pre_window_lag")
    try:
        fit = fit_garch(x, cfg.garch, spec, lag=pre_lag)
    except GarchError:
        return missing("garch_error")
    eps, s2, z = garch_filter(x, fit, pre_lag)
    mu, sigma = forecast_mean(fit, float(x[-1])), forecast_sigma(fit, float(eps[-1]), float(s2[-1]))
    flags = set(fit.flags)
    diag = {
        "mu_next": mu,
        "sigma_next": sigma,
        "filter_mu": fit.mu,
        "filter_phi1": fit.phi1,
        "filter_omega": fit.omega,
        "filter_alpha": fit.alpha,
        "filter_gamma": fit.gamma,
        "filter_beta": fit.beta,
        "filter_nll": fit.nll,
        "filter_start": float(fit.start),
    }
    normal = [(c, *_normal(c)) for c in levels(cfg)]
    out = {
        "garch_n": Forecast(
            "garch_n",
            {c: mu + sigma * v for c, v, _ in normal},
            {c: mu + sigma * e for c, _, e in normal},
            tuple(sorted(flags)),
            diag,
        ),
        "fhs": _scaled("fhs", hs(z, cfg), mu, sigma, flags, dict(diag)),
    }
    for key, q in zip(keys, qs, strict=True):
        evt_z = pot_gpd(z, cfg, q)
        d = evt_z.diagnostics
        evt_diag = {**diag, "u_z": d["u"], "n_u": d["n_u"], "xi": d["xi"], "beta": d["beta"]}
        out[key] = _scaled("garch_evt", evt_z, mu, sigma, flags, evt_diag)
    return out


def forecast_all(
    window: npt.ArrayLike,
    pre_lag: float | None,
    cfg: Config,
    q_star: Fraction,
    q_c: Fraction,
) -> dict[str, Forecast]:
    """The six forecasts, keyed and ordered as :data:`MODELS`."""
    out = {
        "gaussian": gaussian(window, cfg),
        "hs": hs(window, cfg),
        "pot_gpd": pot_gpd(window, cfg, q_star),
        **conditional(window, pre_lag, cfg, q_c),
    }
    return {name: out[name] for name in MODELS}
