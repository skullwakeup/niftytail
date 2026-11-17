"""GARCH-family filters, Gaussian QMLE (D38, D39; methodology 18).

Four specifications (D39 (c)), named by ``SPECS``: F1 constant mean with GARCH(1,1) (the D38
filter and the default everywhere), F2 AR(1) with GARCH(1,1), F3 constant mean with
GJR(1,1), F4 AR(1) with GJR(1,1).

Loss space: ``L_t = mu_t + eps_t``, ``eps_t = sigma_t z_t``, with ``mu_t = mu`` (constant)
or ``mu_t = mu + phi1 L_{t-1}`` (AR(1); ``mu`` is then the intercept ``phi0``), and
``sigma2_t = omega + (alpha + gamma 1{eps_{t-1} > 0}) eps2_{t-1} + beta sigma2_{t-1}`` for
``t >= 2`` (``gamma = 0`` without GJR; ``eps > 0`` is a loss above its conditional mean).
Backcast ``sigma2_1 = mean(eps_t^2)`` over the window. An AR(1) window of ``W`` losses needs
ONE pre-window observation, ``lag``, for its first residual, so every window has ``W``
residuals (D39 (d)); ``lag`` is ignored by the constant-mean specs.

``NLL = 0.5 * sum(ln 2pi + ln sigma2_t + eps2_t / sigma2_t)`` over the window, minimised by
Nelder-Mead over ``theta = (t, a, b1, b2[, g][, p])``: with ``m`` and ``v`` the window mean
and variance (ddof = 0), ``phi1 = tanh(p)``, ``mu = (m + sqrt(v) t) (1 - phi1)``,
``omega = v exp(a)`` and ``(alpha, gamma / 2, beta, rest) = softmax(b1, g, b2, 0)``. The
constraints ``omega > 0``, ``alpha, gamma, beta >= 0``, ``alpha + gamma / 2 + beta < 1`` and
``|phi1| < 1`` hold by construction. Pinned starts; the usable one with the lowest NLL is
kept. Nothing is clipped or repaired: a failed fit raises :class:`GarchError`. Error
messages never contain data values (D23).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.signal import lfilter

from niftytail.config import GarchConfig

__all__ = [
    "ALPHA_FLOOR",
    "GJR_STARTS",
    "PERSISTENCE_CEILING",
    "SPECS",
    "STARTS",
    "GarchError",
    "GarchFit",
    "bic",
    "constrain",
    "fit_garch",
    "forecast_mean",
    "forecast_sigma",
    "garch_filter",
    "garch_nll",
    "n_params",
    "select_filter",
    "unconstrain",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
SPECS: dict[str, tuple[bool, bool]] = {  # name -> (AR(1) mean, GJR variance), D39 (c)
    "F1": (False, False),
    "F2": (True, False),
    "F3": (False, True),
    "F4": (True, True),
}
STARTS: tuple[tuple[float, float], ...] = ((0.05, 0.90), (0.10, 0.85))  # (alpha0, beta0), D38
GJR_STARTS: tuple[tuple[float, float, float], ...] = (  # (alpha0, gamma0, beta0), D39 (c)
    (0.03, 0.08, 0.88),
    (0.05, 0.10, 0.83),
)
PERSISTENCE_CEILING = 0.999  # "near_boundary" if alpha + gamma/2 + beta is above this
ALPHA_FLOOR = 1e-6  # "near_boundary" if alpha is below this (report only)
_LOG_2PI = math.log(2.0 * math.pi)


class GarchError(Exception):
    """The fit failed (non-finite or constant input, no usable start, bad forecast)."""


@dataclass(frozen=True, slots=True)
class GarchFit:
    """Result of :func:`fit_garch`. ``start`` is the index of the kept optimum's start.

    ``mu`` is the constant mean, or the AR(1) intercept ``phi0`` when ``spec`` has an AR
    mean. ``phi1`` and ``gamma`` are exactly 0 where the spec has no such term. The variance
    path is not stored; :func:`garch_filter` recomputes it.
    """

    mu: float
    omega: float
    alpha: float
    beta: float
    nll: float
    start: int
    flags: tuple[str, ...]
    phi1: float = 0.0
    gamma: float = 0.0
    spec: str = "F1"


def n_params(spec: str) -> int:
    """Number of free parameters ``k`` of ``spec``: 4, 5, 5, 6 for F1...F4 (D39 (c))."""
    ar, gjr = SPECS[spec]
    return 4 + ar + gjr


def _expand(
    theta: npt.ArrayLike, mean: float, var: float, spec: str
) -> tuple[float, float, float, float, float, float]:
    """``theta`` to ``(mu, omega, alpha, beta, gamma, phi1)``; zeros where the spec has no term."""
    ar, gjr = SPECS[spec]
    values = [float(x) for x in np.asarray(theta, dtype=np.float64)]
    t, a, b1, b2 = values[:4]
    g = values[4] if gjr else -math.inf
    phi1 = math.tanh(values[-1]) if ar else 0.0
    top = max(b1, b2, 0.0, g)
    e1, e2, e0, eg = math.exp(b1 - top), math.exp(b2 - top), math.exp(-top), math.exp(g - top)
    total = e1 + e2 + e0 + eg
    omega = var * math.exp(a) if a < 709.0 else math.inf
    mu = (mean + math.sqrt(var) * t) * (1.0 - phi1)
    return mu, omega, e1 / total, e2 / total, 2.0 * eg / total, phi1


def constrain(theta: npt.ArrayLike, mean: float, var: float, spec: str = "F1") -> tuple[float, ...]:
    """``theta = (t, a, b1, b2[, g][, p])`` to ``(mu, omega, alpha, beta[, gamma][, phi1])``.

    D38 (2), D39 (c): ``phi1 = tanh(p)``, ``mu = (mean + sqrt(var) t) (1 - phi1)``,
    ``omega = var exp(a)``, ``(alpha, gamma / 2, beta, rest) = softmax(b1, g, b2, 0)``,
    computed after subtracting the maximum. ``exp(a)`` overflows to ``inf`` (never raises).
    """
    ar, gjr = SPECS[spec]
    mu, omega, alpha, beta, gamma, phi1 = _expand(theta, mean, var, spec)
    return (mu, omega, alpha, beta, *((gamma,) if gjr else ()), *((phi1,) if ar else ()))


def unconstrain(
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    mean: float,
    var: float,
    *,
    gamma: float | None = None,
    phi1: float | None = None,
) -> FloatArray:
    """Inverse of :func:`constrain`; ``gamma`` and ``phi1`` are given iff the spec has them.

    Needs ``omega, alpha, beta (, gamma) > 0``, ``alpha + gamma / 2 + beta < 1``, ``|phi1| < 1``.
    """
    half = 0.0 if gamma is None else gamma / 2
    rest = 1.0 - alpha - half - beta
    theta = [
        (mu / (1.0 - (phi1 or 0.0)) - mean) / math.sqrt(var),
        math.log(omega / var),
        math.log(alpha / rest),
        math.log(beta / rest),
    ]
    if gamma is not None:
        theta.append(math.log(half / rest))
    if phi1 is not None:
        theta.append(math.atanh(phi1))
    return np.array(theta)


def _eps(x: FloatArray, mu: float, phi1: float, lag: float | None) -> FloatArray:
    """Residuals ``L_t - mu - phi1 L_{t-1}``; ``lag is None`` means a constant mean."""
    if lag is None:
        return x - mu
    return np.asarray(x - mu - phi1 * np.concatenate(([lag], x[:-1])), dtype=np.float64)


def _sigma2(eps: FloatArray, omega: float, alpha: float, beta: float, gamma: float) -> FloatArray:
    """Variance path: backcast ``mean(eps2)``, then the GJR recursion (GARCH if ``gamma = 0``)."""
    eps2 = eps**2
    u = np.empty_like(eps2)
    u[0] = eps2.mean()
    u[1:] = omega + (alpha + gamma * (eps[:-1] > 0)) * eps2[:-1]
    return np.asarray(lfilter([1.0], [1.0, -beta], u), dtype=np.float64)


def _nll(eps: FloatArray, omega: float, alpha: float, beta: float, gamma: float) -> float:
    s2 = _sigma2(eps, omega, alpha, beta, gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 0.5 * float(np.sum(_LOG_2PI + np.log(s2) + eps**2 / s2))
    return value if math.isfinite(value) else math.inf


def _vector(losses: npt.ArrayLike) -> FloatArray:
    x = np.asarray(losses, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("losses: must be one-dimensional with at least 2 values")
    return x


def garch_nll(
    losses: npt.ArrayLike,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    *,
    gamma: float = 0.0,
    phi1: float = 0.0,
    lag: float | None = None,
) -> float:
    """Gaussian negative log-likelihood of the filter; ``+inf`` if it is not finite.

    ``lag`` (the pre-window observation) switches the AR(1) mean on; without it the mean is
    the constant ``mu``.

    Raises:
        ValueError: if ``losses`` is not 1-D with at least 2 values.
    """
    return _nll(_eps(_vector(losses), mu, phi1, lag), omega, alpha, beta, gamma)


def _lag(fit_spec: str, lag: float | None) -> float | None:
    """The lag an AR spec needs (``ValueError`` if missing); ``None`` for a constant mean."""
    if not SPECS[fit_spec][0]:
        return None
    if lag is None:
        raise ValueError("lag: an AR(1) spec needs the pre-window observation")
    return float(lag)


def garch_filter(
    losses: npt.ArrayLike, fit: GarchFit, lag: float | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(eps, sigma2, z)`` of ``losses`` under ``fit``; ``z = eps / sqrt(sigma2)``, same length.

    Raises:
        ValueError: if ``losses`` is not 1-D with at least 2 values, or ``fit`` has an AR(1)
            mean and ``lag`` is missing.
    """
    eps = _eps(_vector(losses), fit.mu, fit.phi1, _lag(fit.spec, lag))
    s2 = _sigma2(eps, fit.omega, fit.alpha, fit.beta, fit.gamma)
    return eps, s2, eps / np.sqrt(s2)


def forecast_sigma(fit: GarchFit, eps_last: float, sigma2_last: float) -> float:
    """``sigma_{t+1} = sqrt(omega + (alpha + gamma 1{eps_t > 0}) eps_t^2 + beta sigma2_t)``."""
    arch = fit.alpha + fit.gamma * (eps_last > 0)
    return math.sqrt(fit.omega + arch * eps_last**2 + fit.beta * sigma2_last)


def forecast_mean(fit: GarchFit, loss_last: float) -> float:
    """One-step mean ``mu_{t+1} = mu + phi1 L_t`` (``mu`` itself for a constant mean)."""
    return fit.mu + fit.phi1 * loss_last


def _objective(
    theta: FloatArray,
    x: FloatArray,
    mean: float,
    var: float,
    spec: str = "F1",
    lag: float | None = None,
) -> float:
    """NLL at ``constrain(theta)``; ``+inf`` outside the floating-point domain (never clipped)."""
    mu, omega, alpha, beta, gamma, phi1 = _expand(theta, mean, var, spec)
    inside = 0 < omega < math.inf and alpha + gamma / 2 + beta < 1 and abs(phi1) < 1
    if not (math.isfinite(mu) and inside):
        return math.inf
    return _nll(_eps(x, mu, phi1, lag), omega, alpha, beta, gamma)


def _ols_slope(x: FloatArray, lag: float) -> float:
    """OLS slope of ``L_t`` on ``(1, L_{t-1})`` over the window; ``nan`` if it is undefined."""
    prev = np.concatenate(([lag], x[:-1]))
    d = prev - prev.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(d @ (x - x.mean())) / np.float64(d @ d))


def _starts(
    x: FloatArray, mean: float, var: float, spec: str, lag: float | None
) -> list[tuple[int, FloatArray]]:
    """The usable pinned starts ``(index k, theta0)`` of ``spec`` (D38 (2), D39 (c)).

    Variance starts ``j``: ``STARTS`` or ``GJR_STARTS``, ``omega = var x rest``. AR specs
    cross them with the ``phi1`` starts ``i`` = 0 and the OLS slope: ``k = i x J + j``. A
    slope that is not finite or not inside (-1, 1) drops its starts.
    """
    gjr = SPECS[spec][1]
    variance = GJR_STARTS if gjr else tuple((a0, 0.0, b0) for a0, b0 in STARTS)
    phis = (None,) if lag is None else (0.0, _ols_slope(x, lag))
    out: list[tuple[int, FloatArray]] = []
    for i, phi in enumerate(phis):
        if phi is not None and not abs(phi) < 1:
            continue
        for j, (alpha0, gamma0, beta0) in enumerate(variance):
            theta = unconstrain(
                mean * (1.0 - (phi or 0.0)),
                var * (1 - alpha0 - gamma0 / 2 - beta0),
                alpha0,
                beta0,
                mean,
                var,
                gamma=gamma0 if gjr else None,
                phi1=phi,
            )
            out.append((i * len(variance) + j, theta))
    return out


def fit_garch(
    losses: npt.ArrayLike, cfg: GarchConfig, spec: str = "F1", *, lag: float | None = None
) -> GarchFit:
    """Gaussian QMLE of ``spec`` by Nelder-Mead from its pinned starts (D38 (2), D39 (c)).

    Every start has the window mean as its unconditional mean; see :func:`_starts`. The
    initial simplex is the start and the start plus ``cfg.simplex_step`` in each transformed
    coordinate. A start is usable iff scipy reports success with a finite NLL; the usable
    start with the lowest NLL is kept (tie: lower index). ``lag`` is the pre-window
    observation of an AR(1) spec and is ignored by the constant-mean specs.

    Raises:
        GarchError: on a non-finite or constant input, if no start is usable, or if the
            variance path or the one-step forecast at the optimum is not finite.
        ValueError: if ``losses`` is not 1-D with at least 2 values, or an AR(1) spec has
            no ``lag``.
    """
    x = _vector(losses)
    lag = _lag(spec, lag)
    if not (np.all(np.isfinite(x)) and (lag is None or math.isfinite(lag))):
        raise GarchError("losses: all values must be finite")
    mean, var = float(x.mean()), float(x.var())
    if not var > 0:
        raise GarchError("losses: constant series (zero variance)")

    step = float(cfg.simplex_step)
    best: tuple[float, int, FloatArray] | None = None
    for k, x0 in _starts(x, mean, var, spec, lag):
        res = minimize(
            _objective,
            x0,
            args=(x, mean, var, spec, lag),
            method="Nelder-Mead",
            options={
                "xatol": float(cfg.xatol),
                "fatol": float(cfg.fatol),
                "maxiter": cfg.maxiter,
                "maxfev": cfg.maxfev,
                "initial_simplex": np.vstack([x0, x0 + step * np.eye(x0.size)]),
            },
        )
        value = _objective(res.x, x, mean, var, spec, lag)
        if bool(res.success) and math.isfinite(value) and (best is None or value < best[0]):
            best = (value, k, res.x)
    if best is None:
        raise GarchError("GARCH: no start converged")
    nll, start, theta = best
    mu, omega, alpha, beta, gamma, phi1 = _expand(theta, mean, var, spec)
    near = alpha + gamma / 2 + beta > PERSISTENCE_CEILING or alpha < ALPHA_FLOOR
    flags = ("near_boundary",) if near else ()
    fit = GarchFit(mu, omega, alpha, beta, nll, start, flags, phi1, gamma, spec)
    eps, s2, z = garch_filter(x, fit, lag)
    if not (np.all(np.isfinite(z)) and math.isfinite(forecast_sigma(fit, eps[-1], s2[-1]))):
        raise GarchError("GARCH: non-finite variance path or forecast")
    return fit


def bic(nll: float, k: int, n: int) -> float:
    """``BIC = 2 NLL + k ln(n)`` (D39 (c))."""
    return 2.0 * nll + k * math.log(n)


def select_filter(losses: npt.ArrayLike, cfg: GarchConfig) -> tuple[dict[str, GarchFit], str]:
    """Fit F1...F4 on ``losses[1:]`` with ``losses[0]`` as the pre-sample lag; pick by BIC.

    D39 (c): all four specs share the conditional sample (``n - 1`` likelihood terms). The
    winner has the lowest BIC; ties go to the smaller ``k``, then to the lower index.

    Raises:
        GarchError: if any candidate fails (nothing is chosen).
        ValueError: if ``losses`` is not 1-D with at least 3 values.
    """
    x = _vector(losses)
    window = _vector(x[1:])
    fits = {spec: fit_garch(window, cfg, spec, lag=float(x[0])) for spec in SPECS}
    winner = min(SPECS, key=lambda s: (bic(fits[s].nll, n_params(s), window.size), n_params(s)))
    return fits, winner
