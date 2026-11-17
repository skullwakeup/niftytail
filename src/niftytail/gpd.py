"""GPD core for peaks over threshold (D30; methodology 6).

* Excesses ``y = L - u`` over a threshold, strict ``L > u`` (Day-4 convention, D29).
* ``G(y; xi, beta) = 1 - (1 + xi y / beta)^(-1/xi)``, ``beta > 0``; ``xi = 0`` limit
  ``1 - exp(-y / beta)``; location fixed at 0. Same sign as ``scipy genpareto(c=xi)``.
* MLE minimises our own negative log-likelihood over ``(xi, ln beta)`` with Nelder-Mead,
  started from probability-weighted moments (Hosking-Wallis 1987, loc = 0), or from the
  exponential start if PWM fails or lies outside the likelihood domain. Parameters are
  never clipped or repaired.
* Tail VaR/ES from the POT tail estimator ``F(x) = (N_u/n)(1 + xi (x-u)/beta)^(-1/xi)``.

Day 6 (D31): SE(xi), the profile-likelihood CI for xi, delta-method CIs for VaR/ES from the
expected information, and the edge-case policy (xi = -1 boundary raises; irregular and
xi >= 1 fits are returned). Error messages never contain data values (D23).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import brentq, minimize, minimize_scalar

from niftytail.config import GpdConfig

__all__ = [
    "DeltaCI",
    "EstimationError",
    "GpdFit",
    "ProfileCI",
    "delta_ci",
    "excesses",
    "expected_covariance",
    "fit_mle",
    "fit_pwm",
    "nll",
    "profile_ci_xi",
    "risk_gradient",
    "se_xi",
    "tail_es",
    "tail_risk",
    "tail_var",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
_LOG_BETA_MAX = 700.0  # exp() overflows beyond ~709.78; outside this the nll is +inf


class EstimationError(Exception):
    """GPD estimation failed (too few excesses, PWM undefined, optimiser not converged)."""


@dataclass(frozen=True, slots=True)
class GpdFit:
    """Result of :func:`fit_mle`. ``regular`` is ``xi > -0.5`` (MLE regularity [L04, F])."""

    xi: float
    beta: float
    nll: float
    n_excesses: int
    method: Literal["mle"]
    start: Literal["pwm", "exp_fallback"]
    n_iter: int
    regular: bool


def _vector(x: npt.ArrayLike, name: str) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name}: must be one-dimensional")
    if a.size < 1:
        raise ValueError(f"{name}: must not be empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: all values must be finite")
    return a


def _excess_vector(y: npt.ArrayLike) -> FloatArray:
    a = _vector(y, "y")
    if not np.all(a > 0):
        raise ValueError("y: excesses must all be > 0")
    return a


def _min_excesses(m: int) -> int:
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise ValueError("min_excesses: must be an integer >= 2")
    return m


def _finite(name: str, *xs: float) -> None:
    if not all(math.isfinite(x) for x in xs):
        raise ValueError(f"{name}: inputs must be finite")


def excesses(losses: npt.ArrayLike, u: float) -> FloatArray:
    """Excesses ``losses[losses > u] - u`` (strict, as the Day-4 exceedance mask).

    Raises:
        ValueError: if ``losses`` is not 1-D finite or ``u`` is not finite.
    """
    x = _vector(losses, "losses")
    _finite("u", u)
    return x[x > u] - u


def _nll(y: FloatArray, xi: float, beta: float, xi_zero_tol: float) -> float:
    if not (math.isfinite(xi) and math.isfinite(beta)) or beta <= 0 or xi <= -1:
        return math.inf
    n = y.size
    z = y / beta
    t = xi * z
    if np.any(t <= -1):  # 1 + xi z <= 0: outside the support, in every branch (D31 (1))
        return math.inf
    if abs(xi) < xi_zero_tol:
        return float(n * math.log(beta) + np.sum(z))
    lg = np.log1p(t)
    return float(n * math.log(beta) + np.sum(lg / xi) + np.sum(lg))


def nll(y: npt.ArrayLike, xi: float, beta: float, xi_zero_tol: float) -> float:
    """GPD negative log-likelihood; ``+inf`` outside the domain.

    ``+inf`` if ``beta <= 0`` or non-finite, ``xi <= -1`` or any ``1 + xi y / beta <= 0``
    (checked in every branch, D31 (1)).
    With ``z = y / beta``: ``n ln beta + sum z`` if ``|xi| < xi_zero_tol``, else
    ``n ln beta + sum(log1p(xi z) / xi) + sum(log1p(xi z))`` (no cancellation near 0).

    Raises:
        ValueError: if ``y`` is not a 1-D finite vector of positive excesses.
    """
    return _nll(_excess_vector(y), xi, beta, xi_zero_tol)


def fit_pwm(y: npt.ArrayLike, *, min_excesses: int) -> tuple[float, float]:
    """Probability-weighted-moment estimates ``(xi, beta)`` (Hosking-Wallis 1987, loc = 0).

    ``a0 = mean(y)``, ``a1 = sum y_(i) (n - i) / (n (n - 1))`` (ascending order),
    ``xi = 2 - a0 / (a0 - 2 a1)``, ``beta = 2 a0 a1 / (a0 - 2 a1)``.

    Raises:
        EstimationError: if ``n < min_excesses``, ``a0 - 2 a1 <= 0`` or the estimates are
            not finite with ``beta > 0``.
        ValueError: on invalid ``y`` or ``min_excesses``.
    """
    m = _min_excesses(min_excesses)
    ys = np.sort(_excess_vector(y))
    n = ys.size
    if n < m:
        raise EstimationError(f"PWM: needs at least {m} excesses, got {n}")
    a0 = float(np.mean(ys))
    a1 = float(ys @ (n - np.arange(1, n + 1, dtype=np.float64))) / (n * (n - 1))
    denom = a0 - 2.0 * a1
    if not denom > 0:
        raise EstimationError("PWM: a0 - 2 a1 <= 0 (estimator undefined)")
    xi = 2.0 - a0 / denom
    beta = 2.0 * a0 * a1 / denom
    if not (math.isfinite(xi) and math.isfinite(beta) and beta > 0):
        raise EstimationError("PWM: estimates are not finite with beta > 0")
    return xi, beta


def fit_mle(
    y: npt.ArrayLike, *, min_excesses: int, xi_zero_tol: float, settings: GpdConfig
) -> GpdFit:
    """Maximum-likelihood GPD fit over ``(xi, ln beta)`` with Nelder-Mead (D30 item 5).

    Start: PWM if it succeeds with a finite nll, else ``(0, mean(y))`` ("exp_fallback").
    Converged iff the optimiser reports success, the nll is finite and not above the
    start's. Edge-case policy (D31 (2)): ``xi <= -1 + xi_boundary_tol`` raises; a fit with
    ``-1 + tol < xi <= -0.5`` is returned with ``regular = False``; ``xi >= 1`` is returned
    (its ES is infinite, see :func:`tail_es`).

    Raises:
        EstimationError: if ``n < min_excesses``, the optimiser does not converge or the
            optimum is at the ``xi = -1`` boundary.
        ValueError: on invalid ``y``, ``min_excesses`` or settings.
    """
    m = _min_excesses(min_excesses)
    a = _excess_vector(y)
    n = a.size
    if n < m:
        raise EstimationError(f"MLE: needs at least {m} excesses, got {n}")
    if settings.optimizer != "Nelder-Mead":
        raise ValueError("settings.optimizer: only 'Nelder-Mead' is specified (D30)")

    start: Literal["pwm", "exp_fallback"] = "exp_fallback"
    x0 = (0.0, math.log(float(np.mean(a))))
    try:
        xi0, beta0 = fit_pwm(a, min_excesses=m)
    except EstimationError:
        pass
    else:
        if math.isfinite(_nll(a, xi0, beta0, xi_zero_tol)):
            start, x0 = "pwm", (xi0, math.log(beta0))
    nll0 = _nll(a, x0[0], math.exp(x0[1]), xi_zero_tol)

    def objective(theta: FloatArray) -> float:
        if not abs(theta[1]) < _LOG_BETA_MAX:
            return math.inf
        return _nll(a, float(theta[0]), math.exp(float(theta[1])), xi_zero_tol)

    res = minimize(
        objective,
        np.array(x0),
        method="Nelder-Mead",
        options={
            "xatol": float(settings.xatol),
            "fatol": float(settings.fatol),
            "maxiter": settings.maxiter,
            "maxfev": settings.maxfev,
        },
    )
    xi, log_beta = float(res.x[0]), float(res.x[1])
    value = objective(np.array([xi, log_beta]))
    if not (bool(res.success) and math.isfinite(value) and value <= nll0):
        raise EstimationError("MLE: optimiser did not converge")
    if xi <= -1 + float(settings.xi_boundary_tol):
        raise EstimationError("MLE at the xi = -1 boundary")
    return GpdFit(
        xi=xi,
        beta=math.exp(log_beta),
        nll=value,
        n_excesses=n,
        method="mle",
        start=start,
        n_iter=int(res.nit),
        regular=xi > -0.5,
    )


def tail_var(
    u: float, xi: float, beta: float, n: int, n_u: int, p: Fraction, xi_zero_tol: float
) -> float:
    """POT VaR at tail probability ``p`` (``c = 1 - p``).

    ``r = (n / n_u) p`` as an exact rational rounded once; ``u - beta ln r`` if
    ``|xi| < xi_zero_tol``, else ``u + (beta / xi) expm1(-xi ln r)``.

    Raises:
        ValueError: unless ``n >= n_u >= 1`` are ints, ``p`` is a Fraction with
            ``0 < p < n_u / n``, the floats are finite and ``beta > 0``.
    """
    for name, k in (("n", n), ("n_u", n_u)):
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError(f"{name}: must be an integer")
    if not n >= n_u >= 1:
        raise ValueError("need n >= n_u >= 1")
    if not isinstance(p, Fraction) or not 0 < p < 1:
        raise ValueError("p: must be a fractions.Fraction strictly between 0 and 1")
    if not p < Fraction(n_u, n):
        raise ValueError("p: the tail estimator needs p < n_u / n")
    _finite("tail_var", u, xi, beta)
    if beta <= 0:
        raise ValueError("beta: must be > 0")
    log_r = math.log(float(Fraction(n, n_u) * p))
    if abs(xi) < xi_zero_tol:
        return u - beta * log_r
    return u + (beta / xi) * math.expm1(-xi * log_r)


def tail_es(var: float, u: float, xi: float, beta: float) -> float:
    """POT ES ``(VaR + (beta - xi u)) / (1 - xi)``; ``math.inf`` if ``xi >= 1`` (never clipped).

    Raises:
        ValueError: if an input is not finite or ``beta <= 0``.
    """
    _finite("tail_es", var, u, xi, beta)
    if beta <= 0:
        raise ValueError("beta: must be > 0")
    if xi >= 1:
        return math.inf
    return (var + (beta - xi * u)) / (1 - xi)


def tail_risk(
    u: float, xi: float, beta: float, n: int, n_u: int, p: Fraction, xi_zero_tol: float
) -> tuple[float, float]:
    """``(VaR, ES)`` at tail probability ``p`` (see :func:`tail_var`, :func:`tail_es`)."""
    var = tail_var(u, xi, beta, n, n_u, p, xi_zero_tol)
    return var, tail_es(var, u, xi, beta)


# ---------------------------------------------------------------------------
# Uncertainty (D31)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ProfileCI:
    """Profile-likelihood CI for xi; ``lower_open`` means the set reaches the xi = -1 limit."""

    lower: float
    upper: float
    level: float
    lower_open: bool


@dataclass(frozen=True, slots=True)
class DeltaCI:
    """Delta-method CI ``estimate +- z se`` for VaR or ES (u and n_u/n treated as fixed)."""

    estimate: float
    lower: float
    upper: float
    se: float
    level: float


def se_xi(xi: float, n_u: int) -> float:
    """Asymptotic ``SE(xi_hat) = (1 + xi) / sqrt(n_u)`` [L04/L05, F].

    Raises:
        ValueError: unless ``xi > -1/2`` is finite and ``n_u >= 1`` is an int.
    """
    _finite("se_xi", xi)
    if not xi > -0.5:
        raise ValueError("se_xi: needs xi > -1/2")
    if isinstance(n_u, bool) or not isinstance(n_u, int) or n_u < 1:
        raise ValueError("n_u: must be an integer >= 1")
    return (1 + xi) / math.sqrt(n_u)


def _profile_nll(y: FloatArray, xi0: float, xi_zero_tol: float, xatol: float) -> float:
    """``min over ln(beta)`` of the nll at fixed ``xi0`` (bounded Brent; unimodal in beta)."""
    log_ymax = math.log(float(np.max(y)))
    # finite-nll bounds: for xi0 < 0 the support needs beta > -xi0 max(y)
    lo = math.log(-xi0) + log_ymax + 1e-10 if xi0 < 0 else log_ymax - 30.0
    hi = log_ymax + 30.0
    res = minimize_scalar(
        lambda lb: _nll(y, xi0, math.exp(lb), xi_zero_tol),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": xatol, "maxiter": 1000},
    )
    if not (bool(res.success) and math.isfinite(float(res.fun))):
        raise EstimationError("profile likelihood: inner minimisation failed")
    return float(res.fun)


def profile_ci_xi(
    y: npt.ArrayLike, fit: GpdFit, *, xi_zero_tol: float, settings: GpdConfig
) -> ProfileCI:
    """Profile-likelihood CI ``{xi0 : D(xi0) <= chi2.ppf(ci_level, 1)}`` (D31 (4)).

    ``D(xi0) = 2 (min_beta nll(y, xi0, beta) - fit.nll)``. Bounds by brentq on ``D - c``,
    bracketed by steps ``max((1 + xi)/sqrt(n), 0.05) * 2^k`` outward from ``fit.xi``.
    Lower side stops at ``-1 + xi_boundary_tol`` (then ``lower = -1``, ``lower_open``).

    Raises:
        EstimationError: if no upper crossing exists by ``profile_xi_max``.
        ValueError: if ``y`` does not match ``fit``.
    """
    a = _excess_vector(y)
    if a.size != fit.n_excesses:
        raise ValueError("profile_ci_xi: y does not match the fit")
    level = float(settings.ci_level)
    c = float(stats.chi2.ppf(level, 1))
    xtol = float(settings.profile_xtol)
    xi_max = float(settings.profile_xi_max)
    floor = -1.0 + float(settings.xi_boundary_tol)
    if not floor < fit.xi < xi_max:
        raise EstimationError("profile CI: the estimate lies outside (-1 + tol, profile_xi_max)")

    def excess_dev(xi0: float) -> float:
        return 2.0 * (_profile_nll(a, xi0, xi_zero_tol, xtol) - fit.nll) - c

    step = max((1 + fit.xi) / math.sqrt(a.size), 0.05)
    inner, k = fit.xi, 0
    while True:
        outer = min(fit.xi + step * 2**k, xi_max)
        if excess_dev(outer) > 0:
            break
        if outer >= xi_max:
            raise EstimationError("profile CI: no upper crossing by profile_xi_max")
        inner, k = outer, k + 1
    upper = float(brentq(excess_dev, inner, outer, xtol=xtol))

    inner, k = fit.xi, 0
    while True:
        outer = max(fit.xi - step * 2**k, floor)
        if excess_dev(outer) > 0:
            break
        if outer <= floor:
            return ProfileCI(lower=-1.0, upper=upper, level=level, lower_open=True)
        inner, k = outer, k + 1
    lower = float(brentq(excess_dev, outer, inner, xtol=xtol))
    return ProfileCI(lower=lower, upper=upper, level=level, lower_open=False)


def expected_covariance(xi: float, beta: float, n_excesses: int) -> FloatArray:
    """Inverse expected Fisher information of ``(xi, beta)``: ``((1+xi)/N) [[1+xi, -beta],
    [-beta, 2 beta^2]]`` (negative off-diagonal; Coles 2001 4.3, Hosking-Wallis 1987).

    Raises:
        EstimationError: unless ``xi > -1/2``.
    """
    if not xi > -0.5:
        raise EstimationError("expected information needs xi > -1/2")
    return ((1 + xi) / n_excesses) * np.array([[1 + xi, -beta], [-beta, 2 * beta**2]])


Quantity = Literal["var", "es"]


def _risk(
    quantity: Quantity, u: float, xi: float, beta: float, n: int, n_u: int, p: Fraction, tol: float
) -> float:
    var = tail_var(u, xi, beta, n, n_u, p, tol)
    return var if quantity == "var" else tail_es(var, u, xi, beta)


def risk_gradient(
    xi: float,
    beta: float,
    u: float,
    n: int,
    n_u: int,
    p: Fraction,
    quantity: Quantity,
    *,
    xi_zero_tol: float,
    settings: GpdConfig,
) -> tuple[float, float]:
    """Central-difference gradient ``(d/dxi, d/dbeta)`` of VaR or ES at ``(xi, beta)``.

    Steps: ``delta_step_xi`` in xi and ``delta_step_beta_rel * beta`` in beta.

    Raises:
        EstimationError: if the measure is not finite at a difference point.
    """
    if quantity not in ("var", "es"):
        raise ValueError("quantity: must be 'var' or 'es'")
    hx = float(settings.delta_step_xi)
    hb = float(settings.delta_step_beta_rel) * beta

    def f(x: float, b: float) -> float:
        return _risk(quantity, u, x, b, n, n_u, p, xi_zero_tol)

    gx = (f(xi + hx, beta) - f(xi - hx, beta)) / (2 * hx)
    gb = (f(xi, beta + hb) - f(xi, beta - hb)) / (2 * hb)
    if not (math.isfinite(gx) and math.isfinite(gb)):
        raise EstimationError("delta method: the risk measure is not finite near the estimate")
    return gx, gb


def delta_ci(
    fit: GpdFit,
    u: float,
    n: int,
    n_u: int,
    p: Fraction,
    quantity: Quantity,
    *,
    xi_zero_tol: float,
    settings: GpdConfig,
) -> DeltaCI:
    """Delta-method CI for VaR or ES at tail probability ``p`` (D31 (5)).

    Uses :func:`expected_covariance` and :func:`risk_gradient`;
    ``z = norm.ppf((1 + ci_level) / 2)``. Ignores the uncertainty of ``u`` and ``n_u/n``.

    Raises:
        EstimationError: if ``xi <= -1/2`` or (ES) ``xi >= 1``.
        ValueError: on an invalid quantity or tail arguments.
    """
    if quantity not in ("var", "es"):
        raise ValueError("quantity: must be 'var' or 'es'")
    if quantity == "es" and fit.xi >= 1:
        raise EstimationError("delta method: ES is not finite for xi >= 1")
    cov = expected_covariance(fit.xi, fit.beta, fit.n_excesses)
    est = _risk(quantity, u, fit.xi, fit.beta, n, n_u, p, xi_zero_tol)
    g = np.array(
        risk_gradient(
            fit.xi, fit.beta, u, n, n_u, p, quantity, xi_zero_tol=xi_zero_tol, settings=settings
        )
    )
    se = math.sqrt(float(g @ cov @ g))
    level = float(settings.ci_level)
    z = float(stats.norm.ppf((1 + level) / 2))
    return DeltaCI(estimate=est, lower=est - z * se, upper=est + z * se, se=se, level=level)
