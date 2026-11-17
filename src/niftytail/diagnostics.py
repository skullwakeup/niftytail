"""Dependence diagnostics for the Day-4 battery (D29 items 3-7; methodology 9).

* :func:`acf`: sample autocorrelation, mean-removed, biased denominator
  ``sum (x - mean)^2``, lags ``1..max_lag``.
* :func:`ljung_box`: ``Q(m) = n (n + 2) sum_{k=1..m} rho_k^2 / (n - k)``, df = m (no
  adjustment), p from the chi-square survival function.
* :func:`arch_lm`: Engle (1982). ``e = r - mean(r)``; OLS of ``e_t^2`` on a constant and
  ``q`` lags of ``e^2``; ``LM = (n - q) R^2``, df = q.
* :func:`sign_bias`: Engle-Ng sign-bias tests on standardized residuals (D39 (e); the
  convention is pinned there and was not checked against the paper [L38, F]).
* :func:`inter_exceedance_times`: ``T_i = S_{i+1} - S_i`` in observation counts.
* :func:`extremal_index_fs`: Ferro-Segers intervals estimator (F-status: formula checked
  against two secondary descriptions, the paper was not read). No CI.
* :func:`decluster_fs`: Ferro-Segers intervals declustering (D36 (c); convention read in
  Ferro's thesis, section 4.1.5 [L11a]). Diagnostic only: never used for VaR or ES.

Error messages never contain data values (D23).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

__all__ = [
    "ArchLMResult",
    "DeclusterResult",
    "DiagnosticsError",
    "ExtremalIndexResult",
    "LjungBoxResult",
    "SignBiasResult",
    "acf",
    "arch_lm",
    "decluster_fs",
    "extremal_index_fs",
    "inter_exceedance_times",
    "ljung_box",
    "sign_bias",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]


class DiagnosticsError(ValueError):
    """Invalid input for a diagnostic (too short, constant, too few exceedances, ...)."""


@dataclass(frozen=True, slots=True)
class LjungBoxResult:
    """Ljung-Box statistic ``q`` at lag ``m`` with ``df = m`` and chi-square p-value."""

    m: int
    q: float
    df: int
    p: float


@dataclass(frozen=True, slots=True)
class ArchLMResult:
    """Engle ARCH-LM statistic with ``df = q`` lags, p-value and regression size."""

    lags: int
    lm: float
    df: int
    p: float
    nobs: int


@dataclass(frozen=True, slots=True)
class SignBiasResult:
    """Engle-Ng tests: ``(t statistic, two-sided p)`` per single test, and the joint LM test."""

    sign: tuple[float, float]
    negative_size: tuple[float, float]
    positive_size: tuple[float, float]
    joint_lm: float
    joint_df: int
    joint_p: float
    nobs: int


@dataclass(frozen=True, slots=True)
class ExtremalIndexResult:
    """Ferro-Segers extremal index (clipped to 1), branch used and mean cluster size."""

    theta: float
    branch: Literal["T<=2", "T>2"]
    mean_cluster_size: float
    n_exceedances: int


@dataclass(frozen=True, slots=True)
class DeclusterResult:
    """Intervals declustering: clusters are numbered 0..C-1 in time order.

    ``cluster_index[i]`` is the cluster of the ``i``-th exceedance, ``cluster_maxima[j]``
    the largest loss in cluster ``j`` and ``run_length`` the pinned ``r = T_(C)``.
    """

    cluster_index: IntArray
    cluster_maxima: FloatArray
    n_clusters: int
    run_length: int


def _series(x: npt.ArrayLike, name: str) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise DiagnosticsError(f"{name}: must be one-dimensional")
    if not np.all(np.isfinite(a)):
        raise DiagnosticsError(f"{name}: all values must be finite")
    return a


def _lag(m: int, name: str) -> int:
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise DiagnosticsError(f"{name}: must be a positive integer")
    return m


def acf(x: npt.ArrayLike, max_lag: int) -> FloatArray:
    """Autocorrelations at lags ``1..max_lag`` (biased, mean-removed).

    Raises:
        DiagnosticsError: if ``n <= max_lag``, ``x`` is constant or contains non-finite values.
    """
    a = _series(x, "x")
    m = _lag(max_lag, "max_lag")
    n = a.size
    if n <= m:
        raise DiagnosticsError(f"acf: needs n > max_lag (n = {n}, max_lag = {m})")
    d = a - a.mean()
    denom = float(d @ d)
    if denom == 0.0:
        raise DiagnosticsError("acf: series is constant")
    return np.array([float(d[:-k] @ d[k:]) / denom for k in range(1, m + 1)])


def ljung_box(x: npt.ArrayLike, m: int) -> LjungBoxResult:
    """Ljung-Box ``Q(m)``, df = m, chi-square p-value.

    Raises:
        DiagnosticsError: as :func:`acf`.
    """
    rho = acf(x, m)
    n = np.asarray(x).size
    k = np.arange(1, m + 1)
    q = float(n * (n + 2) * np.sum(rho**2 / (n - k)))
    return LjungBoxResult(m=m, q=q, df=m, p=float(stats.chi2.sf(q, m)))


def arch_lm(r: npt.ArrayLike, q: int) -> ArchLMResult:
    """Engle ARCH-LM test with ``q`` lags on ``e = r - mean(r)``.

    Raises:
        DiagnosticsError: if there are not more regression rows than regressors, or
            ``e^2`` is constant.
    """
    a = _series(r, "r")
    q = _lag(q, "q")
    n = a.size
    nobs = n - q
    if nobs <= q + 1:
        raise DiagnosticsError(f"arch_lm: needs n - q > q + 1 (n = {n}, q = {q})")
    e2 = (a - a.mean()) ** 2
    y = e2[q:]
    x = np.column_stack([np.ones(nobs)] + [e2[q - j : n - j] for j in range(1, q + 1)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst == 0.0:
        raise DiagnosticsError("arch_lm: squared residuals are constant")
    lm = nobs * (1.0 - float(resid @ resid) / sst)
    return ArchLMResult(lags=q, lm=lm, df=q, p=float(stats.chi2.sf(lm, q)), nobs=nobs)


def sign_bias(eps: npt.ArrayLike, z: npt.ArrayLike) -> SignBiasResult:
    """Engle-Ng sign-bias tests of ``z_t^2`` on the previous shock (D39 (e)).

    ``eps`` are loss-space residuals and ``z`` their standardized values. With the return
    shock ``e = -eps``, ``S- = 1{e_{t-1} < 0}`` (bad news) and ``S+ = 1 - S-``, rows
    ``t = 2..n``: three simple regressions of ``z_t^2`` on a constant and ``S-``,
    ``S- e_{t-1}`` or ``S+ e_{t-1}`` (slope t statistic, classical OLS standard error,
    Student-t p-value with ``rows - 2`` df), and the joint regression on all three
    (``LM = rows x R^2``, chi-square with 3 df).

    Raises:
        DiagnosticsError: if the inputs differ in length, there are fewer than 6 values, or
            ``z^2`` or a regressor is constant over the rows.
    """
    a, b = _series(eps, "eps"), _series(z, "z")
    if a.size != b.size or a.size < 6:
        raise DiagnosticsError("sign_bias: eps and z need the same length, at least 6")
    e, y = -a[:-1], b[1:] ** 2
    nobs = y.size
    neg = (e < 0).astype(np.float64)
    cols = [neg, neg * e, (1.0 - neg) * e]
    dy = y - y.mean()
    sst = float(dy @ dy)
    if sst == 0.0 or any(float(np.ptp(c)) == 0.0 for c in cols):
        raise DiagnosticsError("sign_bias: squared residuals or a regressor are constant")

    def single(x: FloatArray) -> tuple[float, float]:
        dx = x - x.mean()
        sxx = float(dx @ dx)
        slope = float(dx @ dy) / sxx
        rss = float(np.sum((dy - slope * dx) ** 2))
        t = slope / math.sqrt(rss / (nobs - 2) / sxx)
        return t, float(2.0 * stats.t.sf(abs(t), nobs - 2))

    design = np.column_stack([np.ones(nobs), *cols])
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    lm = nobs * (1.0 - float(resid @ resid) / sst)
    return SignBiasResult(
        sign=single(cols[0]),
        negative_size=single(cols[1]),
        positive_size=single(cols[2]),
        joint_lm=lm,
        joint_df=3,
        joint_p=float(stats.chi2.sf(lm, 3)),
        nobs=nobs,
    )


def inter_exceedance_times(mask: npt.ArrayLike) -> IntArray:
    """Times ``S_{i+1} - S_i`` between consecutive exceedances (``i = 1..N-1``).

    Raises:
        DiagnosticsError: if ``mask`` is not a 1-D boolean array or has fewer than 2 exceedances.
    """
    b = np.asarray(mask)
    if b.ndim != 1 or b.dtype != np.bool_:
        raise DiagnosticsError("mask: must be a one-dimensional boolean array")
    s = np.flatnonzero(b)
    if s.size < 2:
        raise DiagnosticsError(f"inter_exceedance_times: needs N >= 2 exceedances, got {s.size}")
    return np.diff(s).astype(np.int64)


def _theta_fs(times: npt.ArrayLike) -> tuple[Fraction, Literal["T<=2", "T>2"], list[int]]:
    """Exact intervals estimate, the branch used and the times as Python integers."""
    t_arr = np.asarray(times)
    if t_arr.ndim != 1 or not np.issubdtype(t_arr.dtype, np.integer):
        raise DiagnosticsError("times: must be a one-dimensional integer array")
    if t_arr.size < 1:
        raise DiagnosticsError("extremal_index_fs: needs N >= 2 exceedances")
    t = [int(v) for v in t_arr]
    if min(t) < 1:
        raise DiagnosticsError("times: must be positive integers")
    n_minus_1 = len(t)
    branch: Literal["T<=2", "T>2"]
    if max(t) <= 2:
        branch = "T<=2"
        ratio = Fraction(2 * sum(t) ** 2, n_minus_1 * sum(v * v for v in t))
    else:
        branch = "T>2"
        ratio = Fraction(
            2 * sum(v - 1 for v in t) ** 2, n_minus_1 * sum((v - 1) * (v - 2) for v in t)
        )
    return min(Fraction(1), ratio), branch, t


def extremal_index_fs(times: npt.ArrayLike) -> ExtremalIndexResult:
    """Ferro-Segers intervals estimator from inter-exceedance times (``N = len(times) + 1``).

    ``max T <= 2``: ``min(1, 2 (sum T)^2 / ((N-1) sum T^2))``; otherwise
    ``min(1, 2 (sum (T-1))^2 / ((N-1) sum (T-1)(T-2)))``. Computed in exact integer
    arithmetic, then rounded once to float.

    Raises:
        DiagnosticsError: if ``N < 2`` or a time is not a positive integer.
    """
    theta, branch, t = _theta_fs(times)
    return ExtremalIndexResult(
        theta=float(theta),
        branch=branch,
        mean_cluster_size=float(1 / theta),
        n_exceedances=len(t) + 1,
    )


def decluster_fs(
    losses: npt.ArrayLike, u: float, *, theta: Fraction | None = None
) -> DeclusterResult:
    """Ferro-Segers intervals declustering of the exceedances ``losses > u`` (D36 (c)).

    With ``N`` exceedances and inter-exceedance times ``T_1..T_{N-1}``: ``C - 1 =
    floor(theta (N - 1))`` in exact integer arithmetic, ``r = T_(C)`` the ``C``-th largest
    time (``r = 0`` if ``C = N``); the times ``T_i > r`` separate clusters. Ties: every time
    equal to ``r`` stays inside a cluster, so the final ``C = 1 + #{T_i > r}`` (the source's
    "decrease C until T_(C-1) > T_(C)"). ``theta`` defaults to the exact intervals estimate
    of :func:`extremal_index_fs`. Diagnostic only: never used for VaR or ES.

    Raises:
        DiagnosticsError: on invalid ``losses`` or ``u``, fewer than 2 exceedances, or a
            ``theta`` that is not a Fraction in [0, 1].
    """
    x = _series(losses, "losses")
    if not math.isfinite(u):
        raise DiagnosticsError("u: must be finite")
    mask = x > u
    times = inter_exceedance_times(mask)
    estimate, _, t = _theta_fs(times)
    if theta is None:
        theta = estimate
    elif not isinstance(theta, Fraction) or not 0 <= theta <= 1:
        raise DiagnosticsError("theta: must be a fractions.Fraction in [0, 1]")
    c = 1 + (theta.numerator * len(t)) // theta.denominator
    r = 0 if c == len(t) + 1 else sorted(t, reverse=True)[c - 1]
    gap = times > r
    index = np.concatenate(([0], np.cumsum(gap))).astype(np.int64)
    starts = np.flatnonzero(np.concatenate(([True], gap)))
    return DeclusterResult(
        cluster_index=index,
        cluster_maxima=np.maximum.reduceat(x[mask], starts),
        n_clusters=int(starts.size),
        run_length=r,
    )
