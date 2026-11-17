"""Log returns, losses, the training slice and the tail-threshold convention (D29 items 1-2).

Conventions:

* ``r_t = ln(P_t / P_{t-1})`` between consecutive *available* trading days. Nothing is
  filled: a return that spans a holiday gap is one observation. Return ``k`` is dated at
  the later price. Loss ``L = -r``.
* Training = the first ``T0`` returns (needs ``T0 + 1`` prices; D3).
* Tail threshold at an exact rational level ``q``: with ``n`` losses sorted ascending,
  ``u = L_(ceil(n q))`` (1-based order statistic), the ceiling computed in integer
  arithmetic. An exceedance is ``L > u`` strictly, so ``N_u = n - ceil(n q)`` without ties.

Error messages never contain data values (D23).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "exceedance_mask",
    "log_losses",
    "log_returns",
    "return_dates",
    "tail_threshold",
    "training_slice",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]


def _vector(x: npt.ArrayLike, name: str, min_len: int) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name}: must be one-dimensional")
    if a.size < min_len:
        raise ValueError(f"{name}: needs at least {min_len} values, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: all values must be finite")
    return a


def log_returns(close: npt.ArrayLike) -> FloatArray:
    """Log returns ``np.log(p[1:] / p[:-1])`` of a price series (length ``len(close) - 1``).

    Raises:
        ValueError: if ``close`` is not 1-D, has fewer than 2 values, or any value is
            non-finite or <= 0.
    """
    p = _vector(close, "close", 2)
    if not np.all(p > 0):
        raise ValueError("close: all values must be > 0")
    return np.log(p[1:] / p[:-1])


def log_losses(close: npt.ArrayLike) -> FloatArray:
    """Losses ``L = -r`` (see :func:`log_returns`)."""
    return -log_returns(close)


def training_slice(x: npt.ArrayLike, t0: int) -> FloatArray:
    """The first ``t0`` observations of a return or loss series.

    Raises:
        ValueError: if ``t0 < 1`` or ``x`` has fewer than ``t0`` values.
    """
    if isinstance(t0, bool) or not isinstance(t0, int) or t0 < 1:
        raise ValueError("t0: must be a positive integer")
    return _vector(x, "x", t0)[:t0]


def return_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dates of the returns: each return is dated at its later price (``dates[1:]``).

    Raises:
        ValueError: if there are fewer than 2 dates or they are not strictly increasing.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("dates: must be a pandas DatetimeIndex")
    if len(dates) < 2:
        raise ValueError("dates: needs at least 2 dates")
    if not (dates.is_monotonic_increasing and dates.is_unique):
        raise ValueError("dates: must be strictly increasing")
    return dates[1:]


def tail_threshold(losses: npt.ArrayLike, q: Fraction) -> float:
    """Threshold ``u = L_(ceil(n q))`` of the ascending order statistics (exact ceiling).

    Raises:
        ValueError: if ``q`` is not a Fraction in (0, 1) or ``losses`` is empty, not 1-D
            or contains non-finite values.
    """
    if not isinstance(q, Fraction) or not 0 < q < 1:
        raise ValueError("q: must be a fractions.Fraction strictly between 0 and 1")
    x = np.sort(_vector(losses, "losses", 1))
    n = x.size
    rank = -((-n * q.numerator) // q.denominator)  # ceil(n q), integer arithmetic
    return float(x[rank - 1])


def exceedance_mask(losses: npt.ArrayLike, u: float) -> BoolArray:
    """Boolean exceedance indicator ``L > u`` (strict).

    Raises:
        ValueError: if ``u`` is not finite or ``losses`` is invalid.
    """
    if not np.isfinite(u):
        raise ValueError("u: must be finite")
    return _vector(losses, "losses", 1) > u
