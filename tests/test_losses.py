"""Tests for niftytail.losses (D29 items 1-2)."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from niftytail.losses import (
    exceedance_mask,
    log_losses,
    log_returns,
    return_dates,
    tail_threshold,
    training_slice,
)

RNG = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111)))


def test_hand_case() -> None:
    r = log_returns([100.0, 110.0, 99.0])
    assert r.tolist() == [math.log(1.1), math.log(0.9)]
    assert np.array_equal(log_losses([100.0, 110.0, 99.0]), -r)


def test_matches_diff_of_log() -> None:
    # Absolute tolerance only: both float formulas carry an absolute rounding error of
    # about 1e-15 (the ulp of ln P), so for returns near 0 their RELATIVE error is
    # unbounded and a relative tolerance is meaningless.
    p = 1000.0 * np.exp(np.cumsum(RNG.normal(0, 0.01, 5000)))
    assert np.allclose(log_returns(p), np.diff(np.log(p)), rtol=0, atol=1e-14)
    assert np.array_equal(log_losses(p), -log_returns(p))


def test_matches_50_digit_decimal_reference() -> None:
    p = [100.0, 100.01, 99.5, 101.25, 101.2501, 87.0, 87.0]
    with localcontext(prec=50):
        ref = [float((Decimal(repr(b)) / Decimal(repr(a))).ln()) for a, b in pairwise(p)]
    assert np.allclose(log_returns(p), ref, rtol=0, atol=1e-14)


def test_no_fill_across_date_gaps() -> None:
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-20", "2020-03-01"])
    p = [100.0, 101.0, 99.0, 102.0]
    assert len(log_returns(p)) == len(p) - 1
    rd = return_dates(dates)
    assert list(rd) == list(dates[1:])  # each return dated at the later price


@pytest.mark.parametrize(
    "bad",
    [[100.0], [], [100.0, 0.0], [100.0, -1.0], [100.0, np.nan], [100.0, np.inf], [[1.0, 2.0]]],
)
def test_bad_close_raises(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        log_returns(bad)


def test_return_dates_validation() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        return_dates(pd.DatetimeIndex(["2020-01-01"]))
    with pytest.raises(ValueError, match="strictly increasing"):
        return_dates(pd.DatetimeIndex(["2020-01-02", "2020-01-01"]))
    with pytest.raises(ValueError, match="strictly increasing"):
        return_dates(pd.DatetimeIndex(["2020-01-01", "2020-01-01"]))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        return_dates(["2020-01-01", "2020-01-02"])  # type: ignore[arg-type]


def test_training_slice() -> None:
    x = np.arange(2500, dtype=float)
    tr = training_slice(x, 2000)
    assert len(tr) == 2000 and np.array_equal(tr, x[:2000])
    assert len(training_slice(x[:2000], 2000)) == 2000
    with pytest.raises(ValueError, match="at least 2000"):
        training_slice(x[:1999], 2000)
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="t0"):
            training_slice(x, bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("n", "q", "rank"),
    [(1000, "0.99", 990), (2000, "0.80", 1600), (2000, "0.95", 1900), (2000, "0.90", 1800)],
)
def test_tail_threshold_hand_checks(n: int, q: str, rank: int) -> None:
    x = np.arange(1, n + 1, dtype=float)  # order statistic k has value k
    perm = RNG.permutation(x)  # tail_threshold must sort internally
    u = tail_threshold(perm, Fraction(q))
    assert u == float(rank)
    assert int(exceedance_mask(perm, u).sum()) == n - rank


def test_tail_threshold_exact_ceiling() -> None:
    # n q = 3.0000...1 in floats would be ceil 4; exactly 3 -> rank 3
    x = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert tail_threshold(x, Fraction(3, 10)) == 3.0
    assert tail_threshold(x, Fraction(31, 100)) == 4.0  # ceil(3.1) = 4
    assert tail_threshold([7.0], Fraction(1, 2)) == 7.0


def test_strict_exceedance_and_ties() -> None:
    x = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    u = tail_threshold(x, Fraction(1, 2))  # ceil(2.5) = 3 -> u = 2.0
    assert u == 2.0
    assert exceedance_mask(x, u).tolist() == [False, False, False, False, True]


def test_tail_threshold_validation() -> None:
    for bad_q in (0.95, Fraction(0), Fraction(1), Fraction(3, 2)):
        with pytest.raises(ValueError, match="q:"):
            tail_threshold([1.0, 2.0], bad_q)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        tail_threshold([], Fraction(1, 2))
    with pytest.raises(ValueError, match="finite"):
        tail_threshold([1.0, np.nan], Fraction(1, 2))
    with pytest.raises(ValueError, match="u:"):
        exceedance_mask([1.0], np.nan)
