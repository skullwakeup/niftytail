"""Tests for niftytail.diagnostics (D29 items 3-7) against independent references."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf as sm_acf

from niftytail.diagnostics import (
    DiagnosticsError,
    acf,
    arch_lm,
    decluster_fs,
    extremal_index_fs,
    inter_exceedance_times,
    ljung_box,
    sign_bias,
)
from niftytail.losses import exceedance_mask, tail_threshold
from niftytail.threshold_study import armax_frechet

SEED = 20261111


def rng() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED)))


# --- ACF and Ljung-Box -----------------------------------------------------------
def test_acf_hand_case() -> None:
    rho = acf([1, 2, 3, 4, 5], 2)
    assert rho[0] == 0.4
    assert rho[1] == -0.1


def test_ljung_box_hand_case() -> None:
    res = ljung_box([1, 2, 3, 4, 5], 2)
    expected = 35 * (0.16 / 4 + 0.01 / 3)
    assert res.q == pytest.approx(expected, rel=1e-12)
    assert res.q == pytest.approx(1.5166666666666666, rel=1e-12)
    assert (res.m, res.df) == (2, 2)
    assert res.p == pytest.approx(math.exp(-expected / 2), rel=1e-12)


def test_acf_and_ljung_box_match_statsmodels() -> None:
    x = rng().standard_t(5, 3000)
    ref_acf = sm_acf(x, nlags=20, adjusted=False, fft=False)[1:]
    np.testing.assert_allclose(acf(x, 20), ref_acf, rtol=1e-10, atol=0)
    ref = acorr_ljungbox(x, lags=[10, 20])
    for m in (10, 20):
        res = ljung_box(x, m)
        assert res.q == pytest.approx(float(ref.loc[m, "lb_stat"]), rel=1e-10)
        assert res.p == pytest.approx(float(ref.loc[m, "lb_pvalue"]), rel=1e-10)


def test_acf_errors() -> None:
    with pytest.raises(DiagnosticsError, match="n > max_lag"):
        acf([1.0, 2.0], 2)
    with pytest.raises(DiagnosticsError, match="constant"):
        acf([1.0, 1.0, 1.0], 1)
    with pytest.raises(DiagnosticsError, match="positive integer"):
        acf([1.0, 2.0, 3.0], 0)
    with pytest.raises(DiagnosticsError, match="finite"):
        acf([1.0, np.nan, 3.0], 1)
    with pytest.raises(DiagnosticsError, match="one-dimensional"):
        acf([[1.0, 2.0]], 1)
    with pytest.raises(DiagnosticsError, match="n > max_lag"):
        ljung_box([1.0, 2.0, 3.0], 3)


# --- ARCH-LM ---------------------------------------------------------------------
def test_arch_lm_matches_statsmodels_het_arch() -> None:
    r = rng().standard_t(5, 2500) * 0.01 + 0.0005
    e = r - r.mean()
    for q in (5, 10):
        lm, lmp, _, _ = het_arch(e, nlags=q, ddof=0)
        res = arch_lm(r, q)
        assert res.lm == pytest.approx(lm, rel=1e-10)
        assert res.p == pytest.approx(lmp, rel=1e-10)
        assert (res.lags, res.df, res.nobs) == (q, q, 2500 - q)


def test_arch_lm_iid_not_rejected() -> None:
    r = rng().standard_normal(2000)
    for q in (5, 10):
        assert arch_lm(r, q).p > 0.01


def test_arch_lm_detects_arch1() -> None:
    g = rng()
    n, burn = 2000, 500
    z = g.standard_normal(n + burn)
    r = np.empty(n + burn)
    prev = 0.0
    for t in range(n + burn):
        r[t] = math.sqrt(0.1 + 0.6 * prev**2) * z[t]
        prev = r[t]
    for q in (5, 10):
        assert arch_lm(r[burn:], q).p < 0.001


def test_arch_lm_errors() -> None:
    with pytest.raises(DiagnosticsError, match="n - q > q \\+ 1"):
        arch_lm(np.arange(10.0), 5)
    with pytest.raises(DiagnosticsError, match="constant"):
        arch_lm(np.ones(50), 5)


# --- exceedances and extremal index ------------------------------------------------
def test_inter_exceedance_times_hand_case() -> None:
    mask = np.array([False, True, True, False, False, True, False, True])
    assert inter_exceedance_times(mask).tolist() == [1, 3, 2]
    with pytest.raises(DiagnosticsError, match="N >= 2"):
        inter_exceedance_times(np.array([False, True, False]))
    with pytest.raises(DiagnosticsError, match="boolean"):
        inter_exceedance_times(np.array([0, 1, 1]))


def test_extremal_index_hand_case_t_gt_2() -> None:
    res = extremal_index_fs(np.array([1, 1, 20, 1, 1, 25]))
    assert res.theta == 1849 / 2682
    assert res.theta == pytest.approx(0.689411, abs=1e-6)
    assert res.branch == "T>2"
    assert res.mean_cluster_size == 2682 / 1849
    assert res.n_exceedances == 7


def test_extremal_index_hand_case_t_le_2_clipped() -> None:
    assert Fraction(2 * 6**2, 4 * 10) == Fraction(9, 5)  # 1.8 before clipping
    res = extremal_index_fs(np.array([1, 2, 2, 1]))
    assert (res.theta, res.branch, res.mean_cluster_size) == (1.0, "T<=2", 1.0)


def test_extremal_index_errors() -> None:
    with pytest.raises(DiagnosticsError, match="N >= 2"):
        extremal_index_fs(np.array([], dtype=np.int64))
    with pytest.raises(DiagnosticsError, match="positive"):
        extremal_index_fs(np.array([0, 3]))
    with pytest.raises(DiagnosticsError, match="integer"):
        extremal_index_fs(np.array([1.0, 3.0]))


def _theta_at_98(x: np.ndarray) -> float:
    u = tail_threshold(x, Fraction(98, 100))  # empirical 0.98 quantile, D29 convention
    return extremal_index_fs(inter_exceedance_times(exceedance_mask(x, u))).theta


def test_extremal_index_iid_is_one() -> None:
    assert _theta_at_98(rng().standard_normal(200_000)) == 1.0


@pytest.mark.parametrize("a", [0.6, 0.3])
def test_extremal_index_armax_frechet(a: float) -> None:
    """ARMAX X_t = max(a X_{t-1}, (1-a) Z_t), unit Frechet Z: extremal index 1 - a (A3)."""
    n = 200_000
    z = -1.0 / np.log(rng().uniform(size=n))
    x = np.empty(n)
    x[0] = z[0]
    for t in range(1, n):
        x[t] = max(a * x[t - 1], (1 - a) * z[t])
    assert abs(_theta_at_98(x) - (1 - a)) < 0.05


# --- Ferro-Segers intervals declustering (D36 (c)) -------------------------------
def spikes(positions: list[int], values: list[float], n: int = 200) -> np.ndarray:
    """A loss series that is 0 except for ``values`` at ``positions`` (exceedances of u = 0)."""
    x = np.zeros(n)
    x[positions] = values
    return x


def test_decluster_known_clusters_with_estimated_theta() -> None:
    """T = [1, 1, 48, 1, 69]: theta = 2 * 115^2 / (5 * 6718), floor(5 theta) = 3, so C = 4 and
    r = T_(4) = 1; the ties at 1 reduce C to 3."""
    x = spikes([0, 1, 2, 50, 51, 120], [3.0, 9.0, 4.0, 2.0, 1.0, 5.0])
    res = decluster_fs(x, 0.0)
    assert Fraction(2 * 115**2, 5 * 6718) * 5 // 1 == 3
    assert res.cluster_index.tolist() == [0, 0, 0, 1, 1, 2]
    assert res.cluster_maxima.tolist() == [9.0, 2.0, 5.0]
    assert (res.n_clusters, res.run_length) == (3, 1)
    assert res.cluster_index.dtype == np.int64


def test_decluster_ties_rule_and_explicit_theta() -> None:
    x = spikes([0, 5, 10, 15, 17, 22], [1.0, 2.0, 3.0, 4.0, 6.0, 5.0])  # T = [5, 5, 5, 2, 5]
    half = decluster_fs(x, 0.0, theta=Fraction(1, 2))  # C = 3, r = T_(3) = 5: nothing exceeds it
    assert (half.n_clusters, half.run_length) == (1, 5)
    assert half.cluster_index.tolist() == [0] * 6 and half.cluster_maxima.tolist() == [6.0]
    four = decluster_fs(x, 0.0, theta=Fraction(4, 5))  # C = 5, r = T_(5) = 2, no tie at the cut
    assert (four.n_clusters, four.run_length) == (5, 2)
    assert four.cluster_index.tolist() == [0, 1, 2, 3, 3, 4]
    assert four.cluster_maxima.tolist() == [1.0, 2.0, 3.0, 6.0, 5.0]
    exact = decluster_fs(x, 0.0, theta=Fraction(3, 5))  # floor(3) exactly: C = 4, r = 5 (tie)
    assert exact.n_clusters == 1


def test_decluster_theta_one_and_small_theta() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 6.0, 5.0]
    x = spikes([0, 5, 10, 15, 20, 25], values)  # regular spacing: the estimate is clipped to 1
    assert extremal_index_fs(inter_exceedance_times(x > 0)).theta == 1.0
    for res in (decluster_fs(x, 0.0), decluster_fs(x, 0.0, theta=Fraction(1))):
        assert (res.n_clusters, res.run_length) == (6, 0)  # C = N: T_(C) := 0
        assert res.cluster_index.tolist() == [0, 1, 2, 3, 4, 5]
        assert res.cluster_maxima.tolist() == values
    for small in (Fraction(0), Fraction(1, 100), Fraction(19, 100)):  # floor(5 theta) = 0
        res = decluster_fs(x, 0.0, theta=small)
        assert (res.n_clusters, res.cluster_maxima.tolist()) == (1, [6.0])


@pytest.mark.parametrize("q", [Fraction(80, 100), Fraction(90, 100), Fraction(95, 100)])
def test_decluster_partition_maxima_and_shift_invariance(q: Fraction) -> None:
    x = np.round(rng().standard_t(4, 2000) * 1000)  # integer-valued: x + c is exact
    u = tail_threshold(x, q)
    res = decluster_fs(x, u)
    exc = x[x > u]
    idx = res.cluster_index
    assert idx.size == exc.size and idx[0] == 0 and idx[-1] == res.n_clusters - 1
    assert set(np.diff(idx).tolist()) <= {0, 1}  # contiguous in time, every cluster non-empty
    assert res.cluster_maxima.size == res.n_clusters
    for j in range(res.n_clusters):
        assert res.cluster_maxima[j] == exc[idx == j].max()
    times = inter_exceedance_times(x > u)
    assert np.array_equal(np.diff(idx) == 1, times > res.run_length)
    t = [int(v) for v in times]  # independent exact theta (T>2 branch) and the pinned rule
    theta = min(
        Fraction(1),
        Fraction(2 * sum(v - 1 for v in t) ** 2, len(t) * sum((v - 1) * (v - 2) for v in t)),
    )
    c0 = 1 + math.floor(theta * len(t))
    cut = 0 if c0 == len(t) + 1 else sorted(t, reverse=True)[c0 - 1]
    assert res.run_length == cut
    assert res.n_clusters == 1 + sum(v > cut for v in t) <= c0
    shifted = decluster_fs(x + 4096.0, u + 4096.0)
    assert np.array_equal(shifted.cluster_index, idx)
    assert np.array_equal(shifted.cluster_maxima, res.cluster_maxima + 4096.0)
    assert (shifted.n_clusters, shifted.run_length) == (res.n_clusters, res.run_length)


@pytest.mark.parametrize("a", [0.3, 0.6])
def test_decluster_armax_end_to_end(a: float) -> None:
    """ARMAX has extremal index 1 - a: clusters per exceedance and theta land near it."""
    x = armax_frechet(rng().uniform(size=20000), a)
    u = tail_threshold(x, Fraction(95, 100))
    res = decluster_fs(x, u)
    n_exc = int((x > u).sum())
    assert res.n_clusters / n_exc == pytest.approx(1 - a, abs=0.1)
    assert n_exc / res.n_clusters == pytest.approx(1 / (1 - a), rel=0.2)  # mean cluster size


def test_decluster_errors() -> None:
    x = spikes([0, 5, 10], [1.0, 2.0, 3.0])
    with pytest.raises(DiagnosticsError, match="u: must be finite"):
        decluster_fs(x, math.inf)
    with pytest.raises(DiagnosticsError, match="N >= 2"):
        decluster_fs(x, 2.5)  # one exceedance
    for bad in (0.5, Fraction(3, 2), Fraction(-1, 2)):
        with pytest.raises(DiagnosticsError, match="theta"):
            decluster_fs(x, 0.0, theta=bad)  # type: ignore[arg-type]
    with pytest.raises(DiagnosticsError, match="finite"):
        decluster_fs([1.0, math.nan, 2.0], 0.0)


# --- Engle-Ng sign-bias tests (D39 (e)) -----------------------------------------------------
def _sign_bias_inputs(leverage: float) -> tuple[np.ndarray, np.ndarray]:
    """Residuals whose next-day variance is ``1 + leverage`` after a loss shock ``eps > 0``."""
    g = np.random.default_rng(39)
    zs = g.standard_normal(3000)
    eps = np.empty(3000)
    scale = np.ones(3000)
    for t in range(3000):
        if t > 0 and eps[t - 1] > 0:
            scale[t] = math.sqrt(1 + leverage)
        eps[t] = 0.01 * scale[t] * zs[t]
    return eps, eps / 0.01  # "standardized" by a filter that ignores the leverage


def test_sign_bias_matches_statsmodels_ols() -> None:
    eps, z = _sign_bias_inputs(0.5)
    res = sign_bias(eps, z)
    e, y = -eps[:-1], z[1:] ** 2
    neg = (e < 0).astype(float)
    cols = {"sign": neg, "negative_size": neg * e, "positive_size": (1 - neg) * e}
    for name, x in cols.items():
        fit = OLS(y, np.column_stack([np.ones(y.size), x])).fit()
        assert getattr(res, name) == pytest.approx((fit.tvalues[1], fit.pvalues[1]), rel=1e-9)
    joint = OLS(y, np.column_stack([np.ones(y.size), *cols.values()])).fit()
    assert res.joint_lm == pytest.approx(y.size * joint.rsquared, rel=1e-9)
    assert (res.joint_df, res.nobs) == (3, 2999)
    assert res.joint_p < 1e-6 and res.sign[0] > 0 and res.sign[1] < 1e-6  # leverage is found
    scaled = sign_bias(eps * 100.0, z)  # the scale of eps does not matter
    assert scaled.joint_lm == pytest.approx(res.joint_lm, rel=1e-9)
    assert scaled.negative_size[0] == pytest.approx(res.negative_size[0], rel=1e-9)


def test_sign_bias_not_rejected_without_leverage_and_errors() -> None:
    eps, z = _sign_bias_inputs(0.0)
    res = sign_bias(eps, z)
    assert min(res.joint_p, res.sign[1], res.negative_size[1], res.positive_size[1]) > 0.01
    with pytest.raises(DiagnosticsError, match="same length"):
        sign_bias(eps, z[:-1])
    with pytest.raises(DiagnosticsError, match="same length"):
        sign_bias(eps[:5], z[:5])
    with pytest.raises(DiagnosticsError, match="constant"):
        sign_bias(np.abs(eps), z)  # every shock is a loss shock: S- is constant
    with pytest.raises(DiagnosticsError, match="constant"):
        sign_bias(eps, np.ones(3000))
    with pytest.raises(DiagnosticsError, match="finite"):
        sign_bias(np.where(np.arange(3000) == 7, np.nan, eps), z)
