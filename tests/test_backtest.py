"""Backtests (D42): reference values R5-R10, DM vs statsmodels, mask, families, determinism.

Synthetic data only. Tolerances are those of methodology Appendix A (R5: 5e-7 on 6-dp values,
1e-9 on the E[FZ0] identity, 1e-12 on homogeneity; R6: 5e-7, 5e-5 on the 4-dp LR value; R8/R9
exact at 4 dp; R10 exact) and relative 1e-10 against statsmodels (A5b).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import integrate, stats

from niftytail import backtest as bt
from niftytail import rolling
from niftytail.config import Config, load_config
from niftytail.threshold_study import garch_returns

N01 = NormalDist()
LEVELS = ("0.95", "0.99", "0.995", "0.999")
SENS = ("pot_gpd@0.86", "pot_gpd@0.9", "pot_gpd@0.91", "garch_evt@0.86", "garch_evt@0.9")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "base.toml")


def _gen(*key: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=key)))


def _f(x: Any) -> float:
    return float(np.asarray(x))


# ---------------------------------------------------------------------------
# R5 FZ0
# ---------------------------------------------------------------------------
def _ve(a: float) -> tuple[float, float]:
    v = N01.inv_cdf(a)
    return v, -N01.pdf(v) / a


@pytest.mark.parametrize(
    ("a", "y", "ref"),
    [
        (0.05, -1.0, 0.521445),
        (0.05, -2.0, 3.964933),
        (0.05, -3.0, 13.660902),
        (0.05, None, 0.521445),  # the tie Y = v
        (0.01, -2.0, 0.853140),
        (0.01, -3.0, 26.128862),
    ],
)
def test_r5_worked_examples(a: float, y: float | None, ref: float) -> None:
    v, e = _ve(a)
    assert abs(_f(bt.fz0(v if y is None else y, v, e, a)) - ref) <= 5e-7


@pytest.mark.parametrize(("a", "ref"), [(0.05, 0.724022), (0.01, 0.980284)])
def test_r5_expected_score_is_log_minus_e(a: float, ref: float) -> None:
    v, e = _ve(a)

    def g(y: float) -> float:
        return _f(bt.fz0(y, v, e, a)) * N01.pdf(y)

    lo = integrate.quad(g, -np.inf, v, epsabs=1e-13, epsrel=1e-13)[0]
    hi = integrate.quad(g, v, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    assert abs(lo + hi - math.log(-e)) <= 1e-9
    assert abs(math.log(-e) - ref) <= 5e-7


def test_r5_homogeneity_and_conversion() -> None:
    y, v1, e1, v2, e2, k = -2.5, -1.6, -2.1, -1.9, -2.4, 3.7
    d1 = _f(bt.fz0(y, v1, e1, 0.05)) - _f(bt.fz0(y, v2, e2, 0.05))
    d2 = _f(bt.fz0(k * y, k * v1, k * e1, 0.05)) - _f(bt.fz0(k * y, k * v2, k * e2, 0.05))
    assert abs(d2 - d1) <= 1e-12
    var_l = N01.inv_cdf(0.95)
    es_l = N01.pdf(var_l) / 0.05
    yy, vv, ee = bt.to_return_space(3.0, var_l, es_l)
    assert _f(yy) == -3.0 and _f(vv) == -var_l and _f(ee) == -es_l
    assert abs(_f(bt.fz0(yy, vv, ee, 0.05)) - 13.660902) <= 5e-7


@pytest.mark.parametrize(
    "args",
    [
        (math.nan, -1.0, -2.0, 0.05),
        (-1.0, math.nan, -2.0, 0.05),
        (-1.0, -1.0, math.nan, 0.05),
        (-1.0, -1.0, -2.0, math.nan),
        (-math.inf, -1.0, -2.0, 0.05),
        (-1.0, -1.0, -2.0, 0.0),
        (-1.0, -1.0, -2.0, 1.0),
        (-1.0, 0.0, -1.0, 0.05),
        (-1.0, 0.5, -1.0, 0.05),
        (-1.0, -2.0, -1.0, 0.05),
    ],
    ids=[
        "y nan",
        "v nan",
        "e nan",
        "alpha nan",
        "y -inf",
        "alpha 0",
        "alpha 1",
        "v 0",
        "v>0",
        "e>v",
    ],
)
def test_r5_invalid_inputs_raise(args: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        bt.fz0(*args)


def test_r5_boundary_e_equals_v_accepted() -> None:
    assert abs(_f(bt.fz0(-1.0, -1.5, -1.5, 0.05)) - (1.0 + math.log(1.5) - 1.0)) <= 1e-12


# ---------------------------------------------------------------------------
# R6, R6b, R7 Kupiec and equal-tailed binomial
# ---------------------------------------------------------------------------
def test_r6_kupiec_values() -> None:
    assert abs(bt.kupiec_critical(0.05) - 3.841459) <= 5e-7
    assert abs(_f(bt.kupiec_lr(5, 250, 0.01)) - 1.9568) <= 5e-5
    assert abs(_f(bt.kupiec_lr(5, 250, 0.01)) - 1.956810) <= 5e-7
    lr0 = _f(bt.kupiec_lr(0, 250, 0.01))
    assert abs(lr0 - (-2 * 250 * math.log(0.99))) <= 1e-12
    assert abs(lr0 - 5.025168) <= 5e-7 and lr0 > bt.kupiec_critical(0.05)


def _region(accept: np.ndarray[Any, Any]) -> tuple[int, int]:
    idx = np.flatnonzero(accept)
    assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))  # an interval
    return int(idx[0]), int(idx[-1])


LR_REGIONS = {
    (5650, 0.05): (251, 315),
    (5650, 0.01): (43, 71),
    (250, 0.01): (1, 6),
    (250, 0.05): (7, 19),
}
BIN_REGIONS = {
    (5650, 0.05): (251, 315), (5650, 0.01): (42, 72), (5650, 0.001): (2, 11),
    (6650, 0.05): (298, 368), (6650, 0.01): (51, 83), (6650, 0.001): (2, 12),
    (4770, 0.05): (209, 268), (4770, 0.01): (35, 62), (4770, 0.001): (1, 9),
    (250, 0.05): (6, 20), (250, 0.01): (0, 6), (250, 0.001): (0, 2),
}  # fmt: skip


def _lr_region(t: int, p0: float) -> tuple[int, int]:
    return _region(bt.kupiec_lr(np.arange(t + 1), t, p0) <= bt.kupiec_critical(0.05))


def _bin_region(t: int, p0: float) -> tuple[int, int]:
    return _region(bt.binomial_equal_tailed(np.arange(t + 1), t, p0) > 0.05)


@pytest.mark.parametrize(("cell", "ref"), LR_REGIONS.items())
def test_r6_lr_acceptance_regions(cell: tuple[int, float], ref: tuple[int, int]) -> None:
    assert _lr_region(*cell) == ref


@pytest.mark.parametrize(("cell", "ref"), BIN_REGIONS.items())
def test_r7_equal_tailed_regions(cell: tuple[int, float], ref: tuple[int, int]) -> None:
    assert _bin_region(*cell) == ref


def test_r7_example_and_never_interchange() -> None:
    assert abs(_f(bt.binomial_equal_tailed(5, 250, 0.01)) - 0.215625) <= 5e-7
    rows = {(5650, 0.05): True, (5650, 0.01): False, (250, 0.05): False, (250, 0.01): False}
    for cell, same in rows.items():
        assert (_lr_region(*cell) == _bin_region(*cell)) is same
    assert _f(bt.binomial_equal_tailed(0, 250, 0.01)) > 0.05  # x = 0 accepted by D19 ...
    assert _f(bt.kupiec_lr(0, 250, 0.01)) > bt.kupiec_critical(0.05)  # ... rejected by LR


def test_kupiec_input_checks() -> None:
    for args in ((1, 0, 0.01), (1, 10, 0.0), (1, 10, 1.0), (11, 10, 0.01), (-1, 10, 0.01)):
        with pytest.raises(ValueError):
            bt.kupiec_lr(*args)


# ---------------------------------------------------------------------------
# R8, R9, R10
# ---------------------------------------------------------------------------
R8 = [
    (0.01, 0.01, 0.0523), (0.01, 0.0125, 0.4508), (0.01, 0.015, 0.9295), (0.01, 0.02, 1.0),
    (0.05, 0.05, 0.0472), (0.05, 0.055, 0.3879), (0.05, 0.06, 0.9070), (0.05, 0.07, 1.0),
]  # fmt: skip
R9 = [
    (0.01, 0.01, 1.000, 0.0523), (0.01, 0.05, 1.084, 0.0623),
    (0.01, 0.10, 1.200, 0.0764), (0.01, 0.20, 1.475, 0.1099),
    (0.05, 0.05, 1.000, 0.0472), (0.05, 0.10, 1.111, 0.0598),
    (0.05, 0.20, 1.375, 0.0906), (0.05, 0.30, 1.714, 0.1296),
]  # fmt: skip


@pytest.mark.parametrize(("p0", "p", "ref"), R8)
def test_r8_exact_size_power(p0: float, p: float, ref: float) -> None:
    assert round(bt.kupiec_power_exact(5650, p0, p, 0.05), 4) == ref


@pytest.mark.parametrize(("p0", "pi11", "infl", "ref"), R9)
def test_r9_markov_true_size(p0: float, pi11: float, infl: float, ref: float) -> None:
    assert round(bt.kupiec_size_markov(5650, p0, pi11, 0.05), 4) == ref
    assert round(bt.variance_inflation(p0, pi11), 3) == infl


def test_markov_size_edge_cases() -> None:
    assert math.isnan(bt.kupiec_size_markov(100, 0.01, math.nan, 0.05))
    with pytest.raises(ValueError):
        bt.kupiec_size_markov(100, 0.01, 1.5, 0.05)
    # pi11 = p0 is independence: the Markov size equals the i.i.d. exact size
    assert bt.kupiec_size_markov(300, 0.05, 0.05, 0.05) == pytest.approx(
        bt.kupiec_power_exact(300, 0.05, 0.05, 0.05), rel=1e-12
    )


def test_r10_integer_rules() -> None:
    assert bt.dm_lag(5650) == 9 and bt.dm_lag(5716) == 9
    assert bt.block_length(2000) == 13 and bt.block_length(5650) == 18
    assert bt.block_length(5716) == 18
    for t in range(1, 20001, 7):  # exact rule agrees with the float formula away from ties
        assert bt.dm_lag(t) == math.floor(4 * (t / 100) ** (2 / 9))
        b = bt.block_length(t)
        assert (b - 1) ** 3 < t <= b**3


# ---------------------------------------------------------------------------
# DM vs statsmodels; Holm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lag", [9, 18])
def test_dm_matches_statsmodels_hac(lag: int) -> None:
    rng = _gen(42, 10)
    e = rng.standard_normal(600)
    d = 0.1 + np.convolve(e, [1.0, 0.6, 0.3])[:600]  # serially correlated differences
    ours = bt.diebold_mariano(d, lag)
    fit = sm.OLS(d, np.ones((d.size, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
    )
    assert ours["d_bar"] == pytest.approx(fit.params[0], rel=1e-10)
    assert ours["omega"] == pytest.approx(fit.bse[0] ** 2 * d.size, rel=1e-10)
    assert ours["dm"] == pytest.approx(fit.tvalues[0], rel=1e-10)
    assert ours["p"] == pytest.approx(2 * stats.norm.sf(abs(fit.tvalues[0])), rel=1e-10)


def test_dm_degenerate_cases() -> None:
    assert bt.diebold_mariano([1.0], 9)["flag"] == "too_few_dates"
    out = bt.diebold_mariano(np.zeros(50), 9)
    assert out["flag"] == "omega_nonpositive" and math.isnan(out["dm"]) and math.isnan(out["p"])


def test_holm_hand_set_with_ties() -> None:
    p = [0.01, 0.04, 0.03, 0.03, 0.005, math.nan]
    adj, rej = bt.holm(p)
    want = [0.05, 0.12, 0.12, 0.12, 0.03]
    for got, w in zip(adj[:5], [want[0], want[1], want[2], want[3], want[4]], strict=True):
        assert got == pytest.approx(w, rel=1e-12)
    assert math.isnan(adj[5])
    assert rej == [True, False, False, False, True, False]
    assert bt.holm([0.02, 0.02]) == ([0.04, 0.04], [True, True])


# ---------------------------------------------------------------------------
# Christoffersen
# ---------------------------------------------------------------------------
def _lr_ind_hand(n00: int, n01: int, n10: int, n11: int) -> float:
    def xl(k: float, p: float) -> float:
        return 0.0 if k == 0 else k * math.log(p)

    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
    l0 = xl(n00 + n10, 1 - p) + xl(n01 + n11, p)
    return -2 * (l0 - l1)


def test_christoffersen_hand_counts_and_lrs() -> None:
    h = [0, 0, 1, 1, 0, 1, 0, 0, 0, 1]
    assert bt.transition_counts(h) == (3, 3, 2, 1)
    out = bt.christoffersen(h, 0.05, 99, _gen(1), _gen(2))
    ref = -2 * (5 * math.log(5 / 9) + 4 * math.log(4 / 9)) + 2 * (
        3 * math.log(0.5) + 3 * math.log(0.5) + 2 * math.log(2 / 3) + math.log(1 / 3)
    )
    assert out["lr_ind"] == pytest.approx(ref, rel=1e-12)
    assert out["lr_cc"] == pytest.approx(ref + _f(bt.kupiec_lr(4, 10, 0.05)), rel=1e-12)
    h0 = [0, 1, 0, 1, 0]  # n11 = 0: 0 ln 0 = 0
    assert bt.transition_counts(h0) == (0, 2, 2, 0)
    assert bt.christoffersen(h0, 0.05, 9, _gen(1), _gen(2))["lr_ind"] == pytest.approx(
        _lr_ind_hand(0, 2, 2, 0), rel=1e-12
    )


@pytest.mark.parametrize("ones", [False, True])
def test_christoffersen_x_zero_and_x_t(ones: bool) -> None:
    h = np.full(50, ones, dtype=bool)
    out = bt.christoffersen(h, 0.01, 999, _gen(1), _gen(2))
    assert out["lr_ind"] == 0.0 and out["p_ind"] == 1.0
    assert out["lr_cc"] == pytest.approx(_f(bt.kupiec_lr(50 if ones else 0, 50, 0.01)), rel=1e-15)
    assert 0.0 < out["p_cc"] <= 1.0
    single = bt.christoffersen([1], 0.01, 9, _gen(1), _gen(2))
    assert single["lr_ind"] == 0.0


def _mc_by_hand(
    h: np.ndarray[Any, Any], p0: float, p_sim: float, reps: int, key: int, cc: bool
) -> float:
    """Independent p-value: same draws, LRs by the hand formula, (1 + count)/(R + 1)."""
    t = h.size

    def stat(s: np.ndarray[Any, Any]) -> float:
        n = bt.transition_counts(s)
        lr = _lr_ind_hand(*n)
        if cc:
            x = int(s.sum())
            ph = x / t
            ll0 = (t - x) * math.log(1 - p0) + (x * math.log(p0) if x else 0.0)
            ll1 = (t - x) * math.log(1 - ph) if x < t else 0.0
            ll1 += x * math.log(ph) if x else 0.0
            lr += -2 * (ll0 - ll1)
        return lr

    obs = stat(h)
    sims = _gen(key).random((reps, t)) < p_sim
    count = sum(stat(s) >= obs - 1e-9 * max(1.0, abs(obs)) for s in sims)
    return (1 + count) / (reps + 1)


def test_christoffersen_mc_nulls_ind_at_phat_cc_at_p0() -> None:
    h = np.zeros(50, dtype=bool)
    h[[3, 4, 10, 11, 12, 20, 30, 31, 40, 45]] = True  # p_hat = 0.2, clustered
    p0, reps = 0.01, 999
    out = bt.christoffersen(h, p0, reps, _gen(7), _gen(8))
    assert out["p_ind"] == _mc_by_hand(h, p0, 0.2, reps, 7, cc=False)
    assert out["p_ind"] != _mc_by_hand(h, p0, p0, reps, 7, cc=False)
    assert out["p_cc"] == _mc_by_hand(h, p0, p0, reps, 8, cc=True)
    assert out["p_cc"] != _mc_by_hand(h, p0, 0.2, reps, 8, cc=True)
    assert out["p_cc"] == 1 / (reps + 1)


# ---------------------------------------------------------------------------
# Traffic light
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("t", "partial"), [(5650, 150), (5716, 216), (500, None)])
def test_traffic_light_blocks(cfg: Config, t: int, partial: int | None) -> None:
    blocks = bt.traffic_light(np.zeros(t, dtype=int), np.ones(t, dtype=bool), cfg.backtest)
    full = [b for b in blocks if b["zone"] is not None]
    assert len(full) == t // 250 and all(b["n_obs"] == 250 for b in full)
    assert blocks[0]["start_row"] == 0 and blocks[-1]["end_row"] == t - 1
    for a, b in itertools.pairwise(blocks):
        assert b["start_row"] == a["end_row"] + 1
    if partial is None:
        assert len(blocks) == len(full)
    else:
        assert blocks[-1]["n_obs"] == partial and blocks[-1]["zone"] is None
    zc = bt._zone_counts(blocks)
    assert zc["zones"]["green"] == t // 250
    assert zc["partial_block"] == (None if partial is None else {"n_obs": partial, "breaches": 0})


def test_traffic_light_zone_boundaries_and_flag(cfg: Config) -> None:
    h = np.zeros(1000, dtype=int)
    for i, k in enumerate((4, 5, 9, 10)):
        h[250 * i : 250 * i + k] = 1
    ok = np.ones(1000, dtype=bool)
    zones = [b["zone"] for b in bt.traffic_light(h, ok, cfg.backtest)]
    assert zones == ["green", "yellow", "yellow", "red"]
    ok[3] = False  # an inadmissible row in block 0 that carries a hit: not counted, flagged
    b0 = bt.traffic_light(h, ok, cfg.backtest)[0]
    assert b0["breaches"] == 3 and b0["inadmissible_rows"] == 1 and b0["zone"] == "green"
    assert bt._zone_counts(bt.traffic_light(h, ok, cfg.backtest))["flagged_blocks"] == 1


# ---------------------------------------------------------------------------
# ES residual test (A5a checks (1)-(4))
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_es_residual_uniform_under_null(cfg: Config) -> None:
    ps = []
    for r in range(200):
        rng = _gen(42, 900, r)
        z = 0.25 * rng.standard_normal(100)
        out = bt.es_residual_test(
            2.0 * (1.0 + z), np.full(100, 2.0), cfg.backtest.es_residual_bootstrap_reps, 20, rng
        )
        ps.append(out["p"])
    assert stats.kstest(ps, "uniform").pvalue >= 0.01
    assert 0.01 <= np.mean(np.asarray(ps) <= 0.05) <= 0.10


def test_es_residual_detects_underforecast(cfg: Config) -> None:
    rng = _gen(42, 901)
    loss = 2.0 + rng.exponential(0.5, 56)
    out = bt.es_residual_test(
        loss, np.full(56, 2.5 / 1.2), cfg.backtest.es_residual_bootstrap_reps, 20, rng
    )
    assert not out["descriptive"] and out["p"] < 0.01


def test_es_residual_hand_mean_and_descriptive() -> None:
    out = bt.es_residual_test([3.0, 2.5, 4.0], [2.0, 2.5, 2.0], 99, 20, _gen(1))
    assert out == {"n_breach": 3, "z_bar": 0.5, "p": out["p"], "descriptive": True}
    assert math.isnan(out["p"])
    out19 = bt.es_residual_test(np.full(19, 3.0), np.full(19, 2.0), 99, 20, _gen(1))
    assert out19["descriptive"] and math.isnan(out19["p"]) and out19["z_bar"] == 0.5
    out20 = bt.es_residual_test(np.full(20, 3.0), np.full(20, 2.0), 99, 20, _gen(1))
    assert not out20["descriptive"] and out20["p"] == 1 / 100  # centred z~* = 0 < z_bar: count 0
    empty = bt.es_residual_test([], [], 99, 0, _gen(1))
    assert empty["descriptive"] and math.isnan(empty["z_bar"])


# ---------------------------------------------------------------------------
# Breach-rate CIs
# ---------------------------------------------------------------------------
def test_block_and_clopper_pearson() -> None:
    assert bt.clopper_pearson(0, 10)[0] == 0.0 and bt.clopper_pearson(10, 10)[1] == 1.0
    lo, hi = bt.clopper_pearson(5, 250)
    assert stats.binom.sf(4, 250, lo) == pytest.approx(0.025, rel=1e-9)
    assert stats.binom.cdf(5, 250, hi) == pytest.approx(0.025, rel=1e-9)
    h = (_gen(3).random(2000) < 0.05).astype(int)
    lo, hi = bt.block_ci_rate(h, 13, 1999, _gen(4))
    assert lo < h.mean() < hi
    # the block sums reproduce a direct resample built from the same starts
    starts = _gen(4).integers(0, 2000 - 13 + 1, size=(1999, math.ceil(2000 / 13)))
    rates = [np.concatenate([h[s : s + 13] for s in row])[:2000].mean() for row in starts]
    assert (lo, hi) == tuple(np.quantile(rates, [0.025, 0.975]))
    assert bt.block_ci_rate(h[:5], 99, 9, _gen(4)) == (h[:5].mean(), h[:5].mean())  # b capped at n
    with pytest.raises(ValueError):
        bt.block_ci_rate([], 3, 9, _gen(4))


# ---------------------------------------------------------------------------
# Constructed panel: admissibility, mask, scores, alignment, evaluate
# ---------------------------------------------------------------------------
KEYS = (*bt.PRIMARY_KEYS, "pot_gpd@0.86")
LOSS = [0.5, 1.2, 2.0, 0.1, 0.3, 1.8, 0.2, 0.0, -0.5, 3.0]


def _panel(t: int = 10, loss: list[float] | None = None) -> pd.DataFrame:
    data: dict[str, Any] = {"loss": loss if loss is not None else LOSS[:t]}
    for k in KEYS:
        for c in LEVELS:
            data[f"{k}:var:{c}"] = np.full(t, 1.0)
            data[f"{k}:es:{c}"] = np.full(t, 1.5)
        data[f"{k}:flags"] = [""] * t
    idx = pd.MultiIndex.from_arrays(
        [list(range(1000, 1000 + t)), [f"d{i:04d}" for i in range(t)]], names=["j", "date"]
    )
    return pd.DataFrame(data, index=idx)


def _set(tab: pd.DataFrame, row: int, key: str, kind: str, value: Any) -> None:
    cols = [f"{key}:{kind}:{c}" for c in LEVELS] if kind != "flags" else [f"{key}:flags"]
    for col in cols:
        tab.iloc[row, tab.columns.get_loc(col)] = value


def _inadmissible_panel() -> pd.DataFrame:
    tab = _panel()
    _set(tab, 2, "gaussian", "var", math.nan)  # missing:other
    _set(tab, 4, "pot_gpd", "es", math.inf)  # es_infinite
    _set(tab, 4, "pot_gpd", "flags", "es_infinite")
    _set(tab, 5, "hs", "var", -0.1)  # var_nonpositive
    _set(tab, 6, "fhs", "es", 0.9)  # es_below_var
    for k in ("garch_n", "fhs", "garch_evt"):  # missing:garch_error
        _set(tab, 7, k, "var", math.nan)
        _set(tab, 7, k, "es", math.nan)
        _set(tab, 7, k, "flags", "garch_error")
    _set(tab, 8, "pot_gpd@0.86", "var", math.nan)  # a sensitivity NaN: C unchanged
    _set(tab, 8, "pot_gpd@0.86", "flags", "var_not_in_tail")
    return tab


C_ROWS = [0, 1, 3, 8, 9]


def test_admissibility_reasons_precedence() -> None:
    v = [math.nan, 1.0, 1.0, -1.0, 2.0, 1.0, math.inf, -math.inf, 1.0]
    e = [1.0, math.inf, math.nan, 0.5, 1.0, 1.0, math.inf, 1.0, math.nan]
    f = ["gpd_error", "", "no_pre_window_lag;garch_error", "", "", "", "", "", "near_boundary"]
    assert bt.admissibility(v, e, f) == [
        "missing:gpd_error",
        "es_infinite",
        "missing:garch_error",
        "var_nonpositive",
        "es_below_var",
        "",  # ES = VaR accepted
        "es_infinite",
        "var_nonpositive",
        "missing:other",
    ]


def test_common_mask_constructed_panel(cfg: Config) -> None:
    tab = _inadmissible_panel()
    agg, per = bt.evaluate(tab, cfg)
    cm = agg["common_mask"]["0.99"]
    assert cm["total"] == 10 and cm["common"] == 5 and cm["below_min_share"]
    assert cm["sensitivity_masks"] == {"pot_gpd@0.86": 4}
    assert [e["j"] - 1000 for e in per["excluded"]["0.99"]] == [2, 4, 5, 6, 7]
    assert per["excluded"]["0.99"][4]["reasons"] == {
        k: "missing:garch_error" for k in ("garch_n", "fhs", "garch_evt")
    }
    adm = agg["admissibility"]["0.95"]
    assert adm["gaussian"]["reasons"] == {
        "missing": 1,
        "es_infinite": 0,
        "var_nonpositive": 0,
        "es_below_var": 0,
    }
    assert adm["gaussian"]["missing_by_flag"]["other"] == 1
    assert (
        adm["pot_gpd"]["reasons"]["es_infinite"] == 1
        and adm["hs"]["reasons"]["var_nonpositive"] == 1
    )
    assert adm["fhs"]["reasons"] == {
        "missing": 1,
        "es_infinite": 0,
        "var_nonpositive": 0,
        "es_below_var": 1,
    }
    assert (
        adm["garch_evt"]["missing_by_flag"]["garch_error"] == 1
        and adm["garch_evt"]["admissible"] == 9
    )
    assert adm["pot_gpd@0.86"]["missing_by_flag"]["var_not_in_tail"] == 1
    # mean FZ0 on C by hand (VaR 1, ES 1.5 for every key)
    for lvl, a in (("0.95", 0.05), ("0.99", 0.01)):
        hand = []
        for r in C_ROWS:
            y, v, e = -LOSS[r], -1.0, -1.5
            hand.append(-(1 / (a * e)) * (v - y if y <= v else 0.0) + v / e + math.log(-e) - 1)
        for k in bt.PRIMARY_KEYS:
            assert agg["fz0"][lvl][k] == {"n": 5, "mean": pytest.approx(sum(hand) / 5, rel=1e-14)}
    # coverage on each key's OWN admissible dates
    assert agg["coverage"]["0.99"]["gaussian"]["n"] == 9 and agg["coverage"]["0.99"]["hs"]["n"] == 9
    assert (
        agg["coverage"]["0.99"]["garch_n"]["n"] == 9 and agg["coverage"]["0.999"]["fhs"]["n"] == 8
    )


def test_scores_use_only_mask_dates(cfg: Config) -> None:
    tab = _inadmissible_panel()
    mask = np.zeros(10, dtype=bool)
    mask[C_ROWS] = True
    base = bt.score_models(tab, bt.PRIMARY_KEYS, Fraction("0.99"), mask)
    moved = tab.copy()
    for r in (2, 4, 5, 6, 7):
        moved.iloc[r, moved.columns.get_loc("loss")] = 1e6
        for k in bt.PRIMARY_KEYS:
            _set(moved, r, k, "var", 7.0)
    again = bt.score_models(moved, bt.PRIMARY_KEYS, Fraction("0.99"), mask)
    for k in bt.PRIMARY_KEYS:
        assert base[k].size == 5 and np.array_equal(base[k], again[k])
    a1, _ = bt.evaluate(tab, cfg)
    a2, _ = bt.evaluate(moved, cfg)
    assert a1["fz0"] == a2["fz0"] and a1["dm"] == a2["dm"]


def test_sensitivity_nan_does_not_change_c(cfg: Config) -> None:
    tab = _inadmissible_panel()
    clean = tab.copy()
    _set(clean, 8, "pot_gpd@0.86", "var", 1.0)
    a1, _ = bt.evaluate(tab, cfg)
    a2, _ = bt.evaluate(clean, cfg)
    for lvl in ("0.95", "0.99"):
        assert a1["common_mask"][lvl]["common"] == a2["common_mask"][lvl]["common"] == 5
        assert a1["fz0"][lvl]["pot_gpd"] == a2["fz0"][lvl]["pot_gpd"]
    assert a1["common_mask"]["0.99"]["sensitivity_masks"]["pot_gpd@0.86"] == 4
    assert a2["common_mask"]["0.99"]["sensitivity_masks"]["pot_gpd@0.86"] == 5


def test_w2000_pass_mask_is_its_own(cfg: Config) -> None:
    p1000 = _inadmissible_panel()
    p2000 = _panel()
    _set(p2000, 0, "garch_n", "var", math.nan)  # e.g. no_pre_window_lag on the first target
    _set(p2000, 0, "garch_n", "flags", "no_pre_window_lag")
    a2000, per2000 = bt.evaluate(p2000, cfg, window=2000)
    assert a2000["window"] == 2000 and a2000["common_mask"]["0.99"]["common"] == 9
    assert [e["j"] for e in per2000["excluded"]["0.99"]] == [1000]
    bt.evaluate(p1000, cfg, window=1000)  # evaluating the other pass changes nothing
    assert bt.to_json(bt.evaluate(p2000, cfg, window=2000)[0]) == bt.to_json(a2000)


def test_inadmissible_values_still_raise_in_fz0() -> None:
    tab = _inadmissible_panel()
    for r, k in ((2, "gaussian"), (4, "pot_gpd"), (5, "hs"), (6, "fhs"), (7, "garch_n")):
        y, v, e = bt.to_return_space(
            tab["loss"].iloc[r], tab[f"{k}:var:0.99"].iloc[r], tab[f"{k}:es:0.99"].iloc[r]
        )
        with pytest.raises(ValueError):
            bt.fz0(y, v, e, 0.01)


def test_alignment_shift_changes_hits(cfg: Config) -> None:
    t = 20
    loss = [10.0 if j % 2 else 0.0 for j in range(t)]
    tab = _panel(t, loss)
    var = np.array([5.0 if j % 2 else 20.0 for j in range(t)])
    for k in KEYS:
        tab[f"{k}:var:0.99"] = var
        tab[f"{k}:es:0.99"] = var + 1.0
    agg, per = bt.evaluate(tab, cfg)
    assert agg["coverage"]["0.99"]["hs"]["breaches"] == 10  # every odd row: L = 10 > VaR = 5
    assert per["hits"]["0.99"]["hs"] == "01" * 10
    shifted = tab.copy()
    for k in KEYS:  # row j now carries the VaR of row j - 1 (a one-row misalignment)
        shifted[f"{k}:var:0.99"] = np.roll(var, 1)
        shifted[f"{k}:es:0.99"] = np.roll(var, 1) + 1.0
    assert bt.evaluate(shifted, cfg)[0]["coverage"]["0.99"]["hs"]["breaches"] == 0


def test_evaluate_input_errors(cfg: Config) -> None:
    tab = _panel()
    bad = tab.copy()
    bad.iloc[0, bad.columns.get_loc("loss")] = math.nan
    with pytest.raises(ValueError, match="finite"):
        bt.evaluate(bad, cfg)
    with pytest.raises(ValueError, match="primary keys missing"):
        bt.evaluate(tab.drop(columns=[c for c in tab.columns if c.startswith("fhs:")]), cfg)
    extra = tab.copy()
    extra["nonsense:flags"] = ""
    with pytest.raises(ValueError, match="unknown keys"):
        bt.evaluate(extra, cfg)
    with pytest.raises(ValueError, match="primary keys missing"):
        bt.common_mask({"gaussian": np.ones(3, dtype=bool)})
    with pytest.raises(ValueError, match="1e-4"):
        bt.seed_key(1, 1000, "pot_gpd@0.12345", Fraction("0.99"))


def test_seed_key_encoding() -> None:
    assert bt.seed_key(bt.TEST_IND, 1000, "gaussian", Fraction("0.95")) == (42, 1, 1000, 0, 0, 9500)
    assert bt.seed_key(bt.TEST_CC, 2000, "garch_evt@0.9", Fraction("0.99")) == (
        42,
        2,
        2000,
        5,
        9000,
        9900,
    )
    assert bt.seed_key(6, 1000, "pot_gpd@0.86", Fraction("0.99")) == (42, 6, 1000, 2, 8600, 9900)


def test_all_inadmissible_key_and_pi11_undefined(cfg: Config) -> None:
    tab = _panel()
    _set(tab, slice(None), "pot_gpd@0.86", "var", math.nan)  # type: ignore[arg-type]
    agg, _ = bt.evaluate(tab, cfg)
    assert agg["coverage"]["0.99"]["pot_gpd@0.86"] == {
        "n": 0,
        "breaches": 0,
        "note": "no_admissible_dates",
    }
    assert math.isnan(agg["coverage"]["0.999"]["pot_gpd@0.86"]["p_hat"])
    assert agg["fz0"]["0.99"]["pot_gpd@0.86"]["n"] == 0
    # LOSS has hits (L > 1) at rows 1, 2, 5, 9 only: n10 + n11 = 3 transitions out of a hit
    assert agg["coverage"]["0.99"]["hs"]["markov"]["reason"] is None
    one = _panel(t=4, loss=[0.0, 0.0, 0.0, 2.0])  # the only hit is the last row
    m = bt.evaluate(one, cfg)[0]["coverage"]["0.99"]["hs"]["markov"]
    assert m["reason"] == "pi11_undefined" and math.isnan(m["pi11_hat"]) and math.isnan(m["size"])
    assert "null" in bt.to_json(
        {"x": math.nan, "y": (np.int64(1), np.bool_(True), np.float64(1.5))}
    )


# ---------------------------------------------------------------------------
# Determinism, order and worker independence
# ---------------------------------------------------------------------------
def _cov_args(cfg: Config) -> list[tuple[Any, ...]]:
    rng = _gen(42, 20)
    out = []
    for i, key in enumerate(("gaussian", "garch_evt", "pot_gpd@0.9")):
        for c in (Fraction("0.95"), Fraction("0.99")):
            loss = rng.standard_normal(400)
            var = np.full(400, 1.2 + 0.3 * i)
            out.append((loss, var, var + 0.4, key, c, cfg, 1000))
    return out


def test_mc_bootstrap_independent_of_order_and_workers(cfg: Config) -> None:
    small = dataclasses.replace(
        cfg,
        backtest=dataclasses.replace(
            cfg.backtest, christoffersen_mc_reps=2500, es_residual_min_breaches=5
        ),
    )
    args = _cov_args(small)
    forward = [bt._coverage(*a) for a in args]
    backward = [bt._coverage(*a) for a in reversed(args)][::-1]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = list(pool.map(bt._coverage, *zip(*args, strict=True)))
    assert bt.to_json({"r": forward}) == bt.to_json({"r": backward}) == bt.to_json({"r": pooled})
    # a different key gives different draws (seed keys differ)
    assert bt.seed_key(1, 1000, "gaussian", Fraction("0.95")) != bt.seed_key(
        1, 1000, "garch_evt", Fraction("0.95")
    )


# ---------------------------------------------------------------------------
# evaluate() end-to-end on a rolling.py table; per-date writer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rolled(cfg: Config) -> pd.DataFrame:
    n, t0, w = 300, 150, 100
    r, _ = garch_returns(_gen(41, 0).standard_t(6, n) / math.sqrt(1.5), 0.02, 0.08, 0.90)
    small = dataclasses.replace(cfg, design=dataclasses.replace(cfg.design, train_size=t0))
    table, _ = rolling.run_rolling(
        -0.01 * np.asarray(r), [f"day{i:05d}" for i in range(n)], small, w, 4
    )
    return table


def test_evaluate_end_to_end(cfg: Config, rolled: pd.DataFrame, tmp_path: Path) -> None:
    agg, per = bt.evaluate(rolled, cfg, window=100)
    agg2, per2 = bt.evaluate(rolled, cfg, window=100)
    assert bt.to_json(agg) == bt.to_json(agg2) and bt.to_json(per) == bt.to_json(per2)
    text = bt.to_json(agg)
    assert "day0" not in text and "excluded" not in agg and "hits" not in agg
    assert set(per) == {"window", "excluded", "traffic_light", "hits"}
    assert agg["sensitivity_keys"] == sorted((*SENS, "garch_evt@0.91"))
    assert len(agg["dm"]["unconditional"]) == 6 and len(agg["dm"]["conditional"]) == 6
    assert len(agg["dm"]["descriptive"]) == 2 * (9 + 6)
    assert all("p_holm" in e and "reject" in e for e in agg["dm"]["conditional"])
    assert all("reject" not in e for e in agg["dm"]["descriptive"])
    assert set(agg["coverage"]) == set(LEVELS)
    assert agg["coverage"]["0.995"]["hs"]["descriptive"] is True
    assert set(agg["traffic_light"]) == set(agg["primary_keys"] + agg["sensitivity_keys"])
    json.loads(text)
    path = bt.write_per_date(per, tmp_path / "reports" / "runs" / "abc")
    assert path.read_text(encoding="utf-8") == bt.to_json(per)
    with pytest.raises(ValueError, match="reports/runs"):
        bt.write_per_date(per, tmp_path / "elsewhere")
