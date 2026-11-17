"""Tests for the Day-7A threshold procedure core (D34). Synthetic data only."""

from __future__ import annotations

import ast
import dataclasses
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from niftytail import threshold
from niftytail.config import Config, ThresholdConfig, load_config
from niftytail.gpd import (
    EstimationError,
    delta_ci,
    excesses,
    fit_mle,
    profile_ci_xi,
    se_xi,
    tail_risk,
)
from niftytail.losses import tail_threshold
from niftytail.threshold import (
    Steps,
    ThresholdDecision,
    candidate_table,
    forward_stop,
    gof_ad_bootstrap,
    select_threshold,
    split_half,
    split_half_flag,
    stable_region,
    window_convergence_share,
)

SEED = 20261111
B = 9  # tiny bootstrap size: these tests check structure and wiring, not p-value accuracy
ROOT = Path(__file__).resolve().parents[1]
SS = np.random.SeedSequence
Q = Fraction
ALPHA = Q(1, 10)
Q_MAX = Q(9, 10)
CANDIDATE_KEYS = {
    "q", "u", "n_u", "n_u_window", "xi_hat", "beta_hat", "beta_star", "se_xi", "xi_ci",
    "xi_ci_lower_open", "ad_stat", "ad_p", "xi_hat_scipy_gof", "fit_error",
    "convergence_share", "flags", "var95", "var95_ci", "es95", "es95_ci", "var99",
    "var99_ci", "es99", "es99_ci",
}  # fmt: skip
ROW_KEYS = CANDIDATE_KEYS | {"fs_stat", "rejected", "stable_fail"}


def rng(child: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(SS(SEED).spawn(child + 1)[child]))


def gpd_sample(xi: float, n: int, child: int) -> np.ndarray:
    return np.asarray(stats.genpareto.rvs(xi, size=n, random_state=rng(child)))


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(ROOT / "configs" / "base.toml")


@pytest.fixture(scope="module")
def th(cfg: Config) -> ThresholdConfig:
    return cfg.threshold


@pytest.fixture(scope="module")
def x() -> np.ndarray:
    """2000 i.i.d. GPD(0.25) losses: the excesses over every threshold are exactly GPD."""
    return gpd_sample(0.25, 2000, 0)


@pytest.fixture(scope="module")
def decision(x: np.ndarray, cfg: Config) -> ThresholdDecision:
    return select_threshold(x, cfg, ss=SS(SEED), gof_reps=B, convergence_share=False)


@pytest.fixture
def fake_gof(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the bootstrap by a constant (statistic, p, xi_scipy) = (1, 0.5, 0)."""
    monkeypatch.setattr(threshold, "gof_ad_bootstrap", lambda y, *, reps, ss: (1.0, 0.5, 0.0))


# --- forward_stop ---------------------------------------------------------------------------
def test_forward_stop_hand_values() -> None:
    fs, k_hat = forward_stop([0.5], ALPHA)
    assert fs == pytest.approx([math.log(2)], rel=1e-15) and k_hat == 0
    t = -math.log(0.95)  # 0.05129...
    fs, k_hat = forward_stop([0.05, 0.05, 0.05], ALPHA)
    assert fs == pytest.approx([t, t, t], rel=1e-12) and k_hat == 3  # k_hat = m
    assert forward_stop([], ALPHA) == ([], 0)


def test_forward_stop_takes_the_maximum_not_the_first_crossing() -> None:
    fs, k_hat = forward_stop([0.2, 0.01, 0.01, 0.9], ALPHA)
    a, b = -math.log(0.8), -math.log(0.99)
    assert fs == pytest.approx([a, (a + b) / 2, (a + 2 * b) / 3, (a + 2 * b + math.log(10)) / 4])
    assert fs[0] > ALPHA and fs[1] > ALPHA and fs[2] <= ALPHA and fs[3] > ALPHA
    assert k_hat == 3


def test_forward_stop_p_one_is_inf_and_stops() -> None:
    fs, k_hat = forward_stop([0.05, 1.0, 0.0, 0.0], ALPHA)
    assert fs[0] <= ALPHA and fs[1:] == [math.inf] * 3
    assert k_hat == 1
    assert forward_stop([1.0, 0.0], ALPHA) == ([math.inf, math.inf], 0)


def test_forward_stop_comparison_is_inclusive() -> None:
    assert forward_stop([0.0, 0.0], Q(0)) == ([0.0, 0.0], 2)


@pytest.mark.parametrize("child", range(6))
def test_forward_stop_matches_numpy(child: int) -> None:
    p = rng(child).uniform(0.0, 0.3, size=16)
    ref = np.cumsum(-np.log1p(-p)) / np.arange(1, 17)
    ok = np.nonzero(ref <= 0.1)[0]
    fs, k_hat = forward_stop(p.tolist(), ALPHA)
    assert fs == pytest.approx(ref.tolist(), rel=1e-12)
    assert k_hat == (int(ok[-1]) + 1 if ok.size else 0)


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan])
def test_forward_stop_rejects_invalid_p(bad: float) -> None:
    with pytest.raises(ValueError, match="pvalues"):
        forward_stop([0.5, bad], ALPHA)


# --- stable_region on hand-built tables -----------------------------------------------------
def table(th: ThresholdConfig, k_hat: int = 0) -> list[dict[str, Any]]:
    """A perfectly flat 16-row table: every window passes every tolerance."""
    return [
        {"q": q, "xi_hat": 0.25, "var99": 2.0, "es99": 4.0, "flags": [], "rejected": k < k_hat}
        for k, q in enumerate(th.grid)
    ]


def put(rows: list[dict[str, Any]], th: ThresholdConfig, q: str, **kw: Any) -> None:
    rows[th.grid.index(Q(q))].update(kw)


def test_stable_region_smallest_q_wins(th: ThresholdConfig) -> None:
    q_star, failure, fails = stable_region(table(th), 0, q_max=Q_MAX, th=th)
    assert (q_star, failure) == (Q("0.80"), None)
    assert fails[:11] == [None] * 11 and fails[11:] == ["above_q_max"] * 5

    q_star, failure, fails = stable_region(table(th, 3), 3, q_max=Q_MAX, th=th)  # q_FS = 0.83
    assert (q_star, failure) == (Q("0.83"), None)
    assert fails[:4] == ["rejected", "rejected", "rejected", None]

    rows = table(th)
    put(rows, th, "0.81", xi_hat=0.50)  # breaks the windows starting at 0.80 and at 0.81
    q_star, failure, fails = stable_region(rows, 0, q_max=Q_MAX, th=th)
    assert (q_star, failure) == (Q("0.82"), None)
    assert fails[:3] == ["tol_xi@81/100", "tol_xi@41/50", None]


def test_stable_region_real_tolerances(th: ThresholdConfig) -> None:
    for key, inside, outside, name in [
        ("xi_hat", 0.34, 0.36, "tol_xi"),  # 0.10 absolute
        ("xi_hat", 0.16, 0.14, "tol_xi"),
        ("var99", 2.09, 2.11, "tol_var99"),  # 5% relative
        ("var99", 1.91, 1.89, "tol_var99"),
        ("es99", 4.39, 4.41, "tol_es99"),  # 10% relative
        ("es99", 3.61, 3.59, "tol_es99"),
    ]:
        rows = table(th)
        put(rows, th, "0.82", **{key: inside})
        assert stable_region(rows, 0, q_max=Q_MAX, th=th)[2][0] is None
        put(rows, th, "0.82", **{key: outside})
        assert stable_region(rows, 0, q_max=Q_MAX, th=th)[2][0] == f"{name}@41/50"


@pytest.mark.parametrize(
    ("key", "tie", "name"),
    [
        ("xi_hat", 0.375, "tol_xi"),
        ("xi_hat", 0.125, "tol_xi"),
        ("var99", 2.125, "tol_var99"),
        ("var99", 1.875, "tol_var99"),
        ("es99", 4.5, "tol_es99"),
        ("es99", 3.5, "tol_es99"),
    ],
)
def test_stable_region_boundaries_are_inclusive(
    th: ThresholdConfig, key: str, tie: float, name: str
) -> None:
    """Dyadic tolerances make the tie exactly representable: equality passes, 4 ulp more fails."""
    dyadic = dataclasses.replace(th, tol_xi=Q(1, 8), tol_var99_rel=Q(1, 16), tol_es99_rel=Q(1, 8))
    base = table(th)[0][key]
    rows = table(th)
    put(rows, th, "0.83", **{key: tie})
    assert stable_region(rows, 0, q_max=Q_MAX, th=dyadic)[2][0] is None
    beyond = tie + math.copysign(4 * math.ulp(tie), tie - base)
    put(rows, th, "0.83", **{key: beyond})
    assert stable_region(rows, 0, q_max=Q_MAX, th=dyadic)[2][0] == f"{name}@83/100"


def test_stable_region_window_must_exist(th: ThresholdConfig) -> None:
    high = Q("0.95")  # lift q_max so that only the missing window can disqualify
    q_star, failure, fails = stable_region(table(th, 13), 13, q_max=high, th=th)  # q_FS = 0.93
    assert (q_star, failure) == (None, "no_stable_region")
    assert fails == ["rejected"] * 13 + ["no_window"] * 3
    q_star, failure, _ = stable_region(table(th, 12), 12, q_max=high, th=th)  # q_FS = 0.92
    assert (q_star, failure) == (Q("0.92"), None)


def test_stable_region_q_max_binds_q_star_but_not_the_window(th: ThresholdConfig) -> None:
    rows = table(th, 10)  # q_FS = 0.90 = q_max: the window 0.90..0.93 lies above q_max
    for q in ("0.91", "0.92", "0.93"):
        put(rows, th, q, flags=["below_n_min_window"])
    q_star, failure, fails = stable_region(rows, 10, q_max=Q_MAX, th=th)
    assert (q_star, failure) == (Q("0.90"), None)
    assert fails[10] is None and fails[11:] == ["above_q_max"] * 5

    put(rows, th, "0.93", xi_hat=0.50)  # a member above q_max is still tested
    q_star, failure, fails = stable_region(rows, 10, q_max=Q_MAX, th=th)
    assert (q_star, failure, fails[10]) == (None, "no_stable_region", "tol_xi@93/100")

    q_star, failure, fails = stable_region(table(th, 11), 11, q_max=Q_MAX, th=th)  # q_FS = 0.91
    assert (q_star, failure) == (None, "no_stable_region")
    assert fails == ["rejected"] * 11 + ["above_q_max"] * 5


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"flags": ["fit_failed"], "xi_hat": None, "var99": None, "es99": None}, "fit_failed"),
        ({"flags": ["es_infinite"], "es99": None}, "es_infinite"),
        ({"var99": None}, "not_finite"),
        ({"es99": math.inf}, "not_finite"),
        ({"var99": math.nan}, "not_finite"),
        ({"rejected": True}, "rejected"),  # impossible after ForwardStop; the check exists
    ],
)
def test_stable_region_bad_member_fails_its_windows(
    th: ThresholdConfig, update: dict[str, Any], reason: str
) -> None:
    rows = table(th)
    put(rows, th, "0.82", **update)
    q_star, failure, fails = stable_region(rows, 0, q_max=Q_MAX, th=th)
    assert (q_star, failure) == (Q("0.83"), None)
    assert fails[:4] == [f"{reason}@41/50"] * 3 + [None]


def test_stable_region_zero_base_is_undefined(th: ThresholdConfig) -> None:
    rows = table(th)
    put(rows, th, "0.80", var99=0.0)
    q_star, _, fails = stable_region(rows, 0, q_max=Q_MAX, th=th)
    assert fails[0] == "not_finite@4/5" and q_star == Q("0.81")


def test_stable_region_rule_failures(th: ThresholdConfig) -> None:
    assert stable_region(table(th, 16), 16, q_max=Q_MAX, th=th) == (
        None,
        "all_rejected",
        ["rejected"] * 16,
    )
    rows = table(th)
    for i, r in enumerate(rows):
        r["xi_hat"] = 0.2 * i  # no two neighbours within 0.10
    q_star, failure, fails = stable_region(rows, 0, q_max=Q_MAX, th=th)
    assert (q_star, failure) == (None, "no_stable_region")
    assert all(f is not None for f in fails)


# --- gof_ad_bootstrap -----------------------------------------------------------------------
def test_gof_is_deterministic_and_bounded(x: np.ndarray) -> None:
    y = excesses(x, tail_threshold(x, Q("0.90")))
    a = gof_ad_bootstrap(y, reps=19, ss=SS(5))
    assert gof_ad_bootstrap(y, reps=19, ss=SS(5)) == a
    stat, p, xi = a
    assert stat > 0 and 1 / 20 <= p <= 1
    assert p * 20 == pytest.approx(round(p * 20), abs=1e-12)  # (1 + count) / (B + 1)
    assert all(isinstance(v, float) for v in a)
    assert len({gof_ad_bootstrap(y, reps=19, ss=SS(s))[1] for s in range(5, 10)}) > 1  # seeded
    assert xi == pytest.approx(stats.genpareto.fit(y, floc=0)[0], abs=1e-12)


def test_gof_refits_fix_loc_at_zero(x: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fit scipy performs (the data fit and all B replicate refits) passes floc = 0."""
    y = excesses(x, tail_threshold(x, Q("0.90")))
    calls: list[tuple[np.ndarray, tuple[Any, ...], dict[str, Any]]] = []
    original = stats.genpareto.fit

    def recording_fit(data: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((np.array(data), args, dict(kwargs)))
        return original(data, *args, **kwargs)

    monkeypatch.setattr(stats.genpareto, "fit", recording_fit)
    reps = 19
    stat, _, xi = gof_ad_bootstrap(y, reps=reps, ss=SS(5))

    # the null fit to the data, the observed-statistic refit, and one refit per replicate
    assert len(calls) == reps + 2
    assert all(args == () and kwargs == {"floc": 0} for _, args, kwargs in calls)
    assert sum(np.array_equal(data, y) for data, _, _ in calls) == 2  # the data fits
    assert len({data.tobytes() for data, _, _ in calls}) == reps + 1  # distinct replicates
    assert all(data.shape == y.shape and np.all(data > 0) for data, _, _ in calls)

    # the statistic is the AD distance to the loc = 0 fit (independent formula)
    c, loc, scale = original(y, floc=0)
    assert loc == 0 and xi == c
    ys, n = np.sort(y), y.size
    i = np.arange(1, n + 1)
    dist = stats.genpareto(c, loc=0, scale=scale)
    ad = -n - np.sum((2 * i - 1) / n * (dist.logcdf(ys) + dist.logsf(ys[::-1])))
    assert stat == pytest.approx(ad, rel=1e-10)


@pytest.mark.slow
def test_gof_p_values_roughly_uniform_under_the_null() -> None:
    g = rng(1)
    seeds = SS(SEED, spawn_key=(7,)).spawn(40)
    p = np.array(
        [
            gof_ad_bootstrap(stats.genpareto.rvs(0.25, size=100, random_state=g), reps=39, ss=s)[1]
            for s in seeds
        ]
    )
    assert stats.kstest(p, "uniform").statistic < 0.30  # loose: the 1% critical value is 0.25
    assert 0.35 < p.mean() < 0.65


def test_gof_rejects_a_lognormal_tail() -> None:
    y = stats.lognorm.rvs(0.25, size=300, random_state=rng(2))  # mode far from zero
    assert gof_ad_bootstrap(y, reps=99, ss=SS(3))[1] == pytest.approx(0.01)


# --- candidate_table ------------------------------------------------------------------------
def test_candidate_counts_and_threshold_convention(
    decision: ThresholdDecision, x: np.ndarray, cfg: Config
) -> None:
    rows = decision.candidates
    xs = np.sort(x)
    assert [r["q"] for r in rows] == list(cfg.threshold.grid) and len(rows) == 16
    assert [r["n_u"] for r in rows] == list(range(400, 99, -20))  # no ties: n - ceil(n q)
    assert [r["n_u_window"] for r in rows] == [Q(n) for n in range(200, 49, -10)]
    for k, r in enumerate(rows):
        assert r["u"] == xs[1600 + 20 * k - 1]  # u = L_(ceil(n q)), 1-based
        assert r["n_u"] == int(np.sum(x > r["u"]))
        assert all(isinstance(r[key], int) for key in ("n_u",))
        assert set(r) == ROW_KEYS


def test_candidate_n_u_is_the_actual_count_under_ties(cfg: Config, fake_gof: None) -> None:
    x = gpd_sample(0.25, 2000, 3)
    order = np.argsort(x)
    x[order[1598:1602]] = x[order[1599]]  # ties straddle rank 1600 (q = 0.80)
    row = candidate_table(x, cfg, ss=SS(1), gof_reps=B)[0]
    assert row["u"] == np.sort(x)[1599]
    assert row["n_u"] == int(np.sum(x > row["u"])) == 398  # not n - ceil(n q) = 400


def test_candidate_fields_equal_direct_gpd_calls(
    decision: ThresholdDecision, x: np.ndarray, cfg: Config
) -> None:
    tol = float(cfg.threshold.xi_zero_tol)
    for r in decision.candidates:
        y = excesses(x, r["u"])
        fit = fit_mle(y, min_excesses=2, xi_zero_tol=tol, settings=cfg.gpd)
        ci = profile_ci_xi(y, fit, xi_zero_tol=tol, settings=cfg.gpd)
        assert (r["xi_hat"], r["beta_hat"]) == (fit.xi, fit.beta)
        assert r["beta_star"] == fit.beta - fit.xi * r["u"]
        assert r["se_xi"] == se_xi(fit.xi, r["n_u"])
        assert r["xi_ci"] == [ci.lower, ci.upper] and r["xi_ci_lower_open"] is ci.lower_open
        assert abs(r["xi_hat_scipy_gof"] - r["xi_hat"]) < 1e-3  # two optimisers, one optimum
        assert r["fit_error"] is None and r["convergence_share"] is None
        for label, c in (("95", Q("0.95")), ("99", Q("0.99"))):
            p = 1 - c
            if label == "95" and r["q"] == Q("0.95"):
                continue
            var, es = tail_risk(r["u"], fit.xi, fit.beta, 2000, r["n_u"], p, tol)
            assert (r[f"var{label}"], r[f"es{label}"]) == (var, es)
            for name in ("var", "es"):
                d = delta_ci(
                    fit, r["u"], 2000, r["n_u"], p, name, xi_zero_tol=tol, settings=cfg.gpd
                )
                assert r[f"{name}{label}_ci"] == [d.lower, d.upper]


def test_candidate_flags_on_an_exact_gpd_sample(decision: ThresholdDecision) -> None:
    for r in decision.candidates:
        expected = ["below_n_min_window"] if r["q"] > Q_MAX else []
        if r["q"] == Q("0.95"):  # p = 0.05 is not < N_u / n = 0.05
            expected.append("var95_not_in_tail")
            assert r["var95"] is None and r["es95"] is None
            assert r["var95_ci"] is None and r["es95_ci"] is None
        else:
            assert r["es95"] > r["var95"] > r["u"]
        assert r["flags"] == expected
        assert r["es99"] > r["var99"] > r["u"]  # VaR99 stays inside the tail at every q


def test_candidate_gof_seed_layout(decision: ThresholdDecision, x: np.ndarray, cfg: Config) -> None:
    """Full sample = spawn(3)[0]; candidate k = its spawn(16)[k - 1] (real numpy spawning)."""
    full = SS(SEED).spawn(3)[0]
    rows = candidate_table(x, cfg, ss=full, gof_reps=B)
    assert set(rows[0]) == CANDIDATE_KEYS
    for r, d in zip(rows, decision.candidates, strict=True):
        assert r == {k: d[k] for k in CANDIDATE_KEYS}
    k = 7
    y = excesses(x, rows[k - 1]["u"])
    stat, p, xi = gof_ad_bootstrap(y, reps=B, ss=SS(SEED).spawn(3)[0].spawn(16)[k - 1])
    assert (rows[k - 1]["ad_stat"], rows[k - 1]["ad_p"], rows[k - 1]["xi_hat_scipy_gof"]) == (
        stat,
        p,
        xi,
    )


def test_candidate_infinite_es_and_xi_gap(cfg: Config, fake_gof: None) -> None:
    rows = candidate_table(gpd_sample(1.6, 2000, 2), cfg, ss=SS(1), gof_reps=B)
    r = rows[0]
    assert r["xi_hat"] >= 1
    assert "es_infinite" in r["flags"] and r["flags"].count("es_infinite") == 1
    assert r["es95"] is None and r["es99"] is None and r["es99_ci"] is None
    assert math.isfinite(r["var99"]) and r["var99_ci"] is not None and r["se_xi"] is not None
    assert "ci_unavailable" not in r["flags"]
    assert "xi_gap" in r["flags"]  # the fake scipy xi is 0: |0 - xi_hat| > 0.05


def test_candidate_irregular_fit_has_no_se_or_delta_ci(cfg: Config, fake_gof: None) -> None:
    r = candidate_table(gpd_sample(-0.7, 2000, 2), cfg, ss=SS(1), gof_reps=B)[0]
    assert -1 < r["xi_hat"] <= -0.5
    assert r["se_xi"] is None and r["var99_ci"] is None and r["es99_ci"] is None
    assert r["xi_ci"] is not None  # the profile CI does not need regularity
    assert math.isfinite(r["var99"]) and math.isfinite(r["es99"])
    assert "ci_unavailable" in r["flags"]


def test_candidate_profile_failure_is_flagged(
    x: np.ndarray, cfg: Config, fake_gof: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_crossing(*args: Any, **kwargs: Any) -> Any:
        raise EstimationError("profile CI: no upper crossing by profile_xi_max")

    monkeypatch.setattr(threshold, "profile_ci_xi", no_crossing)
    r = candidate_table(x, cfg, ss=SS(1), gof_reps=B)[0]
    assert r["xi_ci"] is None and r["xi_ci_lower_open"] is None and r["se_xi"] is not None
    assert r["flags"] == ["xi_gap", "ci_unavailable"]


def test_candidate_scipy_failure_is_fit_failed(
    x: np.ndarray, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gof(y: np.ndarray, *, reps: int, ss: np.random.SeedSequence) -> tuple[float, float, float]:
        if y.size == 300:  # the q = 0.85 candidate
            raise RuntimeError("secret data value 1.2345")
        return 1.0, 0.01, 0.25

    monkeypatch.setattr(threshold, "gof_ad_bootstrap", gof)
    rows = candidate_table(x, cfg, ss=SS(1), gof_reps=B)
    bad = rows[5]
    assert bad["flags"] == ["fit_failed"] and bad["fit_error"] == "RuntimeError"
    assert bad["ad_p"] == 1.0 and bad["ad_stat"] is None
    assert all(bad[k] is None for k in ("xi_hat", "beta_hat", "se_xi", "var99", "es99"))
    assert "1.2345" not in json.dumps(threshold._jsonable(rows))  # type name only (D23)
    assert all("fit_failed" not in r["flags"] for i, r in enumerate(rows) if i != 5)
    fs, k_hat = forward_stop([r["ad_p"] for r in rows], ALPHA)
    assert k_hat == 5 and fs[5:] == [math.inf] * 11  # p = 1 stops ForwardStop


def test_tied_tail_fails_every_fit_and_the_rule(cfg: Config) -> None:
    x = np.asarray(stats.norm.rvs(size=2000, random_state=rng(3)))
    x[::4] = 10.0  # a quarter of each half sits at the maximum: no loss exceeds any u_k
    d = select_threshold(x, cfg, ss=SS(1), gof_reps=B, convergence_share=False)
    assert (d.q_star, d.q_fs, d.k_hat, d.rule_failure) == (None, Q("0.80"), 0, "no_stable_region")
    assert d.flags == ()  # q* undefined: halves are reported, nothing is compared
    for steps in (d, *d.halves):
        assert steps.rule_failure == "no_stable_region" and steps.q_star is None
        for r in steps.candidates:
            assert r["n_u"] == 0 and r["u"] == 10.0 and "fit_failed" in r["flags"]
            assert r["fit_error"] == "ValueError" and r["ad_p"] == 1.0
            assert r["fs_stat"] == math.inf and r["rejected"] is False
    assert d.candidates[0]["stable_fail"] == "fit_failed@4/5"
    parsed = json.loads(d.to_json())
    assert parsed["candidates"][0]["fs_stat"] == "inf" and parsed["q_star"] is None


# --- split_half -----------------------------------------------------------------------------
def test_split_half_uses_exactly_the_two_halves(
    decision: ThresholdDecision, x: np.ndarray, cfg: Config
) -> None:
    _, s1, s2 = SS(SEED).spawn(3)
    halves = split_half(x, cfg, seeds=(s1, s2), gof_reps=B)
    assert halves == decision.halves  # select_threshold uses spawn(3)[1] and [2]
    for steps, part, seed in ((halves[0], x[:1000], s1), (halves[1], x[1000:], s2)):
        assert isinstance(steps, Steps) and len(steps.candidates) == 16
        direct = candidate_table(part, cfg, ss=seed, gof_reps=B)
        for r, d in zip(steps.candidates, direct, strict=True):
            assert r["u"] == tail_threshold(part, r["q"])  # the half's own quantile
            assert r["n_u"] == int(np.sum(part > r["u"])) == 2000 * (1 - r["q"]) / 2
            assert r["n_u_window"] == (1 - r["q"]) * 1000  # the rolling window, not the half
            assert {k: r[k] for k in CANDIDATE_KEYS} == d
        own_fs, own_k = forward_stop([r["ad_p"] for r in direct], ALPHA)  # own ForwardStop
        assert steps.k_hat == own_k and [r["fs_stat"] for r in steps.candidates] == own_fs
    assert halves[0].candidates[0]["u"] != halves[1].candidates[0]["u"]


def test_split_half_flag_rule() -> None:
    limit = Q("0.03")
    q = Q("0.85")
    assert not split_half_flag(q, [Q("0.85"), Q("0.85")], limit)
    assert not split_half_flag(q, [Q("0.88"), Q("0.82")], limit)  # exactly 0.03: not flagged
    assert split_half_flag(q, [Q("0.89"), Q("0.85")], limit)
    assert split_half_flag(q, [Q("0.85"), Q("0.81")], limit)
    assert split_half_flag(q, [None, Q("0.85")], limit)  # a half without q*
    assert split_half_flag(q, [Q("0.85"), None], limit)
    assert not split_half_flag(None, [Q("0.85"), None], limit)  # q* undefined: halves only
    assert not split_half_flag(None, [None, None], limit)
    assert 0.88 - 0.85 > 0.03  # the float trap that exact Fractions avoid


def test_select_sets_the_split_half_flag(
    x: np.ndarray, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def steps(q_star: Fraction | None) -> Steps:
        return Steps(q_star=q_star, q_fs=None, k_hat=0, rule_failure=None, candidates=())

    results = iter([steps(Q("0.85")), steps(Q("0.89")), steps(Q("0.85"))])
    monkeypatch.setattr(threshold, "_steps", lambda *a, **k: next(results))
    d = select_threshold(x, cfg, ss=SS(1), gof_reps=B, convergence_share=False)
    assert d.flags == ("split_half_flag",) and d.q_star == Q("0.85")
    assert [h.q_star for h in d.halves] == [Q("0.89"), Q("0.85")]


# --- select_threshold -----------------------------------------------------------------------
def test_select_structure(decision: ThresholdDecision, cfg: Config) -> None:
    grid = cfg.threshold.grid
    for steps in (decision, *decision.halves):
        rows = steps.candidates
        assert len(rows) == 16 and all(set(r) == ROW_KEYS for r in rows)
        assert 0 <= steps.k_hat <= 16
        assert [r["rejected"] for r in rows] == [k < steps.k_hat for k in range(16)]
        fs, k_hat = forward_stop([r["ad_p"] for r in rows], ALPHA)
        assert ([r["fs_stat"] for r in rows], steps.k_hat) == (fs, k_hat)
        assert steps.q_fs == (grid[k_hat] if k_hat < 16 else None)
        assert all(1 / (B + 1) <= r["ad_p"] <= 1 for r in rows)
        passing = [r["q"] for r in rows if r["stable_fail"] is None]
        if steps.rule_failure is None:
            assert steps.q_star == passing[0] and steps.q_fs is not None
            assert steps.q_fs <= steps.q_star <= Q_MAX
        else:
            assert steps.q_star is None and passing == []
            assert steps.rule_failure in ("all_rejected", "no_stable_region")
        assert stable_region(rows, k_hat, q_max=Q_MAX, th=cfg.threshold) == (
            steps.q_star,
            steps.rule_failure,
            [r["stable_fail"] for r in rows],
        )
    assert set(decision.flags) <= {"split_half_flag"}
    assert ("split_half_flag" in decision.flags) == split_half_flag(
        decision.q_star, [h.q_star for h in decision.halves], Q("0.03")
    )


def test_select_is_deterministic_and_does_not_advance_the_seed(
    decision: ThresholdDecision, x: np.ndarray, cfg: Config
) -> None:
    ss = SS(SEED)
    ss.spawn(5)  # an already-used SeedSequence gives the same answer
    again = select_threshold(x, cfg, ss=ss, gof_reps=B, convergence_share=False)
    assert again == decision
    assert again.to_json().encode() == decision.to_json().encode()
    other = select_threshold(x, cfg, ss=SS(SEED + 1), gof_reps=B, convergence_share=False)
    assert [r["ad_p"] for r in other.candidates] != [r["ad_p"] for r in decision.candidates]
    assert [r["xi_hat"] for r in other.candidates] == [r["xi_hat"] for r in decision.candidates]


def test_decision_json_round_trip(decision: ThresholdDecision) -> None:
    text = decision.to_json()
    assert text.endswith("}\n") and "\r" not in text and "NaN" not in text
    parsed = json.loads(text)
    assert set(parsed) == {
        "q_star", "q_fs", "k_hat", "rule_failure", "candidates", "halves", "flags",
    }  # fmt: skip
    assert list(parsed) == sorted(parsed)
    assert json.dumps(parsed, sort_keys=True, indent=2) + "\n" == text  # floats keep their repr
    assert [r["q"] for r in parsed["candidates"]] == [str(r["q"]) for r in decision.candidates]
    assert parsed["candidates"][0]["q"] == "4/5" and parsed["candidates"][0]["n_u_window"] == "200"
    assert [Q(r["q"]) for r in parsed["halves"][1]["candidates"]] == [
        r["q"] for r in decision.halves[1].candidates
    ]
    for got, row in zip(parsed["candidates"], decision.candidates, strict=True):
        assert got["xi_hat"] == row["xi_hat"] and got["var99_ci"] == row["var99_ci"]
        assert got["flags"] == row["flags"] and got["rejected"] is row["rejected"]
    q_star = decision.q_star
    assert parsed["q_star"] == (None if q_star is None else str(q_star))


@pytest.mark.parametrize("n", [1999, 2001, 1000, 0])
def test_select_rejects_any_other_length(cfg: Config, n: int) -> None:
    with pytest.raises(ValueError, match="train_size"):
        select_threshold(np.ones(n), cfg, ss=SS(1), gof_reps=B)
    with pytest.raises(ValueError, match="train_size"):
        select_threshold(np.ones((2, 1000)), cfg, ss=SS(1), gof_reps=B)


def test_select_needs_n_min(x: np.ndarray, cfg: Config) -> None:
    unset = dataclasses.replace(cfg, threshold=dataclasses.replace(cfg.threshold, n_min=None))
    with pytest.raises(ValueError, match="n_min"):
        select_threshold(x, unset, ss=SS(1), gof_reps=B)


def test_threshold_module_does_not_touch_the_data_layer() -> None:
    source = Path(threshold.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {m for m in modules if m.startswith("niftytail")} == {
        "niftytail.config",
        "niftytail.gpd",
        "niftytail.losses",
    }
    assert not any(part in ("data", "pandas", "pathlib") for m in modules for part in m.split("."))
    assert "load_raw" not in source and "open(" not in source


# --- window_convergence_share ---------------------------------------------------------------
def test_window_convergence_share_small_series(cfg: Config) -> None:
    tol, q = float(cfg.threshold.xi_zero_tol), Q("0.80")
    kw: dict[str, Any] = {"xi_zero_tol": tol, "settings": cfg.gpd}
    y = gpd_sample(0.25, 130, 4)
    assert window_convergence_share(y, q, 100, **kw) == 1.0  # 31 windows, 20 excesses each

    # 100 equal low values, then 30 larger GPD values: window s holds s values above the
    # constant, so s = 0 has no excess, s = 1 has one, and s >= 20 has 20 distinct excesses
    mixed = np.concatenate([np.full(100, 3.0), y[:30] + 5.0])
    converged = 0
    for start in range(31):
        w = mixed[start : start + 100]
        try:
            fit_mle(excesses(w, tail_threshold(w, q)), min_excesses=2, **kw)
        except (EstimationError, ValueError):
            continue
        converged += 1
    share = window_convergence_share(mixed, q, 100, **kw)
    assert share == converged / 31 and 11 / 31 <= share <= 29 / 31
    assert window_convergence_share(np.full(120, 3.0), q, 100, **kw) == 0.0
    assert window_convergence_share(y[:100], q, 100, **kw) == 1.0  # a single window
    with pytest.raises(ValueError, match="window"):
        window_convergence_share(y[:99], q, 100, **kw)


def test_select_fills_the_convergence_share_on_a_small_design() -> None:
    """Default call (convergence_share=True) on T0 = 150, W = 100: 51 windows per candidate."""
    # A copy of the live config, not a re-parse: q_max = 0.5 is below the grid, so the rule
    # must fail, and no grid value of q_selected (D37) or q_c (D39) is valid for this design.
    base = load_config(ROOT / "configs" / "base.toml")
    small = dataclasses.replace(
        base,
        design=dataclasses.replace(
            base.design, train_size=150, window=100, sensitivity_windows=(150,)
        ),
        threshold=dataclasses.replace(base.threshold, n_min=50, q_selected=None),
    )
    x = gpd_sample(0.25, 150, 5)
    d = select_threshold(x, small, ss=SS(SEED), gof_reps=B)
    assert d.q_star is None and d.rule_failure in ("all_rejected", "no_stable_region")
    kw: dict[str, Any] = {"xi_zero_tol": float(small.threshold.xi_zero_tol), "settings": small.gpd}
    for r in d.candidates:
        assert isinstance(r["convergence_share"], float) and 0 <= r["convergence_share"] <= 1
        assert "below_n_min_window" in r["flags"]
    q = d.candidates[3]["q"]
    assert d.candidates[3]["convergence_share"] == window_convergence_share(x, q, 100, **kw)
    assert all(r["convergence_share"] is None for h in d.halves for r in h.candidates)
    json.loads(d.to_json())
