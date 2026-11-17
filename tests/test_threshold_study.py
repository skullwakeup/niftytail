"""Tests for the Day-7B A3 threshold-procedure study (D35). Synthetic data only."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import re
import statistics
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats
from statsmodels.stats.proportion import proportion_confint

from niftytail import diagnostics as dg
from niftytail import threshold_study as ts
from niftytail.__main__ import main
from niftytail.config import Config, load_config
from niftytail.gpd import excesses as gpd_excesses
from niftytail.gpd import fit_mle, tail_risk
from niftytail.losses import tail_threshold
from niftytail.threshold import ThresholdDecision, select_threshold

SEED = 20261111
ROOT = Path(__file__).resolve().parents[1]
SS = np.random.SeedSequence
Q = Fraction
RULE_KEYS = (
    "q", "u", "n_u", "xi_hat", "beta_hat", "beta_star", "se_xi", "ad_stat", "ad_p",
    "var95", "es95", "var99", "es99", "fs_stat", "rejected", "stable_fail",
)  # fmt: skip
CI_KEYS = ("xi_ci", "xi_ci_lower_open", "var95_ci", "es95_ci", "var99_ci", "es99_ci")


def rng(child: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(SS(SEED, spawn_key=(1000 + child,))))


def s1_cdf(x: Any, q0: float, xi: float) -> np.ndarray:
    """Independent S1 distribution function (scipy norm and genpareto)."""
    u0 = stats.norm.ppf(q0)
    beta0 = (1 - q0) / stats.norm.pdf(u0)
    x = np.asarray(x, dtype=float)
    tail = q0 + (1 - q0) * stats.genpareto.cdf(x - u0, xi, scale=beta0)
    return np.asarray(np.where(x <= u0, stats.norm.cdf(x), tail))


def _set(text: str, section: str, key: str, value: str) -> str:
    """Replace ``key = ...`` (one or more lines up to the next key) inside ``[section]``."""
    head, sep, tail = text.partition(f"\n[{section}]")
    assert sep
    pattern = rf"^{key} = (\[[^\]]*\]|[^\n#]*)"
    new, count = re.subn(pattern, f"{key} = {value} ", tail, count=1, flags=re.MULTILINE)
    assert count == 1, (section, key)
    return head + sep + new


def tiny_text() -> str:
    """base.toml shrunk to 4 cells, a 6-point grid, n = 200, W = 100, R = 2 and B = 9."""
    text = (ROOT / "configs" / "base.toml").read_text(encoding="utf-8")
    edits = [
        ("design", "train_size", "200"),
        ("design", "window", "100"),
        ("design", "sensitivity_windows", "[200]"),
        ("threshold", "grid", '["0.85", "0.86", "0.87", "0.88", "0.89", "0.90"]'),
        ("threshold", "n_min", "10"),
        ("nmin_study", "n_u", "[10, 50, 75, 100, 150]"),
        ("dependence", "exceedance_levels", '["0.85", "0.90"]'),
        ("simstudy", "sample_size", "200"),
        ("simstudy", "window_refit", "100"),
        ("simstudy", "replications", "2"),
        ("simstudy", "replications_fallback", "2"),
        ("simstudy", "gof_bootstrap_reps", "9"),
        ("simstudy", "s1_q0", '["0.87"]'),
        ("simstudy", "s1_xi", '["0.25"]'),
        ("simstudy", "s3_q0", '"0.87"'),
        ("simstudy", "s3_armax_a", '["0.3"]'),
        ("simstudy", "s4_burn_in", "100"),
        ("simstudy", "s4_reference_size", "20000"),
    ]
    for section, key, value in edits:
        text = _set(text, section, key, value)
    return text


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(ROOT / "configs" / "base.toml")


@pytest.fixture(scope="module")
def tiny_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny") / "configs" / "base.toml"
    path.parent.mkdir()
    path.write_text(tiny_text(), encoding="utf-8", newline="\n")
    return path


@pytest.fixture(scope="module")
def tiny(tiny_path: Path) -> Config:
    return load_config(tiny_path)


@pytest.fixture(scope="module")
def study(tiny: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return ts.run_study(tiny, config_sha256="0" * 64, workers=1)


# --- cells and seeds ------------------------------------------------------------------------
def test_cells_in_the_pinned_order(cfg: Config) -> None:
    study_cells = ts.cells(cfg)
    assert [c.index for c in study_cells] == list(range(10))
    assert [c.dgp for c in study_cells] == ["S1"] * 6 + ["S2", "S3", "S3", "S4"]
    assert [(c.q0, c.xi) for c in study_cells[:6]] == [
        (Q(q0), Q(xi)) for q0 in ("0.85", "0.90") for xi in ("0.10", "0.25", "0.40")
    ]
    assert (study_cells[6].q0, study_cells[6].xi, study_cells[6].a) == (None, Q(1, 4), None)
    assert [(c.q0, c.xi, c.a) for c in study_cells[7:9]] == [
        (Q("0.90"), Q("0.25"), Q("0.3")),
        (Q("0.90"), Q("0.25"), Q("0.6")),
    ]
    assert (study_cells[9].q0, study_cells[9].xi, study_cells[9].a) == (None, None, None)
    assert len({c.label for c in study_cells}) == 10


def _fake_selection(captured: dict[str, Any]) -> Any:
    def fake(x: np.ndarray, cfg: Config, *, ss: Any, gof_reps: int, **kw: Any) -> Any:
        captured.update(x=x.copy(), ss=ss, gof_reps=gof_reps, kw=kw)
        return ThresholdDecision(
            q_star=None,
            q_fs=None,
            k_hat=0,
            rule_failure="no_stable_region",
            candidates=(),
            halves=None,
            flags=(),
        )

    return fake


@pytest.mark.parametrize("index", [0, 5, 6, 8, 9])
@pytest.mark.parametrize("rep", [0, 3, 199])
def test_seed_layout_uses_real_spawn_keys(
    cfg: Config, index: int, rep: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cell = spawn(11)[c]; replication = cell.spawn(R)[r]; data and GoF = its spawn(2)."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(ts, "select_threshold", _fake_selection(captured))
    cell = ts.cells(cfg)[index]
    record = ts.run_replication(cfg, cell, rep)
    rep_ss = SS(cfg.runtime.seed).spawn(11)[index].spawn(cfg.simstudy.replications)[rep]
    data_ss, gof_ss = rep_ss.spawn(2)
    expected = ts.sample(cell, np.random.Generator(np.random.PCG64(data_ss)), cfg)
    assert captured["x"].shape == (2000,) and np.array_equal(captured["x"], expected)
    assert np.array_equal(captured["ss"].generate_state(8), gof_ss.generate_state(8))
    assert captured["gof_reps"] == 199
    assert captured["kw"] == {"convergence_share": False, "halves": False, "intervals": False}
    assert record == {
        "cell": index,
        "rep": rep,
        "k_hat": 0,
        "v": None if cell.q0 is None else 0,
        "q_star": None,
        "rule_failure": "no_stable_region",
        "n_fit_failed": 0,
        "full": None,
        "refit": None,
    }


def test_data_stream_does_not_depend_on_the_gof_settings(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(ts, "select_threshold", _fake_selection(captured))
    cell = ts.cells(cfg)[7]
    ts.run_replication(cfg, cell, 1)
    first = captured["x"]
    other = dataclasses.replace(
        cfg, simstudy=dataclasses.replace(cfg.simstudy, gof_bootstrap_reps=7, replications=50)
    )
    ts.run_replication(other, cell, 1)
    assert np.array_equal(first, captured["x"]) and captured["gof_reps"] == 7
    ts.run_replication(cfg, cell, 2)
    assert not np.array_equal(first, captured["x"])


def test_s4_reference_uses_the_last_child(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(g: np.random.Generator, n: int, ss: Any) -> np.ndarray:
        seen.update(state=g.bit_generator.state, n=n)
        return np.arange(float(n))

    monkeypatch.setattr(ts, "s4_losses", fake)
    ref = ts.s4_reference(cfg)
    expected = np.random.PCG64(SS(cfg.runtime.seed).spawn(11)[10]).state
    assert seen == {"state": expected, "n": 10_000_000}
    assert (ref["n"], ref["batches"]) == (10_000_000, ts.S4_REFERENCE_BATCHES) == (10_000_000, 10)


# --- S1 -------------------------------------------------------------------------------------
S1_CASES = [(Q(q0), Q(xi)) for q0 in ("0.85", "0.90") for xi in ("0.10", "0.25", "0.40")]


@pytest.mark.parametrize(("q0", "xi"), S1_CASES)
def test_s1_round_trip_and_continuity(q0: Fraction, xi: Fraction) -> None:
    q = float(q0)
    v = np.concatenate([np.linspace(0.001, 0.999, 999), [q - 1e-9, q, q + 1e-9, 0.99999]])
    x = ts.s1_ppf(v, q0, xi)
    assert np.all(np.diff(x[:999]) > 0)
    assert s1_cdf(x, q, float(xi)) == pytest.approx(v, abs=1e-12)
    u0 = stats.norm.ppf(q)
    assert float(ts.s1_ppf(q, q0, xi)) == pytest.approx(u0, abs=1e-12)
    # continuous density at u0: dF^-1/dv = 1 / f has the same one-sided limits, 1 / phi(u0)
    h = 1e-6
    left = (ts.s1_ppf(q, q0, xi) - ts.s1_ppf(q - h, q0, xi)) / h
    right = (ts.s1_ppf(q + h, q0, xi) - ts.s1_ppf(q, q0, xi)) / h
    assert left == pytest.approx(1 / stats.norm.pdf(u0), rel=1e-4)
    assert right == pytest.approx(1 / stats.norm.pdf(u0), rel=1e-4)
    assert ts.s1_ppf(1.0, q0, xi) == math.inf and ts.s1_ppf(0.0, q0, xi) == -math.inf


@pytest.mark.parametrize(("q0", "xi"), S1_CASES)
@pytest.mark.parametrize("c", [Q("0.95"), Q("0.99")])
def test_s1_truth_against_numerical_integration(q0: Fraction, xi: Fraction, c: Fraction) -> None:
    var, es = ts.s1_truth(c, q0, xi)
    assert float(s1_cdf(var, float(q0), float(xi))) == pytest.approx(float(c), abs=1e-13)
    # ES = (1 / p) int_c^1 F^-1(v) dv; the (1 - v)^-xi end-point singularity is the weight
    x, q = float(xi), float(q0)
    limit = (1 - q) / stats.norm.pdf(stats.norm.ppf(q)) / x * (1 - q) ** x  # value at v = 1

    def smooth(v: float) -> float:
        return limit if v >= 1 else float(ts.s1_ppf(v, q0, xi)) * (1 - v) ** x

    integral, _ = integrate.quad(
        smooth, float(c), 1.0, weight="alg", wvar=(0.0, -x), epsabs=1e-10, epsrel=1e-10
    )
    assert es == pytest.approx(integral / float(1 - c), rel=1e-7)
    assert es > var


def test_s1_truth_guards() -> None:
    with pytest.raises(ValueError, match="q0 <= c"):
        ts.s1_truth(Q("0.85"), Q("0.90"), Q("0.25"))
    assert ts.s1_truth(Q("0.90"), Q("0.90"), Q("0.25"))[0] == pytest.approx(stats.norm.ppf(0.9))
    for q0, xi in [(Q(0), Q("0.25")), (Q("0.9"), Q(0)), (Q("0.9"), Q(1))]:
        with pytest.raises(ValueError, match="S1"):
            ts.s1_ppf(0.5, q0, xi)


def test_s1_sample_is_uniform_under_its_cdf(cfg: Config) -> None:
    big = dataclasses.replace(cfg, simstudy=dataclasses.replace(cfg.simstudy, sample_size=200_000))
    cell = ts.cells(cfg)[1]  # q0 = 0.85, xi = 0.25
    loss = ts.sample(cell, rng(0), big)
    assert loss.shape == (200_000,)
    assert stats.kstest(s1_cdf(loss, 0.85, 0.25), "uniform").pvalue > 0.01
    var, es = ts.s1_truth(Q("0.99"), Q("0.85"), Q("0.25"))
    assert np.quantile(loss, 0.99) == pytest.approx(var, rel=0.03)
    assert loss[loss > var].mean() == pytest.approx(es, rel=0.05)


# --- S2 -------------------------------------------------------------------------------------
@pytest.mark.parametrize("c", [Q("0.95"), Q("0.99")])
def test_s2_truth_against_numerical_integration(c: Fraction) -> None:
    var, es = ts.s2_truth(c, 4)
    assert stats.t.cdf(var, 4) == pytest.approx(float(c), abs=1e-13)
    integral, _ = integrate.quad(  # ES = (1 / p) int_VaR^inf x f(x) dx
        lambda x: x * stats.t.pdf(x, 4), var, np.inf, epsabs=1e-13, epsrel=1e-13
    )
    assert es == pytest.approx(integral / float(1 - c), rel=1e-7)


def test_s2_sample_quantiles(cfg: Config) -> None:
    big = dataclasses.replace(cfg, simstudy=dataclasses.replace(cfg.simstudy, sample_size=400_000))
    cell = ts.cells(cfg)[6]
    loss = ts.sample(cell, rng(1), big)
    assert stats.kstest(loss, stats.t(4).cdf).pvalue > 0.01
    for c in (Q("0.95"), Q("0.99")):
        var, es = ts.s2_truth(c, 4)
        assert np.quantile(loss, float(c)) == pytest.approx(var, rel=0.02)
        assert loss[loss > var].mean() == pytest.approx(es, rel=0.05)
    truth = ts.cell_truth(cell, cfg, {})
    assert (truth["xi"], truth["xi_kind"]) == (0.25, "asymptotic")
    assert truth["var99"] == {"value": ts.s2_truth(Q("0.99"), 4)[0], "rel_se": None}


# --- S3 -------------------------------------------------------------------------------------
@pytest.mark.parametrize("a", [0.3, 0.6])
def test_s3_recursion_equals_the_closed_form(a: float) -> None:
    u = rng(2).uniform(size=40)
    z = -1.0 / np.log(u)
    x = ts.armax_frechet(u, a)
    assert x[0] == z[0]
    for t in range(1, 40):
        closed = max([a**t * z[0]] + [a**j * (1 - a) * z[t - j] for j in range(t)])
        assert x[t] == pytest.approx(closed, rel=1e-12)


@pytest.mark.parametrize(("a", "child"), [(Q("0.3"), 3), (Q("0.6"), 4)])
def test_s3_marginal_and_extremal_index(a: Fraction, child: int) -> None:
    n = 200_000
    u = rng(child).uniform(size=n)
    x = ts.armax_frechet(u, float(a))
    # U' = exp(-1 / X) is uniform; dependent data, so only a loose bound on the KS distance
    assert stats.kstest(np.exp(-1.0 / x), "uniform").statistic < 0.02
    loss = ts.s3_losses(u, a, Q("0.90"), Q("0.25"))
    assert stats.kstest(s1_cdf(loss, 0.90, 0.25), "uniform").statistic < 0.02
    # extremal index 1 - a (Ferro-Segers intervals estimator at the 99% quantile): +- 0.05
    mask = loss > np.quantile(loss, 0.99)
    theta = dg.extremal_index_fs(dg.inter_exceedance_times(mask)).theta
    assert theta == pytest.approx(1 - float(a), abs=0.05)


@pytest.mark.parametrize("position", [0, 10])
@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_s3_non_finite_raises(bad: float, position: int) -> None:
    u = rng(5).uniform(size=50)
    assert np.all(np.isfinite(ts.s3_losses(u, Q("0.3"), Q("0.90"), Q("0.25"))))
    u[position] = bad  # Z = 0 or 1 / 0: the maximum must not absorb it silently
    with pytest.raises(ValueError, match="S3: non-finite"):
        ts.s3_losses(u, Q("0.3"), Q("0.90"), Q("0.25"))


def test_s3_non_finite_loss_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts, "s1_ppf", lambda v, q0, xi: np.where(v > 0.5, np.inf, 0.0))
    with pytest.raises(ValueError, match="S3: non-finite loss"):
        ts.s3_losses(rng(5).uniform(size=50), Q("0.3"), Q("0.90"), Q("0.25"))


def test_s3_truth_is_the_s1_marginal(cfg: Config) -> None:
    truth = ts.cell_truth(ts.cells(cfg)[8], cfg, {})
    var, es = ts.s1_truth(Q("0.95"), Q("0.90"), Q("0.25"))
    assert (truth["var95"]["value"], truth["es95"]["value"]) == (var, es)
    assert (truth["xi"], truth["xi_kind"]) == (0.25, "exact")


# --- S4 -------------------------------------------------------------------------------------
def test_s4_generator_equals_a_plain_loop(cfg: Config) -> None:
    """The generator used by the study and by the reference, against an independent loop."""
    ss = cfg.simstudy
    n = 100_000
    z = rng(6).standard_t(6, size=n) * np.sqrt(4.0 / 6.0)
    omega, a1, b1 = 0.02, 0.08, 0.90
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = 1.0
    r[0] = z[0]
    for t in range(1, n):
        sigma2[t] = omega + a1 * r[t - 1] ** 2 + b1 * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * z[t]
    fast, last = ts.garch_returns(z, omega, a1, b1)
    np.testing.assert_allclose(fast, r, rtol=1e-10, atol=0)
    assert last == pytest.approx(omega + a1 * r[-1] ** 2 + b1 * sigma2[-1], rel=1e-10)

    head, s2 = ts.garch_returns(z[:777], omega, a1, b1)  # block continuation is exact
    tail, _ = ts.garch_returns(z[777:], omega, a1, b1, s2)
    assert np.array_equal(np.concatenate([head, tail]), fast)

    # s4_losses: burn-in, scaling, omega = 1 - a1 - b1, sign
    losses = ts.s4_losses(rng(7), 2000, ss)
    z2 = rng(7).standard_t(ss.s4_t_df, size=ss.s4_burn_in + 2000) * math.sqrt(4.0 / 6.0)
    expected, _ = ts.garch_returns(z2, float(1 - ss.s4_garch_a1 - ss.s4_garch_b1), a1, b1)
    assert (ss.s4_burn_in, ss.s4_t_df) == (1000, 6) and losses.shape == (2000,)
    assert np.array_equal(losses, -expected[1000:])


def test_s4_blocks_do_not_change_the_path(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    one = ts.s4_losses(rng(8), 5000, cfg.simstudy)
    monkeypatch.setattr(ts, "S4_CHUNK", 1024)
    np.testing.assert_allclose(ts.s4_losses(rng(8), 5000, cfg.simstudy), one, rtol=1e-12)


def test_s4_unit_variance(cfg: Config) -> None:
    losses = ts.s4_losses(rng(9), 400_000, cfg.simstudy)
    assert np.var(losses, ddof=1) == pytest.approx(1.0, abs=0.1)
    assert abs(losses.mean()) < 0.01


def test_reference_stats_hand_calculation() -> None:
    ref = ts.reference_stats(np.arange(1.0, 401.0), 2)  # batches 1..200 and 201..400
    assert (ref["n"], ref["batches"]) == (400, 2)
    # full sample: ceil(0.95 * 400) = 380, ceil(0.99 * 400) = 396
    assert ref["var95"]["value"] == 380.0 and ref["es95"]["value"] == 390.5
    assert ref["var99"]["value"] == 396.0 and ref["es99"]["value"] == 398.5
    # batch estimates (190, 390), (195.5, 395.5), (198, 398), (199.5, 399.5): SD 200 / sqrt(2)
    for key in ("var95", "es95", "var99", "es99"):
        assert ref[key]["se"] == pytest.approx(100.0, rel=1e-12)
        assert ref[key]["rel_se"] == pytest.approx(100.0 / ref[key]["value"], rel=1e-12)
    assert ref["variance"]["value"] == pytest.approx(400 * 401 / 12)
    assert ref["variance"]["se"] == pytest.approx(0.0, abs=1e-9)  # equal batch variances
    assert ref["variance_within_3_se"] is False
    with pytest.raises(ValueError, match="batches"):
        ts.reference_stats(np.arange(401.0), 2)
    with pytest.raises(ValueError, match="batches"):
        ts.reference_stats(np.arange(400.0), 1)


def test_s4_truth_is_the_reference(cfg: Config) -> None:
    ref = ts.reference_stats(ts.s4_losses(rng(10), 20_000, cfg.simstudy), 10)
    truth = ts.cell_truth(ts.cells(cfg)[9], cfg, ref)
    assert (truth["xi"], truth["xi_kind"]) == (None, None)
    for key in ts.RISK_KEYS:
        assert truth[key] == {"value": ref[key]["value"], "rel_se": ref[key]["rel_se"]}
        assert 0 < ref[key]["rel_se"] < 0.2
    assert abs(ref["variance"]["value"] - 1) < 0.5


# --- window refit ---------------------------------------------------------------------------
def test_window_refit_equals_direct_gpd_calls(cfg: Config) -> None:
    x = ts.sample(ts.cells(cfg)[4], rng(11), cfg)
    tol = float(cfg.threshold.xi_zero_tol)
    q = Q("0.90")
    out = ts.window_refit(x, q, cfg)
    w = x[-1000:]
    u = tail_threshold(w, q)
    y = gpd_excesses(w, u)
    fit = fit_mle(y, min_excesses=2, xi_zero_tol=tol, settings=cfg.gpd)
    assert (out["converged"], out["n_u"], out["xi"]) == (True, 100, fit.xi)
    for label, p in (("95", Q("0.05")), ("99", Q("0.01"))):
        var, es = tail_risk(u, fit.xi, fit.beta, 1000, 100, p, tol)
        assert (out[f"var{label}"], out[f"es{label}"]) == (var, es)
    assert set(out) == {"converged", "n_u", "xi", *ts.RISK_KEYS}

    high = ts.window_refit(x, Q("0.95"), cfg)  # p = 0.05 is not < N_u / n = 0.05
    assert high["var95"] is None and high["es95"] is None and high["var99"] is not None


def test_window_refit_failure_and_infinite_es(cfg: Config) -> None:
    x = ts.sample(ts.cells(cfg)[4], rng(11), cfg)
    tied = x.copy()
    tied[-1000::4] = 50.0  # no loss of the window exceeds its threshold
    failed = ts.window_refit(tied, Q("0.90"), cfg)
    assert failed == {"converged": False, "n_u": 0, "xi": None, **dict.fromkeys(ts.RISK_KEYS)}

    heavy = np.asarray(stats.genpareto.rvs(1.6, size=2000, random_state=rng(12)))
    out = ts.window_refit(heavy, Q("0.85"), cfg)
    assert out["converged"] and out["xi"] >= 1
    assert out["es95"] is None and out["es99"] is None and math.isfinite(out["var99"])


# --- aggregation on hand-built records ------------------------------------------------------
HAND_CELL = ts.Cell(3, "S1", "hand", Q("0.87"), Q("0.25"))
HAND_TRUTH: dict[str, Any] = {
    "xi": 0.25,
    "xi_kind": "exact",
    "var95": {"value": 2.0, "rel_se": None},
    "es95": {"value": 4.0, "rel_se": None},
    "var99": {"value": 5.0, "rel_se": 0.01},
    "es99": {"value": 10.0, "rel_se": None},
}


def _est(xi: float | None, *risk: float | None) -> dict[str, Any]:
    return {"xi": xi, **dict(zip(ts.RISK_KEYS, risk, strict=True))}


def hand_records() -> list[dict[str, Any]]:
    def rec(rep: int, q: str | None, k_hat: int, v: int, failed: int, **kw: Any) -> dict[str, Any]:
        base = {"cell": 3, "rep": rep, "q_star": q, "k_hat": k_hat, "v": v, "n_fit_failed": failed}
        return {**base, "rule_failure": None, "full": None, "refit": None, **kw}

    return [
        rec(  # false selection (0.85 < q0 = 0.87); ES infinite in the full fit
            0, "17/20", 5, 0, 0,
            full=_est(1.10, 2.2, None, 6.0, None),
            refit={"converged": True, "n_u": 150, **_est(0.55, 2.4, 4.4, 5.5, 12.0)},
        ),
        rec(  # one true null rejected (0.87 itself): V = 1 of k_hat = 8
            1, "87/100", 8, 1, 1,
            full=_est(0.20, 1.8, 3.6, 4.0, 9.0),
            refit={"converged": False, "n_u": 0, **_est(None, None, None, None, None)},
        ),
        rec(
            2, "9/10", 0, 0, 0,
            full=_est(0.60, 2.0, 4.8, 5.0, 11.0),
            refit={"converged": True, "n_u": 100, **_est(1.20, 1.6, None, 4.5, None)},
        ),
        rec(3, None, 10, 3, 16, rule_failure="no_stable_region"),
        rec(4, None, 16, 9, 2, rule_failure="all_rejected"),
    ]  # fmt: skip


def test_aggregate_m1_m2_m3(cfg: Config) -> None:
    m = ts.aggregate_cell(HAND_CELL, hand_records(), HAND_TRUTH, cfg)
    assert (m["cell"], m["label"], m["replications"], m["truth"]) == (3, "hand", 5, HAND_TRUTH)
    m1 = m["m1_q_star"]
    assert list(m1["counts"]) == [str(q) for q in cfg.threshold.grid]
    assert {k: v for k, v in m1["counts"].items() if v} == {"17/20": 1, "87/100": 1, "9/10": 1}
    assert (m1["n_defined"], m1["all_rejected"], m1["no_stable_region"]) == (3, 1, 1)
    assert (m1["rule_failures"], m1["rule_failure_rate"]) == (2, 0.4)

    m2 = m["m2_false_selection"]
    assert (m2["count"], m2["primary"], m2["n_defined"]) == (1, 0.2, 3)  # failures: not false
    assert m2["conditional"] == 1 / 3
    assert m2["primary_wilson95"] == ts.wilson(1, 5)
    assert m2["conditional_wilson95"] == ts.wilson(1, 3)

    fdp = [0.0, 1 / 8, 0.0, 3 / 10, 9 / 16]  # V / max(1, k_hat), over ALL replications
    m3 = m["m3_fdr"]
    assert m3["mean_exact"] == "79/400" and m3["mean"] == pytest.approx(0.1975, rel=1e-15)
    assert m3["sd"] == pytest.approx(statistics.stdev(fdp), rel=1e-12)
    half = 1.96 * statistics.stdev(fdp) / math.sqrt(5)
    assert m3["ci95"] == pytest.approx([0.1975 - half, 0.1975 + half], rel=1e-12)
    assert (m3["alpha"], m3["exceeds_alpha"], m3["exceeds_2alpha"]) == ("1/10", True, False)


def test_aggregate_fdr_uses_max_1_k_hat(cfg: Config) -> None:
    records = hand_records()[:2]
    records[0].update(k_hat=0, v=0)  # nothing rejected: 0 / max(1, 0) = 0, not 0 / 0
    m3 = ts.aggregate_cell(HAND_CELL, records, HAND_TRUTH, cfg)["m3_fdr"]
    assert m3["mean_exact"] == "1/16" and m3["exceeds_alpha"] is False


def test_aggregate_m4_m5_m6(cfg: Config) -> None:
    m = ts.aggregate_cell(HAND_CELL, hand_records(), HAND_TRUTH, cfg)
    xi = [1.10, 0.20, 0.60]
    m4 = m["m4_xi"]
    assert (m4["n"], m4["truth"], m4["truth_kind"]) == (3, 0.25, "exact")
    assert m4["mean"] == pytest.approx(statistics.fmean(xi))
    assert m4["sd"] == pytest.approx(statistics.stdev(xi))
    assert m4["bias"] == pytest.approx(statistics.fmean(xi) - 0.25)
    assert m4["rmse"] == pytest.approx(math.sqrt(statistics.fmean((v - 0.25) ** 2 for v in xi)))

    m5 = m["m5_risk"]
    assert (m5["var99"]["truth"], m5["var99"]["truth_rel_se"]) == (5.0, 0.01)
    rel = [0.1, -0.1, 0.0]  # full-sample VaR95: (2.2, 1.8, 2.0) against 2.0
    assert m5["var95"]["full"] == {
        "rel_bias": pytest.approx(0.0, abs=1e-15),
        "rel_rmse": pytest.approx(math.sqrt(statistics.fmean(v * v for v in rel))),
        "n_used": 3,
        "n_excluded": 0,
    }
    rel = [-0.1, 0.2]  # full-sample ES95: the infinite one is excluded
    assert m5["es95"]["full"] == {
        "rel_bias": pytest.approx(0.05),
        "rel_rmse": pytest.approx(math.sqrt(0.025)),
        "n_used": 2,
        "n_excluded": 1,
    }
    rel = [0.2, -0.2]  # window VaR95: the failed refit is excluded
    assert m5["var95"]["window"] == {
        "rel_bias": pytest.approx(0.0, abs=1e-15),
        "rel_rmse": pytest.approx(0.2),
        "n_used": 2,
        "n_excluded": 1,
    }
    assert m5["es99"]["window"] == {
        "rel_bias": pytest.approx(0.2),
        "rel_rmse": pytest.approx(0.2),
        "n_used": 1,
        "n_excluded": 2,
    }

    m6 = m["m6_failures"]
    assert (m6["pairs"], m6["fit_failed_pairs"]) == (80, 19)
    assert m6["fit_failed_share"] == 19 / 80
    assert m6["full_nonconverged_at_q_star"] == 0
    assert (m6["window_nonconverged"], m6["window_nonconvergence_rate"]) == (1, 1 / 3)
    assert (m6["full_p_xi_ge_0_5"], m6["full_p_xi_ge_1"]) == (2 / 3, 1 / 3)
    assert (m6["window_p_xi_ge_0_5"], m6["window_p_xi_ge_1"]) == (2 / 3, 1 / 3)


def test_aggregate_without_boundary_or_defined_q_star(cfg: Config) -> None:
    records = [dict(r, v=None) for r in hand_records()[3:]]
    cell = ts.Cell(9, "S4", "no truth")
    truth = dict(HAND_TRUTH, xi=None, xi_kind=None)
    m = ts.aggregate_cell(cell, records, truth, cfg)
    assert m["m2_false_selection"] is None and m["m3_fdr"] is None
    assert m["m1_q_star"]["rule_failure_rate"] == 1.0
    assert m["m4_xi"] == {
        "n": 0, "mean": None, "sd": None, "bias": None, "rmse": None,
        "truth": None, "truth_kind": None,
    }  # fmt: skip
    assert m["m5_risk"]["es99"]["window"] == {
        "rel_bias": None, "rel_rmse": None, "n_used": 0, "n_excluded": 0,
    }  # fmt: skip
    assert m["m6_failures"]["window_nonconvergence_rate"] is None
    assert m["m6_failures"]["full_p_xi_ge_1"] is None

    one = ts.aggregate_cell(cell, [dict(hand_records()[1], v=None)], truth, cfg)["m4_xi"]
    assert (one["n"], one["mean"], one["sd"], one["bias"]) == (1, 0.20, None, None)
    with_boundary = ts.aggregate_cell(HAND_CELL, hand_records()[3:], HAND_TRUTH, cfg)
    assert with_boundary["m2_false_selection"]["conditional"] is None
    assert with_boundary["m2_false_selection"]["conditional_wilson95"] is None


def test_wilson_interval() -> None:
    for k, n in [(10, 100), (1, 5), (40, 200), (0, 50), (50, 50)]:
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert ts.wilson(k, n) == pytest.approx([lo, hi], abs=1e-12)
    assert ts.wilson(10, 100) == pytest.approx([0.05522914, 0.17436566], abs=1e-7)
    assert ts.wilson(0, 50)[0] == pytest.approx(0.0, abs=1e-15)


def _trigger_metrics(
    count: int, n_defined: int, fdr: str, upper: float, failures: int
) -> dict[str, Any]:
    return {
        "replications": 200,
        "m1_q_star": {"rule_failures": failures, "rule_failure_rate": failures / 200},
        "m2_false_selection": {
            "count": count,
            "primary": count / 200,
            "primary_wilson95": ts.wilson(count, 200),
            "n_defined": n_defined,
            "conditional": count / n_defined if n_defined else None,
            "conditional_wilson95": ts.wilson(count, n_defined) if n_defined else None,
        },
        "m3_fdr": {"mean": float(Q(fdr)), "mean_exact": fdr, "ci95": [0.0, upper]},
    }


def test_triggers_are_strict_and_flag_near_misses(cfg: Config) -> None:
    study_cells = ts.cells(cfg)
    s3 = study_cells[7:9]

    def run(*metrics: dict[str, Any]) -> list[dict[str, Any]]:
        return ts.triggers(s3, metrics, cfg)

    # exactly on every line (0.20, 0.20, 0.10): nothing fires; clearly below: no near miss
    on_line, below = run(
        _trigger_metrics(40, 200, "1/5", 0.19, 20), _trigger_metrics(10, 200, "1/20", 0.1, 0)
    )
    for t in (on_line, below):
        assert [t[k]["fired"] for k in ("false_selection", "fdr", "rule_failure")] == [False] * 3
        assert t["any_fired"] is False
        assert (t["false_selection"]["near_miss"], t["fdr"]["near_miss"]) == (False, False)
    assert (on_line["cell"], below["cell"]) == (7, 8)
    assert (on_line["label"], below["label"]) == ("S3 a=0.3", "S3 a=0.6")
    assert on_line["false_selection"]["limit"] == "1/5" and on_line["fdr"]["limit"] == "1/5"
    assert on_line["rule_failure"] == {"value": 0.1, "limit": "1/10", "fired": False}
    assert on_line["false_selection"]["value"] == 0.2 and on_line["fdr"]["value"] == 0.2
    assert on_line["false_selection"]["wilson95"] == ts.wilson(40, 200)

    # just above each line: fired, and a fired trigger is never a near miss
    fs, fdr = run(
        _trigger_metrics(41, 150, "1/5", 0.3, 20), _trigger_metrics(40, 200, "201/1000", 0.3, 20)
    )
    assert fs["false_selection"] == {**fs["false_selection"], "fired": True, "near_miss": False}
    assert (fs["fdr"]["fired"], fs["fdr"]["near_miss"], fs["any_fired"]) == (False, True, True)
    assert (fdr["fdr"]["fired"], fdr["fdr"]["near_miss"], fdr["any_fired"]) == (True, False, True)
    assert fdr["false_selection"]["fired"] is False
    rf, cond = run(_trigger_metrics(0, 179, "0", 0.0, 21), _trigger_metrics(40, 199, "1/5", 0.2, 1))
    assert rf["rule_failure"]["fired"] is True and rf["any_fired"] is True
    # near miss: primary 40/200 does not fire, the conditional 40/199 is above 0.20
    assert (cond["false_selection"]["fired"], cond["false_selection"]["near_miss"]) == (False, True)
    assert cond["false_selection"]["conditional"] == 40 / 199 and cond["any_fired"] is False
    none_defined = run(_trigger_metrics(0, 0, "0", 0.0, 200), _trigger_metrics(0, 0, "0", 0.0, 200))
    assert none_defined[0]["false_selection"]["near_miss"] is False

    # only S3 cells are evaluated
    filler = [_trigger_metrics(200, 200, "1", 1.0, 200)] * 10
    assert [t["cell"] for t in ts.triggers(study_cells, filler, cfg)] == [7, 8]


# --- the two select_threshold switches (D35 (7)) --------------------------------------------
@pytest.mark.parametrize("xi", [0.25, -0.7])
def test_switches_leave_rule_relevant_outputs_unchanged(tiny: Config, xi: float) -> None:
    x = np.asarray(stats.genpareto.rvs(xi, size=200, random_state=rng(13)))
    on = select_threshold(x, tiny, ss=SS(3), gof_reps=9, convergence_share=False)
    off = select_threshold(
        x, tiny, ss=SS(3), gof_reps=9, convergence_share=False, halves=False, intervals=False
    )
    for key in ("q_star", "q_fs", "k_hat", "rule_failure"):
        assert getattr(on, key) == getattr(off, key)
    assert len(on.candidates) == len(off.candidates) == 6
    for a, b in zip(on.candidates, off.candidates, strict=True):
        assert {k: a[k] for k in RULE_KEYS} == {k: b[k] for k in RULE_KEYS}
        assert set(a) == set(b)
        assert all(b[k] is None for k in CI_KEYS)
        assert "ci_unavailable" not in b["flags"]  # the skip is deliberate
        assert [f for f in a["flags"] if f != "ci_unavailable"] == b["flags"]
    assert any(a["xi_ci"] is not None for a in on.candidates)
    if xi < -0.5:
        assert any("ci_unavailable" in a["flags"] for a in on.candidates)
    assert on.halves is not None and off.halves is None and off.flags == ()
    parsed = json.loads(off.to_json())
    assert parsed["halves"] is None and len(parsed["candidates"]) == 6

    mixed = select_threshold(
        x, tiny, ss=SS(3), gof_reps=9, convergence_share=False, intervals=False
    )
    assert mixed.halves is not None and mixed.flags == on.flags
    for got, want in zip(mixed.halves, on.halves, strict=True):
        assert (got.q_star, got.k_hat, got.rule_failure) == (
            want.q_star,
            want.k_hat,
            want.rule_failure,
        )
        assert all(r[k] is None for r in got.candidates for k in CI_KEYS)


# --- runner ---------------------------------------------------------------------------------
def test_study_structure(study: tuple[dict[str, Any], list[dict[str, Any]]], tiny: Config) -> None:
    summary, records = study
    assert set(summary) == {
        "label", "decision", "config_sha256", "design", "seeds", "cells", "s4_reference",
        "metrics", "triggers", "records_sha256",
    }  # fmt: skip
    assert summary["label"] == ts.LABEL == "simulation evidence (DGPs S1-S4)"
    assert (summary["decision"], summary["config_sha256"]) == ("D35", "0" * 64)
    assert summary["design"] == {
        "replications": 2,
        "gof_bootstrap_reps": 9,
        "sample_size": 200,
        "window_refit": 100,
        "grid": ["17/20", "43/50", "87/100", "22/25", "89/100", "9/10"],
        "levels": ["19/20", "99/100"],
        "alpha_fs": "1/10",
        "q_max": "9/10",
    }
    assert summary["seeds"]["seed"] == SEED and summary["seeds"]["children"] == 5
    assert [c["dgp"] for c in summary["cells"]] == ["S1", "S2", "S3", "S4"]
    assert [(r["cell"], r["rep"]) for r in records] == [(c, r) for c in range(4) for r in range(2)]
    assert [m["cell"] for m in summary["metrics"]] == [0, 1, 2, 3]
    assert all(m["replications"] == 2 for m in summary["metrics"])
    assert [t["cell"] for t in summary["triggers"]] == [2]
    assert summary["records_sha256"] == ts.records_sha256(records)
    assert summary["s4_reference"]["n"] == 20_000
    text = ts.summary_json(summary)
    assert text.endswith("}\n") and "\r" not in text and json.loads(text) == summary
    for word in ("seconds", "workers", "cpu", "platform", "time"):
        assert f'"{word}' not in text  # no runtime or hardware in the artifact

    for record, cell in zip(records, [c for c in ts.cells(tiny) for _ in range(2)], strict=True):
        assert record == ts.run_replication(tiny, cell, record["rep"])  # a pure function
        assert set(record) == {
            "cell", "rep", "k_hat", "v", "q_star", "rule_failure", "n_fit_failed", "full", "refit",
        }  # fmt: skip
        assert (record["v"] is None) == (cell.q0 is None)
        if record["q_star"] is None:
            assert record["rule_failure"] in ("all_rejected", "no_stable_region")
            assert record["full"] is None and record["refit"] is None
        else:
            assert Q(record["q_star"]) in tiny.threshold.grid and record["rule_failure"] is None
            assert set(record["full"]) == {"xi", *ts.RISK_KEYS}
            assert record["refit"] == ts.window_refit(
                ts.sample(
                    cell,
                    np.random.Generator(
                        np.random.PCG64(SS(SEED, spawn_key=(cell.index, record["rep"], 0)))
                    ),
                    tiny,
                ),
                Q(record["q_star"]),
                tiny,
            )


def test_records_sha256_is_canonical() -> None:
    records = hand_records()
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    assert ts.records_sha256(records) == hashlib.sha256(lines.encode()).hexdigest()
    assert ts.records_sha256(records[::-1]) != ts.records_sha256(records)


def test_worker_count_does_not_change_the_result(
    study: tuple[dict[str, Any], list[dict[str, Any]]], tiny: Config
) -> None:
    summary, records = ts.run_study(tiny, config_sha256="0" * 64, workers=2)
    assert records == study[1]
    assert ts.summary_json(summary).encode() == ts.summary_json(study[0]).encode()
    with pytest.raises(ValueError, match="workers"):
        ts.run_records(tiny, 1, 0)


def test_check_design(cfg: Config, tiny: Config) -> None:
    ts.check_design(cfg)
    ts.check_design(tiny)
    sim = cfg.simstudy
    for bad, message in [
        (
            dataclasses.replace(cfg, simstudy=dataclasses.replace(sim, sample_size=1000)),
            "train_size",
        ),
        (
            dataclasses.replace(cfg, simstudy=dataclasses.replace(sim, replications=1)),
            "replications",
        ),
        (
            dataclasses.replace(cfg, simstudy=dataclasses.replace(sim, s4_reference_size=1005)),
            "batches",
        ),
        (
            dataclasses.replace(cfg, threshold=dataclasses.replace(cfg.threshold, n_min=None)),
            "n_min",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            ts.check_design(bad)
        with pytest.raises(ValueError, match=message):
            ts.run_study(bad, config_sha256="x", workers=1)


def test_runtime_guard_projection() -> None:
    assert ts.project_seconds(30.0, 2, 200) == 3000.0
    assert ts.project_seconds(0.0, 2, 200) == 0.0
    assert ts.PILOT_REPLICATIONS == 2
    for args in [(-1.0, 2, 200), (math.nan, 2, 200), (1.0, 0, 200), (1.0, 2, 0)]:
        with pytest.raises(ValueError, match="projection"):
            ts.project_seconds(*args)


@pytest.mark.parametrize(
    ("cpus", "expected"), [(32, 31), (64, 31), (17, 16), (4, 3), (1, 1), (None, 1)]
)
def test_default_workers(cpus: int | None, expected: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts.os, "cpu_count", lambda: cpus)
    assert ts.default_workers() == expected


def test_study_module_does_not_touch_the_data_layer() -> None:
    source = Path(ts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {m for m in modules if m.startswith("niftytail")} == {
        "niftytail.config",
        "niftytail.gpd",
        "niftytail.losses",
        "niftytail.threshold",
    }
    assert not any(part in ("data", "pandas") for m in modules for part in m.split("."))
    assert "load_raw" not in source and "arch_model" not in source  # no data, no GARCH fitting


# --- command and outputs --------------------------------------------------------------------
def _sha(paths: list[Path]) -> list[str]:
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def test_cli_writes_identical_outputs_twice(
    tiny_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tiny_path.parents[1])
    assert main(["threshold-study", "configs/base.toml", "--workers", "1"]) == 0
    log = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    assert [r["message"] for r in log] == ["pilot done", "OK"]
    names = [ts.METRICS_NAME, *ts.FIGURE_NAMES]
    assert log[1]["outputs"] == [f"reports/threshold_selection/simstudy/{n}" for n in names]
    assert set(log[1]) == {
        "time",
        "level",
        "logger",
        "message",
        "outputs",
        "seconds",
        "workers",
        "pilot_seconds",
        "projected_seconds",
    }  # fmt: skip  (file names and wall-clock only: no result is logged)
    assert log[1]["workers"] == 1 and log[1]["seconds"] >= log[1]["pilot_seconds"] > 0
    assert log[1]["projected_seconds"] == pytest.approx(log[1]["pilot_seconds"])  # R = pilot size

    paths = [ts.STUDY_DIR / n for n in names]
    first = _sha(paths)
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    assert summary["config_sha256"] == hashlib.sha256(tiny_path.read_bytes()).hexdigest()
    assert paths[0].read_bytes() == ts.summary_json(summary).encode("utf-8")
    for p in paths[1:]:
        svg = p.read_bytes()
        assert svg.startswith(b"<?xml") and b"\r" not in svg and b"<dc:date>" not in svg
    assert main(["threshold-study", "configs/base.toml", "--workers", "1"]) == 0
    assert _sha(paths) == first  # metrics and all five figures: byte-identical


def test_cli_errors_write_nothing(
    tiny_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "configs").mkdir()
    target = tmp_path / "configs" / "base.toml"
    monkeypatch.chdir(tmp_path)
    assert main(["threshold-study", "configs/base.toml"]) == 1  # no such file

    target.write_text(
        _set(tiny_path.read_text(encoding="utf-8"), "simstudy", "sample_size", "100"),
        encoding="utf-8",
    )
    assert main(["threshold-study", "configs/base.toml", "--workers", "1"]) == 1
    assert "train_size" in json.loads(capsys.readouterr().err.strip().splitlines()[-1])["error"]

    # runtime guard: a projection above max_runtime_hours stops before the study
    target.write_text(tiny_path.read_text(encoding="utf-8"), encoding="utf-8")
    calls: list[int] = []
    monkeypatch.setattr(ts, "run_records", lambda cfg, reps, workers: calls.append(reps) or [])
    monkeypatch.setattr(ts, "project_seconds", lambda *a: 2 * 3600 + 1.0)
    assert main(["threshold-study", "configs/base.toml"]) == 1
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["message"].startswith("runtime guard") and record["projected_seconds"] == 7201.0
    assert record["workers"] == ts.default_workers() and calls == [2]  # only the pilot ran
    assert not (tmp_path / "reports").exists()


def test_figures_handle_cells_without_estimates(
    study: tuple[dict[str, Any], list[dict[str, Any]]], tmp_path: Path
) -> None:
    summary, records = study
    empty = [
        dict(r, q_star=None, rule_failure="no_stable_region", full=None, refit=None)
        for r in records
    ]
    paths = ts.write_figures(summary, empty, tmp_path / "figs")
    assert [p.name for p in paths] == list(ts.FIGURE_NAMES) and all(
        p.stat().st_size > 0 for p in paths
    )
