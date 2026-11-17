"""Tests for niftytail.gpd (D30) against independent references. Synthetic data only."""

from __future__ import annotations

import dataclasses
import math
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from niftytail import gpd as gpd_module
from niftytail.config import GpdConfig, load_config
from niftytail.gpd import (
    EstimationError,
    excesses,
    fit_mle,
    fit_pwm,
    nll,
    tail_es,
    tail_risk,
    tail_var,
)
from niftytail.losses import exceedance_mask, tail_threshold

SEED = 20261111
TOL = 1e-8  # [threshold].xi_zero_tol, checked against the config below
U, BETA, N, N_U = 2.0, 0.5, 1000, 100  # methodology 6 worked example (R4)


def rng() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED)))


@pytest.fixture
def settings(base_config_path: Path) -> GpdConfig:
    cfg = load_config(base_config_path)
    assert float(cfg.threshold.xi_zero_tol) == TOL
    return cfg.gpd


def gpd_sample(g: np.random.Generator, xi: float, n: int, beta: float = 1.0) -> np.ndarray:
    return np.asarray(stats.genpareto.rvs(xi, scale=beta, size=n, random_state=g))


def mle(y: np.ndarray, s: GpdConfig) -> object:
    return fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=s)


# --- nll ---------------------------------------------------------------------------
def test_nll_hand_cases() -> None:
    assert nll([1, 2, 3], 0.0, 2.0, TOL) == pytest.approx(3 * math.log(2) + 3, abs=1e-12)
    assert nll([1, 2, 3], 0.0, 2.0, TOL) == pytest.approx(5.0794415416798, abs=1e-12)
    assert nll([1, 2], 0.5, 1.0, TOL) == pytest.approx(3 * math.log(3), abs=1e-12)


@pytest.mark.parametrize("xi", [-0.3, 0.1, 0.25, 0.4, 0.9])
def test_nll_matches_scipy_logpdf(xi: float) -> None:
    beta = 1.3
    y = gpd_sample(rng(), xi, 500, beta)
    ref = -float(np.sum(stats.genpareto.logpdf(y, c=xi, loc=0, scale=beta)))
    assert nll(y, xi, beta, TOL) == pytest.approx(ref, rel=1e-12)


def test_nll_continuity_across_xi_zero_tol() -> None:
    # The nll has a genuine non-zero slope at xi = 0, so the branch switch is tested
    # against the first-order expansion f(0) + xi f'(0), with f'(0) = sum(z - z^2 / 2).
    beta = 1.0
    y = rng().exponential(1.0, 200)
    z = y / beta
    at0, slope = nll(y, 0.0, beta, TOL), float(np.sum(z - z**2 / 2))
    for xi in (1.1e-8, -1.1e-8):  # log1p branch
        assert abs(nll(y, xi, beta, TOL) - at0 - xi * slope) <= 1e-9, xi
    # The zero branch (|xi| < xi_zero_tol) intentionally drops the first-order term; the
    # bound documents the cost of the pinned xi_zero_tol = 1e-8.
    for xi in (0.9e-8, -0.9e-8):
        f = nll(y, xi, beta, TOL)
        assert f == at0, xi
        assert abs(f - at0 - xi * slope) <= abs(xi) * abs(slope) * (1 + 1e-6) + 1e-12, xi


def test_nll_infinite_outside_domain() -> None:
    y = [0.5, 1.0, 2.0]
    for beta in (0.0, -1.0, math.inf, math.nan):
        assert nll(y, 0.2, beta, TOL) == math.inf
    for xi in (-1.0, -1.5, math.nan):
        assert nll(y, xi, 10.0, TOL) == math.inf
    assert nll(y, -0.5, 1.0, TOL) == math.inf  # 1 + xi y / beta = 0 at y = 2
    assert nll(y, -0.6, 1.0, TOL) == math.inf  # < 0
    assert math.isfinite(nll(y, -0.4, 1.0, TOL))
    assert nll([1e9], -5e-9, 1.0, TOL) == math.inf  # D31 (1): support check in the zero branch


def test_nll_rejects_invalid_y() -> None:
    for bad in ([], [1.0, 0.0], [1.0, -2.0], [1.0, math.nan], [[1.0]]):
        with pytest.raises(ValueError):
            nll(bad, 0.1, 1.0, TOL)


# --- PWM ---------------------------------------------------------------------------
def test_fit_pwm_hand_case() -> None:
    xi, beta = fit_pwm([4, 1, 3, 2], min_excesses=2)  # a0 = 2.5, a1 = 10/12
    assert xi == pytest.approx(-1.0, abs=1e-12)
    assert beta == pytest.approx(5.0, abs=1e-12)


def test_fit_pwm_large_sample() -> None:
    xi, beta = fit_pwm(gpd_sample(rng(), 0.25, 200_000), min_excesses=2)
    assert abs(xi - 0.25) < 0.02
    assert abs(beta - 1.0) < 0.02


def test_fit_pwm_errors() -> None:
    with pytest.raises(EstimationError, match="at least 5"):
        fit_pwm([1.0, 2.0, 3.0], min_excesses=5)
    with pytest.raises(EstimationError, match="a0 - 2 a1"):
        fit_pwm([1.5, 1.5, 1.5, 1.5], min_excesses=2)  # a0 = 2 a1 for constant y
    for bad in ([1.0, 0.0, 2.0], [1.0, -1.0, 2.0]):
        with pytest.raises(ValueError, match="> 0"):
            fit_pwm(bad, min_excesses=2)
    for m in (1, 0, True):
        with pytest.raises(ValueError, match="min_excesses"):
            fit_pwm([1.0, 2.0, 3.0], min_excesses=m)
    with pytest.raises(TypeError):
        fit_pwm([1.0, 2.0, 3.0])  # type: ignore[call-arg]  # min_excesses is required


# --- MLE ---------------------------------------------------------------------------
@pytest.mark.parametrize("xi", [0.1, 0.25, 0.4])
def test_fit_mle_matches_scipy_fit(xi: float, settings: GpdConfig) -> None:
    g = rng()
    for _ in range(3):
        y = gpd_sample(g, xi, 500)
        fit = fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=settings)
        c, _, scale = stats.genpareto.fit(y, floc=0)
        assert abs(fit.xi - c) <= 1e-3
        assert abs(fit.beta - scale) / scale <= 1e-3
        assert fit.nll <= nll(y, float(c), float(scale), TOL) + 1e-9
        assert (fit.method, fit.start, fit.n_excesses, fit.regular) == ("mle", "pwm", 500, True)


def test_fit_mle_is_a_local_minimum(settings: GpdConfig) -> None:
    g = rng()
    y = gpd_sample(g, 0.25, 500)
    fit = fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=settings)
    theta = np.array([fit.xi, math.log(fit.beta)])

    def f(t: np.ndarray) -> float:
        return nll(y, float(t[0]), math.exp(float(t[1])), TOL)

    h = 1e-6
    grad = [(f(theta + h * e) - f(theta - h * e)) / (2 * h) for e in np.eye(2)]
    assert float(np.hypot(*grad)) < 1e-4
    for d in g.normal(0.0, 0.05, size=(200, 2)):
        assert f(theta + d) >= fit.nll - 1e-10


def test_fit_mle_exponential_data(settings: GpdConfig) -> None:
    fit = fit_mle(rng().exponential(1.0, 2000), min_excesses=2, xi_zero_tol=TOL, settings=settings)
    assert abs(fit.xi) < 0.1


def test_fit_mle_exp_fallback_start(settings: GpdConfig) -> None:
    # Degenerate case: constant y is the only input where fit_pwm raises, so the fit runs
    # from the exponential fallback start. The MLE drifts to xi -> -1, which under the
    # D31 (2) edge-case policy now raises (Day 5 returned it with regular = False).
    y = [1.5, 1.5, 1.5, 1.5]  # fit_pwm raises (a0 - 2 a1 = 0)
    with pytest.raises(EstimationError):
        fit_pwm(y, min_excesses=2)
    with pytest.raises(EstimationError, match="xi = -1 boundary"):
        fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=settings)


def test_fit_mle_exp_fallback_reaches_pwm_optimum(
    settings: GpdConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    y = gpd_sample(rng(), 0.25, 500)
    ref = fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=settings)

    def failing_pwm(*_: object, **__: object) -> tuple[float, float]:
        raise EstimationError("PWM forced to fail")

    monkeypatch.setattr(gpd_module, "fit_pwm", failing_pwm)
    fit = fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=settings)
    assert (ref.start, fit.start) == ("pwm", "exp_fallback")
    assert abs(fit.xi - ref.xi) <= 1e-6
    assert abs(fit.beta - ref.beta) / ref.beta <= 1e-6
    assert abs(fit.nll - ref.nll) <= 1e-9


def test_fit_mle_non_convergence_raises(settings: GpdConfig) -> None:
    y = gpd_sample(rng(), 0.25, 500)
    with pytest.raises(EstimationError, match="did not converge"):
        fit_mle(
            y, min_excesses=2, xi_zero_tol=TOL, settings=dataclasses.replace(settings, maxiter=1)
        )


def test_fit_mle_deterministic(settings: GpdConfig) -> None:
    y = gpd_sample(rng(), 0.25, 500)
    a = fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=settings)
    b = fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=settings)
    assert a == b
    assert (a.xi.hex(), a.beta.hex(), a.nll.hex()) == (b.xi.hex(), b.beta.hex(), b.nll.hex())


def test_fit_mle_input_errors(settings: GpdConfig) -> None:
    with pytest.raises(EstimationError, match="at least 10"):
        fit_mle([1.0, 2.0, 3.0], min_excesses=10, xi_zero_tol=TOL, settings=settings)
    with pytest.raises(ValueError, match="optimizer"):
        bad = dataclasses.replace(settings, optimizer="BFGS")
        fit_mle([1.0, 2.0, 3.0], min_excesses=2, xi_zero_tol=TOL, settings=bad)


# --- VaR / ES ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("p", "var_ref", "es_ref"),
    [(Fraction("0.01"), 3.556559, 4.742078), (Fraction("0.05"), 2.378414, 3.171219)],
)
def test_tail_worked_example(p: Fraction, var_ref: float, es_ref: float) -> None:
    var, es = tail_risk(U, 0.25, BETA, N, N_U, p, TOL)
    assert var == pytest.approx(var_ref, rel=1e-6)
    assert es == pytest.approx(es_ref, rel=1e-6)
    assert es == var / 0.75  # beta - xi u = 0 exactly here


def test_tail_var_float_trap() -> None:
    assert float(Fraction(N, N_U) * Fraction("0.01")) == 0.1
    assert N / N_U * (1 - 0.99) != 0.1  # the float route from c = 0.99 is not exact
    assert tail_var(U, 0.0, BETA, N, N_U, Fraction("0.01"), TOL) == U - BETA * math.log(0.1)


def test_tail_var_guard() -> None:
    for p in (Fraction(N_U, N), Fraction("0.1"), Fraction("0.2")):
        with pytest.raises(ValueError, match="n_u / n"):
            tail_var(U, 0.25, BETA, N, N_U, p, TOL)
    eps = Fraction(1, 10**9)
    var = tail_var(U, 0.25, BETA, N, N_U, Fraction(N_U, N) - eps, TOL)
    assert U < var < U + 1e-7  # VaR -> u as p -> n_u / n


def _survival(x: float, xi: float, n: int, n_u: int) -> float:
    return n_u / n * float(stats.genpareto.sf(x - U, c=xi, scale=BETA))


CASES = [
    (xi, n, n_u, p)
    for xi in (-0.3, 0.0, 0.25, 0.6)
    for n, n_u in ((1000, 100), (2000, 400), (1000, 37))
    for p in (Fraction(1, 1000), Fraction(1, 200), Fraction(1, 100), Fraction(1, 40))
    if p < Fraction(n_u, n)
]


@pytest.mark.parametrize(("xi", "n", "n_u", "p"), CASES)
def test_tail_var_es_independent(xi: float, n: int, n_u: int, p: Fraction) -> None:
    var, es = tail_risk(U, xi, BETA, n, n_u, p, TOL)
    pf = float(p)
    end = U - BETA / xi if xi < 0 else math.inf
    hi = end if xi < 0 else U + 1.0
    while xi >= 0 and _survival(hi, xi, n, n_u) > pf:
        hi = U + 2 * (hi - U)
    root = optimize.bisect(lambda x: _survival(x, xi, n, n_u) - pf, U, hi, xtol=1e-13)
    assert abs(var - root) <= 1e-10
    isf = U + float(stats.genpareto.isf(pf * n / n_u, c=xi, scale=BETA))
    assert abs(var - isf) <= 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an IntegrationWarning fails the test
        integral, _ = integrate.quad(_survival, var, end, args=(xi, n, n_u), epsabs=0, epsrel=1e-12)
    assert es == pytest.approx(var + integral / pf, rel=1e-8)
    assert es > var


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.25, 0.6, 0.95])
def test_var_increases_with_confidence(xi: float) -> None:
    ps = [Fraction(1, 20), Fraction(1, 40), Fraction(1, 100), Fraction(1, 200), Fraction(1, 1000)]
    risks = [tail_risk(U, xi, BETA, N, N_U, p, TOL) for p in ps]
    vars_ = [v for v, _ in risks]
    assert vars_ == sorted(vars_) and len(set(vars_)) == len(vars_)
    assert all(es > v for v, es in risks)


@pytest.mark.parametrize("xi", [1.0, 1.5])
def test_tail_es_infinite_for_xi_ge_1(xi: float) -> None:
    var, es = tail_risk(U, xi, BETA, N, N_U, Fraction("0.01"), TOL)
    assert math.isfinite(var) and es == math.inf


@pytest.mark.parametrize("p", [Fraction("0.01"), Fraction("0.05")])
def test_tail_continuity_across_xi_zero_tol(p: Fraction) -> None:
    # VaR and ES have a genuine non-zero slope at xi = 0, so the branch switch is tested
    # against the first-order expansion: VaR'(0) = beta (ln r)^2 / 2 and
    # ES'(0) = VaR'(0) + VaR(0) + beta - u.
    var0, es0 = tail_risk(U, 0.0, BETA, N, N_U, p, TOL)
    log_r = math.log(float(Fraction(N, N_U) * p))
    dvar = BETA * log_r**2 / 2
    des = dvar + var0 + BETA - U
    for xi in (1.1e-8, -1.1e-8):  # expm1 branch
        var, es = tail_risk(U, xi, BETA, N, N_U, p, TOL)
        assert abs(var - var0 - xi * dvar) <= 1e-8, ("VaR", xi)
        assert abs(es - es0 - xi * des) <= 1e-8, ("ES", xi)
    # The zero branch (|xi| < xi_zero_tol) intentionally drops the first-order term; the
    # bound documents the cost of the pinned xi_zero_tol = 1e-8.
    for xi in (0.9e-8, -0.9e-8):
        var, es = tail_risk(U, xi, BETA, N, N_U, p, TOL)
        assert var == var0, ("VaR", xi)
        assert es == tail_es(var0, U, xi, BETA), ("ES", xi)
        for name, d, slope in (("VaR", var - var0, dvar), ("ES", es - es0, des)):
            bound = abs(xi) * abs(slope) * (1 + 1e-6) + 1e-12
            assert abs(d - xi * slope) <= bound, (name, xi)


def test_tail_input_validation() -> None:
    p = Fraction("0.01")
    for args in (
        (U, 0.25, BETA, 100, 1000, p),  # n < n_u
        (U, 0.25, BETA, N, 0, p),
        (U, 0.25, BETA, 1000.0, N_U, p),
        (U, 0.25, BETA, N, N_U, 0.01),
        (U, 0.25, BETA, N, N_U, Fraction(0)),
        (U, 0.25, 0.0, N, N_U, p),
        (math.nan, 0.25, BETA, N, N_U, p),
        (U, math.inf, BETA, N, N_U, p),
    ):
        with pytest.raises(ValueError):
            tail_var(*args, TOL)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        tail_es(math.nan, U, 0.25, BETA)
    with pytest.raises(ValueError, match="beta"):
        tail_es(3.0, U, 0.25, -1.0)


# --- excesses ----------------------------------------------------------------------
def test_excesses_strict_and_consistent_with_day4() -> None:
    g = rng()
    losses = g.permutation(np.concatenate([g.standard_t(4, 1990), np.full(10, 1.25)]))
    u = tail_threshold(losses, Fraction("0.9"))
    y = excesses(losses, u)
    mask = exceedance_mask(losses, u)
    assert y.size == int(mask.sum())
    np.testing.assert_array_equal(y, losses[mask] - u)
    assert np.all(y > 0)
    assert excesses([1.0, 2.0, 2.0, 3.0], 2.0).tolist() == [1.0]  # ties at u excluded
    with pytest.raises(ValueError):
        excesses([1.0, math.nan], 0.5)
    with pytest.raises(ValueError):
        excesses([1.0, 2.0], math.inf)


def test_error_messages_contain_no_data_values(settings: GpdConfig) -> None:
    values = [2.718281, 2.718281, 2.718281]
    tokens = ("2.718", "2.71", "718281")
    raised: list[str] = []
    for call in (
        lambda: fit_pwm(values, min_excesses=2),
        lambda: fit_pwm(values, min_excesses=7),
        lambda: fit_mle(values, min_excesses=7, xi_zero_tol=TOL, settings=settings),
        lambda: nll([*values, -3.14159], 0.1, 1.0, TOL),
        lambda: excesses([*values, math.nan], 1.0),
        lambda: tail_var(2.718281, 0.25, BETA, N, N_U, Fraction(1, 10), TOL),
        lambda: tail_es(math.nan, 2.718281, 0.25, BETA),
    ):
        with pytest.raises((EstimationError, ValueError)) as exc:
            call()
        raised.append(str(exc.value))
    for msg in raised:
        assert not any(t in msg for t in (*tokens, "3.14")), msg
