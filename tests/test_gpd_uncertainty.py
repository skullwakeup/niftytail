"""Tests for the Day-6 GPD uncertainty and edge-case policy (D31). Synthetic data only."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, stats

from niftytail.config import GpdConfig, load_config
from niftytail.gpd import (
    EstimationError,
    GpdFit,
    delta_ci,
    expected_covariance,
    fit_mle,
    nll,
    profile_ci_xi,
    risk_gradient,
    se_xi,
    tail_risk,
)

SEED = 20261111
TOL = 1e-8
CHI2_95 = 3.841458820694124
U, BETA, N, N_U, P = 2.0, 0.5, 1000, 100, Fraction("0.01")


def rng(child: int | None = None) -> np.random.Generator:
    ss = np.random.SeedSequence(SEED)
    return np.random.Generator(np.random.PCG64(ss if child is None else ss.spawn(child + 1)[child]))


@pytest.fixture
def settings(base_config_path: Path) -> GpdConfig:
    cfg = load_config(base_config_path)
    assert float(cfg.threshold.xi_zero_tol) == TOL
    return cfg.gpd


def sample(g: np.random.Generator, xi: float, n: int, beta: float = 1.0) -> np.ndarray:
    return np.asarray(stats.genpareto.rvs(xi, scale=beta, size=n, random_state=g))


def fit(y: np.ndarray, s: GpdConfig) -> GpdFit:
    return fit_mle(y, min_excesses=2, xi_zero_tol=TOL, settings=s)


def fake_fit(xi: float, beta: float = BETA, n_excesses: int = 100) -> GpdFit:
    return GpdFit(xi, beta, 0.0, n_excesses, "mle", "pwm", 1, xi > -0.5)


# --- (2) edge-case policy ------------------------------------------------------------
def test_uniform_n30_raises_at_boundary(settings: GpdConfig) -> None:
    """Observed outcome for seeded U(0,1), n = 30: the boundary rule fires (reported)."""
    with pytest.raises(EstimationError, match="xi = -1 boundary"):
        fit(rng().uniform(0.0, 1.0, 30), settings)


def test_irregular_fit_is_returned(settings: GpdConfig) -> None:
    f = fit(sample(rng(), -0.6, 30), settings)
    assert -1 + 1e-6 < f.xi <= -0.5 and f.regular is False


def test_heavy_tail_fit_returned_with_infinite_es(settings: GpdConfig) -> None:
    f = fit(sample(rng(), 1.2, 2000), settings)
    assert f.xi >= 1 and f.regular
    var, es = tail_risk(U, f.xi, f.beta, N, N_U, P, TOL)
    assert math.isfinite(var) and es == math.inf


# --- (3) SE --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("n_u", "se"), [(400, 0.060), (200, 0.085), (144, 0.100), (100, 0.120), (50, 0.170)]
)
def test_se_xi_methodology_table(n_u: int, se: float) -> None:
    assert round(se_xi(0.2, n_u), 3) == se


def test_se_xi_validation() -> None:
    for xi, n_u in ((-0.5, 100), (-0.7, 100), (math.nan, 100), (0.2, 0), (0.2, 10.0), (0.2, True)):
        with pytest.raises(ValueError):
            se_xi(xi, n_u)  # type: ignore[arg-type]


# --- (4) profile CI ------------------------------------------------------------------
def _independent_deviance(y: np.ndarray, xi0: float, nll_hat: float) -> float:
    res = optimize.minimize(
        lambda t: nll(y, xi0, math.exp(float(t[0])), TOL),
        x0=np.array([math.log(2.0 * float(np.max(y)))]),  # a different start, inside the domain
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 10_000},
    )
    return 2.0 * (float(res.fun) - nll_hat)


def test_profile_ci_bounds_hit_chi2_quantile(settings: GpdConfig) -> None:
    y = sample(rng(), 0.25, 200)
    f = fit(y, settings)
    ci = profile_ci_xi(y, f, xi_zero_tol=TOL, settings=settings)
    assert float(stats.chi2.ppf(0.95, 1)) == CHI2_95
    assert ci.lower < f.xi < ci.upper and not ci.lower_open and ci.level == 0.95
    for bound in (ci.lower, ci.upper):
        assert abs(_independent_deviance(y, bound, f.nll) - CHI2_95) <= 1e-6
    assert profile_ci_xi(y, f, xi_zero_tol=TOL, settings=settings) == ci  # deterministic


def test_profile_ci_lower_open(settings: GpdConfig) -> None:
    """Seeded GPD(xi = -0.6, n = 30): uniform n = 30 hits the boundary rule, so it has no fit
    to profile. lower_open is sample-size dependent (False for the same xi at n = 50)."""
    y = sample(rng(), -0.6, 30)
    f = fit(y, settings)
    ci = profile_ci_xi(y, f, xi_zero_tol=TOL, settings=settings)
    assert ci.lower_open and ci.lower == -1.0 and ci.upper > f.xi
    assert abs(_independent_deviance(y, ci.upper, f.nll) - CHI2_95) <= 1e-6


def test_profile_ci_upper_failure(settings: GpdConfig) -> None:
    y = sample(rng(), 0.25, 200)
    f = fit(y, settings)
    assert profile_ci_xi(y, f, xi_zero_tol=TOL, settings=settings).upper > 0.3 > f.xi
    small = dataclasses.replace(settings, profile_xi_max=Fraction("0.3"))
    with pytest.raises(EstimationError, match="no upper crossing"):
        profile_ci_xi(y, f, xi_zero_tol=TOL, settings=small)


def test_profile_ci_rejects_mismatched_sample(settings: GpdConfig) -> None:
    y = sample(rng(), 0.25, 200)
    with pytest.raises(ValueError, match="does not match"):
        profile_ci_xi(y[:-1], fit(y, settings), xi_zero_tol=TOL, settings=settings)


# --- (5) delta method ------------------------------------------------------------------
def _analytic(xi: float, quantity: str) -> tuple[float, float]:
    r = float(Fraction(N, N_U) * P)
    lr = math.log(r)
    if xi == 0:
        h, dvar_dxi = -lr, BETA * lr**2 / 2
    else:
        h = (r**-xi - 1) / xi
        dvar_dxi = BETA * (-lr * r**-xi / xi - (r**-xi - 1) / xi**2)
    if quantity == "var":
        return dvar_dxi, h
    var = U + BETA * h
    des_dxi = (dvar_dxi - U) / (1 - xi) + (var + BETA - xi * U) / (1 - xi) ** 2
    return des_dxi, (h + 1) / (1 - xi)


@pytest.mark.parametrize("quantity", ["var", "es"])
@pytest.mark.parametrize("xi", [-0.3, 0.25, 0.6, 0.0])
def test_numeric_gradient_matches_analytic(xi: float, quantity: str, settings: GpdConfig) -> None:
    num = risk_gradient(
        xi,
        BETA,
        U,
        N,
        N_U,
        P,
        quantity,
        xi_zero_tol=TOL,
        settings=settings,  # type: ignore[arg-type]
    )
    for got, want in zip(num, _analytic(xi, quantity), strict=True):
        assert got == pytest.approx(want, rel=1e-6)


def test_covariance_reproduces_se_xi() -> None:
    cov = expected_covariance(0.25, 1.3, 400)
    assert cov[0, 0] == pytest.approx(se_xi(0.25, 400) ** 2, rel=1e-14)
    assert np.array_equal(cov, cov.T)
    assert cov[0, 1] == pytest.approx(-1.25 * 1.3 / 400, rel=1e-14)  # negative off-diagonal


def test_delta_ci_structure(settings: GpdConfig) -> None:
    f = fake_fit(0.25)
    for q in ("var", "es"):
        ci = delta_ci(f, U, N, N_U, P, q, xi_zero_tol=TOL, settings=settings)  # type: ignore[arg-type]
        assert ci.lower < ci.estimate < ci.upper and ci.level == 0.95
        z = float(stats.norm.ppf(0.975))
        assert ci.upper - ci.estimate == pytest.approx(z * ci.se, rel=1e-12)
        assert ci.estimate == tail_risk(U, 0.25, BETA, N, N_U, P, TOL)[q == "es"]


def test_delta_ci_errors(settings: GpdConfig) -> None:
    for xi in (-0.5, -0.7):
        with pytest.raises(EstimationError, match="xi > -1/2"):
            delta_ci(fake_fit(xi), U, N, N_U, P, "var", xi_zero_tol=TOL, settings=settings)
    for xi in (1.0, 1.5):
        with pytest.raises(EstimationError, match="not finite"):
            delta_ci(fake_fit(xi), U, N, N_U, P, "es", xi_zero_tol=TOL, settings=settings)
    with pytest.raises(EstimationError, match="not finite"):  # ES at xi + h >= 1
        delta_ci(fake_fit(1 - 1e-6), U, N, N_U, P, "es", xi_zero_tol=TOL, settings=settings)
    with pytest.raises(ValueError, match="quantity"):
        delta_ci(fake_fit(0.2), U, N, N_U, P, "cvar", xi_zero_tol=TOL, settings=settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantity"):
        risk_gradient(0.2, BETA, U, N, N_U, P, "cvar", xi_zero_tol=TOL, settings=settings)  # type: ignore[arg-type]


# --- Monte Carlo recovery (seeded; slow, runs in the default gate) ------------------------
@pytest.mark.slow
@pytest.mark.parametrize(("child", "xi"), [(0, -0.25), (1, 0.1), (2, 0.4)])
def test_mc_recovery_of_xi(child: int, xi: float, settings: GpdConfig) -> None:
    g = rng(child)  # SeedSequence(20261111).spawn(3)[child]
    est = np.array([fit(sample(g, xi, 400), settings).xi for _ in range(500)])
    assert abs(float(est.mean()) - xi) < 0.02
    rmse = math.sqrt(float(np.mean((est - xi) ** 2)))
    assert 0.8 <= rmse / ((1 + xi) / math.sqrt(400)) <= 1.2


@pytest.mark.slow
def test_mc_profile_ci_coverage(settings: GpdConfig) -> None:
    g, xi = rng(), 0.25
    hits = 0
    for _ in range(1000):
        y = sample(g, xi, 400)
        ci = profile_ci_xi(y, fit(y, settings), xi_zero_tol=TOL, settings=settings)
        hits += ci.lower <= xi <= ci.upper
    assert 0.92 <= hits / 1000 <= 0.98


@pytest.mark.slow
def test_mc_delta_ci(settings: GpdConfig) -> None:
    g, xi, n, n_u, p = rng(), 0.25, 1000, 400, Fraction("0.01")
    true_var, true_es = tail_risk(0.0, xi, 1.0, n, n_u, p, TOL)
    var_hat, se, params, cover_var, cover_es = [], [], [], 0, 0
    for _ in range(1000):
        f = fit(sample(g, xi, n_u), settings)
        cv = delta_ci(f, 0.0, n, n_u, p, "var", xi_zero_tol=TOL, settings=settings)
        ce = delta_ci(f, 0.0, n, n_u, p, "es", xi_zero_tol=TOL, settings=settings)
        var_hat.append(cv.estimate)
        se.append(cv.se)
        params.append((f.xi, f.beta))
        cover_var += cv.lower <= true_var <= cv.upper
        cover_es += ce.lower <= true_es <= ce.upper
    ratio = float(np.mean(se)) / float(np.std(var_hat, ddof=1))
    off_diag = float(np.cov(np.array(params).T)[0, 1]) * n_u  # empirical N_u cov(xi, beta)
    print(f"\n[descriptive] delta-CI coverage VaR99 {cover_var / 1000} ES99 {cover_es / 1000}")
    print(f"[descriptive] mean delta se / empirical SD of VaR99_hat = {ratio:.4f}")
    print(f"[descriptive] empirical N_u cov(xi_hat, beta_hat) = {off_diag:.4f}")
    assert 0.70 <= ratio <= 1.30
    # D31 (5) sign check: the off-diagonal is negative, -(1 + xi) beta (beta = 1) within 25%
    assert off_diag < 0
    assert abs(off_diag / (-(1 + xi) * 1.0) - 1) <= 0.25


# --- error messages ----------------------------------------------------------------------
def test_new_error_messages_contain_no_data_values(settings: GpdConfig) -> None:
    values = np.array([2.718281, 3.141592, 1.414213])
    f = fake_fit(-0.7182, beta=2.718281, n_excesses=3)
    raised: list[str] = []
    for call in (
        lambda: se_xi(-0.7182, 100),
        lambda: expected_covariance(-0.7182, 2.718281, 3),
        lambda: delta_ci(f, 3.141592, N, N_U, P, "var", xi_zero_tol=TOL, settings=settings),
        lambda: profile_ci_xi(values[:2], f, xi_zero_tol=TOL, settings=settings),
        lambda: profile_ci_xi(values, f, xi_zero_tol=TOL, settings=settings),
    ):
        with pytest.raises((EstimationError, ValueError)) as exc:
            call()
        raised.append(str(exc.value))
    for msg in raised:
        assert not any(t in msg for t in ("2.718", "3.141", "1.414", "0.718", "7182")), msg
