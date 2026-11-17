"""Tests for the GARCH(1,1) filter (D38). Synthetic data only."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from niftytail import garch as ga
from niftytail.config import GarchConfig, load_config
from niftytail.threshold_study import garch_returns

OMEGA, ALPHA, BETA = 0.02, 0.08, 0.90  # D38 (8): unit unconditional variance
BURN_IN, REPS = 1000, 50
CELLS = (("normal", 1000), ("normal", 2000), ("t6", 1000), ("t6", 2000))  # D38 (8), in order
TOL_ALPHA, TOL_BETA, TOL_Z = 0.02, 0.04, 0.02  # pinned in D38 (8) before any fit was run


@pytest.fixture(scope="module")
def cfg() -> GarchConfig:
    return load_config(PROJECT_ROOT / "configs" / "base.toml").garch


def simulate(g: np.random.Generator, n: int, dist: str = "normal") -> np.ndarray:
    """GARCH(1,1) path with mean 0 after the burn-in (sigma2_0 = 1)."""
    if dist == "normal":
        z = g.standard_normal(BURN_IN + n)
    else:
        z = g.standard_t(6, BURN_IN + n) * math.sqrt(4 / 6)
    return garch_returns(z, OMEGA, ALPHA, BETA, 1.0)[0][BURN_IN:]


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    return 0.5 + simulate(np.random.default_rng(20261111), 1000)


@pytest.fixture(scope="module")
def fit(series: np.ndarray, cfg: GarchConfig) -> ga.GarchFit:
    return ga.fit_garch(series, cfg)


def loop_nll(x: Any, mu: float, omega: float, alpha: float, beta: float) -> float:
    """Independent implementation: plain Python recursion, math.fsum."""
    eps = [float(v) - mu for v in x]
    s2 = math.fsum(e * e for e in eps) / len(eps)
    terms = []
    for t, e in enumerate(eps):
        if t > 0:
            s2 = omega + alpha * eps[t - 1] ** 2 + beta * s2
        terms.append(math.log(2 * math.pi) + math.log(s2) + e * e / s2)
    return 0.5 * math.fsum(terms)


def recovery_cell(c: int, cfg: GarchConfig) -> dict[str, float]:
    """Cell ``c`` of the D38 (8) recovery study: biases and residual moments over 50 fits."""
    dist, n = CELLS[c]
    alphas, betas, means, variances = [], [], [], []
    for r in range(REPS):
        ss = np.random.SeedSequence(20261111, spawn_key=(38, c, r))
        x = simulate(np.random.Generator(np.random.PCG64(ss)), n, dist)
        f = ga.fit_garch(x, cfg)  # a GarchError fails the cell
        z = ga.garch_filter(x, f)[2]
        alphas.append(f.alpha)
        betas.append(f.beta)
        means.append(float(z.mean()))
        variances.append(float(z.var()))
    return {
        "bias_alpha": float(np.mean(alphas)) - ALPHA,
        "bias_beta": float(np.mean(betas)) - BETA,
        "mean_z": float(np.mean(means)),
        "var_z": float(np.mean(variances)),
    }


# --- likelihood and recursion ---------------------------------------------------------------
@pytest.mark.parametrize(
    "params", [(0.5, 0.02, 0.08, 0.90), (0.3, 0.2, 0.3, 0.1), (0.0, 1.0, 0.0, 0.0)]
)
def test_nll_matches_an_independent_loop(series: np.ndarray, params: tuple[float, ...]) -> None:
    assert ga.garch_nll(series, *params) == pytest.approx(loop_nll(series, *params), rel=1e-12)


def test_hand_computed_three_observation_recursion_and_forecast() -> None:
    f = ga.GarchFit(mu=1.0, omega=0.5, alpha=0.25, beta=0.5, nll=0.0, start=0, flags=())
    eps, s2, z = ga.garch_filter([2.0, 0.0, 4.0], f)
    assert eps.tolist() == [1.0, -1.0, 3.0]
    expected = [11 / 3, 31 / 12, 49 / 24]  # backcast (1 + 1 + 9)/3; then 0.75 + s2/2, twice
    assert s2 == pytest.approx(expected, rel=1e-15)
    pairs = list(zip(eps.tolist(), expected, strict=True))
    assert z == pytest.approx([e / math.sqrt(v) for e, v in pairs], rel=1e-15)
    assert ga.forecast_sigma(f, 3.0, 49 / 24) == pytest.approx(math.sqrt(181 / 48), rel=1e-15)
    nll = 0.5 * sum(math.log(2 * math.pi * v) + e * e / v for e, v in pairs)
    assert ga.garch_nll([2.0, 0.0, 4.0], 1.0, 0.5, 0.25, 0.5) == pytest.approx(nll, rel=1e-14)


def test_nll_is_inf_when_not_finite_and_inputs_are_checked() -> None:
    assert ga.garch_nll([1.0, 1.0, 1.0], 1.0, 0.0, 0.1, 0.8) == math.inf  # sigma2 = 0
    for bad in ([[1.0, 2.0]], [1.0]):
        with pytest.raises(ValueError, match="one-dimensional"):
            ga.garch_nll(bad, 0.0, 1.0, 0.1, 0.8)


# --- constraint map -------------------------------------------------------------------------
def test_constraint_map_round_trip_and_stationarity_by_construction() -> None:
    mean, var = 0.3, 2.5
    for params in ((0.1, 0.4, 0.05, 0.90), (-2.0, 3.0, 0.5, 0.1), (0.3, 1e-6, 1e-4, 0.9998)):
        theta = ga.unconstrain(*params, mean, var)
        assert ga.constrain(theta, mean, var) == pytest.approx(params, rel=1e-10)
    g = np.random.default_rng(1)
    for theta in g.uniform(-30, 30, size=(2000, 4)):
        mu, omega, alpha, beta = ga.constrain(theta, mean, var)
        assert omega > 0 and alpha > 0 and beta > 0 and alpha + beta < 1
        assert math.isfinite(mu)


def test_starts_map_to_the_pinned_values() -> None:
    assert ga.STARTS == ((0.05, 0.90), (0.10, 0.85))
    for a0, b0 in ga.STARTS:
        theta = ga.unconstrain(0.3, 2.5 * (1 - a0 - b0), a0, b0, 0.3, 2.5)
        assert theta[0] == 0.0 and theta[1] == pytest.approx(math.log(1 - a0 - b0))
        assert ga.constrain(theta, 0.3, 2.5)[2:] == pytest.approx((a0, b0), rel=1e-12)


def test_objective_is_inf_outside_the_floating_point_domain(series: np.ndarray) -> None:
    m, v = float(series.mean()), float(series.var())
    ok = ga.unconstrain(m, v * 0.05, 0.05, 0.90, m, v)
    assert math.isfinite(ga._objective(ok, series, m, v))
    for i, value in ((1, 800.0), (1, -800.0), (3, 800.0), (0, math.inf)):
        theta = ok.copy()
        theta[i] = value  # omega = inf, omega = 0, alpha + beta = 1, mu = inf
        assert ga._objective(theta, series, m, v) == math.inf


# --- the fit --------------------------------------------------------------------------------
def test_fit_is_a_local_minimum_with_consistent_fields(
    series: np.ndarray, fit: ga.GarchFit
) -> None:
    params = (fit.mu, fit.omega, fit.alpha, fit.beta)
    assert fit.nll == pytest.approx(loop_nll(series, *params), rel=1e-12)
    assert fit.omega > 0 and fit.alpha > 0 and fit.beta > 0 and fit.alpha + fit.beta < 1
    assert fit.start in (0, 1) and fit.flags == ()
    for i in range(4):
        for factor in (1 - 1e-3, 1 + 1e-3):
            p = list(params)
            p[i] *= factor
            assert ga.garch_nll(series, *p) > fit.nll
    eps, s2, z = ga.garch_filter(series, fit)
    assert eps.size == s2.size == z.size == series.size  # W residuals from W observations
    assert s2[0] == pytest.approx(float(np.mean(eps**2)), rel=1e-14)
    assert z == pytest.approx(eps / np.sqrt(s2))


def test_scale_and_location_equivariance(
    series: np.ndarray, fit: ga.GarchFit, cfg: GarchConfig
) -> None:
    c, a, rel = 100.0, 3.0, 1e-6
    scaled, shifted = ga.fit_garch(series * c, cfg), ga.fit_garch(series + a, cfg)
    assert scaled.mu == pytest.approx(fit.mu * c, rel=rel)
    assert scaled.omega == pytest.approx(fit.omega * c * c, rel=rel)
    assert (scaled.alpha, scaled.beta) == pytest.approx((fit.alpha, fit.beta), rel=rel)
    assert shifted.mu == pytest.approx(fit.mu + a, rel=rel)
    assert (shifted.omega, shifted.alpha, shifted.beta) == pytest.approx(
        (fit.omega, fit.alpha, fit.beta), rel=rel
    )
    eps, s2, _ = ga.garch_filter(series, fit)
    base = ga.forecast_sigma(fit, eps[-1], s2[-1])
    eps_c, s2_c, _ = ga.garch_filter(series * c, scaled)
    assert np.sqrt(s2_c) == pytest.approx(np.sqrt(s2) * c, rel=rel)
    assert ga.forecast_sigma(scaled, eps_c[-1], s2_c[-1]) == pytest.approx(base * c, rel=rel)
    eps_a, s2_a, _ = ga.garch_filter(series + a, shifted)
    assert ga.forecast_sigma(shifted, eps_a[-1], s2_a[-1]) == pytest.approx(base, rel=rel)


def test_two_start_rule_keeps_the_lower_nll(
    series: np.ndarray, cfg: GarchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned starts, then a deliberately poor start in second and in first position."""
    for starts in (ga.STARTS, ((0.05, 0.90), (0.90, 0.05)), ((0.90, 0.05), (0.05, 0.90))):
        single = []
        for start in starts:
            monkeypatch.setattr(ga, "STARTS", (start,))
            single.append(ga.fit_garch(series, cfg).nll)
        monkeypatch.setattr(ga, "STARTS", starts)
        both = ga.fit_garch(series, cfg)
        assert both.nll == min(single) and both.start == single.index(min(single))


def test_fit_is_deterministic(series: np.ndarray, fit: ga.GarchFit, cfg: GarchConfig) -> None:
    assert ga.fit_garch(series.copy(), cfg) == fit


def test_near_boundary_flag(
    series: np.ndarray, cfg: GarchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (ga.PERSISTENCE_CEILING, ga.ALPHA_FLOOR) == (0.999, 1e-6)
    with monkeypatch.context() as mp:
        mp.setattr(ga, "PERSISTENCE_CEILING", 0.5)
        assert ga.fit_garch(series, cfg).flags == ("near_boundary",)
    monkeypatch.setattr(ga, "ALPHA_FLOOR", 0.5)
    assert ga.fit_garch(series, cfg).flags == ("near_boundary",)


def test_fit_failures_raise_garch_error(
    series: np.ndarray, cfg: GarchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ga.GarchError, match="constant") as exc:
        ga.fit_garch(np.full(1000, 0.123456789), cfg)
    assert "0.1234" not in str(exc.value)  # messages hold no data values (D23)
    for bad in (math.nan, math.inf):
        x = series.copy()
        x[10] = bad
        with pytest.raises(ga.GarchError, match="finite"):
            ga.fit_garch(x, cfg)
    with pytest.raises(ValueError, match="one-dimensional"):
        ga.fit_garch(series.reshape(2, -1), cfg)
    with monkeypatch.context() as mp:
        mp.setattr(ga, "forecast_sigma", lambda *a: math.nan)
        with pytest.raises(ga.GarchError, match="non-finite"):
            ga.fit_garch(series, cfg)
    real = ga.minimize

    def not_converged(*args: Any, **kwargs: Any) -> Any:
        res = real(*args, **kwargs)
        res.success = False
        return res

    monkeypatch.setattr(ga, "minimize", not_converged)
    with pytest.raises(ga.GarchError, match="no start converged"):
        ga.fit_garch(series, cfg)


# --- recovery study (D38 (8)) ---------------------------------------------------------------
@pytest.mark.parametrize("c", range(len(CELLS)))
def test_recovery_cell_within_the_pinned_tolerances(c: int, cfg: GarchConfig) -> None:
    out = recovery_cell(c, cfg)
    assert abs(out["bias_alpha"]) <= TOL_ALPHA
    assert abs(out["bias_beta"]) <= TOL_BETA
    assert abs(out["mean_z"]) <= TOL_Z
    assert abs(out["var_z"] - 1) <= TOL_Z


# --- D39: the four specifications F1...F4 ---------------------------------------------------
TRUE = {"phi1": 0.10, "alpha": 0.03, "gamma": 0.08, "beta": 0.88}  # D39 (f)
TOL = {"phi1": 0.03, "alpha": 0.02, "gamma": 0.04, "beta": 0.04}  # pinned in D39 (f) before any fit
RECOVERY_SPECS = ("F2", "F3", "F4")  # D39 (f): cells c = 0, 1, 2
INTERIOR = {"phi1": 0.30, "alpha": 0.08, "gamma": 0.12, "beta": 0.78}  # unit-test paths only
NEW_SPECS = pytest.mark.parametrize("spec", RECOVERY_SPECS)


def applicable(spec: str) -> list[str]:
    ar, gjr = ga.SPECS[spec]
    return [k for k in TRUE if (k != "phi1" or ar) and (k != "gamma" or gjr)]


def simulate_spec(
    g: np.random.Generator, n: int, spec: str, true: dict[str, float] = TRUE
) -> np.ndarray:
    """D39 (f): AR(1)-GJR path from sigma2_0 = 1, eps_0 = L_0 = 0; ``n + 1`` values kept."""
    ar, gjr = ga.SPECS[spec]
    phi1, gamma = true["phi1"] * ar, true["gamma"] * gjr
    alpha, beta = true["alpha"], true["beta"]
    omega = 1 - alpha - gamma / 2 - beta
    s2, eps, loss, out = 1.0, 0.0, 0.0, []
    for z in g.standard_normal(BURN_IN + n + 1).tolist():
        s2 = omega + (alpha + gamma * (eps > 0)) * eps * eps + beta * s2
        eps = math.sqrt(s2) * z
        loss = phi1 * loss + eps
        out.append(loss)
    return np.array(out[BURN_IN:])


def loop_nll_spec(
    x: Any,
    lag: float,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    phi1: float,
) -> float:
    """Independent implementation of the AR(1)-GJR likelihood: plain recursion, math.fsum."""
    prev = [lag, *[float(v) for v in x[:-1]]]
    eps = [float(v) - mu - phi1 * p for v, p in zip(x, prev, strict=True)]
    s2 = math.fsum(e * e for e in eps) / len(eps)
    terms = []
    for t, e in enumerate(eps):
        if t > 0:
            last = eps[t - 1]
            s2 = omega + (alpha + (gamma if last > 0 else 0.0)) * last**2 + beta * s2
        terms.append(math.log(2 * math.pi) + math.log(s2) + e * e / s2)
    return 0.5 * math.fsum(terms)


@pytest.fixture(scope="module")
def paths() -> dict[str, np.ndarray]:
    """One simulated path of 1001 values per new spec (the first value is the lag).

    ``INTERIOR`` parameters: the optimum is away from the alpha = 0 boundary, which the
    local-minimum and equivariance tests need.
    """
    return {
        s: simulate_spec(np.random.default_rng(39 + i), 1000, s, INTERIOR)
        for i, s in enumerate(RECOVERY_SPECS)
    }


def fit_spec(path: np.ndarray, cfg: GarchConfig, spec: str) -> ga.GarchFit:
    return ga.fit_garch(path[1:], cfg, spec, lag=float(path[0]))


def test_specs_and_parameter_counts() -> None:
    assert ga.SPECS == {
        "F1": (False, False),
        "F2": (True, False),
        "F3": (False, True),
        "F4": (True, True),
    }
    assert [ga.n_params(s) for s in ga.SPECS] == [4, 5, 5, 6]
    assert ga.GJR_STARTS == ((0.03, 0.08, 0.88), (0.05, 0.10, 0.83))
    assert ga.bic(10.0, 4, 1999) == 20.0 + 4 * math.log(1999)


@pytest.mark.parametrize("spec", list(ga.SPECS))
def test_nll_matches_an_independent_loop_for_every_spec(series: np.ndarray, spec: str) -> None:
    ar, gjr = ga.SPECS[spec]
    x, lag = series[1:], float(series[0])
    for mu, omega, alpha, beta in ((0.5, 0.02, 0.08, 0.90), (0.3, 0.2, 0.3, 0.1)):
        gamma, phi1 = 0.15 * gjr, -0.3 * ar
        got = ga.garch_nll(
            x, mu, omega, alpha, beta, gamma=gamma, phi1=phi1, lag=lag if ar else None
        )
        expected = loop_nll_spec(x, lag, mu, omega, alpha, beta, gamma, phi1)
        assert got == pytest.approx(expected, rel=1e-12)


def test_hand_computed_three_observation_gjr_recursion() -> None:
    f = ga.GarchFit(1.0, 0.5, 0.25, 0.5, 0.0, 0, (), gamma=0.5, spec="F3")
    eps, s2, _ = ga.garch_filter([2.0, 0.0, 4.0], f, lag=99.0)  # the lag is ignored (no AR)
    assert eps.tolist() == [1.0, -1.0, 3.0]
    # backcast 11/3; eps_1 = 1 > 0: 0.5 + 0.75 + (11/3)/2; eps_2 = -1: 0.5 + 0.25 + (37/12)/2
    expected = [11 / 3, 37 / 12, 55 / 24]
    assert s2 == pytest.approx(expected, rel=1e-15)
    # eps_3 = 3 > 0: 0.5 + 0.75 x 9 + (55/24)/2 = 403/48; a negative shock gets alpha only
    assert ga.forecast_sigma(f, 3.0, 55 / 24) == pytest.approx(math.sqrt(403 / 48), rel=1e-15)
    assert ga.forecast_sigma(f, -3.0, 55 / 24) == pytest.approx(math.sqrt(187 / 48), rel=1e-15)
    assert ga.forecast_mean(f, 4.0) == 1.0
    nll = 0.5 * sum(
        math.log(2 * math.pi * v) + e * e / v for e, v in zip([1, -1, 3], expected, strict=True)
    )
    got = ga.garch_nll([2.0, 0.0, 4.0], 1.0, 0.5, 0.25, 0.5, gamma=0.5)
    assert got == pytest.approx(nll, rel=1e-14)


def test_hand_computed_three_observation_ar1_recursion() -> None:
    f = ga.GarchFit(1.0, 0.5, 0.25, 0.5, 0.0, 0, (), phi1=0.5, spec="F2")
    eps, s2, z = ga.garch_filter([2.0, 0.0, 4.0], f, lag=2.0)
    assert eps.tolist() == [0.0, -2.0, 3.0]  # means 1 + 0.5 x (2, 2, 0)
    expected = [13 / 3, 8 / 3, 17 / 6]  # backcast 13/3; 0.5 + 0 + 13/6; 0.5 + 1 + 4/3
    assert s2 == pytest.approx(expected, rel=1e-15)
    assert z.size == 3  # W residuals from W observations plus one lag
    assert ga.forecast_sigma(f, 3.0, 17 / 6) == pytest.approx(math.sqrt(25 / 6), rel=1e-15)
    assert ga.forecast_mean(f, 4.0) == 3.0
    nll = 0.5 * sum(
        math.log(2 * math.pi * v) + e * e / v for e, v in zip([0, -2, 3], expected, strict=True)
    )
    got = ga.garch_nll([2.0, 0.0, 4.0], 1.0, 0.5, 0.25, 0.5, phi1=0.5, lag=2.0)
    assert got == pytest.approx(nll, rel=1e-14)
    with pytest.raises(ValueError, match="pre-window"):
        ga.garch_filter([2.0, 0.0, 4.0], f)


@pytest.mark.parametrize("spec", list(ga.SPECS))
def test_constraint_maps_hold_by_construction_for_every_spec(spec: str) -> None:
    ar, gjr = ga.SPECS[spec]
    mean, var, k = 0.3, 2.5, ga.n_params(spec)
    kw = {"gamma": 0.12 if gjr else None, "phi1": -0.4 if ar else None}
    params = (0.1, 0.4, 0.05, 0.80)
    theta = ga.unconstrain(*params, mean, var, **kw)
    assert theta.size == k
    extra = [v for v in kw.values() if v is not None]
    assert ga.constrain(theta, mean, var, spec) == pytest.approx((*params, *extra), rel=1e-10)
    g = np.random.default_rng(39)
    for theta in g.uniform(-15, 15, size=(2000, k)):
        mu, omega, alpha, beta, gamma, phi1 = ga._expand(theta, mean, var, spec)
        assert omega > 0 and alpha > 0 and beta > 0 and alpha + gamma / 2 + beta < 1
        assert (gamma > 0) == gjr and (phi1 != 0) == ar and abs(phi1) < 1
        assert math.isfinite(mu)
        assert ga.constrain(theta, mean, var, spec)[:4] == (mu, omega, alpha, beta)


def test_starts_of_every_spec_map_to_the_pinned_values(paths: dict[str, np.ndarray]) -> None:
    x, lag = paths["F4"][1:], float(paths["F4"][0])
    m, v = float(x.mean()), float(x.var())
    slope = float(np.polyfit(np.concatenate(([lag], x[:-1])), x, 1)[0])
    assert ga._ols_slope(x, lag) == pytest.approx(slope, rel=1e-9)
    garch = [(a, 0.0, b) for a, b in ga.STARTS]
    for spec, phis in (("F1", [0.0]), ("F2", [0.0, slope]), ("F3", [0.0]), ("F4", [0.0, slope])):
        ar, gjr = ga.SPECS[spec]
        variance = list(ga.GJR_STARTS) if gjr else garch
        starts = ga._starts(x, m, v, spec, lag if ar else None)
        assert [k for k, _ in starts] == list(range(2 * len(phis)))  # k = 2i + j
        for k, theta in starts:
            phi, (a0, g0, b0) = phis[k // 2], variance[k % 2]
            mu, omega, alpha, beta, gamma, phi1 = ga._expand(theta, m, v, spec)
            assert mu == pytest.approx(m * (1 - phi), rel=1e-12)  # unconditional mean = m
            assert omega == pytest.approx(v * (1 - a0 - g0 / 2 - b0), rel=1e-12)
            assert (alpha, gamma, beta) == pytest.approx((a0, g0, b0), rel=1e-12)
            assert phi1 == pytest.approx(phi, rel=1e-12, abs=0)


def test_an_unusable_ols_slope_drops_its_starts() -> None:
    explosive = 1.1 ** np.arange(1.0, 41.0)  # OLS slope 1.1: not inside (-1, 1)
    m, v = float(explosive.mean()), float(explosive.var())
    assert ga._ols_slope(explosive, 1.0) == pytest.approx(1.1, rel=1e-9)
    assert [k for k, _ in ga._starts(explosive, m, v, "F4", 1.0)] == [0, 1]
    flat_lags = np.array([1.0, 1.0, 1.0, 5.0])  # the lagged series (1, 1, 1, 1) is constant
    assert math.isnan(ga._ols_slope(flat_lags, 1.0))
    assert [k for k, _ in ga._starts(flat_lags, 2.0, 3.0, "F2", 1.0)] == [0, 1]


def test_objective_is_inf_when_phi1_or_the_gjr_persistence_leaves_the_domain(
    paths: dict[str, np.ndarray],
) -> None:
    x, lag = paths["F4"][1:], float(paths["F4"][0])
    m, v = float(x.mean()), float(x.var())
    ok = ga.unconstrain(m * 0.9, v * 0.05, 0.03, 0.88, m, v, gamma=0.08, phi1=0.1)
    assert math.isfinite(ga._objective(ok, x, m, v, "F4", lag))
    for i, value in ((5, 800.0), (5, -800.0), (4, 800.0)):  # phi1 = 1, -1; alpha+gamma/2+beta = 1
        theta = ok.copy()
        theta[i] = value
        assert ga._objective(theta, x, m, v, "F4", lag) == math.inf


@NEW_SPECS
def test_fit_of_every_new_spec_is_a_local_minimum(
    paths: dict[str, np.ndarray], cfg: GarchConfig, spec: str
) -> None:
    ar, gjr = ga.SPECS[spec]
    x, lag = paths[spec][1:], float(paths[spec][0])
    f = fit_spec(paths[spec], cfg, spec)
    params = [f.mu, f.omega, f.alpha, f.beta, f.gamma, f.phi1]
    assert f.spec == spec and f.start in range(4 if ar else 2)
    assert f.nll == pytest.approx(loop_nll_spec(x, lag, *params), rel=1e-12)
    assert (f.gamma > 0) == gjr and (f.phi1 != 0) == ar
    assert f.alpha + f.gamma / 2 + f.beta < 1 and abs(f.phi1) < 1

    def nll(p: list[float]) -> float:
        return ga.garch_nll(x, *p[:4], gamma=p[4], phi1=p[5], lag=lag if ar else None)

    for i in range(6):
        if params[i] == 0.0:
            continue
        for factor in (1 - 1e-3, 1 + 1e-3):
            p = list(params)
            p[i] *= factor
            assert nll(p) > f.nll
    eps, s2, z = ga.garch_filter(x, f, lag)
    assert eps.size == s2.size == z.size == x.size == 1000
    assert s2[0] == pytest.approx(float(np.mean(eps**2)), rel=1e-14)
    assert fit_spec(paths[spec].copy(), cfg, spec) == f  # deterministic


@NEW_SPECS
def test_scale_and_shift_equivariance_of_every_new_spec(
    paths: dict[str, np.ndarray], cfg: GarchConfig, spec: str
) -> None:
    c, a, rel = 100.0, 3.0, 1e-6
    path = paths[spec]
    base, scaled, shifted = (fit_spec(p, cfg, spec) for p in (path, path * c, path + a))

    def shape(f: ga.GarchFit) -> tuple[float, ...]:
        return (f.alpha, f.gamma, f.beta, f.phi1)

    assert scaled.mu == pytest.approx(base.mu * c, abs=rel * c)  # mu is near 0, sd near 1
    assert scaled.omega == pytest.approx(base.omega * c * c, rel=rel)
    assert shape(scaled) == pytest.approx(shape(base), rel=rel)
    assert shifted.mu == pytest.approx(base.mu + a * (1 - base.phi1), rel=rel)
    assert shifted.omega == pytest.approx(base.omega, rel=rel)
    assert shape(shifted) == pytest.approx(shape(base), rel=rel)
    eps, s2, z = ga.garch_filter(path[1:], base, float(path[0]))
    eps_c, s2_c, z_c = ga.garch_filter(path[1:] * c, scaled, float(path[0]) * c)
    eps_a, s2_a, z_a = ga.garch_filter(path[1:] + a, shifted, float(path[0]) + a)
    assert eps_c == pytest.approx(eps * c, rel=1e-4, abs=1e-4)
    assert np.sqrt(s2_c) == pytest.approx(np.sqrt(s2) * c, rel=rel)
    assert np.sqrt(s2_a) == pytest.approx(np.sqrt(s2), rel=rel)
    assert z_c == pytest.approx(z, abs=1e-5) and z_a == pytest.approx(z, abs=1e-5)
    sigma = ga.forecast_sigma(base, eps[-1], s2[-1])
    assert ga.forecast_sigma(scaled, eps_c[-1], s2_c[-1]) == pytest.approx(sigma * c, rel=rel)
    assert ga.forecast_sigma(shifted, eps_a[-1], s2_a[-1]) == pytest.approx(sigma, rel=rel)
    mean = ga.forecast_mean(base, float(path[-1]))
    assert ga.forecast_mean(scaled, float(path[-1]) * c) == pytest.approx(mean * c, abs=1e-4)
    assert ga.forecast_mean(shifted, float(path[-1]) + a) == pytest.approx(mean + a, rel=rel)


def test_ar_spec_needs_a_finite_lag(paths: dict[str, np.ndarray], cfg: GarchConfig) -> None:
    x = paths["F2"][1:]
    with pytest.raises(ValueError, match="pre-window"):
        ga.fit_garch(x, cfg, "F2")
    for bad in (math.nan, math.inf):
        with pytest.raises(ga.GarchError, match="finite"):
            ga.fit_garch(x, cfg, "F4", lag=bad)
    assert ga.fit_garch(x, cfg, "F3", lag=math.nan).spec == "F3"  # ignored: constant mean


def test_select_filter_fits_all_four_on_the_conditional_sample(
    paths: dict[str, np.ndarray], cfg: GarchConfig
) -> None:
    path = paths["F4"]
    fits, winner = ga.select_filter(path, cfg)
    assert list(fits) == ["F1", "F2", "F3", "F4"]
    for spec, f in fits.items():
        assert f == fit_spec(path, cfg, spec)  # window path[1:], lag path[0], for every spec
    bics = {s: 2 * f.nll + ga.n_params(s) * math.log(1000) for s, f in fits.items()}
    assert winner == min(bics, key=lambda s: bics[s])
    with pytest.raises(ValueError, match="one-dimensional"):
        ga.select_filter(path[:2], cfg)


def test_select_filter_ties_go_to_the_smaller_k_then_the_lower_index(
    paths: dict[str, np.ndarray], cfg: GarchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(losses: Any, cfg: Any, spec: str, *, lag: float) -> ga.GarchFit:
        return ga.GarchFit(0.0, 1.0, 0.1, 0.8, float(spec[1]), 0, (), spec=spec)

    monkeypatch.setattr(ga, "fit_garch", fake)
    for table, expected in (
        ({1.0: 7.0, 2.0: 7.0, 3.0: 7.0, 4.0: 7.0}, "F1"),  # all tied: k = 4
        ({1.0: 9.0, 2.0: 7.0, 3.0: 7.0, 4.0: 7.0}, "F2"),  # k = 5 twice: the lower index
        ({1.0: 9.0, 2.0: 8.0, 3.0: 7.0, 4.0: 7.0}, "F3"),  # k = 5 beats k = 6
        ({1.0: 9.0, 2.0: 8.0, 3.0: 7.0, 4.0: 6.0}, "F4"),  # the lowest BIC wins outright
    ):
        monkeypatch.setattr(ga, "bic", lambda nll, k, n, table=table: table[nll])
        assert ga.select_filter(paths["F4"], cfg)[1] == expected


# --- recovery study (D39 (f)) ---------------------------------------------------------------
def recovery_cell_spec(c: int, cfg: GarchConfig) -> dict[str, float]:
    """Cell ``c`` of the D39 (f) study: bias of the mean estimate of each applicable parameter."""
    spec = RECOVERY_SPECS[c]
    est: dict[str, list[float]] = {k: [] for k in applicable(spec)}
    for r in range(REPS):
        ss = np.random.SeedSequence(20261111, spawn_key=(39, c, r))
        path = simulate_spec(np.random.Generator(np.random.PCG64(ss)), 2000, spec)
        f = fit_spec(path, cfg, spec)  # a GarchError fails the cell
        for k in est:
            est[k].append(getattr(f, k))
    return {k: float(np.mean(v)) - TRUE[k] for k, v in est.items()}


@pytest.mark.parametrize("c", range(len(RECOVERY_SPECS)))
def test_recovery_cell_of_every_new_spec_within_the_pinned_tolerances(
    c: int, cfg: GarchConfig
) -> None:
    out = recovery_cell_spec(c, cfg)
    assert list(out) == applicable(RECOVERY_SPECS[c])
    for k, bias in out.items():
        assert abs(bias) <= TOL[k], (k, bias)
