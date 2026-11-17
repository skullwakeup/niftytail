"""Tests for the six forecast models (D40). Synthetic data only."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import math
import tomllib
from fractions import Fraction
from statistics import NormalDist

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from niftytail import models
from niftytail.config import Config, ConfigError, load_config, parse_config
from niftytail.garch import SPECS, fit_garch, forecast_mean, forecast_sigma, garch_filter
from niftytail.gpd import EstimationError, GpdFit
from niftytail.threshold_study import garch_returns

F = Fraction
C95, C99 = F(95, 100), F(99, 100)
Q_C = F(88, 100)
SCALE_REL = 1e-6  # D40 (f), pinned before the tests were run
# D40 (f) as amended: shift tolerance x std(window). Closed-form models 1e-9; models with a
# Nelder-Mead fit 1e-6 (optimiser stopping precision; worst observed 1.9e-7).
SHIFT_ABS = {n: 1e-9 if n in ("gaussian", "hs") else 1e-6 for n in models.MODELS}
SHIFT_EXACT = 1e-12  # D40 (f): pot_gpd on data whose shift is exact in floating point
BASE_TOL_GAUSS, BASE_TOL_HS, BASE_TOL_GARCH = 0.05, 0.15, 0.10  # D40 (f)
Z99 = 2.326348


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(PROJECT_ROOT / "configs" / "base.toml")


def rng(r: int) -> np.random.Generator:
    """D40 (f) seeds: spawn key (40, r) under the project seed."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(40, r))))


def garch_path(r: int, n: int = 1000) -> tuple[np.ndarray, float]:
    """A GARCH(1,1) loss window (unit variance) and its pre-window lag."""
    z = rng(1000 + r).standard_normal(1000 + n + 1)
    x = garch_returns(z, 0.02, 0.08, 0.90, 1.0)[0][1000:]
    return x[1:], float(x[0])


def iid(r: int) -> tuple[np.ndarray, float]:
    x = rng(r).standard_normal(1001)
    return x[1:], float(x[0])


def same(a: float, b: float) -> bool:
    """Exact equality that treats NaN as equal to NaN."""
    return a == b or (math.isnan(a) and math.isnan(b))


def finite_pairs(fc: models.Forecast) -> list[tuple[float, float]]:
    return [(fc.var[c], fc.es[c]) for c in fc.var if math.isfinite(fc.var[c])]


# --- reference values ------------------------------------------------------
@pytest.mark.parametrize(
    ("w", "c", "var", "es"),
    [
        (1000, C99, 990, F(1991, 2)),
        (1000, C95, 950, F(1951, 2)),
        (250, C99, 248, F(1246, 5)),
        (250, C95, 238, F(6106, 25)),
        (20, C95, 19, F(20)),
    ],
)
def test_hs_reference_table_exact(cfg: Config, w: int, c: Fraction, var: int, es: Fraction) -> None:
    """Methodology §4 (R2): losses 1..W, exact."""
    fc = models.hs(rng(0).permutation(np.arange(1.0, w + 1)), cfg)
    assert fc.var[c] == float(var)
    assert fc.es[c] == float(es)
    assert fc.flags == () and fc.model == "hs"


def test_hs_tail_selection(cfg: Config) -> None:
    """R2: 100 losses replaced by large gains leave VaR/ES unchanged; m losses exceed VaR."""
    x = np.arange(1.0, 1001.0)
    base = models.hs(x, cfg)
    y = x.copy()
    y[:100] = -1e6
    assert repr(models.hs(y, cfg)) == repr(base)
    for c in models.levels(cfg):
        m = 1000 * (1 - c)
        assert m.denominator == 1
        assert int(np.sum(x > base.var[c])) == m


def test_gaussian_reference_table(cfg: Config) -> None:
    """Methodology §5 (R3) at mu = 0, sigma = 1 (tolerance 1e-6)."""
    y = rng(1).standard_normal(500)
    x = (y - y.mean()) / y.std(ddof=1)
    fc = models.gaussian(x, cfg)
    for c, var, es in ((C95, 1.644854, 2.062713), (C99, 2.326348, 2.665214)):
        assert fc.var[c] == pytest.approx(var, abs=1e-6)
        assert fc.es[c] == pytest.approx(es, abs=1e-6)
    assert fc.diagnostics["sigma"] == pytest.approx(1.0, rel=1e-12)


def test_pot_gpd_worked_example(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Methodology §6 (R4) through the full pot_gpd path: u = 2, N_u = 100, n = 1000 by
    construction; the fit is pinned at (xi, beta) = (0.25, 0.5)."""
    x = np.concatenate([np.linspace(-1.0, 2.0, 900), np.linspace(2.1, 9.0, 100)])
    pinned = GpdFit(0.25, 0.5, 0.0, 100, "mle", "pwm", 0, True)
    seen: list[np.ndarray] = []

    def fake(y: np.ndarray, **_: object) -> GpdFit:
        seen.append(y)
        return pinned

    monkeypatch.setattr(models, "fit_mle", fake)
    fc = models.pot_gpd(rng(2).permutation(x), cfg, F(9, 10))
    assert (fc.diagnostics["u"], fc.diagnostics["n_u"]) == (2.0, 100.0)
    assert np.all(np.diff(seen[0]) >= 0) and seen[0].size == 100
    for c, var, es in ((C99, 3.556559, 4.742078), (C95, 2.378414, 3.171219)):
        assert fc.var[c] == pytest.approx(var, abs=1e-6)
        assert fc.es[c] == pytest.approx(es, abs=1e-6)
    assert fc.flags == ()


def test_es_not_below_var_wherever_finite(cfg: Config) -> None:
    for window, lag in (garch_path(0), iid(0)):
        for fc in models.forecast_all(window, lag, cfg, F(88, 100), Q_C).values():
            pairs = finite_pairs(fc)
            assert len(pairs) == 4, fc.model
            assert all(es >= var for var, es in pairs), fc.model


# --- equivariance -----------------------------------------------------------
@pytest.fixture(scope="module")
def equivariance(cfg: Config) -> tuple[dict[str, models.Forecast], ...]:
    window, lag = garch_path(1)
    c, a = 100.0, 3.0
    run = lambda w, p: models.forecast_all(w, p, cfg, F(88, 100), Q_C)  # noqa: E731
    return run(window, lag), run(window * c, lag * c), run(window + a, lag + a)


def test_scale_equivariance_all_six(
    equivariance: tuple[dict[str, models.Forecast], ...],
) -> None:
    base, scaled, _ = equivariance
    for name in models.MODELS:
        for c in base[name].var:
            for attr in ("var", "es"):
                v, s = getattr(base[name], attr)[c], getattr(scaled[name], attr)[c]
                assert math.isfinite(v), (name, c)
                assert s == pytest.approx(100.0 * v, rel=SCALE_REL), (name, attr, c)


def test_shift_equivariance_all_six(
    equivariance: tuple[dict[str, models.Forecast], ...],
) -> None:
    base, _, shifted = equivariance
    scale = float(np.std(garch_path(1)[0]))
    for name in models.MODELS:
        for c in base[name].var:
            for attr in ("var", "es"):
                v, s = getattr(base[name], attr)[c], getattr(shifted[name], attr)[c]
                assert abs(s - (v + 3.0)) <= SHIFT_ABS[name] * scale, (name, attr, c)


def test_pot_gpd_shift_exact_in_floating_point(cfg: Config) -> None:
    """With window and shift on a 2^-20 grid, x + a and the excesses are exact."""
    window = np.round(garch_path(1)[0] * 2**20) / 2**20
    assert np.array_equal((window + 3.0) - 3.0, window)
    base, shifted = (models.pot_gpd(w, cfg, Q_C) for w in (window, window + 3.0))
    scale = float(np.std(window))
    for c in base.var:
        assert abs(shifted.var[c] - (base.var[c] + 3.0)) <= SHIFT_EXACT * scale
        assert abs(shifted.es[c] - (base.es[c] + 3.0)) <= SHIFT_EXACT * scale


def test_unconditional_models_permutation_invariant(cfg: Config) -> None:
    window, _ = garch_path(2)
    perm = rng(3).permutation(window)
    for f in (models.gaussian, models.hs, lambda w, k: models.pot_gpd(w, k, Q_C)):
        assert repr(f(perm, cfg)) == repr(f(window, cfg))


# --- conditional consistency ------------------------------------------------
def test_conditional_matches_its_filter(cfg: Config) -> None:
    window, lag = garch_path(3)
    out = models.conditional(window, lag, cfg, Q_C)
    assert list(out) == ["garch_n", "fhs", "garch_evt"]
    fit = fit_garch(window, cfg.garch, "F4", lag=lag)
    eps, s2, z = garch_filter(window, fit, lag)
    mu, sigma = forecast_mean(fit, window[-1]), forecast_sigma(fit, eps[-1], s2[-1])
    hs_z, evt_z = models.hs(z, cfg), models.pot_gpd(z, cfg, Q_C)
    for c in models.levels(cfg):
        nd = NormalDist()
        zc = nd.inv_cdf(float(c))
        assert out["garch_n"].var[c] == mu + sigma * zc
        assert out["garch_n"].es[c] == mu + sigma * (nd.pdf(zc) / float(1 - c))
        assert out["fhs"].var[c] == mu + sigma * hs_z.var[c]
        assert out["fhs"].es[c] == mu + sigma * hs_z.es[c]
        assert same(out["garch_evt"].var[c], mu + sigma * evt_z.var[c])
        assert same(out["garch_evt"].es[c], mu + sigma * evt_z.es[c])
    for c, ref in ((C95, 1.644854), (C99, Z99)):
        assert (out["garch_n"].var[c] - mu) / sigma == pytest.approx(ref, abs=1e-6)
    d = out["garch_evt"].diagnostics
    assert (d["mu_next"], d["sigma_next"], d["filter_phi1"], d["filter_gamma"]) == (
        mu,
        sigma,
        fit.phi1,
        fit.gamma,
    )
    assert (d["u_z"], d["xi"]) == (evt_z.diagnostics["u"], evt_z.diagnostics["xi"])
    assert d["n_u"] == 120.0  # W (1 - q_c) without ties
    assert set(out["fhs"].diagnostics) == set(out["garch_n"].diagnostics)


def test_conditional_multi_q_one_fit(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tuple of levels: one filter fit; each GARCH-EVT equals its single-q call exactly."""
    window, lag = garch_path(9)
    qs = (F(86, 100), Q_C, F(90, 100), F(91, 100))
    calls: list[int] = []
    real = models.fit_garch

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(models, "fit_garch", counted)
    multi = models.conditional(window, lag, cfg, qs)
    assert len(calls) == 1
    keys = ["garch_evt@0.86", "garch_evt@0.88", "garch_evt@0.9", "garch_evt@0.91"]
    assert list(multi) == ["garch_n", "fhs", *keys]
    for q, key in zip(qs, keys, strict=True):
        single = models.conditional(window, lag, cfg, q)
        assert repr(multi[key]) == repr(single["garch_evt"])
        assert repr(multi["garch_n"]) == repr(single["garch_n"])
        assert repr(multi["fhs"]) == repr(single["fhs"])
    assert len(calls) == 1 + len(qs)
    for fc in models.conditional(window, None, cfg, qs).values():
        assert all_nan(fc) and fc.flags == ("no_pre_window_lag",)
    for bad in ((), (Q_C, Q_C)):
        with pytest.raises(ValueError, match="q_c"):
            models.conditional(window, lag, cfg, bad)


# --- baselines (D40 (f)) -----------------------------------------------------
@pytest.mark.slow
def test_baselines_iid_normal(cfg: Config) -> None:
    gauss, hist, gap = [], [], []
    for r in range(20):
        window, lag = iid(r)
        g = models.gaussian(window, cfg).var[C99]
        gauss.append(g)
        hist.append(models.hs(window, cfg).var[C99])
        gap.append(abs(models.conditional(window, lag, cfg, Q_C)["garch_n"].var[C99] - g))
    assert abs(np.mean(gauss) - Z99) <= BASE_TOL_GAUSS
    assert abs(np.mean(hist) - Z99) <= BASE_TOL_HS
    assert np.mean(gap) <= BASE_TOL_GARCH


# --- failure paths ------------------------------------------------------------
def all_nan(fc: models.Forecast) -> bool:
    return all(math.isnan(v) for v in (*fc.var.values(), *fc.es.values()))


def test_garch_error_and_constant_window(cfg: Config) -> None:
    out = models.forecast_all(np.full(1000, 0.01), 0.01, cfg, F(88, 100), Q_C)
    for name in ("garch_n", "fhs", "garch_evt"):
        assert all_nan(out[name]) and out[name].flags == ("garch_error",)
    assert all_nan(out["pot_gpd"]) and out["pot_gpd"].flags == ("gpd_error",)  # no excess
    assert out["hs"].var[C99] == out["gaussian"].var[C99] == 0.01


def test_no_pre_window_lag(cfg: Config) -> None:
    window, _ = garch_path(4)
    for fc in models.conditional(window, None, cfg, Q_C).values():
        assert all_nan(fc) and fc.flags == ("no_pre_window_lag",)
    f1 = dataclasses.replace(cfg, conditional=dataclasses.replace(cfg.conditional, filter="F1"))
    for fc in models.conditional(window, None, f1, Q_C).values():  # constant mean: no lag needed
        assert len(finite_pairs(fc)) == 4


def test_pot_gpd_estimation_error(cfg: Config) -> None:
    x = np.concatenate([np.zeros(999), [1.0]])  # one excess: fit_mle needs two
    fc = models.pot_gpd(x, cfg, Q_C)
    assert all_nan(fc) and fc.flags == ("gpd_error",) and fc.diagnostics["n_u"] == 1.0


def test_gpd_error_inside_garch_evt_only(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> GpdFit:
        raise EstimationError("forced")

    monkeypatch.setattr(models, "fit_mle", fail)
    window, lag = garch_path(5)
    out = models.conditional(window, lag, cfg, Q_C)
    assert all_nan(out["garch_evt"]) and "gpd_error" in out["garch_evt"].flags
    for name in ("garch_n", "fhs"):
        assert len(finite_pairs(out[name])) == 4 and "gpd_error" not in out[name].flags


def test_var_not_in_tail(cfg: Config) -> None:
    window, _ = garch_path(6)
    fc = models.pot_gpd(window, cfg, F(95, 100))  # N_u / W = 0.05 = p at c = 0.95
    assert math.isnan(fc.var[C95]) and math.isnan(fc.es[C95])
    assert math.isfinite(fc.var[C99]) and fc.flags == ("var_not_in_tail",)


def test_es_infinite(cfg: Config) -> None:
    x = rng(7).pareto(0.5, 1000)  # Lomax tail index 0.5: xi = 2
    fc = models.pot_gpd(x, cfg, Q_C)
    assert fc.diagnostics["xi"] >= 1
    assert all(math.isfinite(v) for v in fc.var.values())
    assert all(v == math.inf for v in fc.es.values())
    assert fc.flags == ("es_infinite",)


# --- determinism, contract, config ------------------------------------------
def test_deterministic_and_ordered(cfg: Config) -> None:
    window, lag = garch_path(8)
    a = models.forecast_all(window, lag, cfg, F(88, 100), Q_C)
    b = models.forecast_all(window.copy(), lag, cfg, F(88, 100), Q_C)
    assert repr(a) == repr(b)
    assert (
        list(a)
        == list(models.MODELS)
        == ["gaussian", "hs", "pot_gpd", "garch_n", "fhs", "garch_evt"]
    )
    for name, fc in a.items():
        assert fc.model == name and list(fc.flags) == sorted(fc.flags)
        assert list(fc.var) == list(fc.es) == [F(95, 100), F(99, 100), F(995, 1000), F(999, 1000)]


def test_signature_contract() -> None:
    """Forecast functions take only a window, the pre-window lag, the config and q levels."""
    expected = {
        models.gaussian: ["window", "cfg"],
        models.hs: ["window", "cfg"],
        models.pot_gpd: ["window", "cfg", "q"],
        models.conditional: ["window", "pre_lag", "cfg", "q_c"],
        models.forecast_all: ["window", "pre_lag", "cfg", "q_star", "q_c"],
    }
    for f, names in expected.items():
        assert list(inspect.signature(f).parameters) == names


def test_window_validation(cfg: Config) -> None:
    for bad in ([1.0], [1.0, math.nan], [[1.0, 2.0]]):
        with pytest.raises(ValueError, match="window"):
            models.hs(bad, cfg)


def test_conditional_filter_config(cfg: Config) -> None:
    """D40 (b): the configured filter is the D39 winner and is parsed strictly."""
    tracked = json.loads((PROJECT_ROOT / "reports/conditional/filter_selection.json").read_text())
    assert cfg.conditional.filter == tracked["winner"] == "F4"
    assert cfg.conditional.filter in SPECS
    with (PROJECT_ROOT / "configs" / "base.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    for value in ("F5", "f4", 4, ""):
        bad = copy.deepcopy(raw)
        bad["conditional"]["filter"] = value
        with pytest.raises(ConfigError, match=r"conditional.filter"):
            parse_config(bad)
    del raw["conditional"]["filter"]
    with pytest.raises(ConfigError, match="missing required"):
        parse_config(raw)
