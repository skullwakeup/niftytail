"""Tests for niftytail.config (configuration infrastructure, Day 2)."""

from __future__ import annotations

import copy
import dataclasses
import tomllib
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from niftytail.config import (
    Config,
    ConfigError,
    config_sha256,
    load_config,
    parse_config,
    parse_fraction,
)


@pytest.fixture
def raw(base_config_path: Path) -> dict[str, Any]:
    with base_config_path.open("rb") as fh:
        return tomllib.load(fh)


def _with(raw: dict[str, Any], section: str | None, key: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    (out[section] if section else out)[key] = value
    return out


def _without(raw: dict[str, Any], section: str, key: str) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    del out[section][key]
    return out


# --- the approved base configuration ---------------------------------------
def test_base_config_loads(base_config_path: Path) -> None:
    cfg = load_config(base_config_path)
    assert isinstance(cfg, Config)
    assert cfg.spec_version == "V2.2"


def test_base_config_matches_approved_constants(base_config_path: Path) -> None:
    cfg = load_config(base_config_path)
    assert (cfg.data.series, cfg.data.field, cfg.data.start_date) == (
        "PR",
        "Close",
        date(1995, 11, 3),
    )
    assert (cfg.data.raw_dir, cfg.data.manifest, cfg.data.validation_dir) == (
        "data/raw",
        "data/raw/MANIFEST.toml",
        "data/validation",
    )
    v = cfg.validation
    assert (v.large_return_flag, v.gap_flag_days, v.stale_run_min, v.spot_check_n) == (
        Fraction(1, 20),
        5,
        3,
        10,
    )
    assert (cfg.design.train_size, cfg.design.window, cfg.design.horizon) == (2000, 1000, 1)
    assert cfg.design.sensitivity_windows == (2000,)
    assert cfg.levels.primary == (Fraction(95, 100), Fraction(99, 100))
    assert cfg.levels.descriptive == (Fraction(995, 1000), Fraction(999, 1000))
    t = cfg.threshold
    assert t.grid == tuple(Fraction(k, 100) for k in range(80, 96))
    assert len(t.grid) == 16
    assert t.forwardstop_alpha == Fraction(1, 10)
    assert t.gof_bootstrap_reps == 999
    assert (t.stable_region_points, t.stable_region_width) == (4, Fraction(3, 100))
    assert (t.tol_xi, t.tol_var99_rel, t.tol_es99_rel) == (
        Fraction(1, 10),
        Fraction(1, 20),
        Fraction(1, 10),
    )
    assert t.sensitivity_offsets == (Fraction(-2, 100), Fraction(2, 100), Fraction(3, 100))
    assert t.xi_zero_tol == Fraction(1, 10**8)
    assert t.n_min == 100 and t.q_selected == Fraction(22, 25)  # D33 (Day 6C); D37 (Day 7C-2)
    s = cfg.simstudy
    assert (s.sample_size, s.window_refit, s.replications, s.gof_bootstrap_reps) == (
        2000,
        1000,
        200,
        199,
    )
    assert s.s3_armax_a == (Fraction(3, 10), Fraction(6, 10))
    assert (s.trigger_false_selection, s.trigger_fdr_multiple, s.trigger_rule_failure) == (
        Fraction(1, 5),
        Fraction(2),
        Fraction(1, 10),
    )
    b = cfg.backtest
    assert (b.christoffersen_mc_reps, b.es_residual_bootstrap_reps, b.es_residual_min_breaches) == (
        9999,
        9999,
        20,
    )
    assert (b.traffic_light_level, b.traffic_light_block) == (Fraction(99, 100), 250)
    assert (b.traffic_light_green_max, b.traffic_light_yellow_max) == (4, 9)
    assert cfg.bootstrap.replications == 1999
    assert (cfg.runtime.seed, cfg.runtime.blas_threads) == (20261111, 1)
    d = cfg.dependence  # D29
    assert (d.acf_max_lag, d.ljung_box_lags, d.arch_lm_lags) == (20, (10, 20), (5, 10))
    assert d.exceedance_levels == (Fraction(4, 5), Fraction(9, 10), Fraction(19, 20))
    assert (d.acf_band_z, d.output_dir) == (Fraction(196, 100), "reports/eda")
    g = cfg.gpd  # D30
    assert (g.optimizer, g.xatol, g.fatol, g.maxiter, g.maxfev) == (
        "Nelder-Mead",
        Fraction(1, 10**10),
        Fraction(1, 10**12),
        5000,
        10000,
    )
    assert (g.ci_level, g.profile_xtol, g.profile_xi_max, g.xi_boundary_tol) == (  # D31
        Fraction(95, 100),
        Fraction(1, 10**10),
        Fraction(10),
        Fraction(1, 10**6),
    )
    assert (g.delta_step_xi, g.delta_step_beta_rel) == (Fraction(1, 10**5), Fraction(1, 10**6))
    ga = cfg.garch  # D38
    assert (ga.xatol, ga.fatol, ga.maxiter, ga.maxfev, ga.simplex_step) == (
        Fraction(1, 10**8),
        Fraction(1, 10**10),
        5000,
        10000,
        Fraction(1, 10),
    )
    assert cfg.conditional.q_c == Fraction(22, 25) == t.q_selected  # D39 (a)
    assert cfg.nmin_study.var_level == Fraction(99, 100)  # D32
    assert cfg.nmin_study.override_decision == "D33"


def test_tail_counts_are_exact(base_config_path: Path) -> None:
    """A0 float trap: W*(1-c) must be exactly 50 and 10, never 51 and 11."""
    cfg = load_config(base_config_path)
    w = cfg.design.window
    counts = [w * (1 - c) for c in cfg.levels.primary]
    assert counts == [50, 10]
    assert all(isinstance(m, Fraction) and m.denominator == 1 for m in counts)


def test_config_is_frozen(base_config_path: Path) -> None:
    cfg = load_config(base_config_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.design.window = 500  # type: ignore[misc]


def test_config_sha256_is_deterministic(base_config_path: Path) -> None:
    h1, h2 = config_sha256(base_config_path), config_sha256(base_config_path)
    assert h1 == h2 and len(h1) == 64


# --- exact parsing ----------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.99", Fraction(99, 100)),
        ("0.95", Fraction(19, 20)),
        ("1e-8", Fraction(1, 10**8)),
        ("-0.02", Fraction(-1, 50)),
        (" 0.10 ", Fraction(1, 10)),
    ],
)
def test_parse_fraction_exact(text: str, expected: Fraction) -> None:
    assert parse_fraction(text, "x") == expected


@pytest.mark.parametrize("bad", [0.99, 1, True, None, "abc", "", "1/0"])
def test_parse_fraction_rejects_non_strings_and_garbage(bad: object) -> None:
    with pytest.raises(ConfigError):
        parse_fraction(bad, "x")


def test_float_level_is_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="exact decimal"):
        parse_config(_with(raw, "levels", "primary", [0.95, 0.99]))


# --- structural validation --------------------------------------------------
def test_unknown_top_level_key(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="unknown top-level"):
        parse_config(_with(raw, None, "typo_section", {}))


def test_unknown_key_in_section(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(_with(raw, "design", "windwo", 1000))


def test_missing_required_key(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="missing required"):
        parse_config(_without(raw, "design", "window"))


def test_missing_section(raw: dict[str, Any]) -> None:
    out = copy.deepcopy(raw)
    del out["backtest"]
    with pytest.raises(ConfigError, match=r"\[backtest\]"):
        parse_config(out)


def test_wrong_spec_version(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="spec_version"):
        parse_config(_with(raw, None, "spec_version", "V2.1"))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(p)


# --- semantic validation ----------------------------------------------------
@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("data", "series", "NIFTY", "PR"),
        ("data", "start_date", "03-11-1995", "ISO"),
        ("design", "horizon", 10, "one-day"),
        ("design", "window", 2500, "inside the training period"),
        ("design", "sensitivity_windows", [3000], "inside the training period"),
        ("design", "window", True, "integer"),
        ("levels", "primary", ["0.99", "0.95"], "increasing"),
        ("levels", "primary", ["0.95", "1.0"], "between 0 and 1"),
        ("levels", "descriptive", ["0.99"], "disjoint"),
        ("threshold", "grid", ["0.80", "0.82", "0.83", "0.84"], "equally spaced"),
        ("threshold", "grid", ["0.90", "0.80", "0.85", "0.95"], "increasing"),
        ("threshold", "stable_region_width", "0.04", "grid step"),
        ("threshold", "forwardstop_alpha", "0", "between 0 and 1"),
        ("threshold", "sensitivity_offsets", ["0", "0.02"], "non-zero"),
        ("threshold", "q_selected", "0.905", "not a grid candidate"),
        ("simstudy", "s4_garch_b1", "0.95", "a1 \\+ b1 < 1"),
        ("simstudy", "window_refit", 3000, "window_refit"),
        ("simstudy", "s4_t_df", 2, ">= 3"),
        ("backtest", "traffic_light_level", "0.975", "levels.primary"),
        ("backtest", "traffic_light_green_max", 9, "green_max"),
        ("backtest", "dm_lag_rule", "andrews", "D15"),
        ("bootstrap", "block_rule", "politis_white", "D16"),
        ("runtime", "blas_threads", 4, "pins BLAS"),
        ("data", "raw_dir", "/abs/data/raw", "relative path"),
        ("data", "raw_dir", "C:\\data\\raw", "relative path"),
        ("data", "manifest", "\\\\server\\share\\m.toml", "relative path"),
        ("data", "manifest", "data/../MANIFEST.toml", "'\\.\\.'"),
        ("data", "validation_dir", "..\\outside", "'\\.\\.'"),
        ("data", "validation_dir", "", "non-empty"),
        ("data", "raw_dir", 5, "non-empty string"),
        ("validation", "large_return_flag", "1.5", "between 0 and 1"),
        ("validation", "large_return_flag", "0", "between 0 and 1"),
        ("validation", "large_return_flag", 0.05, "exact decimal"),
        ("validation", "gap_flag_days", 0, ">= 1"),
        ("validation", "stale_run_min", -1, ">= 1"),
        ("validation", "spot_check_n", True, "integer"),
        ("validation", "spot_check_n", "10", "integer"),
        ("dependence", "acf_max_lag", 0, ">= 1"),
        ("dependence", "acf_max_lag", 15, "largest ljung_box_lags"),
        ("dependence", "ljung_box_lags", [20, 10], "increasing"),
        ("dependence", "ljung_box_lags", [0, 10], ">= 1"),
        ("dependence", "ljung_box_lags", [], "non-empty list"),
        ("dependence", "arch_lm_lags", [5, 5], "increasing"),
        ("dependence", "arch_lm_lags", [-1], ">= 1"),
        ("dependence", "exceedance_levels", ["0.95", "0.90"], "increasing"),
        ("dependence", "exceedance_levels", ["1.0"], "between 0 and 1"),
        ("dependence", "exceedance_levels", ["0.97"], "grid candidates"),
        ("dependence", "exceedance_levels", [0.9], "exact decimal"),
        ("dependence", "acf_band_z", "0", "> 0"),
        ("dependence", "acf_band_z", 1.96, "exact decimal"),
        ("dependence", "output_dir", "/abs/eda", "relative path"),
        ("dependence", "output_dir", "reports/../../eda", "'\\.\\.'"),
        ("gpd", "optimizer", "BFGS", "D30"),
        ("gpd", "optimizer", "", "non-empty"),
        ("gpd", "xatol", 1e-10, "exact decimal"),
        ("gpd", "xatol", "0", "> 0"),
        ("gpd", "fatol", "-1e-12", "> 0"),
        ("gpd", "maxiter", 0, ">= 1"),
        ("gpd", "maxiter", True, "integer"),
        ("gpd", "maxfev", "10000", "integer"),
        ("gpd", "maxfev", -5, ">= 1"),
        ("gpd", "ci_level", "1", "between 0 and 1"),
        ("gpd", "ci_level", 0.95, "exact decimal"),
        ("gpd", "profile_xtol", "0", "> 0"),
        ("gpd", "profile_xi_max", "-10", "> 0"),
        ("gpd", "xi_boundary_tol", "1", "between 0 and 1"),
        ("gpd", "xi_boundary_tol", "0", "between 0 and 1"),
        ("gpd", "delta_step_xi", "0", "> 0"),
        ("gpd", "delta_step_beta_rel", "-1e-6", "> 0"),
        ("nmin_study", "var_level", "1", "between 0 and 1"),
        ("nmin_study", "var_level", 0.99, "exact decimal"),
        ("threshold", "n_min", 80, "one of nmin_study.n_u"),
        ("threshold", "n_min", "100", "integer"),
        ("nmin_study", "override_decision", 33, "non-empty string"),
        ("nmin_study", "override_decision", "", "non-empty string"),
        ("nmin_study", "override_decision", ["D33"], "non-empty string"),
    ],
)
def test_semantic_violations(
    raw: dict[str, Any], section: str, key: str, value: Any, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        parse_config(_with(raw, section, key, value))


def test_validation_section_is_strict(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(_with(raw, "validation", "gap_days", 5))
    with pytest.raises(ConfigError, match="missing required"):
        parse_config(_without(raw, "validation", "spot_check_n"))
    out = copy.deepcopy(raw)
    del out["validation"]
    with pytest.raises(ConfigError, match=r"\[validation\]"):
        parse_config(out)


def test_dependence_section_is_strict(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(_with(raw, "dependence", "acf_lags", 20))
    with pytest.raises(ConfigError, match="missing required"):
        parse_config(_without(raw, "dependence", "output_dir"))
    out = copy.deepcopy(raw)
    del out["dependence"]
    with pytest.raises(ConfigError, match=r"\[dependence\]"):
        parse_config(out)


def test_gpd_section_is_strict(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(_with(raw, "gpd", "method", "Nelder-Mead"))
    with pytest.raises(ConfigError, match="missing required"):
        parse_config(_without(raw, "gpd", "maxfev"))
    out = copy.deepcopy(raw)
    del out["gpd"]
    with pytest.raises(ConfigError, match=r"\[gpd\]"):
        parse_config(out)


def test_garch_section_is_strict(raw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(_with(raw, "garch", "optimizer", "Nelder-Mead"))
    with pytest.raises(ConfigError, match="missing required"):
        parse_config(_without(raw, "garch", "maxfev"))
    with pytest.raises(ConfigError, match="must be a string"):
        parse_config(_with(raw, "garch", "xatol", 1e-8))
    with pytest.raises(ConfigError, match="must be > 0"):
        parse_config(_with(raw, "garch", "simplex_step", "0"))
    with pytest.raises(ConfigError, match="must be >= 1"):
        parse_config(_with(raw, "garch", "maxiter", 0))
    out = copy.deepcopy(raw)
    del out["garch"]
    with pytest.raises(ConfigError, match=r"\[garch\]"):
        parse_config(out)


def test_conditional_section_is_strict(raw: dict[str, Any]) -> None:
    """D39 (a): q_c is required, an exact decimal string, a grid member and <= q_max."""
    assert parse_config(_with(raw, "conditional", "q_c", "0.90")).conditional.q_c == Fraction(9, 10)
    for value, match in (
        (0.88, "must be a string"),
        ("0.885", "not a grid candidate"),
        ("0.79", "not a grid candidate"),
        ("0.91", "q_max"),  # (1 - 0.91) x 1000 = 90 < n_min = 100
    ):
        with pytest.raises(ConfigError, match=match):
            parse_config(_with(raw, "conditional", "q_c", value))
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(_with(raw, "conditional", "q_selected", "0.88"))
    with pytest.raises(ConfigError, match="missing required"):
        parse_config(_without(raw, "conditional", "q_c"))
    out = copy.deepcopy(raw)
    del out["conditional"]
    with pytest.raises(ConfigError, match=r"\[conditional\]"):
        parse_config(out)
    no_n_min = _without(raw, "threshold", "n_min")  # before the Day-6 gate: the grid check only
    del no_n_min["threshold"]["q_selected"]
    no_n_min["conditional"]["q_c"] = "0.95"
    assert parse_config(no_n_min).conditional.q_c == Fraction(19, 20)


def test_nmin_override_optional_and_unknown_keys_rejected(raw: dict[str, Any]) -> None:
    assert (
        parse_config(_without(raw, "nmin_study", "override_decision")).nmin_study.override_decision
        is None
    )
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(_with(raw, "nmin_study", "override", "D33"))


def test_valid_alternative_data_paths(raw: dict[str, Any]) -> None:
    out = _with(raw, "data", "raw_dir", "data\\raw_v2")
    out["data"]["validation_dir"] = "data/validation/run_a"
    out["validation"]["large_return_flag"] = "0.075"
    cfg = parse_config(out)
    assert cfg.data.raw_dir == "data\\raw_v2"
    assert cfg.data.validation_dir == "data/validation/run_a"
    assert cfg.validation.large_return_flag == Fraction(3, 40)


def test_gate_values_optional_and_validated(raw: dict[str, Any]) -> None:
    out = _with(raw, "threshold", "n_min", 100)
    out["threshold"]["q_selected"] = "0.90"  # (1 - 0.90) x 1000 = 100 >= n_min
    cfg = parse_config(out)
    assert (cfg.threshold.n_min, cfg.threshold.q_selected) == (100, Fraction(9, 10))
    out["threshold"]["q_selected"] = "0.91"  # (1 - 0.91) x 1000 = 90 < 100
    with pytest.raises(ConfigError, match="q_max"):
        parse_config(out)
