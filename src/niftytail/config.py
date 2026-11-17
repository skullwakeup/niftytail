"""Configuration loading and validation for NIFTYTAIL.

Loads a TOML file (``configs/base.toml``) into frozen dataclasses holding the
pre-registered constants of the approved V2.2 plan.

Design rules (docs/methodology.md 1, 15; plan A0, A5e):

* Probabilities, confidence levels, quantile levels and tolerances must be given
  as TOML *strings* and are parsed with :class:`fractions.Fraction`. Floats are
  rejected so that tail counts such as ``W * (1 - c)`` are always exact.
* Unknown keys and missing required keys are errors: a typo can never silently
  fall back to a default.
* Gate-controlled values (``threshold.n_min`` on Day 6, ``threshold.q_selected``
  on Day 7) are optional; absence means "not yet decided" and is ``None``.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from itertools import pairwise
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

__all__ = [
    "BacktestConfig",
    "BootstrapConfig",
    "ConditionalConfig",
    "Config",
    "ConfigError",
    "DataConfig",
    "DependenceConfig",
    "DesignConfig",
    "GarchConfig",
    "GpdConfig",
    "LevelsConfig",
    "NminStudyConfig",
    "RuntimeConfig",
    "SimStudyConfig",
    "ThresholdConfig",
    "ValidationConfig",
    "config_sha256",
    "load_config",
    "parse_config",
    "parse_fraction",
]

SPEC_VERSION = "V2.2"


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or violates the spec."""


# ---------------------------------------------------------------------------
# Scalar parsers
# ---------------------------------------------------------------------------
def parse_fraction(value: object, name: str) -> Fraction:
    """Parse an exact decimal string (e.g. ``"0.99"``, ``"1e-8"``) into a Fraction.

    Raises:
        ConfigError: if ``value`` is not a string or not a valid decimal.
    """
    if not isinstance(value, str):
        raise ConfigError(
            f'{name}: must be a string holding an exact decimal (e.g. "0.99"), '
            f"got {type(value).__name__} {value!r}"
        )
    try:
        return Fraction(value.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ConfigError(f"{name}: not a valid decimal string: {value!r}") from exc


def _int(value: object, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name}: must be an integer, got {type(value).__name__} {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}: must be >= {minimum}, got {value}")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name}: must be true or false, got {value!r}")
    return value


def _str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}: must be a non-empty string, got {value!r}")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name}: must be a non-empty list, got {value!r}")
    return value


def _fractions(value: object, name: str) -> tuple[Fraction, ...]:
    return tuple(parse_fraction(v, f"{name}[{i}]") for i, v in enumerate(_list(value, name)))


def _ints(value: object, name: str, minimum: int | None = None) -> tuple[int, ...]:
    return tuple(_int(v, f"{name}[{i}]", minimum) for i, v in enumerate(_list(value, name)))


def _open_unit(x: Fraction, name: str) -> Fraction:
    if not 0 < x < 1:
        raise ConfigError(f"{name}: must lie strictly between 0 and 1, got {x}")
    return x


def _positive(x: Fraction, name: str) -> Fraction:
    if x <= 0:
        raise ConfigError(f"{name}: must be > 0, got {x}")
    return x


def _strictly_increasing(xs: tuple[Any, ...], name: str) -> None:
    if any(b <= a for a, b in pairwise(xs)):
        raise ConfigError(f"{name}: values must be strictly increasing, got {list(map(str, xs))}")


def _section(
    raw: dict[str, Any], name: str, required: set[str], optional: frozenset[str] = frozenset()
) -> dict[str, Any]:
    sec = raw.get(name)
    if not isinstance(sec, dict):
        raise ConfigError(f"missing or invalid section [{name}]")
    unknown = set(sec) - required - optional
    missing = required - set(sec)
    if unknown:
        raise ConfigError(f"[{name}]: unknown key(s) {sorted(unknown)}")
    if missing:
        raise ConfigError(f"[{name}]: missing required key(s) {sorted(missing)}")
    return sec


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DataConfig:
    """Data selection and repo-relative data locations (D1, D2, D23)."""

    series: str
    field: str
    start_date: date
    raw_dir: str
    manifest: str
    validation_dir: str


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Data-validation flag thresholds (D25). They affect reporting only, never the data."""

    large_return_flag: Fraction
    gap_flag_days: int
    stale_run_min: int
    spot_check_n: int


@dataclass(frozen=True, slots=True)
class DesignConfig:
    """Sample design (D3): training length T0, rolling window W, horizon."""

    train_size: int
    window: int
    horizon: int
    sensitivity_windows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LevelsConfig:
    """Confidence levels c (A0). Tail probability is p = 1 - c."""

    primary: tuple[Fraction, ...]
    descriptive: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Threshold-selection constants (D4a-D4d, D5, D22)."""

    grid: tuple[Fraction, ...]
    forwardstop_alpha: Fraction
    gof_bootstrap_reps: int
    stable_region_points: int
    stable_region_width: Fraction
    tol_xi: Fraction
    tol_var99_rel: Fraction
    tol_es99_rel: Fraction
    split_half_flag: Fraction
    sensitivity_offsets: tuple[Fraction, ...]
    xi_zero_tol: Fraction
    n_min: int | None
    q_selected: Fraction | None


@dataclass(frozen=True, slots=True)
class NminStudyConfig:
    """Day-6 N_min simulation rule (D4d)."""

    xi_true: tuple[Fraction, ...]
    beta: Fraction
    n_u: tuple[int, ...]
    replications: int
    min_convergence: Fraction
    max_rmse_xi: Fraction
    max_rel_rmse_var99: Fraction
    var_level: Fraction
    override_decision: str | None


@dataclass(frozen=True, slots=True)
class SimStudyConfig:
    """A3 threshold-procedure simulation study (D13)."""

    sample_size: int
    window_refit: int
    replications: int
    replications_fallback: int
    max_runtime_hours: int
    gof_bootstrap_reps: int
    s1_q0: tuple[Fraction, ...]
    s1_xi: tuple[Fraction, ...]
    s2_t_df: int
    s3_q0: Fraction
    s3_xi: Fraction
    s3_armax_a: tuple[Fraction, ...]
    s4_garch_a1: Fraction
    s4_garch_b1: Fraction
    s4_t_df: int
    s4_burn_in: int
    s4_reference_size: int
    trigger_false_selection: Fraction
    trigger_fdr_multiple: Fraction
    trigger_rule_failure: Fraction


@dataclass(frozen=True, slots=True)
class DependenceConfig:
    """Day-4 dependence-battery constants (D29; methodology 9)."""

    acf_max_lag: int
    ljung_box_lags: tuple[int, ...]
    arch_lm_lags: tuple[int, ...]
    exceedance_levels: tuple[Fraction, ...]
    acf_band_z: Fraction
    output_dir: str


@dataclass(frozen=True, slots=True)
class GpdConfig:
    """GPD optimiser (D30) and uncertainty settings (D31). Fractions become floats only at use."""

    optimizer: str
    xatol: Fraction
    fatol: Fraction
    maxiter: int
    maxfev: int
    ci_level: Fraction
    profile_xtol: Fraction
    profile_xi_max: Fraction
    xi_boundary_tol: Fraction
    delta_step_xi: Fraction
    delta_step_beta_rel: Fraction


@dataclass(frozen=True, slots=True)
class GarchConfig:
    """GARCH(1,1) QMLE optimiser settings (D38). Fractions become floats only at use."""

    xatol: Fraction
    fatol: Fraction
    maxiter: int
    maxfev: int
    simplex_step: Fraction


@dataclass(frozen=True, slots=True)
class ConditionalConfig:
    """Conditional-model constants: residual threshold level ``q_c`` (D39), filter spec (D40)."""

    q_c: Fraction
    filter: str


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest constants (D8, D14, D15, D17, D19, D20; A2, A4, A5)."""

    kupiec_level: Fraction
    christoffersen_mc_reps: int
    es_residual_bootstrap_reps: int
    es_residual_min_breaches: int
    traffic_light_level: Fraction
    traffic_light_block: int
    traffic_light_green_max: int
    traffic_light_yellow_max: int
    dm_lag_rule: str
    dm_lag_sensitivity_multiplier: int
    common_mask_min_share: Fraction


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Moving-block bootstrap constants (D16; A5c)."""

    block_rule: str
    replications: int
    sensitivity_multipliers: tuple[Fraction, ...]
    circular: bool


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Reproducibility settings (A5e)."""

    seed: int
    blas_threads: int


@dataclass(frozen=True, slots=True)
class Config:
    """Complete, validated NIFTYTAIL configuration."""

    spec_version: str
    data: DataConfig
    validation: ValidationConfig
    design: DesignConfig
    levels: LevelsConfig
    threshold: ThresholdConfig
    nmin_study: NminStudyConfig
    simstudy: SimStudyConfig
    dependence: DependenceConfig
    gpd: GpdConfig
    garch: GarchConfig
    conditional: ConditionalConfig
    backtest: BacktestConfig
    bootstrap: BootstrapConfig
    runtime: RuntimeConfig


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------
def _rel_path(value: object, name: str) -> str:
    """A non-empty, relative path without '..' components (POSIX or Windows spelling)."""
    s = _str(value, name)
    posix, win = PurePosixPath(s), PureWindowsPath(s)
    if posix.is_absolute() or win.is_absolute() or win.drive or win.root:
        raise ConfigError(f"{name}: must be a relative path, got {s!r}")
    if ".." in posix.parts or ".." in win.parts:
        raise ConfigError(f"{name}: must not contain '..', got {s!r}")
    return s


def _parse_data(raw: dict[str, Any]) -> DataConfig:
    s = _section(
        raw,
        "data",
        {"series", "field", "start_date", "raw_dir", "manifest", "validation_dir"},
    )
    series = _str(s["series"], "data.series")
    if series not in {"PR", "TRI"}:
        raise ConfigError(f"data.series: must be 'PR' or 'TRI' (D2), got {series!r}")
    start = s["start_date"]
    if not isinstance(start, str):
        raise ConfigError(f"data.start_date: must be an ISO date string, got {start!r}")
    try:
        start_date = date.fromisoformat(start)
    except ValueError as exc:
        raise ConfigError(f"data.start_date: not an ISO date: {start!r}") from exc
    return DataConfig(
        series=series,
        field=_str(s["field"], "data.field"),
        start_date=start_date,
        raw_dir=_rel_path(s["raw_dir"], "data.raw_dir"),
        manifest=_rel_path(s["manifest"], "data.manifest"),
        validation_dir=_rel_path(s["validation_dir"], "data.validation_dir"),
    )


def _parse_validation(raw: dict[str, Any]) -> ValidationConfig:
    s = _section(
        raw,
        "validation",
        {"large_return_flag", "gap_flag_days", "stale_run_min", "spot_check_n"},
    )
    flag = _open_unit(
        parse_fraction(s["large_return_flag"], "validation.large_return_flag"),
        "validation.large_return_flag",
    )
    return ValidationConfig(
        large_return_flag=flag,
        gap_flag_days=_int(s["gap_flag_days"], "validation.gap_flag_days", 1),
        stale_run_min=_int(s["stale_run_min"], "validation.stale_run_min", 1),
        spot_check_n=_int(s["spot_check_n"], "validation.spot_check_n", 1),
    )


def _parse_design(raw: dict[str, Any]) -> DesignConfig:
    s = _section(raw, "design", {"train_size", "window", "horizon", "sensitivity_windows"})
    train = _int(s["train_size"], "design.train_size", 1)
    window = _int(s["window"], "design.window", 1)
    horizon = _int(s["horizon"], "design.horizon", 1)
    sens = _ints(s["sensitivity_windows"], "design.sensitivity_windows", 1)
    if horizon != 1:
        raise ConfigError(
            f"design.horizon: only one-day-ahead forecasts are specified, got {horizon}"
        )
    for name, w in [("design.window", window)] + [
        (f"design.sensitivity_windows[{i}]", w) for i, w in enumerate(sens)
    ]:
        if w > train:
            raise ConfigError(
                f"{name}: the first rolling window must lie inside the training period, "
                f"so window ({w}) <= train_size ({train})"
            )
    return DesignConfig(train_size=train, window=window, horizon=horizon, sensitivity_windows=sens)


def _parse_levels(raw: dict[str, Any]) -> LevelsConfig:
    s = _section(raw, "levels", {"primary", "descriptive"})
    primary = tuple(
        _open_unit(c, "levels.primary") for c in _fractions(s["primary"], "levels.primary")
    )
    descr = tuple(
        _open_unit(c, "levels.descriptive")
        for c in _fractions(s["descriptive"], "levels.descriptive")
    )
    _strictly_increasing(primary, "levels.primary")
    _strictly_increasing(descr, "levels.descriptive")
    if set(primary) & set(descr):
        raise ConfigError("levels: primary and descriptive levels must be disjoint")
    return LevelsConfig(primary=primary, descriptive=descr)


def _parse_threshold(raw: dict[str, Any], design: DesignConfig) -> ThresholdConfig:
    s = _section(
        raw,
        "threshold",
        {
            "grid",
            "forwardstop_alpha",
            "gof_bootstrap_reps",
            "stable_region_points",
            "stable_region_width",
            "tol_xi",
            "tol_var99_rel",
            "tol_es99_rel",
            "split_half_flag",
            "sensitivity_offsets",
            "xi_zero_tol",
        },
        frozenset({"n_min", "q_selected"}),
    )
    grid = tuple(_open_unit(q, "threshold.grid") for q in _fractions(s["grid"], "threshold.grid"))
    _strictly_increasing(grid, "threshold.grid")
    points = _int(s["stable_region_points"], "threshold.stable_region_points", 2)
    width = _positive(
        parse_fraction(s["stable_region_width"], "threshold.stable_region_width"),
        "threshold.stable_region_width",
    )
    if len(grid) < points:
        raise ConfigError("threshold.grid: fewer candidates than stable_region_points")
    steps = {b - a for a, b in pairwise(grid)}
    if len(steps) != 1:
        raise ConfigError("threshold.grid: candidates must be equally spaced")
    if width != (points - 1) * next(iter(steps)):
        raise ConfigError(
            "threshold.stable_region_width must equal (stable_region_points - 1) x grid step"
        )
    offsets = _fractions(s["sensitivity_offsets"], "threshold.sensitivity_offsets")
    if any(o == 0 for o in offsets):
        raise ConfigError("threshold.sensitivity_offsets: offsets must be non-zero")

    n_min = _int(s["n_min"], "threshold.n_min", 1) if "n_min" in s else None
    q_sel: Fraction | None = None
    if "q_selected" in s:
        q_sel = parse_fraction(s["q_selected"], "threshold.q_selected")
        if q_sel not in grid:
            raise ConfigError(f"threshold.q_selected: {q_sel} is not a grid candidate")
        if n_min is not None and (1 - q_sel) * design.window < n_min:
            raise ConfigError(
                "threshold.q_selected violates q <= q_max = 1 - n_min / window "
                f"((1 - {q_sel}) x {design.window} < {n_min})"
            )

    def frac01(key: str) -> Fraction:
        return _open_unit(parse_fraction(s[key], f"threshold.{key}"), f"threshold.{key}")

    def fracpos(key: str) -> Fraction:
        return _positive(parse_fraction(s[key], f"threshold.{key}"), f"threshold.{key}")

    return ThresholdConfig(
        grid=grid,
        forwardstop_alpha=frac01("forwardstop_alpha"),
        gof_bootstrap_reps=_int(s["gof_bootstrap_reps"], "threshold.gof_bootstrap_reps", 1),
        stable_region_points=points,
        stable_region_width=width,
        tol_xi=fracpos("tol_xi"),
        tol_var99_rel=fracpos("tol_var99_rel"),
        tol_es99_rel=fracpos("tol_es99_rel"),
        split_half_flag=fracpos("split_half_flag"),
        sensitivity_offsets=offsets,
        xi_zero_tol=fracpos("xi_zero_tol"),
        n_min=n_min,
        q_selected=q_sel,
    )


def _parse_nmin_study(raw: dict[str, Any]) -> NminStudyConfig:
    s = _section(
        raw,
        "nmin_study",
        {
            "xi_true",
            "beta",
            "n_u",
            "replications",
            "min_convergence",
            "max_rmse_xi",
            "max_rel_rmse_var99",
            "var_level",
        },
        frozenset({"override_decision"}),
    )
    xi = tuple(
        _open_unit(x, "nmin_study.xi_true") for x in _fractions(s["xi_true"], "nmin_study.xi_true")
    )
    n_u = _ints(s["n_u"], "nmin_study.n_u", 1)
    _strictly_increasing(n_u, "nmin_study.n_u")
    conv = parse_fraction(s["min_convergence"], "nmin_study.min_convergence")
    if not 0 < conv <= 1:
        raise ConfigError(f"nmin_study.min_convergence: must lie in (0, 1], got {conv}")
    return NminStudyConfig(
        xi_true=xi,
        beta=_positive(parse_fraction(s["beta"], "nmin_study.beta"), "nmin_study.beta"),
        n_u=n_u,
        replications=_int(s["replications"], "nmin_study.replications", 1),
        min_convergence=conv,
        max_rmse_xi=_positive(
            parse_fraction(s["max_rmse_xi"], "nmin_study.max_rmse_xi"), "nmin_study.max_rmse_xi"
        ),
        max_rel_rmse_var99=_positive(
            parse_fraction(s["max_rel_rmse_var99"], "nmin_study.max_rel_rmse_var99"),
            "nmin_study.max_rel_rmse_var99",
        ),
        var_level=_open_unit(
            parse_fraction(s["var_level"], "nmin_study.var_level"), "nmin_study.var_level"
        ),
        override_decision=(
            _str(s["override_decision"], "nmin_study.override_decision")
            if "override_decision" in s
            else None
        ),
    )


def _parse_simstudy(raw: dict[str, Any]) -> SimStudyConfig:
    keys = {
        "sample_size",
        "window_refit",
        "replications",
        "replications_fallback",
        "max_runtime_hours",
        "gof_bootstrap_reps",
        "s1_q0",
        "s1_xi",
        "s2_t_df",
        "s3_q0",
        "s3_xi",
        "s3_armax_a",
        "s4_garch_a1",
        "s4_garch_b1",
        "s4_t_df",
        "s4_burn_in",
        "s4_reference_size",
        "trigger_false_selection",
        "trigger_fdr_multiple",
        "trigger_rule_failure",
    }
    s = _section(raw, "simstudy", keys)
    n = _int(s["sample_size"], "simstudy.sample_size", 1)
    refit = _int(s["window_refit"], "simstudy.window_refit", 1)
    reps = _int(s["replications"], "simstudy.replications", 1)
    reps_fb = _int(s["replications_fallback"], "simstudy.replications_fallback", 1)
    if refit > n:
        raise ConfigError("simstudy.window_refit must not exceed simstudy.sample_size")
    if reps_fb > reps:
        raise ConfigError("simstudy.replications_fallback must not exceed simstudy.replications")
    a1 = parse_fraction(s["s4_garch_a1"], "simstudy.s4_garch_a1")
    b1 = parse_fraction(s["s4_garch_b1"], "simstudy.s4_garch_b1")
    if a1 < 0 or b1 < 0 or a1 + b1 >= 1:
        raise ConfigError("simstudy: GARCH(1,1) needs a1, b1 >= 0 and a1 + b1 < 1 (unit variance)")
    t_df4 = _int(s["s4_t_df"], "simstudy.s4_t_df", 3)  # standardised t needs df > 2

    def u01(key: str) -> Fraction:
        return _open_unit(parse_fraction(s[key], f"simstudy.{key}"), f"simstudy.{key}")

    def u01s(key: str) -> tuple[Fraction, ...]:
        return tuple(
            _open_unit(x, f"simstudy.{key}") for x in _fractions(s[key], f"simstudy.{key}")
        )

    return SimStudyConfig(
        sample_size=n,
        window_refit=refit,
        replications=reps,
        replications_fallback=reps_fb,
        max_runtime_hours=_int(s["max_runtime_hours"], "simstudy.max_runtime_hours", 1),
        gof_bootstrap_reps=_int(s["gof_bootstrap_reps"], "simstudy.gof_bootstrap_reps", 1),
        s1_q0=u01s("s1_q0"),
        s1_xi=u01s("s1_xi"),
        s2_t_df=_int(s["s2_t_df"], "simstudy.s2_t_df", 1),
        s3_q0=u01("s3_q0"),
        s3_xi=u01("s3_xi"),
        s3_armax_a=u01s("s3_armax_a"),
        s4_garch_a1=a1,
        s4_garch_b1=b1,
        s4_t_df=t_df4,
        s4_burn_in=_int(s["s4_burn_in"], "simstudy.s4_burn_in", 0),
        s4_reference_size=_int(s["s4_reference_size"], "simstudy.s4_reference_size", 1),
        trigger_false_selection=u01("trigger_false_selection"),
        trigger_fdr_multiple=_positive(
            parse_fraction(s["trigger_fdr_multiple"], "simstudy.trigger_fdr_multiple"),
            "simstudy.trigger_fdr_multiple",
        ),
        trigger_rule_failure=u01("trigger_rule_failure"),
    )


def _parse_dependence(raw: dict[str, Any], threshold: ThresholdConfig) -> DependenceConfig:
    s = _section(
        raw,
        "dependence",
        {
            "acf_max_lag",
            "ljung_box_lags",
            "arch_lm_lags",
            "exceedance_levels",
            "acf_band_z",
            "output_dir",
        },
    )
    max_lag = _int(s["acf_max_lag"], "dependence.acf_max_lag", 1)
    lb = _ints(s["ljung_box_lags"], "dependence.ljung_box_lags", 1)
    arch = _ints(s["arch_lm_lags"], "dependence.arch_lm_lags", 1)
    _strictly_increasing(lb, "dependence.ljung_box_lags")
    _strictly_increasing(arch, "dependence.arch_lm_lags")
    if max(lb) > max_lag:
        raise ConfigError("dependence.acf_max_lag must be >= the largest ljung_box_lags entry")
    levels = tuple(
        _open_unit(q, "dependence.exceedance_levels")
        for q in _fractions(s["exceedance_levels"], "dependence.exceedance_levels")
    )
    _strictly_increasing(levels, "dependence.exceedance_levels")
    if not set(levels) <= set(threshold.grid):
        raise ConfigError("dependence.exceedance_levels must be threshold.grid candidates")
    return DependenceConfig(
        acf_max_lag=max_lag,
        ljung_box_lags=lb,
        arch_lm_lags=arch,
        exceedance_levels=levels,
        acf_band_z=_positive(
            parse_fraction(s["acf_band_z"], "dependence.acf_band_z"), "dependence.acf_band_z"
        ),
        output_dir=_rel_path(s["output_dir"], "dependence.output_dir"),
    )


def _parse_gpd(raw: dict[str, Any]) -> GpdConfig:
    s = _section(
        raw,
        "gpd",
        {
            "optimizer",
            "xatol",
            "fatol",
            "maxiter",
            "maxfev",
            "ci_level",
            "profile_xtol",
            "profile_xi_max",
            "xi_boundary_tol",
            "delta_step_xi",
            "delta_step_beta_rel",
        },
    )
    opt = _str(s["optimizer"], "gpd.optimizer")
    if opt != "Nelder-Mead":
        raise ConfigError(f"gpd.optimizer: only 'Nelder-Mead' is specified (D30), got {opt!r}")

    def pos(key: str) -> Fraction:
        return _positive(parse_fraction(s[key], f"gpd.{key}"), f"gpd.{key}")

    def unit(key: str) -> Fraction:
        return _open_unit(parse_fraction(s[key], f"gpd.{key}"), f"gpd.{key}")

    return GpdConfig(
        optimizer=opt,
        xatol=pos("xatol"),
        fatol=pos("fatol"),
        maxiter=_int(s["maxiter"], "gpd.maxiter", 1),
        maxfev=_int(s["maxfev"], "gpd.maxfev", 1),
        ci_level=unit("ci_level"),
        profile_xtol=pos("profile_xtol"),
        profile_xi_max=pos("profile_xi_max"),
        xi_boundary_tol=unit("xi_boundary_tol"),
        delta_step_xi=pos("delta_step_xi"),
        delta_step_beta_rel=pos("delta_step_beta_rel"),
    )


def _parse_garch(raw: dict[str, Any]) -> GarchConfig:
    s = _section(raw, "garch", {"xatol", "fatol", "maxiter", "maxfev", "simplex_step"})

    def pos(key: str) -> Fraction:
        return _positive(parse_fraction(s[key], f"garch.{key}"), f"garch.{key}")

    return GarchConfig(
        xatol=pos("xatol"),
        fatol=pos("fatol"),
        maxiter=_int(s["maxiter"], "garch.maxiter", 1),
        maxfev=_int(s["maxfev"], "garch.maxfev", 1),
        simplex_step=pos("simplex_step"),
    )


def _parse_conditional(
    raw: dict[str, Any], threshold: ThresholdConfig, design: DesignConfig
) -> ConditionalConfig:
    s = _section(raw, "conditional", {"q_c", "filter"})
    q_c = parse_fraction(s["q_c"], "conditional.q_c")
    if q_c not in threshold.grid:
        raise ConfigError(f"conditional.q_c: {q_c} is not a grid candidate")
    if threshold.n_min is not None and (1 - q_c) * design.window < threshold.n_min:
        raise ConfigError(
            "conditional.q_c violates q <= q_max = 1 - n_min / window "
            f"((1 - {q_c}) x {design.window} < {threshold.n_min})"
        )
    spec = _str(s["filter"], "conditional.filter")
    if spec not in ("F1", "F2", "F3", "F4"):  # garch.SPECS, D39 (c)
        raise ConfigError(f"conditional.filter: must be one of F1-F4, got {spec!r}")
    return ConditionalConfig(q_c=q_c, filter=spec)


def _parse_backtest(raw: dict[str, Any], levels: LevelsConfig) -> BacktestConfig:
    keys = {
        "kupiec_level",
        "christoffersen_mc_reps",
        "es_residual_bootstrap_reps",
        "es_residual_min_breaches",
        "traffic_light_level",
        "traffic_light_block",
        "traffic_light_green_max",
        "traffic_light_yellow_max",
        "dm_lag_rule",
        "dm_lag_sensitivity_multiplier",
        "common_mask_min_share",
    }
    s = _section(raw, "backtest", keys)
    tl_level = parse_fraction(s["traffic_light_level"], "backtest.traffic_light_level")
    if tl_level not in levels.primary:
        raise ConfigError("backtest.traffic_light_level must be one of levels.primary")
    green = _int(s["traffic_light_green_max"], "backtest.traffic_light_green_max", 0)
    yellow = _int(s["traffic_light_yellow_max"], "backtest.traffic_light_yellow_max", 0)
    if not green < yellow:
        raise ConfigError("backtest: traffic_light_green_max must be < traffic_light_yellow_max")
    rule = _str(s["dm_lag_rule"], "backtest.dm_lag_rule")
    if rule != "newey_west_4_T100_2_9":
        raise ConfigError(
            f"backtest.dm_lag_rule: only 'newey_west_4_T100_2_9' is specified (D15), got {rule!r}"
        )
    share = parse_fraction(s["common_mask_min_share"], "backtest.common_mask_min_share")
    if not 0 < share <= 1:
        raise ConfigError(f"backtest.common_mask_min_share: must lie in (0, 1], got {share}")
    return BacktestConfig(
        kupiec_level=_open_unit(
            parse_fraction(s["kupiec_level"], "backtest.kupiec_level"), "backtest.kupiec_level"
        ),
        christoffersen_mc_reps=_int(
            s["christoffersen_mc_reps"], "backtest.christoffersen_mc_reps", 1
        ),
        es_residual_bootstrap_reps=_int(
            s["es_residual_bootstrap_reps"], "backtest.es_residual_bootstrap_reps", 1
        ),
        es_residual_min_breaches=_int(
            s["es_residual_min_breaches"], "backtest.es_residual_min_breaches", 1
        ),
        traffic_light_level=tl_level,
        traffic_light_block=_int(s["traffic_light_block"], "backtest.traffic_light_block", 1),
        traffic_light_green_max=green,
        traffic_light_yellow_max=yellow,
        dm_lag_rule=rule,
        dm_lag_sensitivity_multiplier=_int(
            s["dm_lag_sensitivity_multiplier"], "backtest.dm_lag_sensitivity_multiplier", 1
        ),
        common_mask_min_share=share,
    )


def _parse_bootstrap(raw: dict[str, Any]) -> BootstrapConfig:
    s = _section(
        raw, "bootstrap", {"block_rule", "replications", "sensitivity_multipliers", "circular"}
    )
    rule = _str(s["block_rule"], "bootstrap.block_rule")
    if rule != "ceil_cube_root":
        raise ConfigError(
            f"bootstrap.block_rule: only 'ceil_cube_root' is specified (D16), got {rule!r}"
        )
    mults = tuple(
        _positive(m, "bootstrap.sensitivity_multipliers")
        for m in _fractions(s["sensitivity_multipliers"], "bootstrap.sensitivity_multipliers")
    )
    return BootstrapConfig(
        block_rule=rule,
        replications=_int(s["replications"], "bootstrap.replications", 1),
        sensitivity_multipliers=mults,
        circular=_bool(s["circular"], "bootstrap.circular"),
    )


def _parse_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    s = _section(raw, "runtime", {"seed", "blas_threads"})
    threads = _int(s["blas_threads"], "runtime.blas_threads", 1)
    if threads != 1:
        raise ConfigError(f"runtime.blas_threads: A5e pins BLAS/OpenMP threads to 1, got {threads}")
    return RuntimeConfig(seed=_int(s["seed"], "runtime.seed", 0), blas_threads=threads)


_TOP_LEVEL = {
    "spec_version",
    "data",
    "validation",
    "design",
    "levels",
    "threshold",
    "nmin_study",
    "simstudy",
    "dependence",
    "gpd",
    "garch",
    "conditional",
    "backtest",
    "bootstrap",
    "runtime",
}


def parse_config(raw: dict[str, Any]) -> Config:
    """Validate an already-parsed TOML mapping and build a :class:`Config`.

    Raises:
        ConfigError: on any unknown, missing, mistyped or out-of-range value.
    """
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ConfigError(f"unknown top-level key(s) {sorted(unknown)}")
    if raw.get("spec_version") != SPEC_VERSION:
        raise ConfigError(
            f"spec_version: expected {SPEC_VERSION!r}, got {raw.get('spec_version')!r}"
        )
    design = _parse_design(raw)
    levels = _parse_levels(raw)
    threshold = _parse_threshold(raw, design)
    nmin_study = _parse_nmin_study(raw)
    if threshold.n_min is not None and threshold.n_min not in nmin_study.n_u:
        raise ConfigError("threshold.n_min must be one of nmin_study.n_u (D32)")
    return Config(
        spec_version=SPEC_VERSION,
        data=_parse_data(raw),
        validation=_parse_validation(raw),
        design=design,
        levels=levels,
        threshold=threshold,
        nmin_study=nmin_study,
        simstudy=_parse_simstudy(raw),
        dependence=_parse_dependence(raw, threshold),
        gpd=_parse_gpd(raw),
        garch=_parse_garch(raw),
        conditional=_parse_conditional(raw, threshold, design),
        backtest=_parse_backtest(raw, levels),
        bootstrap=_parse_bootstrap(raw),
        runtime=_parse_runtime(raw),
    )


def load_config(path: str | Path) -> Config:
    """Read and validate a NIFTYTAIL TOML configuration file.

    Raises:
        ConfigError: if the file cannot be read, is not valid TOML, or fails validation.
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {p}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {p}: {exc}") from exc
    return parse_config(raw)


def config_sha256(path: str | Path) -> str:
    """SHA-256 of the configuration file's bytes (recorded in run manifests, A5e)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
