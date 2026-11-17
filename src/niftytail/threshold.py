"""Threshold-selection procedure (D34; plan 3 Steps 1-5; methodology 7). Training data only.

Steps on one sample of losses (``candidate_table`` -> ``forward_stop`` -> ``stable_region``):

1. Grid ``q_k`` (exact Fractions, ascending), ``u_k = tail_threshold(losses, q_k)`` (D29),
   ``N_u`` = the actual count of losses strictly above ``u_k``.
2. Per candidate: our ``fit_mle`` (estimates, SE, CIs, VaR/ES) and the Anderson-Darling
   parametric-bootstrap p-value from ``scipy.stats.goodness_of_fit`` (loc fixed at 0; the
   AD statistic and the null refits use SCIPY's optimiser, not ``fit_mle``).
3. ForwardStop: ``k_hat = max{k : -(1/k) sum_{i<=k} ln(1 - p_i) <= alpha}`` (the maximum,
   not the first crossing); candidates ``1..k_hat`` are rejected, ``q_FS = q_{k_hat+1}``.
4. Stable region: the smallest ``q >= q_FS`` with ``q <= q_max`` whose window of grid
   points ``[q, q + width]`` passes every tolerance relative to ``q``. Window members above
   ``q_max`` are allowed; only ``q*`` itself is bounded by ``q_max``.
5. Split-half: Steps 1-4 on each half; a flag if a half has no ``q*`` or differs from the
   full-sample ``q*`` by more than ``split_half_flag``.

Nothing is ever chosen automatically after a rule failure, and no value is clipped or
repaired: unavailable quantities are ``None`` with a flag. Error messages never contain data
values (D23). This module reads no files and does not import the data layer.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from niftytail.config import Config, GpdConfig, ThresholdConfig
from niftytail.gpd import (
    EstimationError,
    GpdFit,
    Quantity,
    delta_ci,
    excesses,
    fit_mle,
    profile_ci_xi,
    se_xi,
    tail_risk,
)
from niftytail.losses import tail_threshold

__all__ = [
    "Steps",
    "ThresholdDecision",
    "candidate_table",
    "forward_stop",
    "gof_ad_bootstrap",
    "select_threshold",
    "split_half",
    "split_half_flag",
    "stable_region",
    "window_convergence_share",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
RuleFailure = Literal["all_rejected", "no_stable_region"]
Row = dict[str, Any]

LEVELS = (("95", Fraction(95, 100)), ("99", Fraction(99, 100)))  # confidence levels c (A0)
XI_GAP = 0.05  # D34 (3): |xi_hat_scipy_gof - xi_hat| above this is flagged (diagnostic only)
MIN_EXCESSES = 2  # structural minimum of fit_mle (as D32); N_min acts through q_max


@dataclass(frozen=True, slots=True)
class Steps:
    """Result of Steps 1-4 on one sample; ``candidates`` holds one row per grid point."""

    q_star: Fraction | None
    q_fs: Fraction | None
    k_hat: int
    rule_failure: RuleFailure | None
    candidates: tuple[Row, ...]


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """Full-sample Steps 1-4, the two split-half reruns (``None`` if skipped) and the flags."""

    q_star: Fraction | None
    q_fs: Fraction | None
    k_hat: int
    rule_failure: RuleFailure | None
    candidates: tuple[Row, ...]
    halves: tuple[Steps, Steps] | None
    flags: tuple[str, ...]

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, indent 2, LF, trailing newline, no timestamps.

        Floats are written with ``repr``, Fractions as strings (``"22/25"``) and non-finite
        floats (only the ForwardStop statistic can be ``inf``) as the string ``"inf"``.
        """
        return (
            json.dumps(
                _jsonable(dataclasses.asdict(self)), sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )


def _jsonable(o: Any) -> Any:
    if isinstance(o, Fraction):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    return o


def _children(ss: np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    """``ss.spawn(n)`` of a fresh ``ss``, without advancing ``ss``.

    ``SeedSequence.spawn`` is stateful (a second call returns different children); building
    the children from the spawn key keeps the output independent of call order.
    """
    return [
        np.random.SeedSequence(ss.entropy, spawn_key=(*ss.spawn_key, i), pool_size=ss.pool_size)
        for i in range(n)
    ]


def gof_ad_bootstrap(
    y: npt.ArrayLike, *, reps: int, ss: np.random.SeedSequence
) -> tuple[float, float, float]:
    """``(AD statistic, p-value, scipy's xi_hat)`` of the GPD fit to the excesses ``y``.

    ``scipy.stats.goodness_of_fit(genpareto, y, known_params={"loc": 0}, statistic="ad",
    n_mc_samples=reps, rng=Generator(PCG64(ss)))``: the data and every Monte Carlo sample
    are refitted by ``genpareto.fit(..., floc=0)``; ``p = (1 + #{AD* >= AD}) / (reps + 1)``.
    Whatever scipy raises propagates (the caller records ``fit_failed``).
    """
    res = stats.goodness_of_fit(
        stats.genpareto,
        np.asarray(y, dtype=np.float64),
        known_params={"loc": 0},
        statistic="ad",
        n_mc_samples=reps,
        rng=np.random.Generator(np.random.PCG64(ss)),
    )
    return float(res.statistic), float(res.pvalue), float(res.fit_result.params.c)


def _delta(
    fit: GpdFit, u: float, n: int, n_u: int, p: Fraction, quantity: Quantity, cfg: Config
) -> list[float] | None:
    try:
        ci = delta_ci(
            fit,
            u,
            n,
            n_u,
            p,
            quantity,
            xi_zero_tol=float(cfg.threshold.xi_zero_tol),
            settings=cfg.gpd,
        )
    except EstimationError:
        return None
    return [ci.lower, ci.upper]


def _candidate(
    x: FloatArray,
    q: Fraction,
    cfg: Config,
    ss: np.random.SeedSequence,
    gof_reps: int,
    intervals: bool,
) -> Row:
    th = cfg.threshold
    tol = float(th.xi_zero_tol)
    n = int(x.size)
    u = tail_threshold(x, q)
    y = excesses(x, u)
    n_u = int(y.size)
    n_u_window = (1 - q) * cfg.design.window
    flags: list[str] = []
    risk_keys = [f"{m}{label}" for label, _ in LEVELS for m in ("var", "es")]
    row: Row = {
        "q": q,
        "u": u,
        "n_u": n_u,
        "n_u_window": n_u_window,
        "xi_hat": None,
        "beta_hat": None,
        "beta_star": None,
        "se_xi": None,
        "xi_ci": None,
        "xi_ci_lower_open": None,
        "ad_stat": None,
        "ad_p": 1.0,
        "xi_hat_scipy_gof": None,
        "fit_error": None,
        "convergence_share": None,
        "flags": flags,
    }
    row.update({k + s: None for k in risk_keys for s in ("", "_ci")})
    if th.n_min is not None and n_u_window < th.n_min:
        flags.append("below_n_min_window")
    try:
        fit = fit_mle(y, min_excesses=MIN_EXCESSES, xi_zero_tol=tol, settings=cfg.gpd)
        ad, p_ad, xi_scipy = gof_ad_bootstrap(y, reps=gof_reps, ss=ss)
    except Exception as exc:  # D34 (3): any failure of our fit or of scipy's call
        row["fit_error"] = type(exc).__name__
        flags.append("fit_failed")
        return row
    row.update(
        xi_hat=fit.xi,
        beta_hat=fit.beta,
        beta_star=fit.beta - fit.xi * u,
        se_xi=se_xi(fit.xi, n_u) if fit.regular else None,
        ad_stat=ad,
        ad_p=p_ad,
        xi_hat_scipy_gof=xi_scipy,
    )
    if intervals:
        try:
            profile = profile_ci_xi(y, fit, xi_zero_tol=tol, settings=cfg.gpd)
        except EstimationError:
            pass
        else:
            row.update(xi_ci=[profile.lower, profile.upper], xi_ci_lower_open=profile.lower_open)
    if abs(xi_scipy - fit.xi) > XI_GAP:
        flags.append("xi_gap")
    if fit.xi >= 1:
        flags.append("es_infinite")
    for label, c in LEVELS:
        p = 1 - c
        if not p < Fraction(n_u, n):
            flags.append(f"var{label}_not_in_tail")
            continue
        var, es = tail_risk(u, fit.xi, fit.beta, n, n_u, p, tol)
        row[f"var{label}"] = var
        if intervals:
            row[f"var{label}_ci"] = _delta(fit, u, n, n_u, p, "var", cfg)
        if math.isfinite(es):
            row[f"es{label}"] = es
            if intervals:
                row[f"es{label}_ci"] = _delta(fit, u, n, n_u, p, "es", cfg)
    if intervals and (
        row["se_xi"] is None
        or row["xi_ci"] is None
        or any(row[k] is not None and row[k + "_ci"] is None for k in risk_keys)
    ):
        flags.append("ci_unavailable")
    return row


def candidate_table(
    losses: npt.ArrayLike,
    cfg: Config,
    *,
    ss: np.random.SeedSequence,
    gof_reps: int,
    intervals: bool = True,
) -> list[Row]:
    """Steps 1-2: one row per grid candidate (D34 (1)-(3)); candidate ``k`` uses GoF seed
    child ``k - 1`` of ``ss``.

    Row keys: ``q``, ``u``, ``n_u`` (actual count of losses > u), ``n_u_window`` (exact
    ``(1 - q) W``), ``xi_hat``, ``beta_hat``, ``beta_star``, ``se_xi``, ``xi_ci`` (profile),
    ``xi_ci_lower_open``, ``ad_stat``, ``ad_p``, ``xi_hat_scipy_gof``, ``var95``/``es95``/
    ``var99``/``es99`` with ``*_ci`` (delta method), ``fit_error``, ``convergence_share``
    (filled by :func:`select_threshold`) and ``flags``:

    * ``fit_failed``: ``fit_mle`` or scipy's call raised; ``ad_p`` is 1.0, estimates ``None``
      and ``fit_error`` holds the exception type name (never its message);
    * ``var95_not_in_tail`` (any level): ``p >= n_u / n``, VaR and ES are ``None``;
    * ``es_infinite``: ``xi_hat >= 1``, ES is ``None``;
    * ``xi_gap``: scipy's and our xi differ by more than 0.05 (diagnostic only);
    * ``ci_unavailable``: an SE or CI is undefined (``xi_hat <= -1/2``, no profile crossing);
    * ``below_n_min_window``: ``(1 - q) W < n_min`` (the candidate lies above ``q_max``).

    ``intervals=False`` (D35 (7)) skips the profile and delta-method CIs: the ``*_ci`` fields
    stay ``None`` and ``ci_unavailable`` is not added. Nothing else changes.
    """
    x = np.asarray(losses, dtype=np.float64)
    grid = cfg.threshold.grid
    return [
        _candidate(x, q, cfg, child, gof_reps, intervals)
        for q, child in zip(grid, _children(ss, len(grid)), strict=True)
    ]


def forward_stop(pvalues: Sequence[float], alpha: Fraction) -> tuple[list[float], int]:
    """ForwardStop running statistics and ``k_hat`` (D34 (5); G'Sell et al. 2016, eq. 6).

    ``stat_k = -(1/k) sum_{i<=k} ln(1 - p_i)``, each term ``-log1p(-p_i)`` and ``+inf`` at
    ``p_i == 1``. ``k_hat = max{k : stat_k <= alpha}`` (the maximum, exact comparison of
    the float against the Fraction), 0 if no ``k`` qualifies.

    Raises:
        ValueError: unless every p-value lies in [0, 1].
    """
    out: list[float] = []
    total, k_hat = 0.0, 0
    for k, p in enumerate(pvalues, start=1):
        if not 0 <= p <= 1:
            raise ValueError("pvalues: must lie in [0, 1]")
        total += math.inf if p == 1 else -math.log1p(-p)
        out.append(total / k)
        if out[-1] <= alpha:
            k_hat = k
    return out, k_hat


def _usable(x: object) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _window_fail(
    rows: Sequence[Mapping[str, Any]], i: int, k_hat: int, q_max: Fraction, th: ThresholdConfig
) -> str | None:
    base = rows[i]
    if i < k_hat:
        return "rejected"
    if base["q"] > q_max:
        return "above_q_max"
    if i + th.stable_region_points > len(rows):
        return "no_window"
    for r in rows[i : i + th.stable_region_points]:
        at = f"@{r['q']}"
        if "fit_failed" in r["flags"]:
            return "fit_failed" + at
        if "es_infinite" in r["flags"]:
            return "es_infinite" + at
        if not (_usable(r["var99"]) and _usable(r["es99"])):
            return "not_finite" + at
        if base["var99"] == 0 or base["es99"] == 0:
            return "not_finite" + at  # relative difference undefined
        if abs(r["xi_hat"] - base["xi_hat"]) > th.tol_xi:
            return "tol_xi" + at
        if abs(r["var99"] / base["var99"] - 1) > th.tol_var99_rel:
            return "tol_var99" + at
        if abs(r["es99"] / base["es99"] - 1) > th.tol_es99_rel:
            return "tol_es99" + at
        if r["rejected"]:  # redundant once q >= q_FS (rejections are a prefix); kept, D34 (6)
            return "rejected" + at
    return None


def stable_region(
    rows: Sequence[Mapping[str, Any]], k_hat: int, *, q_max: Fraction, th: ThresholdConfig
) -> tuple[Fraction | None, RuleFailure | None, list[str | None]]:
    """Step 4 (D34 (6)): ``(q*, rule failure, first failing condition per candidate)``.

    ``rows`` need ``q``, ``xi_hat``, ``var99``, ``es99``, ``flags`` and ``rejected``.
    A candidate ``q_i`` qualifies iff ``i >= k_hat`` (``q_i >= q_FS``), ``q_i <= q_max``, its
    window of ``stable_region_points`` grid points exists and every member ``q'`` (members
    above ``q_max`` included) has a converged fit with finite VaR99 and ES99 and satisfies,
    relative to ``q_i`` and inclusively, ``|xi' - xi| <= tol_xi``,
    ``|VaR99'/VaR99 - 1| <= tol_var99_rel``, ``|ES99'/ES99 - 1| <= tol_es99_rel`` (floats
    compared exactly against the Fractions) and is not rejected by ForwardStop. ``q*`` is
    the first qualifying candidate; the third element gives, per candidate, ``None`` or the
    first failing condition (``"tol_xi@22/25"`` names the failing member).

    ``k_hat == len(rows)`` is the rule failure ``"all_rejected"``; no qualifying candidate
    is ``"no_stable_region"``. Nothing is chosen in either case.
    """
    if k_hat >= len(rows):
        return None, "all_rejected", ["rejected"] * len(rows)
    fails = [_window_fail(rows, i, k_hat, q_max, th) for i in range(len(rows))]
    for r, f in zip(rows, fails, strict=True):
        if f is None:
            q_star: Fraction = r["q"]
            return q_star, None, fails
    return None, "no_stable_region", fails


def _steps(
    losses: npt.ArrayLike,
    cfg: Config,
    *,
    ss: np.random.SeedSequence,
    gof_reps: int,
    intervals: bool = True,
) -> Steps:
    th = cfg.threshold
    if th.n_min is None:
        raise ValueError("threshold.n_min is not set (Day-6 gate)")
    q_max = 1 - Fraction(th.n_min, cfg.design.window)
    rows = candidate_table(losses, cfg, ss=ss, gof_reps=gof_reps, intervals=intervals)
    fs, k_hat = forward_stop([r["ad_p"] for r in rows], th.forwardstop_alpha)
    for k, (r, s) in enumerate(zip(rows, fs, strict=True), start=1):
        r["fs_stat"] = s
        r["rejected"] = k <= k_hat
    q_star, failure, fails = stable_region(rows, k_hat, q_max=q_max, th=th)
    for r, f in zip(rows, fails, strict=True):
        r["stable_fail"] = f
    return Steps(
        q_star=q_star,
        q_fs=th.grid[k_hat] if k_hat < len(rows) else None,
        k_hat=k_hat,
        rule_failure=failure,
        candidates=tuple(rows),
    )


def split_half(
    losses: npt.ArrayLike,
    cfg: Config,
    *,
    seeds: tuple[np.random.SeedSequence, np.random.SeedSequence],
    gof_reps: int,
    intervals: bool = True,
) -> tuple[Steps, Steps]:
    """Step 5 (D34 (7)): Steps 1-4 on ``losses[:n // 2]`` and ``losses[n // 2:]``.

    Each half uses its own quantiles, GoF seeds and ForwardStop, with the same grid, alpha
    and ``q_max`` as the full sample.
    """
    x = np.asarray(losses, dtype=np.float64)
    h = x.size // 2
    return (
        _steps(x[:h], cfg, ss=seeds[0], gof_reps=gof_reps, intervals=intervals),
        _steps(x[h:], cfg, ss=seeds[1], gof_reps=gof_reps, intervals=intervals),
    )


def split_half_flag(
    q_star: Fraction | None, q_halves: Sequence[Fraction | None], limit: Fraction
) -> bool:
    """True iff ``q*`` is defined and a half has no ``q*`` or ``|q_half - q*| > limit``.

    Exact Fractions, strictly greater: a difference equal to ``limit`` is not flagged.
    """
    if q_star is None:
        return False
    return any(qh is None or abs(qh - q_star) > limit for qh in q_halves)


def window_convergence_share(
    losses: npt.ArrayLike, q: Fraction, window: int, *, xi_zero_tol: float, settings: GpdConfig
) -> float:
    """Share of the ``n - window + 1`` rolling windows in which ``fit_mle`` converges at ``q``.

    The threshold is recomputed in every window (D5). A window whose fit raises (or that
    has no excess) counts as not converged.

    Raises:
        ValueError: if ``losses`` is shorter than ``window`` or ``window < 1``.
    """
    x = np.asarray(losses, dtype=np.float64)
    if window < 1 or x.ndim != 1 or x.size < window:
        raise ValueError("losses: need a 1-D series of at least `window` values")
    total = x.size - window + 1
    converged = 0
    for start in range(total):
        w = x[start : start + window]
        try:
            fit_mle(
                excesses(w, tail_threshold(w, q)),
                min_excesses=MIN_EXCESSES,
                xi_zero_tol=xi_zero_tol,
                settings=settings,
            )
        except (EstimationError, ValueError):
            continue
        converged += 1
    return converged / total


def select_threshold(
    train_losses: npt.ArrayLike,
    cfg: Config,
    *,
    ss: np.random.SeedSequence,
    gof_reps: int,
    convergence_share: bool = True,
    halves: bool = True,
    intervals: bool = True,
) -> ThresholdDecision:
    """Steps 1-5 on the training losses (D34 (8)).

    Seeds: with ``c = ss.spawn(3)`` of a fresh ``ss`` (``ss`` itself is not advanced), the
    full sample uses ``c[0]`` and the halves ``c[1]`` and ``c[2]``; candidate ``k`` uses
    child ``k - 1`` of the respective seed. The output is a pure function of the
    arguments. ``convergence_share=False`` skips the rolling-window convergence column
    (about 16,000 fits), which changes nothing else. D35 (7): ``halves=False`` skips Step 5
    (``halves`` is ``None`` and no split-half flag is evaluated); ``intervals=False`` skips
    the profile and delta-method CIs. Neither switch changes any rule-relevant output.

    Raises:
        ValueError: unless ``train_losses`` is 1-D with exactly ``design.train_size``
            finite values (a guard against out-of-sample leakage), or if ``n_min`` is unset.
    """
    x = np.asarray(train_losses, dtype=np.float64)
    if x.ndim != 1 or x.size != cfg.design.train_size:
        raise ValueError(
            f"train_losses: need exactly design.train_size = {cfg.design.train_size} values"
        )
    full_seed, seed1, seed2 = _children(ss, 3)
    full = _steps(x, cfg, ss=full_seed, gof_reps=gof_reps, intervals=intervals)
    if convergence_share:
        for r in full.candidates:
            r["convergence_share"] = window_convergence_share(
                x,
                r["q"],
                cfg.design.window,
                xi_zero_tol=float(cfg.threshold.xi_zero_tol),
                settings=cfg.gpd,
            )
    both: tuple[Steps, Steps] | None = None
    flagged = False
    if halves:
        both = split_half(x, cfg, seeds=(seed1, seed2), gof_reps=gof_reps, intervals=intervals)
        flagged = split_half_flag(
            full.q_star, [h.q_star for h in both], cfg.threshold.split_half_flag
        )
    return ThresholdDecision(
        q_star=full.q_star,
        q_fs=full.q_fs,
        k_hat=full.k_hat,
        rule_failure=full.rule_failure,
        candidates=full.candidates,
        halves=both,
        flags=("split_half_flag",) if flagged else (),
    )
