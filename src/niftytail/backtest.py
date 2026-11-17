"""VaR/ES backtests, FZ0 scoring and Diebold-Mariano comparisons (D42; methodology §10-§13, §18).

Input: one pass's D41 table (index ``(j, date)``; columns ``loss``, ``<key>:var:<c>``,
``<key>:es:<c>``, ``<key>:flags``). Row j holds the forecast for target j and its realised
loss L[j]; rows are never shifted. Hit_j = 1{L[j] > VaR_j} (strict). FZ0 is computed in
return space only, through :func:`to_return_space` (the only sign change).

D42 (a): per primary level, the common mask C is the AND of the admissibility of the six
primary keys; a sensitivity key ``<name>@<q>`` is scored on C AND its own admissibility.
(b): three DM families (unconditional, unadjusted; conditional, Holm; descriptive).
(c): coverage tests per key on its own admissible dates. (e): every Monte Carlo or bootstrap
draw comes from its own seed key, so results do not depend on evaluation order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special, stats

from niftytail.config import BacktestConfig, Config
from niftytail.models import MODELS

__all__ = [
    "CONDITIONAL_PAIRS",
    "PER_DATE_JSON",
    "PRIMARY_KEYS",
    "UNCONDITIONAL_PAIRS",
    "admissibility",
    "binomial_equal_tailed",
    "block_ci_rate",
    "block_length",
    "christoffersen",
    "clopper_pearson",
    "common_mask",
    "diebold_mariano",
    "dm_lag",
    "es_residual_test",
    "evaluate",
    "fz0",
    "hits",
    "holm",
    "kupiec_critical",
    "kupiec_lr",
    "kupiec_power_exact",
    "kupiec_size_markov",
    "score_models",
    "seed_key",
    "to_json",
    "to_return_space",
    "traffic_light",
    "transition_counts",
    "variance_inflation",
    "write_per_date",
]

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

PRIMARY_KEYS: tuple[str, ...] = tuple(MODELS)  # D42 (a): the D41 primary keys, MODELS order
UNCONDITIONAL_PAIRS = (("gaussian", "hs"), ("gaussian", "pot_gpd"), ("hs", "pot_gpd"))
CONDITIONAL_PAIRS = (("garch_evt", "pot_gpd"), ("garch_evt", "garch_n"), ("garch_evt", "fhs"))
REASONS = ("missing", "es_infinite", "var_nonpositive", "es_below_var")
MISSING_FLAGS = ("garch_error", "no_pre_window_lag", "gpd_error", "var_not_in_tail")
TEST_LEVEL = 0.05  # D38 (5) Holm level; also the family (i) verdict level (D42 (f)(9))
CI_LEVEL = 0.95  # D42 (f)(10)
PER_DATE_JSON = "backtest_per_date.json"
TEST_IND, TEST_CC, TEST_ES, TEST_BLOCK = 1, 2, 3, 4  # D42 (e) test codes (block: 4, 5, 6)
_MC_CHUNK = 1000  # D42 (f)(4)


# ---------------------------------------------------------------------------
# Elementary pieces
# ---------------------------------------------------------------------------
def _dec(q: Fraction) -> str:
    return str(Decimal(q.numerator) / Decimal(q.denominator))


def _f(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def hits(loss: npt.ArrayLike, var: npt.ArrayLike) -> IntArray:
    """``1{L > VaR}`` element-wise (strict; a tie is not a hit)."""
    return (_f(loss) > _f(var)).astype(np.int64)


def to_return_space(
    loss: npt.ArrayLike, var: npt.ArrayLike, es: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Loss space to return space: ``(Y, v, e) = (-L, -VaR, -ES)``. The ONLY sign change."""
    return -_f(loss), -_f(var), -_f(es)


def fz0(y: npt.ArrayLike, v: npt.ArrayLike, e: npt.ArrayLike, alpha: float) -> FloatArray:
    """FZ0 score in return space (Patton, Ziegel & Chen 2019, eq. 6). Lower is better.

    Raises:
        ValueError: on a non-finite value, ``alpha`` outside (0, 1), ``v >= 0`` or ``e > v``.
            Nothing is repaired (``e == v`` is accepted).
    """
    yy, vv, ee = np.broadcast_arrays(_f(y), _f(v), _f(e))
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"fz0: alpha must lie in (0, 1), got {alpha!r}")
    if not all(np.all(np.isfinite(a)) for a in (yy, vv, ee)):
        raise ValueError("fz0: every Y, v and e must be finite (no NaN, no inf)")
    if np.any(vv >= 0.0):
        raise ValueError("fz0: requires v < 0")
    if np.any(ee > vv):
        raise ValueError("fz0: requires e <= v")
    tail = np.where(yy <= vv, vv - yy, 0.0)
    return _f(-tail / (alpha * ee) + vv / ee + np.log(-ee) - 1.0)


def _ll(n0: npt.ArrayLike, n1: npt.ArrayLike, p: npt.ArrayLike) -> FloatArray:
    """``n0 ln(1 - p) + n1 ln p`` with 0 ln 0 = 0."""
    return _f(special.xlog1py(n0, -_f(p)) + special.xlogy(n1, p))


# ---------------------------------------------------------------------------
# Kupiec, equal-tailed binomial, exact size and power (§10.1-§10.4)
# ---------------------------------------------------------------------------
def kupiec_critical(level: float) -> float:
    """The chi-square(1) quantile at ``1 - level`` (3.841459 at 5%)."""
    return float(stats.chi2.ppf(1.0 - level, 1))


def _check_rate(t: int, p0: float) -> None:
    if t < 1:
        raise ValueError("T must be >= 1")
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must lie in (0, 1), got {p0!r}")


def kupiec_lr(x: npt.ArrayLike, t: int, p0: float) -> FloatArray:
    """Kupiec POF likelihood ratio for ``x`` breaches in ``t`` days (0 ln 0 = 0)."""
    _check_rate(t, p0)
    xx = _f(x)
    if np.any((xx < 0) | (xx > t)):
        raise ValueError("x must lie in 0..T")
    return _f(-2.0 * (_ll(t - xx, xx, p0) - _ll(t - xx, xx, xx / t)))


def binomial_equal_tailed(x: npt.ArrayLike, t: int, p0: float) -> FloatArray:
    """D19: ``min(1, 2 min(P(X <= x), P(X >= x)))``, X ~ Bin(t, p0), via scipy cdf/sf."""
    _check_rate(t, p0)
    xx = _f(x)
    return _f(
        np.minimum(1.0, 2.0 * np.minimum(stats.binom.cdf(xx, t, p0), stats.binom.sf(xx - 1, t, p0)))
    )


def _rejection_set(t: int, p0: float, level: float) -> BoolArray:
    return np.asarray(kupiec_lr(np.arange(t + 1), t, p0) > kupiec_critical(level), dtype=np.bool_)


def kupiec_power_exact(t: int, p0: float, p_true: float, level: float) -> float:
    """P(Kupiec rejects at ``level``) for i.i.d. Bernoulli(``p_true``) hits, by enumeration."""
    x = np.arange(t + 1)
    return float(np.sum(stats.binom.pmf(x[_rejection_set(t, p0, level)], t, p_true)))


def kupiec_size_markov(t: int, p0: float, pi11: float, level: float) -> float:
    """True size of the nominal Kupiec test when hits are a two-state Markov chain with rate
    ``p0`` and persistence ``pi11`` (exact DP over (count, state); NaN if ``pi11`` is NaN)."""
    _check_rate(t, p0)
    if math.isnan(pi11):
        return math.nan
    p01 = p0 * (1.0 - pi11) / (1.0 - p0)
    if not (0.0 <= pi11 <= 1.0 and 0.0 <= p01 <= 1.0):
        raise ValueError("pi11: gives transition probabilities outside [0, 1]")
    d0, d1 = np.zeros(t + 1), np.zeros(t + 1)
    d0[0], d1[1] = 1.0 - p0, p0  # t = 1 from the stationary law
    for _ in range(t - 1):
        up = d0 * p01 + d1 * pi11
        d0 = d0 * (1.0 - p01) + d1 * (1.0 - pi11)
        d1 = np.concatenate(([0.0], up[:-1]))
    return float(np.sum((d0 + d1)[_rejection_set(t, p0, level)]))


def variance_inflation(p0: float, pi11: float) -> float:
    """``(1 + rho) / (1 - rho)`` with ``rho = pi11 - pi01`` (§10.4)."""
    rho = pi11 - p0 * (1.0 - pi11) / (1.0 - p0)
    return (1.0 + rho) / (1.0 - rho)


# ---------------------------------------------------------------------------
# Christoffersen (§10.5)
# ---------------------------------------------------------------------------
def _counts(h: BoolArray) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    a, b = h[..., :-1], h[..., 1:]
    n01 = np.sum(~a & b, axis=-1)
    n10 = np.sum(a & ~b, axis=-1)
    n11 = np.sum(a & b, axis=-1)
    n00 = (h.shape[-1] - 1) - n01 - n10 - n11
    return n00, n01, n10, n11


def transition_counts(h: npt.ArrayLike) -> tuple[int, int, int, int]:
    """``(n00, n01, n10, n11)`` over consecutive pairs of a 0/1 sequence."""
    return tuple(int(c) for c in _counts(np.asarray(h, dtype=np.bool_)))  # type: ignore[return-value]


def _ratio(k: IntArray, n: IntArray) -> FloatArray:
    kk, nn = _f(k), _f(n)
    return _f(np.divide(kk, nn, out=np.zeros_like(kk), where=nn > 0))


def _lr_ind(h: BoolArray) -> FloatArray:
    n00, n01, n10, n11 = _counts(h)
    l1 = _ll(n00, n01, _ratio(n01, n00 + n01)) + _ll(n10, n11, _ratio(n11, n10 + n11))
    l0 = _ll(n00 + n10, n01 + n11, _ratio(n01 + n11, n00 + n01 + n10 + n11))
    return _f(-2.0 * (l0 - l1))


def _lr_cc(h: BoolArray, p0: float) -> FloatArray:
    t = h.shape[-1]
    return _f(kupiec_lr(np.sum(h, axis=-1), t, p0) + _lr_ind(h))


def _mc_p(
    stat: Callable[[BoolArray], FloatArray],
    obs: float,
    p: float,
    t: int,
    reps: int,
    rng: np.random.Generator,
) -> float:
    count, done = 0, 0
    while done < reps:
        rows = min(_MC_CHUNK, reps - done)
        count += int(np.sum(stat(rng.random((rows, t)) < p) >= obs))
        done += rows
    return (1 + count) / (reps + 1)


def christoffersen(
    h: npt.ArrayLike,
    p0: float,
    reps: int,
    rng_ind: np.random.Generator,
    rng_cc: np.random.Generator,
) -> dict[str, Any]:
    """LR_ind and LR_cc with Monte Carlo p-values (IND null Bernoulli(p_hat), CC null
    Bernoulli(p0); p = (1 + #{LR* >= LR_obs}) / (R + 1))."""
    hb = np.asarray(h, dtype=np.bool_)
    t = hb.size
    _check_rate(t, p0)
    p_hat = float(hb.sum()) / t
    lr_ind = float(_lr_ind(hb))
    lr_cc = float(_lr_cc(hb, p0))
    n00, n01, n10, n11 = transition_counts(hb)
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_ind": lr_ind,
        "lr_cc": lr_cc,
        "p_ind": _mc_p(_lr_ind, lr_ind, p_hat, t, reps, rng_ind),
        "p_cc": _mc_p(lambda s: _lr_cc(s, p0), lr_cc, p0, t, reps, rng_cc),
        "reps": reps,
    }


# ---------------------------------------------------------------------------
# Traffic light (§10.6) and ES residual test (§10.7)
# ---------------------------------------------------------------------------
def _zone(breaches: int, bt: BacktestConfig) -> str:
    if breaches <= bt.traffic_light_green_max:
        return "green"
    return "yellow" if breaches <= bt.traffic_light_yellow_max else "red"


def traffic_light(
    h: npt.ArrayLike, admissible: npt.ArrayLike, bt: BacktestConfig
) -> list[dict[str, Any]]:
    """Blocks of ``traffic_light_block`` rows from row 0 (OOS day 1), identical for every key.

    Breaches count hits on admissible rows; a full block gets a zone (and the number of its
    inadmissible rows); the final partial block gets a breach count and no zone.
    """
    hh = np.asarray(h, dtype=np.int64)
    ok = np.asarray(admissible, dtype=np.bool_)
    size = bt.traffic_light_block
    out = []
    for i, start in enumerate(range(0, hh.size, size)):
        stop = min(start + size, hh.size)
        breaches = int(np.sum(hh[start:stop] * ok[start:stop]))
        full = stop - start == size
        out.append(
            {
                "block": i,
                "start_row": start,
                "end_row": stop - 1,
                "n_obs": stop - start,
                "breaches": breaches,
                "zone": _zone(breaches, bt) if full else None,
                "inadmissible_rows": int(np.sum(~ok[start:stop])),
            }
        )
    return out


def _zone_counts(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = [b for b in blocks if b["zone"] is not None]
    partial = [b for b in blocks if b["zone"] is None]
    return {
        "full_blocks": len(full),
        "zones": {z: sum(b["zone"] == z for b in full) for z in ("green", "yellow", "red")},
        "flagged_blocks": sum(b["inadmissible_rows"] > 0 for b in full),
        "partial_block": (
            {"n_obs": partial[0]["n_obs"], "breaches": partial[0]["breaches"]} if partial else None
        ),
    }


def es_residual_test(
    loss: npt.ArrayLike,
    es: npt.ArrayLike,
    reps: int,
    min_breaches: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One-sided test of E[z | breach] = 0, z = (L - ES)/ES on the breach days (A5a).

    Null-centred i.i.d. bootstrap; p = (1 + #{mean(z~*) >= z_bar}) / (B + 1). Fewer than
    ``min_breaches`` breaches: descriptive only, no p-value.
    """
    ll, ee = _f(loss), _f(es)
    z = (ll - ee) / ee
    n = z.size
    z_bar = float(np.mean(z)) if n else math.nan
    if n == 0 or n < min_breaches:
        return {"n_breach": n, "z_bar": z_bar, "p": math.nan, "descriptive": True}
    centred = z - z_bar
    means = np.mean(centred[rng.integers(0, n, size=(reps, n))], axis=1)
    p = (1 + int(np.sum(means >= z_bar))) / (reps + 1)
    return {"n_breach": n, "z_bar": z_bar, "p": p, "descriptive": False}


# ---------------------------------------------------------------------------
# Diebold-Mariano and Holm (§12, §18)
# ---------------------------------------------------------------------------
def dm_lag(t: int) -> int:
    """``floor(4 (T/100)^(2/9))`` exactly: the largest l with 10^4 l^9 <= 4^9 T^2."""
    lag = 0
    while 10**4 * (lag + 1) ** 9 <= 4**9 * t * t:
        lag += 1
    return lag


def diebold_mariano(d: npt.ArrayLike, lag: int) -> dict[str, Any]:
    """DM = d_bar / sqrt(Omega/T), Newey-West Bartlett Omega, no small-sample correction."""
    dd = _f(d)
    t = dd.size
    out: dict[str, Any] = {"n": t, "lag": lag, "d_bar": math.nan, "omega": math.nan}
    out.update(dm=math.nan, p=math.nan, flag=None)
    if t < 2:
        out["flag"] = "too_few_dates"
        return out
    u = dd - dd.mean()
    omega = float(u @ u) / t
    for j in range(1, min(lag, t - 1) + 1):
        omega += 2.0 * (1.0 - j / (lag + 1)) * float(u[j:] @ u[:-j]) / t
    out.update(d_bar=float(dd.mean()), omega=omega)
    if not omega > 0.0:
        out["flag"] = "omega_nonpositive"
        return out
    dm = out["d_bar"] / math.sqrt(omega / t)
    out.update(dm=dm, p=float(2.0 * stats.norm.sf(abs(dm))))
    return out


def holm(p: Sequence[float], level: float = TEST_LEVEL) -> tuple[list[float], list[bool]]:
    """Holm step-down: adjusted p-values and rejections (ties by input order; NaN last)."""
    m = len(p)
    order = sorted(range(m), key=lambda i: (math.isnan(p[i]), 0.0 if math.isnan(p[i]) else p[i], i))
    adj = [math.nan] * m
    running = 0.0
    for rank, i in enumerate(order):
        if not math.isnan(p[i]):
            running = max(running, min(1.0, (m - rank) * p[i]))
            adj[i] = running
    return adj, [not math.isnan(a) and a <= level for a in adj]


# ---------------------------------------------------------------------------
# Breach-rate intervals (§10.4, §13)
# ---------------------------------------------------------------------------
def block_length(n: int) -> int:
    """``ceil(n^(1/3))`` exactly: the smallest b with b^3 >= n."""
    b = 1
    while b**3 < n:
        b += 1
    return b


def clopper_pearson(x: int, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """Exact binomial CI for the breach rate."""
    a = 1.0 - level
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def block_ci_rate(
    h: npt.ArrayLike, b: int, reps: int, rng: np.random.Generator, level: float = CI_LEVEL
) -> tuple[float, float]:
    """Moving-block bootstrap percentile CI of the mean of ``h`` (no wrap; b capped at n)."""
    hh = np.asarray(h, dtype=np.int64)
    n = hh.size
    if n < 1 or b < 1:
        raise ValueError("block_ci_rate: needs n >= 1 and b >= 1")
    b = min(b, n)
    k = -(-n // b)
    r = n - (k - 1) * b
    c = np.concatenate(([0], np.cumsum(hh)))
    full = c[b:] - c[: n - b + 1]  # sum of h[s:s+b], s = 0..n-b
    last = c[r : r + n - b + 1] - c[: n - b + 1]  # sum of h[s:s+r]
    starts = rng.integers(0, n - b + 1, size=(reps, k))
    rates = (np.sum(full[starts[:, :-1]], axis=1) + last[starts[:, -1]]) / n
    a = 1.0 - level
    lo, hi = np.quantile(rates, [a / 2, 1 - a / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Admissibility, mask, scores
# ---------------------------------------------------------------------------
def admissibility(var: npt.ArrayLike, es: npt.ArrayLike, flags: Sequence[str]) -> list[str]:
    """One reason per row, ``""`` if admissible (D42 (f)(1) precedence and missing sub-reason)."""
    vv, ee = _f(var), _f(es)
    out = []
    for v, e, fl in zip(vv, ee, flags, strict=True):
        if math.isnan(v) or math.isnan(e):
            have = set(fl.split(";"))
            out.append("missing:" + next((f for f in MISSING_FLAGS if f in have), "other"))
        elif e == math.inf:
            out.append("es_infinite")
        elif v <= 0.0:
            out.append("var_nonpositive")
        elif e < v:
            out.append("es_below_var")
        else:
            out.append("")
    return out


def common_mask(adm: Mapping[str, BoolArray]) -> dict[str, BoolArray]:
    """D42 (a): C for the six primary keys; ``C & admissible(key)`` for a sensitivity key."""
    missing = [k for k in PRIMARY_KEYS if k not in adm]
    if missing:
        raise ValueError(f"common_mask: primary keys missing: {missing}")
    c = np.logical_and.reduce([adm[k] for k in PRIMARY_KEYS])
    return {k: c if k in PRIMARY_KEYS else c & adm[k] for k in adm}


def _col(table: pd.DataFrame, key: str, kind: str, c: Fraction) -> FloatArray:
    return _f(table[f"{key}:{kind}:{_dec(c)}"].to_numpy(dtype=np.float64))


def score_models(
    table: pd.DataFrame, keys: Sequence[str], c: Fraction, mask: BoolArray
) -> dict[str, FloatArray]:
    """FZ0 of each key on the dates of ``mask`` only (compressed), in return space."""
    loss = _f(table["loss"].to_numpy(dtype=np.float64))[mask]
    out = {}
    for k in keys:
        y, v, e = to_return_space(
            loss, _col(table, k, "var", c)[mask], _col(table, k, "es", c)[mask]
        )
        out[k] = fz0(y, v, e, float(1 - c))
    return out


# ---------------------------------------------------------------------------
# Seeds and serialisation
# ---------------------------------------------------------------------------
def _e4(x: Fraction) -> int:
    y = x * 10000
    if y.denominator != 1:
        raise ValueError(f"seed key: {x} is not a multiple of 1e-4")
    return int(y)


def seed_key(test: int, window: int, key: str, c: Fraction) -> tuple[int, ...]:
    """D42 (e): ``(42, test, W, model index, 10000 q or 0, 10000 c)``."""
    name, _, q = key.partition("@")
    return (42, test, window, list(MODELS).index(name), _e4(Fraction(q)) if q else 0, _e4(c))


def _rng(cfg: Config, key: tuple[int, ...]) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(cfg.runtime.seed, spawn_key=key))
    )


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return float(x) if math.isfinite(x) else None
    return x


def to_json(obj: Mapping[str, Any]) -> str:
    """Sorted keys, indent 2, repr floats, non-finite floats as null, LF, trailing newline."""
    return json.dumps(_clean(obj), sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_per_date(per_date: Mapping[str, Any], directory: str | Path) -> Path:
    """Write the per-date object under a ``reports/runs/`` directory only (D42 (d))."""
    parts = Path(directory).resolve().parts
    if not any(parts[i : i + 2] == ("reports", "runs") for i in range(len(parts) - 1)):
        raise ValueError("write_per_date: per-date output is written only under reports/runs/")
    path = Path(directory) / PER_DATE_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(per_date), encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def _keys(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    keys = {col.split(":")[0] for col in table.columns if col != "loss"}
    missing = [k for k in PRIMARY_KEYS if k not in keys]
    if missing:
        raise ValueError(f"table: primary keys missing: {missing}")
    sens = sorted(keys - set(PRIMARY_KEYS))
    bad = [k for k in sens if k.partition("@")[0] not in MODELS or not k.partition("@")[2]]
    if bad:
        raise ValueError(f"table: unknown keys {bad}")
    return list(PRIMARY_KEYS), sens


def _dm_entry(a: str, b: str, lvl: str, d: FloatArray, mult: int) -> dict[str, Any]:
    lag = dm_lag(d.size)
    main = diebold_mariano(d, lag)
    alt = diebold_mariano(d, mult * lag)
    sens = {k: alt[k] for k in ("lag", "omega", "dm", "p", "flag")}
    return {"a": a, "b": b, "level": lvl, **main, "lag_sensitivity": sens}


def _coverage(
    loss: FloatArray,
    var: FloatArray,
    es: FloatArray,
    key: str,
    c: Fraction,
    cfg: Config,
    window: int,
) -> dict[str, Any]:
    bt, bs = cfg.backtest, cfg.bootstrap
    h = hits(loss, var)
    n, x = int(h.size), int(h.sum())
    if n == 0:
        return {"n": 0, "breaches": 0, "note": "no_admissible_dates"}
    p0, level = float(1 - c), float(bt.kupiec_level)
    lr = float(kupiec_lr(x, n, p0))
    p_bin = float(binomial_equal_tailed(x, n, p0))
    ch = christoffersen(
        h,
        p0,
        bt.christoffersen_mc_reps,
        _rng(cfg, seed_key(TEST_IND, window, key, c)),
        _rng(cfg, seed_key(TEST_CC, window, key, c)),
    )
    den = ch["n10"] + ch["n11"]
    pi11 = ch["n11"] / den if den else math.nan
    b = min(block_length(n), n)
    lengths = [b] + [min(n, max(1, math.ceil(m * b))) for m in bs.sensitivity_multipliers]
    blocks = []
    for i, bl in enumerate(lengths):
        lo, hi = block_ci_rate(
            h, bl, bs.replications, _rng(cfg, seed_key(TEST_BLOCK + i, window, key, c))
        )
        blocks.append({"b": bl, "lower": lo, "upper": hi})
    breach = h == 1
    es_test = es_residual_test(
        loss[breach],
        es[breach],
        bt.es_residual_bootstrap_reps,
        bt.es_residual_min_breaches,
        _rng(cfg, seed_key(TEST_ES, window, key, c)),
    )
    return {
        "n": n,
        "breaches": x,
        "p_hat": x / n,
        "kupiec": {
            "lr": lr,
            "p": float(stats.chi2.sf(lr, 1)),
            "reject": lr > kupiec_critical(level),
        },
        "binomial": {"p": p_bin, "reject": p_bin <= level},
        "christoffersen": ch,
        "markov": {
            "pi11_hat": pi11,
            "size": kupiec_size_markov(n, p0, pi11, level),
            "reason": "pi11_undefined" if den == 0 else None,
        },
        "ci": {
            "clopper_pearson": list(clopper_pearson(x, n)),
            "block": blocks[0],
            "block_sensitivity": blocks[1:],
        },
        "es_residual": es_test,
    }


def evaluate(
    table: pd.DataFrame, cfg: Config, window: int | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply D42 (a)-(c) to ONE pass's D41 table.

    Returns ``(aggregates, per_date)``. Aggregates hold counts, statistics and p-values per
    key and level only; per_date holds every per-date listing (excluded dates and reasons,
    traffic-light blocks with their dates, hit strings) and is written only under
    ``reports/runs/`` (:func:`write_per_date`).

    Raises:
        ValueError: on a non-finite realised loss or a missing primary key.
    """
    w = cfg.design.window if window is None else window
    loss = _f(table["loss"].to_numpy(dtype=np.float64))
    if not np.all(np.isfinite(loss)):
        raise ValueError("table: every realised loss must be finite")
    primary, sens = _keys(table)
    keys = primary + sens
    js = [int(j) for j in table.index.get_level_values("j")]
    dates = [str(d) for d in table.index.get_level_values("date")]
    total = loss.size
    levels = cfg.levels.primary + cfg.levels.descriptive

    reasons: dict[str, dict[str, list[str]]] = {}
    adm_counts: dict[str, Any] = {}
    for c in levels:
        lvl = _dec(c)
        reasons[lvl], adm_counts[lvl] = {}, {}
        for k in keys:
            flags = [str(f) for f in table[f"{k}:flags"]]
            rs = admissibility(_col(table, k, "var", c), _col(table, k, "es", c), flags)
            reasons[lvl][k] = rs
            top = [r.split(":")[0] for r in rs if r]
            adm_counts[lvl][k] = {
                "admissible": rs.count(""),
                "inadmissible": len(top),
                "reasons": {r: top.count(r) for r in REASONS},
                "missing_by_flag": {f: rs.count(f"missing:{f}") for f in (*MISSING_FLAGS, "other")},
            }

    agg: dict[str, Any] = {
        "window": w,
        "n_rows": total,
        "primary_keys": primary,
        "sensitivity_keys": sens,
        "admissibility": adm_counts,
        "common_mask": {},
        "fz0": {},
        "dm": {"unconditional": [], "conditional": [], "descriptive": []},
        "coverage": {},
        "traffic_light": {},
    }
    per: dict[str, Any] = {"window": w, "excluded": {}, "traffic_light": {}, "hits": {}}
    mult = cfg.backtest.dm_lag_sensitivity_multiplier
    tested = set(UNCONDITIONAL_PAIRS) | set(CONDITIONAL_PAIRS)

    for c in cfg.levels.primary:
        lvl = _dec(c)
        adm = {k: np.array([r == "" for r in reasons[lvl][k]], dtype=np.bool_) for k in keys}
        masks = common_mask(adm)
        cm = masks[primary[0]]
        size = int(cm.sum())
        agg["common_mask"][lvl] = {
            "total": total,
            "common": size,
            "share": size / total,
            "below_min_share": Fraction(size, total) < cfg.backtest.common_mask_min_share,
            "sensitivity_masks": {k: int(masks[k].sum()) for k in sens},
        }
        per["excluded"][lvl] = [
            {
                "j": js[i],
                "date": dates[i],
                "reasons": {k: reasons[lvl][k][i] for k in primary if reasons[lvl][k][i]},
            }
            for i in np.flatnonzero(~cm)
        ]
        scores = score_models(table, primary, c, cm)
        agg["fz0"][lvl] = {
            k: {"n": size, "mean": float(np.mean(scores[k])) if size else math.nan} for k in primary
        }
        for a, b in UNCONDITIONAL_PAIRS:
            agg["dm"]["unconditional"].append(_dm_entry(a, b, lvl, scores[a] - scores[b], mult))
        for a, b in CONDITIONAL_PAIRS:
            agg["dm"]["conditional"].append(_dm_entry(a, b, lvl, scores[a] - scores[b], mult))
        for i, a in enumerate(primary):
            for b in primary[i + 1 :]:
                if (a, b) not in tested and (b, a) not in tested:
                    agg["dm"]["descriptive"].append(
                        _dm_entry(a, b, lvl, scores[a] - scores[b], mult)
                    )
        for k in sens:
            base = k.partition("@")[0]
            s = score_models(table, [k, base], c, masks[k])
            n_k = int(masks[k].sum())
            agg["fz0"][lvl][k] = {"n": n_k, "mean": float(np.mean(s[k])) if n_k else math.nan}
            agg["dm"]["descriptive"].append(_dm_entry(k, base, lvl, s[k] - s[base], mult))

        agg["coverage"][lvl] = {}
        per["hits"][lvl] = {}
        for k in keys:
            ok = adm[k]
            var, es = _col(table, k, "var", c), _col(table, k, "es", c)
            agg["coverage"][lvl][k] = _coverage(loss[ok], var[ok], es[ok], k, c, cfg, w)
            hr = hits(loss, var)
            per["hits"][lvl][k] = "".join(
                str(int(x)) if o else "." for x, o in zip(hr, ok, strict=True)
            )
            if c == cfg.backtest.traffic_light_level:
                blocks = traffic_light(hr * ok, ok, cfg.backtest)
                agg["traffic_light"][k] = _zone_counts(blocks)
                per["traffic_light"][k] = [
                    {**bl, "start_date": dates[bl["start_row"]], "end_date": dates[bl["end_row"]]}
                    for bl in blocks
                ]

    for e in agg["dm"]["unconditional"]:
        e["reject"] = not math.isnan(e["p"]) and e["p"] <= TEST_LEVEL
    fam = agg["dm"]["conditional"]
    adj, rej = holm([e["p"] for e in fam])
    for e, p_adj, r in zip(fam, adj, rej, strict=True):
        e.update(p_holm=p_adj, reject=r)

    for c in cfg.levels.descriptive:
        lvl = _dec(c)
        agg["coverage"][lvl] = {}
        for k in keys:
            ok = np.array([r == "" for r in reasons[lvl][k]], dtype=np.bool_)
            h = hits(loss[ok], _col(table, k, "var", c)[ok])
            n, x = int(h.size), int(h.sum())
            agg["coverage"][lvl][k] = {
                "descriptive": True,
                "n": n,
                "breaches": x,
                "p_hat": x / n if n else math.nan,
                "p_binomial": float(binomial_equal_tailed(x, n, float(1 - c))) if n else math.nan,
            }
    return agg, per
