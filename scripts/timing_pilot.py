"""A3 timing pilot (D32 (6); plan A6 Day 6). Synthetic data only; prints timings only.

Run: ``uv run --locked python scripts/timing_pilot.py configs/base.toml``

Measures (median of 3 repeats, single process): scipy ``goodness_of_fit`` (GPD, loc = 0,
AD statistic) with 199 and 999 MC samples and one ``fit_mle`` call, at N_u in {100, 250, 400}
on seeded GPD(xi = 0.25, beta = 1) excesses; the S4 reference simulation once; and W
concurrent copies of the 199-sample call at N_u = 250 (parallel efficiency e).
Projection (pinned): S = sum over the 16 grid candidates of t_gof199(2000 (1 - q_k)),
piecewise-linear in N_u; T_total(R) = R * 10 * S / (W * e) + t_S4; R = 200 if
T_total(200) <= 2 h, else R = 100.
"""

from __future__ import annotations

import niftytail.__main__  # noqa: F401  # isort: skip  (pins BLAS threads at import)

import inspect
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from typing import Any

import numpy as np
import scipy
from scipy import stats

from niftytail.config import Config, load_config
from niftytail.gpd import fit_mle

SEED = 20261111
REPEATS = 3
N_US = (100, 250, 400)
XI = 0.25


def _gen(child: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED).spawn(8)[child]))


def _sample(n_u: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(stats.genpareto.rvs(XI, size=n_u, random_state=_gen(0)))


def _rng_kwarg(g: np.random.Generator) -> dict[str, Any]:
    params = inspect.signature(stats.goodness_of_fit).parameters
    return {"rng": g} if "rng" in params else {"random_state": g}


def gof_seconds(n_u: int, reps: int) -> float:
    """Wall time of one goodness_of_fit call (also used by the worker processes)."""
    y = _sample(n_u)
    kw = _rng_kwarg(_gen(1))
    t = time.perf_counter()
    stats.goodness_of_fit(
        stats.genpareto, y, known_params={"loc": 0}, statistic="ad", n_mc_samples=reps, **kw
    )
    return time.perf_counter() - t


def _median_time(fn: Callable[[], float]) -> float:
    return statistics.median(fn() for _ in range(REPEATS))


def fit_seconds(n_u: int, cfg: Config) -> float:
    y = _sample(n_u)
    t = time.perf_counter()
    fit_mle(y, min_excesses=2, xi_zero_tol=float(cfg.threshold.xi_zero_tol), settings=cfg.gpd)
    return time.perf_counter() - t


def s4_seconds(cfg: Config) -> float:
    """GARCH(1,1), unit variance, standardised t innovations; chunked Python recursion."""
    ss = cfg.simstudy
    a1, b1, nu = float(ss.s4_garch_a1), float(ss.s4_garch_b1), ss.s4_t_df
    n, burn = ss.s4_reference_size, ss.s4_burn_in
    omega = 1.0 - a1 - b1
    g = _gen(2)
    t = time.perf_counter()
    out = np.empty(n + burn)
    s2, chunk = 1.0, 1_000_000
    scale = math.sqrt(nu / (nu - 2))  # standardise t(nu) to unit variance
    for start in range(0, n + burn, chunk):
        z = (g.standard_t(nu, size=min(chunk, n + burn - start)) / scale).tolist()
        r = [0.0] * len(z)
        for i, zi in enumerate(z):
            ri = math.sqrt(s2) * zi
            r[i] = ri
            s2 = omega + a1 * ri * ri + b1 * s2
        out[start : start + len(z)] = r
    losses = -out[burn:]
    var99 = float(np.quantile(losses, 0.99))
    float(np.mean(losses[losses > var99]))  # reference ES, as the study will compute it
    return time.perf_counter() - t


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    cpu = os.cpu_count() or 1
    workers = max(1, cpu - 1)
    print(f"platform: {platform.platform()}; Python {sys.version.split()[0]}")
    print(f"scipy {scipy.__version__}; numpy {np.__version__}")
    print(f"goodness_of_fit seed argument: {next(iter(_rng_kwarg(_gen(1))))}")
    print(f"cpu_count: {cpu}; W = {workers}")

    t199 = {n: _median_time(lambda n=n: gof_seconds(n, 199)) for n in N_US}
    t999 = {n: _median_time(lambda n=n: gof_seconds(n, 999)) for n in N_US}
    tfit = {n: _median_time(lambda n=n: fit_seconds(n, cfg)) for n in N_US}
    print("\n| N_u | t_gof (B = 199) s | t_gof (B = 999) s | t_fit_mle s |")
    print("|---|---|---|---|")
    for n in N_US:
        print(f"| {n} | {t199[n]:.4g} | {t999[n]:.4g} | {tfit[n]:.4g} |")

    t_s4 = s4_seconds(cfg)
    print(f"\nS4 reference simulation ({cfg.simstudy.s4_reference_size} obs): {t_s4:.4g} s")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        par = list(pool.map(gof_seconds, [250] * workers, [199] * workers))
    t_par = statistics.median(par)
    e = t199[250] / t_par
    print(f"parallel: {workers} copies at N_u = 250, B = 199: median {t_par:.4g} s; e = {e:.4g}")

    n_s = cfg.simstudy.sample_size
    n_uk = [int(n_s * (1 - q)) for q in cfg.threshold.grid]
    assert all(Fraction(n_s) * (1 - q) == k for q, k in zip(cfg.threshold.grid, n_uk, strict=True))
    s_cost = float(sum(np.interp(n_uk, N_US, [t199[n] for n in N_US])))

    def total(r: int) -> float:
        return r * 10 * s_cost / (workers * e) + t_s4

    t200, t100 = total(200) / 3600, total(100) / 3600
    if t200 <= 2:
        rule = "R = 200"
    else:
        rule = "R = 100" if t100 <= 2 else "NONE: even R = 100 exceeds 2 h (report, no decision)"
    print(f"\nS = {s_cost:.4g} s per replication (16 candidates, B = 199)")
    print(f"T_total(200) = {t200:.4g} h; T_total(100) = {t100:.4g} h -> rule gives {rule}")
    serial = 16 * statistics.mean(t999.values())
    print(f"Day-7 serial estimate of one real selection (16 x mean t_gof999): {serial:.4g} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/base.toml"))
