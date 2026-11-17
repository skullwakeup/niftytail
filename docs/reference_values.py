"""NIFTYTAIL Day-1 reference values (stdlib only).

Independently re-derives every hand-calculated number in docs/methodology.md
(specification: docs/plan_v2.2.md, V2.2 amendment A0-A6 + V2.1) and ENFORCES it:
every check raises CheckFailed on mismatch, and the script exits with status 1.

Run (PowerShell, from the project root):
    py -3.13 -I docs\\reference_values.py

Rules followed here:
* Standard library only (no numpy/scipy), so the numbers do not depend on the
  package code that will later be tested against them.
* Exact arithmetic (fractions.Fraction / big integers) wherever the spec
  requires it: tail counts, HS VaR/ES, equal-tailed binomial regions,
  DM-lag and block-length integer rules.
* Each closed-form result is cross-checked by a second, independent route
  (numerical integration, root finding, exact piecewise integration, an
  algebraically different LR formula).
* Floating-point results are labelled [float]; exact ones [exact];
  Monte Carlo ones [MC, seed, R].
* Checks use explicit `require(...)`, not `assert`, so they cannot be disabled
  by `python -O`.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable
from fractions import Fraction
from statistics import NormalDist

SEED = 20261111
N01 = NormalDist()
_N_CHECKS = 0


class CheckFailed(AssertionError):
    """Raised when a reference value does not match its specification."""


def require(condition: bool, label: str) -> None:
    """Enforce a reference check; never just print a mismatch."""
    global _N_CHECKS
    if not condition:
        raise CheckFailed(f"Reference check failed: {label}")
    _N_CHECKS += 1


def check(label: str, got: float, ref: float, tol: float) -> None:
    diff = abs(got - ref)
    print(f"    check {label}: {got:.10g} vs {ref:.10g}  |diff|={diff:.2e}  tol={tol:.0e}")
    require(diff <= tol, f"{label}: got {got!r}, expected {ref!r} (tol {tol})")


def require_raises(fn: Callable[[], object], exc: type[BaseException], label: str) -> None:
    try:
        fn()
    except exc:
        print(f"    raises {exc.__name__}: {label}")
        require(True, label)
        return
    raise CheckFailed(f"Expected {exc.__name__} not raised: {label}")


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def simpson(f: Callable[[float], float], a: float, b: float, n: int = 20000) -> float:
    """Composite Simpson rule on [a, b] with n (even) panels."""
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


# ---------------------------------------------------------------------------
# R1. Notation (A0): exact rational tail counts vs the float trap
# ---------------------------------------------------------------------------
def r1_float_trap() -> None:
    header("R1  Notation A0: c = confidence, p = 1 - c; m = W*p must be exact")
    W = 1000
    expected = {"0.95": (51, 50), "0.99": (11, 10), "0.995": (6, 5), "0.999": (2, 1)}
    for cs, (float_count, exact_count) in expected.items():
        c_float = float(cs)
        m_exact = W * (1 - Fraction(cs))
        got_float = math.ceil(W * (1 - c_float))
        print(
            f"  c={cs}: float ceil(W*(1-c)) = {got_float:>3}   "
            f"float W*(1-c) = {W * (1 - c_float)!r:<22}  exact m = {m_exact} [exact]"
        )
        require(m_exact == exact_count, f"exact m at c={cs}")
        require(got_float == float_count, f"float trap reproduced at c={cs}")
    print(
        "  -> float tail counts are unsafe (11 instead of 10, 51 instead of 50);"
        " parse c as Fraction(str)."
    )


# ---------------------------------------------------------------------------
# R2. Historical Simulation (A1)
# ---------------------------------------------------------------------------
def hs_formula(losses: list[Fraction], c: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    """A1: VaR = L_(ceil(W c)); ES = [(m-floor m) L_(W-floor m) + sum top floor m]/m."""
    L = sorted(losses)
    W = len(L)
    m = W * (1 - c)
    var = L[math.ceil(W * c) - 1]  # exact: W*c is a Fraction; 1-based index
    fm = math.floor(m)
    top = sum(L[W - fm :], Fraction(0))
    es = (top + (m - fm) * L[W - fm - 1]) / m if m != fm else top / m
    return var, es, m


def hs_integral(losses: list[Fraction], c: Fraction) -> Fraction:
    """Independent route: ES = (1/p) * integral_c^1 VaR_u du, exactly (step function)."""
    L = sorted(losses)
    W = len(L)
    total = Fraction(0)
    for j in range(1, W + 1):
        lo = max(Fraction(j - 1, W), c)
        hi = Fraction(j, W)
        if hi > lo:
            total += (hi - lo) * L[j - 1]
    return total / (1 - c)


def r2_hs() -> None:
    header("R2  Historical Simulation (A1), losses = 1..W  [exact]")
    rows = [  # W, c, m, VaR, ES  (methodology.md 4)
        (1000, "0.99", Fraction(10), 990, Fraction(1991, 2)),
        (1000, "0.95", Fraction(50), 950, Fraction(1951, 2)),
        (250, "0.99", Fraction(5, 2), 248, Fraction(1246, 5)),
        (250, "0.95", Fraction(25, 2), 238, Fraction(6106, 25)),
        (20, "0.95", Fraction(1), 19, Fraction(20)),
    ]
    print(f"  {'W':>5} {'c':>6} {'m':>6} {'VaR':>6} {'ES (exact)':>14} {'ES float':>10}")
    for W, cs, m_ref, var_ref, es_ref in rows:
        c = Fraction(cs)
        losses = [Fraction(i) for i in range(1, W + 1)]
        var, es, m = hs_formula(losses, c)
        es_int = hs_integral(losses, c)
        print(f"  {W:>5} {cs:>6} {m!s:>6} {var!s:>6} {es!s:>14} {float(es):>10.4f}")
        require((m, var, es) == (m_ref, var_ref, es_ref), f"HS W={W} c={cs} values")
        require(es == es_int, f"HS W={W} c={cs}: formula == exact integral of VaR_u")
        shuffled = losses[:]
        random.Random(SEED).shuffle(shuffled)
        require(hs_formula(shuffled, c)[:2] == (var, es), f"HS W={W} c={cs}: order invariance")
    # tail selection: replacing small losses by large gains must not change VaR/ES
    W, c = 1000, Fraction("0.99")
    base = [Fraction(i) for i in range(1, W + 1)]
    v0, e0, _ = hs_formula(base, c)
    shifted = [x if x > 100 else -Fraction(10**6) for x in base]
    v1, e1, _ = hs_formula(shifted, c)
    print(f"  tail-selection (100 losses -> large gains): VaR {v0}->{v1}, ES {e0}->{e1}")
    require((v0, e0) == (v1, e1), "HS tail selection uses the loss tail")
    above = sum(1 for x in base if x > v0)
    print(f"  exceedances strictly above VaR (W=1000, c=0.99, no ties): {above}")
    require(above == 10, "exactly m losses exceed VaR")


# ---------------------------------------------------------------------------
# R3. Gaussian VaR / ES
# ---------------------------------------------------------------------------
def r3_gaussian() -> None:
    header("R3  Gaussian (mu=0, sigma=1)  [float]")
    refs = {"0.95": (1.644854, 2.062713), "0.99": (2.326348, 2.665214)}
    for cs, (var_ref, es_ref) in refs.items():
        p = 1 - float(cs)
        z = N01.inv_cdf(float(cs))
        es = N01.pdf(z) / p
        es_num = simpson(lambda x: x * N01.pdf(x), z, z + 14.0, 40000) / p
        print(f"  c={cs}: VaR = {z:.6f}   ES = {es:.6f}")
        check(f"VaR c={cs} vs table", z, var_ref, 5e-7)
        check(f"ES c={cs} vs table", es, es_ref, 5e-7)
        check(f"ES c={cs} closed form vs integral", es_num, es, 1e-8)


# ---------------------------------------------------------------------------
# R4. POT-GPD tail VaR / ES (V2 2.3)
# ---------------------------------------------------------------------------
def gpd_var(u: float, xi: float, beta: float, n: int, nu: int, p: float) -> float:
    return u + (beta / xi) * (((n / nu) * p) ** (-xi) - 1)


def gpd_es(var: float, u: float, xi: float, beta: float) -> float:
    return (var + beta - xi * u) / (1 - xi)


def r4_gpd() -> None:
    header("R4  POT-GPD: n=1000, N_u=100, u=2, xi=0.25, beta=0.5  [float]")
    n, nu, u, xi, beta = 1000, 100, 2.0, 0.25, 0.5
    refs = {"0.99": (3.556559, 4.742078), "0.95": (2.378414, 3.171219)}

    def surv(x: float) -> float:  # tail estimator F_bar(x), x >= u
        return (nu / n) * (1 + xi * (x - u) / beta) ** (-1 / xi)

    for cs, (var_ref, es_ref) in refs.items():
        p = 1 - float(cs)
        var = gpd_var(u, xi, beta, n, nu, p)
        es = gpd_es(var, u, xi, beta)
        lo, hi = u, u + 1e4  # independent VaR: bisection on surv = p
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if surv(mid) > p else (lo, mid)
        var_root = (lo + hi) / 2

        def integrand(t: float, var: float = var) -> float:  # y = t/(1-t) substitution
            if t >= 1.0:
                return 0.0
            return surv(var + t / (1 - t)) / (1 - t) ** 2

        es_num = var + simpson(integrand, 0.0, 1.0, 200000) / p
        print(f"  c={cs}: VaR = {var:.6f}   ES = {es:.6f}")
        check(f"VaR c={cs} vs table", var, var_ref, 5e-7)
        check(f"ES  c={cs} vs table", es, es_ref, 5e-7)
        check(f"VaR c={cs} closed form vs bisection", var_root, var, 1e-9)
        check(f"ES  c={cs} closed form vs tail integral", es_num, es, 1e-6)
    p = 0.01
    v0 = u - beta * math.log((n / nu) * p)
    for xe in (1e-9, -1e-9):
        print(
            f"  xi={xe:+.0e}: naive VaR = {gpd_var(u, xe, beta, n, nu, p):.9f}   "
            f"xi=0 limit VaR = {v0:.9f} (ES limit = VaR + beta = {v0 + beta:.9f})"
        )
        check(
            f"xi={xe:+.0e} naive formula near the xi=0 limit",
            gpd_var(u, xe, beta, n, nu, p),
            v0,
            1e-6,
        )
    ses = {400: 0.060, 200: 0.085, 144: 0.100, 100: 0.120, 50: 0.170}
    print(
        "  SE(xi_hat) ~ (1+xi)/sqrt(N_u) at xi=0.2:",
        ", ".join(f"N_u={k}: {1.2 / math.sqrt(k):.3f}" for k in ses),
    )
    for k, ref in ses.items():
        check(f"SE heuristic N_u={k}", round(1.2 / math.sqrt(k), 3), ref, 1e-12)


# ---------------------------------------------------------------------------
# R5. FZ0 in return space (V2.1 2, Patton-Ziegel-Chen 2019 eq. 6)
# ---------------------------------------------------------------------------
def fz0(y: float, v: float, e: float, a: float) -> float:
    """FZ0 loss, return space. Raises ValueError on NaN/inf input or e <= v < 0 violation."""
    for name, val in (("y", y), ("v", v), ("e", e), ("alpha", a)):
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            raise ValueError(f"FZ0 input {name} must be a finite number, got {val!r}")
    if not 0 < a < 1:
        raise ValueError(f"FZ0 alpha must lie in (0, 1), got {a!r}")
    if not (e <= v < 0):
        raise ValueError(f"FZ0 requires e <= v < 0, got v={v!r}, e={e!r}")
    hit = 1.0 if y <= v else 0.0
    return -(1 / (a * e)) * hit * (v - y) + v / e + math.log(-e) - 1


def r5_fz0() -> None:
    header("R5  FZ0 (return space; alpha = tail probability p)  [float]")
    table = {  # alpha: [(Y or None for tie, FZ0)]
        0.05: [(-1.0, 0.521445), (-2.0, 3.964933), (-3.0, 13.660902), (None, 0.521445)],
        0.01: [(-2.0, 0.853140), (-3.0, 26.128862)],
    }
    for a, rows in table.items():
        v = N01.inv_cdf(a)
        e = -N01.pdf(v) / a
        print(f"  alpha={a}: v = {v:.6f}  e = {e:.6f}")
        for y, ref in rows:
            yy = v if y is None else y
            tag = "Y = v (tie)" if y is None else f"Y = {y:+.0f}"
            got = fz0(yy, v, e, a)
            print(f"    FZ0({tag:<12}) = {got:.6f}")
            check(f"FZ0 alpha={a} {tag}", got, ref, 5e-7)
        # E[(v - Y) 1{Y <= v}] by numerical integration (v bound explicitly)
        tail_mean = simpson(lambda y, v=v: (v - y) * N01.pdf(y), v - 14.0, v, 40000)
        efz = -(1 / (a * e)) * tail_mean + v / e + math.log(-e) - 1
        check(f"E[FZ0] alpha={a} numeric vs ln(-e)", efz, math.log(-e), 1e-9)
    a = 0.05
    v, e = -N01.inv_cdf(1 - a), -N01.pdf(N01.inv_cdf(1 - a)) / a
    print(f"  conversion: VaR_L={-v:.6f}, ES_L={-e:.6f}, L=3 -> Y=-3")
    check("loss-space conversion example", fz0(-3.0, v, e, a), 13.660902, 5e-7)
    y, v1, e1, v2, e2, cst = -2.5, -1.6, -2.1, -1.9, -2.4, 3.7
    d1 = fz0(y, v1, e1, a) - fz0(y, v2, e2, a)
    d2 = fz0(cst * y, cst * v1, cst * e1, a) - fz0(cst * y, cst * v2, cst * e2, a)
    check("homogeneity of degree 0 (score differences)", d2, d1, 1e-12)
    print("  invalid inputs must raise ValueError:")
    nan, inf = float("nan"), float("inf")
    invalid = {
        "y = NaN": (nan, -1.0, -2.0, a),
        "v = NaN": (-1.0, nan, -2.0, a),
        "e = NaN": (-1.0, -1.0, nan, a),
        "alpha = NaN": (-1.0, -1.0, -2.0, nan),
        "y = -inf": (-inf, -1.0, -2.0, a),
        "alpha = 0": (-1.0, -1.0, -2.0, 0.0),
        "alpha = 1": (-1.0, -1.0, -2.0, 1.0),
        "v = 0": (-1.0, 0.0, -1.0, a),
        "v > 0": (-1.0, 0.5, -1.0, a),
        "e > v": (-1.0, -2.0, -1.0, a),
    }
    for label, args in invalid.items():
        require_raises(lambda args=args: fz0(*args), ValueError, label)
    check(
        "boundary e == v accepted", fz0(-1.0, -1.5, -1.5, a), -1.5 / -1.5 + math.log(1.5) - 1, 1e-12
    )


# ---------------------------------------------------------------------------
# R6. Kupiec LR; R6b T=250; R7 equal-tailed exact binomial (D19), kept separate
# ---------------------------------------------------------------------------
CRIT = N01.inv_cdf(0.975) ** 2  # chi2_1 0.95 quantile = z_{0.975}^2 (derived, not typed)


def kupiec_lr(x: int, T: int, p0: float) -> float:
    """LR_POF = -2[ln L(p0) - ln L(p_hat)], with 0 ln 0 = 0."""

    def ll(q: float) -> float:
        a = (T - x) * math.log(1 - q) if T - x > 0 else 0.0
        b = x * math.log(q) if x > 0 else 0.0
        return a + b

    ph = x / T
    l1 = ((T - x) * math.log(1 - ph) if 0 < x < T else 0.0) + (x * math.log(ph) if x > 0 else 0.0)
    return -2 * (ll(p0) - l1)


def kupiec_lr_alt(x: int, T: int, p0: float) -> float:
    """Independent algebraic form: LR = 2[x ln(p_hat/p0) + (T-x) ln((1-p_hat)/(1-p0))]."""
    ph = Fraction(x, T)
    t1 = x * math.log(float(ph) / p0) if x > 0 else 0.0
    t2 = (T - x) * math.log(float(1 - ph) / (1 - p0)) if x < T else 0.0
    return 2 * (t1 + t2)


def lr_region(
    T: int, p0: float, lr: Callable[[int, int, float], float] = kupiec_lr
) -> tuple[int, int, float]:
    acc = [x for x in range(T + 1) if lr(x, T, p0) <= CRIT]
    require(acc == list(range(acc[0], acc[-1] + 1)), f"LR region T={T} p0={p0} is an interval")
    margin = min(
        abs(lr(x, T, p0) - CRIT) for x in (acc[0] - 1, acc[0], acc[-1], acc[-1] + 1) if 0 <= x <= T
    )
    return acc[0], acc[-1], margin


def equal_tailed_region(T: int, p: Fraction, tail: Fraction = Fraction(1, 40)) -> tuple[int, int]:
    """Exact big-integer region: [lo, hi] = {x : P(X<=x) > tail and P(X>=x) > tail}.

    With M_x = C(T,x) a^x b^(T-x), p = a/D, b = D - a, total = D^T.
    """
    a, D = p.numerator, p.denominator
    b = D - a
    total = D**T
    M = b**T
    S_le: list[int] = []
    s = 0
    lo: int | None = None
    hi: int | None = None
    x = 0
    while True:
        s += M
        S_le.append(s)
        if lo is None and s * tail.denominator > tail.numerator * total:
            lo = x
        ge = total - (S_le[x - 1] if x > 0 else 0)  # P(X >= x) * total
        if ge * tail.denominator > tail.numerator * total:
            hi = x
        elif lo is not None:
            break
        num, den = M * (T - x) * a, (x + 1) * b
        require(num % den == 0, "exact binomial recursion stays integral")
        M = num // den
        x += 1
    if lo is None or hi is None:
        raise CheckFailed("equal-tailed region not found")
    return lo, hi


def equal_tailed_pvalue(x: int, T: int, p: Fraction) -> Fraction:
    """D19: p = min(1, 2*min(P(X<=x), P(X>=x))), exact."""
    a, D = p.numerator, p.denominator
    b = D - a
    pm = [Fraction(math.comb(T, k) * a**k * b ** (T - k), D**T) for k in range(T + 1)]
    le = sum(pm[: x + 1], Fraction(0))
    ge = sum(pm[x:], Fraction(0))
    return min(Fraction(1), 2 * min(le, ge))


def r6_kupiec_lr() -> None:
    header("R6  Kupiec LR (two-sided, 5%): reject if LR > chi2_1(0.95)")
    print(f"  chi2_1(0.95) = z_0.975^2 = {CRIT:.9f}  [float, derived]")
    check("chi2_1(0.95) critical value", CRIT, 3.841459, 5e-7)
    check("LR(T=250, x=5, p0=0.01)", kupiec_lr(5, 250, 0.01), 1.9568, 5e-5)
    check(
        "LR(T=250, x=5) two independent formulas",
        kupiec_lr_alt(5, 250, 0.01),
        kupiec_lr(5, 250, 0.01),
        1e-10,
    )
    check(
        "LR(T=250, x=0, p0=0.01) = -2T ln(1-p0)",
        kupiec_lr(0, 250, 0.01),
        -2 * 250 * math.log(0.99),
        1e-12,
    )
    for p0, ref in ((0.05, (251, 315)), (0.01, (43, 71))):
        lo, hi, mg = lr_region(5650, p0)
        lo2, hi2, _ = lr_region(5650, p0, kupiec_lr_alt)
        print(
            f"  LR acceptance region T=5650 p0={p0}: [{lo}, {hi}]  "
            f"(alt formula [{lo2}, {hi2}]; min |LR-crit| at boundaries = {mg:.3e})"
        )
        require((lo, hi) == ref, f"Kupiec LR region T=5650 p0={p0} == {list(ref)}")
        require((lo2, hi2) == ref, f"Kupiec LR region T=5650 p0={p0} (alt formula)")
        require(mg > 1e-6, f"LR region T=5650 p0={p0} is float-robust")

    header("R6b Kupiec LR acceptance regions at T=250 (traffic-light block size; DX-1)")
    for p0, ref in ((0.01, (1, 6)), (0.05, (7, 19))):
        lo, hi, mg = lr_region(250, p0)
        lo2, hi2, _ = lr_region(250, p0, kupiec_lr_alt)
        print(
            f"  LR acceptance region T=250 p0={p0}: [{lo}, {hi}]  "
            f"(alt formula [{lo2}, {hi2}]; min |LR-crit| at boundaries = {mg:.3e})"
        )
        require((lo, hi) == ref, f"Kupiec LR region T=250 p0={p0} == {list(ref)}")
        require((lo2, hi2) == ref, f"Kupiec LR region T=250 p0={p0} (alt formula)")
        require(mg > 1e-6, f"LR region T=250 p0={p0} is float-robust")
    lr0 = kupiec_lr(0, 250, 0.01)
    print(f"  zero breaches, T=250, p0=0.01: LR = {lr0:.6f} > {CRIT:.6f} -> rejected")
    require(lr0 > CRIT, "x=0 at T=250, p0=0.01 is rejected by Kupiec LR")


def r7_equal_tailed() -> None:
    header("R7  Equal-tailed exact binomial (D19), separate from LR  [exact big-int]")
    table = {  # (T, p0): region   (V2 2.2 table / methodology.md 10.2)
        (5650, "0.05"): (251, 315),
        (5650, "0.01"): (42, 72),
        (5650, "0.001"): (2, 11),
        (6650, "0.05"): (298, 368),
        (6650, "0.01"): (51, 83),
        (6650, "0.001"): (2, 12),
        (4770, "0.05"): (209, 268),
        (4770, "0.01"): (35, 62),
        (4770, "0.001"): (1, 9),
        (250, "0.05"): (6, 20),
        (250, "0.01"): (0, 6),
        (250, "0.001"): (0, 2),
    }
    print(f"  {'T':>5} {'p0':>6} {'E[x]':>8}  region (each tail > 2.5%)")
    for (T, ps), ref in table.items():
        p = Fraction(ps)
        lo, hi = equal_tailed_region(T, p)
        print(f"  {T:>5} {ps:>6} {float(T * p):>8.2f}  [{lo}, {hi}]")
        require((lo, hi) == ref, f"equal-tailed region T={T} p0={ps} == {list(ref)}")
    pv = equal_tailed_pvalue(5, 250, Fraction("0.01"))
    print(
        f"  D19 example: T=250, x=5, p0=0.01 -> equal-tailed p = {float(pv):.6f}  [exact rational]"
    )
    check("D19 example p-value", float(pv), 0.215625, 5e-7)
    for x in (41, 42, 72, 73):  # region membership <=> p > 0.05 (consistency of D19 with R7)
        pvx = equal_tailed_pvalue(x, 5650, Fraction("0.01"))
        require((pvx > Fraction(1, 20)) == (42 <= x <= 72), f"D19 p-value vs region at x={x}")
    print("  D19 p-value > 0.05 exactly on [42, 72] at T=5650, p0=0.01 (checked at x=41,42,72,73)")
    print(
        "  NOTE: LR and equal-tailed regions coincide at p0=0.05 ([251,315]) but differ at"
        " p0=0.01 (LR [43,71] vs equal-tailed [42,72]); at T=250, p0=0.01: LR [1,6] vs [0,6]."
    )


# ---------------------------------------------------------------------------
# R8. Kupiec size/power under iid hits: exact + MC
# ---------------------------------------------------------------------------
def log_pmf(k: int, n: int, p: float) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log(1 - p)
    )


def r8_power() -> None:
    T, R = 5650, 100_000
    header(
        f"R8  Kupiec LR size/power, iid hits, T={T}: exact [float sum] + MC [seed={SEED}, R={R}]"
    )
    rng = random.Random(SEED)
    grid = {
        0.01: {0.01: 0.0523, 0.0125: 0.4508, 0.015: 0.9295, 0.02: 1.0000},
        0.05: {0.05: 0.0472, 0.055: 0.3879, 0.06: 0.9070, 0.07: 1.0000},
    }
    for p0, cells in grid.items():
        rej = {x for x in range(T + 1) if kupiec_lr(x, T, p0) > CRIT}
        for p1, ref in cells.items():
            exact = sum(math.exp(log_pmf(x, T, p1)) for x in rej)
            mc = sum(1 for _ in range(R) if rng.binomialvariate(T, p1) in rej) / R
            se = math.sqrt(exact * (1 - exact) / R)
            print(f"  p0={p0:<5} true p={p1:<7} exact={exact:.4f}   MC={mc:.4f} (se {se:.4f})")
            check(f"exact power p0={p0} p={p1} (4 dp)", round(exact, 4), ref, 1e-12)
            require(abs(mc - exact) <= 4 * se + 1e-12, f"MC within 4 se of exact, p0={p0} p={p1}")


# ---------------------------------------------------------------------------
# R9. Kupiec true size under Markov-dependent hits (correct unconditional rate)
# ---------------------------------------------------------------------------
def markov_size(T: int, p0: float, p11: float, rej: set[int], cmax: int) -> float:
    p01 = p0 * (1 - p11) / (1 - p0)
    d0 = [0.0] * (cmax + 2)
    d1 = [0.0] * (cmax + 2)
    d0[0], d1[1] = 1 - p0, p0  # t = 1 drawn from the stationary law
    for _ in range(T - 1):
        n0 = [0.0] * (cmax + 2)
        n1 = [0.0] * (cmax + 2)
        for c in range(cmax + 1):
            a, b = d0[c], d1[c]
            if a == 0.0 and b == 0.0:
                continue
            n0[c] += a * (1 - p01) + b * (1 - p11)
            n1[c + 1] += a * p01 + b * p11
        d0, d1 = n0, n1
    mass = sum(d0[: cmax + 1]) + sum(d1[: cmax + 1])
    require(abs(mass - 1) < 1e-9, f"DP probability mass retained (cmax={cmax})")
    return sum(d0[c] + d1[c] for c in range(cmax + 1) if c in rej)


def r9_markov() -> None:
    T = 5650
    header(
        f"R9  Kupiec true size, two-state Markov hits with correct rate p0, T={T}  "
        "[exact DP, float]"
    )
    cells = {
        0.01: ({0.01: 0.0523, 0.05: 0.0623, 0.10: 0.0764, 0.20: 0.1099}, 400),
        0.05: ({0.05: 0.0472, 0.10: 0.0598, 0.20: 0.0906, 0.30: 0.1296}, 900),
    }
    for p0, (sizes, cmax) in cells.items():
        rej = {x for x in range(T + 1) if kupiec_lr(x, T, p0) > CRIT}
        for p11, ref in sizes.items():
            p01 = p0 * (1 - p11) / (1 - p0)
            rho = p11 - p01
            size = markov_size(T, p0, p11, rej, cmax)
            print(
                f"  p0={p0:<5} pi11={p11:<5} rho={rho:.3f}  "
                f"var-inflation={(1 + rho) / (1 - rho):.3f}"
                f"  true size of nominal 5% test = {size:.4f}"
            )
            check(f"Markov size p0={p0} pi11={p11} (4 dp)", round(size, 4), ref, 1e-12)
    R = 2000
    rng = random.Random(SEED)
    p0, p11 = 0.01, 0.10
    p01 = p0 * (1 - p11) / (1 - p0)
    rej = {x for x in range(T + 1) if kupiec_lr(x, T, p0) > CRIT}
    hits = 0
    for _ in range(R):
        s = 1 if rng.random() < p0 else 0
        cnt = s
        for _ in range(T - 1):
            s = 1 if rng.random() < (p11 if s else p01) else 0
            cnt += s
        hits += cnt in rej
    r = hits / R
    exact = markov_size(T, p0, p11, rej, 400)
    se = math.sqrt(exact * (1 - exact) / R)
    print(f"  MC check p0=0.01 pi11=0.10 [seed={SEED}, R={R}]: size = {r:.4f} (se {se:.4f})")
    require(abs(r - exact) <= 4 * se, "Markov MC within 4 se of exact DP")


# ---------------------------------------------------------------------------
# R10. Integer rules: DM lag (A5b) and block length (A5c)
# ---------------------------------------------------------------------------
def r10_integer_rules() -> None:
    header("R10 Integer rules  [exact]")
    T = 5650
    x = Fraction(T, 100) ** 2  # l >= k  <=>  (T/100)^2 >= (k/4)^9
    lag = max(k for k in range(0, 100) if x >= Fraction(k, 4) ** 9)
    lag_float = math.floor(4 * (T / 100) ** (2 / 9))
    print(f"  DM Newey-West lag floor(4*(T/100)^(2/9)) at T={T}: {lag} (float: {lag_float})")
    require(lag == 9 and lag_float == 9, "DM lag at T=5650 == 9 (exact and float)")
    for n, ref in ((2000, 13), (5650, 18)):
        b = next(k for k in range(1, 100) if k**3 >= n)  # ceil(n^(1/3)) exactly
        print(
            f"  block length ceil(n^(1/3)) at n={n}: {b}   "
            f"({b - 1}^3={(b - 1) ** 3} < {n} <= {b}^3={b**3})"
        )
        require(b == ref, f"block length n={n} == {ref}")


def main() -> int:
    print(
        "NIFTYTAIL reference values - stdlib only - Python",
        ".".join(map(str, sys.version_info[:3])),
    )
    try:
        r1_float_trap()
        r2_hs()
        r3_gaussian()
        r4_gpd()
        r5_fz0()
        r6_kupiec_lr()
        r7_equal_tailed()
        r8_power()
        r9_markov()
        r10_integer_rules()
    except CheckFailed as exc:
        print()
        print(f"FAILED after {_N_CHECKS} passing checks: {exc}")
        return 1
    print()
    print(f"ALL {_N_CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
