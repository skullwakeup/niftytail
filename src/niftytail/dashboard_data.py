"""Day-13A dashboard data layer (D55, D58, D59): tracked artefacts in, plain dicts and lists out.

Pure Python (standard library only; no streamlit, no pandas, no estimation module). Every file is
read by path and checked against its protected sha256 before use (``read``); the functions return
display-ready rows for ``app/dashboard.py``. The local daily figures (D53) are offered only in
local mode and only if their bytes match the values recorded in status.md; in public mode
(``NIFTYTAIL_PUBLIC=1``) no run directory is touched.

D59 adds closed-form arithmetic on the stored next-day fit (% of position, mirrored tail
VaR/ES with a self-check, exceedance probabilities, return levels, the volatility scenario)
and Vega-Lite chart specs; nothing is estimated here, and no mirrored value replaces a stored
one on screen. series.json and check.json are read in local mode only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tomllib
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "BANNER",
    "DISPLAY_NAMES",
    "EXPECTED_SHA256",
    "FIGURES",
    "HERE",
    "LEVELS",
    "MODELS",
    "MODEL_COLOURS",
    "PASSES",
    "DashboardDataError",
    "budget_position",
    "budget_rows",
    "coverage_rows",
    "crash_rows",
    "display",
    "dm_rows",
    "drawdown_spec",
    "exceed_prob",
    "forecast_rows",
    "freshness",
    "headline",
    "health_rows",
    "history_count",
    "inr",
    "is_public",
    "kpis",
    "last_session",
    "level_text",
    "live_bundle",
    "live_forecast",
    "local_figures",
    "local_mode",
    "market",
    "market_spec",
    "metrics",
    "p_text",
    "pct",
    "pct_of_position",
    "period_text",
    "pipeline_numbers",
    "provenance",
    "read",
    "recent_spec",
    "replay_rows",
    "return_level",
    "return_level_rows",
    "risk_card",
    "rupee_loss",
    "self_check",
    "sensitivity_rows",
    "stress_rows",
    "svg_data_uri",
    "tail_risk",
    "tail_spec",
    "vol_shock",
    "volatility_ratio",
]

# The repository root for the UI, which runs from it (README); the UI imports no pathlib.
HERE = Path(".")
CONFIG = "configs/base.toml"
PASSES = (
    ("pr_w1000", "PR W1000 (primary)", "reports/oos/w1000"),
    ("pr_w2000", "PR W2000", "reports/oos/w2000"),
    ("tri_w1000", "TRI W1000", "reports/oos/tri_w1000"),
)
FIGURES = {
    "F1": "reports/figures/coverage.svg",
    "F2": "reports/figures/tests_heatmap.svg",
    "F3": "reports/figures/dm.svg",
    "F4": "reports/figures/traffic_light.svg",
    "F5": "reports/figures/sensitivity.svg",
    "F6": "reports/figures/oos_blocks.svg",
    "F7": "reports/figures/xi_blocks.svg",
    "F8": "reports/figures/descriptive_levels.svg",
}
REVIEW = "docs/results_review.md"
FIGURES_DOC = "docs/figures.md"
TL = "reports/oos/traffic_light_blocks.json"
XI = "reports/oos/xi_path/summary.json"
# The protected "keep their sha256" values (a test asserts equality with docs/protected_hashes.md).
EXPECTED_SHA256 = {
    "reports/oos/w1000/backtest.json": (
        "bc79fd419d16ae5ac2cf03800ef6b505088bd9ea27fc559078d78cdb35b7f20a"
    ),
    "reports/oos/w1000/manifest.json": (
        "289f603029705c284b0e1f3996a36fd9ae38b44a6944b883ccce544bed9781b5"
    ),
    "reports/oos/w2000/backtest.json": (
        "f900783ca58644aae3fba27fe51f6b936107959ed01366178ee64e29b44b5f12"
    ),
    "reports/oos/w2000/manifest.json": (
        "289f603029705c284b0e1f3996a36fd9ae38b44a6944b883ccce544bed9781b5"
    ),
    "reports/oos/tri_w1000/backtest.json": (
        "425aba7b13494c4af47a9b689d624966084870a948c999947daec266ac4006ad"
    ),
    "reports/oos/tri_w1000/manifest.json": (
        "5b0c5833fd74b1846e60c6dcc150977c070d58929b7ae54a174ec9869be8cd0a"
    ),
    "reports/oos/traffic_light_blocks.json": (
        "5d736d95d0121937c107873d3d05f3d57953df5f1239b2b37edd964e670af1f0"
    ),
    "reports/oos/xi_path/summary.json": (
        "e2ac9882bd7cd826ff80f46ee9b79a28ab45f2025f0f887c2bc2b300af92c30f"
    ),
    "reports/figures/coverage.svg": (
        "69051fb301613ecb5d4874540a8252a206541474106e11d8f2de8512343d7d76"
    ),
    "reports/figures/tests_heatmap.svg": (
        "82ac0ad2e5e83e728417fca98ae634b03a4b68b623af8dcc2dd2748da717ec51"
    ),
    "reports/figures/dm.svg": ("304d6b3db31908051d08e3ad6812b90c573e73a3410d10538ff0d08d58270cc9"),
    "reports/figures/traffic_light.svg": (
        "b5fd5825ba172b4199ea5c526996d063455a2a6a198a4e8f46e24fa35ed3c23a"
    ),
    "reports/figures/sensitivity.svg": (
        "2a5979cf240dacaedd3a96777b113ac2ef2f6a3f9cc1702f91ff05ebe4b7f382"
    ),
    "reports/figures/oos_blocks.svg": (
        "e4d96237146ded2da29e150565bd810a9b0ee638d99fd21a3770198cf2d23394"
    ),
    "reports/figures/xi_blocks.svg": (
        "21fd325d4bee413ac6f9b78da484a3f5c90b473b586ae4541143a54c56fd805f"
    ),
    "reports/figures/descriptive_levels.svg": (
        "bde61ac2c9d26986b2fcc41896848bfcb957310192896fe1d6916f24cb2b6faf"
    ),
    "docs/results_review.md": ("b2d86d4f810de6c741744155dee8bdbd2fb0052d37c307978a1355559a8259b3"),
    "docs/figures.md": ("c82969e3d64acd6d387e95380707342a82abe9a0d461cb5510a1fbd91142e223"),
}
# D52 display names and colours (a test asserts equality with figures.py).
DISPLAY_NAMES = {
    "gaussian": "Gaussian",
    "hs": "HS",
    "pot_gpd": "POT-GPD",
    "garch_n": "GARCH-N",
    "fhs": "FHS",
    "garch_evt": "GARCH-EVT",
}
MODEL_COLOURS = {
    "gaussian": "#E69F00",
    "hs": "#56B4E9",
    "pot_gpd": "#009E73",
    "garch_n": "#0072B2",
    "fhs": "#D55E00",
    "garch_evt": "#CC79A7",
}
MODELS = tuple(DISPLAY_NAMES)
LEVELS = ("0.95", "0.99")
TEST_LEVEL = 0.05
BORDERLINE = (0.01, 0.10)  # D49 T6, closed interval
TESTS = ("Kupiec", "IND", "CC", "ES")
# D53 local figures: run dir and sha256 recorded in status.md (Day 12C-2; a test asserts it).
LOCAL_RUN_DIR = "reports/runs/53f989e13b86b036d05b0ba0a12f6a09cea33c6b33687784bdbc66194ea764bb"
LOCAL_SHA256 = {
    "figures/var_es_paths.svg": "8f3e34a5fb80e0edd085dd6c3d6e0a501497eccbe56b392d63fc1af8631329b8",
    "figures/xi_path.svg": "e9b2e0bd0578772e91677d255522c3ce603874b13c5b847dab1fc3cb0a4245b3",
    "figures/cum_fz0.svg": "82829741dd75ce3247ddcf7f561e02691e9e20b755c6b7e9376d1870a8cc15c6",
    "figures/traffic_light_calendar.svg": (
        "e9fc54cf319cb21a06b8d34cbce1a7f6c60999334397dead36e88e20f49f7e18"
    ),
    "index.md": "001451effbdd16bc7bc0e91354b435c6ff191190813b487584d2afad4c086364",
}
LOCAL_TITLES = {
    "figures/var_es_paths.svg": "G1: daily loss with VaR and ES at 0.99",
    "figures/xi_path.svg": "G2: rolling xi-hat with profile CIs",
    "figures/cum_fz0.svg": "G3: cumulative FZ0 difference over C",
    "figures/traffic_light_calendar.svg": "G4: traffic light by dated block",
}
BANNER = "Local only: contains daily series derived from NSE data. Never deploy or share."
# D58 (f): the next-day forecast (gitignored output of `forecast`), local mode only.
LIVE_REPORTS = "reports/live"
LIVE_LEVELS = ("0.95", "0.99", "0.995", "0.999")
FAR_TAIL = ("0.995", "0.999")
PASSED_ALL = ("fhs", "garch_evt")  # D46 H: on the price index only (D49 T3)
PASSED_NOTE = "passed every coverage test on the price index (D46 H); pass-dependent, borderline"
FAR_NOTE = "Far tail (descriptive, not backtestable)"
ZONES = ("green", "yellow", "red")
# D59: forecast.json / series.json schema 2, the arithmetic and the chart specs
SCHEMA_VERSION = 2
TAIL_KEYS = {"pot_gpd": ("u", "n_u"), "garch_evt": ("u_z", "n_u")}  # as live.TAIL_KEYS
EVT = ("pot_gpd", "garch_evt")
CONDITIONAL = ("garch_n", "fhs", "garch_evt")
RISK_CARDS = ("fhs", "garch_evt")
TIMES = "\u00d7"  # the multiplication sign after sigma_ratio
SOURCES = ("raw", "live", "api", "demo")  # api: EODHD rows (D60); demo: synthetic (D62)
PCT_CAPTION = "Model output is a log loss x; shown as 1 - e^-x of the position."
SELF_CHECK_TOL = 1e-12
TRADING_DAYS = 250  # "250 trading days a year"
BELOW_TAIL = "below the tail threshold: outside the EVT region"
DESCRIPTIVE = "descriptive, not backtestable"
SCENARIO = "scenario, not a forecast"
RETURN_N = (100, 250, 1000, 2500)
STALE_DAYS = 4
RANGES: dict[str, int | None] = {"1M": 31, "6M": 183, "1Y": 366, "4Y": 1461, "All": None}
CANDLE_MAX_DAYS = 183  # ranges <= 6M as candlesticks
CURVE_POINTS = 60
NSE_URL = "https://www.nseindia.com/"
NO_FORECAST = "Run scripts\\update.cmd to create today's forecast"
# D59 13E-4 display fixes (pins (30)-(44)); numbers unchanged, display only
SOURCE_LABELS = {
    "raw": "niftyindices snapshot",
    "live": "niftyindices file",
    "api": "EODHD API",
    "demo": "synthetic demo",
}
# D62 (b): the public demo (tracked, synthetic; written by `demo-data`)
DEMO_DIR = "app/demo"
DEMO_BANNER = (
    "Synthetic demo data, not the Nifty 50. Real forecasts run only on the owner's machine."
)
DEMO_SECTION = "Today (demo)"
DEMO_TITLE = "synthetic index"
REPLAY_HEADING = "Replay: the worst days of the series, if they happened tomorrow"
COMMITS_NOTE = (
    "Commit IDs refer to the development repository; the public copy is published as one "
    "commit per release (D65 (n))."
)
Y_PAD = 0.03  # M1 y domain: ~3% padding around the visible data
MAX_PERIOD = 100_000  # return periods above this many trading days are capped
CAPPED = "rarer than 1 in 100,000 trading days (beyond the reliable range)"
CAPPED_P = "< 0.001%"
ABOVE_END = "above the fitted tail's upper limit (xi < 0): outside what the model can say"
C3_FLOOR = 1e-6
C3_FLOOR_NOTE = "Below 1 in a million per day: beyond the reliable range."
REPLAY_NOTE = (
    "Judged from the last 1,000 trading days and, for GARCH-EVT, today's volatility. Crashes "
    "like these came after volatility had already risen; try the volatility shock above."
)
STRESS_NOTE = (
    "Scenario, not a forecast: tomorrow's volatility multiplied by k; VaR and ES scale around "
    "the forecast mean."
)
VOL_NOTE = f"Below 1{TIMES} = calmer than its 4-year average; above 1{TIMES} = more volatile."
T3_EXPANDER = "GARCH-EVT caveat (D49 T3, verbatim)"
NEXT = "next"
# D59 (s): colour follows the entity in every chart; text uses text colours only
PALETTE = {
    "dark": {
        "garch_evt": "#c98500",
        "fhs": "#3987e5",
        "pot_gpd": "#d55181",
        "neutral": "#929aa5",
        "hit": "#f6465d",
        "up": "#0ecb81",
        "down": "#f6465d",
        "text": "#eaecef",
        "muted": "#929aa5",
        "grid": "#2b3139",
    },
    "light": {
        "garch_evt": "#b07800",
        "fhs": "#2a78d6",
        "pot_gpd": "#c2477a",
        "neutral": "#707a8a",
        "hit": "#f6465d",
        "up": "#0ecb81",
        "down": "#f6465d",
        "text": "#181a20",
        "muted": "#5e6673",
        "grid": "#eaecef",
    },
}
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
DECISIONS = (
    "D23",
    "D44",
    "D45",
    "D46",
    "D47",
    "D48",
    "D49",
    "D50",
    "D51",
    "D52",
    "D53",
    "D54",
    "D55",
)
# D65 (h): only commands that pass from the public copy (no data, any commit)
REPRODUCE = (
    "uv sync --locked",
    "uv run --locked python -m pytest",
    "uv run --locked python -m niftytail demo-data configs/base.toml --verify",
)
VERIFIED = (
    "Verified in the development repository with the owner's data (byte-identical reruns; "
    "docs/reproducibility.md). From this public copy: the tests and `demo-data --verify`."
)
# D65 (b), (g): the cells of T1 decided by a p-value in [0.01, 0.10] (D49 T6); a test
# rebuilds this line from the three backtest.json files and finds it in the README
# and in the report
T1_BORDERLINE = (
    "Borderline in parts (p in [0.01, 0.10]): GARCH-EVT vs POT-GPD at 0.99, Holm p 0.0109754 on "
    "PR W1000 and 0.0105477 on TRI W1000 (`dm.conditional[a=garch_evt, b=pot_gpd].p_holm`); "
    "IND for GARCH-EVT at 0.95 on TRI W1000, p 0.0887 "
    "(`coverage.0.95.garch_evt.christoffersen.p_ind`); HS vs POT-GPD at 0.95 on PR W2000, "
    "p 0.0620363 (`dm.unconditional[a=hs, b=pot_gpd].p`). Sources: "
    "`reports/oos/w1000/backtest.json`, `reports/oos/w2000/backtest.json`, "
    "`reports/oos/tri_w1000/backtest.json`."
)


class DashboardDataError(Exception):
    """A tracked input is unknown or differs from its docs/protected_hashes.md sha256."""


def read(root: Path, rel: str) -> bytes:
    """The bytes of a tracked artefact, after its sha256 check (D55 (b))."""
    if rel not in EXPECTED_SHA256:
        raise DashboardDataError(f"{rel}: not a protected artefact")
    data = (root / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256[rel]:
        raise DashboardDataError(f"{rel}: sha256 differs from docs/protected_hashes.md")
    return data


def _json(root: Path, rel: str) -> Any:
    return json.loads(read(root, rel))


def _backtests(root: Path) -> dict[str, Any]:
    return {pid: _json(root, f"{d}/backtest.json") for pid, _, d in PASSES}


def display(key: str) -> str:
    """D52 (1): "POT-GPD", or "POT-GPD q=0.86" for a sensitivity key."""
    base, _, q = key.partition("@")
    return DISPLAY_NAMES[base] + (f" q={q}" if q else "")


def _num(x: float | None) -> str:
    return "—" if x is None else format(x, ".6g")  # D44 (8): 6 significant digits


def p_text(p: float | None, reps: int | None) -> str:
    """6 significant digits; the MC floor 1/(B+1) as "≤1e-4"; "—" for no p."""
    if p is None:
        return "—"
    if reps is not None and p <= 1 / (reps + 1):
        mant, _, ex = f"{1 / (reps + 1):.0e}".partition("e")
        return f"≤{mant}e{int(ex)}"
    return _num(p)


def _borderline(p: float | None) -> bool:
    return p is not None and BORDERLINE[0] <= p <= BORDERLINE[1]


def _es_reps(root: Path) -> int:
    """ES bootstrap B from base.toml, whose sha256 must equal the published run config."""
    data = (root / CONFIG).read_bytes()
    man = _json(root, "reports/oos/w1000/manifest.json")
    if hashlib.sha256(data).hexdigest() != man["config"]["sha256"]:
        raise DashboardDataError("configs/base.toml differs from the published run config")
    return int(tomllib.loads(data.decode("utf-8"))["backtest"]["es_residual_bootstrap_reps"])


def _test(
    test: str, c: Mapping[str, Any], es_reps: int
) -> tuple[float | None, bool | None, int | None]:
    """(deciding p, rejected at 5% or None, MC reps or None); the D51 F2 rules."""
    if test == "Kupiec":
        return c["kupiec"]["p"], bool(c["kupiec"]["reject"]), None
    if test in ("IND", "CC"):
        ch = c["christoffersen"]
        p = ch["p_ind" if test == "IND" else "p_cc"]
        return p, p <= TEST_LEVEL, ch["reps"]
    p = c["es_residual"]["p"]
    return p, None if p is None else p <= TEST_LEVEL, es_reps


def headline(root: Path) -> dict[str, str]:
    """The D46 H and the D49 T1 and T3 paragraphs, verbatim from docs/results_review.md."""
    text = read(root, REVIEW).decode("utf-8")
    out = {}
    for tag in ("H", "T1", "T3"):
        m = re.search(rf"^\*\*{tag}\.\*\*.*?(?=\n\nSource:)", text, flags=re.M | re.S)
        if m is None:
            raise DashboardDataError(f"{REVIEW}: no {tag} paragraph")
        out[tag] = m.group(0)
    return out


def metrics(root: Path) -> list[dict[str, str]]:
    """Four metric cards from PR W1000 (values copied from the JSON)."""
    bt = _backtests(root)["pr_w1000"]
    cards = [
        {"label": "OOS days", "value": str(bt["n_rows"]), "help": "OOS days per pass (n_rows)"}
    ]
    for k in ("pot_gpd", "garch_n", "garch_evt"):
        c = bt["coverage"]["0.99"][k]
        cards.append(
            {
                "label": f"{display(k)} 99% breaches",
                "value": str(c["breaches"]),
                "help": f"{display(k)} breaches, VaR 0.99 (PR W1000); n = {c['n']}; "
                f"expected n x 0.01 = {_num(c['n'] * 0.01)}",
            }
        )
    return cards


def coverage_rows(root: Path, pid: str, level: str) -> list[dict[str, str]]:
    """Breaches, p_hat and the four coverage-test p-values per model, with D49 T6 labels."""
    bts = _backtests(root)
    es_reps = _es_reps(root)
    rows = []
    for k in MODELS:
        c = bts[pid]["coverage"][level][k]
        row = {"model": display(k), "breaches": str(c["breaches"]), "p_hat": _num(c["p_hat"])}
        rejected, borderline, dependent = [], [], []
        for test in TESTS:
            p, rej, reps = _test(test, c, es_reps)
            row[f"{test} p"] = p_text(p, reps)
            if rej:
                rejected.append(test)
            if _borderline(p):
                borderline.append(test)
            if len({_test(test, b["coverage"][level][k], es_reps)[1] for b in bts.values()}) > 1:
                dependent.append(test)
        row["rejected at 5%"] = ", ".join(rejected) or "none"
        row["borderline"] = ", ".join(borderline)
        row["pass-dependent"] = ", ".join(dependent)
        rows.append(row)
    return rows


def dm_rows(root: Path, pid: str) -> list[dict[str, str]]:
    """DM families (i) and (ii) with verdicts (unadjusted / Holm) and D49 T6 labels."""
    bts = _backtests(root)
    rows = []
    for fam, name in (("unconditional", "(i)"), ("conditional", "(ii) Holm")):
        for e in bts[pid]["dm"][fam]:
            deciding = e["p_holm"] if fam == "conditional" else e["p"]
            verdicts = {
                x["reject"]
                for b in bts.values()
                for x in b["dm"][fam]
                if (x["a"], x["b"], x["level"]) == (e["a"], e["b"], e["level"])
            }
            rows.append(
                {
                    "family": name,
                    "level": e["level"],
                    "A": display(e["a"]),
                    "B": display(e["b"]),
                    "DM": _num(e["dm"]),
                    "d_bar": _num(e["d_bar"]),
                    "p": _num(e["p"]),
                    "Holm p": _num(e.get("p_holm")),
                    "verdict": "reject" if e["reject"] else "not rejected",
                    "borderline": "borderline" if _borderline(deciding) else "",
                    "pass-dependent": "pass-dependent" if len(verdicts) > 1 else "",
                }
            )
    return rows


def sensitivity_rows(root: Path, level: str) -> list[dict[str, str]]:
    """PR W1000, PR W2000 and TRI W1000 side by side per model at one level."""
    bts = _backtests(root)
    es_reps = _es_reps(root)
    rows = []
    for k in MODELS:
        row = {"model": display(k)}
        for pid, label, _ in PASSES:
            c = bts[pid]["coverage"][level][k]
            row[f"{label}: breaches"] = str(c["breaches"])
            row[f"{label}: Kupiec p"] = p_text(c["kupiec"]["p"], None)
            row[f"{label}: ES p"] = p_text(c["es_residual"]["p"], es_reps)
            row[f"{label}: mean FZ0"] = _num(bts[pid]["fz0"][level][k]["mean"])
        rows.append(row)
    return rows


def svg_data_uri(root: Path, fid: str) -> str:
    """A tracked figure as a ``data:`` URI (no remote URL)."""
    return "data:image/svg+xml;base64," + base64.b64encode(read(root, FIGURES[fid])).decode()


def provenance(root: Path) -> dict[str, Any]:
    """Run and code commits, input sha256, decision IDs and reproduction commands."""
    commits = {label: _json(root, f"{d}/manifest.json")["git"]["commit"] for _, label, d in PASSES}
    commits["traffic_light_blocks.json (code)"] = _json(root, TL)["provenance"]["commit"]
    commits["xi_path summary (code)"] = _json(root, XI)["provenance"]["commit"]
    m = re.search(r"^- Code commit: `([0-9a-f]{40})`", read(root, FIGURES_DOC).decode(), re.M)
    commits["figures (code)"] = m.group(1) if m else "—"
    return {
        "commits": commits,
        "inputs": dict(sorted(EXPECTED_SHA256.items())),
        "decisions": list(DECISIONS),
        "reproduce": list(REPRODUCE),
    }


def is_public(root: Path, environ: Mapping[str, str] | None = None) -> bool:
    """Public mode: NIFTYTAIL_PUBLIC=1, or no local run dir (D55 (e))."""
    env = os.environ if environ is None else environ
    return env.get("NIFTYTAIL_PUBLIC") == "1" or not (root / LOCAL_RUN_DIR).is_dir()


def local_mode(
    environ: Mapping[str, str] | None = None, secrets: Mapping[str, Any] | None = None
) -> bool:
    """Local mode: NIFTYTAIL_PUBLIC is "1" neither in the environment (D58 (13)) nor as a
    top-level key of the app's secrets (D62 (b), Streamlit Community Cloud)."""
    env = os.environ if environ is None else environ
    sec = {} if secrets is None else secrets
    return env.get("NIFTYTAIL_PUBLIC") != "1" and str(sec.get("NIFTYTAIL_PUBLIC", "")) != "1"


def demo_bundle(root: Path) -> dict[str, Any] | None:
    """D62 (b): the tracked synthetic demo in the live-bundle shape; None if invalid."""
    doc, ser = _load(root / DEMO_DIR / "forecast.json"), _load(root / DEMO_DIR / "series.json")
    if not (_valid(doc) and _valid_series(ser)):
        return None
    return {"doc": doc, "series": ser, "check": None, "commit": "synthetic demo", "warning": None}


def local_figures(
    root: Path, environ: Mapping[str, str] | None = None
) -> list[dict[str, str]] | None:
    """G1-G4 as data URIs in local mode if every recorded sha256 matches, else None."""
    if is_public(root, environ):
        return None  # public mode: no run-dir access beyond the existence test
    base = root / LOCAL_RUN_DIR
    blobs = {}
    for rel, want in LOCAL_SHA256.items():
        path = base / rel
        if not path.is_file():
            return None
        blobs[rel] = path.read_bytes()
        if hashlib.sha256(blobs[rel]).hexdigest() != want:
            return None
    return [
        {
            "title": title,
            "uri": "data:image/svg+xml;base64," + base64.b64encode(blobs[rel]).decode(),
        }
        for rel, title in LOCAL_TITLES.items()
    ]


# ---------------------------------------------------------------------------
# D58 (e), (f): next-day forecast (local mode only) and the rupee view; arithmetic only
# ---------------------------------------------------------------------------
def _value(v: Any) -> float:
    """A forecast.json value: a number, "inf" (unbounded ES) or None (missing)."""
    if v is None:
        return math.nan
    return math.inf if v == "inf" else float(v)


def rupee_loss(position: float, loss: float) -> float:
    """``V (1 - e^-x)`` for a long position V at log loss x: exact for VaR (monotone), slightly
    conservative for ES (1 - e^-x is concave). NaN stays NaN, +inf stays +inf ("unbounded")."""
    if math.isnan(loss) or math.isinf(loss):
        return loss
    return position * -math.expm1(-loss)


def budget_position(budget: float, es99: float) -> float:
    """The largest V with ``V (1 - e^-ES_0.99) <= B``; NaN if ES is missing, infinite or <= 0."""
    if not math.isfinite(es99) or es99 <= 0:
        return math.nan
    return budget / -math.expm1(-es99)


def inr(x: float) -> str:
    """Whole rupees with Indian digit grouping (₹12,34,567); "—" for NaN, "unbounded" for inf."""
    if math.isnan(x):
        return "—"
    if math.isinf(x):
        return "unbounded"
    n = round(abs(x))
    head, tail = str(n)[:-3], str(n)[-3:]
    groups: list[str] = []
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ("-" if x < 0 and n else "") + "₹" + ",".join([*groups, tail])


def pct_of_position(x: float) -> float:
    """D59 (f): the log loss x as a share of the position, ``100 (1 - e^-x)``; NaN/inf pass."""
    if math.isnan(x) or math.isinf(x):
        return x
    return 100 * -math.expm1(-x)


def pct(x: float) -> str:
    """% of position with 2 decimals (so % x position = ₹ up to rounding); "—", "unbounded"."""
    v = pct_of_position(x)
    if math.isnan(v):
        return "—"
    return "unbounded" if math.isinf(v) else f"{v:.2f}%"


def _number(v: Any) -> bool:
    return v is None or v == "inf" or (isinstance(v, int | float) and not isinstance(v, bool))


def _real(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _valid(doc: Any) -> bool:
    """Schema check of a forecast.json (D58 (f), schema 2 of D59 (b)); anything else is ignored."""
    try:
        ok = doc["schema_version"] == SCHEMA_VERSION and bool(_ISO.fullmatch(doc["as_of"]))
        ok = ok and set(doc["models"]) == set(MODELS) == set(doc["health"]["models"])
        ok = ok and isinstance(doc["config"]["window"], int) and _real(doc["config"]["xi_zero_tol"])
        ok = ok and isinstance(doc["provenance"]["commit"], str)
        for k in MODELS:
            m, h = doc["models"][k], doc["health"]["models"][k]
            for part in ("var", "es"):
                ok = ok and set(m[part]) == set(LIVE_LEVELS) and all(map(_number, m[part].values()))
            ok = ok and isinstance(m["flags"], list) and isinstance(h["breaches"], int)
            ok = ok and isinstance(h["n_obs"], int) and h["zone"] in (*ZONES, None)
            names = [*TAIL_KEYS.get(k, ())]
            names += ["mu_next", "sigma_next"] * (k in CONDITIONAL) + ["xi", "beta"] * (k in EVT)
            ok = ok and all(_number({**m, **m.get("tail", {})}[n]) for n in names)
        return ok and _number(doc["volatility"]["sigma_ratio"])
    except (KeyError, TypeError):
        return False


def _valid_series(ser: Any) -> bool:
    """Shape check of a series.json (D59 (c))."""
    try:
        ok = ser["schema_version"] == SCHEMA_VERSION and len(ser["history"]) > 0
        ok = ok and all(
            _ISO.fullmatch(h["date"]) and h["source"] in SOURCES and _number(h["close"])
            for h in ser["history"]
        )
        ok = ok and all(map(_ISO.fullmatch, (ser["window"]["first"], ser["window"]["last"])))
        ok = ok and all(_real(v) for v in ser["tail_sample"])
        ok = ok and all(_real(r["loss"]) and _ISO.fullmatch(r["date"]) for r in ser["recent"])
        return ok and all(_real(w["loss"]) and _ISO.fullmatch(w["date"]) for w in ser["worst"])
    except (KeyError, TypeError):
        return False


def _valid_check(chk: Any) -> bool:
    try:
        return (
            isinstance(chk["identical"], int)
            and isinstance(chk["n"], int)
            and all(isinstance(chk[k], str) for k in ("run", "commit"))
        )
    except (KeyError, TypeError):
        return False


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def live_bundle(root: Path, environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """D59 (o): the valid v2 forecast (with its valid series.json) with the latest as-of date,
    the check.json beside it (or None), its commit (12 hex) and the disagreement warning.

    Local mode only; first in path order on a tie (D58 (13)). In public mode reports/live is
    never touched."""
    if not local_mode(environ):
        return None
    found = []
    for path in sorted((root / LIVE_REPORTS).glob("*/forecast.json")):
        doc, ser = _load(path), _load(path.parent / "series.json")
        if _valid(doc) and _valid_series(ser):
            found.append((path.parent, doc, ser))
    if not found:
        return None
    latest = max(doc["as_of"] for _, doc, _ in found)
    same = [f for f in found if f[1]["as_of"] == latest]
    where, doc, ser = same[0]
    commit = doc["provenance"]["commit"][:12]
    numbers = {json.dumps({k: v for k, v in d.items() if k != "provenance"}) for _, d, _ in same}
    warning = (
        f"{len(same)} forecasts for this date disagree; showing commit {commit}; "
        "remove stale reports/live dirs"
        if len(numbers) > 1
        else None
    )
    chk = _load(where / "check.json")
    return {
        "doc": doc,
        "series": ser,
        "check": chk if _valid_check(chk) else None,
        "commit": commit,
        "warning": warning,
    }


def live_forecast(root: Path, environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The forecast.json of :func:`live_bundle`, or None."""
    bundle = live_bundle(root, environ)
    return None if bundle is None else bundle["doc"]


def forecast_rows(doc: Mapping[str, Any], position: float, levels: tuple[str, ...]) -> list[Any]:
    """VaR and ES per model and level, in % of position and in rupees for a long position."""
    rows = []
    for k in MODELS:
        m = doc["models"][k]
        for lvl in levels:
            var, es = _value(m["var"][lvl]), _value(m["es"][lvl])
            rows.append(
                {
                    "model": display(k),
                    "level": lvl,
                    "VaR (% of position)": pct(var),
                    "VaR ₹": inr(rupee_loss(position, var)),
                    "ES (% of position)": pct(es),
                    "ES ₹ (slightly conservative)": inr(rupee_loss(position, es)),
                    "note": PASSED_NOTE if k in PASSED_ALL else "",
                }
            )
    return rows


def budget_rows(doc: Mapping[str, Any], budget: float) -> list[dict[str, str]]:
    """Per model: the largest long position whose 99% ES in rupees stays within the budget."""
    return [
        {
            "model": display(k),
            "largest position (99% ES ₹ ≤ budget)": inr(
                budget_position(budget, _value(doc["models"][k]["es"]["0.99"]))
            ),
        }
        for k in MODELS
    ]


def health_rows(doc: Mapping[str, Any]) -> list[dict[str, str]]:
    """Breaches at the traffic-light level over the last block of targets, and the zone."""
    h = doc["health"]
    return [
        {
            "model": display(k),
            f"breaches at {h['level']}": f"{v['breaches']} of {v['n_obs']}",
            "zone": v["zone"] or "—",
        }
        for k in MODELS
        for v in [h["models"][k]]
    ]


def volatility_ratio(doc: Mapping[str, Any]) -> str:
    """sigma_next / the window's sample std, 2 decimals and the times sign (D59 (f)); "—"."""
    r = _value(doc["volatility"]["sigma_ratio"])
    return "—" if not math.isfinite(r) else f"{r:.2f}{TIMES}"


def risk_card(doc: Mapping[str, Any], key: str, position: float) -> dict[str, str]:
    """99% VaR and ES of one model, in rupees and % of position (stored values)."""
    m = doc["models"][key]
    var, es = _value(m["var"]["0.99"]), _value(m["es"]["0.99"])
    return {
        "model": display(key),
        "VaR ₹": inr(rupee_loss(position, var)),
        "VaR %": pct(var),
        "ES ₹": inr(rupee_loss(position, es)),
        "ES %": pct(es),
    }


# ---------------------------------------------------------------------------
# D59 (g)-(l): closed-form arithmetic on the stored fit (never replaces a stored value)
# ---------------------------------------------------------------------------
def _fit(doc: Mapping[str, Any], key: str) -> dict[str, float] | None:
    """u, n_u, xi, beta (residual scale for GARCH-EVT), mu, sigma, W, tol; None if missing."""
    m = doc["models"][key]
    fit = {
        "u": m["tail"][TAIL_KEYS[key][0]],
        "n_u": m["tail"]["n_u"],
        "xi": m["xi"],
        "beta": m["beta"],
        "mu": m.get("mu_next", 0.0),
        "sigma": m.get("sigma_next", 1.0),
    }
    if not all(_real(v) and math.isfinite(v) for v in fit.values()):
        return None
    if fit["beta"] <= 0 or fit["sigma"] <= 0 or fit["n_u"] < 1:
        return None
    return fit | {"w": doc["config"]["window"], "tol": doc["config"]["xi_zero_tol"]}


def _level_p(level: str) -> tuple[int, int]:
    """p = 1 - c exactly, as (numerator, denominator), from the decimal text of c."""
    digits = level.split(".")[1]
    den = 10 ** len(digits)
    return den - int(digits), den


def tail_risk(doc: Mapping[str, Any], key: str, num: int, den: int) -> tuple[float, float] | None:
    """gpd.tail_var / tail_es mirrored at p = num/den (D59 (g)), scaled by mu_next + sigma_next
    for GARCH-EVT; None if the fit is missing or p >= n_u / W.

    ``r = (W / n_u) p`` is one correctly rounded integer division, as gpd's exact rational."""
    f = _fit(doc, key)
    if f is None:
        return None
    w, n_u = int(f["w"]), int(f["n_u"])
    if not num * w < n_u * den:
        return None
    u, xi, beta = f["u"], f["xi"], f["beta"]
    log_r = math.log(w * num / (n_u * den))
    exponential = abs(xi) < f["tol"]
    var = u - beta * log_r if exponential else u + (beta / xi) * math.expm1(-xi * log_r)
    es = math.inf if xi >= 1 else (var + (beta - xi * u)) / (1 - xi)
    return f["mu"] + f["sigma"] * var, f["mu"] + f["sigma"] * es


def _rel(stored: float, mirror: float) -> float:
    if math.isinf(stored) and stored == mirror:
        return 0.0
    if not (math.isfinite(stored) and math.isfinite(mirror)):
        return math.inf
    return abs(stored - mirror) / (abs(stored) or 1.0)


def self_check(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute VaR and ES at the four levels of POT-GPD and GARCH-EVT from the tail block;
    ✓ iff every relative error <= 1e-12 (a level the stored forecast leaves missing must also
    be missing in the mirror)."""
    worst = 0.0
    for k in EVT:
        for lvl in LIVE_LEVELS:
            stored = (_value(doc["models"][k]["var"][lvl]), _value(doc["models"][k]["es"][lvl]))
            mirror = tail_risk(doc, k, *_level_p(lvl))
            if mirror is None:
                err = 0.0 if all(map(math.isnan, stored)) else math.inf
            else:
                err = max(_rel(s, x) for s, x in zip(stored, mirror, strict=True))
            worst = max(worst, err)
    ok = worst <= SELF_CHECK_TOL
    return {"ok": ok, "max_error": worst, "symbol": "✓" if ok else f"✗ (max error {worst:.3g})"}


def _loss(fall: float) -> float:
    """The log loss of a fall f (a fraction of the position): ``-log(1 - f)``."""
    return -math.log1p(-fall)


def _exceed(doc: Mapping[str, Any], key: str, x: float) -> float | None:
    f = _fit(doc, key)
    if f is None:
        return None
    y = (x - f["mu"]) / f["sigma"] - f["u"]
    if y < 0:
        return None  # below the tail threshold: outside the EVT region
    zeta, xi, beta = f["n_u"] / f["w"], f["xi"], f["beta"]
    if abs(xi) < f["tol"]:
        return zeta * math.exp(-y / beta)
    t = 1 + xi * y / beta
    return 0.0 if t <= 0 else zeta * t ** (-1 / xi)


def exceed_prob(doc: Mapping[str, Any], key: str, fall: float) -> float | None:
    """D59 (h): P(one-day fall >= f) under the stored fit; GARCH-EVT given today's volatility.
    None below the threshold; 0 beyond the upper end of a fit with xi < 0."""
    return _exceed(doc, key, _loss(fall))


def period_text(p: float | None) -> tuple[str, str]:
    """("1 in N trading days", "about every Y years") at 250 trading days a year; capped above
    100,000 trading days (no year figure)."""
    if p is None:
        return BELOW_TAIL, "—"
    if p == 0:
        return ABOVE_END, "—"
    days, years = 1 / p, 1 / (TRADING_DAYS * p)
    if days > MAX_PERIOD:
        return CAPPED, "—"
    return f"1 in {days:,.0f} trading days", f"about every {years:,.1f} years"


def prob_text(p: float | None) -> str:
    """P per day as % (3 significant digits); "< 0.001%" where the period is capped."""
    if p is None or p == 0:
        return "—"
    return CAPPED_P if 1 / p > MAX_PERIOD else f"{100 * p:.3g}%"


def return_level(doc: Mapping[str, Any], key: str, n: int) -> float | None:
    """D59 (i): the log loss exceeded on 1 day in n (p = 1/n); None if p >= n_u / W."""
    r = tail_risk(doc, key, 1, n)
    return None if r is None else r[0]


def return_level_rows(doc: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for n in RETURN_N:
        row = {"once in": f"{n:,} trading days", "level": f"{1 - 1 / n:.4g}"}
        for k in EVT:
            x = return_level(doc, k, n)
            row[display(k)] = "—" if x is None else pct(x)
        row["note"] = DESCRIPTIVE if n > 100 else ""
        rows.append(row)
    return rows


def history_count(doc: Mapping[str, Any], ser: Mapping[str, Any], fall: float) -> int | None:
    """D59 (j): window losses >= the fall's log loss (only at or above POT-GPD's u)."""
    x = _loss(fall)
    u = _value(doc["models"]["pot_gpd"]["tail"]["u"])
    if not x >= u:
        return None
    return sum(1 for v in ser["tail_sample"] if v >= x)


def crash_rows(doc: Mapping[str, Any], fall: float) -> list[dict[str, str]]:
    """Per EVT model: P(fall >= f), "1 in N trading days", "about every Y years"."""
    rows = []
    for k in EVT:
        p = exceed_prob(doc, k, fall)
        days, years = period_text(p)
        rows.append(
            {
                "model": display(k) + (" (given today's volatility)" if k == "garch_evt" else ""),
                "probability per day": prob_text(p),
                "how often": days,
                "in years": years,
            }
        )
    return rows


def vol_shock(doc: Mapping[str, Any], key: str, k: float) -> dict[str, tuple[float, float]]:
    """D59 (k), conditional models only: mu_next + k (VaR - mu_next), the same for ES, per level.
    A scenario, not a forecast; k = 1 returns the stored values."""
    if key not in CONDITIONAL:
        raise DashboardDataError(f"{key}: not a conditional model")
    m = doc["models"][key]
    mu = _value(m["mu_next"])
    out = {}
    for lvl in LIVE_LEVELS:
        var, es = _value(m["var"][lvl]), _value(m["es"][lvl])
        out[lvl] = (var, es) if k == 1 else (mu + k * (var - mu), mu + k * (es - mu))
    return out


def stress_rows(doc: Mapping[str, Any], k: float, position: float) -> list[dict[str, str]]:
    rows = []
    for key in CONDITIONAL:
        var, es = vol_shock(doc, key, k)["0.99"]
        rows.append(
            {
                "model": display(key),
                "99% VaR": pct(var),
                "99% VaR ₹": inr(rupee_loss(position, var)),
                "99% ES": pct(es),
                "99% ES ₹": inr(rupee_loss(position, es)),
            }
        )
    return rows


def replay_rows(
    doc: Mapping[str, Any], ser: Mapping[str, Any], position: float
) -> list[dict[str, str]]:
    """D59 (l): each worst day, its fall, ₹ on the position and today's return periods."""
    rows = []
    for w in ser["worst"]:
        row = {
            "date": w["date"],
            "fall": pct(w["loss"]),
            "₹ on the position": inr(rupee_loss(position, w["loss"])),
        }
        for k in EVT:
            row[display(k)] = period_text(_exceed(doc, k, w["loss"]))[0]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# D59 (m)-(n): market view of the owner's data and freshness
# ---------------------------------------------------------------------------
def level_text(v: Any) -> str:
    return "—" if not _real(v) else f"{v:,.2f}"


def _signed(c: float) -> str:
    return "—" if math.isnan(c) else f"{100 * c:+.2f}%"


def market(ser: Mapping[str, Any], rng: str) -> dict[str, Any]:
    """D59 (m): the sessions of a range (calendar days back from the last session) with the
    change vs the previous close and the drawdown from the running peak within the range;
    the change over the range, the worst drawdown and the last session."""
    hist = ser["history"]
    prev = [math.nan] + [h["close"] for h in hist[:-1]]
    days = RANGES[rng]
    last = date.fromisoformat(hist[-1]["date"])
    start = "" if days is None else (last - timedelta(days=days)).isoformat()
    rows, peak = [], -math.inf
    for h, p in zip(hist, prev, strict=True):
        if h["date"] < start:
            continue
        peak = max(peak, h["close"])
        rows.append(
            h
            | {
                "change": h["close"] / p - 1 if _real(p) else math.nan,
                "drawdown": h["close"] / peak - 1,
            }
        )
    return {
        "rows": rows,
        "change": rows[-1]["close"] / rows[0]["close"] - 1,
        "worst_drawdown": min(r["drawdown"] for r in rows),
        "last": rows[-1],
    }


def kpis(ser: Mapping[str, Any], rng: str) -> list[dict[str, str]]:
    mk = market(ser, rng)
    return [
        {"label": "Last close", "value": level_text(mk["last"]["close"])},
        {"label": change_label(rng), "value": _signed(mk["change"])},
        {"label": f"Worst drawdown in {rng}", "value": _signed(mk["worst_drawdown"])},
        {"label": "Source of the last session", "value": SOURCE_LABELS[mk["last"]["source"]]},
    ]


def last_session(ser: Mapping[str, Any]) -> dict[str, str]:
    h = market(ser, "1M")["last"]
    return {
        "date": h["date"],
        **{k: level_text(h[k]) for k in ("open", "high", "low", "close")},
        "change vs the previous close": _signed(h["change"]),
    }


def freshness(
    doc: Mapping[str, Any], ser: Mapping[str, Any], today: date | None = None
) -> dict[str, Any]:
    """D59 (n): "Data through <as_of>", the source of the last session, and a warning when
    as_of is more than 4 calendar days before today."""
    now = date.today() if today is None else today
    age = (now - date.fromisoformat(doc["as_of"])).days
    warning = (
        f"The newest session is {age} calendar days old: run scripts\\update.cmd (after "
        "16:30) or check the scheduled task."
        if age > STALE_DAYS
        else None
    )
    return {
        "text": f"Data through {doc['as_of']}",
        "source": SOURCE_LABELS[ser["history"][-1]["source"]],
        "warning": warning,
    }


def pipeline_numbers(doc: Mapping[str, Any]) -> dict[str, str]:
    """Today's stored fit values for "How it works", 6 significant digits."""
    pot, evt = doc["models"]["pot_gpd"], doc["models"]["garch_evt"]
    vals = {
        "W": doc["config"]["window"],
        "q": doc["config"]["q_selected"],
        "u": pot["tail"]["u"],
        "n_u": pot["tail"]["n_u"],
        "xi": pot["xi"],
        "beta": pot["beta"],
        "u_z": evt["tail"]["u_z"],
        "n_u_z": evt["tail"]["n_u"],
        "xi_z": evt["xi"],
        "beta_z": evt["beta"],
        "mu_next": evt["mu_next"],
        "sigma_next": evt["sigma_next"],
    }
    return {k: v if isinstance(v, str) else tex_num(v) for k, v in vals.items()}


def tex_num(v: Any) -> str:
    """A number for st.latex: 6 significant digits, 4.96 \\times 10^{-5} style for small or
    large values (never 4.9622e-05); "—" for a missing value."""
    if not _real(v):
        return "—"
    s = format(float(v), ".6g")
    if "e" not in s:
        return s
    e = math.floor(math.log10(abs(float(v))))
    return f"{float(v) / 10**e:.3g} \\times 10^{{{e}}}"


# ---------------------------------------------------------------------------
# D59 (p), (s): Vega-Lite specs for st.vega_lite_chart; colours follow the entity
# ---------------------------------------------------------------------------
def _config(mode: str) -> dict[str, Any]:
    p = PALETTE[mode]
    axis = {
        "labelColor": p["muted"],
        "titleColor": p["text"],
        "gridColor": p["grid"],
        "domainColor": p["grid"],
        "tickColor": p["grid"],
        "labelFont": "Inter",
        "titleFont": "Inter",
    }
    return {
        "background": "transparent",
        "font": "Inter",
        "axis": axis,
        "legend": {"labelColor": p["text"], "titleColor": p["text"], "orient": "bottom"},
        "view": {"stroke": None},
    }


def _colour(names: list[str], colours: list[str]) -> dict[str, Any]:
    return {"scale": {"domain": names, "range": colours}, "legend": {"title": None}}


def _spec(
    mode: str, layers: list[dict[str, Any]], title: str | None = None, **extra: Any
) -> dict[str, Any]:
    out = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "config": _config(mode),
        "layer": layers,
        **extra,
    }
    if title is not None:  # D62 (b): the demo charts say "synthetic index"
        out["title"] = {"text": title, "color": PALETTE[mode]["text"], "anchor": "start"}
    return out


def _tip(*fields: tuple[str, str]) -> list[dict[str, str]]:
    return [{"field": f, "title": t, "type": "nominal"} for f, t in fields]


def _no_axis(enc: dict[str, Any]) -> dict[str, Any]:
    """The same channel without its own axis: one axis per chart (D59 pin (31))."""
    return enc | {"axis": None}


def y_domain(values: list[float], log_scale: bool) -> list[float]:
    """D59 pin (30): the visible range with ~3% padding (multiplicative on a log scale)."""
    lo, hi = min(values), max(values)
    if log_scale:
        r = (hi / lo) ** Y_PAD if hi > lo else 1 + Y_PAD
        return [lo / r, hi * r]
    pad = Y_PAD * (hi - lo) if hi > lo else Y_PAD * abs(hi)
    return [lo - pad, hi + pad]


# D61 (c): month-and-year labels on the time axes of M1 and M2 (day, month, year for 1M)
TICKS: dict[str, dict[str, Any] | None] = {
    "1M": None,
    "6M": {"interval": "month", "step": 1},
    "1Y": {"interval": "month", "step": 2},
    "4Y": {"interval": "month", "step": 6},
    "All": {"interval": "year", "step": 2},
}


def time_axis(rng: str) -> dict[str, Any]:
    """The x axis of M1 and M2: no gridlines, labels with the year ("Apr 2024")."""
    axis: dict[str, Any] = {
        "grid": False,
        "title": None,
        "format": "%d %b %Y" if rng == "1M" else "%b %Y",
        "labelOverlap": "greedy",
    }
    if TICKS[rng] is not None:
        axis["tickCount"] = TICKS[rng]
    return axis


def change_label(rng: str) -> str:
    """D61 (d): the change KPI; no date on screen besides as_of."""
    return "Change since the first session" if rng == "All" else f"Change over {rng}"


def window_caption(doc: Mapping[str, Any], rng: str) -> str | None:
    """D59 pin (32): shading only in "All"; other ranges lie inside the window."""
    if rng == "All":
        return None
    return f"This range lies inside the model's window (last {doc['config']['window']:,} days)"


def market_spec(
    doc: Mapping[str, Any],
    ser: Mapping[str, Any],
    rng: str,
    log_scale: bool,
    mode: str,
    title: str | None = None,
) -> dict[str, Any]:
    """M1: candlesticks for ranges <= 6M, else the close line; the estimation window shaded;
    the worst days as red down-triangles."""
    p, mk = PALETTE[mode], market(ser, rng)
    rows = [
        {
            "date": r["date"],
            "date_t": r["date"],  # tooltip text: "date" itself is parsed as temporal
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "dir": "Up" if not _real(r["open"]) or r["close"] >= r["open"] else "Down",
            **{f"{k}_t": level_text(r[k]) for k in ("open", "high", "low", "close")},
            "change_t": _signed(r["change"]),
            "drawdown_t": _signed(r["drawdown"]),
            "source": SOURCE_LABELS[r["source"]],
        }
        for r in mk["rows"]
    ]
    first, last = rows[0]["date"], rows[-1]["date"]
    days = RANGES[rng]
    candles = days is not None and days <= CANDLE_MAX_DAYS
    shade = rng == "All"
    win = f"Model's window (last {doc['config']['window']:,} days)"
    names = [*(["Up", "Down"] if candles else ["Close"]), *([win] if shade else []), "Worst days"]
    colours = [
        *([p["up"], p["down"]] if candles else [p["neutral"]]),
        *([p["garch_evt"]] if shade else []),
        p["hit"],
    ]
    col = _colour(names, colours)
    x = {
        "field": "date",
        "type": "temporal",
        "title": None,
        "scale": {"domain": [first, last]},
        "axis": time_axis(rng),
    }
    keys = ("low", "high") if candles else ("close",)
    visible = [r[k] for r in rows for k in keys if _real(r[k])]
    y = {
        "type": "log" if log_scale else "linear",
        "zero": False,
        "nice": False,
        "domain": y_domain(visible, log_scale),
    }
    tip = _tip(
        ("date_t", "Date"),
        *((("open_t", "Open"), ("high_t", "High"), ("low_t", "Low")) if candles else ()),
        ("close_t", "Close"),
        ("change_t", "Change vs previous close"),
        ("drawdown_t", "Drawdown"),
        ("source", "Source"),
    )
    layers: list[dict[str, Any]] = []
    if candles:
        size = max(1, min(10, 600 // len(rows)))
        enc = {"x": x, "color": {"field": "dir", **col}, "tooltip": tip}
        layers.append(
            {
                "data": {"values": rows},
                "mark": "rule",
                "encoding": enc
                | {
                    "y": {
                        "field": "low",
                        "type": "quantitative",
                        "scale": y,
                        "title": "Index level",
                    },
                    "y2": {"field": "high"},
                },
            }
        )
        layers.append(
            {
                "data": {"values": rows},
                "mark": {"type": "bar", "size": size},
                "encoding": enc
                | {
                    "x": _no_axis(x),
                    "y": {"field": "open", "type": "quantitative", "scale": y, "axis": None},
                    "y2": {"field": "close"},
                },
            }
        )
    else:
        layers.append(
            {
                "data": {"values": rows},
                "mark": {"type": "line", "strokeWidth": 1.5},
                "encoding": {
                    "x": x,
                    "y": {
                        "field": "close",
                        "type": "quantitative",
                        "scale": y,
                        "title": "Index level",
                    },
                    "color": {"datum": "Close", **col},
                    "tooltip": tip,
                },
            }
        )
    close = {r["date"]: r["close"] for r in rows}
    worst = [
        {"date": w["date"], "date_t": w["date"], "close": close[w["date"]], "fall": pct(w["loss"])}
        for w in ser["worst"]
        if w["date"] in close
    ]
    if worst:
        layers.append(
            {
                "data": {"values": worst},
                "mark": {"type": "point", "shape": "triangle-down", "filled": True, "size": 140},
                "encoding": {
                    "x": _no_axis(x),
                    "y": {"field": "close", "type": "quantitative", "scale": y, "axis": None},
                    "color": {"datum": "Worst days", **col},
                    "tooltip": _tip(("date_t", "Date"), ("fall", "Fall")),
                },
            }
        )
    w0, w1 = max(ser["window"]["first"], first), ser["window"]["last"]
    if shade and w0 <= w1:  # drawn first, under the data
        layers.insert(
            0,
            {
                "data": {"values": [{"date": w0, "end": w1, "label": win}]},
                "mark": {"type": "rect", "opacity": 0.13},
                "encoding": {
                    "x": _no_axis(x),
                    "x2": {"field": "end"},
                    "color": {"datum": win, **col},
                    "tooltip": _tip(("label", "Shaded")),
                },
            },
        )
    return _spec(mode, layers, title, height=360)


def drawdown_spec(
    ser: Mapping[str, Any], rng: str, mode: str, title: str | None = None
) -> dict[str, Any]:
    """M2: the drawdown from the running peak, same x range as M1, a red area at 10%."""
    p, mk = PALETTE[mode], market(ser, rng)
    rows = [
        {
            "date": r["date"],
            "date_t": r["date"],
            "dd": 100 * r["drawdown"],
            "dd_t": _signed(r["drawdown"]),
        }
        for r in mk["rows"]
    ]
    x: dict[str, Any] = {"field": "date", "type": "temporal", "title": None}
    x["scale"] = {"domain": [rows[0]["date"], rows[-1]["date"]]}
    x["axis"] = time_axis(rng)
    layer = {
        "data": {"values": rows},
        "mark": {"type": "area", "color": p["hit"], "opacity": 0.10, "line": {"color": p["hit"]}},
        "encoding": {
            "x": x,
            "y": {"field": "dd", "type": "quantitative", "title": "Drawdown (%)"},
            "tooltip": _tip(("date_t", "Date"), ("dd_t", "Drawdown")),
        },
    }
    return _spec(mode, [layer], title, height=140)


def recent_spec(
    doc: Mapping[str, Any], ser: Mapping[str, Any], mode: str, title: str | None = None
) -> dict[str, Any]:
    """C2: the recent losses with the 0.99 VaR of FHS and GARCH-EVT, breaches as red crosses
    and the next day's VaR as end markers."""
    p = PALETTE[mode]
    names = ["Daily loss", "FHS 99% VaR", "GARCH-EVT 99% VaR", "Breach"]
    col = _colour(names, [p["neutral"], p["fhs"], p["garch_evt"], p["hit"]])
    rows = []
    for r in ser["recent"]:
        row = {
            "x": r["date"],
            "date": r["date"],
            "loss": pct_of_position(r["loss"]),
            "loss_t": pct(r["loss"]),
        }
        for k in RISK_CARDS:
            v = _value(r[k]["var"])
            row[k] = pct_of_position(v) if math.isfinite(v) else None
            row[f"{k}_t"] = pct(v)
            row[f"{k}_hit"] = r[k]["hit"]
        rows.append(row)
    x = {
        "field": "x",
        "type": "ordinal",
        "title": None,
        "sort": None,
        "scale": {"domain": [r["date"] for r in ser["recent"]] + [NEXT]},
        "axis": {"grid": False, "labelOverlap": "greedy", "labelAngle": 0},
    }
    y = {"type": "quantitative", "title": "One-day loss (% of position)"}
    quiet = {"type": "quantitative", "axis": None}
    tip = _tip(
        ("date", "Date"),
        ("loss_t", "Loss"),
        ("fhs_t", "FHS 99% VaR"),
        ("garch_evt_t", "GARCH-EVT 99% VaR"),
    )
    layers: list[dict[str, Any]] = [
        {
            "data": {"values": rows},
            "mark": {"type": "bar"},
            "encoding": {
                "x": x,
                "y": {"field": "loss", **y},
                "color": {"datum": "Daily loss", **col},
                "tooltip": tip,
            },
        }
    ]
    for k in RISK_CARDS:
        name = f"{display(k)} 99% VaR"
        layers.append(
            {
                "data": {"values": rows},
                "mark": {"type": "line", "strokeWidth": 1.5},
                "encoding": {
                    "x": _no_axis(x),
                    "y": {"field": k, **quiet},
                    "color": {"datum": name, **col},
                    "tooltip": tip,
                },
            }
        )
        nxt = _value(doc["models"][k]["var"]["0.99"])
        if math.isfinite(nxt):
            layers.append(
                {
                    "data": {
                        "values": [
                            {"x": NEXT, "v": pct_of_position(nxt), "v_t": pct(nxt), "who": name}
                        ]
                    },
                    "mark": {"type": "point", "shape": "diamond", "filled": True, "size": 120},
                    "encoding": {
                        "x": _no_axis(x),
                        "y": {"field": "v", **quiet},
                        "color": {"datum": name, **col},
                        "tooltip": _tip(("who", "Next trading day"), ("v_t", "VaR")),
                    },
                }
            )
    hits = [
        r | {"who": ", ".join(display(k) for k in RISK_CARDS if r[f"{k}_hit"])}
        for r in rows
        if any(r[f"{k}_hit"] for k in RISK_CARDS)
    ]
    if hits:
        layers.append(
            {
                "data": {"values": hits},
                "mark": {"type": "point", "shape": "cross", "filled": True, "size": 90},
                "encoding": {
                    "x": _no_axis(x),
                    "y": {"field": "loss", **quiet},
                    "color": {"datum": "Breach", **col},
                    "tooltip": _tip(("date", "Date"), ("loss_t", "Loss"), ("who", "Breach of")),
                },
            }
        )
    return _spec(mode, layers, title, height=300)


def tail_spec(
    doc: Mapping[str, Any], ser: Mapping[str, Any], fall: float, mode: str
) -> dict[str, Any]:
    """C3: the empirical exceedance frequency of tail_sample (rank / W), the POT-GPD curve and
    the GARCH-EVT curve given today's volatility, log y, the calculator's fall as a rule."""
    p = PALETTE[mode]
    w = doc["config"]["window"]
    obs, evt, mark = (
        "Observed (last W days)".replace("W", f"{w:,}"),
        "GARCH-EVT (given today's volatility)",
        "Calculator fall",
    )
    names = [obs, "POT-GPD", evt, mark]
    col = _colour(names, [p["neutral"], p["pot_gpd"], p["garch_evt"], p["text"]])
    points = [
        {
            "fall": pct_of_position(v),
            "freq": (i + 1) / w,
            "fall_t": pct(v),
            "freq_t": f"{i + 1} of {w:,} days",
        }
        for i, v in enumerate(ser["tail_sample"])
    ]
    lo = pct_of_position(_value(doc["models"]["pot_gpd"]["tail"]["u"]))
    hi = 1.25 * max([100 * fall, *(q["fall"] for q in points)])
    grid = [lo + (hi - lo) * i / (CURVE_POINTS - 1) for i in range(CURVE_POINTS)]
    ticks = list(range(math.floor(lo), math.ceil(hi) + 1))
    x = {
        "field": "fall",
        "type": "quantitative",
        "title": "One-day fall (% of position)",
        "axis": {"values": ticks},
    }
    y = {
        "field": "freq",
        "type": "quantitative",
        "scale": {"type": "log", "domainMin": C3_FLOOR},
        "title": "Exceedance probability per day",
    }
    layers: list[dict[str, Any]] = [
        {
            "data": {"values": points},
            "mark": {"type": "point", "filled": True, "size": 40},
            "encoding": {
                "x": x,
                "y": y,
                "color": {"datum": obs, **col},
                "tooltip": _tip(("fall_t", "Fall"), ("freq_t", "Observed")),
            },
        }
    ]
    for k, name in (("pot_gpd", "POT-GPD"), ("garch_evt", evt)):
        curve = []
        for g in grid:
            q = exceed_prob(doc, k, g / 100)
            if q is not None and q >= C3_FLOOR:  # clipped at the floor
                curve.append(
                    {"fall": g, "freq": q, "fall_t": f"{g:.2f}%", "freq_t": period_text(q)[0]}
                )
        if curve:
            layers.append(
                {
                    "data": {"values": curve},
                    "mark": {"type": "line", "strokeWidth": 2},
                    "encoding": {
                        "x": _no_axis(x),
                        "y": _no_axis(y),
                        "color": {"datum": name, **col},
                        "tooltip": _tip(("fall_t", "Fall"), ("freq_t", name)),
                    },
                }
            )
    layers.append(
        {
            "data": {"values": [{"fall": 100 * fall, "fall_t": f"{100 * fall:.2f}%"}]},
            "mark": {"type": "rule", "strokeDash": [4, 4]},
            "encoding": {
                "x": _no_axis(x),
                "color": {"datum": mark, **col},
                "tooltip": _tip(("fall_t", mark)),
            },
        }
    )
    return _spec(mode, layers, height=320)
