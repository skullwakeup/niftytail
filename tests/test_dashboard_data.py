"""Day-13A dashboard data layer (D55). The real tracked files are read read-only; error and
local-mode branches use synthetic fixtures in a temporary directory."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import shutil
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from conftest import PROJECT_ROOT
from niftytail import dashboard_data as dd
from niftytail import figures as fg
from niftytail import gpd, live, tl_blocks

BANNED = (
    "models",
    "gpd",
    "garch",
    "threshold",
    "rolling",
    "backtest",
    "oos",
    "xi_path",
    "tri",
    "tl_blocks",
    "data",
    "losses",
    "figures",
    "live",
)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _copy(tmp: Path, rel: str) -> Path:
    (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / rel, tmp / rel)
    return tmp / rel


# --- import discipline (D55 (c)) ---------------------------------------------------------------
def _imports(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out += [mod] + [f"{mod}.{a.name}" for a in node.names]
    return out


def _niftytail_module(name: str) -> str | None:
    parts = name.split(".")
    return parts[1] if parts[0] == "niftytail" and len(parts) > 1 else None


def _banned(mod: str) -> bool:
    return mod.startswith(("threshold", "oos", "figures")) or mod in BANNED


def test_no_estimation_imports() -> None:
    files = [
        PROJECT_ROOT / "src/niftytail/dashboard_data.py",
        *sorted((PROJECT_ROOT / "app").glob("*.py")),
    ]
    for path in files:
        for name in _imports(path):
            mod = _niftytail_module(name)
            assert mod is None or not _banned(mod) or mod == "dashboard_data", (path.name, name)
            assert name.split(".")[0] not in ("pandas", "numpy", "matplotlib", "scipy"), name
    data_imports = {n.split(".")[0] for n in _imports(files[0])}
    assert "streamlit" not in data_imports and "niftytail" not in data_imports
    for path in files[1:]:  # app/*.py: streamlit and niftytail.dashboard_data only
        roots = {n for n in _imports(path) if "." not in n or n.startswith("niftytail")}
        assert roots <= {"__future__", "streamlit", "niftytail", "niftytail.dashboard_data"}, roots


def test_the_ast_check_catches_a_banned_import(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("from niftytail import backtest\nimport niftytail.figures_local\n", "utf-8")
    mods = [m for n in _imports(bad) if (m := _niftytail_module(n))]
    assert [m for m in mods if _banned(m)] == ["backtest", "figures_local"]


# --- constants ---------------------------------------------------------------------------------
def test_expected_hashes_equal_the_protected_list() -> None:
    protected = (PROJECT_ROOT / "docs/protected_hashes.md").read_text(encoding="utf-8")
    listed = dict(re.findall(r"([\w./-]+\.(?:json|md|svg)) \(([0-9a-fA-F]{64})", protected))
    for rel, sha in dd.EXPECTED_SHA256.items():
        assert listed[rel].lower() == sha, rel
    assert set(dd.FIGURES.values()) <= set(dd.EXPECTED_SHA256)


def test_names_colours_and_passes_match_d52() -> None:
    assert dd.DISPLAY_NAMES == fg.DISPLAY_NAMES and dd.MODEL_COLOURS == fg.MODEL_COLOURS
    assert [p[1] for p in dd.PASSES] == [p[1] for p in tl_blocks.PASSES]
    assert dd.display("pot_gpd@0.86") == "POT-GPD q=0.86"
    assert Path(".") == dd.HERE


def test_local_constants_match_status() -> None:
    status = (PROJECT_ROOT / "docs/status.md").read_text(encoding="utf-8")
    assert f"`{dd.LOCAL_RUN_DIR}/`" in status
    for rel, sha in dd.LOCAL_SHA256.items():
        assert f"`{rel}` `{sha}`" in status


# --- the real tracked files (read-only) ----------------------------------------------------------
def test_headline_is_verbatim() -> None:
    text = (PROJECT_ROOT / dd.REVIEW).read_text(encoding="utf-8")
    h = dd.headline(PROJECT_ROOT)
    assert set(h) == {"H", "T1", "T3"}
    for tag, para in h.items():
        assert para.startswith(f"**{tag}.**") and para in text
        assert "Source:" not in para
    assert h["T1"].count("\n- ") == 4


def test_headline_paragraphs_are_byte_identical_after_d64() -> None:
    """D64 (a): rewording the two provenance sentences left the quoted claims unchanged.

    The sha256 values were taken from the file before the change (sha256 f12f46b5...).
    """
    pinned = {
        "H": "93e91302001bad1c7f541ba85bb0f4ef9b123ed7f57d9d2879812346a80fd56d",
        "T1": "81281e0759180c8594e8d2050bcc8d56883bc35901a92f7792fc135ca5b1e56b",
        "T3": "ed95c22c1f887d2bdc86559a7a0666062461b367b5c5574283f25d6ca10210b2",
    }
    assert {k: _sha(v.encode("utf-8")) for k, v in dd.headline(PROJECT_ROOT).items()} == pinned


def test_metrics_copy_the_json() -> None:
    bt = json.loads((PROJECT_ROOT / "reports/oos/w1000/backtest.json").read_bytes())
    cards = dd.metrics(PROJECT_ROOT)
    assert len(cards) == 4
    assert cards[0]["value"] == str(bt["n_rows"])
    assert cards[3]["value"] == str(bt["coverage"]["0.99"]["garch_evt"]["breaches"])
    assert cards[3]["label"] == "GARCH-EVT 99% breaches"  # short (D59 pin (42))
    assert cards[3]["help"].startswith("GARCH-EVT breaches, VaR 0.99 (PR W1000); n = ")
    assert all(len(c["label"]) <= 22 for c in cards)


def test_coverage_rows_and_labels() -> None:
    rows = dd.coverage_rows(PROJECT_ROOT, "tri_w1000", "0.99")
    assert [r["model"] for r in rows] == [dd.display(m) for m in dd.MODELS]
    evt = rows[-1]
    assert (evt["ES p"], evt["rejected at 5%"], evt["borderline"], evt["pass-dependent"]) == (
        "0.0438",
        "ES",
        "ES",
        "ES",
    )
    gaussian = rows[0]
    assert gaussian["IND p"] == "≤1e-4" and "IND" in gaussian["rejected at 5%"]
    pr = dd.coverage_rows(PROJECT_ROOT, "pr_w1000", "0.99")[-1]
    assert pr["rejected at 5%"] == "none" and pr["borderline"] == "ES"


def test_dm_rows_verdicts_and_labels() -> None:
    rows = dd.dm_rows(PROJECT_ROOT, "pr_w1000")
    assert len(rows) == 12
    fhs = next(r for r in rows if r["B"] == "FHS" and r["level"] == "0.95")
    assert (fhs["family"], fhs["Holm p"], fhs["verdict"], fhs["pass-dependent"]) == (
        "(ii) Holm",
        "0.114964",
        "not rejected",
        "pass-dependent",
    )
    assert all(r["Holm p"] == "—" for r in rows if r["family"] == "(i)")


def test_sensitivity_rows_side_by_side() -> None:
    rows = dd.sensitivity_rows(PROJECT_ROOT, "0.99")
    garch_n = rows[3]
    assert garch_n["model"] == "GARCH-N"
    assert garch_n["PR W1000 (primary): breaches"] == "98"
    assert garch_n["TRI W1000: breaches"] == "96"
    assert garch_n["PR W1000 (primary): ES p"] == "≤1e-4"
    assert len(garch_n) == 1 + 4 * 3


def test_svg_and_provenance() -> None:
    uri = dd.svg_data_uri(PROJECT_ROOT, "F5")
    assert uri.startswith("data:image/svg+xml;base64,") and "http" not in uri[:40]
    prov = dd.provenance(PROJECT_ROOT)
    assert prov["commits"]["PR W1000 (primary)"].startswith("04572a3")
    assert prov["commits"]["TRI W1000"].startswith("4c68343")
    assert re.fullmatch(r"[0-9a-f]{40}", prov["commits"]["figures (code)"])
    assert prov["inputs"] == dict(sorted(dd.EXPECTED_SHA256.items()))
    assert "D55" in prov["decisions"] and len(prov["reproduce"]) == 3
    # D65 (h): nothing that needs the owner's data or a development commit
    assert prov["reproduce"][-1].endswith("demo-data configs/base.toml --verify")
    assert not any("tl-blocks" in c or "figures" in c for c in prov["reproduce"])


def test_p_text() -> None:
    assert dd.p_text(None, 9999) == "—"
    assert dd.p_text(0.0001, 9999) == "≤1e-4"
    assert dd.p_text(0.0001, None) == "0.0001"
    assert dd.p_text(0.0438, 9999) == "0.0438"


def test_test_rules_and_missing_es_p() -> None:
    cell = {
        "kupiec": {"p": 0.04, "reject": False},
        "christoffersen": {"p_ind": 0.05, "p_cc": 0.0501, "reps": 9999},
        "es_residual": {"p": None},
    }
    assert dd._test("Kupiec", cell, 9999) == (0.04, False, None)  # the JSON field decides
    assert dd._test("IND", cell, 9999) == (0.05, True, 9999)
    assert dd._test("CC", cell, 9999) == (0.0501, False, 9999)
    assert dd._test("ES", cell, 9999) == (None, None, 9999)
    assert [dd._borderline(p) for p in (0.0099, 0.01, 0.10, 0.1001, None)] == [
        False,
        True,
        True,
        False,
        False,
    ]


# --- refusals (synthetic) ----------------------------------------------------------------------
def test_read_refuses_unknown_and_changed_files(tmp_path: Path) -> None:
    with pytest.raises(dd.DashboardDataError, match="not a protected artefact"):
        dd.read(PROJECT_ROOT, "configs/base.toml")
    f = _copy(tmp_path, dd.REVIEW)
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(dd.DashboardDataError, match="sha256 differs"):
        dd.read(tmp_path, dd.REVIEW)


def test_config_must_match_the_run(tmp_path: Path) -> None:
    for _, _, d in dd.PASSES:
        _copy(tmp_path, f"{d}/backtest.json")
    _copy(tmp_path, "reports/oos/w1000/manifest.json")
    cfg = _copy(tmp_path, dd.CONFIG)
    cfg.write_bytes(cfg.read_bytes() + b"\n")
    with pytest.raises(dd.DashboardDataError, match="published run config"):
        dd.coverage_rows(tmp_path, "pr_w1000", "0.99")


def test_headline_missing_paragraph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / dd.REVIEW
    f.parent.mkdir(parents=True)
    f.write_bytes(b"**H.** text\n\nSource: x\n")
    monkeypatch.setitem(dd.EXPECTED_SHA256, dd.REVIEW, _sha(f.read_bytes()))
    with pytest.raises(dd.DashboardDataError, match="no T1 paragraph"):
        dd.headline(tmp_path)


def test_provenance_without_a_figures_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _, _, d in dd.PASSES:
        _copy(tmp_path, f"{d}/manifest.json")
    _copy(tmp_path, dd.TL)
    _copy(tmp_path, dd.XI)
    f = tmp_path / dd.FIGURES_DOC
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"# no commit line\n")
    monkeypatch.setitem(dd.EXPECTED_SHA256, dd.FIGURES_DOC, _sha(f.read_bytes()))
    assert dd.provenance(tmp_path)["commits"]["figures (code)"] == "—"


# --- local and public mode (D55 (e)) -----------------------------------------------------------
def _local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / dd.LOCAL_RUN_DIR
    shas = {}
    for i, rel in enumerate(dd.LOCAL_SHA256):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(f"<svg>{i}</svg>".encode())
        shas[rel] = _sha((base / rel).read_bytes())
    monkeypatch.setattr(dd, "LOCAL_SHA256", shas)
    return base


def test_local_mode_shows_g1_to_g4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _local(tmp_path, monkeypatch)
    assert dd.is_public(tmp_path, {}) is False
    figs = dd.local_figures(tmp_path, {})
    assert figs is not None and [f["title"][:2] for f in figs] == ["G1", "G2", "G3", "G4"]
    assert all(f["uri"].startswith("data:image/svg+xml;base64,") for f in figs)
    assert dd.BANNER == (
        "Local only: contains daily series derived from NSE data. Never deploy or share."
    )


def test_public_mode_touches_no_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _local(tmp_path, monkeypatch)
    assert dd.is_public(tmp_path, {"NIFTYTAIL_PUBLIC": "1"}) is True

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("run dir accessed in public mode")

    monkeypatch.setattr(Path, "is_dir", boom)
    monkeypatch.setattr(Path, "read_bytes", boom)
    assert dd.local_figures(tmp_path, {"NIFTYTAIL_PUBLIC": "1"}) is None


def test_no_local_dir_is_public(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIFTYTAIL_PUBLIC", raising=False)
    assert dd.is_public(tmp_path) is True and dd.local_figures(tmp_path) is None


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_local_mismatch_hides_the_tab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    base = _local(tmp_path, monkeypatch)
    f = base / "figures/cum_fz0.svg"
    if damage == "missing":
        f.unlink()
    else:
        f.write_bytes(b"<svg>x</svg>")
    assert dd.local_figures(tmp_path, {}) is None


# --- D58: next-day forecast and the rupee view --------------------------------------------------
W_SYN, N_U = 1000, 120
P_LEVELS = {"0.95": Fraction(5, 100), "0.99": Fraction(1, 100), "0.995": Fraction(5, 1000)}
P_LEVELS["0.999"] = Fraction(1, 1000)
TOL = 1e-8


def _gpd_model(
    u: float, xi: float, beta: float, mu: float | None = None, sigma: float | None = None
) -> dict[str, Any]:
    """VaR/ES from niftytail.gpd itself (the code the mirror must reproduce), as models.py."""
    var, es = {}, {}
    for lvl, p in P_LEVELS.items():
        z, e = gpd.tail_risk(u, xi, beta, W_SYN, N_U, p, TOL)
        var[lvl] = z if mu is None or sigma is None else mu + sigma * z
        es[lvl] = e if mu is None or sigma is None else mu + sigma * e
        var[lvl], es[lvl] = (live._num(var[lvl]), live._num(es[lvl]))
    m: dict[str, Any] = {"var": var, "es": es, "flags": [], "xi": xi, "beta": beta}
    if mu is None:
        m["tail"] = {"u": u, "n_u": float(N_U)}
    else:
        m |= {"mu_next": mu, "sigma_next": sigma, "tail": {"u_z": u, "n_u": float(N_U)}}
    return m


def synthetic_forecast(
    as_of: str = "2001-01-05", xi: float = 0.2, xi_z: float = 0.1
) -> dict[str, Any]:
    """A schema-valid v2 forecast.json; POT-GPD and GARCH-EVT from niftytail.gpd."""
    models: dict[str, Any] = {
        k: {
            "var": {lvl: 0.01 * (j + 1) + 0.001 * i for j, lvl in enumerate(dd.LIVE_LEVELS)},
            "es": {lvl: 0.015 * (j + 1) + 0.001 * i for j, lvl in enumerate(dd.LIVE_LEVELS)},
            "flags": [],
        }
        for i, k in enumerate(dd.MODELS)
    }
    for k in ("garch_n", "fhs"):
        models[k] |= {"mu_next": 0.0004, "sigma_next": 0.011}
    models["pot_gpd"] = _gpd_model(0.018, xi, 0.007)
    models["garch_evt"] = _gpd_model(1.2, xi_z, 0.6, mu=0.0004, sigma=0.011)
    health = {
        k: {"breaches": 2, "n_obs": 250, "inadmissible_rows": 0, "zone": "green"} for k in dd.MODELS
    }
    return {
        "schema_version": 2,
        "as_of": as_of,
        "horizon": 1,
        "config": {"window": W_SYN, "q_selected": "0.88", "q_c": "0.88", "xi_zero_tol": TOL},
        "models": models,
        "volatility": {"sigma_next": 0.012, "window_std": 0.01, "sigma_ratio": 1.2},
        "health": {"level": "0.99", "targets": 250, "models": health},
        "provenance": {"commit": "0123456789abcdef0123456789abcdef01234567"},
    }


def display_forecast() -> dict[str, Any]:
    """synthetic_forecast with one missing VaR/ES and one infinite ES (display rules)."""
    doc = synthetic_forecast()
    doc["models"]["pot_gpd"]["var"]["0.999"] = doc["models"]["pot_gpd"]["es"]["0.999"] = None
    doc["models"]["garch_evt"]["es"]["0.999"] = "inf"
    return doc


def _weekdays(end: date, n: int) -> list[str]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return out[::-1]


def synthetic_series(as_of: str = "2001-01-05", n: int = 1300) -> dict[str, Any]:
    """A schema-valid series.json: synthetic closes, the last 30 sessions "live"."""
    dates = _weekdays(date.fromisoformat(as_of), n)
    closes = [1000.0 * math.exp(0.01 * math.sin(i / 7) + 0.0002 * i) for i in range(n)]
    hist = [
        {
            "date": d,
            "open": c * 0.999,
            "high": c * 1.004,
            "low": c * 0.995,
            "close": c,
            "source": "live" if i >= n - 30 else "raw",
        }
        for i, (d, c) in enumerate(zip(dates, closes, strict=True))
    ]
    hist[5]["open"] = None  # a missing Open
    losses = [-math.log(closes[i] / closes[i - 1]) for i in range(1, n)]
    recent = [
        {"date": dates[i + 1], "loss": losses[i]}
        | {k: {"var": 0.02, "hit": int(losses[i] > 0.02)} for k in dd.MODELS}
        for i in range(n - 251, n - 1)
    ]
    recent[-1]["fhs"] = {"var": None, "hit": 0}
    recent[-3] = {"date": recent[-3]["date"], "loss": 0.03} | {
        k: {"var": 0.02, "hit": 1} for k in dd.MODELS
    }  # one breach
    order = sorted(range(n - 1), key=lambda i: -losses[i])[:5]
    return {
        "schema_version": 2,
        "level": "0.99",
        "history": hist,
        "window": {"first": dates[n - W_SYN], "last": dates[-1]},
        "recent": recent,
        "tail_sample": [0.018 + 0.0003 * k for k in range(N_U, 0, -1)],
        "worst": [{"date": dates[i + 1], "loss": losses[i]} for i in order],
    }


def test_banned_list_covers_live() -> None:
    assert _banned("live")  # live.py imports rolling: banned like the other estimation modules


def test_rupee_formulas_hand_values() -> None:
    x = math.log(1 / 0.98)  # a 2% fall
    assert dd.rupee_loss(1_000_000, x) == pytest.approx(20_000, rel=1e-12)
    assert dd.rupee_loss(1_000_000, 0.0) == 0.0
    assert dd.rupee_loss(1_000_000, -x) == pytest.approx(-20_408.163265306, rel=1e-9)
    assert math.isnan(dd.rupee_loss(1.0, math.nan)) and dd.rupee_loss(1.0, math.inf) == math.inf
    assert dd.rupee_loss(1.0, 0.03) < dd.rupee_loss(1.0, 0.04) < 0.04  # monotone, below x
    assert dd.budget_position(20_000, x) == pytest.approx(1_000_000, rel=1e-12)
    for es in (0.01, 0.05, 0.3):
        assert dd.rupee_loss(dd.budget_position(12_345.0, es), es) == pytest.approx(12_345.0)
    for es in (math.nan, math.inf, 0.0, -0.01):
        assert math.isnan(dd.budget_position(1.0, es))


@pytest.mark.parametrize(
    ("x", "text"),
    [
        (1_234_567, "₹12,34,567"),
        (1_000_000, "₹10,00,000"),
        (100_000, "₹1,00,000"),
        (1_000, "₹1,000"),
        (999.4, "₹999"),
        (0, "₹0"),
        (-1_234.6, "-₹1,235"),
        (-0.4, "₹0"),
        (123_456_789_012, "₹1,23,45,67,89,012"),
        (math.nan, "—"),
        (math.inf, "unbounded"),
    ],
)
def test_inr_indian_grouping(x: float, text: str) -> None:
    assert dd.inr(x) == text


def test_forecast_rows_and_labels() -> None:
    doc = display_forecast()
    rows = dd.forecast_rows(doc, 1_000_000, dd.LEVELS)
    assert len(rows) == 12 and [r["level"] for r in rows[:2]] == ["0.95", "0.99"]
    notes = {r["model"]: r["note"] for r in rows}
    assert {m for m, n in notes.items() if n} == {"FHS", "GARCH-EVT"}
    assert set(notes.values()) == {"", dd.PASSED_NOTE}
    assert dd.PASSED_NOTE.endswith("on the price index (D46 H); pass-dependent, borderline")
    first = rows[0]  # Gaussian, 0.95: VaR 0.01, shown as % of position (D59 (f))
    assert first["VaR (% of position)"] == "1.00%" == dd.pct(0.01)
    assert first["VaR ₹"] == dd.inr(1_000_000 * (1 - math.exp(-0.01)))
    far = {(r["model"], r["level"]): r for r in dd.forecast_rows(doc, 1.0, dd.FAR_TAIL)}
    pot = far[("POT-GPD", "0.999")]
    assert pot["VaR ₹"] == "—" == pot["ES (% of position)"]
    assert far[("GARCH-EVT", "0.999")]["ES ₹ (slightly conservative)"] == "unbounded"
    assert far[("GARCH-EVT", "0.999")]["ES (% of position)"] == "unbounded"
    budget = {r["model"]: r["largest position (99% ES ₹ ≤ budget)"] for r in dd.budget_rows(doc, 1)}
    assert budget["Gaussian"] == dd.inr(1 / (1 - math.exp(-0.03)))
    assert dd.health_rows(doc)[0] == {
        "model": "Gaussian",
        "breaches at 0.99": "2 of 250",
        "zone": "green",
    }
    doc["health"]["models"]["hs"]["zone"] = None
    assert dd.health_rows(doc)[1]["zone"] == "—"
    assert dd.volatility_ratio(doc) == "1.20\u00d7"
    doc["volatility"]["sigma_ratio"] = None
    assert dd.volatility_ratio(doc) == "—"


def _break(doc: dict[str, Any], how: str) -> dict[str, Any]:
    if how == "schema":
        doc["schema_version"] = 1
    elif how == "tail":
        del doc["models"]["garch_evt"]["tail"]["u_z"]
    elif how == "mu":
        doc["models"]["fhs"]["mu_next"] = "x"
    elif how == "window":
        doc["config"]["window"] = "1000"
    elif how == "commit":
        doc["provenance"]["commit"] = None
    elif how == "as_of":
        doc["as_of"] = "tomorrow"
    elif how == "model":
        del doc["models"]["hs"]
    elif how == "level":
        del doc["models"]["hs"]["var"]["0.999"]
    elif how == "value":
        doc["models"]["hs"]["es"]["0.99"] = "x"
    elif how == "bool":
        doc["models"]["hs"]["es"]["0.99"] = True
    elif how == "zone":
        doc["health"]["models"]["hs"]["zone"] = "blue"
    elif how == "breaches":
        doc["health"]["models"]["hs"]["breaches"] = "2"
    elif how == "ratio":
        doc["volatility"]["sigma_ratio"] = "high"
    elif how == "missing":
        del doc["volatility"]
    return doc


@pytest.mark.parametrize(
    "how",
    [
        "schema",
        "as_of",
        "model",
        "level",
        "value",
        "bool",
        "zone",
        "breaches",
        "ratio",
        "missing",
        "tail",
        "mu",
        "window",
        "commit",
    ],
)
def test_schema_check_rejects(how: str) -> None:
    assert dd._valid(synthetic_forecast()) is True
    assert dd._valid(_break(synthetic_forecast(), how)) is False
    assert dd._valid([1, 2]) is False


def _write(root: Path, key: str, doc: Any, ser: Any = None, chk: Any = None) -> None:
    p = root / dd.LIVE_REPORTS / key / "forecast.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    ser = (
        synthetic_series(doc["as_of"] if isinstance(doc, dict) else "2001-01-05")
        if ser is None
        else ser
    )
    (p.parent / "series.json").write_text(json.dumps(ser), encoding="utf-8")
    if chk is not None:
        (p.parent / "check.json").write_text(json.dumps(chk), encoding="utf-8")


def test_live_forecast_picks_the_latest_valid(tmp_path: Path) -> None:
    assert dd.live_forecast(tmp_path, {}) is None  # no reports/live
    _write(tmp_path, "a", synthetic_forecast("2001-01-05"))
    _write(tmp_path, "b", synthetic_forecast("2001-01-08"))
    _write(tmp_path, "c", _break(synthetic_forecast("2001-02-01"), "zone"))
    _write(tmp_path, "d", "{not json")
    _write(tmp_path, "e", synthetic_forecast("2001-01-08") | {"horizon": 9})  # tie: first kept
    doc = dd.live_forecast(tmp_path, {})
    assert doc is not None and doc["as_of"] == "2001-01-08" and doc["horizon"] == 1


def test_public_mode_touches_no_live_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "a", synthetic_forecast())

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("reports/live accessed in public mode")

    monkeypatch.setattr(Path, "glob", boom)
    monkeypatch.setattr(Path, "read_text", boom)
    assert dd.live_forecast(tmp_path, {"NIFTYTAIL_PUBLIC": "1"}) is None


# --- D59 (f)-(p): arithmetic and chart specs ------------------------------------------------
def test_pct_of_position() -> None:
    assert dd.pct(0.03) == "2.96%"
    assert dd.pct_of_position(math.log(1 / 0.98)) == pytest.approx(2.0, rel=1e-12)
    assert dd.pct(math.nan) == "—" and dd.pct(math.inf) == "unbounded"
    for x in (0.01, 0.0543, 0.2):  # % x position = ₹ up to rounding
        assert dd.pct_of_position(x) / 100 * 1_000_000 == pytest.approx(dd.rupee_loss(1e6, x))
    assert dd.PCT_CAPTION == "Model output is a log loss x; shown as 1 - e^-x of the position."


@pytest.mark.parametrize("xi", [0.2, -0.25, 1e-9, 1.2])
def test_self_check_reproduces_gpd(xi: float) -> None:
    doc = synthetic_forecast(xi=xi, xi_z=xi)
    out = dd.self_check(doc)
    assert out["ok"] and out["symbol"] == "\u2713" and out["max_error"] <= 1e-12
    bad = synthetic_forecast(xi=xi, xi_z=xi)
    bad["models"]["garch_evt"]["var"]["0.99"] *= 1 + 1e-9
    out = dd.self_check(bad)
    assert not out["ok"] and out["symbol"].startswith("\u2717") and out["max_error"] > 1e-12


def test_self_check_missing_levels() -> None:
    doc = synthetic_forecast()
    doc["models"]["pot_gpd"]["var"]["0.999"] = None  # stored missing, mirror present
    assert not dd.self_check(doc)["ok"]
    doc = synthetic_forecast()
    doc["models"]["pot_gpd"]["tail"]["n_u"] = 1.0  # p >= n_u / W everywhere: mirror missing
    assert not dd.self_check(doc)["ok"]
    for lvl in dd.LIVE_LEVELS:
        doc["models"]["pot_gpd"]["var"][lvl] = doc["models"]["pot_gpd"]["es"][lvl] = None
    assert dd.self_check(doc)["ok"]
    doc["models"]["pot_gpd"]["xi"] = None  # a failed fit: no mirror, all missing
    assert dd.tail_risk(doc, "pot_gpd", 1, 100) is None and dd.self_check(doc)["ok"]


@pytest.mark.parametrize("key", ["pot_gpd", "garch_evt"])
def test_exceed_prob_and_return_level_round_trip(key: str) -> None:
    doc = synthetic_forecast()
    for n in dd.RETURN_N:
        x = dd.return_level(doc, key, n)
        assert x is not None
        assert dd.exceed_prob(doc, key, -math.expm1(-x)) == pytest.approx(1 / n, rel=1e-9)
    assert dd.return_level(doc, key, 5) is None  # p = 0.2 >= n_u / W = 0.12
    stored = doc["models"][key]["var"]["0.99"]
    assert dd.return_level(doc, key, 100) == pytest.approx(stored, rel=1e-12)


def test_exceed_prob_branches() -> None:
    doc = synthetic_forecast(xi=-0.25)
    u, beta = 0.018, 0.007
    end = u + beta / 0.25  # the upper end of the fit
    assert dd.exceed_prob(doc, "pot_gpd", -math.expm1(-(end + 0.01))) == 0.0
    assert dd.period_text(0.0) == (dd.ABOVE_END, "—")
    assert dd.exceed_prob(doc, "pot_gpd", 0.001) is None  # below u
    assert dd.period_text(None) == (dd.BELOW_TAIL, "—")
    doc = synthetic_forecast(xi=1e-9)  # within xi_zero_tol: the exponential branch
    x = 0.03
    want = N_U / W_SYN * math.exp(-(x - u) / beta)
    assert dd.exceed_prob(doc, "pot_gpd", -math.expm1(-x)) == pytest.approx(want, rel=1e-12)
    assert dd.period_text(0.004) == ("1 in 250 trading days", "about every 1.0 years")


def test_crash_and_return_level_tables() -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    rows = dd.crash_rows(doc, 0.05)
    assert [r["model"] for r in rows] == ["POT-GPD", "GARCH-EVT (given today's volatility)"]
    p = dd.exceed_prob(doc, "pot_gpd", 0.05)
    assert p is not None and rows[0]["probability per day"] == f"{100 * p:.3g}%"
    assert rows[0]["how often"] == dd.period_text(p)[0]
    low = dd.crash_rows(doc, 0.001)
    assert low[0]["how often"] == dd.BELOW_TAIL and low[0]["probability per day"] == "—"
    table = dd.return_level_rows(doc)
    assert [r["once in"] for r in table] == [f"{n:,} trading days" for n in dd.RETURN_N]
    assert [r["note"] for r in table] == ["", dd.DESCRIPTIVE, dd.DESCRIPTIVE, dd.DESCRIPTIVE]
    assert table[1]["POT-GPD"] == dd.pct(dd.return_level(doc, "pot_gpd", 250) or 0.0)
    one = synthetic_forecast()
    one["models"]["pot_gpd"]["tail"]["n_u"] = 1.0
    assert dd.return_level_rows(one)[0]["POT-GPD"] == "—"
    # history count: window losses >= x, only at or above u
    x = 0.018 + 0.0003 * 100
    assert dd.history_count(doc, ser, -math.expm1(-x)) == sum(
        1 for v in ser["tail_sample"] if v >= -math.log1p(math.expm1(-x))
    )
    assert dd.history_count(doc, ser, 0.5) == 0 and dd.history_count(doc, ser, 0.001) is None


def test_vol_shock_and_stress() -> None:
    doc = synthetic_forecast()
    m = doc["models"]["garch_evt"]
    for key in dd.CONDITIONAL:
        one = dd.vol_shock(doc, key, 1)
        assert all(
            one[lvl] == (doc["models"][key]["var"][lvl], doc["models"][key]["es"][lvl])
            for lvl in dd.LIVE_LEVELS
        )
    two = dd.vol_shock(doc, "garch_evt", 2)["0.99"]
    mu = m["mu_next"]
    assert two == (mu + 2 * (m["var"]["0.99"] - mu), mu + 2 * (m["es"]["0.99"] - mu))
    with pytest.raises(dd.DashboardDataError, match="conditional"):
        dd.vol_shock(doc, "pot_gpd", 2)
    rows = dd.stress_rows(doc, 1.5, 1_000_000)
    assert [r["model"] for r in rows] == ["GARCH-N", "FHS", "GARCH-EVT"]
    var = dd.vol_shock(doc, "fhs", 1.5)["0.99"][0]
    assert rows[1]["99% VaR"] == dd.pct(var) and rows[1]["99% VaR ₹"] == dd.inr(
        dd.rupee_loss(1e6, var)
    )
    assert dd.SCENARIO == "scenario, not a forecast"


def test_replay_and_cards() -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    rows = dd.replay_rows(doc, ser, 1_000_000)
    assert [r["date"] for r in rows] == [w["date"] for w in ser["worst"]]
    w = ser["worst"][0]
    assert rows[0]["fall"] == dd.pct(w["loss"])
    assert rows[0]["₹ on the position"] == dd.inr(dd.rupee_loss(1e6, w["loss"]))
    assert set(rows[0]) == {"date", "fall", "₹ on the position", "POT-GPD", "GARCH-EVT"}
    card = dd.risk_card(doc, "fhs", 1_000_000)
    v = doc["models"]["fhs"]["var"]["0.99"]
    assert card["VaR %"] == dd.pct(v) and card["VaR ₹"] == dd.inr(dd.rupee_loss(1e6, v))
    nums = dd.pipeline_numbers(doc)
    assert nums["W"] == "1000" and nums["n_u"] == "120" and nums["u_z"] == "1.2"
    assert nums["q"] == "0.88" and nums["sigma_next"] == "0.011"


def _hist(closes: list[float]) -> dict[str, Any]:
    ser = synthetic_series()
    days = _weekdays(date(2001, 1, 5), len(closes))
    ser["history"] = [
        {"date": d, "open": c, "high": c, "low": c, "close": c, "source": "raw"}
        for d, c in zip(days, closes, strict=True)
    ]
    return ser


def test_market_hand_values() -> None:
    ser = _hist([100.0, 110.0, 99.0, 121.0])
    mk = dd.market(ser, "All")
    assert [round(r["drawdown"], 12) for r in mk["rows"]] == [0.0, 0.0, -0.1, 0.0]
    assert mk["change"] == pytest.approx(0.21) and mk["worst_drawdown"] == pytest.approx(-0.1)
    assert math.isnan(mk["rows"][0]["change"]) and mk["rows"][2]["change"] == pytest.approx(-0.1)
    assert [r["date"] for r in dd.market(ser, "1M")["rows"]] == [r["date"] for r in mk["rows"]]
    last = dd.last_session(ser)
    assert last["close"] == "121.00" and last["change vs the previous close"] == "+22.22%"
    kp = {k["label"]: k["value"] for k in dd.kpis(ser, "All")}
    assert kp == {
        "Last close": "121.00",
        "Change since the first session": "+21.00%",
        "Worst drawdown in All": "-10.00%",
        "Source of the last session": "niftyindices snapshot",
    }
    long = synthetic_series()
    rows = dd.market(long, "1Y")["rows"]
    first = date.fromisoformat(long["history"][-1]["date"]) - timedelta(days=366)
    assert rows[0]["date"] >= first.isoformat() > long["history"][-len(rows) - 1]["date"]
    assert dd.level_text(None) == "—" and dd.level_text(24500.356) == "24,500.36"


def test_freshness_with_an_injected_date() -> None:
    doc, ser = synthetic_forecast("2001-01-05"), synthetic_series("2001-01-05")
    ok = dd.freshness(doc, ser, date(2001, 1, 9))
    assert ok == {
        "text": "Data through 2001-01-05",
        "source": "niftyindices file",
        "warning": None,
    }
    old = dd.freshness(doc, ser, date(2001, 1, 10))
    assert old["warning"] is not None and "5 calendar days" in old["warning"]
    assert dd.freshness(doc, ser)["warning"] is not None  # today is long after 2001


def test_live_bundle_selection_warning_and_check(tmp_path: Path) -> None:
    chk = {"run": "r", "identical": 20, "n": 20, "commit": "a" * 40}
    _write(tmp_path, "a", synthetic_forecast("2001-01-05"), chk=chk)
    b = dd.live_bundle(tmp_path, {})
    assert b is not None and b["check"] == chk and b["warning"] is None
    assert b["commit"] == "0123456789ab" and b["series"]["history"]
    other = synthetic_forecast("2001-01-05")
    other["provenance"]["commit"] = "f" * 40  # same numbers, another commit: no warning
    _write(tmp_path, "b", other)
    assert (dd.live_bundle(tmp_path, {}) or {})["warning"] is None
    diff = synthetic_forecast("2001-01-05")
    diff["models"]["hs"]["var"]["0.99"] = 0.5
    _write(tmp_path, "c", diff)
    b = dd.live_bundle(tmp_path, {})
    assert b is not None and b["warning"] == (
        "3 forecasts for this date disagree; showing commit 0123456789ab; "
        "remove stale reports/live dirs"
    )
    _write(tmp_path, "d", synthetic_forecast("2001-01-08"), ser={"schema_version": 1})
    assert (dd.live_bundle(tmp_path, {}) or {})["doc"]["as_of"] == "2001-01-05"  # bad series
    _write(tmp_path, "a", synthetic_forecast("2001-01-05"), chk={"run": "r"})
    assert (dd.live_bundle(tmp_path, {}) or {})["check"] is None


def test_public_mode_reads_no_series_or_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        tmp_path, "a", synthetic_forecast(), chk={"run": "r", "identical": 1, "n": 1, "commit": "c"}
    )

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("reports/live accessed in public mode")

    monkeypatch.setattr(Path, "glob", boom)
    monkeypatch.setattr(Path, "read_text", boom)
    assert dd.live_bundle(tmp_path, {"NIFTYTAIL_PUBLIC": "1"}) is None


def _layers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return spec["layer"]


def _scale_ranges(spec: dict[str, Any]) -> set[str]:
    out = set()
    for layer in _layers(spec):
        col = layer["encoding"].get("color", {})
        out |= set(col.get("scale", {}).get("range", []))
        out |= (
            {v for k, v in layer["mark"].items() if k == "color"}
            if isinstance(layer["mark"], dict)
            else set()
        )
        line = layer["mark"].get("line", {}) if isinstance(layer["mark"], dict) else {}
        out |= {line["color"]} if "color" in line else set()
    return out


def _all_specs(mode: str) -> dict[str, dict[str, Any]]:
    doc, ser = synthetic_forecast(), synthetic_series()
    out = {f"M1 {r}": dd.market_spec(doc, ser, r, True, mode) for r in dd.RANGES}
    out |= {f"M2 {r}": dd.drawdown_spec(ser, r, mode) for r in dd.RANGES}
    out["C2"] = dd.recent_spec(doc, ser, mode)
    out["C3"] = dd.tail_spec(doc, ser, 0.05, mode)
    return out


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_specs_marks_tooltips_and_colours(mode: str) -> None:
    pal = dd.PALETTE[mode]
    for name, spec in _all_specs(mode).items():
        assert spec["config"]["axis"]["labelColor"] == pal["muted"], name
        names: set[str] = set()
        for layer in _layers(spec):
            assert {"mark", "encoding", "data"} <= set(layer), name
            assert layer["encoding"]["tooltip"], name
            col = layer["encoding"].get("color")
            if col is not None:
                names |= set(col["scale"]["domain"])
                assert len(col["scale"]["domain"]) == len(col["scale"]["range"])
        if len(names) >= 2:
            assert all(
                layer["encoding"]["color"]["legend"] is not None
                for layer in _layers(spec)
                if "color" in layer["encoding"]
            )
        colours = {c.lower() for c in _scale_ranges(spec)}
        assert "#fcd535" not in colours, name
        assert colours <= {v.lower() for v in pal.values()}, name
        ys = {
            json.dumps(layer["encoding"]["y"].get("title"))
            for layer in _layers(spec)
            if "y" in layer["encoding"] and layer["encoding"]["y"].get("title")
        }
        assert len(ys) <= 1, name  # one y axis per chart
    specs = _all_specs(mode)
    m1 = specs["M1 4Y"]["layer"]
    full = specs["M1 All"]["layer"]
    assert full[0]["encoding"]["color"]["scale"]["range"][-2:] == [pal["garch_evt"], pal["hit"]]
    assert full[0]["mark"]["opacity"] == 0.13 and full[0]["data"]["values"][0]["label"] == (
        "Model's window (last 1,000 days)"
    )
    assert any(
        layer["mark"].get("shape") == "triangle-down"
        for layer in m1
        if isinstance(layer["mark"], dict)
    )
    line = [
        layer for layer in m1 if isinstance(layer["mark"], dict) and layer["mark"]["type"] == "line"
    ]
    assert line and line[0]["encoding"]["color"]["datum"] == "Close"
    assert line[0]["encoding"]["y"]["scale"]["type"] == "log"
    candles = specs["M1 6M"]["layer"]
    assert {"rule", "bar"} <= {
        la["mark"] if isinstance(la["mark"], str) else la["mark"]["type"] for la in candles
    }
    assert candles[1]["encoding"]["color"]["scale"]["range"][:2] == [pal["up"], pal["down"]]
    m2 = specs["M2 4Y"]["layer"][0]
    assert m2["mark"]["color"] == pal["hit"] and m2["mark"]["opacity"] == 0.10
    assert m2["encoding"]["x"]["scale"] == m1[0]["encoding"]["x"]["scale"]  # same x range
    c2 = specs["C2"]
    assert c2["layer"][0]["encoding"]["color"]["scale"]["range"] == [
        pal["neutral"],
        pal["fhs"],
        pal["garch_evt"],
        pal["hit"],
    ]
    assert any(la["mark"].get("shape") == "cross" for la in c2["layer"])
    assert sum(la["mark"].get("shape") == "diamond" for la in c2["layer"]) == 2  # end markers
    c3 = specs["C3"]
    assert c3["layer"][0]["encoding"]["y"]["scale"]["type"] == "log"
    assert c3["layer"][-1]["mark"]["type"] == "rule"
    assert c3["layer"][-1]["data"]["values"][0]["fall"] == 5.0
    assert [la["encoding"]["color"]["datum"] for la in c3["layer"]] == [
        "Observed (last 1,000 days)",
        "POT-GPD",
        "GARCH-EVT (given today's volatility)",
        "Calculator fall",
    ]
    lin = dd.market_spec(synthetic_forecast(), synthetic_series(), "All", False, mode)
    assert all(
        la["encoding"]["y"]["scale"]["type"] == "linear"
        for la in lin["layer"]
        if "scale" in la["encoding"].get("y", {})
    )


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_no_tooltip_field_is_temporal(mode: str) -> None:
    # 15C-6: a tooltip field that is also a temporal channel shows epoch milliseconds
    doc, ser = synthetic_forecast(), synthetic_series()
    specs = _all_specs(mode)
    specs |= {f"M1 {r} linear": dd.market_spec(doc, ser, r, False, mode) for r in dd.RANGES}
    for name, spec in specs.items():
        for layer in _layers(spec):
            enc = layer["encoding"]
            temporal = {
                v["field"]
                for ch, v in enc.items()
                if ch != "tooltip" and isinstance(v, dict) and v.get("type") == "temporal"
            }
            if enc.get("x", {}).get("type") == "temporal" and "x2" in enc:
                temporal.add(enc["x2"]["field"])  # x2 takes the type of x
            tips = {t["field"] for t in enc["tooltip"]}
            assert not tips & temporal, (name, tips & temporal)


def test_tooltip_date_is_the_iso_string() -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    isos = {h["date"] for h in ser["history"]}
    for spec in (dd.market_spec(doc, ser, "4Y", True, "dark"), dd.drawdown_spec(ser, "4Y", "dark")):
        for layer in _layers(spec):
            dates = [t["field"] for t in layer["encoding"]["tooltip"] if t["title"] == "Date"]
            for row in layer["data"]["values"] if dates else []:
                assert row[dates[0]] == row["date"] and row["date"] in isos
    rows = dd.market_spec(doc, ser, "4Y", True, "dark")["layer"][0]["data"]["values"]
    assert rows[-1]["date_t"] == ser["history"][-1]["date"]


def test_market_spec_window_outside_the_range() -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    ser["window"] = {"first": "1990-01-01", "last": "1990-02-01"}
    ser["worst"] = [{"date": "1990-01-02", "loss": 0.1}]
    spec = dd.market_spec(doc, ser, "1M", False, "dark")
    assert [
        la["mark"] if isinstance(la["mark"], str) else la["mark"]["type"] for la in spec["layer"]
    ] == ["rule", "bar"]


def test_palette_is_d59_s() -> None:
    assert dd.PALETTE["dark"] | {} == {
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
    }
    light = dd.PALETTE["light"]
    assert (light["garch_evt"], light["fhs"], light["pot_gpd"], light["neutral"]) == (
        "#b07800",
        "#2a78d6",
        "#c2477a",
        "#707a8a",
    )
    assert (light["text"], light["muted"]) == ("#181a20", "#5e6673")
    assert dd.TAIL_KEYS == live.TAIL_KEYS and dd.SCHEMA_VERSION == live.SCHEMA_VERSION


def test_local_mode() -> None:
    assert dd.local_mode({}) and not dd.local_mode({"NIFTYTAIL_PUBLIC": "1"})


def test_invalid_series_and_fits() -> None:
    assert not dd._valid_series({}) and not dd._valid_series([1])
    assert dd._valid_series(synthetic_series())
    doc = synthetic_forecast()
    doc["models"]["pot_gpd"]["beta"] = -1.0
    assert dd.exceed_prob(doc, "pot_gpd", 0.05) is None
    assert dd.crash_rows(doc, 0.05)[0]["how often"] == dd.BELOW_TAIL


def test_specs_without_optional_layers() -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    doc["models"]["fhs"]["var"]["0.99"] = None  # no end marker for FHS
    for r in ser["recent"]:
        for k in dd.MODELS:
            r[k]["hit"] = 0
    c2 = dd.recent_spec(doc, ser, "dark")["layer"]
    assert sum(la["mark"].get("shape") == "diamond" for la in c2) == 1
    assert not any(la["mark"].get("shape") == "cross" for la in c2)
    for k in dd.EVT:
        doc["models"][k]["xi"] = None  # no fit: no curve
    c3 = dd.tail_spec(doc, ser, 0.05, "dark")["layer"]
    assert [la["mark"]["type"] for la in c3] == ["point", "rule"]


# --- D59 13E-4 display fixes (pins (30)-(44)) ----------------------------------------------
def _axes(spec: dict[str, Any], ch: str) -> int:
    """Layers that draw an axis for the channel (no "axis": None)."""
    return sum(
        1
        for la in spec["layer"]
        if ch in la["encoding"] and la["encoding"][ch].get("axis", {}) is not None
    )


@pytest.mark.parametrize("log_scale", [True, False])
@pytest.mark.parametrize("rng", list(dd.RANGES))
def test_m1_domain_axes_and_shading(rng: str, log_scale: bool) -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    spec = dd.market_spec(doc, ser, rng, log_scale, "dark")
    rows = dd.market(ser, rng)["rows"]
    keys = ("low", "high") if (dd.RANGES[rng] or 10**9) <= dd.CANDLE_MAX_DAYS else ("close",)
    vis = [r[k] for r in rows for k in keys if r[k] is not None]
    lo, hi = min(vis), max(vis)
    for la in spec["layer"]:
        y = la["encoding"].get("y")
        if y is not None:
            sc = y["scale"]
            assert sc["zero"] is False and sc["nice"] is False
            a, b = sc["domain"]
            assert a < lo and hi < b  # fits the visible data, with padding
            if log_scale:
                assert (hi / lo) ** 0.029 < lo / a < (hi / lo) ** 0.031
            else:
                assert (lo - a) == pytest.approx(0.03 * (hi - lo))
    assert _axes(spec, "x") == 1 and _axes(spec, "y") == 1  # one axis each
    main = next(la for la in spec["layer"] if la["encoding"]["x"].get("axis") is not None)
    assert main["encoding"]["x"]["axis"]["grid"] is False
    rects = [
        la for la in spec["layer"] if isinstance(la["mark"], dict) and la["mark"]["type"] == "rect"
    ]
    assert len(rects) == (1 if rng == "All" else 0)
    cap = dd.window_caption(doc, rng)
    assert cap == (
        None if rng == "All" else "This range lies inside the model's window (last 1,000 days)"
    )
    dd_spec = dd.drawdown_spec(ser, rng, "dark")
    assert (
        _axes(dd_spec, "x") == 1 and dd_spec["layer"][0]["encoding"]["x"]["axis"]["grid"] is False
    )


def test_y_domain_hand_values() -> None:
    assert dd.y_domain([100.0, 200.0], False) == pytest.approx([97.0, 203.0])
    lo, hi = dd.y_domain([100.0, 400.0], True)
    assert lo == pytest.approx(100 / 4**0.03) and hi == pytest.approx(400 * 4**0.03)
    assert dd.y_domain([50.0, 50.0], False) == pytest.approx([48.5, 51.5])
    assert dd.y_domain([50.0, 50.0], True) == pytest.approx([50 / 1.03, 50 * 1.03])


def test_caps_and_labels() -> None:
    assert dd.period_text(1 / 100_001) == (dd.CAPPED, "—")
    assert dd.period_text(1 / 99_999)[0] == "1 in 99,999 trading days"
    assert dd.prob_text(1 / 200_000) == "< 0.001%" == dd.CAPPED_P
    assert dd.prob_text(0.01) == "1%" and dd.prob_text(None) == "—" == dd.prob_text(0.0)
    assert dd.CAPPED == "rarer than 1 in 100,000 trading days (beyond the reliable range)"
    assert dd.ABOVE_END == (
        "above the fitted tail's upper limit (xi < 0): outside what the model can say"
    )
    doc = synthetic_forecast(xi=-0.25)
    rows = dd.crash_rows(doc, 0.4)  # far above the fitted upper end
    assert rows[0]["how often"] == dd.ABOVE_END and rows[0]["in years"] == "—"
    far = dd.crash_rows(synthetic_forecast(xi=0.05), 0.3)  # thin tail: a tiny probability
    assert far[0]["how often"] == dd.CAPPED and far[0]["probability per day"] == dd.CAPPED_P
    assert far[0]["in years"] == "—"


def test_c3_floor_and_ticks() -> None:
    doc, ser = synthetic_forecast(xi=0.05), synthetic_series()
    spec = dd.tail_spec(doc, ser, 0.05, "dark")
    assert dd.C3_FLOOR_NOTE == "Below 1 in a million per day: beyond the reliable range."
    main = spec["layer"][0]["encoding"]
    assert main["y"]["scale"]["domainMin"] == 1e-6
    ticks = main["x"]["axis"]["values"]
    assert ticks == list(range(ticks[0], ticks[-1] + 1))  # every 1%
    for la in spec["layer"][1:-1]:
        assert min(v["freq"] for v in la["data"]["values"]) >= 1e-6  # clipped at the floor
    assert _axes(spec, "x") == 1 and _axes(spec, "y") == 1


def test_c2_dates_on_x_and_next() -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    spec = dd.recent_spec(doc, ser, "dark")
    x = spec["layer"][0]["encoding"]["x"]
    assert x["type"] == "ordinal" and x["scale"]["domain"] == [r["date"] for r in ser["recent"]] + [
        "next"
    ]
    marks = [la for la in spec["layer"] if la["mark"].get("shape") == "diamond"]
    assert {m["data"]["values"][0]["x"] for m in marks} == {"next"}
    assert _axes(spec, "x") == 1 and _axes(spec, "y") == 1


def test_tex_numbers_and_source_labels() -> None:
    assert dd.tex_num(4.9622e-05) == "4.96 \\times 10^{-5}"
    assert dd.tex_num(0.0123) == "0.0123" and dd.tex_num(1000) == "1000"
    assert dd.tex_num(2.5e7) == "2.5 \\times 10^{7}" and dd.tex_num(None) == "—"
    doc = synthetic_forecast()
    doc["models"]["garch_evt"]["mu_next"] = 4.9622e-05
    assert dd.pipeline_numbers(doc)["mu_next"] == "4.96 \\times 10^{-5}"
    assert dd.SOURCE_LABELS == {
        "raw": "niftyindices snapshot",
        "live": "niftyindices file",
        "api": "EODHD API",
        "demo": "synthetic demo",
    }
    ser = synthetic_series()
    tips = {
        r["source"]
        for r in dd.market_spec(doc, ser, "1M", True, "dark")["layer"][0]["data"]["values"]
    }
    assert tips <= set(dd.SOURCE_LABELS.values())


# --- D61 (c), (d): year on the time axes; the All-range change label ----------------------------
@pytest.mark.parametrize("rng", list(dd.RANGES))
def test_time_axes_show_the_year(rng: str) -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    for spec in (dd.market_spec(doc, ser, rng, True, "dark"), dd.drawdown_spec(ser, rng, "dark")):
        (axis,) = [
            la["encoding"]["x"]["axis"]
            for la in spec["layer"]
            if la["encoding"]["x"].get("axis") is not None
        ]
        assert "%Y" in axis["format"] and axis["grid"] is False
        assert axis == dd.time_axis(rng)
    assert dd.time_axis("4Y")["format"] == "%b %Y"
    assert dd.time_axis("4Y")["tickCount"] == {"interval": "month", "step": 6}
    assert "tickCount" not in dd.time_axis("1M")


def test_change_label() -> None:
    assert dd.change_label("All") == "Change since the first session"
    assert dd.change_label("1Y") == "Change over 1Y"
    labels = [k["label"] for k in dd.kpis(synthetic_series(), "All")]
    assert labels[1] == "Change since the first session"


# --- D62 (b): public mode from secrets; the demo bundle; chart titles -----------------------------
def test_local_mode_with_secrets() -> None:
    assert dd.local_mode({}, {}) and dd.local_mode({}, None)
    assert not dd.local_mode({}, {"NIFTYTAIL_PUBLIC": "1"})
    assert not dd.local_mode({}, {"NIFTYTAIL_PUBLIC": 1})
    assert not dd.local_mode({"NIFTYTAIL_PUBLIC": "1"}, {})


def test_demo_bundle(tmp_path: Path) -> None:
    b = dd.demo_bundle(PROJECT_ROOT)
    assert b is not None and b["commit"] == "synthetic demo" and b["check"] is None
    assert dd.SOURCE_LABELS[b["series"]["history"][-1]["source"]] == "synthetic demo"
    assert dd.demo_bundle(tmp_path) is None  # no demo files


def test_spec_titles() -> None:
    doc, ser = synthetic_forecast(), synthetic_series()
    assert "title" not in dd.market_spec(doc, ser, "1Y", True, "dark")
    spec = dd.market_spec(doc, ser, "1Y", True, "dark", "Market (synthetic index)")
    assert spec["title"]["text"] == "Market (synthetic index)"
    assert spec["title"]["color"] == dd.PALETTE["dark"]["text"]
    assert dd.drawdown_spec(ser, "1Y", "light", "D")["title"]["text"] == "D"
    assert dd.recent_spec(doc, ser, "dark", "R")["title"]["text"] == "R"
