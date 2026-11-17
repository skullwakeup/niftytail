"""Streamlit UI tests (D55 (g), D59 (q)). Run in .venv-dashboard on Windows (D56), from the root:

    $env:UV_PROJECT_ENVIRONMENT = ".venv-dashboard"
    uv run --locked --group dashboard python -m pytest app/tests -o addopts=""

Not collected by the Windows run (pyproject testpaths = ["tests"]); streamlit is never installed
into the Windows .venv (D43). Synthetic forecast and series only.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from niftytail import dashboard_data as dd

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app" / "dashboard.py")
sys.path.insert(0, str(ROOT / "tests"))
from test_dashboard_data import synthetic_forecast, synthetic_series  # noqa: E402

COMMON = ["Track record", "How it works", "Proof"]
PUBLIC_SECTIONS = ["Today (demo)", *COMMON]  # D62 (b): the synthetic demo comes first
SECTIONS = ["Today", *COMMON]
ALL_SECTIONS = ["Today", "Today (demo)", *COMMON]
TRACK = ["Overview", "Coverage", "Model comparison", "Traffic light"]
HOW = ["Pipeline", "Threshold robustness", "Tail index", "Sensitivities"]
DAY_BY_DAY = "Day by day (local only)"


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    st.cache_data.clear()


def _run(monkeypatch: pytest.MonkeyPatch, public: bool) -> AppTest:
    monkeypatch.chdir(ROOT)
    if public:
        monkeypatch.setenv("NIFTYTAIL_PUBLIC", "1")
    else:
        monkeypatch.delenv("NIFTYTAIL_PUBLIC", raising=False)
    return AppTest.from_file(APP, default_timeout=180).run()


def _labels(at: AppTest) -> list[str]:
    return [t.label for t in at.tabs]


def _own_text(at: AppTest) -> str:
    """Rendered text without the verbatim tracked paragraphs (D49 T3 itself says "predicted by
    D36 (a)(3)"; claim text is never reworded, so the "predict" rule covers the app's own text)."""
    text = _text(at)
    for para in dd.headline(ROOT).values():
        text = text.replace(para, "")
    return text


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [s.value for s in at.subheader] + [str(t.value) for t in at.table]
    parts += [m.label + " " + m.value for m in at.metric] + [i.value for i in at.info]
    return "\n".join(parts)


@pytest.fixture
def no_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dd, "LIVE_REPORTS", str(tmp_path / "none"))
    return tmp_path


@pytest.fixture
def live_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "key"
    d.mkdir()
    (d / "forecast.json").write_text(json.dumps(synthetic_forecast()), encoding="utf-8")
    (d / "series.json").write_text(json.dumps(synthetic_series()), encoding="utf-8")
    chk = {"run": "r", "identical": 20, "n": 20, "commit": "a" * 40}
    (d / "check.json").write_text(json.dumps(chk), encoding="utf-8")
    monkeypatch.setattr(dd, "LIVE_REPORTS", str(tmp_path))
    return d


def test_public_mode_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(monkeypatch, public=True)
    assert not at.exception
    labels = _labels(at)
    assert [x for x in labels if x in ALL_SECTIONS] == PUBLIC_SECTIONS
    assert [x for x in labels if x in TRACK + HOW] == TRACK + HOW
    assert DAY_BY_DAY not in labels
    text = _text(at)
    assert dd.BANNER not in text and "reports/runs" not in text
    assert dd.DEMO_BANNER in text  # the permanent demo banner
    assert "predict" not in _own_text(at).lower()


def test_public_mode_reads_nothing_under_reports_live(monkeypatch: pytest.MonkeyPatch) -> None:
    real_glob, real_read = Path.glob, Path.read_text

    def guard(path: Path) -> None:
        if "reports/live" in path.as_posix() or "reports/runs" in path.as_posix():
            raise AssertionError(f"public mode touched {path}")

    def glob(self: Path, pattern: str, **kw: Any) -> Any:
        guard(self)
        return real_glob(self, pattern, **kw)

    def read_text(self: Path, *a: Any, **kw: Any) -> str:
        guard(self)
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "read_text", read_text)
    at = _run(monkeypatch, public=True)
    assert not at.exception
    text = _text(at)
    assert "Self-check" not in text and "past days recomputed" not in text
    latex = "\n".join(x.value for x in at.latex)
    assert latex and not re.search(r"=\s*-?\d+\.\d", latex)  # formulas, no plugged values


def test_headline_is_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(monkeypatch, public=True)
    text = "\n".join(m.value for m in at.markdown)
    for para in dd.headline(ROOT).values():
        assert para in text


def test_selectors_drive_the_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(monkeypatch, public=True)
    at.selectbox(key="coverage-pass").set_value("TRI W1000").run()
    at.selectbox(key="coverage-level").set_value("0.99").run()
    assert not at.exception
    rows = dd.coverage_rows(ROOT, "tri_w1000", "0.99")
    table = next(t.value for t in at.table if "ES p" in t.value.columns)
    assert list(table["model"]) == [r["model"] for r in rows]
    assert list(table["ES p"]) == [r["ES p"] for r in rows]


def test_floor_and_labels_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _text(_run(monkeypatch, public=True))
    assert "≤1e-4" in text and "borderline" in text and "pass-dependent" in text


def test_overview_subheaders(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(monkeypatch, public=True)
    heads = [s.value for s in at.subheader]
    main = "Main finding on the price index (pass-dependent, borderline)"  # D62 15A-1
    assert "What holds in every test pass" in heads and main in heads
    assert "Headline and its limits" not in heads
    assert not any("D49 T1" in s or "D46 H" in s for s in heads)


def test_inputs_block_is_sha256sum_format(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(monkeypatch, public=True)
    line = re.compile(r"[0-9a-f]{64}  \S+")
    blocks = [c.value for c in at.code if line.fullmatch(c.value.split("\n")[0])]
    assert len(blocks) == 1
    lines = blocks[0].split("\n")
    assert len(lines) == len(dd.EXPECTED_SHA256)
    assert all(line.fullmatch(s) for s in lines)
    assert [s[66:] for s in lines] == sorted(dd.EXPECTED_SHA256)
    # D65 (h): "Reproduce it yourself" offers only what passes from the public copy
    assert dd.VERIFIED in _text(at)
    code = "\n".join(c.value for c in at.code)
    assert "demo-data configs/base.toml --verify" in code
    assert not any(s in code for s in ("tl-blocks", "figures configs", "update.cmd"))


def test_local_sections_and_day_by_day(monkeypatch: pytest.MonkeyPatch, no_live: Path) -> None:
    at = _run(monkeypatch, public=False)
    assert not at.exception
    labels = _labels(at)
    assert [x for x in labels if x in ALL_SECTIONS] == SECTIONS  # never the demo locally
    assert (DAY_BY_DAY in labels) == (dd.local_figures(ROOT, {}) is not None)
    assert dd.NO_FORECAST in _text(at)  # no valid v2 forecast
    assert dd.BANNER in _text(at)


def _tables(at: AppTest, column: str) -> list[Any]:
    return [t.value for t in at.table if column in t.value.columns]


def test_today_blocks(monkeypatch: pytest.MonkeyPatch, live_fixture: Path) -> None:
    at = _run(monkeypatch, public=False)
    assert not at.exception
    text = _text(at)
    doc, ser = synthetic_forecast(), synthetic_series()
    assert "Data through 2001-01-05" in text and "0123456789ab" in text
    assert any(w.value.startswith("The newest session is") for w in at.warning)
    assert "Last session in your data (end of day). Intraday prices are not shown." in text
    buttons = at.link_button
    assert [b.proto.label for b in buttons] == ["Today's market on NSE ↗"]
    assert buttons[0].proto.url == "https://www.nseindia.com/"
    kpis = {m.label: m.value for m in at.metric}
    assert kpis["Last close"] == dd.level_text(ser["history"][-1]["close"])
    card = dd.risk_card(doc, "garch_evt", 1_000_000)
    assert (
        kpis["GARCH-EVT: 99% VaR"] == card["VaR ₹"]
        and kpis["FHS: 99% ES"] == dd.risk_card(doc, "fhs", 1_000_000)["ES ₹"]
    )
    assert kpis["Volatility vs its 4-year average"] == "1.20" + dd.TIMES
    assert dd.headline(ROOT)["T3"] in text and dd.PASSED_NOTE in text
    assert dd.FAR_NOTE in text and dd.PCT_CAPTION in text
    assert len(at.get("vega_lite_chart")) >= 4  # M1, M2, C2 and C3
    main, far = _tables(at, "ES ₹ (slightly conservative)")
    assert list(main["level"].unique()) == ["0.95", "0.99"]
    assert list(far["level"].unique()) == ["0.995", "0.999"]
    assert "predict" not in _own_text(at).lower()
    # Proof: check.json and the forecast provenance
    assert "20/20 past days recomputed from scratch match the published backtest" in text
    # How it works: today's numbers and the self-check
    assert "**Self-check** ✓" in text
    assert any("u = 0.018" in x.value for x in at.latex)


def test_crash_calculator_one_input(monkeypatch: pytest.MonkeyPatch, live_fixture: Path) -> None:
    at = _run(monkeypatch, public=False)
    at.number_input(key="fall").set_value(7.5).run()
    assert not at.exception
    doc = synthetic_forecast()
    (crash,) = _tables(at, "probability per day")
    want = dd.crash_rows(doc, 0.075)
    assert list(crash["how often"]) == [r["how often"] for r in want]
    assert list(crash["probability per day"]) == [r["probability per day"] for r in want]
    count = dd.history_count(doc, synthetic_series(), 0.075)
    assert any(f"{count} of the last 1,000 trading days" in c.value for c in at.caption)
    (returns,) = _tables(at, "once in")
    assert list(returns["once in"]) == [r["once in"] for r in dd.return_level_rows(doc)]


def test_stress_slider(monkeypatch: pytest.MonkeyPatch, live_fixture: Path) -> None:
    at = _run(monkeypatch, public=False)
    at.slider[0].set_value(2.0).run()
    assert not at.exception
    (stress,) = _tables(at, "99% ES ₹")
    want = dd.stress_rows(synthetic_forecast(), 2.0, 1_000_000)
    assert list(stress["99% VaR"]) == [r["99% VaR"] for r in want]
    (replay,) = _tables(at, "₹ on the position")
    assert len(replay) == 5


def test_today_absent_in_public_mode(monkeypatch: pytest.MonkeyPatch, live_fixture: Path) -> None:
    at = _run(monkeypatch, public=True)
    assert not at.exception
    assert [x for x in _labels(at) if x in ALL_SECTIONS] == PUBLIC_SECTIONS
    assert "2001-01-05" not in _text(at)


def test_config_theme_dark_and_fonts() -> None:
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["toolbarMode"] == "viewer"
    assert cfg["server"]["enableStaticServing"] is True
    t = cfg["theme"]
    assert t["base"] == "dark" and "light" not in t and "dark" not in t  # dark only (D59)
    assert (t["backgroundColor"], t["secondaryBackgroundColor"], t["borderColor"]) == (
        "#0b0e11",
        "#1e2329",
        "#2b3139",
    )
    assert (t["textColor"], t["primaryColor"], t["linkColor"]) == ("#eaecef", "#fcd535", "#fcd535")
    assert t["baseRadius"] == "8px" and t["showWidgetBorder"] is True
    for face in t["fontFaces"]:
        assert face["url"].startswith("app/static/") and "//" not in face["url"]
        assert (ROOT / "app" / face["url"].removeprefix("app/")).is_file()
    assert {f["family"] for f in t["fontFaces"]} == {"Inter", "IBM Plex Mono"}
    fonts = ROOT / "app" / "static" / "fonts"
    for lic in ("LICENSE-Inter.txt", "LICENSE-IBMPlexMono.txt"):
        assert "SIL Open Font License" in (fonts / lic).read_text(encoding="utf-8")


def test_no_white_text_on_primary_and_no_primary_widgets() -> None:
    src = Path(APP).read_text(encoding="utf-8")
    assert 'type="primary"' not in src and "st.checkbox" not in src and "st.toggle" not in src
    assert "stNumberInputStepUp" in src  # the hover override (dark text on the primary)
    assert "#fcd535" not in json.dumps(dd.PALETTE).lower()


# --- D59 13E-4 display fixes ---------------------------------------------------------------------
def test_review_fixes_in_today(monkeypatch: pytest.MonkeyPatch, live_fixture: Path) -> None:
    at = _run(monkeypatch, public=False)
    assert not at.exception
    doc, ser = synthetic_forecast(), synthetic_series()
    captions = [c.value for c in at.caption]
    # (6) the history count without the duplicated phrase
    count = dd.history_count(doc, ser, 0.05)
    assert f"{count} of the last 1,000 trading days had a fall at least this large." in captions
    assert not any("(in the last" in c for c in captions)
    # (7) % of position under each rupee value
    for key in dd.RISK_CARDS:
        card = dd.risk_card(doc, key, 1_000_000)
        assert f"{card['VaR %']} of position" in captions
        assert f"{card['ES %']} of position" in captions
    # (8) source labels, no folder names
    text = _text(at)
    assert "last session from the niftyindices file" in text and "data/live" not in text
    # D61 (b): the source KPI is a small, wrapping card, never a truncated metric
    assert "Source of the last session" not in [m.label for m in at.metric]
    assert any(
        '<div class="nt-kpi-label">Source of the last session</div>' in m.value
        and '<div class="nt-kpi-small">niftyindices file</div>' in m.value
        for m in at.markdown
    )
    css = next(m.value for m in at.markdown if ".nt-kpi-small" in m.value)
    assert "white-space: normal" in css
    # (9) D49 T3 in a collapsed expander, verbatim
    (exp,) = [e for e in at.expander if e.label == dd.T3_EXPANDER]
    assert exp.proto.expanded is False
    assert dd.headline(ROOT)["T3"] in [m.value for m in exp.markdown]
    # (12) volatility caption; (3), (4), (5), (10) captions
    assert dd.VOL_NOTE in captions and dd.REPLAY_NOTE in captions
    assert dd.STRESS_NOTE in captions and dd.C3_FLOOR_NOTE in captions
    assert dd.window_caption(doc, "4Y") in captions  # the default range is 4Y


def test_track_record_labels_short(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(monkeypatch, public=True)
    cards = dd.metrics(ROOT)
    track = list(at.metric)[-len(cards) :]  # after the demo's metrics in "Today (demo)"
    assert [m.label for m in track] == [c["label"] for c in cards]
    assert all(len(m.label) <= 22 for m in track)
    assert [m.proto.help for m in track] == [c["help"] for c in cards]  # full text kept


def test_config_hides_the_welcome_popup() -> None:
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["logger"]["hideWelcomeMessage"] is True


# --- D62 (b): the public demo -------------------------------------------------------------------
def test_public_demo_today(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(monkeypatch, public=True)
    assert not at.exception
    demo = dd.demo_bundle(ROOT)
    assert demo is not None
    text = _text(at)
    assert dd.DEMO_BANNER in text and "synthetic demo" in text
    # D65 (i): the last-session card says it is the demo; no NSE link beside it
    assert "Last session (synthetic demo)" in text
    assert "Last session in your data" not in text and len(at.link_button) == 0
    assert dd.T1_BORDERLINE in text and dd.PASSED_NOTE in text  # D65 (g)
    assert f"Data through {demo['doc']['as_of']}" in text
    assert not any(w.value.startswith("The newest session is") for w in at.warning)
    titles = [
        json.loads(c.proto.spec).get("title", {}).get("text", "") for c in at.get("vega_lite_chart")
    ]
    assert sum("(synthetic index)" in t for t in titles) == 3  # M1, M2, C2
    assert "Self-check" not in text  # How it works stays formula-only in public mode
    assert dd.NO_FORECAST not in text


def test_public_mode_from_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("NIFTYTAIL_PUBLIC", raising=False)
    at = AppTest.from_file(APP, default_timeout=180)
    at.secrets["NIFTYTAIL_PUBLIC"] = "1"
    at.run()
    assert not at.exception
    assert [x for x in _labels(at) if x in ALL_SECTIONS] == PUBLIC_SECTIONS
    assert dd.BANNER not in _text(at)


def test_local_mode_never_shows_the_demo(monkeypatch: pytest.MonkeyPatch, no_live: Path) -> None:
    at = _run(monkeypatch, public=False)
    assert dd.DEMO_SECTION not in _labels(at) and dd.DEMO_BANNER not in _text(at)
