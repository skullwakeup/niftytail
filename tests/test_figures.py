"""Day-12B tracked figures (D51). Synthetic JSON fixtures only, no real data.

Builders are checked through the returned artists (data, labels), not pixels. Command tests run in
a temporary git repository holding the fixtures at the D51 (a) paths, with the expected-hash
constants monkeypatched to the fixture hashes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from conftest import PROJECT_ROOT
from niftytail import figures as fg
from niftytail import oos, tl_blocks
from niftytail.__main__ import configure_logging, main
from test_threshold_real import ISO_DATE
from test_tl_blocks import audit_reads

CONFIG = "configs/base.toml"
MONTH_DATE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}\b"
    r"|\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b"
)
MODELS = fg.MODELS
SENS = (
    "garch_evt@0.86",
    "garch_evt@0.9",
    "garch_evt@0.91",
    "pot_gpd@0.86",
    "pot_gpd@0.9",
    "pot_gpd@0.91",
)
KEYS = (*MODELS, *SENS)
PIDS = tuple(p[0] for p in tl_blocks.PASSES)
BOUNDS = (0.0099, 0.01, 0.0499, 0.05, 0.0501, 0.10, 0.1001, 0.0001)
NB = 4  # full blocks in the tl fixture


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _p(i: int, col: int) -> float:
    return BOUNDS[(i * 8 + col) % len(BOUNDS)]


def _zone(x: int) -> str:
    return "green" if x <= 4 else ("yellow" if x <= 9 else "red")


def make_tl(cfg_sha: str) -> dict[str, Any]:
    passes = {}
    for s, (pid, label, window, *_rest) in enumerate(tl_blocks.PASSES):
        keys = {}
        for i, k in enumerate(KEYS):
            brs = [(b + i + 3 * s) % 12 for b in range(NB)]
            keys[k] = {
                "blocks": [
                    {
                        "block": b,
                        "n_obs": 250,
                        "breaches": x,
                        "zone": _zone(x),
                        "inadmissible_rows": 0,
                    }
                    for b, x in enumerate(brs)
                ],
                "partial_block": {"n_obs": 40, "breaches": i},
            }
        passes[pid] = {"label": label, "window": window, "keys": keys}
    return {
        "decision": "D51",
        "level": "0.99",
        "block_size": 250,
        "provenance": {"commit": "c" * 40, "config_sha256": cfg_sha},
        "passes": passes,
    }


def make_backtest(s: int, tl: dict[str, Any]) -> dict[str, Any]:
    cov: dict[str, Any] = {}
    for li, lvl in enumerate(fg.LEVELS):
        a = 1 - float(lvl)
        cov[lvl] = {}
        for i, k in enumerate(KEYS):
            p_hat = a * (0.8 + 0.05 * i + 0.01 * s)
            col = 4 * li
            pk = _p(i, col)
            cov[lvl][k] = {
                "n": 5716,
                "breaches": round(5716 * p_hat),
                "p_hat": p_hat,
                "ci": {
                    "clopper_pearson": [0.8 * p_hat, 1.2 * p_hat],
                    "block": {"b": 18, "lower": 0.7 * p_hat, "upper": 1.3 * p_hat},
                },
                # the reject field decides Kupiec: it disagrees with p <= 0.05 for model 1
                "kupiec": {"p": pk, "reject": (pk <= 0.05) != (i == 1)},
                "christoffersen": {"p_ind": _p(i, col + 1), "p_cc": _p(i, col + 2), "reps": 9999},
                "es_residual": {
                    "p": None if (i, lvl) == (5, "0.95") else _p(i, col + 3),
                    "z_bar": 0.1,
                    "n_breach": 10,
                    "descriptive": False,
                },
            }
    for lvl in fg.DESCRIPTIVE_LEVELS:
        cov[lvl] = {
            k: {
                "descriptive": True,
                "n": 5716 - i,
                "breaches": 10 + i + s,
                "p_hat": 0.001,
                "p_binomial": 0.001 * (i + 1),
            }
            for i, k in enumerate(KEYS)
        }
    dm: dict[str, list[dict[str, Any]]] = {"unconditional": [], "conditional": []}
    for lvl in fg.LEVELS:
        for j, (a_, b_) in enumerate(
            (("gaussian", "hs"), ("gaussian", "pot_gpd"), ("hs", "pot_gpd"))
        ):
            dm["unconditional"].append(
                {
                    "a": a_,
                    "b": b_,
                    "level": lvl,
                    "dm": 1.0 + j + 0.1 * s,
                    "p": 0.2,
                    "reject": j == 0,
                }
            )
        for j, (a_, b_) in enumerate(
            (("garch_evt", "pot_gpd"), ("garch_evt", "garch_n"), ("garch_evt", "fhs"))
        ):
            dm["conditional"].append(
                {
                    "a": a_,
                    "b": b_,
                    "level": lvl,
                    "dm": -2.0 - j - 0.1 * s,
                    "p": 0.01,
                    "p_holm": 0.03,
                    "reject": j != 1,
                }
            )
    fz0 = {
        lvl: {k: {"n": 5716, "mean": -3.0 - 0.01 * i - 0.1 * s} for i, k in enumerate(KEYS)}
        for lvl in fg.LEVELS
    }
    return {
        "primary_keys": list(MODELS),
        "sensitivity_keys": list(SENS),
        "coverage": cov,
        "dm": dm,
        "fz0": fz0,
        "traffic_light": {k: fg._counts(e) for k, e in tl["passes"][PIDS[s]]["keys"].items()},
    }


OOS_BLOCKS: dict[str, Any] = {
    "layout": {
        "blocks": [
            {"block": "B1", "overlapping": False},
            {"block": "B2", "overlapping": False},
            {"block": "B3", "overlapping": True},
        ]
    },
    "raw": [
        {
            "block": "B1",
            "selection": {"q_star": "4/5"},
            "check": {"q_ref": "22/25", "var99_rel": 0.01, "es99_rel": -0.02},
        },
        {
            "block": "B2",
            "selection": {"q_star": "81/100"},
            "check": {"q_ref": "22/25", "var99_rel": 0.02, "es99_rel": None},
        },
        {
            "block": "B3",
            "selection": {"q_star": None},
            "check": {"q_ref": "22/25", "var99_rel": None, "es99_rel": None},
        },
    ],
    "residuals": [
        {
            "block": "B1",
            "selection": {"q_star": "41/50"},
            "check": {"q_ref": "22/25", "var99_rel": -0.03, "es99_rel": 0.04},
        },
        {
            "block": "B2",
            "selection": {"q_star": "4/5"},
            "check": {"q_ref": "22/25", "var99_rel": 0.0, "es99_rel": 0.05},
        },
        {"block": "B3", "garch": {"garch_error": True}, "check": None},
    ],
}
XI: dict[str, Any] = {
    "keys": {
        "pot_gpd": {
            "blocks": [
                {
                    "block": b,
                    "n_obs": 40 if b == 3 else 250,
                    "partial": b == 3,
                    "xi_median": 0.1 * b,
                    "share_ci_lo_gt_0": 0.2 * b,
                }
                for b in range(4)
            ],
            "training_reference": {"q": "22/25", "xi_hat": 0.12, "xi_ci": [-0.01, 0.27]},
        },
        "garch_evt": {
            "blocks": [
                {
                    "block": b,
                    "n_obs": 40 if b == 3 else 250,
                    "partial": b == 3,
                    "xi_median": None if b == 1 else -0.05 * b,
                    "share_ci_lo_gt_0": 0.0,
                }
                for b in range(4)
            ],
        },
    }
}
TRAINING = {"selection": {"q_star": "22/25"}}


def make_inputs() -> fg.Inputs:
    cfg_bytes = (PROJECT_ROOT / CONFIG).read_bytes()
    tl = make_tl(_sha(cfg_bytes))
    return fg.Inputs(
        backtest={pid: make_backtest(s, tl) for s, pid in enumerate(PIDS)},
        oos_blocks=OOS_BLOCKS,
        xi_path=XI,
        training=TRAINING,
        tl=tl,
        config=tomllib.loads(cfg_bytes.decode("utf-8")),
        sha256={},
    )


@pytest.fixture(scope="module")
def inp() -> fg.Inputs:
    return make_inputs()


@pytest.fixture
def close_figs() -> Any:
    yield
    plt.close("all")


def _lines(fig: Figure) -> dict[str, Any]:
    return {
        ln.get_label(): ln
        for ax in fig.axes
        for ln in ax.lines
        if not ln.get_label().startswith("_")
    }


def _rects(fig: Figure) -> dict[str, Rectangle]:
    return {
        p.get_label(): p
        for ax in fig.axes
        for p in ax.patches
        if isinstance(p, Rectangle) and not p.get_label().startswith("_")
    }


# --- pure rules -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("p", "rejected", "borderline"),
    [
        (0.0099, True, False),
        (0.01, True, True),
        (0.0499, True, True),
        (0.05, True, True),
        (0.0501, False, True),
        (0.10, False, True),
        (0.1001, False, False),
    ],
)
def test_verdict_boundaries(p: float, rejected: bool, borderline: bool) -> None:
    assert fg.verdict(p) == (rejected, borderline)
    assert fg.verdict(p, not rejected) == (not rejected, borderline)  # Kupiec: the field decides


def test_verdict_and_p_text_edge_cases() -> None:
    assert fg.verdict(None) == (None, False)
    assert fg.p_text(None, 9999) == "n/a"
    assert fg.p_text(0.0001, 9999) == "\u22641e-4"
    assert fg.p_text(0.0001, None) == "0.0001"
    assert fg.p_text(0.00011, 9999) == "0.00011"
    assert fg.p_text(8.32291e-07, None) == "8.32e-07"
    assert fg.p_text(0.001, 999) == "\u22641e-3"


def test_p_text_falls_back_without_the_glyph(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fg._has_glyph("\u2264")  # the default font has it, so the Unicode form is used
    monkeypatch.setattr(fg, "_has_glyph", lambda _c: False)
    assert fg.p_text(0.0001, 9999) == "<=1e-4"


def test_display_names() -> None:
    assert [fg.display(k) for k in MODELS] == [
        "Gaussian",
        "HS",
        "POT-GPD",
        "GARCH-N",
        "FHS",
        "GARCH-EVT",
    ]
    assert fg.display("pot_gpd@0.86") == "POT-GPD q=0.86"
    assert fg.display("garch_evt@0.9") == "GARCH-EVT q=0.9"


# --- builders ---------------------------------------------------------------------------------
def test_f1_plots_the_fixture_values(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f1_coverage(inp.backtest)
    assert len(fig.axes) == 6
    for r, lvl in enumerate(fg.LEVELS):
        for c, pid in enumerate(PIDS):
            ax = fig.axes[r * 3 + c]
            assert ax.get_title() == f"{fg.PASS_LABELS[pid]}, VaR {lvl}"
            got = {cont.get_label(): cont for cont in ax.containers}
            assert len(got) == 2 * len(MODELS)
            for k in MODELS:
                cell = inp.backtest[pid]["coverage"][lvl][k]
                for tag, (lo, hi) in (
                    ("cp", cell["ci"]["clopper_pearson"]),
                    ("block", (cell["ci"]["block"]["lower"], cell["ci"]["block"]["upper"])),
                ):
                    cont = got[f"{k}:{tag}"]
                    assert cont.lines[0].get_ydata()[0] == pytest.approx(100 * cell["p_hat"])
                    seg = cont.lines[2][0].get_segments()[0]
                    assert seg[:, 1].tolist() == pytest.approx([100 * lo, 100 * hi])
                    assert cont.lines[0].get_color() == fg.MODEL_COLOURS[k]
            nominal = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
            assert [ln.get_ydata()[0] for ln in nominal] == [pytest.approx(100 * (1 - float(lvl)))]
            ticks = [tk.get_text() for tk in ax.get_xticklabels()]
            assert ticks == [fg.display(k) for k in MODELS]
    assert not any(ax.get_legend() for ax in fig.axes) and len(fig.legends) == 1


def test_f1_legend_overlaps_no_data_artist(inp: fg.Inputs, close_figs: Any) -> None:
    """D52 (2): the figure-level legend intersects no axes' data artist."""
    fig = fg.f1_coverage(inp.backtest)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    boxes = [leg.get_window_extent(renderer) for leg in fig.legends]
    assert boxes
    n = 0
    for ax in fig.axes:
        for art in [*ax.lines, *ax.collections, *ax.patches]:
            bb = art.get_window_extent(renderer)
            if bb.width > 0 or bb.height > 0:
                n += 1
                assert not any(box.overlaps(bb) for box in boxes), art
    assert n > 0


def test_f2_verdicts_borderline_and_p_text(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f2_tests(inp.backtest, inp.config)
    rects = _rects(fig)
    assert len(rects) == len(PIDS) * len(MODELS) * 8
    seen = 0
    for ax_i, pid in enumerate(PIDS):
        texts = {t.get_gid(): t.get_text() for t in fig.axes[ax_i].texts}
        for i, k in enumerate(MODELS):
            for li, lvl in enumerate(fg.LEVELS):
                for ti, test in enumerate(fg.TESTS):
                    p = _p(i, 4 * li + ti)
                    if test == "ES" and (i, lvl) == (5, "0.95"):
                        rejected, border, text = None, False, "n/a"
                    else:
                        rejected = (p <= 0.05) != (i == 1) if test == "Kupiec" else p <= 0.05
                        border = 0.01 <= p <= 0.10
                        floor = p == 0.0001 and test != "Kupiec"
                        text = "\u22641e-4" if floor else f"{p:.3g}"
                    label = f"{pid}|{k}|{lvl}|{test}|{rejected}|{border}"
                    matches = [lab for lab in rects if lab.startswith(f"{pid}|{k}|{lvl}|{test}|")]
                    assert label in matches
                    r = rects[label]
                    assert (r.get_hatch() == "////") == border
                    want = (
                        fg.MISSING_FILL
                        if rejected is None
                        else (fg.REJECT_FILL if rejected else fg.ACCEPT_FILL)
                    )
                    assert r.get_facecolor() == pytest.approx(plt.matplotlib.colors.to_rgba(want))
                    assert texts[f"{k}|{lvl}|{test}"] == text
                    seen += 1
    assert seen == len(PIDS) * len(MODELS) * 8


def test_f3_dm_statistics_and_reject_marks(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f3_dm(inp.backtest)
    lines = _lines(fig)
    n = n_filled_i = 0
    for pid in PIDS:
        for fam in ("unconditional", "conditional"):
            for e in inp.backtest[pid]["dm"][fam]:
                # D52 (3): fill = the JSON verdict field (family (i) unadjusted, (ii) Holm)
                ln = lines[f"{pid}|{e['a']}|{e['b']}|{e['reject']}"]
                assert e["dm"] in ln.get_xdata()
                assert ln.get_marker() == fg.PASS_MARKERS[pid]
                assert ln.get_markerfacecolor() == ("black" if e["reject"] else "white")
                n_filled_i += fam == "unconditional" and e["reject"]
                n += 1
    assert n == 3 * 12 and n_filled_i == 3 * 2  # the fixture rejects one (i) pair per level
    assert fig.axes[0].get_xlabel() == "DM statistic of FZ0 difference A - B (< 0: A scores better)"
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "family (i): reject at 5% (unadjusted)" in legend
    assert "family (ii): Holm reject at 5%" in legend
    ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert ticks[0] == "Gaussian vs HS"
    refs = sorted(
        {
            ln.get_xdata()[0]
            for ax in fg.f3_dm(inp.backtest).axes
            for ln in ax.lines
            if ln.get_label().startswith("_")
        }
    )
    assert refs == [-1.96, 0.0, 1.96]


def test_f4_blocks_partial_and_table(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f4_traffic_light(inp.tl)
    rects = _rects(fig)
    full = [lab for lab in rects if lab.split("|")[2] != "partial"]
    partial = [lab for lab in rects if lab.split("|")[2] == "partial"]
    assert len(full) == len(PIDS) * len(MODELS) * NB
    assert len(partial) == len(PIDS) * len(MODELS)
    for pid in PIDS:
        for i, k in enumerate(MODELS):
            e = inp.tl["passes"][pid]["keys"][k]
            for b in e["blocks"]:
                r = rects[f"{pid}|{k}|{b['block']}|{b['zone']}"]
                assert r.get_x() == b["block"]
                assert r.get_facecolor() == pytest.approx(
                    plt.matplotlib.colors.to_rgba(fg.ZONE_COLOURS[b["zone"]])
                )
            r = rects[f"{pid}|{k}|partial|40|{i}"]
            assert r.get_x() == NB and r.get_facecolor() == pytest.approx((1, 1, 1, 1))
    strip = fig.axes[0]
    assert [t.get_text() for t in strip.get_xticklabels()] == [*map(str, range(NB)), "P (40 d)"]
    table = fig.axes[-1].tables[0]
    cells = table.get_celld()
    assert cells[(0, 1)].get_text().get_text() == "PR W1000 (primary): green / yellow / red"
    for row, k in enumerate(KEYS, start=1):
        assert cells[(row, 0)].get_text().get_text() == fg.display(k)
        z = fg._counts(inp.tl["passes"]["tri_w1000"]["keys"][k])["zones"]
        assert cells[(row, 3)].get_text().get_text() == f"{z['green']} / {z['yellow']} / {z['red']}"


def test_f4_without_a_partial_block(inp: fg.Inputs, close_figs: Any) -> None:
    tl = json.loads(json.dumps(inp.tl))
    for p in tl["passes"].values():
        for e in p["keys"].values():
            e["partial_block"] = None
    fig = fg.f4_traffic_light(tl)
    assert not [lab for lab in _rects(fig) if "partial" in lab]
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == list(map(str, range(NB)))


def test_f5_sensitivity_values(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f5_sensitivity(inp.backtest, inp.training, inp.oos_blocks)
    lines = {}
    for c, pid in enumerate(PIDS):
        for r in range(4):
            for ln in fig.axes[r * 3 + c].lines:
                lines[(pid, ln.get_label())] = ln
            vlines = [ln for ln in fig.axes[r * 3 + c].lines if ln.get_linestyle() == ":"]
            assert [ln.get_xdata()[0] for ln in vlines] == [0.88]
    for pid in PIDS:
        b = inp.backtest[pid]
        for base in ("pot_gpd", "garch_evt"):
            keys = [f"{base}@0.86", base, f"{base}@0.9", f"{base}@0.91"]
            for lvl in fg.LEVELS:
                ln = lines[(pid, f"{base}|fz0|{lvl}")]
                assert list(ln.get_xdata()) == [0.86, 0.88, 0.9, 0.91]
                ref = b["fz0"][lvl][base]["mean"]  # D52 (4): differences from q_ref
                assert list(ln.get_ydata()) == [b["fz0"][lvl][k]["mean"] - ref for k in keys]
                ln = lines[(pid, f"{base}|rate|{lvl}")]
                assert list(ln.get_ydata()) == pytest.approx(
                    [100 * b["coverage"][lvl][k]["p_hat"] for k in keys]
                )
                assert ln.get_color() == fg.MODEL_COLOURS[base]


def test_f5_refuses_inconsistent_q_c(inp: fg.Inputs) -> None:
    bad = json.loads(json.dumps(OOS_BLOCKS))
    bad["residuals"][1]["check"]["q_ref"] = "9/10"
    with pytest.raises(fg.FiguresInputError, match="q_c"):
        fg.f5_sensitivity(inp.backtest, inp.training, bad)


def test_f6_points_and_bands(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f6_oos_blocks(inp.oos_blocks, inp.config)
    lines = _lines(fig)
    assert list(lines["raw|q_star"].get_ydata()) == [0.8, 0.81]  # B3 rule failure skipped
    assert list(lines["raw|q_star"].get_xdata()) == [-0.1, 0.9]
    assert list(lines["residuals|q_star"].get_ydata()) == [0.82, 0.8]  # garch_error skipped
    assert list(lines["raw|var99_rel"].get_ydata()) == pytest.approx([1.0, 2.0])
    assert list(lines["raw|es99_rel"].get_ydata()) == pytest.approx([-2.0])
    assert list(lines["residuals|es99_rel"].get_ydata()) == pytest.approx([4.0, 5.0])
    for ax, tol in zip(fig.axes[1:], (5.0, 10.0), strict=True):
        band = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert [(p.get_y(), p.get_y() + p.get_height()) for p in band] == [
            pytest.approx((-tol, tol))
        ]
    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert ticks == ["B1", "B2", "B3\n(overlapping,\ndescriptive)"]
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "F4 residuals (descriptive only)" in legend
    refs = [ln.get_ydata()[0] for ln in fig.axes[0].lines if ln.get_linestyle() == "--"]
    assert refs == [0.88]


def test_f7_xi_blocks(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f7_xi_blocks(inp.xi_path)
    lines = _lines(fig)
    assert list(lines["pot_gpd|xi_median|full"].get_ydata()) == pytest.approx([0.0, 0.1, 0.2])
    assert list(lines["pot_gpd|xi_median|partial"].get_xdata()) == [3]
    assert list(lines["garch_evt|xi_median|full"].get_xdata()) == [0, 2]  # None skipped
    assert list(lines["pot_gpd|share_ci_lo_gt_0|full"].get_ydata()) == pytest.approx(
        [0.0, 20.0, 40.0]
    )
    band = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert [(p.get_y(), p.get_y() + p.get_height()) for p in band] == [pytest.approx((-0.01, 0.27))]
    assert "descriptive; overlapping windows" in fig.texts[0].get_text()


def test_f8_bars_expected_and_p(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f8_descriptive(inp.backtest)
    for r, lvl in enumerate(fg.DESCRIPTIVE_LEVELS):
        for c, pid in enumerate(PIDS):
            ax = fig.axes[r * 3 + c]
            bars = {cont.get_label(): cont for cont in ax.containers}
            texts = [t.get_text() for t in ax.texts]
            for k in MODELS:
                cell = inp.backtest[pid]["coverage"][lvl][k]
                assert bars[f"{k}|{lvl}"].patches[0].get_height() == cell["breaches"]
                assert f"p={cell['p_binomial']:.3g}" in texts
            exp = sorted(seg[0][1] for coll in ax.collections for seg in coll.get_segments())
            want = sorted(
                inp.backtest[pid]["coverage"][lvl][k]["n"] * (1 - float(lvl)) for k in MODELS
            )
            assert exp == pytest.approx(want)


def test_builders_open_no_file(
    inp: fg.Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Builders receive parsed JSON only: rendering every figure opens nothing in the repository."""
    with audit_reads(PROJECT_ROOT, monkeypatch) as seen:
        files = fg.render_all(dataclasses_replace_sha(inp), "d" * 40)
    assert {p for p in seen if not p.startswith(".venv/")} == set()  # matplotlib's own fonts aside
    assert set(files) == {f"reports/figures/{s.file}" for s in fg.FIGURES} | {"docs/figures.md"}


def dataclasses_replace_sha(inp: fg.Inputs) -> fg.Inputs:
    paths = {p for s in fg.FIGURES for p in s.inputs} | {fg.CONFIG.as_posix()}
    return dataclasses.replace(inp, sha256={p: "e" * 64 for p in paths})


# --- command ----------------------------------------------------------------------------------
def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _write(root: Path, rel: str, obj: Any) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(json.dumps(obj, sort_keys=True).encode())


def make_repo(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inp = make_inputs()
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    cfg_sha = _sha((root / CONFIG).read_bytes())
    for pid, path in fg.BACKTEST.items():
        _write(root, path, inp.backtest[pid])
    _write(root, fg.W1000_MANIFEST, {"config": {"sha256": cfg_sha}})
    _write(root, fg.OOS_BLOCKS, OOS_BLOCKS)
    _write(root, fg.XI_PATH, XI)
    _write(root, fg.TRAINING, TRAINING)
    _write(root, fg.TL, inp.tl)
    expected = {rel: _sha((root / rel).read_bytes()) for rel in fg.EXPECTED_SHA256}
    monkeypatch.setattr(fg, "EXPECTED_SHA256", expected)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")


def _run(root: Path, verify: bool = False) -> tuple[int, list[dict[str, Any]]]:
    buf = io.StringIO()
    status = fg.run(verify, root, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


def _outputs() -> list[str]:
    return [(fg.FIG_DIR / s.file).as_posix() for s in fg.FIGURES] + [fg.DOC.as_posix()]


def test_run_verify_determinism_and_doc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path, monkeypatch)
    commit = oos.clean_commit(tmp_path)
    with audit_reads(tmp_path, monkeypatch) as seen:
        status, logs = _run(tmp_path)
    assert status == 0 and logs[-1]["message"] == "OK"
    assert seen == {*fg.EXPECTED_SHA256, CONFIG, fg.TL}  # D51 (a), (12): nothing else
    files = {n: (tmp_path / n).read_bytes() for n in _outputs()}
    assert len(files) == 9
    inputs = fg.read_inputs(tmp_path)
    assert fg.render_all(inputs, commit) == files  # same commit -> identical bytes
    assert files["docs/figures.md"].decode("utf-8") == fg.render_doc(inputs.sha256, commit)
    for s in fg.FIGURES:
        text = files[f"reports/figures/{s.file}"].decode("utf-8")
        assert not ISO_DATE.search(text) and not MONTH_DATE.search(text), s.file
        assert f"<dc:title>{s.fid} {s.file}</dc:title>" in text
        for p in s.inputs:
            assert f"{p} sha256 {inputs.sha256[p]}" in text
        assert f"commit {commit}" in text
    doc = files["docs/figures.md"].decode("utf-8")
    assert not ISO_DATE.search(doc) and not MONTH_DATE.search(doc)
    assert all(lab in doc for lab in fg.PASS_LABELS.values())
    status, logs = _run(tmp_path, verify=True)
    assert status == 0 and logs[-1]["message"] == "verify: all hashes match"
    assert {n: (tmp_path / n).read_bytes() for n in _outputs()} == files  # verify writes nothing
    (tmp_path / "reports/figures/dm.svg").write_bytes(files["reports/figures/dm.svg"] + b" ")
    assert _run(tmp_path, verify=True)[0] == 1
    with pytest.raises(oos.OosError, match="clean tree"):
        _run(tmp_path)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "figures")
    assert _run(tmp_path)[0] == 1  # no overwrite


def _tamper(root: Path, rel: str, edit: Any) -> None:
    obj = json.loads((root / rel).read_bytes())
    edit(obj)
    _write(root, rel, obj)
    _git(root, "commit", "-qam", "tamper")


@pytest.mark.parametrize(
    ("what", "match"),
    [
        ("hash", "sha256 differs from docs/protected_hashes.md"),
        ("config", "base.toml differs"),
        ("tl_counts", "zone counts differ"),
        ("tl_config", "config or passes differ"),
        ("models", "unexpected primary keys"),
    ],
)
def test_input_checks_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, what: str, match: str
) -> None:
    make_repo(tmp_path, monkeypatch)
    if what == "hash":
        _tamper(tmp_path, fg.XI_PATH, lambda o: o.update(extra=1))
    elif what == "config":
        _tamper(tmp_path, fg.W1000_MANIFEST, lambda o: o.update(config={"sha256": "0" * 64}))
        fg.EXPECTED_SHA256[fg.W1000_MANIFEST] = _sha((tmp_path / fg.W1000_MANIFEST).read_bytes())
    elif what == "tl_counts":
        _tamper(
            tmp_path,
            fg.TL,
            lambda o: o["passes"]["pr_w2000"]["keys"]["hs"]["blocks"][0].update(zone="red"),
        )
    elif what == "tl_config":
        _tamper(tmp_path, fg.TL, lambda o: o["provenance"].update(config_sha256="0" * 64))
    else:
        path = fg.BACKTEST["tri_w1000"]
        _tamper(tmp_path, path, lambda o: o.update(primary_keys=list(reversed(MODELS))))
        fg.EXPECTED_SHA256[path] = _sha((tmp_path / path).read_bytes())
    with pytest.raises(fg.FiguresInputError, match=match):
        _run(tmp_path)
    assert not (tmp_path / fg.FIG_DIR).exists() and not (tmp_path / fg.DOC).exists()


def test_expected_hashes_equal_the_protected_list() -> None:
    protected = (PROJECT_ROOT / "docs/protected_hashes.md").read_text(encoding="utf-8")
    listed = dict(re.findall(r"([\w./-]+\.(?:json|md|svg)) \(([0-9a-fA-F]{64})", protected))
    for rel, sha in fg.EXPECTED_SHA256.items():
        assert listed[rel].lower() == sha, rel


def test_model_colours_are_okabe_ito_and_fixed() -> None:
    okabe_ito = {"#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"}
    assert set(fg.MODEL_COLOURS.values()) <= okabe_ito
    assert len(set(fg.MODEL_COLOURS.values())) == len(fg.MODEL_COLOURS) == 6


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert _cli("figures", "configs/missing.toml")[0] == 1
    status, log = _cli("figures", CONFIG)
    assert status == 0 and log["message"] == "OK"
    status, log = _cli("figures", CONFIG, "--verify")
    assert status == 0 and log["message"] == "verify: all hashes match"
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    status, log = _cli("figures", CONFIG, "--verify")
    assert status == 1 and "clean tree" in log["error"]


def test_d52_labels(inp: fg.Inputs, close_figs: Any) -> None:
    fig = fg.f5_sensitivity(inp.backtest, inp.training, inp.oos_blocks)
    for r in range(2):  # the FZ0 rows: a zero line and the short Day-13-0a label
        ax = fig.axes[r * 3]
        assert ax.get_ylabel() == f"VaR {fg.LEVELS[r]}\n\u0394FZ0 vs q = 0.88"
        assert [ln.get_ydata()[0] for ln in ax.lines if ln.get_linestyle() == "-"][-1] == 0.0
    assert fig.axes[6].get_ylabel() == "VaR 0.95\nbreach rate (%)"
    assert [fig.axes[i].get_ylabel() for i in range(12) if i % 3] == [""] * 8  # left column only
    assert "\u0394FZ0 < 0: better than q = 0.88" in [x.get_text() for x in fig.texts]
    fig6 = fg.f6_oos_blocks(inp.oos_blocks, inp.config)
    assert fg.F6_PASS in fig6.texts[0].get_text()
    assert fg.F6_PASS == "PR OOS losses (D45; window-independent)"
    doc = fg.render_doc(dataclasses_replace_sha(inp).sha256, "d" * 40)
    assert doc.count("Display names (D52)") == 1 and "`pot_gpd` = POT-GPD" in doc
    assert f"Passes shown: {fg.F6_PASS}." in doc


def test_replace_rerenders_only_the_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D54: refuse without --replace; refuse --replace on a dirty tree; replace only the outputs."""
    make_repo(tmp_path, monkeypatch)
    buf = io.StringIO()
    log = configure_logging(stream=buf)
    assert fg.run(False, tmp_path, log) == 0
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "figures")
    before = {n: (tmp_path / n).read_bytes() for n in _outputs()}
    assert fg.run(False, tmp_path, log) == 1  # existing outputs, no --replace
    assert "use --replace" in buf.getvalue()
    assert fg.run(True, tmp_path, log, replace=True) == 1  # exclusive with --verify
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(oos.OosError, match="clean tree"):
        fg.run(False, tmp_path, log, replace=True)
    (tmp_path / "x.txt").unlink()
    assert fg.run(False, tmp_path, log, replace=True) == 0
    changed = tl_blocks.changed_paths(tmp_path)
    assert changed and changed <= set(_outputs())  # only the listed paths were touched
    commit = _head(tmp_path)
    after = {n: (tmp_path / n).read_bytes() for n in _outputs()}
    assert after != before and all(
        f"commit {commit}".encode() in after[n] for n in after if n.endswith(".svg")
    )
    assert fg.run(True, tmp_path, log) == 0  # the replaced files verify


def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_cli_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert _cli("figures", CONFIG)[0] == 0
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "figures")
    assert _cli("figures", CONFIG)[0] == 1
    status, log = _cli("figures", CONFIG, "--replace")
    assert status == 0 and log["message"] == "OK"


def _texts(ax: Any) -> list[Any]:
    """Labels, title and the tick labels that are drawn (tick inside the view interval)."""
    out = [ax.yaxis.label, ax.xaxis.label, ax.title]
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        out += [tick.label1 for tick in axis.get_major_ticks() if lo <= tick.get_loc() <= hi]
    return [x for x in out if x.get_text() and x.get_visible()]


def test_f5_labels_overlap_no_other_axes(inp: fg.Inputs, close_figs: Any) -> None:
    """Day 13-0a: no label, tick label or title of one panel overlaps another panel's frame or
    texts, on the fixture and on the published inputs."""
    for data in (inp, fg.read_inputs(PROJECT_ROOT)):
        fig = fg.f5_sensitivity(data.backtest, data.training, data.oos_blocks)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        boxes = [
            [ax.get_window_extent(renderer)] + [x.get_window_extent(renderer) for x in _texts(ax)]
            for ax in fig.axes
        ]
        n = 0
        for i, ax in enumerate(fig.axes):
            for txt in _texts(ax):
                bb = txt.get_window_extent(renderer)
                n += 1
                for j, others in enumerate(boxes):
                    if j != i:
                        assert not any(o.overlaps(bb) for o in others), (i, j, txt.get_text())
        assert n > 0
        plt.close(fig)
