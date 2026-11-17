"""Day-12B tracked figures (D51), built from tracked JSON only.

``read_inputs`` reads the D51 (a) files by path and checks each before use; the builders take the
parsed JSON and return matplotlib figures (no file I/O); ``run`` renders every figure to SVG with
fixed metadata, writes ``reports/figures/`` and the generated ``docs/figures.md``, or with
``--verify`` compares the in-memory bytes with the files on disk. No figure shows a daily series.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # select the non-interactive backend before pyplot is imported

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from niftytail import backtest, oos, tl_blocks

__all__ = [
    "CONFIG",
    "DISPLAY_NAMES",
    "DOC",
    "EXPECTED_SHA256",
    "FIGURES",
    "FIG_DIR",
    "MODEL_COLOURS",
    "FiguresInputError",
    "Inputs",
    "display",
    "p_text",
    "read_inputs",
    "render_all",
    "render_doc",
    "run",
    "verdict",
]

SVG_HASHSALT = "niftytail-D51"  # fixed, so that matplotlib's SVG ids are reproducible
FIG_DIR = Path("reports/figures")
DOC = Path("docs/figures.md")
CONFIG = Path("configs/base.toml")
BACKTEST = {pid: (tdir / oos.BACKTEST).as_posix() for pid, _, _, tdir, _, _ in tl_blocks.PASSES}
PASS_LABELS = {pid: label for pid, label, *_ in tl_blocks.PASSES}
W1000_MANIFEST = "reports/oos/w1000/manifest.json"
OOS_BLOCKS = "reports/threshold_selection/oos_blocks/selection.json"
XI_PATH = "reports/oos/xi_path/summary.json"
TRAINING = "reports/threshold_selection/real/selection.json"
TL = tl_blocks.OUT.as_posix()
# D51 (a): the protected hashes (a test asserts these equal the docs/protected_hashes.md list).
EXPECTED_SHA256 = {
    BACKTEST["pr_w1000"]: "bc79fd419d16ae5ac2cf03800ef6b505088bd9ea27fc559078d78cdb35b7f20a",
    BACKTEST["pr_w2000"]: "f900783ca58644aae3fba27fe51f6b936107959ed01366178ee64e29b44b5f12",
    BACKTEST["tri_w1000"]: "425aba7b13494c4af47a9b689d624966084870a948c999947daec266ac4006ad",
    W1000_MANIFEST: "289f603029705c284b0e1f3996a36fd9ae38b44a6944b883ccce544bed9781b5",
    OOS_BLOCKS: "5a64bb708a7714f13fdedc7846857c7d8c7b24e6f4c9a68a755b94e75d3cd49c",
    XI_PATH: "e2ac9882bd7cd826ff80f46ee9b79a28ab45f2025f0f887c2bc2b300af92c30f",
    TRAINING: "7929c129a03509cb31b56cea1804cbfd4559e99c09b960d86b658b86e86741f9",
}
# D51 (7): Okabe-Ito, one colour per model in every figure.
MODEL_COLOURS = {
    "gaussian": "#E69F00",
    "hs": "#56B4E9",
    "pot_gpd": "#009E73",
    "garch_n": "#0072B2",
    "fhs": "#D55E00",
    "garch_evt": "#CC79A7",
}
MODELS = tuple(MODEL_COLOURS)
# D52 (1): display names; JSON keys are unchanged.
DISPLAY_NAMES = {
    "gaussian": "Gaussian",
    "hs": "HS",
    "pot_gpd": "POT-GPD",
    "garch_n": "GARCH-N",
    "fhs": "FHS",
    "garch_evt": "GARCH-EVT",
}
F6_PASS = "PR OOS losses (D45; window-independent)"  # D52 (5)
ZONE_COLOURS = {"green": "#009E73", "yellow": "#F0E442", "red": "#D55E00"}
REJECT_FILL, ACCEPT_FILL, MISSING_FILL = "#BBBBBB", "#FFFFFF", "#EEEEEE"
PASS_MARKERS = {"pr_w1000": "o", "pr_w2000": "s", "tri_w1000": "^"}
LEVELS = ("0.95", "0.99")
DESCRIPTIVE_LEVELS = ("0.995", "0.999")
TEST_LEVEL = backtest.TEST_LEVEL
BORDERLINE = (0.01, 0.10)  # D49 T6, closed interval
TESTS = ("Kupiec", "IND", "CC", "ES")


class FiguresInputError(oos.OosError):
    """An input failed its D51 (a) check; nothing is written."""


@dataclass(frozen=True)
class Inputs:
    """The parsed D51 (a) inputs and the sha256 of every file read."""

    backtest: dict[str, dict[str, Any]]  # pass id -> backtest.json
    oos_blocks: dict[str, Any]
    xi_path: dict[str, Any]
    training: dict[str, Any]
    tl: dict[str, Any]
    config: dict[str, Any]  # base.toml, parsed by tomllib
    sha256: dict[str, str]  # repository-relative path -> sha256


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _counts(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Zone counts of one D51 (4) key entry, in the tracked backtest.json layout."""
    blocks = list(entry["blocks"])
    if entry["partial_block"] is not None:
        blocks.append({**entry["partial_block"], "zone": None, "inadmissible_rows": 0})
    return backtest._zone_counts(blocks)


def read_inputs(root: str | Path) -> Inputs:
    """Read and check every D51 (a) input (D51 (6)).

    Raises:
        FiguresInputError: a sha256 differs from the protected-hash list, base.toml differs from
            the run config, or traffic_light_blocks.json does not reconcile with the tracked
            backtest.json.
        OSError: a missing input.
    """
    base = Path(root)
    raw = {rel: (base / rel).read_bytes() for rel in (*EXPECTED_SHA256, CONFIG.as_posix(), TL)}
    sha = {rel: _sha(b) for rel, b in raw.items()}
    for rel, want in EXPECTED_SHA256.items():
        if sha[rel] != want:
            raise FiguresInputError(f"{rel}: sha256 differs from docs/protected_hashes.md; STOP")
    cfg_sha = sha[CONFIG.as_posix()]
    if json.loads(raw[W1000_MANIFEST])["config"]["sha256"] != cfg_sha:
        raise FiguresInputError("configs/base.toml differs from the published run config; STOP")
    bt = {pid: json.loads(raw[path]) for pid, path in BACKTEST.items()}
    for pid, b in bt.items():
        if tuple(b["primary_keys"]) != MODELS:
            raise FiguresInputError(f"{pid}: unexpected primary keys")
    tl = json.loads(raw[TL])
    if tl["provenance"]["config_sha256"] != cfg_sha or set(tl["passes"]) != set(bt):
        raise FiguresInputError(f"{TL}: config or passes differ; STOP")
    for pid, b in bt.items():
        for key, counts in b["traffic_light"].items():
            if _counts(tl["passes"][pid]["keys"][key]) != counts:
                raise FiguresInputError(f"{TL}: {pid} {key} zone counts differ; STOP")
    return Inputs(
        backtest=bt,
        oos_blocks=json.loads(raw[OOS_BLOCKS]),
        xi_path=json.loads(raw[XI_PATH]),
        training=json.loads(raw[TRAINING]),
        tl=tl,
        config=tomllib.loads(raw[CONFIG.as_posix()].decode("utf-8")),
        sha256=sha,
    )


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------
def verdict(p: float | None, reject: bool | None = None) -> tuple[bool | None, bool]:
    """D51 F2: (rejected, borderline). ``reject`` (Kupiec) overrides ``p <= TEST_LEVEL``."""
    if p is None:
        return None, False
    rejected = reject if reject is not None else p <= TEST_LEVEL
    return rejected, BORDERLINE[0] <= p <= BORDERLINE[1]


def display(key: str) -> str:
    """D52 (1): "POT-GPD", or "POT-GPD q=0.86" for a sensitivity key."""
    base, _, q = key.partition("@")
    return DISPLAY_NAMES[base] + (f" q={q}" if q else "")


@cache
def _has_glyph(char: str) -> bool:
    """Whether matplotlib's default font has ``char`` (D52 (6) fallback)."""
    path = font_manager.findfont(font_manager.FontProperties())
    return ord(char) in font_manager.get_font(path).get_charmap()


def _le() -> str:
    return "\u2264" if _has_glyph("\u2264") else "<="


def _exp(x: float) -> str:
    mant, _, ex = f"{x:.0e}".partition("e")
    return f"{mant}e{int(ex)}"


def p_text(p: float | None, reps: int | None) -> str:
    """D52 (6): "\u22641e-4" at the MC floor 1/(B+1), else 3 significant digits; "n/a" if no p."""
    if p is None:
        return "n/a"
    if reps is not None and p <= 1 / (reps + 1):
        return f"{_le()}{_exp(1 / (reps + 1))}"
    return f"{p:.3g}"


def _test_cell(
    test: str, c: Mapping[str, Any], es_reps: int
) -> tuple[float | None, bool | None, int | None]:
    """(deciding p, JSON reject field or None, MC reps or None) for one coverage cell."""
    if test == "Kupiec":
        return c["kupiec"]["p"], c["kupiec"]["reject"], None
    if test in ("IND", "CC"):
        ch = c["christoffersen"]
        return ch["p_ind" if test == "IND" else "p_cc"], None, ch["reps"]
    return c["es_residual"]["p"], None, es_reps


def _frac(s: str) -> float:
    return float(Fraction(s))


def _legend(ax: Axes, handles: Sequence[Any], **kw: Any) -> None:
    ax.legend(handles=list(handles), fontsize=7, framealpha=0.9, **kw)


def _model_axis(ax: Axes, models: Sequence[str]) -> None:
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([display(m) for m in models], rotation=30, ha="right", fontsize=8)


# ---------------------------------------------------------------------------
# builders: parsed JSON in, Figure out (D51 (b))
# ---------------------------------------------------------------------------
def f1_coverage(bt: Mapping[str, Mapping[str, Any]]) -> Figure:
    """Breach rate with Clopper-Pearson and block CIs per model, 0.95 and 0.99, three passes."""
    fig, axes = plt.subplots(2, len(bt), figsize=(13, 7.5), sharey="row", squeeze=False)
    for r, lvl in enumerate(LEVELS):
        for c, (pid, b) in enumerate(bt.items()):
            ax = axes[r][c]
            for i, k in enumerate(b["primary_keys"]):
                cell = b["coverage"][lvl][k]
                p = 100 * cell["p_hat"]
                for dx, ci, fmt, tag in (
                    (-0.13, cell["ci"]["clopper_pearson"], "o", "cp"),
                    (
                        0.13,
                        (cell["ci"]["block"]["lower"], cell["ci"]["block"]["upper"]),
                        "s",
                        "block",
                    ),
                ):
                    ax.errorbar(
                        [i + dx],
                        [p],
                        yerr=[[p - 100 * ci[0]], [100 * ci[1] - p]],
                        fmt=fmt,
                        color=MODEL_COLOURS[k],
                        mfc="white" if tag == "block" else MODEL_COLOURS[k],
                        capsize=3,
                        linewidth=1.2,
                        markersize=5,
                        label=f"{k}:{tag}",
                    )
            ax.axhline(100 * (1 - _frac(lvl)), color="black", linestyle="--", linewidth=0.9)
            _model_axis(ax, b["primary_keys"])
            ax.set_title(f"{PASS_LABELS[pid]}, VaR {lvl}", fontsize=9)
            if c == 0:
                ax.set_ylabel("breach rate (% of OOS days)")
    fig.legend(
        handles=[
            Line2D(
                [], [], color="black", marker="o", linestyle="-", label="rate, Clopper-Pearson CI"
            ),
            Line2D([], [], color="black", marker="s", mfc="white", label="rate, block CI"),
            Line2D([], [], color="black", linestyle="--", label="nominal rate 1 - level"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
    )
    fig.suptitle(
        "Breach rate per model with Clopper-Pearson and block CIs (VaR 0.95, 0.99)", fontsize=10
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def f2_tests(bt: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> Figure:
    """Verdict at 5% per model x test x level x pass; hatch = borderline; p printed."""
    es_reps = int(config["backtest"]["es_residual_bootstrap_reps"])
    cols = [(t, lvl) for lvl in LEVELS for t in TESTS]
    fig, axes = plt.subplots(1, len(bt), figsize=(16, 5.2), squeeze=False)
    for c, (pid, b) in enumerate(bt.items()):
        ax = axes[0][c]
        models = list(b["primary_keys"])
        for r, k in enumerate(models):
            for x, (test, lvl) in enumerate(cols):
                p, field, reps = _test_cell(test, b["coverage"][lvl][k], es_reps)
                rejected, border = verdict(p, field)
                fill = (
                    MISSING_FILL if rejected is None else (REJECT_FILL if rejected else ACCEPT_FILL)
                )
                ax.add_patch(
                    Rectangle(
                        (x, r),
                        1,
                        1,
                        facecolor=fill,
                        edgecolor="black",
                        linewidth=0.4,
                        hatch="////" if border else None,
                        label=f"{pid}|{k}|{lvl}|{test}|{rejected}|{border}",
                    )
                )
                ax.text(
                    x + 0.5,
                    r + 0.5,
                    p_text(p, reps),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    gid=f"{k}|{lvl}|{test}",
                    bbox={"facecolor": fill, "edgecolor": "none", "pad": 1.0},
                )
        ax.set_xlim(0, len(cols))
        ax.set_ylim(len(models), 0)
        ax.set_xticks([x + 0.5 for x in range(len(cols))])
        ax.set_xticklabels([f"{t}\n{lvl}" for t, lvl in cols], fontsize=7)
        ax.set_yticks([r + 0.5 for r in range(len(models))])
        ax.set_yticklabels([display(m) for m in models] if c == 0 else [], fontsize=8)
        ax.set_xlabel("test and VaR level (p-value in the cell)")
        ax.set_title(PASS_LABELS[pid], fontsize=9)
    fig.legend(
        handles=[
            Patch(facecolor=REJECT_FILL, edgecolor="black", label="rejected at 5%"),
            Patch(facecolor=ACCEPT_FILL, edgecolor="black", label="not rejected at 5%"),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="////",
                label="borderline: deciding p in [0.01, 0.10]",
            ),
            Patch(facecolor=MISSING_FILL, edgecolor="black", label="no p (n/a)"),
        ],
        loc="lower center",
        ncol=4,
        fontsize=8,
    )
    fig.suptitle(
        "Coverage-test verdicts at 5% (Kupiec, Christoffersen IND / CC, ES residual)", fontsize=10
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    return fig


def f3_dm(bt: Mapping[str, Mapping[str, Any]]) -> Figure:
    """DM statistic per pair, families (i) and (ii), per level and pass; ±1.96 lines."""
    fams = (
        ("unconditional", "family (i), unconditional"),
        ("conditional", "family (ii), conditional, Holm"),
    )
    fig, axes = plt.subplots(2, len(LEVELS), figsize=(12, 7), squeeze=False)
    first = next(iter(bt.values()))
    for r, (fam, title) in enumerate(fams):
        for c, lvl in enumerate(LEVELS):
            ax = axes[r][c]
            pairs = [(e["a"], e["b"]) for e in first["dm"][fam] if e["level"] == lvl]
            for j, (pid, b) in enumerate(bt.items()):
                ent = {(e["a"], e["b"]): e for e in b["dm"][fam] if e["level"] == lvl}
                for i, pair in enumerate(pairs):
                    e = ent[pair]
                    filled = bool(e["reject"])  # (i) unadjusted verdict, (ii) Holm (D52 (3))
                    ax.plot(
                        [e["dm"]],
                        [i + (j - 1) * 0.22],
                        marker=PASS_MARKERS[pid],
                        color="black",
                        mfc="black" if filled else "white",
                        linestyle="none",
                        markersize=6,
                        label=f"{pid}|{pair[0]}|{pair[1]}|{filled}",
                    )
            for x, style in ((-1.96, "--"), (1.96, "--"), (0.0, "-")):
                ax.axvline(x, color="black", linestyle=style, linewidth=0.7 if x else 0.5)
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{display(a)} vs {display(b_)}" for a, b_ in pairs], fontsize=8)
            ax.set_ylim(len(pairs) - 0.5, -0.5)
            ax.set_xlabel("DM statistic of FZ0 difference A - B (< 0: A scores better)")
            ax.set_title(f"{title}, VaR {lvl}", fontsize=9)
    handles = [
        Line2D(
            [],
            [],
            marker=PASS_MARKERS[pid],
            color="black",
            mfc="white",
            linestyle="none",
            label=PASS_LABELS[pid],
        )
        for pid in bt
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            color="black",
            mfc="black",
            linestyle="none",
            label="family (i): reject at 5% (unadjusted)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="black",
            mfc="black",
            linestyle="none",
            label="family (ii): Holm reject at 5%",
        ),
        Line2D(
            [], [], marker="o", color="black", mfc="white", linestyle="none", label="not rejected"
        ),
        Line2D([], [], color="black", linestyle="--", label="±1.96"),
    ]
    _legend(axes[0][0], handles, loc="best")
    fig.suptitle("Diebold-Mariano statistics on FZ0, test families (i) and (ii)", fontsize=10)
    fig.tight_layout()
    return fig


def f4_traffic_light(tl: Mapping[str, Any]) -> Figure:
    """A4 strip plot: zone by 250-day block per model, one panel per pass, plus the count table."""
    passes = list(tl["passes"].items())
    fig = plt.figure(figsize=(13, 12))
    grid = fig.add_gridspec(len(passes) + 1, 1, height_ratios=[1] * len(passes) + [1.5])
    for r, (pid, p) in enumerate(passes):
        ax = fig.add_subplot(grid[r])
        n_full = 0
        for row, k in enumerate(MODELS):
            entry = p["keys"][k]
            n_full = len(entry["blocks"])
            for bl in entry["blocks"]:
                ax.add_patch(
                    Rectangle(
                        (bl["block"], row),
                        1,
                        1,
                        facecolor=ZONE_COLOURS[bl["zone"]],
                        edgecolor="white",
                        linewidth=1.5,
                        label=f"{pid}|{k}|{bl['block']}|{bl['zone']}",
                    )
                )
                ax.text(
                    bl["block"] + 0.5,
                    row + 0.5,
                    str(bl["breaches"]),
                    ha="center",
                    va="center",
                    fontsize=6,
                )
            part = entry["partial_block"]
            if part is not None:
                ax.add_patch(
                    Rectangle(
                        (n_full, row),
                        1,
                        1,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.6,
                        label=f"{pid}|{k}|partial|{part['n_obs']}|{part['breaches']}",
                    )
                )
                ax.text(
                    n_full + 0.5,
                    row + 0.5,
                    str(part["breaches"]),
                    ha="center",
                    va="center",
                    fontsize=6,
                )
        part0 = p["keys"][MODELS[0]]["partial_block"]
        ticks = list(range(n_full)) + ([n_full] if part0 is not None else [])
        ax.set_xticks([t + 0.5 for t in ticks])
        ax.set_xticklabels(
            [str(t) for t in range(n_full)]
            + ([f"P ({part0['n_obs']} d)"] if part0 is not None else []),
            fontsize=7,
        )
        ax.set_xlim(0, len(ticks))
        ax.set_ylim(len(MODELS), 0)
        ax.set_yticks([i + 0.5 for i in range(len(MODELS))])
        ax.set_yticklabels([display(m) for m in MODELS], fontsize=8)
        ax.set_title(f"{p['label']}: zone by block, breaches printed", fontsize=9)
        ax.set_xlabel(
            "250-day block index (0 = first OOS block; P = partial block, no zone)", fontsize=8
        )
    _legend(
        fig.axes[0],
        [Patch(facecolor=c, label=z) for z, c in ZONE_COLOURS.items()]
        + [Patch(facecolor="white", edgecolor="black", label="partial block (no zone)")],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    ax = fig.add_subplot(grid[len(passes)])
    ax.axis("off")
    keys = [*MODELS, *sorted(k for k in passes[0][1]["keys"] if k not in MODELS)]
    zones = [[_counts(p["keys"][k])["zones"] for _, p in passes] for k in keys]
    cells = [
        [display(k), *(f"{z['green']} / {z['yellow']} / {z['red']}" for z in zs)]
        for k, zs in zip(keys, zones, strict=True)
    ]
    table = ax.table(
        cellText=cells,
        colLabels=["model"] + [f"{p['label']}: green / yellow / red" for _, p in passes],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.15)
    ax.set_title("Zone counts over the full blocks, VaR 0.99 (all 12 keys)", fontsize=9)
    fig.suptitle(
        f"Basel traffic light by 250-day block at VaR {tl['level']} (plan A4)", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    return fig


def f5_sensitivity(
    bt: Mapping[str, Mapping[str, Any]], training: Mapping[str, Any], oos_blocks: Mapping[str, Any]
) -> Figure:
    """Mean FZ0 and breach rate against q for pot_gpd and garch_evt, per level and pass."""
    q_sel = Fraction(training["selection"]["q_star"])
    q_cs = {Fraction(e["check"]["q_ref"]) for e in oos_blocks["residuals"] if e.get("check")}
    if len(q_cs) != 1:
        raise FiguresInputError("residual entries disagree on q_c")
    q_c = q_cs.pop()
    q_base = {"pot_gpd": q_sel, "garch_evt": q_c}
    fig, axes = plt.subplots(4, len(bt), figsize=(13, 12), sharex=True, squeeze=False)
    rows = [("fz0", lvl) for lvl in LEVELS] + [("rate", lvl) for lvl in LEVELS]
    for c, (pid, b) in enumerate(bt.items()):
        for r, (metric, lvl) in enumerate(rows):
            ax = axes[r][c]
            for base, q0 in q_base.items():
                keys = [base] + [k for k in b["sensitivity_keys"] if k.partition("@")[0] == base]
                pts = sorted(
                    (float(q0) if k == base else float(k.partition("@")[2]), k) for k in keys
                )
                ref = b["fz0"][lvl][base]["mean"]  # D52 (4): display difference from q_ref
                ys = [
                    b["fz0"][lvl][k]["mean"] - ref
                    if metric == "fz0"
                    else 100 * b["coverage"][lvl][k]["p_hat"]
                    for _, k in pts
                ]
                ax.plot(
                    [q for q, _ in pts],
                    ys,
                    "o-",
                    color=MODEL_COLOURS[base],
                    linewidth=1.2,
                    markersize=5,
                    label=f"{base}|{metric}|{lvl}",
                )
            if metric == "rate":
                ax.axhline(100 * (1 - _frac(lvl)), color="black", linestyle="--", linewidth=0.9)
            else:
                ax.axhline(0.0, color="black", linewidth=0.6)
            for q in sorted({q_sel, q_c}):
                ax.axvline(float(q), color="black", linestyle=":", linewidth=0.9)
            if c == 0:  # y labels on the left column only (Day 13-0a)
                ax.set_ylabel(
                    f"VaR {lvl}\n\u0394FZ0 vs q = {float(q_sel):g}"
                    if metric == "fz0"
                    else f"VaR {lvl}\nbreach rate (%)",
                    fontsize=8,
                )
            if r == 0:
                ax.set_title(PASS_LABELS[pid], fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("threshold quantile level q")
    marks = [(q_sel, "q_selected = q_c")] if q_sel == q_c else [(q_sel, "q_selected"), (q_c, "q_c")]
    _legend(
        axes[0][0],
        [Line2D([], [], color=MODEL_COLOURS[m], marker="o", label=display(m)) for m in q_base]
        + [
            Line2D([], [], color="black", linestyle=":", label=f"{n} = {float(q):g}")
            for q, n in marks
        ]
        + [Line2D([], [], color="black", linestyle="--", label="nominal rate 1 - level")],
        loc="best",
    )
    fig.suptitle(
        "Threshold sensitivity: mean FZ0 difference and breach rate against q (POT-GPD, GARCH-EVT)",
        fontsize=10,
    )
    fig.text(
        0.5, 0.005, f"\u0394FZ0 < 0: better than q = {float(q_sel):g}", ha="center", fontsize=8
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


def f6_oos_blocks(oos_blocks: Mapping[str, Any], config: Mapping[str, Any]) -> Figure:
    """D45 block q* (raw and residuals) against q_ref; var99_rel / es99_rel with tolerance bands."""
    tol = {
        "var99_rel": _frac(config["threshold"]["tol_var99_rel"]),
        "es99_rel": _frac(config["threshold"]["tol_es99_rel"]),
    }
    layout = oos_blocks["layout"]["blocks"]
    names = [b["block"] for b in layout]
    ticks = [
        f"{b['block']}\n(overlapping,\ndescriptive)" if b["overlapping"] else b["block"]
        for b in layout
    ]
    series = (
        ("raw", -0.1, MODEL_COLOURS["pot_gpd"], "raw OOS losses (post-hoc check)"),
        ("residuals", 0.1, MODEL_COLOURS["garch_evt"], "F4 residuals (descriptive only)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), squeeze=False)
    panels = (
        ("q_star", "block q* (quantile level)"),
        ("var99_rel", "VaR99(q*) / VaR99(q_ref) - 1 (%)"),
        ("es99_rel", "ES99(q*) / ES99(q_ref) - 1 (%)"),
    )
    for ax, (field, ylabel) in zip(axes[0], panels, strict=True):
        for name, dx, colour, _ in series:
            xs, ys = [], []
            for e in oos_blocks[name]:
                chk = e.get("check") or {}
                if field == "q_star":
                    v = (e.get("selection") or {}).get("q_star")
                    y = None if v is None else _frac(v)
                else:
                    v = chk.get(field)
                    y = None if v is None else 100 * v
                if y is not None:
                    xs.append(names.index(e["block"]) + dx)
                    ys.append(y)
            ax.plot(
                xs, ys, "o", color=colour, markersize=7, linestyle="none", label=f"{name}|{field}"
            )
        if field == "q_star":
            refs = sorted(
                {
                    _frac(e["check"]["q_ref"])
                    for n in ("raw", "residuals")
                    for e in oos_blocks[n]
                    if e.get("check")
                }
            )
            for q in refs:
                ax.axhline(q, color="black", linestyle="--", linewidth=0.9)
        else:
            ax.axhspan(-100 * tol[field], 100 * tol[field], color="#DDDDDD", zorder=0)
            ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("2000-day OOS block")
    _legend(
        axes[0][0],
        [
            Line2D([], [], color=c, marker="o", linestyle="none", label=lab)
            for _, _, c, lab in series
        ]
        + [Line2D([], [], color="black", linestyle="--", label="q_ref")],
        loc="best",
    )
    _legend(
        axes[0][1],
        [Patch(facecolor="#DDDDDD", label=f"tolerance ±{100 * tol['var99_rel']:g}%")],
        loc="best",
    )
    _legend(
        axes[0][2],
        [Patch(facecolor="#DDDDDD", label=f"tolerance ±{100 * tol['es99_rel']:g}%")],
        loc="best",
    )
    fig.suptitle(
        "D45 OOS-block threshold check: block q* against q_ref; VaR99 and ES99 relative change"
        f"\nPass: {F6_PASS}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def f7_xi_blocks(xi: Mapping[str, Any]) -> Figure:
    """Per-block median xi and share with ci_lo > 0 (D47), pot_gpd training xi and CI as a band."""
    keys = (
        ("pot_gpd", "POT-GPD (raw losses, q_selected)"),
        ("garch_evt", "GARCH-EVT (F4 residuals, q_c)"),
    )
    ref = xi["keys"]["pot_gpd"]["training_reference"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, squeeze=False)
    for r, (field, ylabel, scale) in enumerate(
        (
            ("xi_median", "median xi per block (GPD shape, unitless)", 1.0),
            ("share_ci_lo_gt_0", "share of targets with CI lower bound > 0 (%)", 100.0),
        )
    ):
        ax = axes[r][0]
        if field == "xi_median":
            ax.axhspan(
                ref["xi_ci"][0],
                ref["xi_ci"][1],
                color=MODEL_COLOURS["pot_gpd"],
                alpha=0.15,
                zorder=0,
            )
            ax.axhline(ref["xi_hat"], color=MODEL_COLOURS["pot_gpd"], linestyle="--", linewidth=0.9)
        for k, _ in keys:
            blocks = xi["keys"][k]["blocks"]
            full = [
                (b["block"], b[field]) for b in blocks if not b["partial"] and b[field] is not None
            ]
            part = [(b["block"], b[field]) for b in blocks if b["partial"] and b[field] is not None]
            ax.plot(
                [x for x, _ in full],
                [scale * y for _, y in full],
                "o-",
                color=MODEL_COLOURS[k],
                linewidth=1.1,
                markersize=4,
                label=f"{k}|{field}|full",
            )
            ax.plot(
                [x for x, _ in part],
                [scale * y for _, y in part],
                "o",
                color=MODEL_COLOURS[k],
                mfc="white",
                markersize=5,
                linestyle="none",
                label=f"{k}|{field}|partial",
            )
        ax.set_ylabel(ylabel, fontsize=8)
    axes[1][0].set_xlabel("250-target block index (PR W1000 path; open marker = partial block)")
    _legend(
        axes[0][0],
        [Line2D([], [], color=MODEL_COLOURS[k], marker="o", label=lab) for k, lab in keys]
        + [
            Patch(
                facecolor=MODEL_COLOURS["pot_gpd"],
                alpha=0.15,
                label=(
                    f"POT-GPD training xi {ref['xi_hat']:.3g},"
                    f" CI [{ref['xi_ci'][0]:.3g}, {ref['xi_ci'][1]:.3g}]"
                ),
            ),
        ],
        loc="best",
    )
    fig.suptitle(
        "Rolling xi path by block, POT-GPD vs GARCH-EVT (D47; descriptive; overlapping windows)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def f8_descriptive(bt: Mapping[str, Mapping[str, Any]]) -> Figure:
    """Breaches vs expected at 0.995 and 0.999 per model and pass, binomial p printed."""
    fig, axes = plt.subplots(len(DESCRIPTIVE_LEVELS), len(bt), figsize=(13, 7.5), squeeze=False)
    for r, lvl in enumerate(DESCRIPTIVE_LEVELS):
        for c, (pid, b) in enumerate(bt.items()):
            ax = axes[r][c]
            models = list(b["primary_keys"])
            top = 0.0
            for i, k in enumerate(models):
                cell = b["coverage"][lvl][k]
                expected = cell["n"] * (1 - _frac(lvl))
                ax.bar(
                    [i], [cell["breaches"]], color=MODEL_COLOURS[k], width=0.6, label=f"{k}|{lvl}"
                )
                ax.hlines(
                    expected, i - 0.38, i + 0.38, color="black", linestyle="--", linewidth=0.9
                )
                ax.text(
                    i,
                    max(cell["breaches"], expected) * 1.03 + 0.5,
                    f"p={p_text(cell['p_binomial'], None)}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
                top = max(top, cell["breaches"], expected)
            ax.set_ylim(0, top * 1.2 + 2)
            _model_axis(ax, models)
            ax.set_title(f"{PASS_LABELS[pid]}, VaR {lvl}", fontsize=9)
            if c == 0:
                ax.set_ylabel("breaches (OOS days)")
    _legend(
        axes[0][0],
        [Line2D([], [], color="black", linestyle="--", label="expected n x (1 - level)")],
        loc="upper left",
    )
    fig.suptitle(
        "Breaches vs expected at the descriptive levels 0.995 and 0.999, binomial p printed",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# registry, rendering, doc, command
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureSpec:
    fid: str
    file: str
    caption: str
    build: Callable[[Inputs], Figure]
    inputs: tuple[str, ...]
    passes: tuple[str, ...]  # pass labels as shown


_ALL = tuple(PASS_LABELS.values())
_BT = tuple(BACKTEST.values())
_CFG = (W1000_MANIFEST, CONFIG.as_posix())
FIGURES = (
    FigureSpec(
        "F1",
        "coverage.svg",
        "Breach rate per model with Clopper-Pearson and block CIs at VaR 0.95 and 0.99;"
        " nominal rate as reference.",
        lambda i: f1_coverage(i.backtest),
        _BT,
        _ALL,
    ),
    FigureSpec(
        "F2",
        "tests_heatmap.svg",
        "Kupiec, IND, CC and ES-residual verdicts at 5% per model, level and pass;"
        " hatch = deciding p in [0.01, 0.10]; p printed.",
        lambda i: f2_tests(i.backtest, i.config),
        (*_BT, *_CFG),
        _ALL,
    ),
    FigureSpec(
        "F3",
        "dm.svg",
        "Diebold-Mariano statistics on FZ0 for test families (i) and (ii) per level and pass;"
        " ±1.96 lines; Holm verdict for (ii).",
        lambda i: f3_dm(i.backtest),
        _BT,
        _ALL,
    ),
    FigureSpec(
        "F4",
        "traffic_light.svg",
        "Basel traffic-light zone by 250-day block at VaR 0.99 per model (plan A4),"
        " partial block unzoned; zone-count table.",
        lambda i: f4_traffic_light(i.tl),
        (TL, *_BT),
        _ALL,
    ),
    FigureSpec(
        "F5",
        "sensitivity.svg",
        "Mean FZ0 difference from q_ref and breach rate against the threshold q for POT-GPD"
        " and GARCH-EVT per level and pass; q_selected / q_c marked.",
        lambda i: f5_sensitivity(i.backtest, i.training, i.oos_blocks),
        (*_BT, TRAINING, OOS_BLOCKS),
        _ALL,
    ),
    FigureSpec(
        "F6",
        "oos_blocks.svg",
        "D45 block q* (raw, residuals) against q_ref, and VaR99 / ES99 relative change"
        " with the tolerance bands.",
        lambda i: f6_oos_blocks(i.oos_blocks, i.config),
        (OOS_BLOCKS, *_CFG),
        (F6_PASS,),
    ),
    FigureSpec(
        "F7",
        "xi_blocks.svg",
        "Per-block median xi and share of profile CIs with lower bound > 0, POT-GPD vs"
        " GARCH-EVT (D47); POT-GPD training xi and CI as a band.",
        lambda i: f7_xi_blocks(i.xi_path),
        (XI_PATH, TRAINING),
        (PASS_LABELS["pr_w1000"],),
    ),
    FigureSpec(
        "F8",
        "descriptive_levels.svg",
        "Breaches vs expected at VaR 0.995 and 0.999 per model and pass, binomial p printed.",
        lambda i: f8_descriptive(i.backtest),
        _BT,
        _ALL,
    ),
)


def _svg(fig: Figure, spec: FigureSpec, sha: Mapping[str, str], commit: str) -> bytes:
    desc = "; ".join(f"{p} sha256 {sha[p]}" for p in spec.inputs)
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="svg",
        metadata={
            "Date": None,
            "Title": f"{spec.fid} {spec.file}",
            "Description": f"inputs: {desc}; commit {commit}",
        },
    )
    plt.close(fig)
    return buf.getvalue().replace(b"\r\n", b"\n")


def render_doc(sha: Mapping[str, str], commit: str) -> str:
    """docs/figures.md (D51 (f), (11))."""
    lines = [
        "# NIFTYTAIL: tracked figures (D51, D52)",
        "",
        "**GENERATED by `python -m niftytail figures` from tracked JSON only;"
        " do not edit by hand.**",
        "",
        f"- Code commit: `{commit}`",
        f"- `{CONFIG.as_posix()}` sha256: `{sha[CONFIG.as_posix()]}`",
        f"- Passes: {', '.join(PASS_LABELS.values())}.",
        "- Display names (D52): "
        + ", ".join(f"`{k}` = {v}" for k, v in DISPLAY_NAMES.items())
        + '; a sensitivity key such as `pot_gpd@0.86` is shown as "POT-GPD q=0.86".',
        "",
    ]
    for s in FIGURES:
        lines += [
            f"## {s.fid}: `{(FIG_DIR / s.file).as_posix()}`",
            "",
            s.caption,
            "",
            f"Passes shown: {', '.join(s.passes)}.",
            "",
            "| input | sha256 |",
            "|---|---|",
            *[f"| `{p}` | `{sha[p]}` |" for p in s.inputs],
            "",
        ]
    return "\n".join(lines)


def render_all(inputs: Inputs, commit: str) -> dict[str, bytes]:
    """Every output (repository-relative POSIX path -> bytes), deterministically (D51 (e))."""
    out: dict[str, bytes] = {}
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT, "svg.fonttype": "path"}):
        for s in FIGURES:
            out[(FIG_DIR / s.file).as_posix()] = _svg(s.build(inputs), s, inputs.sha256, commit)
    out[DOC.as_posix()] = render_doc(inputs.sha256, commit).encode("utf-8")
    return out


def run(verify: bool, root: str | Path, log: logging.Logger, replace: bool = False) -> int:
    """D51 (g). Exit 0 written/verified, 1 refused or mismatch.

    ``replace`` (D54): overwrite the existing outputs, and only those paths, for a deliberate
    re-render; like every write it needs a clean tree, so the files replaced are exactly the
    committed, recoverable versions. Without it an existing output is refused.

    Raises:
        OosError: dirty tree or no commit; FiguresInputError on a failed input check.
    """
    base = Path(root)
    names = [(FIG_DIR / s.file).as_posix() for s in FIGURES] + [DOC.as_posix()]
    if verify and replace:
        log.error("--replace and --verify are exclusive; refusing")
        return 1
    commit = tl_blocks.head_commit(base, set(names), verify)
    if not verify and not replace and any((base / n).exists() for n in names):
        log.error(
            "an output exists; refusing (use --replace to re-render)", extra={"outputs": names}
        )
        return 1
    files = render_all(read_inputs(base), commit)
    hashes = {n: _sha(b) for n, b in files.items()}
    if verify:
        on_disk = {
            n: _sha((base / n).read_bytes()) if (base / n).is_file() else None for n in names
        }
        ok = on_disk == hashes
        log.log(
            logging.INFO if ok else logging.ERROR,
            "verify: all hashes match" if ok else "verify: MISMATCH",
            extra={"computed": hashes, "on_disk": on_disk},
        )
        return 0 if ok else 1
    for n, b in files.items():
        (base / n).parent.mkdir(parents=True, exist_ok=True)
        (base / n).write_bytes(b)
    log.info("OK", extra={"sha256": hashes, "commit": commit})
    return 0
