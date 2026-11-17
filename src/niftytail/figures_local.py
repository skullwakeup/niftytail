"""Day-12C local daily figures (D53). GITIGNORED output only; never committed (D23).

These figures show the daily OOS series (losses, VaR / ES paths, the rolling xi path, cumulative
FZ0 differences and the traffic light with block dates), so they are written only under
``reports/runs/`` after ``git check-ignore`` confirms the paths are ignored. Inputs are the 11A
PR W1000 run dir, the 12A xi-path run dir and configs/base.toml, identified through tracked
files (D53 (1)). Dates appear in the SVGs only, never in logs or tracked files.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # select the non-interactive backend before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from niftytail import backtest, oos, rolling, xi_path
from niftytail import experiment as ex
from niftytail import figures as fg
from niftytail.config import Config

__all__ = ["FIGURES", "LocalInputs", "MeanMismatch", "read_inputs", "render_all", "run", "run_dir"]

SVG_HASHSALT = "niftytail-D53"
CONFIG = Path("configs/base.toml")
XI_SUMMARY = "reports/oos/xi_path/summary.json"
W1000_BACKTEST = "reports/oos/w1000/backtest.json"
TRAINING = "reports/threshold_selection/real/selection.json"
TRACKED = (XI_SUMMARY, W1000_BACKTEST, TRAINING)
XI_RUN_COMMIT = "c70202f3ac1a703b4f2a7831a247794f50ccb14e"  # status.md Day 12A
XI_CSV_SHA256 = "4c90b70fbe5269ff55fcc6a34a30571664a839cd07f5e3c8491339aab9377c0d"  # status.md 12A
G1_KEYS = ("pot_gpd", "fhs", "garch_evt")
G2_KEYS = ("pot_gpd", "garch_evt")
G3_OTHERS = ("pot_gpd", "fhs", "garch_n")
LEVEL = Fraction(99, 100)


class MeanMismatch(oos.OosError):
    """D53 (b) G3: a recomputed mean FZ0 differs from the tracked w1000 value."""


@dataclass(frozen=True)
class LocalInputs:
    table: pd.DataFrame  # the 11A forecasts.csv
    per_date: dict[str, Any]  # the 11A backtest_per_date.json
    xi: pd.DataFrame  # the 12A xi_path.csv
    training: dict[str, Any]
    fz0_means: dict[str, dict[str, float]]  # tracked w1000 fz0 means
    sha256: dict[str, str]  # input label -> sha256


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_dir(root: Path, config_sha: str, commit: str) -> Path:
    """D53 (c): ``reports/runs/<sha256("figures-local", config, commit)>``."""
    key = f"figures-local\n{config_sha}\n{commit}\n".encode()
    return root / oos.RUNS_DIR / hashlib.sha256(key).hexdigest()


def _dlvl(c: Fraction) -> str:
    return backtest._dec(c)


def read_inputs(root: Path, cfg_sha: str) -> LocalInputs:
    """Identify and read the D53 (a) inputs; every check is fatal (D53 (1), (2)).

    Raises:
        oos.OosError: a hash, commit, clean-flag or config check fails; MeanMismatch for G3.
    """
    raw = {rel: (root / rel).read_bytes() for rel in TRACKED}
    sha = {rel: _sha(b) for rel, b in raw.items()}
    for rel in TRACKED:
        if sha[rel] != fg.EXPECTED_SHA256[rel]:
            raise oos.OosError(f"{rel}: sha256 differs from docs/protected_hashes.md; STOP")
    prov = json.loads(raw[XI_SUMMARY])["provenance"]
    if prov["config_sha256"] != cfg_sha or prov["commit"] != XI_RUN_COMMIT:
        raise oos.OosError("xi_path summary provenance: config or commit differs; STOP")
    rd11 = root / oos.RUNS_DIR / prov["run_11a_dir"]
    man = json.loads((rd11 / oos.MANIFEST).read_bytes())
    want_git = {"commit": xi_path.RUN_11A_COMMIT, "dirty": False}
    if man["git"] != want_git or man["config"]["sha256"] != cfg_sha:
        raise oos.OosError("11A run-dir manifest: commit, clean flag or config differs; STOP")
    fc = (rd11 / oos.FORECASTS).read_bytes()
    sha["11A forecasts.csv"] = _sha(fc)
    if sha["11A forecasts.csv"] != prov["run_11a_forecasts_sha256"]:
        raise oos.OosError("11A forecasts.csv sha256 differs from the xi_path provenance; STOP")
    per_bytes = (rd11 / oos.PER_DATE).read_bytes()
    sha["11A backtest_per_date.json"] = _sha(per_bytes)
    rdx = xi_path.run_dir(root, cfg_sha, prov["data_manifest_sha256"], 1000, XI_RUN_COMMIT)
    xcsv = (rdx / xi_path.CSV_NAME).read_bytes()
    sha["12A xi_path.csv"] = _sha(xcsv)
    if sha["12A xi_path.csv"] != XI_CSV_SHA256:
        raise oos.OosError("12A xi_path.csv sha256 differs from status.md; STOP")
    bt = json.loads(raw[W1000_BACKTEST])
    per = json.loads(per_bytes)
    for k, counts in bt["traffic_light"].items():  # the per-date blocks match the tracked counts
        if backtest._zone_counts(per["traffic_light"][k]) != counts:
            raise oos.OosError(f"11A backtest_per_date.json: {k} zone counts differ; STOP")
    table = rolling.read_table(rd11 / oos.FORECASTS)
    xi = pd.read_csv(io.BytesIO(xcsv), keep_default_na=False, na_values=["nan"])
    need = {
        "loss",
        *(
            f"{k}:{m}:{_dlvl(c)}"
            for k in fg.MODELS
            for m in ("var", "es")
            for c in (Fraction(95, 100), LEVEL)
        ),
    }
    need_xi = {f"{k}:{f}" for k in G2_KEYS for f in ("xi", "ci_lo", "ci_hi")}
    if not need <= set(table.columns) or not need_xi <= set(xi.columns) or "date" not in xi:
        raise oos.OosError("an input lacks a column the figures need; STOP")
    return LocalInputs(
        table=table,
        per_date=per,
        xi=xi,
        training=json.loads(raw[TRAINING]),
        fz0_means={lvl: {k: v["mean"] for k, v in d.items()} for lvl, d in bt["fz0"].items()},
        sha256=sha,
    )


def fz0_by_day(table: pd.DataFrame, c: Fraction) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Daily FZ0 of the six primary keys on C, exactly as backtest.evaluate computes it."""
    adm = {
        k: np.array(
            [
                r == ""
                for r in backtest.admissibility(
                    backtest._col(table, k, "var", c),
                    backtest._col(table, k, "es", c),
                    [str(f) for f in table[f"{k}:flags"]],
                )
            ],
            dtype=np.bool_,
        )
        for k in fg.MODELS
    }
    cm = backtest.common_mask(adm)[fg.MODELS[0]]
    return cm, backtest.score_models(table, list(fg.MODELS), c, cm)


def check_means(inp: LocalInputs, levels: tuple[Fraction, ...]) -> dict[str, bool]:
    """D53 (b) G3: each key's mean over C repr-equals the tracked w1000 fz0 mean, else STOP."""
    out = {}
    for c in levels:
        _, scores = fz0_by_day(inp.table, c)
        for k in fg.MODELS:
            ok = repr(float(np.mean(scores[k]))) == repr(inp.fz0_means[_dlvl(c)][k])
            out[f"{_dlvl(c)} {k}"] = ok
            if not ok:
                raise MeanMismatch(f"G3: mean FZ0 of {k} at {_dlvl(c)} differs from w1000; STOP")
    return out


def _dates(values: Any) -> Any:
    return pd.to_datetime(pd.Series(list(values))).to_numpy()


# ---------------------------------------------------------------------------
# builders (parsed inputs in, Figure out)
# ---------------------------------------------------------------------------
def g1_var_es(table: pd.DataFrame, per_date: Mapping[str, Any]) -> Figure:
    """Daily loss with VaR99 and ES99 for POT-GPD, FHS and GARCH-EVT; breaches marked."""
    t = _dates(table.index.get_level_values("date"))
    loss = table["loss"].to_numpy(dtype=np.float64)
    lvl = _dlvl(LEVEL)
    fig, axes = plt.subplots(len(G1_KEYS), 1, figsize=(13, 9), sharex=True, squeeze=False)
    for ax, k in zip(axes[:, 0], G1_KEYS, strict=True):
        col = fg.MODEL_COLOURS[k]
        ax.plot(t, loss, color="#999999", linewidth=0.3, label="daily loss")
        ax.plot(t, table[f"{k}:var:{lvl}"], color=col, linewidth=0.7, label="VaR 0.99")
        ax.plot(
            t, table[f"{k}:es:{lvl}"], color=col, linewidth=0.7, linestyle="--", label="ES 0.99"
        )
        hit = np.array([ch == "1" for ch in per_date["hits"][lvl][k]])
        ax.plot(
            t[hit],
            loss[hit],
            "o",
            color="black",
            markersize=2.5,
            linestyle="none",
            label=f"breach ({int(hit.sum())})",
        )
        ax.set_ylabel("daily log loss (-log return)", fontsize=8)
        ax.set_title(fg.display(k), fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=4)
    axes[-1, 0].set_xlabel("OOS date")
    fig.suptitle(
        "PR W1000 (primary): daily loss with VaR and ES at 0.99, breaches marked", fontsize=10
    )
    fig.tight_layout()
    return fig


def g2_xi_path(xi: pd.DataFrame, training: Mapping[str, Any], ci_level: Fraction) -> Figure:
    """Rolling xi-hat with its profile-CI band for POT-GPD and GARCH-EVT; training band."""
    t = _dates(xi["date"])
    ref = training["xi_comparison"]["q_star"]
    fig, axes = plt.subplots(len(G2_KEYS), 1, figsize=(13, 7), sharex=True, squeeze=False)
    for ax, k in zip(axes[:, 0], G2_KEYS, strict=True):
        col = fg.MODEL_COLOURS[k]
        lo = xi[f"{k}:ci_lo"].to_numpy(dtype=np.float64)
        hi = xi[f"{k}:ci_hi"].to_numpy(dtype=np.float64)
        ax.fill_between(
            t, lo, hi, color=col, alpha=0.25, linewidth=0, label=f"{float(ci_level):.0%} profile CI"
        )
        ax.plot(
            t, xi[f"{k}:xi"].to_numpy(dtype=np.float64), color=col, linewidth=0.7, label="xi-hat"
        )
        ax.axhline(0.0, color="black", linewidth=0.6)
        if k == "pot_gpd":
            ax.axhspan(
                ref["xi_ci"][0],
                ref["xi_ci"][1],
                color="#BBBBBB",
                alpha=0.35,
                zorder=0,
                label=f"training xi {ref['xi_hat']:.3g}, CI [{ref['xi_ci'][0]:.3g}, "
                f"{ref['xi_ci'][1]:.3g}]",
            )
        ax.set_ylabel("xi (GPD shape, unitless)", fontsize=8)
        ax.set_title(fg.display(k), fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    axes[-1, 0].set_xlabel("OOS target date")
    fig.suptitle(
        "PR W1000 (primary): rolling xi-hat with profile CIs (D47; descriptive; "
        "overlapping windows)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def g3_cum_fz0(table: pd.DataFrame, levels: tuple[Fraction, ...]) -> Figure:
    """Cumulative FZ0 difference GARCH-EVT minus POT-GPD, FHS, GARCH-N over C."""
    dates = np.asarray(table.index.get_level_values("date"))
    fig, axes = plt.subplots(len(levels), 1, figsize=(13, 7), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], levels, strict=True):
        cm, scores = fz0_by_day(table, c)
        t = _dates(dates[cm])
        for other in G3_OTHERS:
            ax.plot(
                t,
                np.cumsum(scores["garch_evt"] - scores[other]),
                color=fg.MODEL_COLOURS[other],
                linewidth=0.9,
                label=f"GARCH-EVT - {fg.display(other)}",
            )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylabel(
            f"cumulative FZ0 difference, VaR {_dlvl(c)} (< 0: GARCH-EVT better)", fontsize=8
        )
        ax.legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("OOS date (days in the common mask C)")
    fig.suptitle("PR W1000 (primary): cumulative FZ0 difference over C", fontsize=10)
    fig.tight_layout()
    return fig


def g4_calendar(per_date: Mapping[str, Any]) -> Figure:
    """Traffic-light zone by block at 0.99 for the six primary models; block dates as ticks."""
    tl = per_date["traffic_light"]
    blocks = tl[fg.MODELS[0]]
    fig, ax = plt.subplots(figsize=(14, 5))
    for row, k in enumerate(fg.MODELS):
        for b in tl[k]:
            zone = b["zone"]
            ax.add_patch(
                Rectangle(
                    (b["block"], row),
                    1,
                    1,
                    edgecolor="white",
                    linewidth=1.5,
                    facecolor=fg.ZONE_COLOURS[zone] if zone else "white",
                    label=f"{k}|{b['block']}|{zone}",
                )
            )
            ax.text(
                b["block"] + 0.5,
                row + 0.5,
                str(b["breaches"]),
                ha="center",
                va="center",
                fontsize=6,
            )
    ax.set_xlim(0, len(blocks))
    ax.set_ylim(len(fg.MODELS), 0)
    ax.set_xticks([b["block"] + 0.5 for b in blocks])
    ax.set_xticklabels(
        [f"{b['start_date']} to {b['end_date']}" for b in blocks], rotation=90, fontsize=6
    )
    ax.set_yticks([i + 0.5 for i in range(len(fg.MODELS))])
    ax.set_yticklabels([fg.display(m) for m in fg.MODELS], fontsize=8)
    ax.set_xlabel("250-day block (first and last OOS date); last block partial, no zone")
    ax.legend(
        handles=[Patch(facecolor=c, label=z) for z, c in fg.ZONE_COLOURS.items()]
        + [Patch(facecolor="white", edgecolor="black", label="partial block (no zone)")],
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.suptitle(
        "PR W1000 (primary): traffic-light zone by block at VaR 0.99, breaches printed", fontsize=10
    )
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class LocalFigure:
    gid: str
    file: str
    caption: str
    inputs: tuple[str, ...]


FIGURES = (
    LocalFigure(
        "G1",
        "var_es_paths.svg",
        "Daily loss with VaR99 and ES99 for POT-GPD, FHS and GARCH-EVT; breaches marked.",
        ("11A forecasts.csv", "11A backtest_per_date.json"),
    ),
    LocalFigure(
        "G2",
        "xi_path.svg",
        "Rolling xi-hat with its profile-CI band for POT-GPD and GARCH-EVT; zero line;"
        " POT-GPD training band.",
        ("12A xi_path.csv", TRAINING),
    ),
    LocalFigure(
        "G3",
        "cum_fz0.svg",
        "Cumulative FZ0 difference GARCH-EVT minus POT-GPD, FHS, GARCH-N over C at 0.95 and 0.99.",
        ("11A forecasts.csv", W1000_BACKTEST),
    ),
    LocalFigure(
        "G4",
        "traffic_light_calendar.svg",
        "Traffic-light zone by 250-day block at 0.99 for the six primary models, block"
        " dates as tick labels.",
        ("11A backtest_per_date.json", W1000_BACKTEST),
    ),
)


def render_all(inp: LocalInputs, cfg: Config, commit: str) -> dict[str, bytes]:
    """Every output (run-dir-relative POSIX path -> bytes), deterministically."""
    levels = tuple(cfg.levels.primary)
    builders = {
        "G1": lambda: g1_var_es(inp.table, inp.per_date),
        "G2": lambda: g2_xi_path(inp.xi, inp.training, cfg.gpd.ci_level),
        "G3": lambda: g3_cum_fz0(inp.table, levels),
        "G4": lambda: g4_calendar(inp.per_date),
    }
    out: dict[str, bytes] = {}
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT, "svg.fonttype": "path"}):
        for s in FIGURES:
            fig = builders[s.gid]()
            desc = "; ".join(f"{p} sha256 {inp.sha256[p]}" for p in s.inputs)
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="svg",
                metadata={
                    "Date": None,
                    "Title": f"{s.gid} {s.file}",
                    "Description": f"inputs: {desc}; commit {commit}",
                },
            )
            plt.close(fig)
            out[f"figures/{s.file}"] = buf.getvalue().replace(b"\r\n", b"\n")
    lines = [
        "# NIFTYTAIL: local daily figures (D53)",
        "",
        "**LOCAL ONLY (gitignored, D23). Generated by `python -m niftytail figures-local`.**",
        "",
        f"- Code commit: `{commit}`",
        "",
    ]
    for s in FIGURES:
        lines += [
            f"## {s.gid}: `figures/{s.file}`",
            "",
            s.caption,
            "",
            "| input | sha256 |",
            "|---|---|",
            *[f"| `{p}` | `{inp.sha256[p]}` |" for p in s.inputs],
            "",
        ]
    out["index.md"] = "\n".join(lines).encode("utf-8")
    return out


def _ignored(root: Path, rel: str) -> bool:
    res = ex._git(root, "check-ignore", "-q", "--no-index", rel)
    return res is not None and res.returncode == 0


def run(
    cfg: Config, config_path: str | Path, verify: bool, root: str | Path, log: logging.Logger
) -> int:
    """D53 (d). Exit 0 written/verified, 1 refused or mismatch.

    Raises:
        oos.OosError: dirty tree, a failed identity check, an output that is not gitignored,
            or MeanMismatch.
    """
    base = Path(root)
    commit = oos.clean_commit(base)
    cfg_sha = _sha((base / config_path).read_bytes())
    rd = run_dir(base, cfg_sha, commit)
    names = [f"figures/{s.file}" for s in FIGURES] + ["index.md"]
    rels = [(rd / n).relative_to(base).as_posix() for n in names]
    if not all(_ignored(base, r) for r in rels):
        raise oos.OosError("an output path is not gitignored; STOP, nothing written")
    if not verify and any((rd / n).exists() for n in names):
        log.error("local figures exist; refusing", extra={"run_dir": rd.name})
        return 1
    inp = read_inputs(base, cfg_sha)
    means = check_means(inp, tuple(cfg.levels.primary))
    files = render_all(inp, cfg, commit)
    hashes = {n: _sha(b) for n, b in files.items()}
    info = {"run_dir": rd.name, "fz0_means_equal": all(means.values()), "n_checked": len(means)}
    if verify:
        on_disk = {n: _sha((rd / n).read_bytes()) if (rd / n).is_file() else None for n in names}
        ok = on_disk == hashes
        log.log(
            logging.INFO if ok else logging.ERROR,
            "verify: all hashes match" if ok else "verify: MISMATCH",
            extra={**info, "computed": hashes, "on_disk": on_disk},
        )
        return 0 if ok else 1
    for n, b in files.items():
        (rd / n).parent.mkdir(parents=True, exist_ok=True)
        (rd / n).write_bytes(b)
    log.info("OK", extra={**info, "sha256": hashes, "commit": commit})
    return 0
