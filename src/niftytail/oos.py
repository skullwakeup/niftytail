"""Out-of-sample runs, verification and publication of aggregates (D44; methodology §15).

``run``: one rolling pass (D41) on the real series plus its backtests (D42), written to a
run directory keyed by config, data manifest, window and git commit; only from a clean tree.
``--verify`` recomputes in memory and compares hashes. ``publish`` copies the aggregate files
of every window into the tracked ``reports/oos/`` and generates ``docs/oos.md``. Per-date data
(forecasts.csv, backtest_per_date.json) never leaves ``reports/runs/`` (D23).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from niftytail import backtest, rolling
from niftytail import data as dl
from niftytail import experiment as ex
from niftytail import losses as ls
from niftytail.config import Config, config_sha256

__all__ = [
    "OOS_DIR",
    "OOS_DOC",
    "RUNS_DIR",
    "RUN_FILES",
    "OosError",
    "clean_commit",
    "publish",
    "render_doc",
    "run",
    "run_dir",
    "windows",
]

RUNS_DIR = Path("reports/runs")
OOS_DIR = Path("reports/oos")
OOS_DOC = Path("docs/oos.md")
MANIFEST = "manifest.json"
FORECASTS = rolling.FORECASTS_CSV
SUMMARY = "summary.json"
BACKTEST = "backtest.json"
PER_DATE = backtest.PER_DATE_JSON
TIMING = "timing.json"
RUN_FILES = (MANIFEST, FORECASTS, SUMMARY, BACKTEST, PER_DATE, TIMING)
VERIFIED = (FORECASTS, SUMMARY, BACKTEST, PER_DATE)
TRACKED = (BACKTEST, SUMMARY, MANIFEST)


class OosError(Exception):
    """A refused run or publication. Messages contain no data values."""


def windows(cfg: Config) -> tuple[int, ...]:
    """``([design] window, *sensitivity_windows)``."""
    return (cfg.design.window, *cfg.design.sensitivity_windows)


def run_dir(root: Path, config_sha: str, manifest_sha: str, window: int, commit: str) -> Path:
    """D44 (b): ``reports/runs/<sha256(config, data manifest, W, commit)>``."""
    key = f"{config_sha}\n{manifest_sha}\n{window}\n{commit}\n".encode()
    return root / RUNS_DIR / hashlib.sha256(key).hexdigest()


def clean_commit(root: Path) -> str:
    """The HEAD commit, if there is one and the tree is clean (ignored files allowed)."""
    git = ex.git_state(root)
    if git.commit is None or git.dirty is not False:
        raise OosError("a real run needs a git commit and a clean tree")
    return git.commit


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _summary_json(summary: dict[str, Any]) -> bytes:
    keep = {k: v for k, v in summary.items() if k not in ("workers", "runtime_seconds")}
    return backtest.to_json(keep).encode()


def _keys(cfg: Config, root: Path, config_path: Path, window: int) -> tuple[str, str, Path]:
    commit = clean_commit(root)
    cfg_sha = config_sha256(config_path)
    man_sha = ex.file_sha256(root / cfg.data.manifest)
    return commit, man_sha, run_dir(root, cfg_sha, man_sha, window, commit)


def run(
    cfg: Config,
    config_path: str | Path,
    window: int,
    workers: int,
    verify: bool,
    root: str | Path,
    log: logging.Logger,
) -> int:
    """One OOS pass (D44 (c)-(f)). Exit 0 done/verified, 1 refused/mismatch, 2 stop rule.

    Raises:
        OosError: dirty tree or no commit.
        DataValidationError: from the manifest-verified loader.
    """
    base = Path(root)
    cfg_path = base / config_path
    if window not in windows(cfg):
        log.error("window not in [design] window or sensitivity_windows", extra={"window": window})
        return 1
    _, _, rd = _keys(cfg, base, cfg_path, window)
    if not verify and (rd / FORECASTS).exists():
        log.error("run dir already holds forecasts.csv; refusing", extra={"run_dir": rd.name})
        return 1
    raw, man = base / cfg.data.raw_dir, base / cfg.data.manifest
    df = dl.load_raw(raw, man, start_date=cfg.data.start_date)
    losses = ls.log_losses(df["close"].to_numpy())
    dates = [d.date().isoformat() for d in ls.return_dates(pd.DatetimeIndex(df.index))]
    manifest = ex.build_manifest(cfg_path, [*dl._csv_files(raw), man], base)

    t0 = time.perf_counter()
    table, summary = rolling.run_rolling(losses, dates, cfg, window, workers)
    rolling_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        files = {FORECASTS: rolling.write_table(table, tmp).read_bytes()}
    files[SUMMARY] = _summary_json(summary)
    stop = bool(summary["stop_for_owner"])
    agg: dict[str, Any] | None = None
    per: dict[str, Any] = {}
    evaluate_s = None
    if not stop:
        t1 = time.perf_counter()
        agg, per = backtest.evaluate(table, cfg, window)
        evaluate_s = time.perf_counter() - t1
        files[BACKTEST] = backtest.to_json(agg).encode()
        files[PER_DATE] = backtest.to_json(per).encode()
    timing = {"workers": workers, "rolling_seconds": rolling_s, "evaluate_seconds": evaluate_s}
    counts = {
        "run_dir": rd.name,
        "window": window,
        "n_targets": summary["n_targets"],
        "garch_error": summary["garch_error"],
        "no_pre_window_lag": summary["no_pre_window_lag"],
        "stop_for_owner": stop,
        **timing,
    }

    if verify:
        names = VERIFIED if not stop else (FORECASTS, SUMMARY)
        hashes = {
            n: {
                "computed": _sha(files[n]),
                "on_disk": _sha((rd / n).read_bytes()) if (rd / n).is_file() else None,
            }
            for n in names
        }
        ok = all(h["computed"] == h["on_disk"] for h in hashes.values())
        log.log(
            logging.INFO if ok else logging.ERROR,
            "verify: all hashes match" if ok else "verify: MISMATCH",
            extra={**counts, "hashes": hashes},
        )
        return 1 if not ok else (2 if stop else 0)

    rd.mkdir(parents=True, exist_ok=True)
    ex.write_manifest(manifest, rd / MANIFEST)
    (rd / FORECASTS).write_bytes(files[FORECASTS])
    (rd / SUMMARY).write_bytes(files[SUMMARY])
    if stop:
        (rd / TIMING).write_bytes(backtest.to_json(timing).encode())
        log.error(
            "stop_for_owner: garch_error share above 1%; not evaluated",
            extra={
                "garch_error": summary["garch_error"],
                "garch_error_share": summary["garch_error_share"],
            },
        )
        return 2
    assert agg is not None
    (rd / BACKTEST).write_bytes(files[BACKTEST])
    backtest.write_per_date(per, rd)
    (rd / TIMING).write_bytes(backtest.to_json(timing).encode())
    mask = {
        lvl: {k: m[k] for k in ("common", "share", "below_min_share")}
        for lvl, m in agg["common_mask"].items()
    }
    log.info("OK", extra={**counts, "common_mask": mask})
    return 0


def publish(cfg: Config, config_path: str | Path, root: str | Path) -> list[Path]:
    """D44 (g): copy the aggregates of every window's run dir (current commit) and write the doc.

    Raises:
        OosError: dirty tree, an incomplete run dir, or a published window from another commit.
    """
    base = Path(root)
    cfg_path = base / config_path
    plan = []
    for w in windows(cfg):
        commit, man_sha, rd = _keys(cfg, base, cfg_path, w)
        missing = [f for f in RUN_FILES if not (rd / f).is_file()]
        if missing:
            raise OosError(f"window {w}: no complete run dir for the current commit ({missing})")
        out = base / OOS_DIR / f"w{w}"
        if (out / MANIFEST).is_file():
            old = json.loads((out / MANIFEST).read_text(encoding="utf-8"))["git"]["commit"]
            if old != commit:
                raise OosError(f"window {w}: already published from another commit (D44 (h))")
        plan.append((rd, out, man_sha))
    written = []
    for rd, out, man_sha in plan:
        out.mkdir(parents=True, exist_ok=True)
        for name in (BACKTEST, SUMMARY):
            (out / name).write_bytes((rd / name).read_bytes())
        m = json.loads((rd / MANIFEST).read_text(encoding="utf-8"))
        m.pop("data_files")
        m["data_manifest"] = {"file": Path(cfg.data.manifest).as_posix(), "sha256": man_sha}
        ex.write_manifest(m, out / MANIFEST)
        written += [out / n for n in TRACKED]
    doc = base / OOS_DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render_doc(base), encoding="utf-8", newline="\n")
    return [*written, doc]


# ---------------------------------------------------------------------------
# docs/oos.md (generated; D44 (i))
# ---------------------------------------------------------------------------
def _v(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return format(x, ".6g")
    return str(x)


def _ci(pair: Any) -> str:
    return "—" if pair is None else f"[{_v(pair[0])}, {_v(pair[1])}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def _verdict(e: dict[str, Any]) -> str:
    return "reject" if e["reject"] else "not rejected"


def _coverage_rows(cov: dict[str, Any], keys: list[str]) -> list[list[str]]:
    rows = []
    for k in keys:
        c = cov[k]
        if c["n"] == 0:
            rows.append([k, "0", *["—"] * 15])
            continue
        es = c["es_residual"]
        rows.append(
            [
                k,
                _v(c["n"]),
                _v(c["breaches"]),
                _v(c["p_hat"]),
                _v(c["kupiec"]["lr"]),
                _v(c["kupiec"]["p"]),
                _v(c["binomial"]["p"]),
                _v(c["christoffersen"]["p_ind"]),
                _v(c["christoffersen"]["p_cc"]),
                _v(c["markov"]["pi11_hat"]),
                _v(c["markov"]["size"]),
                _ci(c["ci"]["clopper_pearson"]),
                _ci([c["ci"]["block"]["lower"], c["ci"]["block"]["upper"]]),
                _v(c["ci"]["block"]["b"]),
                _v(es["n_breach"]),
                _v(es["z_bar"]),
                "descriptive" if es["descriptive"] else _v(es["p"]),
            ]
        )
    return rows


COVERAGE_HEADER = [
    "key", "n", "breaches", "p_hat", "Kupiec LR", "Kupiec p", "binomial p", "p_ind", "p_cc",
    "pi11_hat", "Markov size", "Clopper-Pearson CI", "block CI", "b", "ES n_breach", "z_bar",
    "ES p",
]  # fmt: skip
DM_HEADER = ["level", "A", "B", "n", "lag", "d_bar", "DM", "p", "flag", "DM (2 lag)", "p (2 lag)"]


def _dm_cells(e: dict[str, Any]) -> list[str]:
    s = e["lag_sensitivity"]
    return [
        e["level"], e["a"], e["b"], _v(e["n"]), _v(e["lag"]), _v(e["d_bar"]), _v(e["dm"]),
        _v(e["p"]), _v(e["flag"]), _v(s["dm"]), _v(s["p"]),
    ]  # fmt: skip


def _pass_section(w: int, agg: dict[str, Any]) -> list[str]:
    keys = agg["primary_keys"] + agg["sensitivity_keys"]
    out = [f"## W = {w}", "", f"Rows (OOS targets): {agg['n_rows']}.", ""]
    out += ["### Common mask (primary levels)", ""]
    out += _table(
        ["level", "total", r"\|C\|", "share", "below min share"],
        [
            [lvl, _v(m["total"]), _v(m["common"]), _v(m["share"]), _v(m["below_min_share"])]
            for lvl, m in agg["common_mask"].items()
        ],
    )
    out += ["### Inadmissible forecasts by reason", ""]
    rows = []
    for lvl, by_key in agg["admissibility"].items():
        for k in keys:
            a = by_key[k]
            r, mf = a["reasons"], a["missing_by_flag"]
            rows.append(
                [lvl, k, _v(a["admissible"]), _v(a["inadmissible"])]
                + [_v(r[x]) for x in backtest.REASONS]
                + [_v(mf[x]) for x in (*backtest.MISSING_FLAGS, "other")]
            )
    out += _table(
        ["level", "key", "admissible", "inadmissible", *backtest.REASONS]
        + [f"missing:{x}" for x in (*backtest.MISSING_FLAGS, "other")],
        rows,
    )
    for lvl in agg["common_mask"]:
        out += [f"### Coverage, level {lvl} (each key on its own admissible dates)", ""]
        out += _table(COVERAGE_HEADER, _coverage_rows(agg["coverage"][lvl], keys))
    out += [
        "### Mean FZ0 (primary keys on C; a sensitivity key on C and its own admissibility)",
        "",
    ]
    levels = list(agg["fz0"])
    out += _table(
        ["key"] + [f"{lvl} {c}" for lvl in levels for c in ("n", "mean")],
        [[k] + [_v(agg["fz0"][lvl][k][c]) for lvl in levels for c in ("n", "mean")] for k in keys],
    )
    dm = agg["dm"]
    out += ["### DM family (i): unconditional pairs, unadjusted", ""]
    out += _table(
        [*DM_HEADER, "verdict"], [[*_dm_cells(e), _verdict(e)] for e in dm["unconditional"]]
    )
    out += ["### DM family (ii): conditional pairs, Holm step-down", ""]
    out += _table(
        [*DM_HEADER, "Holm p", "verdict"],
        [[*_dm_cells(e), _v(e["p_holm"]), _verdict(e)] for e in dm["conditional"]],
    )
    out += ["### DM family (iii): descriptive, no verdict", ""]
    out += _table(DM_HEADER, [_dm_cells(e) for e in dm["descriptive"]])
    out += ["### Traffic light (0.99): zone counts over full blocks", ""]
    tl = agg["traffic_light"]
    out += _table(
        [
            "key",
            "full blocks",
            "green",
            "yellow",
            "red",
            "flagged blocks",
            "partial n_obs",
            "partial breaches",
        ],
        [
            [
                k,
                _v(tl[k]["full_blocks"]),
                *(_v(tl[k]["zones"][z]) for z in ("green", "yellow", "red")),
                _v(tl[k]["flagged_blocks"]),
                _v(tl[k]["partial_block"]["n_obs"] if tl[k]["partial_block"] else None),
                _v(tl[k]["partial_block"]["breaches"] if tl[k]["partial_block"] else None),
            ]
            for k in keys
        ],
    )
    out += ["### Descriptive levels", ""]
    desc = [lvl for lvl in agg["coverage"] if lvl not in agg["common_mask"]]
    out += _table(
        ["level", "key", "n", "breaches", "p_hat", "binomial p"],
        [
            [lvl, k]
            + [_v(agg["coverage"][lvl][k][c]) for c in ("n", "breaches", "p_hat", "p_binomial")]
            for lvl in desc
            for k in keys
        ],
    )
    return out


def render_doc(root: str | Path) -> str:
    """docs/oos.md from reports/oos/w*/ (deterministic; numbers only, no interpretation)."""
    base = Path(root)
    dirs = sorted((base / OOS_DIR).glob("w*"), key=lambda p: int(p.name[1:]))
    manifests = [json.loads((d / MANIFEST).read_text(encoding="utf-8")) for d in dirs]
    m0 = manifests[0]
    out = [
        "# NIFTYTAIL — Out-of-sample results (generated)",
        "",
        "Generated by `python -m niftytail oos-publish` from `reports/oos/w*/backtest.json` "
        "(D44 (i)). "
        "Do not edit by hand.",
        "",
        f"- Code commit: `{m0['git']['commit']}`",
        f"- Config sha256: `{m0['config']['sha256']}`",
        f"- Data manifest sha256: `{m0['data_manifest']['sha256']}`",
        "",
    ]
    out += _table(
        ["tracked file", "sha256"],
        [
            [f"`{(d / n).relative_to(base).as_posix()}`", f"`{ex.file_sha256(d / n)}`"]
            for d in dirs
            for n in TRACKED
        ],
    )
    for d in dirs:
        agg = json.loads((d / BACKTEST).read_text(encoding="utf-8"))
        out += _pass_section(int(d.name[1:]), agg)
    return "\n".join(out).rstrip("\n") + "\n"
