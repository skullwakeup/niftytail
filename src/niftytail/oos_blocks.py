"""Post-hoc OOS-block threshold check (D45; plan §5a (ii)).

The unchanged D34 procedure (``select_threshold``) is rerun on blocks of ``train_size`` OOS
losses, and on the F4 standardized residuals of the same blocks (descriptive only), to report
the q each block would have chosen. Nothing here changes q_selected or q_c. Two checks fixed
before any block is run (D45 (c)) are attached to every entry.

Outputs hold aggregates only (D23): the run-dir ``oos_blocks.json`` and, via publish, the tracked
``reports/threshold_selection/oos_blocks/selection.json`` and the generated ``docs/oos_blocks.md``.
No date, loss or residual series is written or logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from niftytail import data as dl
from niftytail import experiment as ex
from niftytail import losses as ls
from niftytail import oos
from niftytail import threshold_real as tr
from niftytail import threshold_residuals as trr
from niftytail.config import Config, ThresholdConfig, config_sha256
from niftytail.garch import GarchError, fit_garch, garch_filter
from niftytail.threshold import select_threshold

__all__ = [
    "DOC",
    "OUT_NAME",
    "TRACKED_JSON",
    "block_slice",
    "build_document",
    "checks",
    "layout",
    "publish",
    "render_doc",
    "run",
    "run_dir",
    "spawn_key",
]

OUT_NAME = "oos_blocks.json"
TRACKED_JSON = Path("reports/threshold_selection/oos_blocks/selection.json")
DOC = Path("docs/oos_blocks.md")
W1000_SUMMARY = oos.OOS_DIR / "w1000" / "summary.json"
PLAN = Path("docs/plan_v2.2.md")
SERIES = ("raw", "residuals")
SERIES_LABEL = {
    "raw": "POST-HOC (plan §5a (ii)); does not change q_selected",
    "residuals": "DESCRIPTIVE ONLY; no decision may cite it",
}
SERIES_NAME = {"raw": "raw OOS losses", "residuals": "F4 standardized residuals of the OOS losses"}
CHECK_KEYS = ("q_drift", "q_drift_reason", "var99_rel", "es99_rel", "risk_material", "risk_reason")

Task = tuple[str, int, dict[str, Any], ls.FloatArray, float, Config]


def layout(n_oos: int, length: int) -> dict[str, Any]:
    """D45 (b): full blocks from offset 0, plus one end-aligned overlapping block if r > 0."""
    if n_oos < length:
        raise oos.OosError(f"n_oos = {n_oos} < block length {length}: no OOS block")
    k, r = divmod(n_oos, length)
    blocks = [
        {"block": f"B{i + 1}", "offset": i * length, "overlapping": False, "overlap": 0}
        | {"overlap_with": None}
        for i in range(k)
    ]
    if r:
        blocks.append(
            {
                "block": f"B{k + 1}",
                "offset": n_oos - length,
                "overlapping": True,
                "overlap": length - r,
                "overlap_with": f"B{k}",
            }
        )
    return {"n_oos": n_oos, "block_length": length, "remainder": r, "blocks": blocks}


def block_slice(
    losses: ls.FloatArray, t0: int, offset: int, length: int
) -> tuple[ls.FloatArray, float]:
    """``(losses[t0+off : t0+off+length], losses[t0+off-1])``: the block and its AR lag."""
    start = t0 + offset
    if start < 1 or start + length > losses.size:
        raise ValueError("block outside the loss series")
    return losses[start : start + length].copy(), float(losses[start - 1])


def spawn_key(series: str, block: int) -> tuple[int, int, int]:
    """D45 (d), (e): ``(45, 0, b)`` raw, ``(45, 1, b)`` residuals; b counts from 1."""
    return (45, SERIES.index(series), block)


def _usable(x: object) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def checks(
    rows: list[dict[str, Any]],
    q_block: Fraction | None,
    rule_failure: str | None,
    q_ref: Fraction,
    th: ThresholdConfig,
) -> dict[str, Any]:
    """D45 (c): q_drift and risk_material from the block's own candidate rows (no refit)."""
    out: dict[str, Any] = {"q_ref": str(q_ref), **dict.fromkeys(CHECK_KEYS)}
    if q_block is None:
        out.update(q_drift=True, q_drift_reason=rule_failure)
        return out
    gap = abs(q_block - q_ref) > th.split_half_flag
    out.update(q_drift=gap, q_drift_reason="gap" if gap else None)
    if q_block == q_ref:
        out.update(var99_rel=0.0, es99_rel=0.0, risk_material=False)
        return out
    by_q = {r["q"]: r for r in rows}
    a, b = by_q[str(q_block)], by_q[str(q_ref)]

    def ok(r: dict[str, Any]) -> bool:
        bad = {"fit_failed", "es_infinite"} & set(r["flags"])
        return not bad and _usable(r["var99"]) and _usable(r["es99"])

    if not (ok(a) and ok(b)) or b["var99"] == 0 or b["es99"] == 0:
        out.update(risk_material=True, risk_reason="not_finite")
        return out
    # exact rationals of the floats (D45 (6)): a ratio of exactly 1 + tol is not flagged
    v = Fraction(a["var99"]) / Fraction(b["var99"]) - 1
    e = Fraction(a["es99"]) / Fraction(b["es99"]) - 1
    reason = None
    if abs(v) > th.tol_var99_rel:
        reason = "tol_var99"
    elif abs(e) > th.tol_es99_rel:
        reason = "tol_es99"
    out.update(
        var99_rel=float(v), es99_rel=float(e), risk_material=reason is not None, risk_reason=reason
    )
    return out


def _label(series: str, blk: dict[str, Any]) -> str:
    where = (
        f"overlapping ({blk['overlap']} returns with {blk['overlap_with']}), descriptive"
        if blk["overlapping"]
        else "non-overlapping"
    )
    return f"{SERIES_LABEL[series]}: {SERIES_NAME[series]}, block {blk['block']} ({where})"


def _entry(task: Task) -> tuple[dict[str, Any], float]:
    """One (series, block) entry and its wall seconds (D45 (d)-(f))."""
    series, b, blk, x, pre, cfg = task
    start = time.perf_counter()
    key = spawn_key(series, b)
    q_ref = cfg.threshold.q_selected if series == "raw" else cfg.conditional.q_c
    assert q_ref is not None
    head = {
        "decision": "D45",
        "label": _label(series, blk),
        "series": series,
        "block": blk["block"],
        "descriptive_only": series == "residuals",
    }
    prov = {
        "series": series,
        "block": blk["block"],
        "offset": blk["offset"],
        "n_obs": int(x.size),
        "seed_spawn_key": list(key),
    }
    y, fit = x, None
    if series == "residuals":
        try:
            fit = fit_garch(x, cfg.garch, cfg.conditional.filter, lag=pre)
            y = garch_filter(x, fit, pre)[2]
        except GarchError:
            out = {**head, "provenance": prov, "garch_error": True}
            out["check"] = {"q_ref": str(q_ref), **dict.fromkeys(CHECK_KEYS)}
            out["body_sha256"] = tr.body_sha256(out)
            return out, time.perf_counter() - start
    decision = select_threshold(
        y,
        cfg,
        ss=np.random.SeedSequence(cfg.runtime.seed, spawn_key=key),
        gof_reps=cfg.threshold.gof_bootstrap_reps,
        convergence_share=False,
        halves=True,
        intervals=True,
    )
    s = tr.build_summary(y, decision, cfg, prov)
    s.update(head)
    s["check"] = checks(
        s["candidates"], decision.q_star, decision.rule_failure, q_ref, cfg.threshold
    )
    if fit is not None:
        s["garch_error"] = False
        s["garch"] = {
            "spec": fit.spec,
            "phi1": fit.phi1,
            "gamma": fit.gamma,
            **trr.garch_block(fit, int(y.size)),
        }
    s["body_sha256"] = tr.body_sha256(s)
    return s, time.perf_counter() - start


def build_document(
    losses: ls.FloatArray, cfg: Config, workers: int, provenance: dict[str, Any]
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], float]]]:
    """Every entry (raw blocks, then residual blocks) and the document; output order fixed."""
    t0 = cfg.design.train_size
    lay = layout(losses.size - t0, t0)
    tasks: list[Task] = []
    for series in SERIES:
        for i, blk in enumerate(lay["blocks"]):
            x, pre = block_slice(losses, t0, blk["offset"], t0)
            tasks.append((series, i + 1, blk, x, pre, cfg))
    if workers == 1:
        results = [_entry(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            results = list(pool.map(_entry, tasks))
    doc = {
        "decision": "D45",
        "label": "post-hoc OOS-block threshold check (plan §5a (ii)); aggregates only",
        "provenance": provenance,
        "design": {
            "gof_bootstrap_reps": cfg.threshold.gof_bootstrap_reps,
            "convergence_share": False,
            "halves": True,
            "intervals": True,
            "block_length": t0,
        },
        "layout": lay,
        "raw": [e for e, _ in results if e["series"] == "raw"],
        "residuals": [e for e, _ in results if e["series"] == "residuals"],
    }
    return doc, results


def run_dir(root: Path, config_sha: str, manifest_sha: str, commit: str) -> Path:
    """D45 (h): ``reports/runs/<sha256("oos-blocks", config, data manifest, commit)>``."""
    key = f"oos-blocks\n{config_sha}\n{manifest_sha}\n{commit}\n".encode()
    return root / oos.RUNS_DIR / hashlib.sha256(key).hexdigest()


def _where(cfg: Config, root: Path, config_path: Path) -> tuple[str, str, str, Path]:
    commit = oos.clean_commit(root)
    cfg_sha = config_sha256(config_path)
    man_sha = ex.file_sha256(root / cfg.data.manifest)
    return commit, cfg_sha, man_sha, run_dir(root, cfg_sha, man_sha, commit)


def run(
    cfg: Config,
    config_path: str | Path,
    workers: int,
    verify: bool,
    root: str | Path,
    log: logging.Logger,
) -> int:
    """D45 (h): exit 0 done/verified, 1 refused or mismatch.

    Raises:
        OosError: dirty tree, no commit, an n_oos mismatch or n_oos < block length.
        DataValidationError: from the manifest-verified loader.
    """
    base = Path(root)
    commit, cfg_sha, man_sha, rd = _where(cfg, base, base / config_path)
    path = rd / OUT_NAME
    if not verify and path.exists():
        log.error("run dir already holds oos_blocks.json; refusing", extra={"run_dir": rd.name})
        return 1
    df = dl.load_raw(base / cfg.data.raw_dir, base / cfg.data.manifest, cfg.data.start_date)
    losses = ls.log_losses(df["close"].to_numpy())
    n_oos = losses.size - cfg.design.train_size
    n_targets = json.loads((base / W1000_SUMMARY).read_text(encoding="utf-8"))["n_targets"]
    if n_oos != n_targets:
        raise oos.OosError(f"n_oos = {n_oos} differs from w1000 n_targets = {n_targets}")
    provenance = {
        "commit": commit,
        "config_sha256": cfg_sha,
        "data_manifest_sha256": man_sha,
        "plan_sha256": ex.file_sha256(base / PLAN),
        "oos_w1000_summary_sha256": ex.file_sha256(base / W1000_SUMMARY),
    }
    doc, results = build_document(losses, cfg, workers, provenance)
    for e, secs in results:
        sel = e.get("selection", {})
        log.info(
            "entry",
            extra={
                "run_dir": rd.name,
                "series": e["series"],
                "block": e["block"],
                "offset": e["provenance"]["offset"],
                "q_star": sel.get("q_star"),
                "k_hat": sel.get("k_hat"),
                "rule_failure": sel.get("rule_failure"),
                "garch_error": e.get("garch_error"),
                "q_drift": e["check"]["q_drift"],
                "risk_material": e["check"]["risk_material"],
                "seconds": secs,
            },
        )
    text = tr.summary_json(doc).encode("utf-8")
    computed = hashlib.sha256(text).hexdigest()
    if verify:
        on_disk = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        ok = computed == on_disk
        log.log(
            logging.INFO if ok else logging.ERROR,
            "verify: hash matches" if ok else "verify: MISMATCH",
            extra={"run_dir": rd.name, "computed": computed, "on_disk": on_disk},
        )
        return 0 if ok else 1
    rd.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text)
    log.info("OK", extra={"run_dir": rd.name, "sha256": computed, "workers": workers})
    return 0


def publish(cfg: Config, config_path: str | Path, root: str | Path) -> list[Path]:
    """D45 (i): copy the current commit's oos_blocks.json to the tracked file; write the doc.

    Raises:
        OosError: dirty tree, no run for the current commit, or a tracked file from another
            commit.
    """
    base = Path(root)
    commit, _, _, rd = _where(cfg, base, base / config_path)
    src = rd / OUT_NAME
    if not src.is_file():
        raise oos.OosError("no oos-blocks run dir for the current commit")
    out = base / TRACKED_JSON
    if out.is_file():
        old = json.loads(out.read_text(encoding="utf-8"))["provenance"]["commit"]
        if old != commit:
            raise oos.OosError("selection.json already published from another commit")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(src.read_bytes())
    doc = base / DOC
    doc.write_text(render_doc(base), encoding="utf-8", newline="\n")
    return [out, doc]


# ---------------------------------------------------------------------------
# docs/oos_blocks.md (generated; D45 (j))
# ---------------------------------------------------------------------------
def _q(s: str | None) -> str:
    if s is None:
        return "—"
    f = Fraction(s)
    return str(Decimal(f.numerator) / Decimal(f.denominator))


def _half(h: dict[str, Any]) -> str:
    return _q(h["q_star"]) if h["q_star"] is not None else str(h["rule_failure"])


def _where_note(blk: dict[str, Any]) -> str:
    if not blk["overlapping"]:
        return "non-overlapping"
    return f"overlapping ({blk['overlap']} returns with {blk['overlap_with']}), descriptive"


def _series_section(series: str, entries: list[dict[str, Any]]) -> list[str]:
    v = oos._v
    out = [f"## {SERIES_NAME[series]}: {SERIES_LABEL[series]}", ""]
    header = [
        "block", "q*", "q_FS", "k_hat", "rule_failure", "at grid floor", "half 1", "half 2",
        "q_drift", "q_drift reason", "var99_rel", "es99_rel", "risk_material", "risk reason",
    ]  # fmt: skip
    if series == "residuals":
        header += ["near_boundary", "garch_error"]
    rows = []
    for e in entries:
        c = e["check"]
        checks_cells = [v(c[k]) for k in CHECK_KEYS]
        if e.get("garch_error"):
            rows.append([e["block"], *["—"] * 7, *checks_cells, "—", "yes"])
            continue
        sel, halves = e["selection"], e["split_half"]["halves"]
        row = [
            e["block"],
            _q(sel["q_star"]),
            _q(sel["q_fs"]),
            v(sel["k_hat"]),
            v(sel["rule_failure"]),
            v(sel["q_star_at_grid_floor"]),
            _half(halves[0]),
            _half(halves[1]),
            *checks_cells,
        ]
        if series == "residuals":
            row += [v("near_boundary" in e["garch"]["flags"]), v(e["garch_error"])]
        rows.append(row)
    out += oos._table(header, rows)
    out += [f"### xi_comparison ({series}): xi_hat and 95% profile CI", ""]
    xrows = []
    for e in entries:
        if e.get("garch_error"):
            xrows.append([e["block"], *["—"] * 6])
            continue
        cells = [e["block"]]
        for k in ("q_star", "q_0.90", "q_0.95"):
            x = e["xi_comparison"][k]
            cells += ["—", "—"] if x is None else [v(x["xi_hat"]), oos._ci(x["xi_ci"])]
        xrows.append(cells)
    out += oos._table(
        ["block", "xi at q*", "CI at q*", "xi at 0.90", "CI at 0.90", "xi at 0.95", "CI at 0.95"],
        xrows,
    )
    return out


def render_doc(root: str | Path) -> str:
    """docs/oos_blocks.md from the tracked JSON (deterministic; numbers only)."""
    base = Path(root)
    path = base / TRACKED_JSON
    d = json.loads(path.read_text(encoding="utf-8"))
    p = d["provenance"]
    out = [
        "# NIFTYTAIL — Post-hoc OOS-block threshold check (generated)",
        "",
        "Generated by `python -m niftytail oos-blocks-publish` from "
        f"`{TRACKED_JSON.as_posix()}` (D45 (j)). Do not edit by hand.",
        "",
        f"- Code commit: `{p['commit']}`",
        f"- Config sha256: `{p['config_sha256']}`",
        f"- Data manifest sha256: `{p['data_manifest_sha256']}`",
        f"- Plan sha256: `{p['plan_sha256']}`",
        f"- reports/oos/w1000/summary.json sha256: `{p['oos_w1000_summary_sha256']}`",
        f"- {TRACKED_JSON.name} sha256: `{ex.file_sha256(path)}`",
        "",
        "## Layout",
        "",
    ]
    lay = d["layout"]
    out += oos._table(
        ["n_oos", "block length", "remainder"],
        [[str(lay["n_oos"]), str(lay["block_length"]), str(lay["remainder"])]],
    )
    out += oos._table(
        ["block", "offset", "overlap", "note"],
        [[b["block"], str(b["offset"]), str(b["overlap"]), _where_note(b)] for b in lay["blocks"]],
    )
    for series in SERIES:
        out += _series_section(series, d[series])
    return "\n".join(out).rstrip("\n") + "\n"
