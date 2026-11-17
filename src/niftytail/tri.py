"""TRI sensitivity (D2; design D46 (e); implementation D48).

Data layer for the owner's Nifty 50 TRI downloads (data/raw_tri/, gitignored), mirroring the PR
layer (D25-D27) and reusing its parsers and checks; only the Gross "Total Returns Index" column
is used. The run aligns the TRI return dates with the PR OOS target dates (PR dates only, never
PR values; D2 "never mixed") and passes the TRI losses to the unchanged rolling engine and
backtests with an in-memory ``[design] train_size`` = t0_tri (the first-target index).

No message, flag or tracked output contains an index level, a return or a trading date (D23);
dates appear only under data/validation_tri/ and in the gitignored run dir.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import logging
import math
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from niftytail import backtest, oos, rolling
from niftytail import data as dl
from niftytail import experiment as ex
from niftytail import losses as ls
from niftytail.config import Config, config_sha256

__all__ = [
    "DOC",
    "HEADER",
    "OUT_DIR",
    "TRI_MANIFEST",
    "TRI_RAW",
    "TRI_START",
    "TriAlignmentError",
    "TriReport",
    "align",
    "inspect",
    "load_tri",
    "parse_window",
    "publish",
    "render_doc",
    "run",
    "run_dir",
    "validate",
    "validate_command",
    "write_outputs",
]

TRI_RAW = Path("data/raw_tri")
TRI_MANIFEST = TRI_RAW / "MANIFEST.toml"
TRI_VALIDATION = Path("data/validation_tri")
TRI_START = date(1999, 6, 30)  # NSE-documented TR availability start (D46 (e))
SOURCE_URL = "https://www.niftyindices.com/reports/historical-data"
RETRIEVED = date(2026, 11, 15)
SERIES, FIELD = "TRI", "Total Returns Index"
HEADER = ("IndexName", "Date", "Total Returns Index", "Net Total Return Index")
OUT_DIR = oos.OOS_DIR / "tri_w1000"
DOC = Path("docs/oos_tri.md")
PR_W1000 = oos.OOS_DIR / "w1000" / "backtest.json"
_NAME_RE = re.compile(r"^NIFTY 50_Historical_TR_(\d{2})(\d{2})(\d{4})to(\d{2})(\d{2})(\d{4})\.csv$")


class TriAlignmentError(oos.OosError):
    """A D48 (b) alignment condition failed. Messages hold counts and indices, never a date."""


# ---------------------------------------------------------------------------
# Parsing and validation (D48 (a), (2)-(8))
# ---------------------------------------------------------------------------
def parse_window(name: str) -> tuple[date, date]:
    """Window encoded in ``NIFTY 50_Historical_TR_DDMMYYYYtoDDMMYYYY.csv``.

    Raises:
        DataValidationError: if the name does not match or encodes an invalid window.
    """
    m = _NAME_RE.match(name)
    if m is None:
        raise dl.DataValidationError([f"{name}: file name does not match the expected TRI pattern"])
    d1, m1, y1, d2, m2, y2 = (int(g) for g in m.groups())
    try:
        start, end = date(y1, m1, d1), date(y2, m2, d2)
    except ValueError as exc:
        raise dl.DataValidationError([f"{name}: file name encodes an invalid date"]) from exc
    if end < start:
        raise dl.DataValidationError([f"{name}: window end precedes window start"])
    return start, end


def _parse_file(path: Path) -> tuple[Any, int]:
    """A data.py file record with Gross as ``close`` (no OHL), and the Net "-" count."""
    blob = path.read_bytes()
    out = dl._Parsed(name=path.name, data=blob, window=None)
    dash = 0
    try:
        out.window = parse_window(path.name)
    except dl.DataValidationError as exc:
        out.errors.extend(exc.errors)
        return out, dash
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        out.errors.append(f"{path.name}: not valid UTF-8")
        return out, dash
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or tuple(rows[0]) != HEADER:
        out.errors.append(f"{path.name}: header does not match {list(HEADER)}")
        return out, dash
    start, end = out.window
    for line_no, row in enumerate(rows[1:], start=2):
        where = f"{path.name}: line {line_no}"
        if len(row) != len(HEADER):
            out.errors.append(f"{where}: expected {len(HEADER)} fields, got {len(row)}")
            continue
        name, date_cell, gross_cell, net_cell = row
        if name != dl.EXPECTED_INDEX_NAME:
            out.errors.append(f"{where}: IndexName is not {dl.EXPECTED_INDEX_NAME!r}")
            continue
        d = dl._parse_date(date_cell)
        if d is None:
            out.errors.append(f"{where}: Date is not in 'dd Mon yyyy' format")
            continue
        g = dl._parse_number(gross_cell)
        if g is None or not math.isfinite(g) or g <= 0:
            out.errors.append(f"{where} ({d.isoformat()}): Total Returns Index is not valid")
            continue
        dash += net_cell == "-"  # Net is never parsed (D48 (a))
        if d < start or d > end:
            out.errors.append(
                f"{where}: date {d.isoformat()} outside the file window "
                f"{start.isoformat()}..{end.isoformat()}"
            )
        out.dates.append(d)
        out.lines.append(line_no)
        out.open.append(math.nan)
        out.high.append(math.nan)
        out.low.append(math.nan)
        out.close.append(g)
    return out, dash


def inspect(raw_dir: str | Path) -> tuple[list[Any], list[str], int]:
    """Parse every TRI file: records, parse/structure errors, Net "-" count."""
    paths = dl._csv_files(Path(raw_dir))
    results = [_parse_file(p) for p in paths]
    parsed = [r[0] for r in results]
    errors = [e for p in parsed for e in p.errors] + dl._structure_errors(parsed)
    if not paths:
        errors.append("no CSV files found in the raw directory")
    return parsed, errors, sum(r[1] for r in results)


def _manifest_errors(manifest_path: Path, raw_dir: Path) -> list[str]:
    try:
        man = dl.read_manifest(manifest_path)
    except dl.DataValidationError as exc:
        return list(exc.errors)
    errors: list[str] = []
    if (man.series, man.field) != (SERIES, FIELD):
        errors.append(f"manifest series/field is not {SERIES}/{FIELD}")
    on_disk = {p.name: p for p in dl._csv_files(raw_dir)}
    listed = {e.name: e for e in man.files}
    for name in sorted(set(listed) - set(on_disk)):
        errors.append(f"manifest lists a missing file: {name}")
    for name in sorted(set(on_disk) - set(listed)):
        errors.append(f"file not in manifest: {name}")
    for name in sorted(set(listed) & set(on_disk)):
        blob = on_disk[name].read_bytes()
        if len(blob) != listed[name].bytes or dl._sha256(blob) != listed[name].sha256:
            errors.append(f"sha256/size mismatch with manifest: {name}")
    return errors


def _rows(raw: Path, man: Path) -> tuple[Any, list[str], int]:
    errors = _manifest_errors(man, raw)
    parsed, perr, dash = inspect(raw)
    errors += perr
    infos = tuple(dl._info(p) for p in parsed if p.window is not None)
    errors += dl._manifest_content_errors(man, infos)
    rows = dl._collect(parsed)
    if rows.dates and rows.dates[0] != TRI_START:
        errors.append(f"first date {rows.dates[0].isoformat()} != TRI start {TRI_START}")
    return (rows, infos), errors, dash


def load_tri(raw_dir: str | Path, manifest_path: str | Path) -> pd.DataFrame:
    """Manifest-verified TRI Gross series: DatetimeIndex ``date``, column ``gross``.

    Raises:
        DataValidationError: on any hard error (manifest checked first).
    """
    raw, man = Path(raw_dir), Path(manifest_path)
    errors = _manifest_errors(man, raw)
    if errors:
        raise dl.DataValidationError(errors)
    (rows, _), errors, _ = _rows(raw, man)
    if errors:
        raise dl.DataValidationError(errors)
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in rows.dates], name="date")
    return pd.DataFrame({"gross": rows.close}, index=index)


@dataclass(frozen=True, slots=True)
class TriReport:
    """Validation result: dates, counts and hashes only."""

    report: dl.ValidationReport
    net_dash: int
    differences: dict[str, int]
    difference_dates: tuple[tuple[date, str], ...]


def _spot_check(rows: Any, n: int, seed: int) -> tuple[dl.SpotCheckItem, ...]:
    k = min(n, len(rows.dates))
    if k == 0:
        return ()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(48,))))
    idx = sorted(int(i) for i in rng.choice(len(rows.dates), size=k, replace=False))
    return tuple(dl.SpotCheckItem(rows.dates[i], rows.files[i], rows.lines[i]) for i in idx)


def _differences(
    pr_dates: list[date], tri_dates: list[date], oos_start: date | None
) -> tuple[dict[str, int], tuple[tuple[date, str], ...]]:
    """D48 (7): PR-only and TRI-only dates within the TRI span, and over the OOS span."""
    if not tri_dates:
        return {}, ()
    lo, hi = tri_dates[0], tri_dates[-1]
    pr = {d for d in pr_dates if lo <= d <= hi}
    tri = set(tri_dates)
    diff = sorted([(d, "pr_only") for d in pr - tri] + [(d, "tri_only") for d in tri - pr])
    counts = {
        "pr_only": sum(s == "pr_only" for _, s in diff),
        "tri_only": sum(s == "tri_only" for _, s in diff),
        "pr_only_oos": sum(
            s == "pr_only" and oos_start is not None and d >= oos_start for d, s in diff
        ),
        "tri_only_oos": sum(
            s == "tri_only" and oos_start is not None and d >= oos_start for d, s in diff
        ),
    }
    return counts, tuple(diff)


def validate(
    raw_dir: str | Path,
    manifest_path: str | Path,
    cfg: Config,
    pr_dates: list[date],
    oos_start: date | None,
) -> TriReport:
    """Every hard check (collected) and the report-only flags; the files are only read."""
    raw, man = Path(raw_dir), Path(manifest_path)
    try:
        dl._csv_files(raw)
    except dl.DataValidationError as exc:
        empty = dl.ValidationReport((), 0, None, None, exc.errors, (), {"errors": len(exc.errors)})
        return TriReport(empty, 0, {}, ())
    (rows, infos), errors, dash = _rows(raw, man)
    v = cfg.validation
    flags, counts = dl._flags(
        rows,
        large_return_flag=v.large_return_flag,
        gap_flag_days=v.gap_flag_days,
        stale_run_min=v.stale_run_min,
    )
    counts = {k: n for k, n in counts.items() if not k.startswith("missing_")}
    counts["errors"] = len(errors)
    counts["net_dash"] = dash
    yearly, absent = dl._calendar_audit(rows.dates)
    report = dl.ValidationReport(
        files=infos,
        n_rows=len(rows.dates),
        first_date=rows.dates[0] if rows.dates else None,
        last_date=rows.dates[-1] if rows.dates else None,
        errors=tuple(errors),
        flags=tuple(flags),
        counts=dict(sorted(counts.items())),
        spot_check=_spot_check(rows, v.spot_check_n, cfg.runtime.seed),
        yearly=yearly,
        absent_weekdays=absent,
    )
    diff, diff_dates = _differences(pr_dates, rows.dates, oos_start)
    return TriReport(report, dash, diff, diff_dates)


def write_outputs(tr: TriReport, out_dir: str | Path) -> list[Path]:
    """data/validation_tri/ (gitignored): the D25-style files plus date_differences.csv."""
    out = Path(out_dir)
    paths = list(dl.write_outputs(tr.report, out))
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    summary.pop("first_close_equals_base_value", None)
    summary.update(series=SERIES, field=FIELD, date_differences=tr.differences)
    paths[0].write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sheet = [
        "NIFTYTAIL TRI spot-check sheet (D48 (a); D25 format)",
        "For each date: open the named raw file at the given line, then compare its Total",
        "Returns Index with an independent public source. Record the result below.",
        "No values are written here.",
        "",
        "date        | file | line | matches? (yes/no) | source checked",
        *(f"{s.date.isoformat()} | {s.file} | {s.line} |  |" for s in tr.report.spot_check),
    ]
    paths[2].write_text("\n".join(sheet) + "\n", encoding="utf-8", newline="\n")
    diff_path = out / "date_differences.csv"
    with diff_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date", "side"])
        for d, side in tr.difference_dates:
            w.writerow([d.isoformat(), side])
    return [*paths, diff_path]


def _pr_dates(cfg: Config, root: Path) -> tuple[list[date], list[date]]:
    """PR price dates and PR return dates (manifest-verified loader; dates only)."""
    df = dl.load_raw(root / cfg.data.raw_dir, root / cfg.data.manifest, cfg.data.start_date)
    idx = pd.DatetimeIndex(df.index)
    return [d.date() for d in idx], [d.date() for d in ls.return_dates(idx)]


def validate_command(cfg: Config, root: str | Path, log: logging.Logger) -> int:
    """``validate-tri`` (D48 (a), (5)): manifest once, then the full validation. Exit 0 iff ok."""
    base = Path(root)
    raw, man = base / TRI_RAW, base / TRI_MANIFEST
    if not man.exists():
        parsed, errors, _ = inspect(raw)
        if errors:
            for e in errors:
                log.error("data error", extra={"error": e})
            log.error("TRI manifest not written: the raw files have errors")
            return 1
        infos = [dl._info(p) for p in parsed if p.window is not None]
        dl.write_manifest(
            man, infos, source_url=SOURCE_URL, retrieved=RETRIEVED, series=SERIES, field_name=FIELD
        )
        log.info(
            "TRI manifest written", extra={"manifest": TRI_MANIFEST.as_posix(), "files": len(infos)}
        )
    pr_prices, pr_returns = _pr_dates(cfg, base)
    t0 = cfg.design.train_size
    oos_start = pr_returns[t0] if len(pr_returns) > t0 else None
    tr = validate(raw, man, cfg, pr_prices, oos_start)
    outputs = write_outputs(tr, base / TRI_VALIDATION)
    for e in tr.report.errors:
        log.error("data error", extra={"error": e})
    log.log(
        logging.INFO if tr.report.ok else logging.ERROR,
        "TRI validation passed" if tr.report.ok else "TRI validation failed",
        extra={
            "n_files": len(tr.report.files),
            "n_rows": tr.report.n_rows,
            "n_errors": len(tr.report.errors),
            "counts": tr.report.counts,
            "date_differences": tr.differences,
            "outputs": [p.relative_to(base).as_posix() for p in outputs],
        },
    )
    return 0 if tr.report.ok else 1


# ---------------------------------------------------------------------------
# Alignment and run (D48 (b)-(d))
# ---------------------------------------------------------------------------
def align(pr_returns: list[date], tri_returns: list[date], t0: int, window: int) -> int:
    """t0_tri: the index of the first PR OOS date among the TRI return dates (D48 (b))."""
    pr_oos = pr_returns[t0:]
    if not pr_oos:
        raise TriAlignmentError("no PR OOS dates")
    try:
        t = tri_returns.index(pr_oos[0])
    except ValueError:
        raise TriAlignmentError("the first PR OOS date is missing from the TRI dates") from None
    if t < window + 1:
        raise TriAlignmentError(f"t0_tri = {t} < W + 1 = {window + 1}")
    tail = tri_returns[t:]
    if tail != pr_oos:
        k = next((i for i, (a, b) in enumerate(zip(tail, pr_oos, strict=False)) if a != b), None)
        raise TriAlignmentError(
            f"TRI return dates from t0_tri differ from the PR OOS dates: {len(tail)} vs "
            f"{len(pr_oos)} dates, first difference at OOS index {k}"
        )
    return t


def run_dir(
    root: Path, config_sha: str, pr_sha: str, tri_sha: str, window: int, commit: str
) -> Path:
    """D48 (d): ``reports/runs/<sha256("tri", config, PR manifest, TRI manifest, W, commit)>``."""
    key = f"tri\n{config_sha}\n{pr_sha}\n{tri_sha}\n{window}\n{commit}\n".encode()
    return root / oos.RUNS_DIR / hashlib.sha256(key).hexdigest()


def _where(cfg: Config, root: Path, config_path: Path) -> tuple[str, str, str, str, Path]:
    commit = oos.clean_commit(root)
    cfg_sha = config_sha256(config_path)
    pr_sha = ex.file_sha256(root / cfg.data.manifest)
    tri_sha = ex.file_sha256(root / TRI_MANIFEST)
    rd = run_dir(root, cfg_sha, pr_sha, tri_sha, cfg.design.window, commit)
    return commit, cfg_sha, pr_sha, tri_sha, rd


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run(
    cfg: Config,
    config_path: str | Path,
    workers: int,
    verify: bool,
    root: str | Path,
    log: logging.Logger,
) -> int:
    """The TRI W = 1000 pass (D48 (c), (d)). Exit 0 done/verified, 1 refused/mismatch, 2 stop.

    Raises:
        OosError: dirty tree or no commit; TriAlignmentError for a D48 (b) condition.
        DataValidationError: from the manifest-verified loaders.
    """
    base = Path(root)
    cfg_path = base / config_path
    w = cfg.design.window
    _, _, pr_sha, tri_sha, rd = _where(cfg, base, cfg_path)
    if not verify and (rd / oos.FORECASTS).exists():
        log.error("run dir already holds forecasts.csv; refusing", extra={"run_dir": rd.name})
        return 1
    _, pr_returns = _pr_dates(cfg, base)
    df = load_tri(base / TRI_RAW, base / TRI_MANIFEST)
    idx = pd.DatetimeIndex(df.index)
    tri_returns = [d.date() for d in ls.return_dates(idx)]
    t0_tri = align(pr_returns, tri_returns, cfg.design.train_size, w)
    cfg_tri = dataclasses.replace(cfg, design=dataclasses.replace(cfg.design, train_size=t0_tri))
    losses = ls.log_losses(df["gross"].to_numpy())
    dates = [d.isoformat() for d in tri_returns]
    data_paths = [
        *dl._csv_files(base / cfg.data.raw_dir),
        base / cfg.data.manifest,
        *dl._csv_files(base / TRI_RAW),
        base / TRI_MANIFEST,
    ]
    manifest = ex.build_manifest(cfg_path, data_paths, base)
    manifest["tri"] = {
        "t0_tri": t0_tri,
        "train_size_override": t0_tri,
        "pr_manifest_sha256": pr_sha,
        "tri_manifest_sha256": tri_sha,
    }

    t0 = time.perf_counter()
    table, summary = rolling.run_rolling(losses, dates, cfg_tri, w, workers)
    rolling_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        files = {oos.FORECASTS: rolling.write_table(table, tmp).read_bytes()}
    files[oos.SUMMARY] = oos._summary_json(summary)
    stop = bool(summary["stop_for_owner"])
    agg: dict[str, Any] = {}
    per: dict[str, Any] = {}
    evaluate_s = None
    if not stop:
        t1 = time.perf_counter()
        agg, per = backtest.evaluate(table, cfg_tri, w)
        evaluate_s = time.perf_counter() - t1
        files[oos.BACKTEST] = backtest.to_json(agg).encode()
        files[oos.PER_DATE] = backtest.to_json(per).encode()
    timing = {"workers": workers, "rolling_seconds": rolling_s, "evaluate_seconds": evaluate_s}
    counts = {
        "run_dir": rd.name,
        "window": w,
        "t0_tri": t0_tri,
        "n_targets": summary["n_targets"],
        "garch_error": summary["garch_error"],
        "no_pre_window_lag": summary["no_pre_window_lag"],
        "stop_for_owner": stop,
        **timing,
    }
    if verify:
        names = oos.VERIFIED if not stop else (oos.FORECASTS, oos.SUMMARY)
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
    ex.write_manifest(manifest, rd / oos.MANIFEST)
    (rd / oos.FORECASTS).write_bytes(files[oos.FORECASTS])
    (rd / oos.SUMMARY).write_bytes(files[oos.SUMMARY])
    if stop:
        (rd / oos.TIMING).write_bytes(backtest.to_json(timing).encode())
        log.error(
            "stop_for_owner: garch_error share above 1%; not evaluated",
            extra={
                "garch_error": summary["garch_error"],
                "garch_error_share": summary["garch_error_share"],
            },
        )
        return 2
    (rd / oos.BACKTEST).write_bytes(files[oos.BACKTEST])
    backtest.write_per_date(per, rd)
    (rd / oos.TIMING).write_bytes(backtest.to_json(timing).encode())
    mask = {
        lvl: {k: m[k] for k in ("common", "share", "below_min_share")}
        for lvl, m in agg["common_mask"].items()
    }
    log.info("OK", extra={**counts, "common_mask": mask})
    return 0


def publish(cfg: Config, config_path: str | Path, root: str | Path) -> list[Path]:
    """D48 (d), (e): copy the TRI aggregates to reports/oos/tri_w1000/ and write the doc.

    Raises:
        OosError: dirty tree, an incomplete run dir, or a tracked manifest from another commit.
    """
    base = Path(root)
    commit, _, pr_sha, tri_sha, rd = _where(cfg, base, base / config_path)
    missing = [f for f in oos.RUN_FILES if not (rd / f).is_file()]
    if missing:
        raise oos.OosError(f"no complete TRI run dir for the current commit ({missing})")
    out = base / OUT_DIR
    if (out / oos.MANIFEST).is_file():
        old = json.loads((out / oos.MANIFEST).read_text(encoding="utf-8"))["git"]["commit"]
        if old != commit:
            raise oos.OosError("TRI results already published from another commit")
    out.mkdir(parents=True, exist_ok=True)
    for name in (oos.BACKTEST, oos.SUMMARY):
        (out / name).write_bytes((rd / name).read_bytes())
    m = json.loads((rd / oos.MANIFEST).read_text(encoding="utf-8"))
    m.pop("data_files")
    m["data_manifest"] = {"file": Path(cfg.data.manifest).as_posix(), "sha256": pr_sha}
    m["tri_manifest"] = {"file": TRI_MANIFEST.as_posix(), "sha256": tri_sha}
    ex.write_manifest(m, out / oos.MANIFEST)
    doc = base / DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render_doc(base), encoding="utf-8", newline="\n")
    return [*(out / n for n in oos.TRACKED), doc]


# ---------------------------------------------------------------------------
# docs/oos_tri.md (generated; D48 (e), (12))
# ---------------------------------------------------------------------------
def _es(c: dict[str, Any]) -> str:
    es = c["es_residual"]
    return "descriptive" if es["descriptive"] else oos._v(es["p"])


def _cov_cells(c: dict[str, Any], fz: dict[str, Any]) -> list[str]:
    v = oos._v
    if c["n"] == 0:
        return ["—"] * 6 + [v(fz["mean"])]
    ch = c["christoffersen"]
    return [
        v(c["breaches"]), v(c["p_hat"]), v(c["kupiec"]["p"]), v(ch["p_ind"]), v(ch["p_cc"]),
        _es(c), v(fz["mean"]),
    ]  # fmt: skip


def render_doc(root: str | Path) -> str:
    """docs/oos_tri.md from reports/oos/tri_w1000/ and reports/oos/w1000/backtest.json."""
    base = Path(root)
    out_dir = base / OUT_DIR
    m = json.loads((out_dir / oos.MANIFEST).read_text(encoding="utf-8"))
    tri = json.loads((out_dir / oos.BACKTEST).read_text(encoding="utf-8"))
    pr = json.loads((base / PR_W1000).read_text(encoding="utf-8"))
    v = oos._v
    files = [*(OUT_DIR / n for n in oos.TRACKED), PR_W1000]
    out = [
        "# NIFTYTAIL — TRI sensitivity, W = 1000 (generated)",
        "",
        "Generated by `python -m niftytail tri-publish` from `reports/oos/tri_w1000/backtest.json` "
        "and `reports/oos/w1000/backtest.json` (D48 (e)). Do not edit by hand.",
        "",
        f"- Code commit: `{m['git']['commit']}`",
        f"- Config sha256: `{m['config']['sha256']}`",
        f"- PR data manifest sha256: `{m['data_manifest']['sha256']}`",
        f"- TRI data manifest sha256: `{m['tri_manifest']['sha256']}`",
        f"- t0_tri (in-memory `[design] train_size` override): {m['tri']['t0_tri']}",
        "",
    ]
    out += oos._table(
        ["file", "sha256"],
        [[f"`{f.as_posix()}`", f"`{ex.file_sha256(base / f)}`"] for f in files],
    )
    section = oos._pass_section(1000, tri)
    section[0] = "## TRI pass, W = 1000"
    out += section
    out += ["## PR vs TRI on the identical OOS days (W = 1000)", ""]
    keys = pr["primary_keys"] + pr["sensitivity_keys"]
    cols = ("breaches", "p_hat", "Kupiec p", "p_ind", "p_cc", "ES p", "mean FZ0")
    for lvl in pr["common_mask"]:
        out += [f"### Level {lvl}", ""]
        rows = []
        for k in keys:
            a = _cov_cells(pr["coverage"][lvl][k], pr["fz0"][lvl][k])
            b = _cov_cells(tri["coverage"][lvl][k], tri["fz0"][lvl][k])
            rows.append([k, *(x for pair in zip(a, b, strict=True) for x in pair)])
        out += oos._table(["key", *(f"{c} {s}" for c in cols for s in ("PR", "TRI"))], rows)
    out += ["### DM family (ii): Holm-adjusted p and verdict", ""]
    rows = []
    for e_pr, e_tri in zip(pr["dm"]["conditional"], tri["dm"]["conditional"], strict=True):
        assert (e_pr["level"], e_pr["a"], e_pr["b"]) == (e_tri["level"], e_tri["a"], e_tri["b"])
        rows.append(
            [
                e_pr["level"],
                e_pr["a"],
                e_pr["b"],
                v(e_pr["p_holm"]),
                oos._verdict(e_pr),
                v(e_tri["p_holm"]),
                oos._verdict(e_tri),
            ]
        )
    out += oos._table(
        ["level", "A", "B", "Holm p PR", "verdict PR", "Holm p TRI", "verdict TRI"], rows
    )
    out += ["### Traffic light (0.99): zone counts over full blocks", ""]
    zones = ("green", "yellow", "red")
    out += oos._table(
        ["key", *(f"{z} {s}" for s in ("PR", "TRI") for z in zones), "flagged PR", "flagged TRI"],
        [
            [
                k,
                *(v(pr["traffic_light"][k]["zones"][z]) for z in zones),
                *(v(tri["traffic_light"][k]["zones"][z]) for z in zones),
                v(pr["traffic_light"][k]["flagged_blocks"]),
                v(tri["traffic_light"][k]["flagged_blocks"]),
            ]
            for k in keys
        ],
    )
    return "\n".join(out).rstrip("\n") + "\n"
