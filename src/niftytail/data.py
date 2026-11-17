"""Raw-data layer: manifest, strict loader and validator (plan §6 Day 3; methodology §2; D23, D25).

Input format (found by the Day-3A structure inspection of the owner's NSE downloads):

* one CSV per one-year window, named ``NIFTY 50_Historical_PR_DDMMYYYYtoDDMMYYYY.csv``;
* UTF-8, CRLF line endings, every field double-quoted, comma delimiter, no trailing newline;
* header exactly ``"Index Name","Date","Open","High","Low","Close"``;
* ``Index Name`` is always ``NIFTY 50``; ``Date`` is ``dd Mon yyyy`` (English month abbreviation);
* numbers have no thousands separators; ``Open``/``High``/``Low`` may be ``-`` (missing);
* rows are sorted by date, most recent first.

Rules (D25):

* HARD ERRORS stop the pipeline; they are all collected before reporting.
* FLAGS are reported only. The data are never altered, filled or removed (methodology §2).
* No message, flag or output written by this module contains an index level or a per-day
  return (D23): only file names, line numbers, dates, counts, categories and hashes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BASE_VALUE",
    "EXPECTED_HEADER",
    "MANIFEST_SCHEMA_VERSION",
    "AbsentWeekday",
    "DataValidationError",
    "Flag",
    "Manifest",
    "ManifestEntry",
    "RawFileInfo",
    "SpotCheckItem",
    "ValidationReport",
    "YearAudit",
    "inspect_files",
    "load_raw",
    "parse_window_from_name",
    "read_manifest",
    "validate",
    "write_manifest",
    "write_outputs",
]

MANIFEST_SCHEMA_VERSION = 1
EXPECTED_HEADER = ("Index Name", "Date", "Open", "High", "Low", "Close")
EXPECTED_INDEX_NAME = "NIFTY 50"
BASE_VALUE = 1000.0  # NSE documented base value on 1995-11-03 (informational only)
MISSING_TOKENS = frozenset({"-", ""})

_NAME_RE = re.compile(r"^NIFTY 50_Historical_PR_(\d{2})(\d{2})(\d{4})to(\d{2})(\d{2})(\d{4})\.csv$")
_DATE_RE = re.compile(r"^(\d{2}) ([A-Z][a-z]{2}) (\d{4})$")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_MONTHS = {
    m: i
    for i, m in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
        start=1,
    )
}
_OHL = ("open", "high", "low")

FLAG_CATEGORIES = (
    "calendar_gap",
    "identical_close_pair",
    "large_return",
    "ohlc_inconsistent",
    "stale_close_run",
    "weekend_date",
)


class DataValidationError(Exception):
    """Hard data error(s). ``errors`` holds every message; none contains a data value."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        head = "; ".join(self.errors[:3]) + (" ..." if len(self.errors) > 3 else "")
        super().__init__(f"{len(self.errors)} data error(s): {head}")


@dataclass(frozen=True, slots=True)
class RawFileInfo:
    """Structural facts about one raw file (no data values)."""

    name: str
    bytes: int
    sha256: str
    rows: int
    first_date: date | None
    last_date: date | None
    window_start: date
    window_end: date


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One file as recorded in the manifest."""

    name: str
    bytes: int
    sha256: str
    rows: int
    first_date: date
    last_date: date


@dataclass(frozen=True, slots=True)
class Manifest:
    """Parsed ``MANIFEST.toml``."""

    schema_version: int
    series: str
    field: str
    source_url: str
    retrieved: date
    files: tuple[ManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class Flag:
    """A report-only observation. ``detail`` never contains a data value."""

    category: str
    date: date
    detail: str


@dataclass(frozen=True, slots=True)
class SpotCheckItem:
    """A date the owner checks manually against another public source."""

    date: date
    file: str
    line: int


@dataclass(frozen=True, slots=True)
class YearAudit:
    """Per-calendar-year counts (D26): rows, weekend rows, absent weekdays."""

    year: int
    rows: int
    weekend_rows: int
    absent_weekdays: int


@dataclass(frozen=True, slots=True)
class AbsentWeekday:
    """A Mon-Fri date strictly inside the series with no row (D26; report only).

    ``run_length`` is the number of consecutive absent weekdays (Fri -> Mon adjacent) in the
    maximal run containing ``date``.
    """

    date: date
    run_length: int


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Result of :func:`validate`. Contains dates, counts and hashes only."""

    files: tuple[RawFileInfo, ...]
    n_rows: int
    first_date: date | None
    last_date: date | None
    errors: tuple[str, ...]
    flags: tuple[Flag, ...]
    counts: dict[str, int]
    spot_check: tuple[SpotCheckItem, ...] = ()
    first_close_equals_base_value: bool | None = None
    yearly: tuple[YearAudit, ...] = ()
    absent_weekdays: tuple[AbsentWeekday, ...] = ()

    @property
    def ok(self) -> bool:
        """True when there are no hard errors."""
        return not self.errors


# ---------------------------------------------------------------------------
# File names and hashing
# ---------------------------------------------------------------------------
def parse_window_from_name(name: str) -> tuple[date, date]:
    """Window ``(start, end)`` encoded in an NSE file name (``...PR_DDMMYYYYtoDDMMYYYY.csv``).

    Raises:
        DataValidationError: if the name does not follow the pattern or encodes an invalid window.
    """
    m = _NAME_RE.match(name)
    if m is None:
        raise DataValidationError([f"{name}: file name does not match the expected NSE pattern"])
    d1, m1, y1, d2, m2, y2 = (int(g) for g in m.groups())
    try:
        start, end = date(y1, m1, d1), date(y2, m2, d2)
    except ValueError as exc:
        raise DataValidationError([f"{name}: file name encodes an invalid date"]) from exc
    if end < start:
        raise DataValidationError([f"{name}: window end precedes window start"])
    return start, end


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        raise DataValidationError([f"raw directory not found: {raw_dir.as_posix()}"])
    return sorted(
        (p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"),
        key=lambda p: p.name,
    )


# ---------------------------------------------------------------------------
# Strict parsing
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Parsed:
    name: str
    data: bytes
    window: tuple[date, date] | None
    dates: list[date] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    open: list[float] = field(default_factory=list)
    high: list[float] = field(default_factory=list)
    low: list[float] = field(default_factory=list)
    close: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _parse_date(cell: str) -> date | None:
    m = _DATE_RE.match(cell)
    if m is None or m.group(2) not in _MONTHS:
        return None
    try:
        return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    except ValueError:
        return None


def _parse_number(cell: str) -> float | None:
    """Exact decimal without sign, exponent or separators; ``None`` if not such a number."""
    if _NUMBER_RE.match(cell) is None:
        return None
    return float(cell)


def _parse_file(path: Path) -> _Parsed:
    data = path.read_bytes()
    out = _Parsed(name=path.name, data=data, window=None)
    try:
        out.window = parse_window_from_name(path.name)
    except DataValidationError as exc:
        out.errors.extend(exc.errors)
        return out
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        out.errors.append(f"{path.name}: not valid UTF-8")
        return out
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or tuple(rows[0]) != EXPECTED_HEADER:
        out.errors.append(f"{path.name}: header does not match {list(EXPECTED_HEADER)}")
        return out
    start, end = out.window
    for line_no, row in enumerate(rows[1:], start=2):
        where = f"{path.name}: line {line_no}"
        if len(row) != len(EXPECTED_HEADER):
            out.errors.append(f"{where}: expected {len(EXPECTED_HEADER)} fields, got {len(row)}")
            continue
        name, date_cell, o_cell, h_cell, l_cell, c_cell = row
        if name != EXPECTED_INDEX_NAME:
            out.errors.append(f"{where}: Index Name is not {EXPECTED_INDEX_NAME!r}")
            continue
        d = _parse_date(date_cell)
        if d is None:
            out.errors.append(f"{where}: Date is not in 'dd Mon yyyy' format")
            continue
        c = _parse_number(c_cell)
        if c is None:
            out.errors.append(f"{where} ({d.isoformat()}): Close is not a valid number")
            continue
        if not math.isfinite(c) or c <= 0:
            out.errors.append(f"{where} ({d.isoformat()}): Close is not finite and positive")
            continue
        ohl: list[float] = []
        bad_ohl = False
        for label, cell in (("Open", o_cell), ("High", h_cell), ("Low", l_cell)):
            if cell in MISSING_TOKENS:
                ohl.append(math.nan)
                continue
            v = _parse_number(cell)
            if v is None or not math.isfinite(v) or v <= 0:
                out.errors.append(f"{where} ({d.isoformat()}): {label} is not a valid number")
                bad_ohl = True
                break
            ohl.append(v)
        if bad_ohl:
            continue
        if d < start or d > end:
            out.errors.append(
                f"{where}: date {d.isoformat()} outside the file window "
                f"{start.isoformat()}..{end.isoformat()}"
            )
        out.dates.append(d)
        out.lines.append(line_no)
        out.open.append(ohl[0])
        out.high.append(ohl[1])
        out.low.append(ohl[2])
        out.close.append(c)
    return out


def _info(p: _Parsed) -> RawFileInfo:
    assert p.window is not None
    return RawFileInfo(
        name=p.name,
        bytes=len(p.data),
        sha256=_sha256(p.data),
        rows=len(p.dates),
        first_date=min(p.dates) if p.dates else None,
        last_date=max(p.dates) if p.dates else None,
        window_start=p.window[0],
        window_end=p.window[1],
    )


def _structure_errors(parsed: Sequence[_Parsed]) -> list[str]:
    """Duplicate content, window contiguity/overlap and duplicate dates across files."""
    errors: list[str] = []
    by_hash: dict[str, list[str]] = {}
    for p in parsed:
        by_hash.setdefault(_sha256(p.data), []).append(p.name)
    for names in sorted(by_hash.values()):
        if len(names) > 1:
            errors.append(f"identical file content (same sha256): {', '.join(sorted(names))}")
    windowed = sorted((p for p in parsed if p.window is not None), key=lambda p: p.window or ())
    for prev, cur in pairwise(windowed):
        assert prev.window is not None and cur.window is not None
        if cur.window[0] <= prev.window[1]:
            errors.append(f"overlapping windows: {prev.name} and {cur.name}")
        elif cur.window[0] != prev.window[1] + timedelta(days=1):
            errors.append(f"non-contiguous windows: gap between {prev.name} and {cur.name}")
    counts = Counter(d for p in parsed for d in p.dates)
    dups = sorted(d for d, n in counts.items() if n > 1)
    if dups:
        shown = ", ".join(d.isoformat() for d in dups[:10]) + (" ..." if len(dups) > 10 else "")
        errors.append(f"duplicate dates ({len(dups)}): {shown}")
    return errors


def inspect_files(raw_dir: str | Path) -> tuple[tuple[RawFileInfo, ...], tuple[str, ...]]:
    """Parse every raw file strictly; return per-file facts and all parse/structure errors."""
    paths = _csv_files(Path(raw_dir))
    parsed = [_parse_file(p) for p in paths]
    errors = [e for p in parsed for e in p.errors] + _structure_errors(parsed)
    if not paths:
        errors.append("no CSV files found in the raw directory")
    infos = tuple(_info(p) for p in parsed if p.window is not None)
    return infos, tuple(errors)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _toml_str(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def write_manifest(
    path: str | Path,
    files: Sequence[RawFileInfo],
    *,
    source_url: str,
    retrieved: date,
    series: str = "PR",
    field_name: str = "Close",
) -> None:
    """Write ``MANIFEST.toml`` deterministically (sorted, LF, no timestamps). Never overwrites.

    Raises:
        FileExistsError: if ``path`` already exists.
        DataValidationError: if a file has no parsed rows.
    """
    lines = [
        f"schema_version = {MANIFEST_SCHEMA_VERSION}",
        f"series = {_toml_str(series)}",
        f"field = {_toml_str(field_name)}",
        f"source_url = {_toml_str(source_url)}",
        f"retrieved = {_toml_str(retrieved.isoformat())}",
    ]
    for f in sorted(files, key=lambda x: x.name):
        if f.first_date is None or f.last_date is None:
            raise DataValidationError([f"{f.name}: no rows; cannot be recorded in the manifest"])
        lines += [
            "",
            "[[files]]",
            f"name = {_toml_str(f.name)}",
            f"bytes = {f.bytes}",
            f"sha256 = {_toml_str(f.sha256)}",
            f"rows = {f.rows}",
            f"first_date = {_toml_str(f.first_date.isoformat())}",
            f"last_date = {_toml_str(f.last_date.isoformat())}",
        ]
    with Path(path).open("x", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def read_manifest(path: str | Path) -> Manifest:
    """Read and type-check ``MANIFEST.toml``.

    Raises:
        DataValidationError: if the manifest is missing, malformed or of another schema.
    """
    p = Path(path)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError([f"manifest not found: {p.as_posix()}"]) from exc
    except tomllib.TOMLDecodeError as exc:
        raise DataValidationError([f"manifest is not valid TOML: {exc}"]) from exc
    try:
        if raw["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise DataValidationError([f"manifest schema_version is not {MANIFEST_SCHEMA_VERSION}"])
        entries = tuple(
            ManifestEntry(
                name=str(e["name"]),
                bytes=int(e["bytes"]),
                sha256=str(e["sha256"]),
                rows=int(e["rows"]),
                first_date=date.fromisoformat(str(e["first_date"])),
                last_date=date.fromisoformat(str(e["last_date"])),
            )
            for e in raw.get("files", [])
        )
        return Manifest(
            schema_version=int(raw["schema_version"]),
            series=str(raw["series"]),
            field=str(raw["field"]),
            source_url=str(raw["source_url"]),
            retrieved=date.fromisoformat(str(raw["retrieved"])),
            files=entries,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError([f"manifest is malformed: {exc!r}"]) from exc


def _manifest_errors(manifest_path: Path, raw_dir: Path) -> list[str]:
    """Compare the manifest with the files on disk (names, sizes, sha256)."""
    try:
        man = read_manifest(manifest_path)
    except DataValidationError as exc:
        return list(exc.errors)
    errors: list[str] = []
    if man.series != "PR" or man.field != "Close":
        errors.append("manifest series/field is not PR/Close")
    on_disk = {p.name: p for p in _csv_files(raw_dir)}
    listed = {e.name: e for e in man.files}
    for name in sorted(set(listed) - set(on_disk)):
        errors.append(f"manifest lists a missing file: {name}")
    for name in sorted(set(on_disk) - set(listed)):
        errors.append(f"file not in manifest: {name}")
    for name in sorted(set(listed) & set(on_disk)):
        data = on_disk[name].read_bytes()
        e = listed[name]
        if len(data) != e.bytes or _sha256(data) != e.sha256:
            errors.append(f"sha256/size mismatch with manifest: {name}")
    return errors


def _manifest_content_errors(manifest_path: Path, infos: Sequence[RawFileInfo]) -> list[str]:
    try:
        man = read_manifest(manifest_path)
    except DataValidationError:
        return []  # already reported by _manifest_errors
    listed = {e.name: e for e in man.files}
    errors: list[str] = []
    for f in infos:
        e = listed.get(f.name)
        if e is not None and (e.rows, e.first_date, e.last_date) != (
            f.rows,
            f.first_date,
            f.last_date,
        ):
            errors.append(f"rows/date range differ from manifest: {f.name}")
    return errors


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Rows:
    dates: list[date]
    files: list[str]
    lines: list[int]
    open: np.ndarray[Any, np.dtype[np.float64]]
    high: np.ndarray[Any, np.dtype[np.float64]]
    low: np.ndarray[Any, np.dtype[np.float64]]
    close: np.ndarray[Any, np.dtype[np.float64]]


def _collect(parsed: Sequence[_Parsed]) -> _Rows:
    recs = sorted(
        (
            (p.dates[i], p.name, p.lines[i], p.open[i], p.high[i], p.low[i], p.close[i])
            for p in parsed
            for i in range(len(p.dates))
        ),
        key=lambda r: (r[0], r[1], r[2]),
    )
    return _Rows(
        dates=[r[0] for r in recs],
        files=[r[1] for r in recs],
        lines=[r[2] for r in recs],
        open=np.array([r[3] for r in recs], dtype=np.float64),
        high=np.array([r[4] for r in recs], dtype=np.float64),
        low=np.array([r[5] for r in recs], dtype=np.float64),
        close=np.array([r[6] for r in recs], dtype=np.float64),
    )


def _frame(rows: _Rows) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in rows.dates], name="date")
    return pd.DataFrame(
        {"open": rows.open, "high": rows.high, "low": rows.low, "close": rows.close},
        index=index,
    )


def load_raw(
    raw_dir: str | Path, manifest_path: str | Path, start_date: date | None = None
) -> pd.DataFrame:
    """Verify the manifest, then parse every raw file strictly and return one ascending frame.

    Columns ``open, high, low, close`` (float64; missing Open/High/Low are NaN), DatetimeIndex
    named ``date``. Nothing is filled, interpolated or removed.

    Raises:
        DataValidationError: on any manifest mismatch (checked first) or other hard error.
    """
    raw, man = Path(raw_dir), Path(manifest_path)
    errors = _manifest_errors(man, raw)
    if errors:
        raise DataValidationError(errors)
    parsed = [_parse_file(p) for p in _csv_files(raw)]
    errors = [e for p in parsed for e in p.errors] + _structure_errors(parsed)
    infos = [_info(p) for p in parsed if p.window is not None]
    errors += _manifest_content_errors(man, infos)
    rows = _collect(parsed)
    if start_date is not None and rows.dates and rows.dates[0] != start_date:
        errors.append(f"first date {rows.dates[0].isoformat()} != start_date {start_date}")
    if errors:
        raise DataValidationError(errors)
    return _frame(rows)


# ---------------------------------------------------------------------------
# Validation (flags never alter the data)
# ---------------------------------------------------------------------------
def _flags(
    rows: _Rows, *, large_return_flag: Fraction, gap_flag_days: int, stale_run_min: int
) -> tuple[list[Flag], dict[str, int]]:
    flags: list[Flag] = []
    for d in rows.dates:
        if d.weekday() >= 5:
            flags.append(Flag("weekend_date", d, d.strftime("%A")))
    for prev, cur in pairwise(rows.dates):
        gap = (cur - prev).days
        if gap >= gap_flag_days:
            flags.append(Flag("calendar_gap", cur, f"{gap} calendar days since {prev.isoformat()}"))
    if len(rows.close) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            logret = np.abs(np.diff(np.log(rows.close)))
        threshold = float(large_return_flag)
        for k in np.flatnonzero(logret >= threshold).tolist():
            flags.append(Flag("large_return", rows.dates[k + 1], f"|log return| >= {threshold:g}"))
    run_start = 0
    for i in range(1, len(rows.close) + 1):
        if i == len(rows.close) or rows.close[i] != rows.close[run_start]:
            run = i - run_start
            if run >= stale_run_min:
                flags.append(
                    Flag(
                        "stale_close_run",
                        rows.dates[run_start],
                        f"{run} identical consecutive closes through "
                        f"{rows.dates[i - 1].isoformat()}",
                    )
                )
            elif run == 2:  # D28: exactly two; disjoint from stale_close_run
                flags.append(
                    Flag(
                        "identical_close_pair",
                        rows.dates[run_start + 1],
                        f"2 identical consecutive closes since {rows.dates[run_start].isoformat()}",
                    )
                )
            run_start = i
    o, h, lo, c = rows.open, rows.high, rows.low, rows.close
    full = ~(np.isnan(o) | np.isnan(h) | np.isnan(lo))
    for k in np.flatnonzero(full & (lo > np.minimum(o, c))).tolist():
        flags.append(Flag("ohlc_inconsistent", rows.dates[k], "low above min(open, close)"))
    for k in np.flatnonzero(full & (h < np.maximum(o, c))).tolist():
        flags.append(Flag("ohlc_inconsistent", rows.dates[k], "high below max(open, close)"))
    flags.sort(key=lambda f: (f.date, f.category, f.detail))
    counts = {cat: 0 for cat in FLAG_CATEGORIES}
    for f in flags:
        counts[f.category] += 1
    counts["missing_open"] = int(np.isnan(o).sum())
    counts["missing_high"] = int(np.isnan(h).sum())
    counts["missing_low"] = int(np.isnan(lo).sum())
    return flags, counts


def _calendar_audit(
    dates: Sequence[date],
) -> tuple[tuple[YearAudit, ...], tuple[AbsentWeekday, ...]]:
    """Absent weekdays strictly between the first and last date, and per-year counts (D26)."""
    if not dates:
        return (), ()
    present = set(dates)
    first, last = min(present), max(present)
    runs: list[list[date]] = []
    for k in range(1, (last - first).days):
        d = first + timedelta(days=k)
        if d.weekday() >= 5 or d in present:
            continue
        prev_weekday = d - timedelta(days=3 if d.weekday() == 0 else 1)
        if runs and runs[-1][-1] == prev_weekday:
            runs[-1].append(d)
        else:
            runs.append([d])
    absent = tuple(AbsentWeekday(d, len(run)) for run in runs for d in run)
    absent_by_year = Counter(a.date.year for a in absent)
    yearly = tuple(
        YearAudit(
            year=y,
            rows=sum(1 for d in dates if d.year == y),
            weekend_rows=sum(1 for d in dates if d.year == y and d.weekday() >= 5),
            absent_weekdays=absent_by_year[y],
        )
        for y in range(first.year, last.year + 1)
    )
    return yearly, absent


def _spot_check(rows: _Rows, n: int, seed: int) -> tuple[SpotCheckItem, ...]:
    k = min(n, len(rows.dates))
    if k == 0:
        return ()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    idx = sorted(int(i) for i in rng.choice(len(rows.dates), size=k, replace=False))
    return tuple(SpotCheckItem(rows.dates[i], rows.files[i], rows.lines[i]) for i in idx)


def validate(
    raw_dir: str | Path,
    manifest_path: str | Path,
    *,
    start_date: date,
    large_return_flag: Fraction,
    gap_flag_days: int,
    stale_run_min: int,
    spot_check_n: int,
    seed: int,
) -> ValidationReport:
    """Run every hard check (all errors collected) and compute report-only flags.

    The raw files are only read. Returns a report; it does not raise for data errors.
    """
    raw, man = Path(raw_dir), Path(manifest_path)
    try:
        paths = _csv_files(raw)
    except DataValidationError as exc:
        return ValidationReport((), 0, None, None, exc.errors, (), {"errors": len(exc.errors)})
    errors = _manifest_errors(man, raw)
    parsed = [_parse_file(p) for p in paths]
    errors += [e for p in parsed for e in p.errors] + _structure_errors(parsed)
    if not paths:
        errors.append("no CSV files found in the raw directory")
    infos = tuple(_info(p) for p in parsed if p.window is not None)
    errors += _manifest_content_errors(man, infos)
    rows = _collect(parsed)
    if rows.dates and rows.dates[0] != start_date:
        errors.append(f"first date {rows.dates[0].isoformat()} != start_date {start_date}")
    flags, counts = _flags(
        rows,
        large_return_flag=large_return_flag,
        gap_flag_days=gap_flag_days,
        stale_run_min=stale_run_min,
    )
    counts["errors"] = len(errors)
    yearly, absent = _calendar_audit(rows.dates)
    return ValidationReport(
        files=infos,
        n_rows=len(rows.dates),
        first_date=rows.dates[0] if rows.dates else None,
        last_date=rows.dates[-1] if rows.dates else None,
        errors=tuple(errors),
        flags=tuple(flags),
        counts=dict(sorted(counts.items())),
        spot_check=_spot_check(rows, spot_check_n, seed),
        first_close_equals_base_value=bool(rows.close[0] == BASE_VALUE) if rows.dates else None,
        yearly=yearly,
        absent_weekdays=absent,
    )


# ---------------------------------------------------------------------------
# Outputs (deterministic; dates, counts and hashes only)
# ---------------------------------------------------------------------------
def _iso(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None


def write_outputs(report: ValidationReport, out_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    """Write ``validation_report.json``, ``flags.csv``, ``spot_check_sheet.txt`` and
    ``absent_weekdays.csv`` (D26).

    Byte-identical for identical inputs: sorted keys, LF endings, no timestamps or absolute paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "ok": report.ok,
        "n_files": len(report.files),
        "n_rows": report.n_rows,
        "first_date": _iso(report.first_date),
        "last_date": _iso(report.last_date),
        "errors": list(report.errors),
        "counts": report.counts,
        "first_close_equals_base_value": report.first_close_equals_base_value,
        "files": [
            {
                "name": f.name,
                "bytes": f.bytes,
                "sha256": f.sha256,
                "rows": f.rows,
                "first_date": _iso(f.first_date),
                "last_date": _iso(f.last_date),
                "window_start": f.window_start.isoformat(),
                "window_end": f.window_end.isoformat(),
            }
            for f in sorted(report.files, key=lambda f: f.name)
        ],
        "spot_check_dates": [s.date.isoformat() for s in report.spot_check],
        "yearly": [
            {
                "year": y.year,
                "rows": y.rows,
                "weekend_rows": y.weekend_rows,
                "absent_weekdays": y.absent_weekdays,
            }
            for y in report.yearly
        ],
    }
    report_path = out / "validation_report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    flags_path = out / "flags.csv"
    with flags_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["date", "category", "detail"])
        for f in report.flags:
            writer.writerow([f.date.isoformat(), f.category, f.detail])
    sheet_path = out / "spot_check_sheet.txt"
    lines = [
        "NIFTYTAIL spot-check sheet (D25)",
        "For each date: open the named raw file at the given line, then compare its Close",
        "with an independent public source. Record the result below. No values are written here.",
        "",
        "date        | file | line | matches? (yes/no) | source checked",
    ]
    lines += [f"{s.date.isoformat()} | {s.file} | {s.line} |  |" for s in report.spot_check]
    with sheet_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    absent_path = out / "absent_weekdays.csv"
    with absent_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["date", "weekday", "run_length"])
        for a in report.absent_weekdays:
            writer.writerow([a.date.isoformat(), a.date.strftime("%A"), a.run_length])
    return report_path, flags_path, sheet_path, absent_path
