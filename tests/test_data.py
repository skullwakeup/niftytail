"""Tests for niftytail.data (Day 3A, D25).

All data here is SYNTHETIC and built in tmp_path in the exact NSE format found by the Day-3A
structure inspection (quoted fields, CRLF, newest row first, no trailing newline). The real
data/raw directory is never read.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from niftytail.__main__ import main
from niftytail.data import (
    AbsentWeekday,
    DataValidationError,
    RawFileInfo,
    ValidationReport,
    YearAudit,
    inspect_files,
    load_raw,
    parse_window_from_name,
    read_manifest,
    validate,
    write_manifest,
    write_outputs,
)

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
HEADER = '"Index Name","Date","Open","High","Low","Close"'
START = date(1995, 11, 3)
WINDOWS = (
    (date(1995, 11, 3), date(1995, 11, 30)),
    (date(1995, 12, 1), date(1995, 12, 31)),
    (date(1996, 1, 1), date(1996, 1, 19)),
)
SEED = 20261111
Row = tuple[date, str, str, str, str]  # date, open, high, low, close (as written)


# --- synthetic NSE-format builders ----------------------------------------------
def fmt_date(d: date) -> str:
    return f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}"


def window_name(start: date, end: date) -> str:
    return f"NIFTY 50_Historical_PR_{start:%d%m%Y}to{end:%d%m%Y}.csv"


def make_rows(start: date, end: date, first_close: float, seed: int) -> list[Row]:
    """Business days in [start, end]; small random-walk closes; consistent OHLC; 2 dp."""
    rng = np.random.default_rng(seed)
    rows: list[Row] = []
    close = first_close
    d = start
    while d <= end:
        if d.weekday() < 5:
            if rows:
                close = round(close * float(np.exp(rng.normal(0.0, 0.004))), 2)
            o = round(close * (1 + float(rng.uniform(-0.002, 0.002))), 2)
            h = round(max(o, close) * (1 + float(rng.uniform(0.0005, 0.003))), 2)
            lo = round(min(o, close) * (1 - float(rng.uniform(0.0005, 0.003))), 2)
            rows.append((d, f"{o:.2f}", f"{h:.2f}", f"{lo:.2f}", f"{close:.2f}"))
        d += timedelta(days=1)
    return rows


def write_nse_csv(path: Path, rows: list[Row], index_name: str = "NIFTY 50") -> None:
    lines = [HEADER] + [
        f'"{index_name}","{fmt_date(d)}","{o}","{h}","{lo}","{c}"'
        for d, o, h, lo, c in sorted(rows, key=lambda r: r[0], reverse=True)
    ]
    path.write_bytes("\r\n".join(lines).encode("utf-8"))


def build_raw(raw: Path, mutate: Callable[[int, list[Row]], list[Row]] | None = None) -> None:
    """Three contiguous synthetic windows from 1995-11-03; first close is the base value."""
    raw.mkdir(parents=True, exist_ok=True)
    close = 1000.0
    for i, (s, e) in enumerate(WINDOWS):
        rows = make_rows(s, e, close, SEED + i)
        close = float(rows[-1][4])
        if i > 0:  # continue the walk across files
            rows = make_rows(s, e, close * 1.001, SEED + i)
        if mutate is not None:
            rows = mutate(i, rows)
        write_nse_csv(raw / window_name(s, e), rows)


def make_manifest(raw: Path, man: Path) -> None:
    infos, errors = inspect_files(raw)
    assert errors == ()
    write_manifest(
        man, infos, source_url="https://example.invalid/nse", retrieved=date(2026, 11, 11)
    )


def run_validate(
    raw: Path,
    man: Path,
    *,
    start_date: date = START,
    large_return_flag: Fraction = Fraction(1, 20),
    gap_flag_days: int = 5,
    stale_run_min: int = 3,
    spot_check_n: int = 10,
    seed: int = SEED,
) -> ValidationReport:
    return validate(
        raw,
        man,
        start_date=start_date,
        large_return_flag=large_return_flag,
        gap_flag_days=gap_flag_days,
        stale_run_min=stale_run_min,
        spot_check_n=spot_check_n,
        seed=seed,
    )


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, Path]:
    raw, man = tmp_path / "raw", tmp_path / "raw" / "MANIFEST.toml"
    build_raw(raw)
    make_manifest(raw, man)
    return raw, man


def errors_of(raw: Path, man: Path, start_date: date = START) -> tuple[str, ...]:
    return run_validate(raw, man, start_date=start_date).errors


def first_file(raw: Path) -> Path:
    return raw / window_name(*WINDOWS[0])


# --- file names --------------------------------------------------------------------------
def test_parse_window_from_name() -> None:
    assert parse_window_from_name("NIFTY 50_Historical_PR_03111995to02111996.csv") == (
        date(1995, 11, 3),
        date(1996, 11, 2),
    )


@pytest.mark.parametrize(
    "name",
    [
        "NIFTY 50_Historical_TRI_03111995to02111996.csv",
        "nifty.csv",
        "NIFTY 50_Historical_PR_32111995to02111996.csv",
        "NIFTY 50_Historical_PR_03111996to02111995.csv",
    ],
)
def test_parse_window_from_name_rejects(name: str) -> None:
    with pytest.raises(DataValidationError):
        parse_window_from_name(name)


# --- successful load -----------------------------------------------------------------------
def test_clean_data_loads_sorted_and_typed(env: tuple[Path, Path]) -> None:
    raw, man = env
    df = load_raw(raw, man, start_date=START)
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert all(str(t) == "float64" for t in df.dtypes)
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index[0].date() == START
    infos, _ = inspect_files(raw)
    assert len(df) == sum(f.rows for f in infos)
    assert df["close"].iloc[0] == 1000.0


def test_clean_data_has_no_errors_or_flags(env: tuple[Path, Path]) -> None:
    rep = run_validate(*env)
    assert rep.ok and rep.errors == ()
    assert rep.flags == ()
    assert rep.first_close_equals_base_value is True
    assert rep.counts["missing_open"] == 0 and rep.counts["errors"] == 0
    assert len(rep.files) == 3


def test_base_value_flag_is_informational(tmp_path: Path) -> None:
    raw, man = tmp_path / "raw", tmp_path / "raw" / "MANIFEST.toml"

    def shift(i: int, rows: list[Row]) -> list[Row]:
        if i == 0:
            d, o, h, lo, _ = rows[0]
            rows[0] = (d, o, h, lo, "1001.00")
        return rows

    build_raw(raw, shift)
    make_manifest(raw, man)
    rep = run_validate(raw, man)
    assert rep.ok  # informational only: never an error
    assert rep.first_close_equals_base_value is False


# --- manifest -------------------------------------------------------------------------------
def test_manifest_round_trip_and_deterministic_bytes(
    env: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, man = env
    infos, _ = inspect_files(raw)
    m = read_manifest(man)
    assert (m.schema_version, m.series, m.field) == (1, "PR", "Close")
    assert m.retrieved == date(2026, 11, 11)
    assert [e.name for e in m.files] == sorted(f.name for f in infos)
    for e, f in zip(m.files, sorted(infos, key=lambda x: x.name), strict=True):
        assert (e.bytes, e.sha256, e.rows, e.first_date, e.last_date) == (
            f.bytes,
            f.sha256,
            f.rows,
            f.first_date,
            f.last_date,
        )
    other = tmp_path / "other_manifest.toml"
    write_manifest(
        other, infos, source_url="https://example.invalid/nse", retrieved=date(2026, 11, 11)
    )
    data = man.read_bytes()
    assert data == other.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert str(tmp_path).encode() not in data


def test_manifest_refuses_overwrite(env: tuple[Path, Path]) -> None:
    raw, man = env
    infos, _ = inspect_files(raw)
    before = man.read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(man, infos, source_url="https://x.invalid", retrieved=date(2026, 1, 1))
    assert man.read_bytes() == before


def test_manifest_string_escaping(tmp_path: Path, env: tuple[Path, Path]) -> None:
    raw, _ = env
    infos, _ = inspect_files(raw)
    p = tmp_path / "esc.toml"
    url = 'https://example.invalid/a"b\\c\td'
    write_manifest(p, infos, source_url=url, retrieved=date(2026, 11, 11))
    assert read_manifest(p).source_url == url


def test_manifest_rejects_file_without_rows(tmp_path: Path) -> None:
    info = RawFileInfo("x.csv", 1, "0" * 64, 0, None, None, START, START)
    with pytest.raises(DataValidationError):
        write_manifest(tmp_path / "m.toml", [info], source_url="https://x", retrieved=START)


def test_missing_and_malformed_manifest(env: tuple[Path, Path]) -> None:
    raw, man = env
    man.unlink()
    assert any("manifest not found" in e for e in errors_of(raw, man))
    man.write_text("schema_version = = 1", encoding="utf-8")
    assert any("not valid TOML" in e for e in errors_of(raw, man))
    man.write_text('schema_version = 2\nseries = "PR"\n', encoding="utf-8")
    assert any("schema_version" in e for e in errors_of(raw, man))
    man.write_text("schema_version = 1\n", encoding="utf-8")
    assert any("malformed" in e for e in errors_of(raw, man))


def test_manifest_series_and_row_mismatch(env: tuple[Path, Path], tmp_path: Path) -> None:
    raw, _ = env
    infos, _ = inspect_files(raw)
    bad = tmp_path / "bad.toml"
    wrong = [replace(infos[0], rows=infos[0].rows + 1), *infos[1:]]
    write_manifest(bad, wrong, source_url="https://x", retrieved=START, series="TRI")
    errs = errors_of(raw, bad)
    assert any("series/field" in e for e in errs)
    assert any("rows/date range differ" in e for e in errs)


# --- hard errors ------------------------------------------------------------------------------
def test_tampered_file_sha_mismatch_checked_first_by_load_raw(env: tuple[Path, Path]) -> None:
    raw, man = env
    p = first_file(raw)
    p.write_bytes(p.read_bytes().replace(b'"1000.00"', b'"abc"', 1))
    with pytest.raises(DataValidationError) as ei:
        load_raw(raw, man)
    assert ei.value.errors == (f"sha256/size mismatch with manifest: {p.name}",)
    assert any("sha256/size mismatch" in e for e in errors_of(raw, man))


def test_bad_file_name_and_extra_file(env: tuple[Path, Path]) -> None:
    raw, man = env
    shutil.copy(first_file(raw), raw / "nifty_extra.csv")
    errs = errors_of(raw, man)
    assert any("does not match the expected NSE pattern" in e for e in errs)
    assert any("file not in manifest: nifty_extra.csv" in e for e in errs)


def test_missing_file(env: tuple[Path, Path]) -> None:
    raw, man = env
    (raw / window_name(*WINDOWS[2])).unlink()
    assert any("manifest lists a missing file" in e for e in errors_of(raw, man))


def test_duplicate_file_content(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    build_raw(raw)
    (raw / window_name(*WINDOWS[2])).write_bytes(first_file(raw).read_bytes())
    errs = inspect_files(raw)[1]
    assert any("identical file content" in e for e in errs)
    assert any("outside the file window" in e for e in errs)


def test_duplicate_dates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    build_raw(raw, lambda i, rows: [*rows, rows[3]] if i == 1 else rows)
    assert any("duplicate dates (1)" in e for e in inspect_files(raw)[1])


@pytest.mark.parametrize(
    ("close", "message"),
    [
        ("0.00", "Close is not finite and positive"),
        ("abc", "Close is not a valid number"),
        ("", "Close is not a valid number"),
        ("1,000.00", "Close is not a valid number"),
        (" 1000.00", "Close is not a valid number"),
        ("-5.00", "Close is not a valid number"),
        ("1e3", "Close is not a valid number"),
        ("nan", "Close is not a valid number"),
    ],
)
def test_bad_close_is_hard_error(tmp_path: Path, close: str, message: str) -> None:
    raw = tmp_path / "raw"

    def bad(i: int, rows: list[Row]) -> list[Row]:
        if i == 0:
            d, o, h, lo, _ = rows[5]
            rows[5] = (d, o, h, lo, close)
        return rows

    build_raw(raw, bad)
    errs = inspect_files(raw)[1]
    assert any(message in e for e in errs), errs
    assert all(close not in e for e in errs if close)  # no values echoed (D23)


def test_bad_open_high_low_and_structure(tmp_path: Path) -> None:
    raw = tmp_path / "raw"

    def bad(i: int, rows: list[Row]) -> list[Row]:
        if i == 0:
            d, _, h, lo, c = rows[2]
            rows[2] = (d, "x1", h, lo, c)
        return rows

    build_raw(raw, bad)
    assert any("Open is not a valid number" in e for e in inspect_files(raw)[1])


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda b: b.replace(b'"Index Name"', b'"Name"', 1), "header does not match"),
        (lambda b: b.replace(b'"NIFTY 50"', b'"NIFTY 500"', 1), "Index Name is not"),
        (lambda b: b.replace(b' Nov 1995"', b' NOV 1995"', 1), "Date is not in"),
        (lambda b: b.replace(b'"03 Nov 1995"', b'"3 Nov 1995"', 1), "Date is not in"),
        (lambda b: b.replace(b'"03 Nov 1995"', b'"31 Nov 1995"', 1), "Date is not in"),
        (lambda b: b.replace(b'","1000.00"', b'","1000.00","9"', 1), "expected 6 fields"),
        (lambda b: b"\xff\xfe" + b, "not valid UTF-8"),
        (lambda b: b"", "header does not match"),
    ],
)
def test_strict_format_errors(
    tmp_path: Path, transform: Callable[[bytes], bytes], message: str
) -> None:
    raw = tmp_path / "raw"
    build_raw(raw)
    p = first_file(raw)
    p.write_bytes(transform(p.read_bytes()))
    errs = inspect_files(raw)[1]
    assert any(message in e for e in errs), errs


def test_date_outside_window(tmp_path: Path) -> None:
    raw = tmp_path / "raw"

    def late(i: int, rows: list[Row]) -> list[Row]:
        if i == 0:
            _, o, h, lo, c = rows[-1]
            rows[-1] = (date(1995, 12, 4), o, h, lo, c)
        return rows

    build_raw(raw, late)
    assert any("outside the file window" in e for e in inspect_files(raw)[1])


def test_overlapping_and_non_contiguous_windows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_nse_csv(
        raw / window_name(date(1995, 11, 3), date(1995, 11, 30)),
        make_rows(date(1995, 11, 3), date(1995, 11, 30), 1000.0, 1),
    )
    write_nse_csv(
        raw / window_name(date(1995, 11, 20), date(1995, 12, 20)),
        make_rows(date(1995, 12, 1), date(1995, 12, 20), 1000.0, 2),
    )
    write_nse_csv(
        raw / window_name(date(1996, 1, 5), date(1996, 1, 31)),
        make_rows(date(1996, 1, 5), date(1996, 1, 31), 1000.0, 3),
    )
    errs = inspect_files(raw)[1]
    assert any("overlapping windows" in e for e in errs)
    assert any("non-contiguous windows" in e for e in errs)


def test_wrong_first_date(env: tuple[Path, Path]) -> None:
    raw, man = env
    assert any("!= start_date" in e for e in errors_of(raw, man, start_date=date(1995, 11, 6)))
    with pytest.raises(DataValidationError, match="start_date"):
        load_raw(raw, man, start_date=date(1995, 11, 6))


def test_missing_and_empty_raw_dir(tmp_path: Path) -> None:
    rep = run_validate(tmp_path / "nope", tmp_path / "m.toml")
    assert not rep.ok and "raw directory not found" in rep.errors[0]
    with pytest.raises(DataValidationError):
        load_raw(tmp_path / "nope", tmp_path / "m.toml")
    (tmp_path / "empty").mkdir()
    assert any("no CSV files" in e for e in inspect_files(tmp_path / "empty")[1])


def test_load_raw_reports_parse_errors_after_manifest(tmp_path: Path) -> None:
    raw, man = tmp_path / "raw", tmp_path / "raw" / "MANIFEST.toml"
    build_raw(raw, lambda i, rows: [*rows, rows[3]] if i == 1 else rows)
    infos, _ = inspect_files(raw)
    write_manifest(man, infos, source_url="https://x", retrieved=START)
    with pytest.raises(DataValidationError, match="duplicate dates"):
        load_raw(raw, man)


# --- flags (report only; data never altered) ---------------------------------------------------
def _flag_dates(rep: ValidationReport, category: str) -> list[date]:
    return [f.date for f in rep.flags if f.category == category]


def _validated(tmp_path: Path, mutate: Callable[[int, list[Row]], list[Row]]) -> ValidationReport:
    raw, man = tmp_path / "raw", tmp_path / "raw" / "MANIFEST.toml"
    build_raw(raw, mutate)
    make_manifest(raw, man)
    rep = run_validate(raw, man)
    assert rep.ok, rep.errors
    return rep


def test_flag_weekend(tmp_path: Path) -> None:
    sat = date(1995, 12, 9)
    rep = _validated(
        tmp_path, lambda i, r: [*r, (sat, "1.00", "1.10", "0.90", r[-1][4])] if i == 1 else r
    )
    assert _flag_dates(rep, "weekend_date") == [sat]
    assert rep.counts["weekend_date"] == 1


def test_flag_calendar_gap(tmp_path: Path) -> None:
    def gap(i: int, rows: list[Row]) -> list[Row]:
        if i == 1:
            return [r for r in rows if not date(1995, 12, 11) <= r[0] <= date(1995, 12, 15)]
        return rows

    rep = _validated(tmp_path, gap)
    assert _flag_dates(rep, "calendar_gap") == [date(1995, 12, 18)]
    assert "10 calendar days" in next(f.detail for f in rep.flags if f.category == "calendar_gap")


def test_flag_large_return(tmp_path: Path) -> None:
    def spike(i: int, rows: list[Row]) -> list[Row]:
        if i == 1:
            d, _, _, _, c = rows[6]
            v = round(float(c) * 1.10, 2)
            rows[6] = (d, f"{v:.2f}", f"{v * 1.001:.2f}", f"{v * 0.999:.2f}", f"{v:.2f}")
        return rows

    rep = _validated(tmp_path, spike)
    dates = _flag_dates(rep, "large_return")
    assert len(dates) == 2  # into and out of the spike
    assert all(
        f.detail == "|log return| >= 0.05" for f in rep.flags if f.category == "large_return"
    )


def test_flag_stale_run(tmp_path: Path) -> None:
    def stale(i: int, rows: list[Row]) -> list[Row]:
        if i == 2:
            c = rows[4][4]
            for k in (5, 6):
                d, _, _, _, _ = rows[k]
                rows[k] = (d, c, f"{float(c) + 1:.2f}", f"{float(c) - 1:.2f}", c)
        return rows

    rep = _validated(tmp_path, stale)
    assert rep.counts["stale_close_run"] == 1
    assert "3 identical consecutive closes" in next(
        f.detail for f in rep.flags if f.category == "stale_close_run"
    )


def test_flag_ohlc_inconsistent(tmp_path: Path) -> None:
    def bad(i: int, rows: list[Row]) -> list[Row]:
        if i == 0:
            d, o, h, _, c = rows[7]
            rows[7] = (d, o, h, f"{max(float(o), float(c)) + 5:.2f}", c)
            d, o, _, lo, c = rows[8]
            rows[8] = (d, o, f"{min(float(o), float(c)) - 5:.2f}", lo, c)
        return rows

    rep = _validated(tmp_path, bad)
    details = sorted(f.detail for f in rep.flags if f.category == "ohlc_inconsistent")
    assert "low above min(open, close)" in details and "high below max(open, close)" in details


def test_missing_open_high_low_kept_as_nan_and_counted(tmp_path: Path) -> None:
    def missing(i: int, rows: list[Row]) -> list[Row]:
        if i == 0:
            d, _, _, _, c = rows[3]
            rows[3] = (d, "-", "-", "-", c)
        return rows

    raw, man = tmp_path / "raw", tmp_path / "raw" / "MANIFEST.toml"
    build_raw(raw, missing)
    make_manifest(raw, man)
    rep = run_validate(raw, man)
    assert rep.ok
    assert (rep.counts["missing_open"], rep.counts["missing_high"], rep.counts["missing_low"]) == (
        1,
        1,
        1,
    )
    df = load_raw(raw, man)
    assert int(df[["open", "high", "low"]].isna().sum().sum()) == 3
    assert not df["close"].isna().any()


def test_validation_never_modifies_raw_files(env: tuple[Path, Path]) -> None:
    raw, man = env
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    run_validate(raw, man, large_return_flag=Fraction(1, 1000), gap_flag_days=1, stale_run_min=1)
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == before


# --- outputs and spot check ----------------------------------------------------------------------
def test_outputs_deterministic_and_value_free(env: tuple[Path, Path], tmp_path: Path) -> None:
    raw, man = env
    a, b = tmp_path / "out_a", tmp_path / "out_b"
    pa = write_outputs(run_validate(raw, man), a)
    pb = write_outputs(run_validate(raw, man), b)
    values = {
        v
        for p in raw.glob("*.csv")
        for v in re.findall(r'"(\d+\.\d+)"', p.read_text(encoding="utf-8"))
    }
    assert "1000.00" in values and len(values) > 100
    for x, y in zip(pa, pb, strict=True):
        data = x.read_bytes()
        assert data == y.read_bytes()
        assert b"\r\n" not in data
        text = data.decode("utf-8")
        assert str(tmp_path) not in text and tmp_path.as_posix() not in text
        assert not any(v in text for v in values), x.name  # D23: no data values written
    report = json.loads(pa[0].read_text(encoding="utf-8"))
    assert report["ok"] is True and report["n_files"] == 3
    assert "time" not in report and "timestamp" not in report
    assert pa[1].read_text(encoding="utf-8").splitlines()[0] == "date,category,detail"


def test_spot_check_reproducible_and_seeded(env: tuple[Path, Path]) -> None:
    raw, man = env
    r1, r2 = run_validate(raw, man), run_validate(raw, man)
    r3 = run_validate(raw, man, seed=SEED + 1)
    d1 = [s.date for s in r1.spot_check]
    assert d1 == [s.date for s in r2.spot_check]
    assert d1 != [s.date for s in r3.spot_check]
    assert len(d1) == 10 and d1 == sorted(set(d1))
    assert all(s.line >= 2 and s.file.endswith(".csv") for s in r1.spot_check)
    small = run_validate(raw, man, spot_check_n=10_000)
    assert len(small.spot_check) == small.n_rows


# --- CLI ------------------------------------------------------------------------------------------
@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_root: Path) -> Path:
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    build_raw(tmp_path / "data" / "raw")
    monkeypatch.chdir(tmp_path)
    return tmp_path


INIT = [
    "--init-manifest",
    "--retrieved",
    "2026-11-11",
    "--source-url",
    "https://example.invalid/nse",
]


def test_cli_validate_init_then_rerun_identical(cli_env: Path) -> None:
    cfg = "configs/base.toml"
    assert main(["validate", cfg, *INIT]) == 0
    out = cli_env / "data" / "validation"
    names = ("validation_report.json", "flags.csv", "spot_check_sheet.txt")
    first = {n: (out / n).read_bytes() for n in names}
    assert main(["validate", cfg]) == 0
    assert {n: (out / n).read_bytes() for n in names} == first
    assert main(["validate", cfg, *INIT]) == 1  # refuses to overwrite the manifest


def test_cli_validate_exit_1_on_errors(cli_env: Path) -> None:
    cfg = "configs/base.toml"
    assert main(["validate", cfg]) == 1  # no manifest yet
    assert main(["validate", cfg, *INIT]) == 0
    p = cli_env / "data" / "raw" / window_name(*WINDOWS[1])
    p.write_bytes(p.read_bytes() + b" ")
    assert main(["validate", cfg]) == 1
    report = json.loads((cli_env / "data" / "validation" / "validation_report.json").read_text())
    assert report["ok"] is False and report["counts"]["errors"] >= 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--init-manifest"],
        ["--init-manifest", "--retrieved", "11-11-2026", "--source-url", "https://x.invalid"],
        ["--init-manifest", "--retrieved", "2026-11-11", "--source-url", "ftp://x.invalid"],
        ["--retrieved", "2026-11-11"],
    ],
)
def test_cli_validate_argument_errors(cli_env: Path, extra: list[str]) -> None:
    assert main(["validate", "configs/base.toml", *extra]) == 1
    assert not (cli_env / "data" / "raw" / "MANIFEST.toml").exists()


def test_cli_init_refuses_when_raw_files_have_errors(cli_env: Path) -> None:
    p = cli_env / "data" / "raw" / window_name(*WINDOWS[0])
    p.write_bytes(p.read_bytes().replace(b'"1000.00"', b'"abc"', 1))
    assert main(["validate", "configs/base.toml", *INIT]) == 1
    assert not (cli_env / "data" / "raw" / "MANIFEST.toml").exists()
    shutil.rmtree(cli_env / "data" / "raw")
    assert main(["validate", "configs/base.toml", *INIT]) == 1


def test_cli_validate_invalid_config(cli_env: Path) -> None:
    (cli_env / "configs" / "bad.toml").write_text('spec_version = "V2.1"\n', encoding="utf-8")
    assert main(["validate", "configs/bad.toml"]) == 1


def test_validation_outputs_are_gitignored(project_root: Path) -> None:
    for rel in (
        "data/validation/validation_report.json",
        "data/validation/flags.csv",
        "data/validation/spot_check_sheet.txt",
        "data/raw/MANIFEST.toml",
    ):
        res = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", rel],
            cwd=project_root,
            capture_output=True,
            check=False,
        )
        assert res.returncode == 0, f"{rel} is NOT gitignored"


# --- absent-weekday audit (D26; report only) ------------------------------------------------------
def _business_days(start: date, end: date) -> int:
    return sum(1 for k in range((end - start).days + 1) if (start + timedelta(k)).weekday() < 5)


def _drop(*days: date) -> Callable[[int, list[Row]], list[Row]]:
    return lambda i, rows: [r for r in rows if r[0] not in days]


def _with_saturday(sat: date) -> Callable[[int, list[Row]], list[Row]]:
    def add(i: int, rows: list[Row]) -> list[Row]:
        if WINDOWS[i][0] <= sat <= WINDOWS[i][1]:
            prev = max(r for r in rows if r[0] < sat)
            rows = [*rows, (sat, prev[1], prev[2], prev[3], prev[4])]
        return rows

    return add


def test_no_absent_weekdays_and_partial_years(env: tuple[Path, Path]) -> None:
    rep = run_validate(*env)
    assert rep.absent_weekdays == ()
    # partial first year (from 1995-11-03, a Friday) and partial last year (to 1996-01-19)
    assert rep.yearly == (
        YearAudit(1995, _business_days(START, date(1995, 12, 31)), 0, 0),
        YearAudit(1996, _business_days(date(1996, 1, 1), date(1996, 1, 19)), 0, 0),
    )
    assert sum(y.rows for y in rep.yearly) == rep.n_rows


def test_trailing_missing_days_are_not_absent(tmp_path: Path) -> None:
    rep = _validated(tmp_path, _drop(date(1996, 1, 18), date(1996, 1, 19)))
    assert rep.last_date == date(1996, 1, 17)
    assert rep.absent_weekdays == ()
    assert rep.yearly[-1] == YearAudit(1996, _business_days(date(1996, 1, 1), rep.last_date), 0, 0)


def test_single_absent_weekday(tmp_path: Path) -> None:
    rep = _validated(tmp_path, _drop(date(1995, 12, 13)))
    assert rep.absent_weekdays == (AbsentWeekday(date(1995, 12, 13), 1),)
    assert [y.absent_weekdays for y in rep.yearly] == [1, 0]


def test_absent_run_across_weekend_counts_weekdays_only(tmp_path: Path) -> None:
    days = (date(1995, 12, 15), date(1995, 12, 18), date(1995, 12, 19))  # Fri, Mon, Tue
    drop, sat = _drop(*days), _with_saturday(date(1995, 12, 16))
    rep = _validated(tmp_path, lambda i, rows: sat(i, drop(i, rows)))
    assert rep.absent_weekdays == tuple(AbsentWeekday(d, 3) for d in days)
    assert rep.yearly[0].weekend_rows == 1


def test_weekend_rows_are_not_absent_and_are_counted(tmp_path: Path) -> None:
    sat = date(1995, 12, 9)
    rep = _validated(tmp_path, _with_saturday(sat))
    assert rep.absent_weekdays == ()
    assert rep.yearly[0] == YearAudit(1995, _business_days(START, date(1995, 12, 31)) + 1, 1, 0)
    assert rep.yearly[1].weekend_rows == 0


def test_absent_run_across_year_boundary(tmp_path: Path) -> None:
    days = (date(1995, 12, 29), date(1996, 1, 1))  # Fri, Mon; different files and years
    rep = _validated(tmp_path, _drop(*days))
    assert rep.absent_weekdays == tuple(AbsentWeekday(d, 2) for d in days)
    assert [(y.year, y.absent_weekdays) for y in rep.yearly] == [(1995, 1), (1996, 1)]


def test_absent_weekday_outputs_deterministic_and_value_free(tmp_path: Path) -> None:
    rep = _validated(tmp_path, _drop(date(1995, 12, 13), date(1995, 12, 29), date(1996, 1, 1)))
    pa = write_outputs(rep, tmp_path / "a")
    pb = write_outputs(rep, tmp_path / "b")
    assert [p.name for p in pa][3] == "absent_weekdays.csv"
    assert pa[3].read_bytes() == pb[3].read_bytes()
    assert pa[0].read_bytes() == pb[0].read_bytes()
    text = pa[3].read_bytes().decode("utf-8")
    assert "\r" not in text
    assert text.splitlines() == [
        "date,weekday,run_length",
        "1995-12-13,Wednesday,1",
        "1995-12-29,Friday,2",
        "1996-01-01,Monday,2",
    ]
    report = json.loads(pa[0].read_text(encoding="utf-8"))
    assert report["yearly"] == [
        {"absent_weekdays": 2, "rows": rep.yearly[0].rows, "weekend_rows": 0, "year": 1995},
        {"absent_weekdays": 1, "rows": rep.yearly[1].rows, "weekend_rows": 0, "year": 1996},
    ]
    values = {
        v
        for p in (tmp_path / "raw").glob("*.csv")
        for v in re.findall(r'"(\d+\.\d+)"', p.read_text(encoding="utf-8"))
    }
    yearly_text = json.dumps(report["yearly"])
    assert not any(v in text or v in yearly_text for v in values)  # D23


def test_absent_weekdays_output_is_gitignored(project_root: Path) -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "data/validation/absent_weekdays.csv"],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    assert res.returncode == 0


def test_data_doc_has_no_index_levels(project_root: Path) -> None:
    """docs/data.md (D23, D26) must not contain anything shaped like an index level."""
    doc = project_root / "docs" / "data.md"
    text = doc.read_text(encoding="utf-8") if doc.exists() else ""
    assert re.findall(r"\b\d{3,6}\.\d{2}\b", text) == []


def test_data_doc_lists_no_flagged_day(project_root: Path) -> None:
    """D65 (j): large-return and identical-close days are counts only, and the halts
    table has no per-day flag column."""
    text = (project_root / "docs" / "data.md").read_text(encoding="utf-8")
    iso = re.compile(r"\d{4}-\d{2}-\d{2}")
    large = text.split("### 5.3 Large returns")[1].split("**Market-wide halts")[0]
    pair = next(ln for ln in text.splitlines() if ln.startswith("- **identical_close_pair"))
    weekend = text.split("### 5.1 Weekend dates")[1].split("| date |")[0]
    assert not iso.search(large) and not iso.search(pair) and not iso.search(weekend)
    assert "80 dates" in large and "):** 8." in pair
    assert "large_return flag |" not in text and "| in series |" not in text


# --- identical-close pair flag (D28; report only) -------------------------------------------------
def _repeat_close(window: int, *ks: int) -> Callable[[int, list[Row]], list[Row]]:
    """Give row k of the window the close of row k - 1 (consistent OHLC)."""

    def mutate(i: int, rows: list[Row]) -> list[Row]:
        if i == window:
            for k in ks:
                c = rows[k - 1][4]
                rows[k] = (rows[k][0], c, f"{float(c) + 1:.2f}", f"{float(c) - 1:.2f}", c)
        return rows

    return mutate


def _pairs(rep: ValidationReport) -> list[tuple[date, str]]:
    return [(f.date, f.detail) for f in rep.flags if f.category == "identical_close_pair"]


def test_identical_close_pair_is_flagged_at_second_row(tmp_path: Path) -> None:
    rep = _validated(tmp_path, _repeat_close(2, 5))
    first, second = date(1996, 1, 5), date(1996, 1, 8)  # rows 4 and 5 of the third window
    assert _pairs(rep) == [(second, f"2 identical consecutive closes since {first.isoformat()}")]
    assert rep.counts["identical_close_pair"] == 1 and rep.counts["stale_close_run"] == 0


def test_run_of_three_is_stale_not_pair(tmp_path: Path) -> None:
    rep = _validated(tmp_path, _repeat_close(2, 5, 6))
    assert rep.counts["stale_close_run"] == 1
    assert rep.counts["identical_close_pair"] == 0 and _pairs(rep) == []


def test_two_separate_pairs(tmp_path: Path) -> None:
    rep = _validated(tmp_path, _repeat_close(1, 3, 10))
    assert [d for d, _ in _pairs(rep)] == [date(1995, 12, 6), date(1995, 12, 15)]
    assert rep.counts["identical_close_pair"] == 2


def test_identical_close_pair_across_file_boundary(tmp_path: Path) -> None:
    last: dict[str, Row] = {}

    def mutate(i: int, rows: list[Row]) -> list[Row]:
        if i == 0:
            last["row"] = rows[-1]
        if i == 1:
            c = last["row"][4]
            rows[0] = (rows[0][0], c, f"{float(c) + 1:.2f}", f"{float(c) - 1:.2f}", c)
        return rows

    rep = _validated(tmp_path, mutate)
    assert _pairs(rep) == [(date(1995, 12, 1), "2 identical consecutive closes since 1995-11-30")]


def test_weekend_row_repeating_previous_close_is_a_pair(tmp_path: Path) -> None:
    sat = date(1995, 12, 9)
    rep = _validated(tmp_path, _with_saturday(sat))
    assert _pairs(rep) == [(sat, "2 identical consecutive closes since 1995-12-08")]
    assert [f.date for f in rep.flags if f.category == "weekend_date"] == [sat]


def test_pair_flags_sorted_deterministic_and_value_free(tmp_path: Path) -> None:
    mutate_a, mutate_b = _repeat_close(0, 4), _repeat_close(2, 7)
    rep = _validated(tmp_path, lambda i, rows: mutate_b(i, mutate_a(i, rows)))
    raw, man = tmp_path / "raw", tmp_path / "raw" / "MANIFEST.toml"
    again = run_validate(raw, man)
    assert rep.flags == again.flags
    assert list(rep.flags) == sorted(rep.flags, key=lambda f: (f.date, f.category, f.detail))
    assert rep.counts["identical_close_pair"] == 2
    paths = write_outputs(rep, tmp_path / "out")
    values = {
        v
        for p in raw.glob("*.csv")
        for v in re.findall(r'"(\d+\.\d+)"', p.read_text(encoding="utf-8"))
    }
    flags_text = paths[1].read_text(encoding="utf-8")
    assert flags_text.count(",identical_close_pair,") == 2
    for p in paths[:2]:
        text = p.read_text(encoding="utf-8")
        assert not any(v in text for v in values), p.name  # D23
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    assert report["counts"]["identical_close_pair"] == 2


def test_counts_include_identical_close_pair_key(env: tuple[Path, Path]) -> None:
    rep = run_validate(*env)
    assert rep.counts["identical_close_pair"] == 0
