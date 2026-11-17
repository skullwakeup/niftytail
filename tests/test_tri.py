"""Day-11C TRI sensitivity (D48). Synthetic data only.

TRI files are written in the owner's format (quoted, CRLF, newest first; header IndexName, Date,
Total Returns Index, Net Total Return Index). Integration tests use a temporary git repository
with a synthetic PR series from 1995-11-03 and a synthetic TRI series from 1999-06-30 on the same
business-day calendar, a small design (W = 100, 40 OOS targets) and a PR W = 100 backtest.json
written in the tracked place for the side-by-side.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from conftest import PROJECT_ROOT
from niftytail import backtest, models, oos, rolling, tri
from niftytail import data as dl
from niftytail import losses as ls
from niftytail.__main__ import configure_logging, main
from niftytail.config import Config, load_config
from niftytail.garch import GarchError
from niftytail.threshold_study import garch_returns
from test_data import fmt_date, make_manifest, make_rows, window_name, write_nse_csv
from test_threshold_real import ISO_DATE

CONFIG = "configs/base.toml"
PR_START = date(1995, 11, 3)
END = date(2000, 2, 15)
W, TARGETS = 100, 40
HEADER_LINE = '"IndexName","Date","Total Returns Index","Net Total Return Index"'


def _bdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def tri_name(start: date, end: date) -> str:
    return f"NIFTY 50_Historical_TR_{start:%d%m%Y}to{end:%d%m%Y}.csv"


def write_tri(path: Path, rows: list[tuple[date, str, str]], header: str = HEADER_LINE) -> None:
    lines = [header] + [
        f'"NIFTY 50","{fmt_date(d)}","{g}","{n}"' for d, g, n in sorted(rows, reverse=True)
    ]
    path.write_bytes("\r\n".join(lines).encode("utf-8"))


def tri_rows(dates: list[date], seed: int = 48, n_dash: int = 5) -> list[tuple[date, str, str]]:
    g = np.random.default_rng(seed)
    lv = 1500.0 * np.exp(np.cumsum(0.01 * g.standard_t(6, len(dates))))
    return [
        (d, f"{v:.2f}", "-" if i < n_dash else f"{0.99 * v:.2f}")
        for i, (d, v) in enumerate(zip(dates, lv, strict=True))
    ]


def _one(tmp: Path, rows: list[tuple[date, str, str]], name: str | None = None) -> tuple[Any, int]:
    p = tmp / (name or tri_name(rows[0][0], rows[-1][0]))
    write_tri(p, rows)
    return tri._parse_file(p)


# --- parser ---------------------------------------------------------------------------------
def test_parser_ignores_net_and_parses_gross(tmp_path: Path) -> None:
    rows = tri_rows(_bdays(tri.TRI_START, date(1999, 8, 31)), n_dash=7)
    rows[10] = (rows[10][0], rows[10][1], "garbage")  # Net is never parsed
    parsed, dash = _one(tmp_path, rows)
    assert parsed.errors == [] and dash == 7 and len(parsed.dates) == len(rows)
    assert parsed.close == [float(g) for _, g, _ in reversed(rows)]  # file order: newest first
    assert all(np.isnan(parsed.open)) and parsed.dates == sorted(parsed.dates, reverse=True)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: (r[0], "-", r[2]), "Total Returns Index is not valid"),
        (lambda r: (r[0], "0", r[2]), "Total Returns Index is not valid"),
        (lambda r: (r[0], "1,234.5", r[2]), "Total Returns Index is not valid"),
        (lambda r: (r[0] + timedelta(days=400), r[1], r[2]), "outside the file window"),
    ],
)
def test_parser_hard_errors(tmp_path: Path, edit: Any, message: str) -> None:
    rows = tri_rows(_bdays(tri.TRI_START, date(1999, 8, 31)))
    name = tri_name(rows[0][0], rows[-1][0])
    rows[3] = edit(rows[3])
    parsed, _ = _one(tmp_path, rows, name)
    assert any(message in e for e in parsed.errors)


def test_parser_header_name_and_index_errors(tmp_path: Path) -> None:
    rows = tri_rows(_bdays(tri.TRI_START, date(1999, 7, 30)))
    p = tmp_path / tri_name(rows[0][0], rows[-1][0])
    write_tri(p, rows, header='"Index Name","Date","Total Returns Index","Net Total Return Index"')
    assert any("header" in e for e in tri._parse_file(p)[0].errors)
    bad = tmp_path / "NIFTY 50_Historical_PR_30061999to30071999.csv"
    write_tri(bad, rows)
    assert any("TRI pattern" in e for e in tri._parse_file(bad)[0].errors)
    p.write_bytes(
        p.read_bytes().replace(b'"NIFTY 50"', b'"NIFTY50"', 1).replace(b"Index Name", b"IndexName")
    )
    assert any("IndexName" in e for e in tri._parse_file(p)[0].errors)
    with pytest.raises(dl.DataValidationError):
        tri.parse_window("NIFTY 50_Historical_TR_31021999to30061999.csv")
    with pytest.raises(dl.DataValidationError, match="precedes"):
        tri.parse_window("NIFTY 50_Historical_TR_30061999to01011999.csv")
    p.write_bytes(b"\xff\xfe")
    assert any("UTF-8" in e for e in tri._parse_file(p)[0].errors)


# --- repository fixtures --------------------------------------------------------------------
def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _small(cfg: Config, n_pr_returns: int) -> Config:
    design = dataclasses.replace(cfg.design, train_size=n_pr_returns - TARGETS, window=W)
    return dataclasses.replace(cfg, design=design)


def make_repo(
    root: Path, *, pr_backtest: bool = False, tri_dates: list[date] | None = None
) -> Config:
    """PR + TRI raw data (ignored), configs, and optionally a PR W = 100 backtest.json (tracked)."""
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    pr = make_rows(PR_START, END, 1000.0, 20261111)
    write_nse_csv(raw / window_name(PR_START, END), pr)
    make_manifest(raw, raw / "MANIFEST.toml")
    traw = root / tri.TRI_RAW
    traw.mkdir(parents=True)
    dates = tri_dates if tri_dates is not None else _bdays(tri.TRI_START, END)
    write_tri(traw / tri_name(tri.TRI_START, END), tri_rows(dates))
    (root / ".gitignore").write_text("/data/\n/reports/runs/\n", encoding="utf-8", newline="\n")
    cfg = _small(load_config(root / CONFIG), len(pr) - 1)
    if pr_backtest:
        df = dl.load_raw(raw, raw / "MANIFEST.toml", cfg.data.start_date)
        x = ls.log_losses(df["close"].to_numpy())
        d = [t.date().isoformat() for t in ls.return_dates(pd.DatetimeIndex(df.index))]
        table, _ = rolling.run_rolling(x, d, cfg, W, 4)
        agg, _ = backtest.evaluate(table, cfg, W)
        out = root / tri.PR_W1000
        out.parent.mkdir(parents=True)
        out.write_text(backtest.to_json(agg), encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return cfg


def _validate(cfg: Config, root: Path) -> tuple[int, list[dict[str, Any]]]:
    buf = io.StringIO()
    status = tri.validate_command(cfg, root, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


def _run(cfg: Config, root: Path, verify: bool = False, workers: int = 1) -> tuple[int, list[Any]]:
    buf = io.StringIO()
    status = tri.run(cfg, CONFIG, workers, verify, root, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


def _nodate(r: dict[str, Any]) -> bool:
    return not ISO_DATE.search(json.dumps({k: v for k, v in r.items() if k != "time"}))


# --- validation and manifest ----------------------------------------------------------------
def test_validate_tri_writes_manifest_once(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    status, logs = _validate(cfg, tmp_path)
    assert status == 0 and logs[0]["message"] == "TRI manifest written"
    assert all(_nodate(r) for r in logs)
    man = tmp_path / tri.TRI_MANIFEST
    m = dl.read_manifest(man)
    assert (m.series, m.field, m.retrieved, m.source_url) == (
        "TRI",
        "Total Returns Index",
        date(2026, 11, 15),
        tri.SOURCE_URL,
    )
    before = (man.read_bytes(), man.stat().st_mtime_ns)
    status, logs = _validate(cfg, tmp_path)
    assert status == 0 and (man.read_bytes(), man.stat().st_mtime_ns) == before
    assert not any(r["message"] == "TRI manifest written" for r in logs)
    report = json.loads(
        (tmp_path / tri.TRI_VALIDATION / "validation_report.json").read_text("utf-8")
    )
    assert report["counts"]["net_dash"] == 5 and report["counts"]["errors"] == 0
    assert "missing_open" not in report["counts"] and "first_close_equals_base_value" not in report
    assert report["date_differences"] == {
        "pr_only": 0,
        "tri_only": 0,
        "pr_only_oos": 0,
        "tri_only_oos": 0,
    }
    assert len(report["spot_check_dates"]) == cfg.validation.spot_check_n
    assert "Total" in (tmp_path / tri.TRI_VALIDATION / "spot_check_sheet.txt").read_text("utf-8")


def test_date_differences_and_first_date(tmp_path: Path) -> None:
    dates = _bdays(tri.TRI_START, END)
    dates.remove(dates[50])  # one PR-only date
    dates.insert(60, date(1999, 9, 18))  # a Saturday: TRI-only, and a weekend flag
    cfg = make_repo(tmp_path, tri_dates=sorted(dates))
    status, _ = _validate(cfg, tmp_path)
    report = json.loads(
        (tmp_path / tri.TRI_VALIDATION / "validation_report.json").read_text("utf-8")
    )
    assert status == 0 and report["counts"]["weekend_date"] == 1
    assert (report["date_differences"]["pr_only"], report["date_differences"]["tri_only"]) == (1, 1)
    diff = (tmp_path / tri.TRI_VALIDATION / "date_differences.csv").read_text("utf-8").splitlines()
    assert diff[0] == "date,side" and len(diff) == 3


def test_wrong_first_date_and_no_manifest_on_errors(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path, tri_dates=_bdays(date(1999, 7, 1), END))
    (tmp_path / tri.TRI_RAW / tri_name(tri.TRI_START, END)).rename(
        tmp_path / tri.TRI_RAW / tri_name(date(1999, 7, 1), END)
    )
    status, logs = _validate(cfg, tmp_path)
    assert status == 1 and any("TRI start" in r.get("error", "") for r in logs)
    broken = tmp_path / "b"
    cfg2 = make_repo(broken)
    f = broken / tri.TRI_RAW / tri_name(tri.TRI_START, END)
    f.write_bytes(f.read_bytes().replace(b'"Total Returns Index"', b'"TRI"', 1))
    status, logs = _validate(cfg2, broken)
    assert status == 1 and not (broken / tri.TRI_MANIFEST).exists()
    assert logs[-1]["message"] == "TRI manifest not written: the raw files have errors"


def test_manifest_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    assert _validate(cfg, tmp_path)[0] == 0
    f = tmp_path / tri.TRI_RAW / tri_name(tri.TRI_START, END)
    f.write_bytes(f.read_bytes() + b" ")
    status, logs = _validate(cfg, tmp_path)
    assert status == 1 and any("sha256/size mismatch" in r.get("error", "") for r in logs)
    with pytest.raises(dl.DataValidationError):
        tri.load_tri(tmp_path / tri.TRI_RAW, tmp_path / tri.TRI_MANIFEST)


# --- alignment ------------------------------------------------------------------------------
def _days(n: int, start: date = date(2000, 1, 3)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def test_alignment() -> None:
    pr = _days(400)
    tri_ret = pr[30:]  # TRI starts at PR return index 30; PR OOS = pr[250:]
    assert tri.align(pr, tri_ret, 250, 100) == 220
    with pytest.raises(tri.TriAlignmentError, match="missing"):
        tri.align(pr, pr[30:250] + pr[251:], 250, 100)
    with pytest.raises(tri.TriAlignmentError, match=r"W \+ 1"):
        tri.align(pr, pr[200:], 250, 100)  # t0_tri = 50 < 101
    with pytest.raises(tri.TriAlignmentError, match="differ"):
        tri.align(pr, [*tri_ret[:280], date(2099, 1, 1), *tri_ret[280:]], 250, 100)  # inserted
    with pytest.raises(tri.TriAlignmentError, match="differ"):
        tri.align(pr, tri_ret[:290] + tri_ret[291:], 250, 100)  # one missing
    shifted = list(tri_ret)
    shifted[300] = shifted[300] + timedelta(days=1000)
    with pytest.raises(tri.TriAlignmentError, match="first difference at OOS index 80"):
        tri.align(pr, shifted, 250, 100)
    with pytest.raises(tri.TriAlignmentError, match="no PR OOS"):
        tri.align(pr, tri_ret, 400, 100)


def test_train_size_override_only_moves_the_first_target() -> None:
    cfg = load_config(PROJECT_ROOT / CONFIG)
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(48, 1))))
    r, _ = garch_returns(g.standard_t(6, 210) / np.sqrt(1.5), 0.02, 0.08, 0.90)
    x, dates, s = -0.01 * np.asarray(r), [f"d{i}" for i in range(210)], 30

    def at(t0: int) -> Config:
        return dataclasses.replace(
            cfg, design=dataclasses.replace(cfg.design, train_size=t0, window=W)
        )

    full, fs = rolling.run_rolling(x, dates, at(170), W, 4)
    cut, cs = rolling.run_rolling(x[s:], dates[s:], at(170 - s), W, 4)
    assert list(full.index.get_level_values("date")) == list(cut.index.get_level_values("date"))
    assert [j - s for j in full.index.get_level_values("j")] == list(
        cut.index.get_level_values("j")
    )
    a, b = full.reset_index(drop=True), cut.reset_index(drop=True)
    assert a.equals(b)
    assert (fs["first_target"], cs["first_target"]) == (170, 170 - s)
    assert {k: v for k, v in fs.items() if k not in ("first_target", "runtime_seconds")} == {
        k: v for k, v in cs.items() if k not in ("first_target", "runtime_seconds")
    }


# --- run, verify, publish -------------------------------------------------------------------
def _walk(x: Any) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            assert k != "date" and not k.endswith("_date"), k
            _walk(v)
    elif isinstance(x, list):
        assert len(x) < 100
        for v in x:
            _walk(v)


def test_run_verify_publish(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path, pr_backtest=True)
    assert _validate(cfg, tmp_path)[0] == 0  # writes the TRI manifest under the ignored data/
    with pytest.raises(oos.OosError, match="no complete TRI run dir"):
        tri.publish(cfg, CONFIG, tmp_path)
    status, logs = _run(cfg, tmp_path, workers=1)
    assert status == 0 and logs[-1]["message"] == "OK" and all(_nodate(r) for r in logs)
    n_tri = len(_bdays(tri.TRI_START, END)) - 1
    assert logs[-1]["t0_tri"] == n_tri - TARGETS and logs[-1]["n_targets"] == TARGETS
    rd = tmp_path / "reports" / "runs" / logs[-1]["run_dir"]
    assert sorted(p.name for p in rd.iterdir()) == sorted(oos.RUN_FILES)
    m = json.loads((rd / "manifest.json").read_text("utf-8"))
    assert m["tri"]["t0_tri"] == m["tri"]["train_size_override"] == n_tri - TARGETS
    assert len(m["data_files"]) == 4 and m["git"]["dirty"] is False
    assert _run(cfg, tmp_path)[0] == 1  # no overwrite
    before = {p.name: p.read_bytes() for p in rd.iterdir()}
    status, logs = _run(cfg, tmp_path, verify=True, workers=3)  # workers 1 vs 3: same bytes
    assert status == 0 and logs[-1]["message"] == "verify: all hashes match"
    assert {p.name: p.read_bytes() for p in rd.iterdir()} == before
    (rd / "summary.json").write_bytes(before["summary.json"] + b" ")
    assert _run(cfg, tmp_path, verify=True, workers=3)[0] == 1
    (rd / "summary.json").write_bytes(before["summary.json"])
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(oos.OosError, match="clean tree"):
        tri.publish(cfg, CONFIG, tmp_path)
    with pytest.raises(oos.OosError, match="clean tree"):
        _run(cfg, tmp_path)
    (tmp_path / "x.txt").unlink()
    outputs = tri.publish(cfg, CONFIG, tmp_path)
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in outputs)
    assert rel == sorted([f"reports/oos/tri_w1000/{n}" for n in oos.TRACKED] + ["docs/oos_tri.md"])
    for p in outputs:
        text = p.read_text(encoding="utf-8")
        assert not ISO_DATE.search(text), p.name
        if p.suffix == ".json":
            _walk(json.loads(text))
    tm = json.loads((tmp_path / tri.OUT_DIR / "manifest.json").read_text("utf-8"))
    assert "data_files" not in tm and tm["tri_manifest"]["file"] == "data/raw_tri/MANIFEST.toml"
    assert (tmp_path / tri.OUT_DIR / "backtest.json").read_bytes() == before["backtest.json"]
    doc = (tmp_path / tri.DOC).read_text(encoding="utf-8")
    assert doc == tri.render_doc(tmp_path) == tri.render_doc(tmp_path)
    assert "## TRI pass, W = 1000" in doc and "## PR vs TRI on the identical OOS days" in doc


def test_publish_refuses_another_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    assert _validate(cfg, tmp_path)[0] == 0
    commit = oos.clean_commit(tmp_path)
    rd = tri._where(cfg, tmp_path, tmp_path / CONFIG)[4]
    rd.mkdir(parents=True)
    for n in oos.RUN_FILES:
        (rd / n).write_text("{}", encoding="utf-8")
    out = tmp_path / tri.OUT_DIR
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"git": {"commit": "0" * 40}}), encoding="utf-8")
    monkeypatch.setattr(oos, "clean_commit", lambda _root: commit)
    with pytest.raises(oos.OosError, match="another commit"):
        tri.publish(cfg, CONFIG, tmp_path)


def test_stop_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)
    assert _validate(cfg, tmp_path)[0] == 0

    def broken(*args: Any, **kw: Any) -> Any:
        raise GarchError("injected")

    monkeypatch.setattr(models, "fit_garch", broken)
    status, logs = _run(cfg, tmp_path)
    assert status == 2 and logs[-1]["garch_error_share"] == 1.0
    rd = next((tmp_path / "reports" / "runs").iterdir())
    assert sorted(p.name for p in rd.iterdir()) == sorted(
        ["manifest.json", "forecasts.csv", "summary.json", "timing.json"]
    )
    status, logs = _run(cfg, tmp_path, verify=True)
    assert status == 2 and set(logs[-1]["hashes"]) == {"forecasts.csv", "summary.json"}


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    for cmd in ("validate-tri", "tri-run", "tri-publish"):
        assert _cli(cmd, "configs/missing.toml")[0] == 1
    status, log = _cli("tri-run", CONFIG)  # no TRI manifest yet
    assert status == 1 and "MANIFEST" in log["error"]
    assert _cli("validate-tri", CONFIG)[0] == 0
    status, log = _cli("tri-run", CONFIG, "--workers", "1")  # full design: T0 = 2000 > PR length
    assert status == 1 and "no PR OOS dates" in log["error"]
    status, log = _cli("tri-publish", CONFIG)
    assert status == 1 and "no complete TRI run dir" in log["error"]
    (tmp_path / tri.TRI_MANIFEST).write_text("broken", encoding="utf-8")
    status, log = _cli("tri-run", CONFIG)
    assert status == 1 and log["message"] == "data error"
    (tmp_path / "data" / "raw" / "MANIFEST.toml").write_text("broken", encoding="utf-8")
    status, log = _cli("validate-tri", CONFIG)
    assert status == 1 and log["message"] == "data error"


def test_edge_paths(tmp_path: Path) -> None:
    rows = tri_rows(_bdays(tri.TRI_START, date(1999, 7, 30)))
    p = tmp_path / tri_name(rows[0][0], rows[-1][0])
    write_tri(p, rows)
    text = p.read_bytes().decode("utf-8").split("\r\n")
    text[2] = '"NIFTY 50","x"'  # wrong field count
    text[3] = text[3].replace(fmt_date(sorted(rows, reverse=True)[2][0]), "2 Jul 99")  # bad date
    p.write_bytes("\r\n".join(text).encode("utf-8"))
    errs = tri._parse_file(p)[0].errors
    assert any("expected 4 fields" in e for e in errs) and any("dd Mon yyyy" in e for e in errs)
    man = tmp_path / "MANIFEST.toml"
    parsed, _, _ = tri.inspect(tmp_path)
    dl.write_manifest(
        man, [dl._info(x) for x in parsed], source_url="u", retrieved=date(2026, 1, 1)
    )
    assert "manifest series/field is not TRI/Total Returns Index" in tri._manifest_errors(
        man, tmp_path
    )
    with pytest.raises(dl.DataValidationError):
        tri.load_tri(tmp_path, man)  # series/field
    man.unlink()
    dl.write_manifest(
        man, [dl._info(x) for x in parsed], source_url="u", retrieved=date(2026, 1, 1),
        series="TRI", field_name="Total Returns Index",
    )  # fmt: skip
    with pytest.raises(dl.DataValidationError, match="expected 4 fields"):
        tri.load_tri(tmp_path, man)  # manifest fine, parse errors second
    extra = tmp_path / tri_name(date(1999, 7, 31), date(1999, 8, 31))
    shutil.copy(p, extra)
    p.unlink()
    errs = tri._manifest_errors(man, tmp_path)
    assert any("missing file" in e for e in errs) and any("not in manifest" in e for e in errs)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no CSV files found in the raw directory" in tri.inspect(empty)[1]
    cfg = load_config(PROJECT_ROOT / CONFIG)
    gone = tri.validate(tmp_path / "missing", man, cfg, [], None)
    assert not gone.report.ok and gone.differences == {}
    assert tri._spot_check(dl._collect([]), 10, 1) == ()
    assert tri._differences([date(2000, 1, 3)], [], None) == ({}, ())
    assert tri._cov_cells({"n": 0}, {"mean": None}) == ["—"] * 7
