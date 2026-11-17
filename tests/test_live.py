"""Day-13D next-day forecast and live data (D58). Synthetic data only.

A temporary git repository holds a synthetic raw snapshot (1995-11-03 .. 1996-06-28) and,
where needed, a synthetic live file continuing it. Small design: W = 100, a 20-target health
block; the real config is otherwise unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import shutil
import subprocess
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from niftytail import data as dl
from niftytail import live, models, oos, rolling
from niftytail.__main__ import configure_logging, main
from niftytail.config import Config, config_sha256, load_config
from niftytail.threshold_study import garch_returns
from test_data import make_manifest, make_rows, window_name, write_nse_csv
from test_threshold_real import ISO_DATE

CONFIG = "configs/base.toml"
RAW_START, RAW_END = date(1995, 11, 3), date(1996, 6, 28)
LIVE_START, LIVE_END = date(1996, 6, 29), date(1996, 8, 30)
W, BLOCK = 100, 20
RETRIEVED = date(2026, 11, 15)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _small(cfg: Config, t0: int = 150) -> Config:
    design = dataclasses.replace(cfg.design, window=W, train_size=t0)
    bt = dataclasses.replace(cfg.backtest, traffic_light_block=BLOCK)
    return dataclasses.replace(cfg, design=design, backtest=bt)


def make_repo(root: Path) -> Config:
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    write_nse_csv(raw / window_name(RAW_START, RAW_END), make_rows(RAW_START, RAW_END, 1000, 7))
    make_manifest(raw, raw / "MANIFEST.toml")
    (root / ".gitignore").write_text("/data/\n/reports/\n", encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return _small(load_config(root / CONFIG))


def add_live(root: Path, start: date = LIVE_START, end: date = LIVE_END, **kw: Any) -> Path:
    d = root / live.LIVE_DIR
    d.mkdir(parents=True, exist_ok=True)
    rows = kw.get("rows") or make_rows(start, end, 1100.0, 8)
    p = d / window_name(start, end)
    write_nse_csv(p, rows)
    return p


def _logged(fn: Any, *args: Any) -> tuple[int, list[dict[str, Any]]]:
    buf = io.StringIO()
    status = fn(*args, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


def _nodate(logs: list[dict[str, Any]]) -> bool:
    return not any(
        ISO_DATE.search(json.dumps({k: v for k, v in r.items() if k != "time"})) for r in logs
    )


# --- the one code path (D58 (b)) ------------------------------------------------------------
def _bits(v: Any) -> Any:
    return v if isinstance(v, str) else np.float64(v).view(np.uint64).item()


def test_forecast_target_equals_run_rolling_rows() -> None:
    cfg = _small(load_config(PROJECT_ROOT / CONFIG), t0=170)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(58,))))
    r, _ = garch_returns(rng.standard_t(6, 180) / math.sqrt(1.5), 0.02, 0.08, 0.90)
    x = -0.01 * np.asarray(r)
    table, _ = rolling.run_rolling(x, [f"d{i}" for i in range(180)], cfg, W, 1)
    q, qc = cfg.threshold.q_selected, cfg.conditional.q_c
    assert q is not None
    full = (tuple(rolling._levels(q, cfg)), tuple(rolling._levels(qc, cfg)))
    for j in range(170, 180):
        row = table.loc[j].iloc[0]
        for qs, keys in ((full, None), (((q,), (qc,)), set(models.MODELS))):
            fc = rolling.forecast_target(x[:j], W, cfg, *qs)
            if keys is not None:
                assert set(fc) == keys
            for k, f in fc.items():
                assert ";".join(f.flags) == row[f"{k}:flags"]
                for c in f.var:
                    lvl = rolling._dec(c)
                    assert _bits(f.var[c]) == _bits(row[f"{k}:var:{lvl}"])
                    assert _bits(f.es[c]) == _bits(row[f"{k}:es:{lvl}"])


# --- live data (D58 (c)) --------------------------------------------------------------------
def test_raw_only_without_live_dir(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    df, sha = live.load_series(cfg, tmp_path)
    ref = dl.load_raw(tmp_path / "data/raw", tmp_path / "data/raw/MANIFEST.toml", RAW_START)
    assert sha is None and df.equals(ref)


def test_validate_live_writes_manifest_and_loads(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    add_live(tmp_path)
    with pytest.raises(dl.DataValidationError, match="manifest not found"):
        live.load_series(cfg, tmp_path)
    status, logs = _logged(live.validate_command, cfg, tmp_path, RETRIEVED)
    assert status == 0 and logs[-1]["manifest"] == "written" and _nodate(logs)
    assert set(logs[-1]["flag_counts"]) == set(dl.FLAG_CATEGORIES)
    man = dl.read_manifest(tmp_path / live.LIVE_MANIFEST)
    assert (man.retrieved, man.source_url) == (RETRIEVED, "https://example.invalid/nse")
    first = (tmp_path / live.LIVE_MANIFEST).read_bytes()
    status, logs = _logged(live.validate_command, cfg, tmp_path, date(2027, 1, 1))
    assert status == 0 and logs[-1]["manifest"] == "unchanged"
    assert (tmp_path / live.LIVE_MANIFEST).read_bytes() == first
    df, sha = live.load_series(cfg, tmp_path)
    assert df.index.is_monotonic_increasing and df.index[-1].date() <= LIVE_END
    assert df.index[0].date() == RAW_START and sha == live.ex.file_sha256(
        tmp_path / live.LIVE_MANIFEST
    )
    assert len(df) == len(make_rows(RAW_START, RAW_END, 1, 7)) + len(
        make_rows(LIVE_START, LIVE_END, 1, 8)
    )
    # a changed live file is refused until validate-live runs again
    p = add_live(tmp_path, end=date(1996, 9, 30))
    (tmp_path / live.LIVE_DIR / window_name(LIVE_START, LIVE_END)).unlink()
    with pytest.raises(dl.DataValidationError, match="not in manifest"):
        live.load_series(cfg, tmp_path)
    status, logs = _logged(live.validate_command, cfg, tmp_path, RETRIEVED)
    assert status == 0 and logs[-1]["manifest"] == "written"
    p.write_bytes(p.read_bytes() + b"\r\n")
    with pytest.raises(dl.DataValidationError, match="mismatch"):
        live.load_series(cfg, tmp_path)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (date(1996, 7, 1), LIVE_END, "non-contiguous"),
        (date(1996, 6, 20), LIVE_END, "overlapping"),
        (date(1995, 6, 1), date(1995, 11, 2), "does not start the day after"),
    ],
)
def test_validate_live_continuity(tmp_path: Path, start: date, end: date, message: str) -> None:
    cfg = make_repo(tmp_path)
    add_live(tmp_path, start, end)
    status, logs = _logged(live.validate_command, cfg, tmp_path, RETRIEVED)
    assert status == 1 and any(message in r.get("error", "") for r in logs)
    assert not (tmp_path / live.LIVE_MANIFEST).exists()


def test_validate_live_duplicate_date_and_empty_dir(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    (tmp_path / live.LIVE_DIR).mkdir(parents=True)
    status, logs = _logged(live.validate_command, cfg, tmp_path, RETRIEVED)
    assert status == 1 and any("no CSV files" in r.get("error", "") for r in logs)
    rows = make_rows(LIVE_START, LIVE_END, 1100.0, 8)
    add_live(tmp_path, rows=[*rows, rows[3]])
    status, logs = _logged(live.validate_command, cfg, tmp_path, RETRIEVED)
    assert status == 1 and any("duplicate dates" in r.get("error", "") for r in logs)
    assert not (tmp_path / live.LIVE_MANIFEST).exists()


@pytest.mark.parametrize(
    ("last_row", "window_end", "ok"),
    [
        (date(1996, 8, 30), date(1996, 9, 1), True),  # Friday row, window ends on Sunday
        (date(1996, 8, 28), LIVE_END, False),  # weekday (Thu, Fri) after the last row
        (date(1996, 8, 30), date(1996, 9, 2), False),  # trailing Monday (a holiday, say)
    ],
)
def test_live_trailing_weekday_guard(
    tmp_path: Path, last_row: date, window_end: date, ok: bool
) -> None:
    cfg = make_repo(tmp_path)
    add_live(tmp_path, end=window_end, rows=make_rows(LIVE_START, last_row, 1100.0, 8))
    status, logs = _logged(live.validate_command, cfg, tmp_path, RETRIEVED)
    assert status == (0 if ok else 1) and _nodate(logs)
    assert (tmp_path / live.LIVE_MANIFEST).exists() == ok
    if ok:
        assert live.load_series(cfg, tmp_path)[1] is not None
        return
    assert any(r.get("error") == live.TRAILING_WEEKDAY for r in logs)
    assert "1996" not in live.TRAILING_WEEKDAY and ".csv" not in live.TRAILING_WEEKDAY
    with pytest.raises(dl.DataValidationError) as exc:
        live.load_series(cfg, tmp_path)
    assert live.TRAILING_WEEKDAY in exc.value.errors


# --- forecast (D58 (d)) ---------------------------------------------------------------------
def _run(cfg: Config, root: Path, verify: bool = False, workers: int = 1) -> tuple[int, Any]:
    return _logged(live.run, cfg, CONFIG, workers, verify, root)


def _doc(root: Path) -> tuple[Path, dict[str, Any]]:
    (path,) = (root / live.LIVE_REPORTS).glob(f"*/{live.FORECAST_JSON}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_forecast_raw_only_document(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    status, logs = _run(cfg, tmp_path)
    assert status == 0 and _nodate(logs)
    path, doc = _doc(tmp_path)
    assert path.parent.name == logs[-1]["key"]
    df = dl.load_raw(tmp_path / "data/raw", tmp_path / "data/raw/MANIFEST.toml", RAW_START)
    assert doc["as_of"] == df.index[-1].date().isoformat()
    assert doc["provenance"]["live_manifest_sha256"] is None
    assert doc["provenance"]["n_losses"] == len(df) - 1
    assert sorted(doc["models"]) == sorted(models.MODELS)  # primary keys only, no "@q"
    for k, m in doc["models"].items():
        assert list(m["var"]) == ["0.95", "0.99", "0.995", "0.999"] == list(m["es"])
        assert ("mu_next" in m) == (k in live.CONDITIONAL) and ("xi" in m) == (k in live.GPD)
    vol = doc["volatility"]
    assert vol["sigma_ratio"] == vol["sigma_next"] / vol["window_std"]
    h = doc["health"]
    assert (h["level"], h["targets"]) == ("0.99", BLOCK)
    for v in h["models"].values():
        assert v["n_obs"] == BLOCK and v["zone"] in ("green", "yellow", "red")
    # the next-day forecast is the rolling forecast of target n
    x, _, _ = live._series(cfg, tmp_path)
    fc = rolling.forecast_target(x, W, cfg, (Fraction(88, 100),), (Fraction(88, 100),))
    for k in models.MODELS:
        for c, lvl in ((Fraction(99, 100), "0.99"), (Fraction(999, 1000), "0.999")):
            assert doc["models"][k]["var"][lvl] == live._num(fc[k].var[c])
            assert doc["models"][k]["es"][lvl] == live._num(fc[k].es[c])
    assert doc["models"]["garch_evt"]["xi"] == live._num(fc["garch_evt"].diagnostics["xi"])
    # schema v2 (D59 (b)): the tail block is copied from the diagnostics
    assert doc["schema_version"] == 2 and doc["config"]["xi_zero_tol"] == 1e-8
    for k, names in (("pot_gpd", ("u", "n_u")), ("garch_evt", ("u_z", "n_u"))):
        d = fc[k].diagnostics
        assert doc["models"][k]["tail"] == {n: d[n] for n in names}
    assert all("tail" not in doc["models"][k] for k in models.MODELS if k not in live.GPD)


def test_series_document(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    add_live(tmp_path)
    assert _logged(live.validate_command, cfg, tmp_path, RETRIEVED)[0] == 0
    assert _run(cfg, tmp_path)[0] == 0
    path, doc = _doc(tmp_path)
    ser = json.loads((path.parent / live.SERIES_JSON).read_text(encoding="utf-8"))
    df, _, x, dates, _, _ = live._loaded(cfg, tmp_path)
    n_raw = len(make_rows(RAW_START, RAW_END, 1, 7))
    assert [h["source"] for h in ser["history"]] == ["raw"] * n_raw + ["live"] * (len(df) - n_raw)
    assert ser["history"][-1]["date"] == doc["as_of"] == dates[-1]
    assert ser["history"][0]["close"] == float(df["close"].iloc[0])
    assert set(ser["history"][0]) == {"date", "open", "high", "low", "close", "source"}
    assert ser["window"] == {"first": dates[-W], "last": dates[-1]}
    assert [r["date"] for r in ser["recent"]] == dates[-BLOCK:]
    assert [r["loss"] for r in ser["recent"]] == x[-BLOCK:].tolist()
    for k, h in doc["health"]["models"].items():  # the same forecasts as health
        assert sum(r[k]["hit"] for r in ser["recent"]) == h["breaches"]
    tail = ser["tail_sample"]
    assert (
        tail == sorted(tail, reverse=True) and len(tail) == doc["models"]["pot_gpd"]["tail"]["n_u"]
    )
    assert min(tail) > doc["models"]["pot_gpd"]["tail"]["u"]
    worst = ser["worst"]
    assert [w["loss"] for w in worst] == sorted(x.tolist(), reverse=True)[:5]
    assert all(dates[x.tolist().index(w["loss"])] == w["date"] for w in worst)
    # deterministic: another run (2 workers) from scratch gives the same bytes
    blob = (path.parent / live.SERIES_JSON).read_bytes()
    shutil.rmtree(tmp_path / live.LIVE_REPORTS)
    assert _run(cfg, tmp_path, workers=2)[0] == 0
    assert (path.parent / live.SERIES_JSON).read_bytes() == blob


def test_forecast_refuses_verifies_and_is_deterministic(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path)
    add_live(tmp_path)
    assert _logged(live.validate_command, cfg, tmp_path, RETRIEVED)[0] == 0
    assert _run(cfg, tmp_path, workers=2)[0] == 0
    path, doc = _doc(tmp_path)
    assert doc["as_of"] <= LIVE_END.isoformat() and doc["provenance"]["live_manifest_sha256"]
    status, logs = _run(cfg, tmp_path)  # an existing forecast.json acts as --verify (D59 (b))
    assert status == 0 and logs[-1]["message"] == "unchanged"
    status, logs = _run(cfg, tmp_path, verify=True)
    assert status == 0 and logs[-1]["message"] == "verify: bytes match" and _nodate(logs)
    x, dates, _ = live._series(cfg, tmp_path)
    one = live.build(x, dates[-1], cfg, 1, doc["provenance"])
    three = live.build(x, dates[-1], cfg, 3, doc["provenance"])
    assert live.backtest.to_json(one).encode() == live.backtest.to_json(three).encode()
    assert live.backtest.to_json(one).encode() == path.read_bytes()
    series = path.parent / live.SERIES_JSON
    good = series.read_bytes()
    series.write_bytes(good.replace(b'"raw"', b'"live"', 1))
    assert _run(cfg, tmp_path, verify=True)[0] == 1
    status, logs = _run(cfg, tmp_path)
    assert status == 1 and logs[-1]["message"] == "verify: MISMATCH"
    assert series.read_bytes() != good  # nothing rewritten
    series.write_bytes(good)
    path.write_bytes(path.read_bytes().replace(b'"horizon": 1', b'"horizon": 2'))
    assert _run(cfg, tmp_path, verify=True)[0] == 1
    (tmp_path / "untracked.txt").write_text("x", encoding="utf-8")
    with pytest.raises(oos.OosError, match="clean tree"):
        _run(cfg, tmp_path)


def test_build_input_checks() -> None:
    cfg = _small(load_config(PROJECT_ROOT / CONFIG))
    with pytest.raises(ValueError, match="block"):
        live.build(np.zeros(W + BLOCK), "x", cfg, 1, {})
    unset = dataclasses.replace(cfg, threshold=dataclasses.replace(cfg.threshold, q_selected=None))
    with pytest.raises(ValueError, match="q_selected"):
        live._primary_qs(unset)


def test_num_encoding() -> None:
    assert live._num(1.5) == 1.5 and live._num(math.inf) == "inf" and live._num(math.nan) is None


def test_check_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_repo(tmp_path)
    x, dates, _ = live._series(cfg, tmp_path)
    run_cfg = _small(cfg, t0=x.size - 30)
    table, summary = rolling.run_rolling(x, dates, run_cfg, W, 1)
    rd = tmp_path / "reports" / "runs" / "r1"
    rolling.write_table(table, rd)
    (rd / oos.MANIFEST).write_text(
        json.dumps({"config": {"sha256": config_sha256(tmp_path / CONFIG)}}), encoding="utf-8"
    )
    (rd / oos.SUMMARY).write_text(json.dumps({"window": summary["window"]}), encoding="utf-8")
    add_live(tmp_path)  # the live data only extend the series; the run's targets are unchanged
    assert _logged(live.validate_command, cfg, tmp_path, RETRIEVED)[0] == 0
    capsys.readouterr()
    assert live.check_run(cfg, CONFIG, "reports/runs/r1", 2, tmp_path) == 0
    assert capsys.readouterr().out == "20/20 identical\n"
    (check,) = (tmp_path / live.LIVE_REPORTS).glob(f"*/{live.CHECK_JSON}")
    assert json.loads(check.read_text(encoding="utf-8")) == {
        "commit": oos.clean_commit(tmp_path),
        "identical": 20,
        "n": 20,
        "run": "r1",
    }
    assert _run(cfg, tmp_path)[0] == 0  # the same key as forecast; --verify ignores check.json
    assert (check.parent / live.FORECAST_JSON).is_file()
    assert _run(cfg, tmp_path, verify=True)[0] == 0
    table["loss"] += 1.0
    rolling.write_table(table, rd)
    assert live.check_run(cfg, CONFIG, "reports/runs/r1", 1, tmp_path) == 1
    assert capsys.readouterr().out == "0/20 identical\n"
    assert json.loads(check.read_text(encoding="utf-8"))["identical"] == 0
    (rd / oos.MANIFEST).write_text(json.dumps({"config": {"sha256": "0"}}), encoding="utf-8")
    with pytest.raises(oos.OosError, match="another config"):
        live.check_run(cfg, CONFIG, "reports/runs/r1", 1, tmp_path)


def test_forecast_dir_is_a_content_key() -> None:
    a = live.forecast_dir(Path("r"), "c", "m", None, "g")
    assert a.parent == Path("r") / live.LIVE_REPORTS and len(a.name) == 64
    assert a != live.forecast_dir(Path("r"), "c", "m", "l", "g")
    assert a != live.forecast_dir(Path("r"), "c", "m", None, "h")


# --- CLI ------------------------------------------------------------------------------------
def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, out.getvalue(), err.getvalue()


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    for cmd in ("validate-live", "forecast"):
        assert _cli(cmd, "configs/missing.toml")[0] == 1
    status, _, err = _cli("validate-live", CONFIG)  # no data/live directory
    assert status == 1 and "data/live" in err
    add_live(tmp_path)
    status, _, err = _cli("validate-live", CONFIG)
    assert status == 0 and "validate-live passed" in err
    status, _, err = _cli("forecast", CONFIG, "--workers", "1")  # W = 1000: series too short
    assert status == 1 and "forecast refused or failed" in err
    assert _nodate([json.loads(line) for line in err.strip().splitlines()])
    status, _, err = _cli("forecast", CONFIG, "--check-run", "reports/runs/none")
    assert status == 1 and "forecast refused or failed" in err
    (tmp_path / live.LIVE_MANIFEST).write_text("broken", encoding="utf-8")
    status, _, err = _cli("forecast", CONFIG, "--workers", "1")
    assert status == 1 and "data error" in err
    shutil.rmtree(tmp_path / "data" / "raw")
    assert _cli("validate-live", CONFIG)[0] == 1
