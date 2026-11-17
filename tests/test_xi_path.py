"""Day-12A rolling xi path (D47). Synthetic data only.

Small design n = 210, T0 = 150, W = 100 (60 targets; all models real). CLI-level tests run in a
temporary git repository with a synthetic raw file and an 11A-style forecasts.csv written by
rolling.run_rolling + write_table under the D44 (b) key for RUN_11A_COMMIT.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from conftest import PROJECT_ROOT
from niftytail import backtest, models, oos, rolling, xi_path
from niftytail import data as dl
from niftytail import experiment as ex
from niftytail import losses as ls
from niftytail.__main__ import configure_logging, main
from niftytail.config import Config, config_sha256, load_config
from niftytail.gpd import EstimationError
from niftytail.threshold_study import garch_returns
from test_data import make_manifest, make_rows, window_name, write_nse_csv
from test_threshold_real import ISO_DATE

N, T0, W = 210, 150, 100
CONFIG = "configs/base.toml"
START, END = date(1995, 11, 3), date(1996, 8, 23)  # ~210 business days


def _small(cfg: Config) -> Config:
    return dataclasses.replace(cfg, design=dataclasses.replace(cfg.design, train_size=T0, window=W))


@pytest.fixture(scope="module")
def cfg() -> Config:
    return _small(load_config(PROJECT_ROOT / CONFIG))


@pytest.fixture(scope="module")
def losses() -> np.ndarray[Any, np.dtype[np.float64]]:
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(47, 0))))
    r, _ = garch_returns(g.standard_t(6, N) / math.sqrt(1.5), 0.02, 0.08, 0.90)
    return -0.01 * np.asarray(r)


DATES = [f"day{i:05d}" for i in range(N)]


@pytest.fixture(scope="module")
def table(cfg: Config, losses: Any) -> pd.DataFrame:
    return rolling.run_rolling(losses, DATES, cfg, W, 4)[0]


@pytest.fixture(scope="module")
def path1(cfg: Config, losses: Any) -> tuple[list[int], list[dict[str, Any]]]:
    return xi_path.compute(losses, cfg, W, 1)


def test_forecasts_equal_run_rolling(
    cfg: Config, table: pd.DataFrame, path1: tuple[list[int], list[dict[str, Any]]]
) -> None:
    js, results = path1
    assert js == list(range(T0, N))
    cells = xi_path.check_forecasts(js, DATES[T0:], results, table, cfg)
    assert cells == (N - T0) * 2 * 4 * 2  # targets x keys x levels x (VaR, ES)


def test_window_and_lag_indexing(cfg: Config, losses: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, float, float | None]] = []
    real = models.conditional

    def spy(window: Any, pre: Any, c: Config, q: Any) -> Any:
        x = np.asarray(window)
        seen.append((x.size, float(x[0]), pre))
        return real(window, pre, c, q)

    monkeypatch.setattr(models, "conditional", spy)
    xi_path.compute(losses[: T0 + 3], cfg, W, 1)
    assert seen == [(W, float(losses[j - W]), float(losses[j - W - 1])) for j in range(T0, T0 + 3)]


def test_workers_1_and_3_identical(
    cfg: Config, losses: Any, path1: tuple[list[int], list[dict[str, Any]]]
) -> None:
    js3, r3 = xi_path.compute(losses, cfg, W, 3)
    js1, r1 = path1
    assert xi_path._csv(js1, DATES[T0:], r1) == xi_path._csv(js3, DATES[T0:], r3)
    s1 = {k: xi_path.key_summary([r[k] for r in r1], cfg) for k in xi_path.KEYS}
    s3 = {k: xi_path.key_summary([r[k] for r in r3], cfg) for k in xi_path.KEYS}
    assert backtest.to_json(s1) == backtest.to_json(s3)
    text = xi_path._csv(js1, DATES[T0:], r1).decode("utf-8")
    assert text.splitlines()[0].split(",")[:4] == ["j", "date", "pot_gpd:xi", "pot_gpd:beta"]
    assert "\r" not in text


@pytest.mark.parametrize(
    ("col", "row"), [("pot_gpd:var:0.99", 3), ("garch_evt:es:0.999", 10), ("pot_gpd:es:0.95", 59)]
)
def test_checks_fire_on_a_perturbed_table(
    cfg: Config,
    table: pd.DataFrame,
    path1: tuple[list[int], list[dict[str, Any]]],
    col: str,
    row: int,
) -> None:
    bad = table.copy()
    v = float(bad[col].iloc[row])
    bad.iloc[row, bad.columns.get_loc(col)] = 1.0 if math.isnan(v) else math.nextafter(v, math.inf)
    with pytest.raises(xi_path.XiPathMismatch, match=f"target {T0 + row}"):
        xi_path.check_forecasts(*path1[:1], DATES[T0:], path1[1], bad, cfg)


def test_check_fires_on_index_mismatch(
    cfg: Config, table: pd.DataFrame, path1: tuple[list[int], list[dict[str, Any]]]
) -> None:
    dates = list(DATES[T0:])
    dates[0] = "other"
    with pytest.raises(xi_path.XiPathMismatch, match="index"):
        xi_path.check_forecasts(path1[0], dates, path1[1], table, cfg)


def test_refit_mismatch_is_fatal(cfg: Config, losses: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    real = xi_path.fit_mle

    def off(*args: Any, **kw: Any) -> Any:
        f = real(*args, **kw)
        return dataclasses.replace(f, xi=f.xi + 1e-12)

    monkeypatch.setattr(xi_path, "fit_mle", off)
    with pytest.raises(xi_path.XiPathMismatch, match="refit"):
        xi_path.compute(losses[: T0 + 1], cfg, W, 1)


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("profile CI: no upper crossing by profile_xi_max", "no_upper_crossing"),
        (
            "profile CI: the estimate lies outside (-1 + tol, profile_xi_max)",
            "estimate_outside_range",
        ),
        ("something else", "estimation_error"),
    ],
)
def test_ci_failure_is_counted_not_raised(
    cfg: Config, losses: Any, monkeypatch: pytest.MonkeyPatch, message: str, reason: str
) -> None:
    def fail(*args: Any, **kw: Any) -> Any:
        raise EstimationError(message)

    monkeypatch.setattr(xi_path, "profile_ci_xi", fail)
    _, results = xi_path.compute(losses[: T0 + 2], cfg, W, 1)
    for r in results:
        for k in xi_path.KEYS:
            e = r[k]
            assert e["ci_reason"] == reason and math.isnan(e["ci_lo"]) and not math.isnan(e["xi"])
    s = xi_path.key_summary([r["pot_gpd"] for r in results], cfg)
    assert s["ci_failures"][reason] == 2 and s["n_with_ci"] == 0


def test_no_xi_entry() -> None:
    f = models.Forecast("garch_evt", {}, {}, ("garch_error",), {})
    e = xi_path._xi_entry(0, f, np.zeros(3), "u_z", load_config(PROJECT_ROOT / CONFIG))
    assert e["ci_reason"] == "no_xi" and e["n_u"] is None and math.isnan(e["xi"])


def _e(xi: float, lo: float, hi: float, reason: str | None) -> dict[str, Any]:
    return {"xi": xi, "ci_lo": lo, "ci_hi": hi, "ci_reason": reason, "lower_open": False}


def test_quantiles_and_shares_on_a_hand_built_path(cfg: Config) -> None:
    nan = math.nan
    entries = [
        _e(0.1, 0.05, 0.2, None),
        _e(0.2, -0.1, 0.3, None),
        _e(-0.3, -0.5, -0.1, None),
        _e(nan, nan, nan, "no_xi"),
        _e(0.5, nan, nan, "no_upper_crossing"),
    ]
    s = xi_path.key_summary(entries, cfg)
    assert s["xi_quantiles"] == {
        "0": -0.3, "0.05": -0.3, "0.25": -0.3, "0.5": 0.1, "0.75": 0.2, "0.95": 0.5, "1": 0.5,
    }  # fmt: skip
    assert s["shares_over_targets_with_ci"] == {
        "ci_lo_gt_0": 1 / 3,
        "ci_hi_lt_0": 1 / 3,
        "ci_contains_0": 1 / 3,
        "xi_lt_0": 1 / 3,
    }
    assert (s["n_targets"], s["n_with_xi"], s["n_with_ci"]) == (5, 4, 3)
    assert s["ci_failures"] == {
        "no_xi": 1,
        "no_upper_crossing": 1,
        "estimate_outside_range": 0,
        "estimation_error": 0,
    }
    assert s["blocks"] == [
        {"block": 0, "n_obs": 5, "partial": True, "xi_median": 0.1, "share_ci_lo_gt_0": 1 / 3}
    ]
    empty = xi_path.key_summary([_e(nan, nan, nan, "no_xi")], cfg)
    assert empty["xi_quantiles"]["0.5"] is None and empty["blocks"][0]["xi_median"] is None


@pytest.mark.parametrize(("n", "partial"), [(5716, 216), (5650, 150)])
def test_block_layout_is_the_traffic_light_layout(cfg: Config, n: int, partial: int) -> None:
    ours = xi_path.blocks(n, cfg)
    tl = backtest.traffic_light(np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool), cfg.backtest)
    assert [(b["block"], b["start_row"], b["end_row"], b["n_obs"]) for b in ours] == [
        (b["block"], b["start_row"], b["end_row"], b["n_obs"]) for b in tl
    ]
    assert sum(not b["partial"] for b in ours) == 22 and ours[-1] == ours[-1] | {
        "partial": True,
        "n_obs": partial,
    }


# --- command, verify, publish ---------------------------------------------------------------
def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def make_repo(root: Path) -> Config:
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    write_nse_csv(raw / window_name(START, END), make_rows(START, END, 1000.0, 20261111))
    make_manifest(raw, raw / "MANIFEST.toml")
    (root / "docs").mkdir()
    (root / "docs" / "plan_v2.2.md").write_text("plan\n", encoding="utf-8", newline="\n")
    sel = root / xi_path.TRAINING
    sel.parent.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / xi_path.TRAINING, sel)
    (root / ".gitignore").write_text("/data/\n/reports/runs/\n", encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return _small(load_config(root / CONFIG))


def _write_11a(cfg: Config, root: Path) -> Path:
    df = dl.load_raw(root / cfg.data.raw_dir, root / cfg.data.manifest, cfg.data.start_date)
    x = ls.log_losses(df["close"].to_numpy())
    dates = [d.date().isoformat() for d in ls.return_dates(pd.DatetimeIndex(df.index))]
    table, _ = rolling.run_rolling(x, dates, cfg, W, 4)
    man = ex.file_sha256(root / cfg.data.manifest)
    rd = oos.run_dir(root, config_sha256(root / CONFIG), man, W, xi_path.RUN_11A_COMMIT)
    return rolling.write_table(table, rd)


def _run(cfg: Config, root: Path, verify: bool = False, workers: int = 4) -> tuple[int, list[Any]]:
    buf = io.StringIO()
    status = xi_path.run(cfg, CONFIG, workers, verify, root, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


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
    cfg = make_repo(tmp_path)
    status, logs = _run(cfg, tmp_path)
    assert status == 1 and "no 11A run dir" in logs[-1]["message"]
    f11 = _write_11a(cfg, tmp_path)
    with pytest.raises(oos.OosError, match="no complete xi-path run dir"):
        xi_path.publish(cfg, CONFIG, tmp_path)
    status, logs = _run(cfg, tmp_path)
    assert status == 0 and logs[-1]["message"] == "OK"
    assert [r["key"] for r in logs if r["message"] == "key"] == list(xi_path.KEYS)
    assert all(
        not ISO_DATE.search(json.dumps({k: v for k, v in r.items() if k != "time"})) for r in logs
    )
    rd = tmp_path / "reports" / "runs" / logs[-1]["run_dir"]
    assert sorted(p.name for p in rd.iterdir()) == ["summary.json", "xi_path.csv"]
    assert _run(cfg, tmp_path)[0] == 1  # no overwrite
    before = {p.name: p.read_bytes() for p in rd.iterdir()}
    status, logs = _run(cfg, tmp_path, verify=True, workers=1)
    assert status == 0 and logs[-1]["message"] == "verify: all hashes match"
    assert {p.name: p.read_bytes() for p in rd.iterdir()} == before
    (rd / "xi_path.csv").write_bytes(before["xi_path.csv"] + b"x")
    assert _run(cfg, tmp_path, verify=True)[0] == 1
    (rd / "xi_path.csv").write_bytes(before["xi_path.csv"])
    good = f11.read_bytes()  # a perturbed 11A table stops the command (D47 (c))
    tab = rolling.read_table(f11)
    tab.iloc[0, tab.columns.get_loc("pot_gpd:var:0.99")] += 1e-9
    rolling.write_table(tab, f11.parent)
    with pytest.raises(xi_path.XiPathMismatch):
        _run(cfg, tmp_path, verify=True)
    f11.write_bytes(good)
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(oos.OosError, match="clean tree"):
        xi_path.publish(cfg, CONFIG, tmp_path)
    (tmp_path / "x.txt").unlink()
    outputs = xi_path.publish(cfg, CONFIG, tmp_path)
    tracked = tmp_path / xi_path.TRACKED
    assert outputs == [tracked, tmp_path / xi_path.DOC]
    assert tracked.read_bytes() == before["summary.json"]
    for p in outputs:
        assert not ISO_DATE.search(p.read_text(encoding="utf-8")), p.name
    s = json.loads(tracked.read_text(encoding="utf-8"))
    _walk(s)
    assert set(s["provenance"]) == {
        "commit",
        "config_sha256",
        "data_manifest_sha256",
        "plan_sha256",
        "run_11a_dir",
        "run_11a_forecasts_sha256",
    }
    assert s["provenance"]["run_11a_forecasts_sha256"] == ex.file_sha256(f11)
    ref = s["keys"]["pot_gpd"]["training_reference"]
    assert ref["q"] == "22/25" and ref["source_sha256"] == ex.file_sha256(
        PROJECT_ROOT / xi_path.TRAINING
    )
    assert "training_reference" not in s["keys"]["garch_evt"]
    doc = (tmp_path / xi_path.DOC).read_text(encoding="utf-8")
    assert doc == xi_path.render_doc(tmp_path) == xi_path.render_doc(tmp_path)
    assert xi_path.LABEL in doc
    other = dict(s, provenance=dict(s["provenance"], commit="0" * 40))
    tracked.write_text(json.dumps(other), encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "published")
    with pytest.raises(oos.OosError, match="no complete xi-path run dir"):
        xi_path.publish(cfg, CONFIG, tmp_path)


def test_publish_refuses_a_tracked_summary_from_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_repo(tmp_path)
    commit = oos.clean_commit(tmp_path)
    rd = xi_path._where(cfg, tmp_path, tmp_path / CONFIG)[3]
    rd.mkdir(parents=True)
    for n in ("xi_path.csv", "summary.json"):
        (rd / n).write_text("{}", encoding="utf-8")
    tracked = tmp_path / xi_path.TRACKED
    tracked.parent.mkdir(parents=True)
    tracked.write_text(json.dumps({"provenance": {"commit": "0" * 40}}), encoding="utf-8")
    monkeypatch.setattr(oos, "clean_commit", lambda _root: commit)
    with pytest.raises(oos.OosError, match="another commit"):
        xi_path.publish(cfg, CONFIG, tmp_path)


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def test_cli_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _cli("xi-path", "configs/missing.toml")[0] == 1
    assert _cli("xi-path-publish", "configs/missing.toml")[0] == 1
    status, log = _cli("xi-path", CONFIG, "--workers", "1")  # full design: no 11A run dir here
    assert status == 1 and "no 11A run dir" in log["message"]
    status, log = _cli("xi-path-publish", CONFIG)
    assert status == 1 and "no complete xi-path run dir" in log["error"]
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    status, log = _cli("xi-path", CONFIG)
    assert status == 1 and "clean tree" in log["error"]
