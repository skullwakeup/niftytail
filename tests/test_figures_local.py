"""Day-12C local daily figures (D53). Synthetic run dirs and tracked files only.

A temporary git repository holds a copy of base.toml, synthetic tracked files (xi_path summary,
w1000 backtest.json, real/selection.json) and synthetic 11A / 12A run dirs; the expected-hash
constants are monkeypatched to the fixtures. Synthetic dates live only in the temporary repo.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from conftest import PROJECT_ROOT
from niftytail import backtest, oos, rolling, xi_path
from niftytail import figures as fg
from niftytail import figures_local as fl
from niftytail.__main__ import configure_logging, main
from niftytail.config import Config, config_sha256, load_config
from test_threshold_real import ISO_DATE

CONFIG = "configs/base.toml"
N = 600  # two full 250-row blocks and a partial one
RUN_11A = "r11a" + "0" * 60


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(PROJECT_ROOT / CONFIG)


def make_table() -> pd.DataFrame:
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(53, 0))))
    dates = [d.date().isoformat() for d in pd.bdate_range("2001-01-01", periods=N)]
    cols: dict[str, Any] = {"loss": 0.01 * g.standard_t(5, N)}
    for i, k in enumerate(fg.MODELS):
        cols[f"{k}:flags"] = [""] * N
        for lvl, z in (("0.95", 1.6), ("0.99", 2.3), ("0.995", 2.6), ("0.999", 3.1)):
            var = 0.01 * z * (1 + 0.05 * i) * (1 + 0.1 * g.random(N))
            cols[f"{k}:var:{lvl}"] = var
            cols[f"{k}:es:{lvl}"] = var * 1.2
    index = pd.MultiIndex.from_arrays([list(range(N)), dates], names=["j", "date"])
    return pd.DataFrame(cols, index=index)


def make_repo(root: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    cfg_sha = config_sha256(root / CONFIG)
    table = make_table()
    rd11 = root / oos.RUNS_DIR / RUN_11A
    rolling.write_table(table, rd11)
    man = {"git": {"commit": xi_path.RUN_11A_COMMIT, "dirty": False}, "config": {"sha256": cfg_sha}}
    (rd11 / "manifest.json").write_bytes(backtest.to_json(man).encode())
    loss = table["loss"].to_numpy()
    per: dict[str, Any] = {"hits": {"0.99": {}}, "traffic_light": {}}
    counts = {}
    for k in fg.MODELS:
        h = backtest.hits(loss, table[f"{k}:var:0.99"].to_numpy())
        per["hits"]["0.99"][k] = "".join(map(str, h))
        blocks = backtest.traffic_light(h, np.ones(N, dtype=np.bool_), cfg.backtest)
        dates = list(table.index.get_level_values("date"))
        per["traffic_light"][k] = [
            {**b, "start_date": dates[b["start_row"]], "end_date": dates[b["end_row"]]}
            for b in blocks
        ]
        counts[k] = backtest._zone_counts(blocks)
    (rd11 / "backtest_per_date.json").write_bytes(backtest.to_json(per).encode())
    means = {}
    for c in cfg.levels.primary:
        _, scores = fl.fz0_by_day(table, c)
        means[backtest._dec(c)] = {
            k: {"mean": float(np.mean(scores[k])), "n": N} for k in fg.MODELS
        }
    tracked = {
        fl.XI_SUMMARY: {
            "provenance": {
                "commit": fl.XI_RUN_COMMIT,
                "config_sha256": cfg_sha,
                "data_manifest_sha256": "a" * 64,
                "run_11a_dir": RUN_11A,
                "run_11a_forecasts_sha256": _sha((rd11 / "forecasts.csv").read_bytes()),
            }
        },
        fl.W1000_BACKTEST: {"fz0": means, "traffic_light": counts},
        fl.TRAINING: {"xi_comparison": {"q_star": {"xi_hat": 0.11, "xi_ci": [-0.01, 0.27]}}},
    }
    for rel, obj in tracked.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(backtest.to_json(obj).encode())
        monkeypatch.setitem(fg.EXPECTED_SHA256, rel, _sha((root / rel).read_bytes()))
    rdx = xi_path.run_dir(root, cfg_sha, "a" * 64, 1000, fl.XI_RUN_COMMIT)
    rdx.mkdir(parents=True)
    xi = pd.DataFrame({"j": range(N), "date": list(table.index.get_level_values("date"))})
    for k in fl.G2_KEYS:
        base = 0.1 * np.sin(np.arange(N) / 50.0)
        xi[f"{k}:xi"] = base
        xi[f"{k}:ci_lo"] = base - 0.1
        xi[f"{k}:ci_hi"] = base + 0.1
        xi[f"{k}:ci_reason"] = ""
    xi.to_csv(rdx / "xi_path.csv", index=False, lineterminator="\n")
    monkeypatch.setattr(fl, "XI_CSV_SHA256", _sha((rdx / "xi_path.csv").read_bytes()))
    (root / ".gitignore").write_text("/reports/runs/\n", encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return {"rd11": rd11, "rdx": rdx}


def _run(cfg: Config, root: Path, verify: bool = False) -> tuple[int, list[dict[str, Any]]]:
    buf = io.StringIO()
    status = fl.run(cfg, CONFIG, verify, root, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


def _out_dir(root: Path) -> Path:
    return fl.run_dir(root, config_sha256(root / CONFIG), oos.clean_commit(root))


def test_run_verify_no_overwrite(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    status, logs = _run(cfg, tmp_path)
    assert status == 0 and logs[-1]["message"] == "OK"
    assert logs[-1]["fz0_means_equal"] is True and logs[-1]["n_checked"] == 12
    for r in logs:  # D53 (e): no date in a log record
        assert not ISO_DATE.search(json.dumps({k: v for k, v in r.items() if k != "time"}))
    rd = _out_dir(tmp_path)
    names = sorted(p.relative_to(rd).as_posix() for p in rd.rglob("*") if p.is_file())
    assert names == sorted([f"figures/{s.file}" for s in fl.FIGURES] + ["index.md"])
    assert oos.clean_commit(tmp_path)  # the outputs are ignored: the tree stays clean
    index = (rd / "index.md").read_text(encoding="utf-8")
    assert not ISO_DATE.search(index) and "LOCAL ONLY" in index
    assert _run(cfg, tmp_path)[0] == 1  # no overwrite
    status, logs = _run(cfg, tmp_path, verify=True)
    assert status == 0 and logs[-1]["message"] == "verify: all hashes match"
    svg = rd / "figures/cum_fz0.svg"
    svg.write_bytes(svg.read_bytes() + b" ")
    assert _run(cfg, tmp_path, verify=True)[0] == 1


def test_renders_are_byte_identical(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    inp = fl.read_inputs(tmp_path, config_sha256(tmp_path / CONFIG))
    a = fl.render_all(inp, cfg, "c" * 40)
    b = fl.render_all(inp, cfg, "c" * 40)
    assert a == b and len(a) == 5
    for s in fl.FIGURES:
        text = a[f"figures/{s.file}"].decode("utf-8")
        assert f"<dc:title>{s.gid} {s.file}</dc:title>" in text
        assert "commit " + "c" * 40 in text


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        ("manifest_commit", "commit, clean flag or config"),
        ("manifest_dirty", "commit, clean flag or config"),
        ("forecasts", "forecasts.csv sha256"),
        ("xi_csv", "xi_path.csv sha256"),
        ("tracked_hash", "sha256 differs from docs/protected_hashes.md"),
        ("xi_commit", "provenance: config or commit"),
        ("per_date_blocks", "zone counts differ"),
        ("missing_column", "lacks a column"),
    ],
)
def test_identity_checks_stop(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch, damage: str, match: str
) -> None:
    dirs = make_repo(tmp_path, cfg, monkeypatch)
    rd11 = dirs["rd11"]
    if damage in ("manifest_commit", "manifest_dirty"):
        man = json.loads((rd11 / "manifest.json").read_bytes())
        if damage == "manifest_commit":
            man["git"]["commit"] = "0" * 40
        else:
            man["git"]["dirty"] = True
        (rd11 / "manifest.json").write_bytes(backtest.to_json(man).encode())
    elif damage == "forecasts":
        f = rd11 / "forecasts.csv"
        f.write_bytes(f.read_bytes() + b"\n")
    elif damage == "xi_csv":
        f = dirs["rdx"] / "xi_path.csv"
        f.write_bytes(f.read_bytes() + b"\n")
    elif damage == "tracked_hash":
        monkeypatch.setitem(fg.EXPECTED_SHA256, fl.TRAINING, "0" * 64)
    elif damage == "xi_commit":
        monkeypatch.setattr(fl, "XI_RUN_COMMIT", "0" * 40)
    elif damage == "per_date_blocks":
        per = json.loads((rd11 / "backtest_per_date.json").read_bytes())
        per["traffic_light"]["hs"][0]["zone"] = (
            "red" if per["traffic_light"]["hs"][0]["zone"] != "red" else "green"
        )
        (rd11 / "backtest_per_date.json").write_bytes(backtest.to_json(per).encode())
    else:
        f = dirs["rdx"] / "xi_path.csv"
        df = pd.read_csv(f).drop(columns=["garch_evt:ci_hi"])
        df.to_csv(f, index=False, lineterminator="\n")
        monkeypatch.setattr(fl, "XI_CSV_SHA256", _sha(f.read_bytes()))
    with pytest.raises(oos.OosError, match=match):
        _run(cfg, tmp_path)
    assert not _out_dir(tmp_path).exists()


def test_g3_mean_equality_stop(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    rel = fl.W1000_BACKTEST
    bt = json.loads((tmp_path / rel).read_bytes())
    m = bt["fz0"]["0.99"]["fhs"]["mean"]
    bt["fz0"]["0.99"]["fhs"]["mean"] = float(np.nextafter(m, np.inf))  # one ulp
    (tmp_path / rel).write_bytes(backtest.to_json(bt).encode())
    monkeypatch.setitem(fg.EXPECTED_SHA256, rel, _sha((tmp_path / rel).read_bytes()))
    _git(tmp_path, "commit", "-qam", "ulp")
    with pytest.raises(fl.MeanMismatch, match=r"fhs at 0.99"):
        _run(cfg, tmp_path)
    assert not _out_dir(tmp_path).exists()


def test_g4_block_date_ticks(cfg: Config) -> None:
    table = make_table()
    dates = list(table.index.get_level_values("date"))
    h = np.zeros(N, dtype=np.int64)
    h[::40] = 1
    blocks = backtest.traffic_light(h, np.ones(N, dtype=np.bool_), cfg.backtest)
    per = {
        "traffic_light": {
            k: [
                {**b, "start_date": dates[b["start_row"]], "end_date": dates[b["end_row"]]}
                for b in blocks
            ]
            for k in fg.MODELS
        }
    }
    fig = fl.g4_calendar(per)
    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert ticks == [f"{dates[b['start_row']]} to {dates[b['end_row']]}" for b in blocks]
    assert len(ticks) == 3 and blocks[-1]["zone"] is None
    rects = {p.get_label(): p for p in fig.axes[0].patches if not p.get_label().startswith("_")}
    assert rects["gaussian|2|None"].get_facecolor() == pytest.approx((1, 1, 1, 1))  # partial
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
        fg.display(m) for m in fg.MODELS
    ]


def test_check_ignore_guard(tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    (tmp_path / ".gitignore").write_text("", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "unignore")
    # the run dirs are now untracked files: commit them so the tree is clean
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "runs")
    with pytest.raises(oos.OosError, match="not gitignored"):
        _run(cfg, tmp_path)
    assert not _out_dir(tmp_path).exists()


def test_constants_match_status() -> None:
    status = (PROJECT_ROOT / "docs/status.md").read_text(encoding="utf-8")
    assert f"xi_path.csv sha256 is `{fl.XI_CSV_SHA256}`" in status
    assert f"from commit `{fl.XI_RUN_COMMIT}`" in status
    assert re.fullmatch(r"[0-9a-f]{64}", fl.XI_CSV_SHA256)


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def test_cli(tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert _cli("figures-local", "configs/missing.toml")[0] == 1
    status, log = _cli("figures-local", CONFIG)
    assert status == 0 and log["message"] == "OK"
    status, log = _cli("figures-local", CONFIG, "--verify")
    assert status == 0 and log["message"] == "verify: all hashes match"
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    status, log = _cli("figures-local", CONFIG)
    assert status == 1 and "clean tree" in log["error"]
