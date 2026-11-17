"""Day-11A OOS run command, provenance and publication (D44). Synthetic data only.

Each test works in a temporary git repository: a copy of configs/base.toml, a synthetic raw
file with its manifest under the ignored data/raw/, and one commit. The design is shrunk
with dataclasses.replace (T0 = 150, W = 100, sensitivity W = 150; about 110 targets).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from conftest import PROJECT_ROOT
from niftytail import models, oos
from niftytail.__main__ import configure_logging, main
from niftytail.config import Config, load_config
from niftytail.garch import GarchError
from test_data import make_manifest, make_rows, window_name, write_nse_csv
from test_threshold_real import ISO_DATE

START, END = date(1995, 11, 3), date(1996, 10, 31)  # ~260 business days
T0, W, W2 = 150, 100, 150
CONFIG = "configs/base.toml"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def make_repo(root: Path) -> Config:
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    write_nse_csv(raw / window_name(START, END), make_rows(START, END, 1000.0, 20261111))
    make_manifest(raw, raw / "MANIFEST.toml")
    (root / ".gitignore").write_text("/data/\n/reports/runs/\n", encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    cfg = load_config(root / CONFIG)
    design = dataclasses.replace(cfg.design, train_size=T0, window=W, sensitivity_windows=(W2,))
    return dataclasses.replace(cfg, design=design)


def _run(cfg: Config, root: Path, **kw: Any) -> tuple[int, dict[str, Any]]:
    buf = io.StringIO()
    log = configure_logging(stream=buf)
    args = {"window": W, "workers": 4, "verify": False} | kw
    status = oos.run(cfg, CONFIG, args["window"], args["workers"], args["verify"], root, log)
    return status, json.loads(buf.getvalue().strip().splitlines()[-1])


def _no_date(record: dict[str, Any]) -> bool:
    return not ISO_DATE.search(json.dumps({k: v for k, v in record.items() if k != "time"}))


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Config, int, dict[str, Any]]:
    root = tmp_path_factory.mktemp("oos")
    cfg = make_repo(root)
    status, log = _run(cfg, root)
    return root, cfg, status, log


def test_clean_tree_refusal_and_commit_in_key(tmp_path: Path) -> None:
    make_repo(tmp_path)
    c1 = oos.clean_commit(tmp_path)
    original = (tmp_path / CONFIG).read_bytes()
    (tmp_path / CONFIG).write_bytes(original + b"\n# edit\n")  # tracked change
    with pytest.raises(oos.OosError, match="clean tree"):
        oos.clean_commit(tmp_path)
    (tmp_path / CONFIG).write_bytes(original)
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")  # untracked, not ignored
    with pytest.raises(oos.OosError, match="clean tree"):
        oos.clean_commit(tmp_path)
    (tmp_path / "new.txt").unlink()
    (tmp_path / "data" / "raw" / "scratch.txt").write_text("x", encoding="utf-8")  # ignored: fine
    assert oos.clean_commit(tmp_path) == c1
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "second")
    c2 = oos.clean_commit(tmp_path)
    d1, d2 = (oos.run_dir(tmp_path, "c", "m", W, c) for c in (c1, c2))
    assert d1 != d2 and d1.parent == tmp_path / "reports" / "runs"
    assert d1.name == hashlib.sha256(f"c\nm\n{W}\n{c1}\n".encode()).hexdigest()
    bare = tmp_path / "bare"
    bare.mkdir()
    _git(bare, "init", "-q")
    with pytest.raises(oos.OosError, match="git commit"):
        oos.clean_commit(bare)


def test_run_writes_six_files_and_refuses_overwrite(
    repo: tuple[Path, Config, int, dict[str, Any]],
) -> None:
    root, cfg, status, log = repo
    assert status == 0 and log["message"] == "OK" and _no_date(log)
    assert log["garch_error"] == 0 and log["stop_for_owner"] is False
    rd = root / "reports" / "runs" / log["run_dir"]
    assert sorted(p.name for p in rd.iterdir()) == sorted(oos.RUN_FILES)
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert "workers" not in summary and "runtime_seconds" not in summary
    assert summary["window"] == W and summary["first_target"] == T0
    timing = json.loads((rd / "timing.json").read_text(encoding="utf-8"))
    assert set(timing) == {"workers", "rolling_seconds", "evaluate_seconds"}
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git"] == {"commit": oos.clean_commit(root), "dirty": False}
    files = {f["file"] for f in manifest["data_files"]}
    assert files == {"data/raw/MANIFEST.toml", f"data/raw/{window_name(START, END)}"}
    status2, log2 = _run(cfg, root)
    assert status2 == 1 and "refusing" in log2["message"]
    status3, log3 = _run(cfg, root, window=123)
    assert status3 == 1 and "window" in log3["message"]
    assert _git(root, "status", "--porcelain") == ""


def test_verify_matches_writes_nothing_and_detects_corruption(
    repo: tuple[Path, Config, int, dict[str, Any]],
) -> None:
    root, cfg, _, log = repo
    rd = root / "reports" / "runs" / log["run_dir"]
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in rd.iterdir()}
    status, vlog = _run(cfg, root, workers=2, verify=True)
    assert status == 0 and vlog["message"] == "verify: all hashes match"
    assert set(vlog["hashes"]) == {
        "forecasts.csv",
        "summary.json",
        "backtest.json",
        "backtest_per_date.json",
    }
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in rd.iterdir()} == before
    assert sorted(p.name for p in (root / "reports" / "runs").iterdir()) == [rd.name]
    good = (rd / "forecasts.csv").read_bytes()
    (rd / "forecasts.csv").write_bytes(good + b"x")
    try:
        status, vlog = _run(cfg, root, workers=2, verify=True)
        assert status == 1 and vlog["message"] == "verify: MISMATCH"
        h = vlog["hashes"]["forecasts.csv"]
        assert h["computed"] != h["on_disk"] and vlog["hashes"]["summary.json"]["on_disk"]
    finally:
        (rd / "forecasts.csv").write_bytes(good)


def test_stop_rule_writes_no_backtest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_repo(tmp_path)

    def broken(*args: Any, **kw: Any) -> Any:
        raise GarchError("injected")

    monkeypatch.setattr(models, "fit_garch", broken)
    status, log = _run(cfg, tmp_path, workers=1)
    assert status == 2 and set(log) - {"time", "level", "logger", "message"} == {
        "garch_error",
        "garch_error_share",
    }
    assert log["garch_error_share"] == 1.0
    rd = next((tmp_path / "reports" / "runs").iterdir())
    assert sorted(p.name for p in rd.iterdir()) == sorted(
        ["manifest.json", "forecasts.csv", "summary.json", "timing.json"]
    )
    assert json.loads((rd / "timing.json").read_text(encoding="utf-8"))["evaluate_seconds"] is None
    status, vlog = _run(cfg, tmp_path, workers=1, verify=True)
    assert status == 2 and set(vlog["hashes"]) == {"forecasts.csv", "summary.json"}
    with pytest.raises(oos.OosError, match="no complete run dir"):
        oos.publish(cfg, CONFIG, tmp_path)


def _walk(x: Any) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            assert k != "date" and not k.endswith("_date"), k
            _walk(v)
    elif isinstance(x, list):
        assert len(x) < 100
        for v in x:
            _walk(v)


def test_publish(
    repo: tuple[Path, Config, int, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg, _, log = repo
    with pytest.raises(oos.OosError, match=f"window {W2}"):
        oos.publish(cfg, CONFIG, root)
    status, _ = _run(cfg, root, window=W2)
    assert status == 0
    (root / "scratch.txt").write_text("x", encoding="utf-8")
    with pytest.raises(oos.OosError, match="clean tree"):
        oos.publish(cfg, CONFIG, root)
    (root / "scratch.txt").unlink()
    commit = oos.clean_commit(root)
    outputs = oos.publish(cfg, CONFIG, root)
    rel = sorted(p.relative_to(root).as_posix() for p in outputs)
    assert rel == sorted(
        [f"reports/oos/w{w}/{n}" for w in (W, W2) for n in oos.TRACKED] + ["docs/oos.md"]
    )
    rd = root / "reports" / "runs" / log["run_dir"]
    for p in outputs:
        text = p.read_text(encoding="utf-8")
        assert not ISO_DATE.search(text), p.name
        if p.suffix == ".json":
            _walk(json.loads(text))
    assert (root / "reports/oos/w100/backtest.json").read_bytes() == (
        rd / "backtest.json"
    ).read_bytes()
    m = json.loads((root / "reports/oos/w100/manifest.json").read_text(encoding="utf-8"))
    assert "data_files" not in m and m["git"]["commit"] == commit
    assert m["data_manifest"] == {
        "file": "data/raw/MANIFEST.toml",
        "sha256": hashlib.sha256((root / "data/raw/MANIFEST.toml").read_bytes()).hexdigest(),
    }
    doc = (root / "docs" / "oos.md").read_text(encoding="utf-8")
    assert doc == oos.render_doc(root) == oos.render_doc(root)
    assert commit in doc and "## W = 100" in doc and "## W = 150" in doc
    # D44 (h): a window already published from another commit is never overwritten
    monkeypatch.setattr(oos, "clean_commit", lambda _root: commit)  # the tree is now dirty
    mpath = root / "reports/oos/w150/manifest.json"
    other = json.loads(mpath.read_text(encoding="utf-8"))
    other["git"]["commit"] = "0" * 40
    mpath.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(oos.OosError, match="another commit"):
        oos.publish(cfg, CONFIG, root)


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def test_cli_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _cli("run", "configs/missing.toml", "--window", "1000")[0] == 1
    assert _cli("oos-publish", "configs/missing.toml")[0] == 1
    assert _cli("run", CONFIG, "--window", "999")[0] == 1  # not a configured window
    status, log = _cli("run", CONFIG, "--window", "1000", "--workers", "1")
    assert status == 1 and "T0 + 1" in log["error"]  # the synthetic series is too short
    status, log = _cli("oos-publish", CONFIG)
    assert status == 1 and "no complete run dir" in log["error"]
    man = tmp_path / "data" / "raw" / "MANIFEST.toml"
    man.write_text("broken", encoding="utf-8")  # ignored file: the tree stays clean
    status, log = _cli("run", CONFIG, "--window", "1000")
    assert status == 1 and log["message"] == "data error"
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    status, log = _cli("run", CONFIG, "--window", "1000")
    assert status == 1 and "clean tree" in log["error"]
