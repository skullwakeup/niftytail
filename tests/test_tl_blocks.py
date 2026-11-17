"""Day-12B traffic-light block table (D51 (j)). Synthetic run dirs only.

Each test builds a temporary git repository holding a copy of base.toml, synthetic tracked
manifests and backtest.json files, and synthetic run dirs at the D44 (b) / D48 keys the command
derives from those manifests; the expected-hash constants are monkeypatched to the fixtures.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from niftytail import backtest, oos, tl_blocks
from niftytail import experiment as ex
from niftytail.__main__ import configure_logging, main
from niftytail.config import Config, load_config
from test_threshold_real import ISO_DATE

CONFIG = "configs/base.toml"
KEYS = ("garch_evt", "pot_gpd", "pot_gpd@0.9")
N_ROWS = 3 * 250 + 40  # three full blocks and a partial one


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _hits(seed: int) -> str:
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(51, seed))))
    u = g.random(N_ROWS)
    return "".join("." if x < 0.002 else ("1" if x < 0.03 else "0") for x in u)


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(PROJECT_ROOT / CONFIG)


def make_repo(root: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Write the synthetic tracked files and run dirs; commit; patch the expected hashes."""
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    cfg_sha = _sha((root / CONFIG).read_bytes())
    lvl = backtest._dec(cfg.backtest.traffic_light_level)
    rds: dict[str, Path] = {}
    bt_sha, per_sha, fc_sha = {}, {}, {}
    for n, (pid, _, window, tdir, commit, is_tri) in enumerate(tl_blocks.PASSES):
        hits = {k: _hits(10 * n + i) for i, k in enumerate(KEYS)}
        per: dict[str, Any] = {
            "hits": {lvl: hits},
            "traffic_light": {},
            "excluded": {},
            "window": window,
        }
        counts = {}
        for k, h in hits.items():
            hh = np.array([c == "1" for c in h], dtype=np.int64)
            ok = np.array([c != "." for c in h], dtype=np.bool_)
            blocks = backtest.traffic_light(hh, ok, cfg.backtest)
            per["traffic_light"][k] = [
                {**b, "start_date": f"d{b['start_row']}", "end_date": f"d{b['end_row']}"}
                for b in blocks
            ]
            counts[k] = backtest._zone_counts(blocks)
        bt = backtest.to_json({"traffic_light": counts}).encode()
        man: dict[str, Any] = {
            "git": {"commit": commit, "dirty": False},
            "config": {"sha256": cfg_sha},
        }
        if is_tri:
            man["tri"] = {"pr_manifest_sha256": "a" * 64, "tri_manifest_sha256": "b" * 64}
        else:
            man["data_manifest"] = {"sha256": "a" * 64}
        (root / tdir).mkdir(parents=True)
        (root / tdir / "backtest.json").write_bytes(bt)
        (root / tdir / "manifest.json").write_bytes(backtest.to_json(man).encode())
        rd = tl_blocks._run_dir(root, window, man, is_tri)
        rd.mkdir(parents=True)
        (rd / "manifest.json").write_bytes(backtest.to_json({**man, "data_files": []}).encode())
        (rd / "backtest.json").write_bytes(bt)
        per_bytes = backtest.to_json(per).encode()
        (rd / "backtest_per_date.json").write_bytes(per_bytes)
        (rd / "forecasts.csv").write_bytes(f"j,date\n{n}\n".encode())
        rds[pid] = rd
        bt_sha[pid], per_sha[pid] = _sha(bt), _sha(per_bytes)
        fc_sha[pid] = _sha((rd / "forecasts.csv").read_bytes())
    monkeypatch.setattr(tl_blocks, "BACKTEST_SHA256", bt_sha)
    monkeypatch.setattr(tl_blocks, "PER_DATE_SHA256", {"tri_w1000": per_sha["tri_w1000"]})
    monkeypatch.setattr(
        tl_blocks, "FORECASTS_SHA256", {p: fc_sha[p] for p in ("pr_w1000", "tri_w1000")}
    )
    (root / ".gitignore").write_text("/reports/runs/\n", encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return rds


def _run(cfg: Config, root: Path, verify: bool = False) -> tuple[int, list[dict[str, Any]]]:
    buf = io.StringIO()
    status = tl_blocks.run(cfg, CONFIG, verify, root, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


def _walk(x: Any) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            assert "date" not in k, k
            _walk(v)
    elif isinstance(x, list):
        assert len(x) < 100
        for v in x:
            _walk(v)


def test_run_reproduces_counts_and_verify(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    status, logs = _run(cfg, tmp_path)
    assert status == 0 and logs[-1]["message"] == "OK"
    out = tmp_path / tl_blocks.OUT
    text = out.read_text(encoding="utf-8")
    assert not ISO_DATE.search(text)
    doc = json.loads(text)
    _walk(doc)
    assert doc["decision"] == "D51" and doc["level"] == "0.99" and doc["block_size"] == 250
    assert doc["provenance"]["commit"] == ex.git_state(tmp_path).commit
    for pid, _, _, tdir, _, _ in tl_blocks.PASSES:
        p = doc["passes"][pid]
        tracked = json.loads((tmp_path / tdir / "backtest.json").read_bytes())
        for k in KEYS:
            e = p["keys"][k]
            assert len(e["blocks"]) == 3 and e["partial_block"]["n_obs"] == 40
            assert [b["block"] for b in e["blocks"]] == [0, 1, 2]
            assert set(e["blocks"][0]) == {
                "block",
                "n_obs",
                "breaches",
                "zone",
                "inadmissible_rows",
            }
            blocks = [*e["blocks"], {**e["partial_block"], "zone": None, "inadmissible_rows": 0}]
            assert backtest._zone_counts(blocks) == tracked["traffic_light"][k]
    status, logs = _run(cfg, tmp_path, verify=True)  # only the output differs from HEAD
    assert status == 0 and logs[-1]["message"] == "verify: hash matches"
    good = out.read_bytes()
    out.write_bytes(good + b" ")
    assert _run(cfg, tmp_path, verify=True)[0] == 1
    out.write_bytes(good)
    with pytest.raises(oos.OosError, match="clean tree"):
        _run(cfg, tmp_path)  # writing needs a clean tree
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(oos.OosError, match="clean tree"):
        _run(cfg, tmp_path, verify=True)
    (tmp_path / "x.txt").unlink()
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "published")
    assert _run(cfg, tmp_path)[0] == 1  # no overwrite


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        ("stored_block", "recomputed block 1 differs"),
        ("counts", "zone counts differ"),
        ("forecasts", "forecasts.csv"),
        ("run_backtest", "run-dir backtest.json"),
        ("run_commit", "run-dir manifest"),
        ("tracked_backtest", "tracked backtest.json"),
        ("tracked_commit", "tracked manifest"),
        ("missing_dir", "run dir missing"),
    ],
)
def test_checks_stop(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch, damage: str, match: str
) -> None:
    rds = make_repo(tmp_path, cfg, monkeypatch)
    rd = rds["pr_w1000"]
    tdir = tmp_path / "reports" / "oos" / "w1000"
    if damage == "stored_block":
        per = json.loads((rd / "backtest_per_date.json").read_bytes())
        per["traffic_light"]["pot_gpd"][1]["breaches"] += 1
        (rd / "backtest_per_date.json").write_bytes(backtest.to_json(per).encode())
    elif damage == "counts":  # tracked and run-dir backtest.json agree, but not with the blocks
        bt = json.loads((tdir / "backtest.json").read_bytes())
        bt["traffic_light"]["pot_gpd"]["zones"]["green"] += 1
        b = backtest.to_json(bt).encode()
        for d in (tdir, rd):
            (d / "backtest.json").write_bytes(b)
        monkeypatch.setitem(tl_blocks.BACKTEST_SHA256, "pr_w1000", _sha(b))
        _git(tmp_path, "commit", "-qam", "counts")
    elif damage == "forecasts":
        (rd / "forecasts.csv").write_bytes(b"x")
    elif damage == "run_backtest":
        (rd / "backtest.json").write_bytes(b"{}")
    elif damage == "run_commit":
        man = json.loads((rd / "manifest.json").read_bytes())
        man["git"]["commit"] = "0" * 40
        (rd / "manifest.json").write_bytes(backtest.to_json(man).encode())
    elif damage == "tracked_backtest":
        monkeypatch.setitem(tl_blocks.BACKTEST_SHA256, "pr_w1000", "0" * 64)
    elif damage == "tracked_commit":
        monkeypatch.setattr(
            tl_blocks,
            "PASSES",
            (("pr_w1000", "x", 1000, oos.OOS_DIR / "w1000", "0" * 40, False),),
        )
    else:
        shutil.rmtree(rd)
    with pytest.raises(tl_blocks.TlBlocksMismatch, match=match):
        _run(cfg, tmp_path)
    assert not (tmp_path / tl_blocks.OUT).exists()


def test_tri_per_date_hash_checked(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    monkeypatch.setitem(tl_blocks.PER_DATE_SHA256, "tri_w1000", "0" * 64)
    with pytest.raises(tl_blocks.TlBlocksMismatch, match="tri_w1000: backtest_per_date"):
        _run(cfg, tmp_path)


@contextlib.contextmanager
def audit_reads(root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[set[str]]:
    """Record every file under ``root`` opened for reading (D51 (12))."""
    seen: set[str] = set()
    real_open, real_path_open = builtins.open, Path.open

    def note(f: Any, mode: str) -> None:
        if isinstance(f, (str, Path)) and not any(c in mode for c in "wax"):
            p = Path(f).resolve()
            if p.is_relative_to(root.resolve()):
                seen.add(p.relative_to(root.resolve()).as_posix())

    def fake_open(f: Any, mode: str = "r", *a: Any, **kw: Any) -> Any:
        note(f, mode)
        return real_open(f, mode, *a, **kw)

    def fake_path_open(self: Path, mode: str = "r", *a: Any, **kw: Any) -> Any:
        note(self, mode)
        return real_path_open(self, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(io, "open", fake_open)
    monkeypatch.setattr(Path, "open", fake_path_open)
    yield seen


def test_opens_only_the_allowed_files(
    tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    rds = make_repo(tmp_path, cfg, monkeypatch)
    with audit_reads(tmp_path, monkeypatch) as seen:
        assert _run(cfg, tmp_path)[0] == 0
    runs = {pid: rd.relative_to(tmp_path).as_posix() for pid, rd in rds.items()}
    allowed = {CONFIG}
    for pid, _, _, tdir, _, _ in tl_blocks.PASSES:
        allowed |= {f"{tdir.as_posix()}/manifest.json", f"{tdir.as_posix()}/backtest.json"}
        allowed |= {
            f"{runs[pid]}/{n}" for n in ("manifest.json", "backtest.json", "backtest_per_date.json")
        }
    allowed |= {f"{runs[p]}/forecasts.csv" for p in ("pr_w1000", "tri_w1000")}
    assert seen == allowed
    assert not any(p.startswith("data/") for p in seen)


def test_constants_match_their_tracked_sources() -> None:
    protected = (PROJECT_ROOT / "docs/protected_hashes.md").read_text(encoding="utf-8")
    for pid, _, _, tdir, _, _ in tl_blocks.PASSES:
        assert (
            f"{(tdir / 'backtest.json').as_posix()} ({tl_blocks.BACKTEST_SHA256[pid]})" in protected
        )
    xi = json.loads((PROJECT_ROOT / "reports/oos/xi_path/summary.json").read_bytes())
    assert xi["provenance"]["run_11a_forecasts_sha256"] == tl_blocks.FORECASTS_SHA256["pr_w1000"]
    status = (PROJECT_ROOT / "docs/status.md").read_text(encoding="utf-8")
    assert f"backtest_per_date.json `{tl_blocks.PER_DATE_SHA256['tri_w1000']}`" in status
    assert f"forecasts.csv `{tl_blocks.FORECASTS_SHA256['tri_w1000']}`" in status
    labels = [p[1] for p in tl_blocks.PASSES]
    assert labels == ["PR W1000 (primary)", "PR W2000", "TRI W1000"]
    assert re.fullmatch(r"[0-9a-f]{40}", tl_blocks.RUN_11A_COMMIT)
    assert tl_blocks.RUN_11C_COMMIT.startswith("4c68343")
    assert tl_blocks.RUN_11A_COMMIT.startswith("04572a3")


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def test_cli(tmp_path: Path, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path, cfg, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert _cli("tl-blocks", "configs/missing.toml")[0] == 1
    status, log = _cli("tl-blocks", CONFIG)
    assert status == 0 and log["message"] == "OK"
    status, log = _cli("tl-blocks", CONFIG, "--verify")
    assert status == 0 and log["message"] == "verify: hash matches"
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    status, log = _cli("tl-blocks", CONFIG, "--verify")
    assert status == 1 and "clean tree" in log["error"]


def test_git_status_failure_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "_git", lambda *_a: None)
    with pytest.raises(oos.OosError, match="git status failed"):
        tl_blocks.changed_paths(tmp_path)
