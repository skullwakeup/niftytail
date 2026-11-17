"""Day-11B post-hoc OOS-block threshold check (D45). Synthetic data only.

Small design: train_size (= block length) 500 and 19 GoF replications via dataclasses.replace;
everything else as configs/base.toml. CLI-level tests run in temporary git repositories.
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
from niftytail import oos, oos_blocks
from niftytail import threshold_real as tr
from niftytail import threshold_residuals as trr
from niftytail.__main__ import configure_logging, main
from niftytail.config import Config, load_config
from niftytail.garch import GarchError, fit_garch
from niftytail.threshold_study import garch_returns
from test_data import make_manifest, make_rows, window_name, write_nse_csv
from test_threshold_real import ISO_DATE

T0 = 500
CONFIG = "configs/base.toml"
START, END = date(1995, 11, 3), date(2002, 2, 28)


def _small(cfg: Config) -> Config:
    return dataclasses.replace(
        cfg,
        design=dataclasses.replace(cfg.design, train_size=T0),
        threshold=dataclasses.replace(cfg.threshold, gof_bootstrap_reps=19),
    )


@pytest.fixture(scope="module")
def cfg() -> Config:
    return _small(load_config(PROJECT_ROOT / CONFIG))


@pytest.fixture(scope="module")
def losses() -> np.ndarray[Any, np.dtype[np.float64]]:
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(45, 99))))
    r, _ = garch_returns(g.standard_t(6, T0 + 1100) / math.sqrt(1.5), 0.02, 0.08, 0.90)
    return -0.01 * np.asarray(r)


# --- layout, slices, seeds ------------------------------------------------------------------
def test_layout_current_data_and_edges() -> None:
    lay = oos_blocks.layout(5716, 2000)
    assert [b["offset"] for b in lay["blocks"]] == [0, 2000, 3716]
    assert lay["remainder"] == 1716 and lay["n_oos"] == 5716
    assert [b["overlapping"] for b in lay["blocks"]] == [False, False, True]
    assert lay["blocks"][2] | {} == {
        "block": "B3",
        "offset": 3716,
        "overlapping": True,
        "overlap": 284,
        "overlap_with": "B2",
    }
    six = oos_blocks.layout(6000, 2000)
    assert [b["offset"] for b in six["blocks"]] == [0, 2000, 4000] and six["remainder"] == 0
    assert not any(b["overlapping"] for b in six["blocks"])
    with pytest.raises(oos.OosError, match="1999"):
        oos_blocks.layout(1999, 2000)


def test_block_slices_and_ar_lag() -> None:
    x = np.arange(7716, dtype=np.float64)
    blk, pre = oos_blocks.block_slice(x, 2000, 3716, 2000)
    assert blk[0] == 5716 and blk[-1] == 7715 and blk.size == 2000 and pre == 5715
    blk, pre = oos_blocks.block_slice(x, 2000, 0, 2000)
    assert blk[0] == 2000 and pre == 1999  # the lag of B1 is the last training loss
    with pytest.raises(ValueError):
        oos_blocks.block_slice(x, 2000, 3717, 2000)


def test_spawn_keys_distinct_from_7c_and_8a() -> None:
    keys = [oos_blocks.spawn_key(s, b) for s in oos_blocks.SERIES for b in (1, 2, 3)]
    assert keys[0] == (45, 0, 1) and keys[3] == (45, 1, 1) and len(set(keys)) == 6
    seed = 20261111
    states = {tuple(np.random.SeedSequence(seed, spawn_key=k).generate_state(4)) for k in keys}
    assert tuple(np.random.SeedSequence(seed).generate_state(4)) not in states  # Day 7C
    assert tuple(trr.residual_seed(seed).generate_state(4)) not in states  # Day 8A
    assert len(states) == 6


# --- (c) checks -----------------------------------------------------------------------------
def _row(q: str, var: Any = 1.0, es: Any = 1.0, flags: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"q": str(Fraction(q)), "var99": var, "es99": es, "flags": list(flags)}


def _check(cfg: Config, q: str | None, rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    qb = None if q is None else Fraction(q)
    return oos_blocks.checks(rows, qb, kw.get("rf"), Fraction("0.88"), cfg.threshold)


def test_q_drift_gap_boundary(cfg: Config) -> None:
    c = _check(cfg, "0.85", [_row("0.85"), _row("0.88")])
    assert (c["q_drift"], c["q_drift_reason"], c["risk_material"]) == (False, None, False)
    c = _check(cfg, "0.84", [_row("0.84"), _row("0.88")])
    assert (c["q_drift"], c["q_drift_reason"]) == (True, "gap")
    assert c["q_ref"] == "22/25" and c["var99_rel"] == 0.0


def test_risk_material_tolerance_boundaries(cfg: Config) -> None:
    at = _check(cfg, "0.80", [_row("0.80", 21.0, 22.0), _row("0.88", 20.0, 20.0)])
    assert at["risk_material"] is False and at["risk_reason"] is None  # exactly 0.05 and 0.10
    assert (at["var99_rel"], at["es99_rel"]) == (0.05, 0.1)
    above = math.nextafter(21.0, math.inf)
    v = _check(cfg, "0.80", [_row("0.80", above, 22.0), _row("0.88", 20.0, 20.0)])
    assert (v["risk_material"], v["risk_reason"]) == (True, "tol_var99")
    e = _check(
        cfg, "0.80", [_row("0.80", 21.0, math.nextafter(22.0, math.inf)), _row("0.88", 20.0, 20.0)]
    )
    assert (e["risk_material"], e["risk_reason"]) == (True, "tol_es99")
    below = _check(cfg, "0.80", [_row("0.80", 19.0, 18.0), _row("0.88", 20.0, 20.0)])
    assert below["risk_material"] is False  # -0.05 and -0.10: absolute value, inclusive


@pytest.mark.parametrize(
    ("block_row", "ref_row"),
    [
        (_row("0.80", None), _row("0.88")),
        (_row("0.80", es=math.inf), _row("0.88")),
        (_row("0.80", flags=("es_infinite",)), _row("0.88")),
        (_row("0.80", flags=("fit_failed",)), _row("0.88")),
        (_row("0.80"), _row("0.88", es=None)),
        (_row("0.80"), _row("0.88", var=0.0)),
    ],
)
def test_risk_material_not_finite(
    cfg: Config, block_row: dict[str, Any], ref_row: dict[str, Any]
) -> None:
    c = _check(cfg, "0.80", [block_row, ref_row])
    assert (c["risk_material"], c["risk_reason"], c["var99_rel"]) == (True, "not_finite", None)


def test_rule_failure_and_equal_q(cfg: Config) -> None:
    c = _check(cfg, None, [], rf="no_stable_region")
    assert (c["q_drift"], c["q_drift_reason"]) == (True, "no_stable_region")
    assert c["risk_material"] is None and c["var99_rel"] is None and c["risk_reason"] is None
    same = _check(cfg, "0.88", [])  # no lookup needed
    assert (same["q_drift"], same["risk_material"], same["var99_rel"]) == (False, False, 0.0)


# --- entries and documents ------------------------------------------------------------------
@pytest.fixture(scope="module")
def doc1(cfg: Config, losses: Any) -> tuple[dict[str, Any], list[Any]]:
    return oos_blocks.build_document(losses, cfg, 1, {"commit": "x"})


def test_document_identical_for_workers_1_and_3(
    cfg: Config, losses: Any, doc1: tuple[dict[str, Any], list[Any]]
) -> None:
    doc3, _ = oos_blocks.build_document(losses, cfg, 3, {"commit": "x"})
    assert tr.summary_json(doc1[0]) == tr.summary_json(doc3)
    doc = doc1[0]
    assert [b["offset"] for b in doc["layout"]["blocks"]] == [0, 500, 600]
    assert (
        [e["block"] for e in doc["raw"]]
        == ["B1", "B2", "B3"]
        == [e["block"] for e in doc["residuals"]]
    )
    for e in doc["raw"] + doc["residuals"]:
        assert e["decision"] == "D45" and e["body_sha256"] == tr.body_sha256(e)
        assert set(e["check"]) == {"q_ref", *oos_blocks.CHECK_KEYS}
        assert "date" not in json.dumps(e["provenance"])
    assert doc["raw"][0]["check"]["q_ref"] == "22/25" and doc["residuals"][0]["descriptive_only"]
    assert doc["raw"][2]["label"].endswith(
        "block B3 (overlapping (400 returns with B2), descriptive)"
    )


def test_residual_fit_is_the_rolling_window_fit(
    cfg: Config, losses: Any, doc1: tuple[dict[str, Any], list[Any]]
) -> None:
    x, pre = oos_blocks.block_slice(losses, T0, 500, T0)
    fit = fit_garch(x, cfg.garch, cfg.conditional.filter, lag=pre)  # as models.conditional
    g = doc1[0]["residuals"][1]["garch"]
    assert g["nll"] == fit.nll and g["spec"] == "F4" and g["phi1"] == fit.phi1
    assert doc1[0]["residuals"][1]["garch_error"] is False


def test_garch_error_entry(cfg: Config, losses: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kw: Any) -> Any:
        raise GarchError("injected")

    monkeypatch.setattr(oos_blocks, "fit_garch", broken)
    doc, _ = oos_blocks.build_document(losses, cfg, 1, {})
    for e in doc["residuals"]:
        assert e["garch_error"] is True and "selection" not in e
        assert e["check"] == {"q_ref": "22/25", **dict.fromkeys(oos_blocks.CHECK_KEYS)}
    assert all("selection" in e for e in doc["raw"])


# --- command, verify, publish ---------------------------------------------------------------
def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def make_repo(root: Path, n_targets_delta: int = 0) -> Config:
    (root / "configs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / CONFIG, root / CONFIG)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    rows = make_rows(START, END, 1000.0, 20261111)
    write_nse_csv(raw / window_name(START, END), rows)
    make_manifest(raw, raw / "MANIFEST.toml")
    (root / "docs").mkdir()
    (root / "docs" / "plan_v2.2.md").write_text("plan\n", encoding="utf-8", newline="\n")
    w = root / "reports" / "oos" / "w1000"
    w.mkdir(parents=True)
    n_oos = len(rows) - 1 - T0
    (w / "summary.json").write_text(
        json.dumps({"n_targets": n_oos + n_targets_delta}), encoding="utf-8"
    )
    (root / ".gitignore").write_text("/data/\n/reports/runs/\n", encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return _small(load_config(root / CONFIG))


def _run(cfg: Config, root: Path, verify: bool = False) -> tuple[int, list[dict[str, Any]]]:
    buf = io.StringIO()
    status = oos_blocks.run(cfg, CONFIG, 3, verify, root, configure_logging(stream=buf))
    return status, [json.loads(line) for line in buf.getvalue().strip().splitlines()]


def _clean(record: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in record.items() if k != "time"})


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
    with pytest.raises(oos.OosError, match="no oos-blocks run dir"):
        oos_blocks.publish(cfg, CONFIG, tmp_path)
    status, logs = _run(cfg, tmp_path)
    assert status == 0 and logs[-1]["message"] == "OK"
    entries = [r for r in logs if r["message"] == "entry"]
    assert [(r["series"], r["block"]) for r in entries] == [
        (s, b) for s in oos_blocks.SERIES for b in ("B1", "B2", "B3")
    ]
    assert all(not ISO_DATE.search(_clean(r)) for r in logs)
    rd = tmp_path / "reports" / "runs" / logs[-1]["run_dir"]
    assert [p.name for p in rd.iterdir()] == [oos_blocks.OUT_NAME]
    assert _run(cfg, tmp_path)[0] == 1  # no overwrite
    before = (rd / oos_blocks.OUT_NAME).read_bytes()
    status, logs = _run(cfg, tmp_path, verify=True)
    assert status == 0 and logs[-1]["computed"] == logs[-1]["on_disk"]
    assert (rd / oos_blocks.OUT_NAME).read_bytes() == before and len(list(rd.iterdir())) == 1
    (rd / oos_blocks.OUT_NAME).write_bytes(before + b" ")
    assert _run(cfg, tmp_path, verify=True)[0] == 1
    (rd / oos_blocks.OUT_NAME).write_bytes(before)
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(oos.OosError, match="clean tree"):
        oos_blocks.run(cfg, CONFIG, 1, False, tmp_path, configure_logging(stream=io.StringIO()))
    with pytest.raises(oos.OosError, match="clean tree"):
        oos_blocks.publish(cfg, CONFIG, tmp_path)
    (tmp_path / "x.txt").unlink()
    outputs = oos_blocks.publish(cfg, CONFIG, tmp_path)
    tracked = tmp_path / oos_blocks.TRACKED_JSON
    assert outputs == [tracked, tmp_path / oos_blocks.DOC] and tracked.read_bytes() == before
    for p in outputs:
        assert not ISO_DATE.search(p.read_text(encoding="utf-8")), p.name
    _walk(json.loads(tracked.read_text(encoding="utf-8")))
    doc = (tmp_path / oos_blocks.DOC).read_text(encoding="utf-8")
    assert doc == oos_blocks.render_doc(tmp_path) == oos_blocks.render_doc(tmp_path)
    assert "DESCRIPTIVE ONLY; no decision may cite it" in doc and "POST-HOC (plan §5a (ii))" in doc
    p = json.loads(tracked.read_text(encoding="utf-8"))["provenance"]
    assert set(p) == {
        "commit",
        "config_sha256",
        "data_manifest_sha256",
        "plan_sha256",
        "oos_w1000_summary_sha256",
    }
    other = json.loads(tracked.read_text(encoding="utf-8"))  # D45 (11)
    other["provenance"]["commit"] = "0" * 40
    tracked.write_text(json.dumps(other), encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "published")
    with pytest.raises(oos.OosError, match="no oos-blocks run dir"):  # new commit: no run yet
        oos_blocks.publish(cfg, CONFIG, tmp_path)


def test_publish_refuses_a_tracked_file_from_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_repo(tmp_path)
    commit = oos.clean_commit(tmp_path)
    rd = oos_blocks._where(cfg, tmp_path, tmp_path / CONFIG)[3]
    rd.mkdir(parents=True)
    (rd / oos_blocks.OUT_NAME).write_text("{}", encoding="utf-8")
    tracked = tmp_path / oos_blocks.TRACKED_JSON
    tracked.parent.mkdir(parents=True)
    tracked.write_text(json.dumps({"provenance": {"commit": "0" * 40}}), encoding="utf-8")
    monkeypatch.setattr(oos, "clean_commit", lambda _root: commit)  # the tracked file is untracked
    with pytest.raises(oos.OosError, match="another commit"):
        oos_blocks.publish(cfg, CONFIG, tmp_path)


def test_n_oos_mismatch_refuses(tmp_path: Path) -> None:
    cfg = make_repo(tmp_path, n_targets_delta=1)
    with pytest.raises(oos.OosError, match="differs from w1000 n_targets"):
        _run(cfg, tmp_path)
    assert not (tmp_path / "reports" / "runs").exists()


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(list(argv))
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def test_cli_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _cli("oos-blocks", "configs/missing.toml")[0] == 1
    assert _cli("oos-blocks-publish", "configs/missing.toml")[0] == 1
    status, log = _cli("oos-blocks", CONFIG)  # full design: n_oos differs from the summary
    assert status == 1 and "n_targets" in log["error"]
    status, log = _cli("oos-blocks-publish", CONFIG)
    assert status == 1 and "no oos-blocks run dir" in log["error"]
    (tmp_path / "data" / "raw" / "MANIFEST.toml").write_text("broken", encoding="utf-8")
    status, log = _cli("oos-blocks", CONFIG)
    assert status == 1 and log["message"] == "data error"
