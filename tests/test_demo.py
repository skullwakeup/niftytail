"""Day-15A public demo (D62 (b)): the synthetic index and the demo-data command. No real data."""

from __future__ import annotations

import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

from conftest import PROJECT_ROOT
from niftytail import dashboard_data as dd
from niftytail import demo, live
from niftytail.__main__ import configure_logging, main
from niftytail.config import load_config

CONFIG = "configs/base.toml"


def test_synthetic_index_shape_and_determinism() -> None:
    cfg = load_config(PROJECT_ROOT / CONFIG)
    a, b = demo.synthetic_index(cfg), demo.synthetic_index(cfg)
    assert a.equals(b) and len(a) == demo.SESSIONS == 4500
    assert list(a.columns) == ["open", "high", "low", "close"]
    assert a.index.is_monotonic_increasing and a.index.is_unique
    assert all(d.weekday() < 5 for d in a.index)
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a > 0).all().all() and (a.round(2) == a).all().all()


def test_run_writes_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / CONFIG, tmp_path / CONFIG)
    cfg = load_config(tmp_path / CONFIG)
    buf = io.StringIO()
    log = configure_logging(stream=buf)
    assert demo.run(cfg, CONFIG, 8, True, tmp_path, log) == 1  # nothing to verify yet
    assert demo.run(cfg, CONFIG, 8, False, tmp_path, log) == 0
    assert demo.run(cfg, CONFIG, 3, True, tmp_path, log) == 0  # worker-count independent
    out = tmp_path / demo.DEMO_DIR
    assert sorted(p.name for p in out.iterdir()) == sorted(demo.FILES)
    doc = json.loads((out / live.FORECAST_JSON).read_text(encoding="utf-8"))
    ser = json.loads((out / live.SERIES_JSON).read_text(encoding="utf-8"))
    assert dd._valid(doc) and dd._valid_series(ser)
    assert doc["provenance"]["commit"] == "synthetic-demo"
    assert {h["source"] for h in ser["history"]} == {"demo"}
    assert dd.self_check(doc)["ok"]
    (out / live.SERIES_JSON).write_bytes(b"{}")
    assert demo.run(cfg, CONFIG, 8, True, tmp_path, log) == 1
    logs: list[dict[str, Any]] = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert [r["message"] for r in logs][-1] == "verify: MISMATCH"


def test_max_rel_diff_synthetic() -> None:
    """D66: floats within relative 1e-5 pass; beyond, or any structure or string change, fails."""
    a = {"m": {"var": [0.05, 1.0e-3], "n": 3, "ok": True, "s": "fhs"}, "z": 0.0}
    near = {"m": {"var": [0.05 * (1 + 1e-6), 1.0e-3], "n": 3, "ok": True, "s": "fhs"}, "z": 0.0}
    far = {"m": {"var": [0.05 * (1 + 1e-4), 1.0e-3], "n": 3, "ok": True, "s": "fhs"}, "z": 0.0}
    assert demo.max_rel_diff(a, a) == 0.0
    assert demo.max_rel_diff(a, near) == pytest.approx(1e-6, rel=1e-3)
    assert demo.max_rel_diff(a, near) <= demo.REL_TOL
    assert demo.max_rel_diff(a, far) > demo.REL_TOL
    for change in (
        {"m": {**a["m"], "s": "hs"}},  # string
        {"m": {**a["m"], "n": 4}},  # count
        {"m": {**a["m"], "ok": False}},  # flag
        {"m": {**a["m"], "n": 3.0}},  # int vs float
        {"m": {**a["m"], "var": [0.05]}},  # list length
        {"m": {**a["m"], "extra": 1}},  # keys
        {"z": None},  # float vs null
    ):
        assert demo.max_rel_diff(a, {**a, **change}) == math.inf, change
    assert demo.max_rel_diff({"z": 0.0}, {"z": 1e-13}) == pytest.approx(0.1)  # floor 1e-12


def test_max_rel_diff_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    """15C-9: the same infinity or nan vs nan match; any other non-finite pair is a mismatch."""
    inf, nan = math.inf, math.nan
    assert demo.max_rel_diff(inf, inf) == 0.0
    assert demo.max_rel_diff(-inf, inf) == inf
    assert demo.max_rel_diff(nan, nan) == 0.0
    assert demo.max_rel_diff(nan, 1.0) == inf and demo.max_rel_diff(1.0, nan) == inf
    assert demo.max_rel_diff([0.0, nan], [0.0, 1.0]) == inf  # max() must not hide it
    assert demo.max_rel_diff([nan, 0.0], [1.0, 0.0]) == inf
    monkeypatch.setattr(demo, "BYTES_ONLY", False)
    doc = b'{"v": [NaN, Infinity, -Infinity, 1.0]}'
    assert demo.compare(doc, b'{"v":[NaN,Infinity,-Infinity,1.0]}') == 0.0
    assert demo.compare(doc, b'{"v": [1.0, Infinity, -Infinity, 1.0]}') == inf
    assert demo.compare(doc, b'{"v": [NaN, -Infinity, -Infinity, 1.0]}') == inf


def test_compare_and_verify_off_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D66: bytes on Windows; elsewhere the parsed values within relative 1e-5."""
    old, near = b'{"v": 1.0}', b'{"v": 1.000001}'
    monkeypatch.setattr(demo, "BYTES_ONLY", True)
    assert demo.compare(old, old) == 0.0 and demo.compare(old, near) == math.inf
    monkeypatch.setattr(demo, "BYTES_ONLY", False)
    assert demo.compare(old, near) == pytest.approx(1e-6, rel=1e-3)
    assert demo.compare(old, b"not json") == math.inf
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / CONFIG, tmp_path / CONFIG)
    cfg = load_config(tmp_path / CONFIG)
    buf = io.StringIO()
    log = configure_logging(stream=buf)
    assert demo.run(cfg, CONFIG, 8, False, tmp_path, log) == 0
    path = tmp_path / demo.DEMO_DIR / live.FORECAST_JSON
    doc = json.loads(path.read_text(encoding="utf-8"))
    for scale in (1 + 1e-6, 1 + 1e-4):
        bad = json.loads(json.dumps(doc))
        bad["models"]["fhs"]["es"]["0.95"] *= scale
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert demo.run(cfg, CONFIG, 8, True, tmp_path, log) == (0 if scale < 1.00001 else 1)
    logs: list[dict[str, Any]] = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert logs[-2]["message"] == (
        "verify: values match within relative 1e-5 (not byte-identical; docs/reproducibility.md)"
    )
    assert logs[-1]["message"] == "verify: MISMATCH"
    assert float(logs[-1]["max_rel_diff"]) == pytest.approx(1e-4, rel=1e-3)


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["demo-data", "configs/missing.toml"]) == 1
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / CONFIG, tmp_path / CONFIG)
    (tmp_path / "app").write_text("not a directory", encoding="utf-8")
    assert main(["demo-data", CONFIG, "--workers", "4"]) == 1  # OSError: logged, exit 1
