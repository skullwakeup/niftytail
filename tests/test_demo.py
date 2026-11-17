"""Day-15A public demo (D62 (b)): the synthetic index and the demo-data command. No real data."""

from __future__ import annotations

import io
import json
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


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["demo-data", "configs/missing.toml"]) == 1
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / CONFIG, tmp_path / CONFIG)
    (tmp_path / "app").write_text("not a directory", encoding="utf-8")
    assert main(["demo-data", CONFIG, "--workers", "4"]) == 1  # OSError: logged, exit 1
