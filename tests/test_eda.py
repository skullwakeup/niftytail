"""Tests for the Day-4 ``eda`` command and docs/eda.md (D29). Synthetic data only."""

from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from niftytail.__main__ import main
from niftytail.plots import FIGURE_NAMES
from test_data import make_manifest, make_rows, window_name, write_nse_csv

START, END = date(1995, 11, 3), date(2004, 12, 31)  # ~2,390 business days > T0 + 1
ALLOWED_KEYS = {
    "decision", "config_sha256", "manifest_sha256", "train_size", "training", "full_sample",
    "label", "n_returns", "first_return_date", "last_return_date",
    "descriptive_stats", "n", "mean", "std", "skewness", "excess_kurtosis", "jarque_bera",
    "stat", "p", "acf_band", "series", "r", "abs_r", "r2", "acf", "ljung_box", "q", "df",
    "arch_lm", "lm", "nobs", "exceedances", "u", "n_exceedances", "theta", "branch",
    "mean_cluster_size", "indicator_acf", "indicator_ljung_box",
    "5", "10", "20", "0.8", "0.9", "0.95",
}  # fmt: skip


@pytest.fixture
def eda_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_root: Path) -> Path:
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    write_nse_csv(raw / window_name(START, END), make_rows(START, END, 1000.0, 20261111))
    make_manifest(raw, raw / "MANIFEST.toml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _walk(node: Any, keys: set[str], lists: list[int]) -> None:
    if isinstance(node, dict):
        keys.update(node)
        for v in node.values():
            _walk(v, keys, lists)
    elif isinstance(node, list):
        lists.append(len(node))
        for v in node:
            _walk(v, keys, lists)


def test_eda_deterministic_whitelisted_and_short_lists(eda_env: Path) -> None:
    out = eda_env / "reports" / "eda"
    assert main(["eda", "configs/base.toml"]) == 0
    first = (out / "eda_summary.json").read_bytes()
    assert main(["eda", "configs/base.toml"]) == 0
    assert (out / "eda_summary.json").read_bytes() == first
    assert first.endswith(b"}\n") and b"\r" not in first
    summary = json.loads(first)
    keys: set[str] = set()
    lists: list[int] = []
    _walk(summary, keys, lists)
    assert keys <= ALLOWED_KEYS, sorted(keys - ALLOWED_KEYS)
    assert lists and max(lists) <= 20
    tr = summary["training"]
    assert (tr["n_returns"], tr["label"], summary["train_size"]) == (
        2000,
        "decision-relevant",
        2000,
    )
    assert summary["full_sample"]["label"] == "descriptive"
    for k, n_u in (("0.8", 400), ("0.9", 200), ("0.95", 100)):  # N_u = n - ceil(n q); ties lower
        assert 0 < tr["exceedances"][k]["n_exceedances"] <= n_u
    assert sorted(p.name for p in (out / "figures").iterdir()) == sorted(FIGURE_NAMES)


def test_eda_fails_without_manifest(eda_env: Path) -> None:
    (eda_env / "data" / "raw" / "MANIFEST.toml").unlink()
    assert main(["eda", "configs/base.toml"]) == 1
    assert not (eda_env / "reports" / "eda" / "eda_summary.json").exists()


def test_eda_bad_config(eda_env: Path) -> None:
    assert main(["eda", "configs/missing.toml"]) == 1


def test_eda_md_has_no_index_level_like_numbers(project_root: Path) -> None:
    text = (project_root / "docs" / "eda.md").read_text(encoding="utf-8")
    assert re.findall(r"\b\d{3,6}\.\d{2}\b", text) == []
