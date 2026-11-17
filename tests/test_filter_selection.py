"""Tests for the Day-8A-2 ``filter-select`` command (D39 (c), (e)). Synthetic data only.

The temporary tree is the Day-7C one (``test_threshold_real.make_env``): a synthetic raw file
with its manifest and the pinned artefacts.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from niftytail import filter_selection as fs
from niftytail import garch as ga
from niftytail import threshold_real as tr
from niftytail import threshold_residuals as trr
from niftytail.__main__ import main
from niftytail.config import load_config
from niftytail.data import load_raw
from niftytail.losses import log_returns
from test_threshold_real import ISO_DATE, PINNED, SHORT_END, _ignored, make_env, walk

COMMAND = "filter-select"
TOP_KEYS = {
    "body_sha256", "candidates", "decision", "design", "diagnostics", "label", "provenance",
    "winner",
}  # fmt: skip
CANDIDATE_KEYS = {
    "mean", "variance", "k", "nll", "bic", "mu", "phi1", "omega", "alpha", "gamma", "beta",
    "persistence", "start", "flags",
}  # fmt: skip
DIAGNOSTIC_KEYS = {
    "spec", "moments_z", "ljung_box", "arch_lm", "acf_z", "sign_bias",
    "ljung_box_rejections_at_1pct",
}  # fmt: skip
LOG_KEYS = {
    "time", "level", "logger", "message", "outputs", "seconds", "n_train", "winner", "bic",
}  # fmt: skip


def run_cli() -> tuple[int, dict[str, Any]]:
    """Run the command in the current directory: (exit status, last log record)."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main([COMMAND, "configs/base.toml"])
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


@dataclasses.dataclass(frozen=True)
class Run:
    root: Path
    status: int
    log: dict[str, Any]
    first: bytes
    second: bytes
    summary: dict[str, Any]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    """Two full runs of the real code path on synthetic data."""
    root = tmp_path_factory.mktemp("filter")
    make_env(root)
    out = root / fs.FILTER_DIR / fs.FILTER_NAME
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        status, log = run_cli()
        first = out.read_bytes()
        assert run_cli()[0] == status
        yield Run(root, status, log, first, out.read_bytes(), json.loads(first))


def training_losses(root: Path) -> np.ndarray:
    df = load_raw(root / "data" / "raw", root / "data" / "raw" / "MANIFEST.toml")
    return -log_returns(df["close"].to_numpy())[:2000]


def rule_winner(candidates: dict[str, Any]) -> str:
    """D39 (c), restated: lowest BIC, then smaller k, then lower index."""
    return min(candidates, key=lambda s: (candidates[s]["bic"], candidates[s]["k"], s))


def assert_schema(s: dict[str, Any]) -> None:
    assert set(s) == TOP_KEYS
    assert (s["decision"], s["label"]) == ("D39", fs.LABEL)
    assert s["design"]["n_likelihood_terms"] == s["provenance"]["n_train"] - 1 == 1999
    assert (s["design"]["acf_lags"], s["design"]["ljung_box_reject_level"]) == (5, 0.01)
    assert list(s["candidates"]) == ["F1", "F2", "F3", "F4"]
    for spec, c in s["candidates"].items():
        ar, gjr = ga.SPECS[spec]
        assert set(c) == CANDIDATE_KEYS
        assert c["k"] == ga.n_params(spec) == 4 + ar + gjr
        assert c["bic"] == 2 * c["nll"] + c["k"] * math.log(1999)
        assert (c["phi1"] is not None) == ar and (c["gamma"] is not None) == gjr
        assert (c["mean"], c["variance"]) == (
            "AR(1)" if ar else "constant",
            "GJR(1,1)" if gjr else "GARCH(1,1)",
        )
        assert c["omega"] > 0 and c["alpha"] > 0 and c["beta"] > 0 and c["persistence"] < 1
        assert c["persistence"] == c["alpha"] + (c["gamma"] or 0.0) / 2 + c["beta"]
        assert c["start"] in range(4 if ar else 2) and set(c["flags"]) <= {"near_boundary"}
    assert s["winner"] == rule_winner(s["candidates"])
    assert set(s["diagnostics"]) == {"winner", "F1"}
    assert (s["diagnostics"]["winner"]["spec"], s["diagnostics"]["F1"]["spec"]) == (
        s["winner"],
        "F1",
    )
    for d in s["diagnostics"].values():
        assert set(d) == DIAGNOSTIC_KEYS
        assert set(d["moments_z"]) == {"mean", "variance", "skewness", "excess_kurtosis"}
        assert set(d["ljung_box"]) == {"level", "squared"}
        for lb in d["ljung_box"].values():
            assert set(lb) == {"10", "20"}
            assert all(set(v) == {"q", "df", "p"} for v in lb.values())
        assert set(d["arch_lm"]) == {"5", "10"}
        assert len(d["acf_z"]) == 5 and all(abs(r) < 1 for r in d["acf_z"])
        sb = d["sign_bias"]
        assert set(sb) == {"sign", "negative_size", "positive_size", "joint", "nobs"}
        assert all(set(sb[k]) == {"t", "p"} for k in ("sign", "negative_size", "positive_size"))
        assert set(sb["joint"]) == {"lm", "df", "p"} and (sb["joint"]["df"], sb["nobs"]) == (
            3,
            1998,
        )
        assert d["ljung_box_rejections_at_1pct"] == [
            f"{kind}_{lag}"
            for kind in ("level", "squared")
            for lag in ("10", "20")
            if d["ljung_box"][kind][lag]["p"] < 0.01
        ]
    assert tr.body_sha256(s) == s["body_sha256"]


def assert_aggregates_only(text: str, s: dict[str, Any]) -> list[float]:
    """Exactly the first and last training dates; no list longer than the 5 ACF lags."""
    prov = s["provenance"]
    assert ISO_DATE.findall(text) == [
        prov["first_train_return_date"],
        prov["last_train_return_date"],
    ]
    floats: list[float] = []
    lists: list[int] = []
    walk(s, floats, lists)
    assert max(lists) == 5
    return floats


# --- the command on synthetic data ----------------------------------------------------------
def test_cli_exit_status_log_and_output(run: Run) -> None:
    assert run.status == 0 and set(run.log) == LOG_KEYS  # aggregates only
    assert run.log["outputs"] == [(fs.FILTER_DIR / fs.FILTER_NAME).as_posix()]
    assert (run.log["winner"], run.log["n_train"]) == (run.summary["winner"], 2000)
    assert run.log["bic"] == {k: c["bic"] for k, c in run.summary["candidates"].items()}
    assert [p.name for p in (run.root / fs.FILTER_DIR).iterdir()] == [fs.FILTER_NAME]
    assert not (run.root / trr.RESIDUAL_DIR).exists() and not (run.root / tr.REAL_DIR).exists()


def test_two_runs_are_byte_identical(run: Run) -> None:
    assert run.first == run.second
    assert b"\r" not in run.first and run.first.endswith(b"}\n")


def test_json_schema_rule_and_recomputation(run: Run) -> None:
    s = run.summary
    assert_schema(s)
    assert {k: s["provenance"][k] for k in PINNED} == PINNED
    losses = training_losses(run.root)
    cfg = load_config(run.root / "configs" / "base.toml")
    fits, winner = ga.select_filter(losses, cfg.garch)
    assert winner == s["winner"]
    for spec, f in fits.items():
        c = s["candidates"][spec]
        assert (c["nll"], c["mu"], c["omega"], c["alpha"], c["beta"]) == (
            f.nll,
            f.mu,
            f.omega,
            f.alpha,
            f.beta,
        )
        assert (c["phi1"] or 0.0, c["gamma"] or 0.0) == (f.phi1, f.gamma)
    eps, _, z = ga.garch_filter(losses[1:], fits[winner], float(losses[0]))
    block = fs.diagnostics_block(eps, z, cfg.dependence)
    assert s["diagnostics"]["winner"] == {"spec": winner, **block}


def test_json_holds_no_per_day_data(run: Run) -> None:
    found = set(assert_aggregates_only(run.first.decode("utf-8"), run.summary)) - {0.0}
    df = load_raw(run.root / "data" / "raw", run.root / "data" / "raw" / "MANIFEST.toml")
    r = log_returns(df["close"].to_numpy())
    losses = -r[:2000]
    cfg = load_config(run.root / "configs" / "base.toml")
    for f in ga.select_filter(losses, cfg.garch)[0].values():
        for series in ga.garch_filter(losses[1:], f, float(losses[0])):
            assert not found & set(series.tolist())
    for series in (r, -r, df["close"].to_numpy()):
        assert not found & set(series.tolist())


def test_cli_error_paths_write_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main([COMMAND, "configs/missing.toml"]) == 1

    def fail(*args: object, **kwargs: object) -> None:
        raise ga.GarchError("synthetic")

    with monkeypatch.context() as mp:
        mp.setattr(fs, "select_filter", fail)  # one failed candidate: nothing is chosen
        status, log = run_cli()
    assert status == 1 and log["message"] == f"{COMMAND} failed"
    (tmp_path / "data" / "raw" / "MANIFEST.toml").write_text("broken", encoding="utf-8")
    status, log = run_cli()
    assert status == 1 and log["message"] == "data error"
    assert not (tmp_path / fs.FILTER_DIR).exists()


def test_refuses_a_series_shorter_than_train_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_env(tmp_path, end=SHORT_END)
    monkeypatch.chdir(tmp_path)
    status, log = run_cli()
    assert status == 1 and "train_size = 2000" in log["error"]
    assert not (tmp_path / fs.FILTER_DIR).exists()


# --- blocks on synthetic series -------------------------------------------------------------
def test_diagnostics_block_lists_the_ljung_box_rejections_at_one_percent() -> None:
    dep = load_config(PROJECT_ROOT / "configs" / "base.toml").dependence
    g = np.random.default_rng(39)
    iid = g.standard_normal(2000)
    clean = fs.diagnostics_block(iid * 0.01, iid, dep)
    assert set(clean) == DIAGNOSTIC_KEYS - {"spec"}
    assert clean["ljung_box_rejections_at_1pct"] == []
    assert clean["moments_z"]["variance"] == pytest.approx(1.0, abs=0.1)
    ar = np.empty(2000)
    ar[0] = iid[0]
    for t in range(1, 2000):
        ar[t] = 0.5 * ar[t - 1] + iid[t]  # strong serial correlation in the level
    dirty = fs.diagnostics_block(ar * 0.01, ar, dep)
    assert dirty["ljung_box_rejections_at_1pct"][:2] == ["level_10", "level_20"]
    assert dirty["acf_z"][0] == pytest.approx(0.5, abs=0.06)


def test_candidate_block_reports_null_for_a_term_the_spec_lacks() -> None:
    f1 = ga.GarchFit(0.1, 0.2, 0.05, 0.9, 7.0, 1, ("near_boundary",))
    block = fs.candidate_block(f1, 1999)
    assert (block["phi1"], block["gamma"], block["k"]) == (None, None, 4)
    assert block["bic"] == 14.0 + 4 * math.log(1999) and block["flags"] == ["near_boundary"]
    f4 = ga.GarchFit(0.1, 0.2, 0.05, 0.8, 7.0, 3, (), phi1=0.1, gamma=0.1, spec="F4")
    block = fs.candidate_block(f4, 1999)
    assert (block["phi1"], block["gamma"], block["k"]) == (0.1, 0.1, 6)
    assert block["persistence"] == 0.05 + 0.1 / 2 + 0.8


# --- repository layout ----------------------------------------------------------------------
def test_committed_filter_selection_is_aggregate_only_and_intact() -> None:
    """The tracked filter_selection.json: schema, the rule, hashes, two dates, short lists."""
    path = PROJECT_ROOT / fs.FILTER_DIR / fs.FILTER_NAME
    assert not _ignored(path.relative_to(PROJECT_ROOT).as_posix())
    text = path.read_text(encoding="utf-8")
    s = json.loads(text)
    assert_schema(s)
    assert tr.summary_json(s) == text
    assert {k: s["provenance"][k] for k in PINNED} == PINNED
    assert_aggregates_only(text, s)
    assert [p.name for p in path.parent.iterdir()] == [fs.FILTER_NAME]
    real = json.loads((PROJECT_ROOT / tr.REAL_DIR / tr.SELECTION_NAME).read_text("utf-8"))
    for key in ("n_train", "first_train_return_date", "last_train_return_date"):
        assert s["provenance"][key] == real["provenance"][key]  # the same training slice
