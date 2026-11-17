"""Tests for the Day-8A ``threshold-select-residuals`` command (D38 (9)). Synthetic data only.

The temporary tree is the Day-7C one (``test_threshold_real.make_env``): B = 19 GoF
replications, a synthetic raw file with its manifest and the pinned artefacts.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from niftytail import filter_selection as fs
from niftytail import threshold_real as tr
from niftytail import threshold_residuals as trr
from niftytail.__main__ import main
from niftytail.config import load_config
from niftytail.data import load_raw
from niftytail.garch import (
    GarchError,
    GarchFit,
    fit_garch,
    garch_filter,
    n_params,
    select_filter,
)
from niftytail.losses import log_returns
from niftytail.plots import FIGURE_NAMES as EDA_FIGURE_NAMES
from test_threshold_real import (
    CANDIDATE_KEYS,
    DEPENDENCE_KEYS,
    LOG_KEYS,
    PINNED,
    TOP_KEYS,
    B,
    _ignored,
    assert_aggregates_only,
    make_env,
    walk,
)

GARCH_KEYS = {
    "n_obs", "mu", "omega", "alpha", "beta", "persistence", "unconditional_sd", "nll",
    "start", "flags",
}  # fmt: skip
FILTER_KEYS = {"spec", "k", "phi1", "gamma", "n_residuals"}
COMMAND = "threshold-select-residuals"
DAY_8A_SHA256 = (
    "97249cef7ced190edae6d07cdf7e406cdea500ac7768805c204d0f9acb98ac44"  # docs/protected_hashes.md
)
DESCRIPTIVE_JSON = f"{trr.DESCRIPTIVE_NAME}.json"


def run_cli(*flags: str) -> tuple[int, dict[str, Any]]:
    """Run the command in the current directory: (exit status, last log record)."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main([COMMAND, "configs/base.toml", *flags])
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def tracked_bytes(root: Path) -> dict[str, bytes]:
    out = root / trr.RESIDUAL_DIR
    return {p.name: p.read_bytes() for p in sorted(out.iterdir()) if p.is_file()}


@dataclasses.dataclass(frozen=True)
class Run:
    root: Path
    status: int
    log: dict[str, Any]
    first: dict[str, bytes]
    second: dict[str, bytes]
    summary: dict[str, Any]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    """Two full runs of the real code path on synthetic data."""
    root = tmp_path_factory.mktemp("residuals")
    make_env(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        status, log = run_cli()
        first = tracked_bytes(root)
        status2, _ = run_cli()
        assert status2 == status
        second = tracked_bytes(root)
        yield Run(root, status, log, first, second, json.loads(first[tr.SELECTION_NAME]))


def assert_schema(s: dict[str, Any], descriptive: bool = False) -> None:
    """The Day-8A schema; ``descriptive``: the D39 (g) run adds its label and a filter block."""
    extra = {"descriptive_only", "filter"} if descriptive else set()
    assert set(s) == TOP_KEYS | {"garch", "residual_diagnostics"} | extra
    g = s["garch"]
    half_gamma = 0.0
    if descriptive:
        f = s["filter"]
        assert set(f) == FILTER_KEYS and s["descriptive_only"] is True
        assert (s["decision"], s["label"]) == ("D39", trr.DESCRIPTIVE_LABEL.format(f["spec"]))
        assert s["label"].startswith("DESCRIPTIVE ONLY")
        assert f["n_residuals"] == g["n_obs"] == s["provenance"]["n_train"] - 1 == 1999
        assert (
            f["k"] == n_params(f["spec"]) == 4 + (f["phi1"] is not None) + (f["gamma"] is not None)
        )
        half_gamma = (f["gamma"] or 0.0) / 2
    else:
        assert (s["decision"], s["label"]) == ("D38", trr.LABEL)
    assert s["design"]["seed_spawn_key"] == [1]
    assert set(g) == GARCH_KEYS
    assert g["omega"] > 0 and g["alpha"] > 0 and g["beta"] > 0 and g["persistence"] < 1
    assert g["persistence"] == g["alpha"] + half_gamma + g["beta"] and g["start"] in range(4)
    assert set(g["flags"]) <= {"near_boundary"}
    d = s["residual_diagnostics"]
    assert set(d) == {"moments_z", "residuals", "raw_losses"}
    assert set(d["moments_z"]) == {"mean", "variance", "skewness", "excess_kurtosis"}
    for block in (d["residuals"], d["raw_losses"]):
        assert set(block) == {"ljung_box", "arch_lm"}
        assert set(block["ljung_box"]) == {"level", "squared"}
        for lb in block["ljung_box"].values():
            assert set(lb) == {"10", "20"}
            assert all(set(v) == {"q", "df", "p"} for v in lb.values())
        assert set(block["arch_lm"]) == {"5", "10"}
        assert all(set(v) == {"lm", "df", "p", "nobs"} for v in block["arch_lm"].values())
    assert len(s["candidates"]) == len(s["dependence"]) == len(s["mrl"]) == 16
    assert all(set(r) == CANDIDATE_KEYS for r in s["candidates"])
    assert all(set(r) == DEPENDENCE_KEYS for r in s["dependence"])
    assert tr.body_sha256(s) == s["body_sha256"]


def test_seed_node_is_separate_from_day_7c() -> None:
    ss = trr.residual_seed(20261111)
    assert (ss.entropy, ss.spawn_key) == (20261111, (1,))
    assert np.random.SeedSequence(20261111).spawn_key == ()  # the 7C node


def test_cli_exit_status_log_and_outputs(run: Run) -> None:
    sel = run.summary["selection"]
    assert run.status == (0 if sel["rule_failure"] is None else 1)
    assert set(run.log) == LOG_KEYS | {"garch_flags"}  # aggregates only
    assert (run.log["q_star"], run.log["n_train"], run.log["n_candidates"]) == (
        sel["q_star"],
        2000,
        16,
    )
    assert run.log["garch_flags"] == run.summary["garch"]["flags"]
    assert set(run.first) == {tr.SELECTION_NAME, *tr.FIGURE_NAMES}
    private = run.root / trr.RESIDUAL_DIR / tr.PRIVATE_DIR
    expected = set(EDA_FIGURE_NAMES[3:]) if sel["q_star"] is not None else set()
    assert {p.name for p in private.iterdir()} == expected if expected else not private.exists()
    assert not (run.root / tr.REAL_DIR).exists()  # the 7C outputs are not touched
    assert tr.LABEL == "real training data (aggregates only)"  # restored after the figures


def test_two_runs_are_byte_identical(run: Run) -> None:
    assert run.first == run.second
    assert all(b"\r" not in data for data in run.first.values())
    assert all(run.first[n].startswith(b"<?xml") for n in tr.FIGURE_NAMES)


def test_json_key_set_and_blocks(run: Run) -> None:
    s = run.summary
    assert_schema(s)
    assert s["garch"]["n_obs"] == s["provenance"]["n_train"] == 2000
    assert {k: s["provenance"][k] for k in PINNED} == PINNED
    assert (s["design"]["gof_bootstrap_reps"], s["design"]["q_max"]) == (B, "9/10")
    assert s["residual_diagnostics"]["moments_z"]["variance"] == pytest.approx(1.0, abs=0.05)


def test_json_holds_no_per_day_data_and_matches_a_recomputation(run: Run) -> None:
    text = run.first[tr.SELECTION_NAME].decode("utf-8")
    assert_aggregates_only(text, run.summary)  # two dates; no list longer than the grid
    df = load_raw(run.root / "data" / "raw", run.root / "data" / "raw" / "MANIFEST.toml")
    r = log_returns(df["close"].to_numpy())
    losses = -r[:2000]
    fit = fit_garch(losses, load_config(run.root / "configs" / "base.toml").garch)
    eps, s2, z = garch_filter(losses, fit)
    g = run.summary["garch"]
    assert (g["mu"], g["omega"], g["alpha"], g["beta"]) == (
        fit.mu,
        fit.omega,
        fit.alpha,
        fit.beta,
    )
    floats: list[float] = []
    walk(run.summary, floats, [])
    found = {x for x in floats if x != 0.0}
    thresholds = {row["u"] for row in run.summary["candidates"]}  # order statistics of z (D29)
    assert found & set(z.tolist()) <= thresholds
    for series in (eps, s2, np.sqrt(s2), -r, r, df["close"].to_numpy()):
        assert not found & set(series.tolist())
    raw = run.summary["residual_diagnostics"]["raw_losses"]
    assert raw == trr.serial_tests(losses, (10, 20), (5, 10))


def test_rule_failure_writes_evidence_and_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_env(tmp_path, config_edit=('tol_xi = "0.10" ', 'tol_xi = "0.000000001" '))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("niftytail.threshold.window_convergence_share", lambda *a, **k: 1.0)
    status, log = run_cli()
    assert status == 1 and log["level"] == "ERROR"
    assert log["rule_failure"] in ("no_stable_region", "all_rejected")
    assert (log["q_star"], log["q_star_at_grid_floor"]) == (None, None)
    files = tracked_bytes(tmp_path)
    assert set(files) == {tr.SELECTION_NAME, *tr.FIGURE_NAMES}  # the evidence is written
    assert not (tmp_path / trr.RESIDUAL_DIR / tr.PRIVATE_DIR).exists()
    s = json.loads(files[tr.SELECTION_NAME])
    assert_schema(s)
    assert s["selection"]["q_star"] is None
    assert s["selection"]["rule_failure"] == log["rule_failure"]


def test_selected_filter_run_is_descriptive_and_leaves_day_8a_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D39 (g): the BIC-selected filter's 1999 residuals; its own file names; twice identical.

    The convergence column is stubbed for speed only.
    """
    make_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("niftytail.threshold.window_convergence_share", lambda *a, **k: 1.0)
    out = tmp_path / trr.RESIDUAL_DIR

    def written() -> dict[str, bytes]:
        files = [out / DESCRIPTIVE_JSON, *sorted((out / trr.DESCRIPTIVE_NAME).iterdir())]
        return {p.name: p.read_bytes() for p in files}

    status, log = run_cli("--selected-filter")
    first = written()
    assert run_cli("--selected-filter")[0] == status and written() == first
    assert set(first) == {DESCRIPTIVE_JSON, *tr.FIGURE_NAMES}
    assert [p.name for p in out.iterdir() if p.is_file()] == [DESCRIPTIVE_JSON]  # no selection.json
    assert all(b"\r" not in data for data in first.values())
    text = first[DESCRIPTIVE_JSON].decode("utf-8")
    s = json.loads(text)
    assert_schema(s, descriptive=True)
    assert_aggregates_only(text, s)
    sel = s["selection"]
    assert status == (0 if sel["rule_failure"] is None else 1)
    assert set(log) == LOG_KEYS | {"garch_flags"} and log["q_star"] == sel["q_star"]
    assert len(s["split_half"]["halves"]) == 2 and len(s["candidates"]) == 16
    private = out / tr.PRIVATE_DIR / trr.DESCRIPTIVE_NAME
    expected = set(EDA_FIGURE_NAMES[3:]) if sel["q_star"] is not None else set()
    assert {p.name for p in private.iterdir()} == expected if expected else not private.exists()
    assert tr.LABEL == "real training data (aggregates only)"  # restored after the figures

    df = load_raw(tmp_path / "data" / "raw", tmp_path / "data" / "raw" / "MANIFEST.toml")
    losses = -log_returns(df["close"].to_numpy())[:2000]
    fits, winner = select_filter(losses, load_config(tmp_path / "configs" / "base.toml").garch)
    fit = fits[winner]
    eps, s2, z = garch_filter(losses[1:], fit, float(losses[0]))
    assert s["filter"] == trr.filter_block(fit, 1999) and s["filter"]["spec"] == winner
    assert s["garch"] == trr.garch_block(fit, 1999)
    assert s["residual_diagnostics"]["moments_z"] == trr.moments(z)
    floats: list[float] = []
    walk(s, floats, [])
    found = {x for x in floats if x != 0.0}
    assert found & set(z.tolist()) <= {row["u"] for row in s["candidates"]}
    for series in (eps, s2, losses, df["close"].to_numpy()):
        assert not found & set(series.tolist())


def test_filter_block_reports_null_for_a_term_the_spec_lacks() -> None:
    f1 = GarchFit(0.1, 0.2, 0.05, 0.9, 7.0, 0, ())
    assert trr.filter_block(f1, 10) == {
        "spec": "F1",
        "k": 4,
        "phi1": None,
        "gamma": None,
        "n_residuals": 10,
    }
    f4 = GarchFit(0.1, 0.2, 0.05, 0.8, 7.0, 0, (), phi1=0.1, gamma=0.2, spec="F4")
    assert trr.filter_block(f4, 10) == {
        "spec": "F4",
        "k": 6,
        "phi1": 0.1,
        "gamma": 0.2,
        "n_residuals": 10,
    }
    assert trr.garch_block(f4, 10)["persistence"] == 0.05 + 0.2 / 2 + 0.8


def test_cli_error_paths_write_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main([COMMAND, "configs/missing.toml"]) == 1

    def fail(*args: object, **kwargs: object) -> None:
        raise GarchError("synthetic")

    with monkeypatch.context() as mp:
        mp.setattr(trr, "fit_garch", fail)
        status, log = run_cli()
    assert status == 1 and log["message"] == f"{COMMAND} failed"
    (tmp_path / "data" / "raw" / "MANIFEST.toml").write_text("broken", encoding="utf-8")
    status, log = run_cli()
    assert status == 1 and log["message"] == "data error"
    assert not (tmp_path / trr.RESIDUAL_DIR).exists()


def test_refuses_a_series_shorter_than_train_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_threshold_real import SHORT_END

    make_env(tmp_path, end=SHORT_END)
    monkeypatch.chdir(tmp_path)
    status, log = run_cli()
    assert status == 1 and "train_size = 2000" in log["error"]
    assert not (tmp_path / trr.RESIDUAL_DIR).exists()


def test_tracked_and_gitignored_output_split() -> None:
    base = trr.RESIDUAL_DIR.as_posix()
    for name in (tr.SELECTION_NAME, *tr.FIGURE_NAMES):
        assert not _ignored(f"{base}/{name}"), name
    for name in EDA_FIGURE_NAMES[3:]:
        assert _ignored(f"{base}/{tr.PRIVATE_DIR}/{name}"), name
        assert _ignored(f"{base}/{tr.PRIVATE_DIR}/{trr.DESCRIPTIVE_NAME}/{name}"), name
    assert not _ignored(f"{base}/{DESCRIPTIVE_JSON}")
    for name in tr.FIGURE_NAMES:
        assert not _ignored(f"{base}/{trr.DESCRIPTIVE_NAME}/{name}"), name


def test_committed_residual_evidence_is_aggregate_only_and_intact() -> None:
    """The tracked real_residuals/selection.json: schema, hashes, two dates, short lists."""
    path = PROJECT_ROOT / trr.RESIDUAL_DIR / tr.SELECTION_NAME
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DAY_8A_SHA256  # kept by D39
    text = path.read_text(encoding="utf-8")
    s = json.loads(text)
    assert_schema(s)
    assert tr.summary_json(s) == text
    assert {k: s["provenance"][k] for k in PINNED} == PINNED
    assert (s["provenance"]["n_train"], s["design"]["gof_bootstrap_reps"]) == (2000, 999)
    assert_aggregates_only(text, s)
    tracked = {p.name for p in path.parent.iterdir() if p.is_file()}
    assert tracked == {tr.SELECTION_NAME, DESCRIPTIVE_JSON, *tr.FIGURE_NAMES}
    real = json.loads((PROJECT_ROOT / tr.REAL_DIR / tr.SELECTION_NAME).read_text("utf-8"))
    for key in ("n_train", "first_train_return_date", "last_train_return_date"):
        assert s["provenance"][key] == real["provenance"][key]  # the same training slice


def test_committed_descriptive_evidence_matches_the_filter_selection() -> None:
    """The tracked selection_filter.json: descriptive label, schema, the D39 winner's fit."""
    path = PROJECT_ROOT / trr.RESIDUAL_DIR / DESCRIPTIVE_JSON
    text = path.read_text(encoding="utf-8")
    s = json.loads(text)
    assert_schema(s, descriptive=True)
    assert tr.summary_json(s) == text
    assert {k: s["provenance"][k] for k in PINNED} == PINNED
    assert (s["provenance"]["n_train"], s["design"]["gof_bootstrap_reps"]) == (2000, 999)
    assert_aggregates_only(text, s)
    figures = {p.name for p in (path.parent / trr.DESCRIPTIVE_NAME).iterdir()}
    assert figures == set(tr.FIGURE_NAMES)
    chosen = json.loads((PROJECT_ROOT / fs.FILTER_DIR / fs.FILTER_NAME).read_text("utf-8"))
    winner = chosen["candidates"][chosen["winner"]]
    assert s["filter"]["spec"] == chosen["winner"]
    assert (s["filter"]["phi1"], s["filter"]["gamma"]) == (winner["phi1"], winner["gamma"])
    for key in ("mu", "omega", "alpha", "beta", "nll", "start", "flags", "persistence"):
        assert s["garch"][key] == winner[key], key
    assert s["provenance"] == chosen["provenance"]  # the same slice and configuration
