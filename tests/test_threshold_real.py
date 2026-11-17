"""Tests for the Day-7C ``threshold-select`` command (D36). Synthetic data only.

The CLI tests run in a temporary tree: a copy of ``configs/base.toml`` with 19 GoF bootstrap
replications (speed; nothing else differs), a synthetic raw file with its manifest, and
copies of the three pinned artefacts whose sha256 the command records.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from niftytail import threshold_real as tr
from niftytail.__main__ import main
from niftytail.config import Config, load_config
from niftytail.data import load_raw
from niftytail.gpd import EstimationError
from niftytail.losses import log_returns
from niftytail.plots import FIGURE_NAMES as EDA_FIGURE_NAMES
from niftytail.threshold import ThresholdDecision, select_threshold
from test_data import make_manifest, make_rows, window_name, write_nse_csv
from test_repo_hygiene import FROZEN_PLAN_SHA256

START, END = date(1995, 11, 3), date(2004, 12, 31)  # ~2,390 business days > T0 + 1
SHORT_END = date(2000, 12, 29)  # ~1,345 business days < T0 + 1
B = 19
PINNED = {  # docs/protected_hashes.md
    "plan_sha256": FROZEN_PLAN_SHA256,
    "nmin_study_sha256": "5a8349a7773fcb38dc2fe264ccf6fa32d5bb5493df5c17a555c35aee6964fc97",
    "simstudy_metrics_sha256": "cfa424e0fa1049eca51fc98d37c00fa66aca285d699d4673ff38274b003212b5",
}
TOP_KEYS = {
    "body_sha256", "candidates", "decision", "dependence", "dependence_at_q_star", "design",
    "label", "mrl", "provenance", "selection", "split_half", "xi_comparison",
}  # fmt: skip
CANDIDATE_KEYS = {
    "q", "u", "n_u", "n_u_window", "xi_hat", "beta_hat", "beta_star", "se_xi", "xi_ci",
    "xi_ci_lower_open", "ad_stat", "ad_p", "xi_hat_scipy_gof", "fit_error",
    "convergence_share", "flags", "var95", "var95_ci", "es95", "es95_ci", "var99",
    "var99_ci", "es99", "es99_ci", "fs_stat", "rejected", "stable_fail",
}  # fmt: skip
DEPENDENCE_KEYS = {
    "q", "n_exceedances", "theta", "branch", "mean_cluster_size", "n_clusters", "run_length",
    "exceedances_per_cluster", "xi_all", "xi_all_ci", "declustered_status", "xi_declustered",
    "xi_declustered_ci", "xi_difference", "xi_all_in_declustered_ci",
    "xi_declustered_in_all_ci",
}  # fmt: skip
LOG_KEYS = {
    "time", "level", "logger", "message", "outputs", "seconds", "n_train", "n_candidates",
    "q_star", "q_fs", "k_hat", "rule_failure", "q_star_at_grid_floor", "split_half_flag",
}  # fmt: skip
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def make_env(root: Path, end: date = END, config_edit: tuple[str, str] | None = None) -> None:
    """Config (B = 19), synthetic raw data with manifest and the pinned artefacts in ``root``."""
    text = (PROJECT_ROOT / "configs" / "base.toml").read_text(encoding="utf-8")
    edits = [("gof_bootstrap_reps = 999 ", f"gof_bootstrap_reps = {B} ")]
    for old, new in edits + ([config_edit] if config_edit else []):
        assert text.count(old) == 1
        text = text.replace(old, new)
    (root / "configs").mkdir()
    (root / "configs" / "base.toml").write_text(text, encoding="utf-8", newline="\n")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    write_nse_csv(raw / window_name(START, end), make_rows(START, end, 1000.0, 20261111))
    make_manifest(raw, raw / "MANIFEST.toml")
    for path in tr.PINNED_FILES.values():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PROJECT_ROOT / path, root / path)


def run_cli() -> tuple[int, dict[str, Any]]:
    """Run the command in the current directory: (exit status, last log record)."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = main(["threshold-select", "configs/base.toml"])
    return status, json.loads(err.getvalue().strip().splitlines()[-1])


def tracked_bytes(root: Path) -> dict[str, bytes]:
    out = root / tr.REAL_DIR
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
    """Two full runs of the real code path on synthetic data (about 30 s each)."""
    root = tmp_path_factory.mktemp("select")
    make_env(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        status, log = run_cli()
        first = tracked_bytes(root)
        status2, _ = run_cli()
        assert status2 == status
        second = tracked_bytes(root)
        yield Run(root, status, log, first, second, json.loads(first[tr.SELECTION_NAME]))


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(PROJECT_ROOT / "configs" / "base.toml")


@pytest.fixture(scope="module")
def losses() -> np.ndarray:
    return np.random.default_rng(20261111).standard_t(4, 2000)


@pytest.fixture(scope="module")
def decision(cfg: Config, losses: np.ndarray) -> ThresholdDecision:
    return select_threshold(
        losses, cfg, ss=np.random.SeedSequence(1), gof_reps=B, convergence_share=False
    )


def walk(node: Any, floats: list[float], lists: list[int]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            walk(v, floats, lists)
    elif isinstance(node, list):
        lists.append(len(node))
        for v in node:
            walk(v, floats, lists)
    elif isinstance(node, float):
        floats.append(node)


def assert_aggregates_only(text: str, summary: dict[str, Any]) -> None:
    """Exactly the first and last training dates, and no list longer than the grid."""
    prov = summary["provenance"]
    assert ISO_DATE.findall(text) == [
        prov["first_train_return_date"],
        prov["last_train_return_date"],
    ]
    floats: list[float] = []
    lists: list[int] = []
    walk(summary, floats, lists)
    assert max(lists) == len(summary["design"]["grid"]) == 16
    assert prov["n_train"] not in lists


# --- the command on synthetic data ----------------------------------------------------------
def test_cli_exit_status_log_and_outputs(run: Run) -> None:
    sel = run.summary["selection"]
    assert run.status == (0 if sel["rule_failure"] is None else 1)
    assert set(run.log) == LOG_KEYS  # aggregates only: no per-day field can appear
    assert (run.log["q_star"], run.log["n_train"], run.log["n_candidates"]) == (
        sel["q_star"],
        2000,
        16,
    )
    assert set(run.first) == {tr.SELECTION_NAME, *tr.FIGURE_NAMES}
    private = run.root / tr.REAL_DIR / tr.PRIVATE_DIR
    expected = set(EDA_FIGURE_NAMES[3:]) if sel["q_star"] is not None else set()
    assert {p.name for p in private.iterdir()} == expected if expected else not private.exists()
    assert len(run.log["outputs"]) == 6 + len(expected)


def test_two_runs_are_byte_identical(run: Run) -> None:
    assert run.first == run.second
    for name, data in run.first.items():
        assert b"\r" not in data, name
    assert run.first[tr.SELECTION_NAME].endswith(b"}\n")
    assert all(run.first[n].startswith(b"<?xml") for n in tr.FIGURE_NAMES)


def test_json_key_set_and_selection_block(run: Run) -> None:
    s = run.summary
    assert set(s) == TOP_KEYS
    assert set(s["selection"]) == {
        "q_star", "q_fs", "k_hat", "rule_failure", "q_star_at_grid_floor", "flags",
    }  # fmt: skip
    assert set(s["xi_comparison"]) == {"q_star", "q_0.90", "q_0.95"}
    assert len(s["candidates"]) == len(s["dependence"]) == len(s["mrl"]) == 16
    for row in s["candidates"]:
        assert set(row) == CANDIDATE_KEYS
        assert isinstance(row["convergence_share"], float)
    for row in s["dependence"]:
        assert set(row) == DEPENDENCE_KEYS
    assert [r["q"] for r in s["candidates"]] == s["design"]["grid"]
    assert (s["design"]["gof_bootstrap_reps"], s["design"]["q_max"]) == (B, "9/10")
    q_star = s["selection"]["q_star"]
    assert s["selection"]["q_star_at_grid_floor"] == (None if q_star is None else q_star == "4/5")
    assert (s["xi_comparison"]["q_0.90"]["q"], s["xi_comparison"]["q_0.95"]["q"]) == (
        "9/10",
        "19/20",
    )
    assert len(s["split_half"]["halves"]) == 2
    assert s["split_half"]["flag"] == ("split_half_flag" in s["selection"]["flags"])


def test_json_has_provenance_and_a_valid_body_hash(run: Run) -> None:
    prov = run.summary["provenance"]
    assert prov["n_train"] == 2000
    assert {k: prov[k] for k in PINNED} == PINNED
    assert len(prov["config_sha256"]) == len(prov["manifest_sha256"]) == 64
    assert tr.body_sha256(run.summary) == run.summary["body_sha256"]
    tampered = {**run.summary, "decision": "D0"}
    assert tr.body_sha256(tampered) != run.summary["body_sha256"]


def test_json_holds_no_per_day_data(run: Run) -> None:
    text = run.first[tr.SELECTION_NAME].decode("utf-8")
    assert_aggregates_only(text, run.summary)
    df = load_raw(run.root / "data" / "raw", run.root / "data" / "raw" / "MANIFEST.toml")
    r = log_returns(df["close"].to_numpy())
    floats: list[float] = []
    walk(run.summary, floats, [])
    floats = [x for x in floats if x != 0.0]  # an unchanged synthetic close gives r = 0
    thresholds = {row["u"] for row in run.summary["candidates"]}  # order statistics (D29)
    assert set(floats) & set((-r).tolist()) <= thresholds  # training and later losses
    assert not set(floats) & set(r.tolist())
    assert not set(floats) & set(df["close"].tolist())
    first, last = df.index[1].date().isoformat(), df.index[2000].date().isoformat()
    prov = run.summary["provenance"]
    assert (prov["first_train_return_date"], prov["last_train_return_date"]) == (first, last)


def test_rule_failure_writes_evidence_and_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tolerance no window can meet gives a genuine "no_stable_region" (unless every
    candidate is rejected). The convergence column is stubbed for speed only."""
    make_env(tmp_path, config_edit=('tol_xi = "0.10" ', 'tol_xi = "0.000000001" '))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("niftytail.threshold.window_convergence_share", lambda *a, **k: 1.0)
    status, log = run_cli()
    assert status == 1 and log["level"] == "ERROR"
    assert log["rule_failure"] in ("no_stable_region", "all_rejected")
    assert (log["q_star"], log["q_star_at_grid_floor"]) == (None, None)
    files = tracked_bytes(tmp_path)
    assert set(files) == {tr.SELECTION_NAME, *tr.FIGURE_NAMES}  # the evidence is written
    assert not (tmp_path / tr.REAL_DIR / tr.PRIVATE_DIR).exists()  # nothing at q* to draw
    s = json.loads(files[tr.SELECTION_NAME])
    assert (s["selection"]["q_star"], s["selection"]["q_star_at_grid_floor"]) == (None, None)
    assert s["selection"]["rule_failure"] == log["rule_failure"]
    assert (s["xi_comparison"]["q_star"], s["dependence_at_q_star"]) == (None, None)
    assert s["xi_comparison"]["q_0.90"] is not None
    assert all(r["stable_fail"] is not None for r in s["candidates"])


def test_refuses_a_series_shorter_than_train_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_env(tmp_path, end=SHORT_END)
    monkeypatch.chdir(tmp_path)
    status, log = run_cli()
    assert status == 1 and log["message"] == "threshold-select failed"
    assert "train_size = 2000" in log["error"]
    assert not (tmp_path / tr.REAL_DIR).exists()


def test_cli_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["threshold-select", "configs/missing.toml"]) == 1
    (tmp_path / "data" / "raw" / "MANIFEST.toml").write_text("broken", encoding="utf-8")
    status, log = run_cli()
    assert status == 1 and log["message"] == "data error"
    (tmp_path / "docs" / "plan_v2.2.md").unlink()
    status, log = run_cli()
    assert status == 1 and log["message"] == "threshold-select failed"
    assert not (tmp_path / tr.REAL_DIR).exists()


# --- summary blocks on a synthetic decision -------------------------------------------------
def test_floor_flag_true_false_null(
    cfg: Config, losses: np.ndarray, decision: ThresholdDecision
) -> None:
    grid = cfg.threshold.grid
    assert tr.floor_flag(grid[0], grid) is True
    assert tr.floor_flag(grid[1], grid) is False
    assert tr.floor_flag(None, grid) is None
    for q_star, failure, expected in (
        (Fraction(4, 5), None, True),
        (Fraction(81, 100), None, False),
        (None, "no_stable_region", None),
    ):
        d = dataclasses.replace(decision, q_star=q_star, rule_failure=failure)
        s = tr.build_summary(losses, d, cfg, {})
        assert s["selection"]["q_star_at_grid_floor"] is expected
        entry = s["xi_comparison"]["q_star"]
        assert (entry is None) == (q_star is None)
        if q_star is not None:
            assert entry["q"] == str(q_star) == s["dependence_at_q_star"]["q"]
            row = next(r for r in s["candidates"] if r["q"] == str(q_star))
            assert (entry["xi_hat"], entry["xi_ci"]) == (row["xi_hat"], row["xi_ci"])


def test_summary_is_deterministic_and_aggregate_only(
    cfg: Config, losses: np.ndarray, decision: ThresholdDecision
) -> None:
    prov = {"first_train_return_date": "2000-01-03", "last_train_return_date": "2007-12-31"}
    s = tr.build_summary(losses, decision, cfg, {**prov, "n_train": 2000})
    text = tr.summary_json(s)
    assert text == tr.summary_json(
        tr.build_summary(losses, decision, cfg, {**prov, "n_train": 2000})
    )
    assert json.loads(text) == s  # nothing but JSON types; floats survive by repr
    assert_aggregates_only(text, s)
    floats: list[float] = []
    walk(s, floats, [])
    assert set(floats) & set(losses.tolist()) <= {r["u"] for r in s["candidates"]}


def test_dependence_table_and_mrl(
    cfg: Config, losses: np.ndarray, decision: ThresholdDecision
) -> None:
    rows = json.loads(decision.to_json())["candidates"]
    dep = tr.dependence_table(losses, rows, cfg)
    for d, r in zip(dep, rows, strict=True):
        assert (d["q"], d["xi_all"], d["xi_all_ci"]) == (r["q"], r["xi_hat"], r["xi_ci"])
        assert d["n_exceedances"] == r["n_u"]
        assert 1 <= d["n_clusters"] <= d["n_exceedances"]
        assert d["exceedances_per_cluster"] == d["n_exceedances"] / d["n_clusters"]
        assert d["mean_cluster_size"] == pytest.approx(1 / d["theta"])
        assert d["declustered_status"] == "estimated"  # i.i.d. t(4): far more than 30 clusters
        lo, hi = d["xi_declustered_ci"]
        assert lo < d["xi_declustered"] < hi
        assert d["xi_difference"] == d["xi_declustered"] - d["xi_all"]
        assert d["xi_all_in_declustered_ci"] == (lo <= d["xi_all"] <= hi)
        assert d["xi_declustered_in_all_ci"] == (
            r["xi_ci"][0] <= d["xi_declustered"] <= r["xi_ci"][1]
        )
    for m, r in zip(tr.mrl_table(losses, rows), rows, strict=True):
        y = losses[losses > r["u"]] - r["u"]
        assert (m["q"], m["u"], m["n_u"]) == (r["q"], r["u"], r["n_u"])
        assert m["mean_excess"] == pytest.approx(y.mean(), rel=1e-12)
        half = 1.96 * y.std(ddof=1) / np.sqrt(y.size)
        assert m["band"] == pytest.approx([y.mean() - half, y.mean() + half], rel=1e-12)


def test_declustered_fit_not_estimated_failed_and_without_ci(
    cfg: Config, losses: np.ndarray, decision: ThresholdDecision, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = json.loads(decision.to_json())["candidates"]
    empty = ("xi_declustered", "xi_declustered_ci", "xi_difference", "xi_all_in_declustered_ci")

    def fail(*args: object, **kwargs: object) -> None:
        raise EstimationError("synthetic")

    with monkeypatch.context() as mp:
        mp.setattr(tr, "profile_ci_xi", fail)
        no_ci = tr.dependence_table(losses, rows, cfg)
        assert all(d["declustered_status"] == "estimated" for d in no_ci)
        assert all(d["xi_declustered_ci"] is None for d in no_ci)
        assert all(d["xi_all_in_declustered_ci"] is None for d in no_ci)
        assert all(isinstance(d["xi_declustered_in_all_ci"], bool) for d in no_ci)
        mp.setattr(tr, "fit_mle", fail)
        failed = tr.dependence_table(losses, rows, cfg)
        assert all(d["declustered_status"] == "fit_failed" for d in failed)
        assert all(d[k] is None for d in failed for k in empty)
    monkeypatch.setattr(tr, "MIN_CLUSTER_MAXIMA", 10**6)
    cut = tr.dependence_table(losses, rows, cfg)
    assert all(d["declustered_status"] == "not estimated" for d in cut)
    assert all(d[k] is None for d in cut for k in empty)
    assert all(d["n_clusters"] >= 1 and d["theta"] > 0 for d in cut)  # still reported
    no_fit = [{**r, "xi_hat": None, "xi_ci": None} for r in rows]  # a failed candidate fit
    monkeypatch.setattr(tr, "MIN_CLUSTER_MAXIMA", 30)
    for d in tr.dependence_table(losses, no_fit, cfg):
        assert d["declustered_status"] == "estimated" and d["xi_difference"] is None
        assert d["xi_declustered_in_all_ci"] is None and d["xi_all_in_declustered_ci"] is None


def test_figures_are_deterministic_with_and_without_q_star(
    cfg: Config, losses: np.ndarray, decision: ThresholdDecision, tmp_path: Path
) -> None:
    rejected = dataclasses.replace(
        decision, q_star=None, q_fs=None, k_hat=16, rule_failure="all_rejected"
    )
    for name, d in (("a", decision), ("b", decision), ("none", rejected)):
        summary = tr.build_summary(losses, d, cfg, {})
        for row in summary["candidates"]:
            row["rejected"] = d.k_hat == 16  # the rejected markers of the "all_rejected" case
        paths = tr.write_figures(summary, tmp_path / name)
        assert [p.name for p in paths] == list(tr.FIGURE_NAMES)
    for fig in tr.FIGURE_NAMES:
        data = (tmp_path / "a" / fig).read_bytes()
        assert data == (tmp_path / "b" / fig).read_bytes()
        assert b"\r" not in data and b"<dc:date>" not in data


# --- repository layout ----------------------------------------------------------------------
def _ignored(path: str) -> bool:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=PROJECT_ROOT, check=False
    )
    return res.returncode == 0


def test_tracked_and_gitignored_output_split() -> None:
    base = tr.REAL_DIR.as_posix()
    for name in (tr.SELECTION_NAME, *tr.FIGURE_NAMES):
        assert not _ignored(f"{base}/{name}"), name
    for name in EDA_FIGURE_NAMES[3:]:
        assert _ignored(f"{base}/{tr.PRIVATE_DIR}/{name}"), name


def test_committed_selection_evidence_is_aggregate_only_and_intact() -> None:
    """The tracked selection.json: valid body hash, pinned hashes, two dates, short lists."""
    path = PROJECT_ROOT / tr.REAL_DIR / tr.SELECTION_NAME
    text = path.read_text(encoding="utf-8")
    s = json.loads(text)
    assert set(s) == TOP_KEYS
    assert tr.summary_json(s) == text
    assert tr.body_sha256(s) == s["body_sha256"]
    assert {k: s["provenance"][k] for k in PINNED} == PINNED
    assert (s["provenance"]["n_train"], s["design"]["gof_bootstrap_reps"]) == (2000, 999)
    assert_aggregates_only(text, s)
    tracked = {p.name for p in path.parent.iterdir() if p.is_file()}
    assert tracked == {tr.SELECTION_NAME, *tr.FIGURE_NAMES}
