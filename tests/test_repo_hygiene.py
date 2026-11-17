"""Repository hygiene: no market data in Git, pinned interpreter (D12; A5e).

Raw Nifty data and derived daily series must never be committed (critical research
rule 8). These tests fail if such files are tracked or would not be ignored.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

DATA_SUFFIXES = (".csv", ".parquet", ".xlsx", ".xls", ".feather", ".pkl", ".pickle")

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

# SHA-256 of the frozen, approved plan (docs/plan_v2.2.md); any byte change must fail.
FROZEN_PLAN_SHA256 = "44f42b13a9daf43d1857cfb9f258f6b46d9f3fc9cace9fdca58f82416465d66d"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def _is_repo(root: Path) -> bool:
    return _git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


@requires_git
def test_no_files_tracked_under_data(project_root: Path) -> None:
    if not _is_repo(project_root):
        pytest.skip("not a git repository")
    tracked = _git(project_root, "ls-files", "--", "data").stdout.split()
    assert tracked == [], f"files under data/ are tracked: {tracked}"


@requires_git
def test_no_tabular_data_files_tracked(project_root: Path) -> None:
    if not _is_repo(project_root):
        pytest.skip("not a git repository")
    tracked = _git(project_root, "ls-files").stdout.splitlines()
    offenders = [f for f in tracked if f.lower().endswith(DATA_SUFFIXES)]
    assert offenders == [], f"tabular data files are tracked: {offenders}"


@requires_git
@pytest.mark.parametrize(
    "path",
    [
        "data/raw/nifty50_pr_2026-11-11.csv",
        "data/raw/MANIFEST.toml",
        "data/processed/losses.parquet",
        "reports/runs/20261111_base/forecasts.csv",
        "reports/runs/20261111_base/manifest.json",
        "somewhere/else/nifty.csv",
        "notebooks/scratch.xlsx",
        "reports/eda/figures/acf_returns.png",
        "data/live/NIFTY 50_Historical_PR_01012026to31122026.csv",
        "data/live/MANIFEST.toml",
        "reports/live/0123abcd/forecast.json",
        "reports/live/0123abcd/series.json",
        "reports/live/0123abcd/check.json",
        "data/api/1996-07-01_1996-07-02.csv",
        "data/api/MANIFEST.toml",
    ],
)
def test_data_and_run_outputs_are_ignored(project_root: Path, path: str) -> None:
    """Accidental `git add` of raw data or run outputs must be impossible without --force."""
    if not _is_repo(project_root):
        pytest.skip("not a git repository")
    res = _git(project_root, "check-ignore", "--no-index", "-q", path)
    assert res.returncode == 0, f"{path} is NOT ignored by .gitignore"


@requires_git
def test_threshold_selection_evidence_is_not_ignored(project_root: Path) -> None:
    """Pre-registration evidence (reports/threshold_selection/) must be committable."""
    if not _is_repo(project_root):
        pytest.skip("not a git repository")
    res = _git(
        project_root, "check-ignore", "--no-index", "-q", "reports/threshold_selection/table.md"
    )
    assert res.returncode == 1


@requires_git
def test_eda_summary_is_not_ignored(project_root: Path) -> None:
    """The aggregate-only EDA summary (D29) is tracked; its figures are not."""
    if not _is_repo(project_root):
        pytest.skip("not a git repository")
    res = _git(project_root, "check-ignore", "--no-index", "-q", "reports/eda/eda_summary.json")
    assert res.returncode == 1


def test_python_pinned_to_313(project_root: Path) -> None:
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.13,<3.14"
    assert (project_root / ".python-version").read_text(encoding="utf-8").strip() == "3.13"


def test_running_interpreter_is_313() -> None:
    import sys

    assert sys.version_info[:2] == (3, 13)


def test_lockfile_present(project_root: Path) -> None:
    assert (project_root / "uv.lock").is_file()


def test_frozen_plan_sha256(project_root: Path) -> None:
    """The approved plan is frozen: its exact bytes (read in binary mode) must not change."""
    data = (project_root / "docs" / "plan_v2.2.md").read_bytes()
    assert hashlib.sha256(data).hexdigest() == FROZEN_PLAN_SHA256


def test_pyproject_authors_have_no_email(project_root: Path) -> None:
    """D62 15A-1: the author entry keeps the name only (no e-mail in a tracked file)."""
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    authors = pyproject["project"]["authors"]
    assert authors and all(set(a) == {"name"} for a in authors)


def test_readme_headline_is_t1_only(project_root: Path) -> None:
    """D62 15A-1 (D49 T6): the headline section holds T1 only; H follows with its labels."""
    text = (project_root / "README.md").read_text(encoding="utf-8")
    head = text.index("## Headline result (holds in all three passes)\n")
    main = text.index("## Main finding on the price index (pass-dependent, borderline)\n")
    assert head < main
    assert "> **T1.**" in text[head:main] and "> **H.**" not in text[head:main]
    assert "> **H.**" in text[main : text.index("\n## ", main + 1)]
    assert "Market data is never included in this repository." in text
    assert "local-only, not yet published" not in text


@requires_git
def test_demo_dir_holds_exactly_the_generated_files(project_root: Path) -> None:
    """D62 (b): app/demo is tracked and holds the two files that demo-data regenerates."""
    if not _is_repo(project_root):
        pytest.skip("not a git repository")
    from niftytail import demo
    from niftytail.config import load_config

    tracked = _git(project_root, "ls-files", "--", "app/demo").stdout.split()
    assert sorted(tracked) == sorted(f"app/demo/{n}" for n in demo.FILES)
    cfg = load_config(project_root / "configs" / "base.toml")
    blobs = demo.documents(cfg, project_root / "configs" / "base.toml", 8)
    for name, blob in blobs.items():
        assert (project_root / "app" / "demo" / name).read_bytes() == blob


def test_cloud_requirements_pinned_to_uv_lock(project_root: Path) -> None:
    """D62 (e): app/requirements.txt lists streamlit, pyarrow (uv.lock versions) and the package."""
    lines = [
        ln.strip()
        for ln in (project_root / "app" / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
    version = {p["name"]: p["version"] for p in lock["package"] if "version" in p}
    assert lines == [
        f"streamlit=={version['streamlit']}",
        f"pyarrow=={version['pyarrow']}",
        "-e .",
    ]
