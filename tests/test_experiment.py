"""Tests for the run-manifest metadata infrastructure (A5e item 4; A6 Day 2)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from niftytail.config import config_sha256
from niftytail.experiment import (
    MANIFEST_SCHEMA_VERSION,
    GitState,
    blas_info,
    build_manifest,
    file_sha256,
    git_state,
    installed_packages,
    write_manifest,
)

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # FIPS 180-2 vector


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


# --- file hashing -------------------------------------------------------------
def test_file_sha256_known_vector(tmp_path: Path) -> None:
    p = tmp_path / "abc.bin"
    p.write_bytes(b"abc")
    assert file_sha256(p) == SHA256_ABC


def test_file_sha256_streams_large_file(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 20_000  # ~5 MB, spans several read chunks
    p = tmp_path / "big.bin"
    p.write_bytes(payload)
    assert file_sha256(p) == hashlib.sha256(payload).hexdigest()


def test_file_sha256_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.bin")


# --- environment -------------------------------------------------------------
def test_installed_packages_versions_and_order() -> None:
    pkgs = installed_packages()
    assert pkgs["numpy"] == metadata.version("numpy")
    assert pkgs["niftytail"] == metadata.version("niftytail")
    assert list(pkgs) == sorted(pkgs)


def test_blas_info_records_numpy_and_thread_pinning() -> None:
    import numpy as np

    info = blas_info()
    assert info["numpy_version"] == np.__version__
    assert isinstance(info["build_dependencies"], dict)
    assert set(info["build_dependencies"]) <= {"blas", "lapack"}
    assert info["thread_env"] == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


# --- git state ---------------------------------------------------------------
@requires_git
def test_git_state_clean_then_dirty(tmp_repo: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert git_state(tmp_repo) == GitState(commit=head, dirty=False)
    (tmp_repo / "a.txt").write_text("changed\n", encoding="utf-8")
    assert git_state(tmp_repo) == GitState(commit=head, dirty=True)


@requires_git
def test_git_state_untracked_counts_ignored_does_not(tmp_repo: Path) -> None:
    (tmp_repo / ".gitignore").write_text("ignored.csv\n", encoding="utf-8")
    _git(tmp_repo, "add", ".gitignore")
    _git(tmp_repo, "commit", "-q", "-m", "ignore")
    (tmp_repo / "ignored.csv").write_text("1,2\n", encoding="utf-8")
    assert git_state(tmp_repo).dirty is False
    (tmp_repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert git_state(tmp_repo).dirty is True


@requires_git
def test_git_state_repo_without_commits(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    assert git_state(tmp_path).commit is None


def test_git_state_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()
    # tmp_path may itself sit inside no repository; ensure we test a non-repo directory
    probe = (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=outside,
            capture_output=True,
            text=True,
        )
        if shutil.which("git")
        else None
    )
    if probe is not None and probe.stdout.strip() == "true":
        pytest.skip("temporary directory is inside a git work tree")
    assert git_state(outside) == GitState(commit=None, dirty=None)


# --- manifest ----------------------------------------------------------------
REQUIRED_KEYS = {
    "schema_version",
    "python",
    "platform",
    "packages",
    "blas",
    "git",
    "config",
    "data_files",
}


def test_manifest_fields(base_config_path: Path, project_root: Path, tmp_path: Path) -> None:
    d1, d2 = tmp_path / "b_input.bin", tmp_path / "a_input.bin"
    d1.write_bytes(b"abc")
    d2.write_bytes(b"")
    m = build_manifest(base_config_path, [d1, d2], repo_root=project_root)
    assert set(m) == REQUIRED_KEYS
    assert m["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert m["python"]["version"] == sys.version
    assert m["packages"]["numpy"] == metadata.version("numpy")
    assert m["config"] == {"file": "configs/base.toml", "sha256": config_sha256(base_config_path)}
    # outside the repo: bare file names only (no absolute paths); sorted by name
    assert m["data_files"] == [
        {"file": "a_input.bin", "sha256": hashlib.sha256(b"").hexdigest()},
        {"file": "b_input.bin", "sha256": SHA256_ABC},
    ]
    assert set(m["git"]) == {"commit", "dirty"}


def test_manifest_missing_data_file_raises(base_config_path: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(base_config_path, [tmp_path / "absent.csv"], repo_root=tmp_path)


def test_manifest_has_no_timestamps_or_absolute_paths(
    base_config_path: Path, project_root: Path
) -> None:
    text = json.dumps(build_manifest(base_config_path, repo_root=project_root))
    assert str(project_root) not in text and project_root.as_posix() not in text
    for key in ("time", "timestamp", "date", "hostname", "node"):
        assert f'"{key}"' not in text


def test_write_manifest_is_deterministic(
    base_config_path: Path, project_root: Path, tmp_path: Path
) -> None:
    m = build_manifest(base_config_path, repo_root=project_root)
    p1, p2 = tmp_path / "m1.json", tmp_path / "m2.json"
    write_manifest(m, p1)
    write_manifest(build_manifest(base_config_path, repo_root=project_root), p2)
    assert p1.read_bytes() == p2.read_bytes()
    raw = p1.read_bytes()
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    assert json.loads(raw) == m
