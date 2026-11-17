"""Experiment bookkeeping. Day 2 scope: run-manifest metadata only (plan A5e item 4; A6 Day 2).

A manifest records everything needed to reproduce or audit a run:

* Python version and every installed distribution with its version (``importlib.metadata``);
* platform string (OS and architecture; no hostname);
* numpy BLAS/LAPACK build configuration and the thread-pinning environment variables;
* git commit hash plus a dirty-tree flag;
* SHA-256 of the configuration file;
* SHA-256 of every input data file.

Paths are recorded relative to the repository root where possible, and the manifest
contains no timestamps (A5e item 10). Experiment execution is added on Day 11.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from niftytail.__main__ import THREAD_ENV_VARS
from niftytail.config import config_sha256

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "GitState",
    "blas_info",
    "build_manifest",
    "file_sha256",
    "git_state",
    "installed_packages",
    "write_manifest",
]

MANIFEST_SCHEMA_VERSION = 1
_CHUNK = 1 << 20


def file_sha256(path: str | Path) -> str:
    """Streaming SHA-256 of a file's bytes.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def installed_packages() -> dict[str, str]:
    """All installed distributions as ``{normalised name: version}``, sorted by name."""
    pkgs: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name.lower().replace("_", "-")] = dist.version
    return dict(sorted(pkgs.items()))


def blas_info() -> dict[str, Any]:
    """numpy's BLAS/LAPACK build configuration plus the thread-pinning variables in force."""
    import numpy as np  # imported lazily: callers pin threads first (A5e item 5)

    info: dict[str, Any] = {"numpy_version": np.__version__}
    try:
        cfg = np.show_config(mode="dicts")
    except (TypeError, ValueError) as exc:  # pragma: no cover - older numpy only
        info["build_dependencies"] = f"unavailable: {exc}"
    else:
        raw = cfg.get("Build Dependencies", {}) if isinstance(cfg, dict) else {}
        deps: dict[str, Any] = dict(raw)
        info["build_dependencies"] = {k: deps[k] for k in sorted(deps) if k in {"blas", "lapack"}}
    info["thread_env"] = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
    return info


@dataclass(frozen=True, slots=True)
class GitState:
    """Commit hash and dirty flag. ``None`` values mean git or a commit is unavailable."""

    commit: str | None
    dirty: bool | None


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:  # git not installed
        return None


def git_state(repo_root: str | Path) -> GitState:
    """Current HEAD commit and whether the work tree has uncommitted or untracked changes.

    Ignored files (data, run outputs, caches) do not count as dirty.
    """
    root = Path(repo_root)
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    if inside is None or inside.returncode != 0 or inside.stdout.strip() != "true":
        return GitState(commit=None, dirty=None)
    head = _git(root, "rev-parse", "--verify", "HEAD")
    commit = head.stdout.strip() if head is not None and head.returncode == 0 else None
    status = _git(root, "status", "--porcelain")
    dirty = bool(status.stdout.strip()) if status is not None and status.returncode == 0 else None
    return GitState(commit=commit, dirty=dirty)


def _display_path(path: Path, repo_root: Path) -> str:
    """Repository-relative POSIX path if inside the repo, else the bare file name."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_manifest(
    config_path: str | Path,
    data_paths: Iterable[str | Path] = (),
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the run manifest (A5e item 4). Contains no timestamps or hostnames.

    Args:
        config_path: configuration file used for the run.
        data_paths: every input data file; each is hashed (and must exist).
        repo_root: repository root for git state and relative paths (default: cwd).

    Raises:
        FileNotFoundError: if the configuration file or a data file is missing.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    cfg = Path(config_path)
    data = sorted({Path(p) for p in data_paths}, key=lambda p: _display_path(p, root))
    git = git_state(root)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "python": {"version": sys.version, "implementation": platform.python_implementation()},
        "platform": platform.platform(),
        "packages": installed_packages(),
        "blas": blas_info(),
        "git": {"commit": git.commit, "dirty": git.dirty},
        "config": {"file": _display_path(cfg, root), "sha256": config_sha256(cfg)},
        "data_files": [{"file": _display_path(p, root), "sha256": file_sha256(p)} for p in data],
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write ``manifest`` as deterministic JSON (sorted keys, LF line endings)."""
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
