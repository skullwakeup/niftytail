"""Shared test setup.

Pins BLAS/OpenMP threads to 1 for every test run (plan A5e item 5) before any
test module can import numpy. pytest imports conftest.py first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from niftytail.__main__ import pin_threads

pin_threads()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    """Absolute path of the repository root."""
    return PROJECT_ROOT


@pytest.fixture
def base_config_path() -> Path:
    """Path of the approved base configuration."""
    return PROJECT_ROOT / "configs" / "base.toml"
