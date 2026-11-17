"""Tests for the CLI entry point: thread pinning, JSON logging, check-config."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from niftytail.__main__ import THREAD_ENV_VARS, configure_logging, main, pin_threads


def test_thread_env_vars_cover_a5e() -> None:
    assert set(THREAD_ENV_VARS) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}


def test_pin_threads_sets_all_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in THREAD_ENV_VARS:
        monkeypatch.setenv(var, "8")
    pin_threads()
    assert all(os.environ[var] == "1" for var in THREAD_ENV_VARS)


def test_threads_pinned_in_test_session() -> None:
    assert all(os.environ.get(var) == "1" for var in THREAD_ENV_VARS)


def test_entry_point_pins_before_numpy_import() -> None:
    """In a fresh interpreter, running the entry module pins threads even if the parent set 8."""
    env = {**os.environ, **dict.fromkeys(THREAD_ENV_VARS, "8")}
    code = (
        "import runpy, sys, os; sys.argv=['niftytail','--help'];\n"
        "try:\n    runpy.run_module('niftytail', run_name='__main__')\n"
        "except SystemExit:\n    pass\n"
        "names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')\n"
        "print(','.join(os.environ[v] for v in names))\n"
        "print('numpy' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    lines = out.stdout.strip().splitlines()
    assert lines[-2] == "1,1,1"
    assert lines[-1] == "False"  # nothing numeric imported by the entry point itself


def test_json_lines_logging() -> None:
    buf = io.StringIO()
    log = configure_logging(logging.INFO, stream=buf)
    log.info("hello", extra={"run_id": "abc", "n": 3})
    record = json.loads(buf.getvalue().strip())
    assert record["level"] == "INFO" and record["message"] == "hello"
    assert record["logger"] == "niftytail"
    assert (record["run_id"], record["n"]) == ("abc", 3)
    assert "time" in record


def test_configure_logging_is_idempotent() -> None:
    configure_logging(stream=io.StringIO())
    log = configure_logging(stream=io.StringIO())
    assert len(log.handlers) == 1


def test_check_config_valid(base_config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-config", str(base_config_path)]) == 0
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["message"] == "configuration valid"
    assert len(record["config_sha256"]) == 64
    assert record["n_min_set"] is True and record["q_selected_set"] is True  # D37


def test_publish_is_not_part_of_the_public_copy(
    monkeypatch: pytest.MonkeyPatch,
    base_config_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """D65 (o): without niftytail.publish the package runs; only `publish` is refused."""
    monkeypatch.setitem(sys.modules, "niftytail.publish", None)
    assert main(["check-config", str(base_config_path)]) == 0
    assert main(["publish", "x"]) == 1
    assert "publish is not part of this copy" in capsys.readouterr().err


def test_check_config_invalid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('spec_version = "V2.1"\n', encoding="utf-8")
    assert main(["check-config", str(bad)]) == 1
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["level"] == "ERROR" and "spec_version" in record["error"]
