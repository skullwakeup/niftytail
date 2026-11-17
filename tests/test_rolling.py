"""Rolling engine (D41): alignment, window contract, leakage invariances, determinism, failures.

Synthetic data only. Small design n = 300, W = 100, T0 = 150 (all models real); one alignment
test at the real sizes (n = 7716, T0 = 2000, W = 1000 and 2000) with the GPD and GARCH
models replaced by cheap spies.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from niftytail import models, rolling
from niftytail.config import Config, load_config
from niftytail.garch import GarchError
from niftytail.models import Forecast, evt_key
from niftytail.threshold_study import garch_returns

N, W, T0 = 300, 100, 150
POOL = 8  # workers for the extra invariance passes (worker count is shown irrelevant below)
SENS = ("0.86", "0.9", "0.91")
CONDITIONAL = ("garch_n", "fhs", "garch_evt", *(f"garch_evt@{q}" for q in SENS))
UNCONDITIONAL = ("gaussian", "hs", "pot_gpd", *(f"pot_gpd@{q}" for q in SENS))


def _rng(k: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111, spawn_key=(41, k))))


def _series(n: int, k: int = 0) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Synthetic GARCH(1,1) losses with standardized t(6) innovations (daily scale 1%)."""
    r, _ = garch_returns(_rng(k).standard_t(6, n) / math.sqrt(1.5), 0.02, 0.08, 0.90)
    return -0.01 * np.asarray(r)


def _dates(n: int) -> list[str]:
    return [f"day{i:05d}" for i in range(n)]


def _with_t0(cfg: Config, t0: int) -> Config:
    return dataclasses.replace(cfg, design=dataclasses.replace(cfg.design, train_size=t0))


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "base.toml")


@pytest.fixture(scope="module")
def small(cfg: Config) -> Config:
    return _with_t0(cfg, T0)


@pytest.fixture(scope="module")
def losses() -> np.ndarray[Any, np.dtype[np.float64]]:
    return _series(N)


@pytest.fixture(scope="module")
def base(small: Config, losses: np.ndarray[Any, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return rolling.run_rolling(losses, _dates(N), small, W, 1)


def _forecast_bits(table: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Every forecast cell of every row, floats as their 64-bit patterns (the loss excluded)."""
    cols = [c for c in table.columns if c != "loss"]
    out = []
    for _, row in table[cols].iterrows():
        out.append(
            tuple(v if isinstance(v, str) else np.float64(v).view(np.uint64).item() for v in row)
        )
    return out


# --- Alignment and the window contract -------------------------------------------------------


def test_hs_var_is_the_order_statistic_of_the_window(base: Any, losses: Any, small: Config) -> None:
    table, _ = base
    assert list(table.index.get_level_values("j")) == list(range(T0, N))
    assert list(table.index.get_level_values("date")) == _dates(N)[T0:]
    assert table["loss"].tolist() == losses[T0:].tolist()
    for j in range(T0, N):
        x = np.sort(losses[j - W : j])
        assert table.loc[j, "hs:var:0.95"].item() == x[95 - 1]  # ceil(100 * 0.95) = 95
        assert table.loc[j, "hs:var:0.99"].item() == x[99 - 1]
        assert table.loc[j, "hs:var:0.999"].item() == x[100 - 1]


def test_column_set(base: Any) -> None:
    table, summary = base
    keys = {c.split(":")[0] for c in table.columns if c != "loss"}
    assert keys == {*UNCONDITIONAL, *CONDITIONAL}
    assert list(table.columns) == sorted(table.columns)
    assert summary["n_targets"] == N - T0 and summary["first_target"] == T0
    assert summary["no_pre_window_lag"] == 0


def test_step_change_first_affects_the_next_target(base: Any, losses: Any, small: Config) -> None:
    table, _ = base
    k = 200
    stepped = losses.copy()
    stepped[k:] += 0.05
    other, _ = rolling.run_rolling(stepped, _dates(N), small, W, POOL)
    a, b = _forecast_bits(table), _forecast_bits(other)
    assert a[: k - T0 + 1] == b[: k - T0 + 1]  # targets T0 .. k unchanged
    for col in ("gaussian:var:0.99", "hs:es:0.95", "garch_n:var:0.99"):
        assert table.loc[k + 1, col].item() != other.loc[k + 1, col].item()


class _Spy:
    """Records the arguments of every model call and returns cheap constant forecasts."""

    def __init__(self, cfg: Config) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {
            "gaussian": [],
            "hs": [],
            "pot": [],
            "cond": [],
        }
        self.cfg = cfg
        self.real_hs = models.hs

    def _f(self, name: str) -> Forecast:
        nan = dict.fromkeys(models.levels(self.cfg), 0.0)
        return Forecast(name, nan, dict(nan))

    def gaussian(self, x: Any, cfg: Config) -> Forecast:
        self.calls["gaussian"].append((np.array(x),))
        return self._f("gaussian")

    def hs(self, x: Any, cfg: Config) -> Forecast:
        self.calls["hs"].append((np.array(x),))
        return self.real_hs(x, cfg)

    def pot_gpd(self, x: Any, cfg: Config, q: Fraction) -> Forecast:
        self.calls["pot"].append((np.array(x), q))
        return self._f("pot_gpd")

    def conditional(
        self, x: Any, pre: float | None, cfg: Config, qs: tuple[Fraction, ...]
    ) -> dict[str, Forecast]:
        self.calls["cond"].append((np.array(x), pre, qs))
        return {
            "garch_n": self._f("garch_n"),
            "fhs": self._f("fhs"),
            **{evt_key(q): self._f("garch_evt") for q in qs},
        }


def _spy_run(
    monkeypatch: pytest.MonkeyPatch, losses: Any, cfg: Config, window: int
) -> tuple[_Spy, pd.DataFrame]:
    spy = _Spy(cfg)
    for name in ("gaussian", "hs", "pot_gpd", "conditional"):
        monkeypatch.setattr(models, name, getattr(spy, name))
    table, _ = rolling.run_rolling(losses, _dates(len(losses)), cfg, window, 1)
    return spy, table


def _check_contract(spy: _Spy, table: pd.DataFrame, losses: Any, t0: int, window: int) -> None:
    n = len(losses)
    for name in ("gaussian", "hs", "cond"):
        assert len(spy.calls[name]) == n - t0
    assert len(spy.calls["pot"]) == 4 * (n - t0)
    for i, j in enumerate(range(t0, n)):
        expect = losses[j - window : j]
        for name in ("gaussian", "hs"):
            assert np.array_equal(spy.calls[name][i][0], expect)
        for x, _ in spy.calls["pot"][4 * i : 4 * i + 4]:
            assert np.array_equal(x, expect)
        x, pre, qs = spy.calls["cond"][i]
        assert np.array_equal(x, expect)
        assert pre == (None if j - window - 1 < 0 else losses[j - window - 1])
        assert qs == (Fraction(88, 100), Fraction(86, 100), Fraction(90, 100), Fraction(91, 100))
    rank = -(-window * 99 // 100)  # ceil(W * 0.99)
    hs99 = table["hs:var:0.99"].to_numpy()
    assert all(
        hs99[i] == np.sort(losses[j - window : j])[rank - 1] for i, j in enumerate(range(t0, n))
    )


def test_window_contract_small(monkeypatch: pytest.MonkeyPatch, small: Config, losses: Any) -> None:
    spy, table = _spy_run(monkeypatch, losses, small, W)
    _check_contract(spy, table, losses, T0, W)
    flags = table["pot_gpd@0.91:flags"].unique().tolist()  # (1 - q) W = 9 < n_min
    assert flags == ["below_n_min_window"]


def test_window_contract_no_pre_lag_when_window_equals_t0(
    monkeypatch: pytest.MonkeyPatch, small: Config, losses: Any
) -> None:
    spy, table = _spy_run(monkeypatch, losses, small, T0)
    assert spy.calls["cond"][0][1] is None and spy.calls["cond"][1][1] == losses[0]
    _check_contract(spy, table, losses, T0, T0)


@pytest.mark.parametrize("window", [1000, 2000])
def test_window_contract_real_sizes(
    monkeypatch: pytest.MonkeyPatch, cfg: Config, window: int
) -> None:
    """n = 7716, T0 = 2000: W = 1000 gives L[1000:2000] and pre_lag L[999] first; W = 2000 None."""
    losses = _series(7716, k=1)
    spy, table = _spy_run(monkeypatch, losses, cfg, window)
    first_x, first_pre, _ = spy.calls["cond"][0]
    assert np.array_equal(first_x, losses[2000 - window : 2000])
    assert first_pre == (losses[999] if window == 1000 else None)
    assert sum(pre is None for _, pre, _ in spy.calls["cond"]) == (window == 2000)
    _check_contract(spy, table, losses, 2000, window)
    below = {
        k
        for k in table.columns
        if k.endswith(":flags") and "below_n_min_window" in table[k].iloc[0]
    }
    assert below == ({"pot_gpd@0.91:flags", "garch_evt@0.91:flags"} if window == 1000 else set())


# --- Leakage invariances (all models real) -------------------------------------------------


def test_future_perturbation_invariance(base: Any, losses: Any, small: Config) -> None:
    table, _ = base
    a = _forecast_bits(table)
    for i, m in enumerate((T0, 220, N - 1)):
        perturbed = losses.copy()
        perturbed[m:] = _rng(10 + i).normal(0.3, 5.0, N - m)  # arbitrary values
        other, _ = rolling.run_rolling(perturbed, _dates(N), small, W, POOL)
        b = _forecast_bits(other)
        assert a[: m - T0 + 1] == b[: m - T0 + 1], m  # every target j <= m, every model
        assert a[m - T0 + 1 :] != b[m - T0 + 1 :] or m == N - 1


def test_past_beyond_window_invariance(base: Any, losses: Any, small: Config) -> None:
    table, _ = base
    k = 80  # change L[:80]: targets j >= 181 have j - W - 1 >= 80
    changed = losses.copy()
    changed[:k] = _rng(20).normal(1.0, 3.0, k)
    other, _ = rolling.run_rolling(changed, _dates(N), small, W, POOL)
    a, b = _forecast_bits(table), _forecast_bits(other)
    assert a[k + W + 1 - T0 :] == b[k + W + 1 - T0 :]
    # target 180: same window L[80:180], but its pre_lag L[79] changed -> only the filter moves
    j = k + W
    for col in ("gaussian:var:0.99", "hs:var:0.99", "pot_gpd:var:0.99"):
        assert table.loc[j, col].item() == other.loc[j, col].item()
    assert table.loc[j, "garch_n:var:0.99"].item() != other.loc[j, "garch_n:var:0.99"].item()


# --- Determinism ---------------------------------------------------------------------------


def test_deterministic_and_worker_count_free(
    base: Any, losses: Any, small: Config, tmp_path: Path
) -> None:
    again, _ = rolling.run_rolling(losses, _dates(N), small, W, 1)
    three, s3 = rolling.run_rolling(losses, _dates(N), small, W, 3)
    paths = [
        rolling.write_table(t, tmp_path / name)
        for t, name in ((base[0], "a"), (again, "b"), (three, "c"))
    ]
    blobs = [p.read_bytes() for p in paths]
    assert blobs[0] == blobs[1] == blobs[2]
    drop = ("runtime_seconds", "workers")
    assert {k: v for k, v in s3.items() if k not in drop} == {
        k: v for k, v in base[1].items() if k not in drop
    }


# --- Failure handling ----------------------------------------------------------------------


def test_injected_garch_error(
    monkeypatch: pytest.MonkeyPatch, small: Config, losses: Any, base: Any
) -> None:
    calls = {"n": 0}
    real = models.fit_garch

    def flaky(*args: Any, **kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise GarchError("injected")
        return real(*args, **kw)

    monkeypatch.setattr(models, "fit_garch", flaky)
    table, summary = rolling.run_rolling(losses[: T0 + 3], _dates(T0 + 3), small, W, 1)
    bad = T0 + 1
    for key in CONDITIONAL:
        assert "garch_error" in table.loc[bad, f"{key}:flags"].item().split(";")
        assert all(
            math.isnan(table.loc[bad, f"{key}:{m}:{c}"].item())
            for m in ("var", "es")
            for c in ("0.95", "0.99", "0.995", "0.999")
        )
    # the unconditional models of that day, and every model of the other days, are untouched
    ref = _forecast_bits(base[0].iloc[:3])
    got = _forecast_bits(table)
    assert got[0] == ref[0] and got[2] == ref[2]
    keep = [
        i
        for i, c in enumerate(c for c in table.columns if c != "loss")
        if c.split(":")[0] in UNCONDITIONAL
    ]
    assert [got[1][i] for i in keep] == [ref[1][i] for i in keep]
    assert summary["garch_error"] == 1 and summary["no_pre_window_lag"] == 0
    assert summary["flag_counts"]["garch_evt@0.86"]["garch_error"] == 1
    assert summary["missing_var"]["garch_n"] == 1 and summary["missing_var"]["gaussian"] == 0
    assert summary["garch_error_share"] == 1 / 3 and summary["stop_for_owner"] is True


def test_no_pre_window_lag_counted_apart(small: Config, losses: Any) -> None:
    table, summary = rolling.run_rolling(losses[: T0 + 2], _dates(T0 + 2), small, T0, 1)
    for key in CONDITIONAL:
        assert "no_pre_window_lag" in table.loc[T0, f"{key}:flags"].item().split(";")
    assert "no_pre_window_lag" not in table.loc[T0 + 1, "garch_n:flags"].item()
    assert summary["no_pre_window_lag"] == 1 and summary["garch_error"] == 0
    assert summary["stop_for_owner"] is False


@pytest.mark.parametrize(("errors", "stop"), [(1, False), (2, True)])
def test_stop_flag_above_one_percent(
    monkeypatch: pytest.MonkeyPatch, small: Config, errors: int, stop: bool
) -> None:
    """100 targets: 1 GarchError (= 1%) does not stop; 2 (> 1%) does."""
    calls = {"n": 0}

    def fake_conditional(x: Any, pre: Any, cfg: Config, qs: Any) -> dict[str, Forecast]:
        calls["n"] += 1
        flags = ("garch_error",) if calls["n"] <= errors else ()
        zero = dict.fromkeys(models.levels(cfg), 0.0)
        names = [("garch_n", "garch_n"), ("fhs", "fhs"), *(("garch_evt", evt_key(q)) for q in qs)]
        return {k: Forecast(n, zero, dict(zero), flags) for n, k in names}

    zero = dict.fromkeys(models.levels(small), 0.0)
    monkeypatch.setattr(models, "conditional", fake_conditional)
    monkeypatch.setattr(models, "pot_gpd", lambda x, cfg, q: Forecast("pot_gpd", zero, zero))
    _, summary = rolling.run_rolling(_series(T0 + 100), _dates(T0 + 100), small, W, 1)
    assert summary["garch_error"] == errors and summary["stop_for_owner"] is stop


# --- Output --------------------------------------------------------------------------------


def test_run_dir_is_a_content_hash() -> None:
    a = rolling.run_dir("root", "c" * 64, "m" * 64, 1000)
    assert a == rolling.run_dir("root", "c" * 64, "m" * 64, 1000)
    assert a.parent == Path("root") / "reports" / "runs"
    assert len(a.name) == 64 and set(a.name) <= set("0123456789abcdef")
    assert a != rolling.run_dir("root", "c" * 64, "m" * 64, 2000)
    assert a != rolling.run_dir("root", "d" * 64, "m" * 64, 1000)
    assert a != rolling.run_dir("root", "c" * 64, "n" * 64, 1000)


def test_csv_round_trip_and_location(base: Any, tmp_path: Path, project_root: Path) -> None:
    table, _ = base
    directory = rolling.run_dir(tmp_path, "c" * 64, "m" * 64, W)
    path = rolling.write_table(table, directory)
    back = rolling.read_table(path)
    pd.testing.assert_frame_equal(back, table, check_exact=True)
    assert _forecast_bits(back) == _forecast_bits(table)
    assert rolling.write_table(back, tmp_path / "again").read_bytes() == path.read_bytes()
    written = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {path, tmp_path / "again" / rolling.FORECASTS_CSV}
    assert path.is_relative_to(tmp_path / "reports" / "runs")
    assert b"\r" not in path.read_bytes()
    run = rolling.run_dir(Path(), "c" * 64, "m" * 64, W)
    for rel in (run / rolling.FORECASTS_CSV, run / "summary.json"):  # not only via *.csv
        cmd = ["git", "check-ignore", "--no-index", "-q", rel.as_posix()]
        assert subprocess.run(cmd, cwd=project_root, check=False).returncode == 0, rel


def test_writes_nothing_outside_runs(
    small: Config, losses: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rolling.run_rolling(losses[: T0 + 1], _dates(T0 + 1), small, W, 1)
    assert list(tmp_path.iterdir()) == []


# --- Input validation ----------------------------------------------------------------------


def test_input_validation(small: Config, losses: Any) -> None:
    d = _dates(N)
    with pytest.raises(ValueError, match="T0 \\+ 1"):
        rolling.run_rolling(losses[:T0], d[:T0], small, W, 1)
    with pytest.raises(ValueError, match="dates"):
        rolling.run_rolling(losses, d[:-1], small, W, 1)
    bad = losses.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rolling.run_rolling(bad, d, small, W, 1)
    with pytest.raises(ValueError, match="window"):
        rolling.run_rolling(losses, d, small, T0 + 1, 1)
    with pytest.raises(ValueError, match="workers"):
        rolling.run_rolling(losses, d, small, W, 0)
    unset = dataclasses.replace(
        small, threshold=dataclasses.replace(small.threshold, q_selected=None)
    )
    with pytest.raises(ValueError, match="q_selected"):
        rolling.run_rolling(losses, d, unset, W, 1)


def test_chunk_refuses_a_target_outside_the_split(small: Config, losses: Any) -> None:
    qs = (Fraction(88, 100),)
    with pytest.raises(ValueError, match="outside"):
        rolling._chunk((losses[:T0], T0, [T0 - 1], W, small, qs, qs))
