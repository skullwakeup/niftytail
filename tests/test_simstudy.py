"""Tests for the Day-6 N_min study (D32) and the D33 override guard, on tiny synthetic designs."""

from __future__ import annotations

import copy
import json
import math
import shutil
import tomllib
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from niftytail.__main__ import main
from niftytail.config import Config, parse_config
from niftytail.gpd import fit_mle, tail_var
from niftytail.simstudy import (
    adopted_block,
    apply_rule,
    nmin_cell,
    run_nmin_study,
    sample_gpd_excesses,
    summary_json,
)

ALLOWED_KEYS = {
    "decision", "config_sha256", "design", "cells", "rule",
    "xi", "beta", "n_u", "replications", "seed", "window", "var_level",
    "converged", "convergence_rate", "n_estimation_error", "bias_xi", "rmse_xi",
    "rel_rmse_var", "p_xi_ge_0_5", "p_xi_ge_1", "p_irregular",
    "criteria", "min_convergence", "max_rmse_xi", "max_rel_rmse_var",
    "by_n_u", "per_xi", "qualifies", "convergence", "n_min", "q_max", "q_max_exact",
    "0.1", "0.25", "100",
    "u_over_beta_ref", "rel_rmse_var_level_ref",  # D33, descriptive only
}  # fmt: skip
GRID = dict(min_convergence=Fraction(99, 100), max_rmse_xi=Fraction(15, 100))


@pytest.fixture
def tiny(base_config_path: Path) -> Config:
    with base_config_path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)
    raw = copy.deepcopy(raw)
    raw["nmin_study"].update(xi_true=["0.1", "0.25"], n_u=[100], replications=20)
    return parse_config(raw)


def _walk(node: Any, keys: set[str], lists: list[int]) -> None:
    if isinstance(node, dict):
        keys.update(node)
        for v in node.values():
            _walk(v, keys, lists)
    elif isinstance(node, list):
        lists.append(len(node))
        for v in node:
            _walk(v, keys, lists)


def _g() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261111)))


def test_sampler_matches_gpd() -> None:
    y = sample_gpd_excesses(_g(), 0.25, 1.0, 20_000)
    assert stats.kstest(y, stats.genpareto(0.25).cdf).pvalue > 0.01
    u = _g().uniform(size=5)
    expected = (1.0 / 0.4) * np.expm1(-0.4 * np.log1p(-u))
    assert np.array_equal(sample_gpd_excesses(_g(), 0.4, 1.0, 5), expected)
    assert np.array_equal(sample_gpd_excesses(_g(), 0.0, 2.0, 5), -2.0 * np.log1p(-u))


def test_tiny_study_deterministic_whitelisted_short_lists(tiny: Config) -> None:
    a = summary_json(run_nmin_study(tiny, config_sha256="test"))
    b = summary_json(run_nmin_study(tiny, config_sha256="test"))
    assert a == b and a.endswith("}\n") and "\r" not in a
    summary = json.loads(a)
    keys: set[str] = set()
    lists: list[int] = []
    _walk(summary, keys, lists)
    assert keys <= ALLOWED_KEYS, sorted(keys - ALLOWED_KEYS)
    assert max(lists) <= 20
    assert [c["xi"] for c in summary["cells"]] == [0.1, 0.25]
    assert all(c["replications"] == 20 for c in summary["cells"])
    assert all(
        c["converged"] + c["n_estimation_error"] == c["replications"] for c in summary["cells"]
    )


def test_cell_uses_spawned_child(tiny: Config) -> None:
    """Cell k = len(n_u) * i_xi + i_nu draws from SeedSequence(seed).spawn(n_cells)[k]."""
    child = np.random.SeedSequence(20261111).spawn(2)[1]
    cell = nmin_cell(
        0.25,
        100,
        20,
        np.random.Generator(np.random.PCG64(child)),
        beta=1.0,
        window=1000,
        p=Fraction(1, 100),
        xi_zero_tol=1e-8,
        settings=tiny.gpd,
    )
    assert run_nmin_study(tiny, config_sha256="x")["cells"][1] == cell


def _cell(xi: float, n_u: int, conv: int, rmse: float | None, rel: float | None) -> dict[str, Any]:
    return {
        "xi": xi,
        "n_u": n_u,
        "replications": 2000,
        "converged": conv,
        "rmse_xi": rmse,
        "rel_rmse_var": rel,
    }


def _rule(cells: list[dict[str, Any]]) -> tuple[dict[str, Any], int | None]:
    return apply_rule(
        cells,
        n_u_grid=(50, 100),
        xi_grid=(Fraction(1, 10), Fraction(1, 4)),
        max_rel_rmse=Fraction(15, 100),
        **GRID,
    )


def test_rule_qualifying_and_smallest() -> None:
    cells = [
        _cell(0.1, 50, 2000, 0.2, 0.1),  # rmse fails at 50
        _cell(0.25, 50, 2000, 0.1, 0.1),
        _cell(0.1, 100, 2000, 0.1, 0.1),
        _cell(0.25, 100, 1990, 0.1, 0.1),
    ]
    table, n_min = _rule(cells)
    assert n_min == 100
    assert table["50"]["qualifies"] is False and table["100"]["qualifies"] is True
    assert table["50"]["per_xi"]["0.1"] == {
        "convergence": True,
        "rmse_xi": False,
        "rel_rmse_var": True,
    }


def test_rule_failure_none_qualifies() -> None:
    cells = [_cell(xi, n, 1979, 0.1, 0.1) for xi in (0.1, 0.25) for n in (50, 100)]
    table, n_min = _rule(cells)  # 1979 / 2000 < 0.99 everywhere
    assert n_min is None and not any(v["qualifies"] for v in table.values())
    cells = [_cell(xi, n, 0, None, None) for xi in (0.1, 0.25) for n in (50, 100)]
    assert _rule(cells)[1] is None  # no converged fits -> fails, no crash


def test_rule_ties_pass() -> None:
    cells = [_cell(xi, n, 1980, 0.15, 0.15) for xi in (0.1, 0.25) for n in (50, 100)]
    table, n_min = _rule(cells)  # 1980/2000 = 0.99 exactly; 0.15 vs Fraction("0.15")
    assert Fraction(15, 100) >= 0.15  # float 0.15 is just below 3/20
    assert n_min == 50 and table["50"]["per_xi"]["0.25"]["convergence"] is True
    cells = [_cell(xi, n, 2000, 0.1500001, 0.1) for xi in (0.1, 0.25) for n in (50, 100)]
    assert _rule(cells)[1] is None


def test_cli_nmin_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_root: Path, tiny: Config
) -> None:
    lines = (project_root / "configs" / "base.toml").read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if not ln.lstrip("# ").startswith("n_min ")]  # unset n_min
    text = "\n".join(lines) + "\n"
    text = text.replace('xi_true = ["0.1", "0.25", "0.4"]', 'xi_true = ["0.1"]')
    text = text.replace("n_u = [50, 75, 100, 150]", "n_u = [50, 100]")
    text = text.replace("replications = 2000", "replications = 10")
    cfg_path = tmp_path / "configs" / "base.toml"
    cfg_path.parent.mkdir()
    # n_min = 100 equals the seeded rule result of this tiny design (D33 guard case (a))
    cfg_path.write_text(
        text.replace("[threshold]", "[threshold]\nn_min = 100", 1), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["nmin-study", "configs/base.toml"]) == 0
    out = tmp_path / "reports" / "threshold_selection" / "nmin_study.json"
    first = out.read_bytes()
    assert main(["nmin-study", "configs/base.toml"]) == 0
    assert out.read_bytes() == first
    n_min = json.loads(first)["rule"]["n_min"]
    wrong = 100 if n_min != 100 else 50
    cfg_path.write_text(
        text.replace("[threshold]", f"[threshold]\nn_min = {wrong}", 1), encoding="utf-8"
    )
    assert main(["nmin-study", "configs/base.toml"]) == 1  # config n_min disagrees
    assert main(["nmin-study", "configs/missing.toml"]) == 1
    shutil.rmtree(tmp_path / "reports")


# --- D33: secondary descriptive metric ------------------------------------------------------
def _cell_args(tiny: Config) -> dict[str, Any]:
    return dict(beta=1.0, window=1000, p=Fraction(1, 100), xi_zero_tol=1e-8, settings=tiny.gpd)


def test_secondary_metric_formula(tiny: Config) -> None:
    cell = nmin_cell(0.25, 100, 20, _g(), **_cell_args(tiny))
    u_ref = float(stats.norm.ppf(0.9))  # q = 1 - 100/1000
    s = u_ref * float(stats.norm.pdf(u_ref)) / float(stats.norm.sf(u_ref))
    assert cell["u_over_beta_ref"] == pytest.approx(s, rel=1e-14)
    x_p = tail_var(0.0, 0.25, 1.0, 1000, 100, Fraction(1, 100), 1e-8)
    expected = cell["rel_rmse_var"] * x_p / (s + x_p)
    assert cell["rel_rmse_var_level_ref"] == pytest.approx(expected, rel=1e-12)


def test_pre_existing_cell_keys_bit_identical(tiny: Config) -> None:
    """The D32 per-cell values are unchanged by D33: re-implement the D32 loop and compare."""
    cell = nmin_cell(0.25, 100, 20, _g(), **_cell_args(tiny))
    g, x_p = _g(), tail_var(0.0, 0.25, 1.0, 1000, 100, Fraction(1, 100), 1e-8)
    xi_hat, rel = [], []
    for _ in range(20):
        y = sample_gpd_excesses(g, 0.25, 1.0, 100)
        f = fit_mle(y, min_excesses=2, xi_zero_tol=1e-8, settings=tiny.gpd)
        xi_hat.append(f.xi)
        rel.append((tail_var(0.0, f.xi, f.beta, 1000, 100, Fraction(1, 100), 1e-8) - x_p) / x_p)
    est, r = np.array(xi_hat), np.array(rel)
    assert cell["bias_xi"] == float(est.mean() - 0.25)
    assert cell["rmse_xi"] == math.sqrt(float(np.mean((est - 0.25) ** 2)))
    assert cell["rel_rmse_var"] == math.sqrt(float(np.mean(r**2)))


def test_secondary_metric_never_enters_the_rule() -> None:
    cells = [_cell(xi, n, 2000, 0.1, 0.1) for xi in (0.1, 0.25) for n in (50, 100)]
    before = _rule(cells)
    for c in cells:
        c.update(u_over_beta_ref=1.0, rel_rmse_var_level_ref=99.0)
    assert _rule(cells) == before


# --- D33: adopted N_min guard ---------------------------------------------------------------
def test_adopted_block_six_cases() -> None:
    with pytest.raises(ValueError, match="disagrees"):  # (a)
        adopted_block(75, 100, None, 1000)
    with pytest.raises(ValueError, match="unset or different"):  # (a) with config n_min None
        adopted_block(75, None, None, 1000)
    with pytest.raises(ValueError, match="unset or different"):  # override cannot replace a rule
        adopted_block(75, None, "D33", 1000)
    with pytest.raises(ValueError, match="override_decision"):  # (b)
        adopted_block(None, 100, None, 1000)
    assert adopted_block(None, 100, "D33", 1000) == {  # (c)
        "n_min": 100,
        "source": "D33",
        "rule_n_min": None,
        "q_max": "9/10",
    }
    assert adopted_block(75, 75, None, 1000) == {  # (d)
        "n_min": 75,
        "source": "rule",
        "rule_n_min": 75,
        "q_max": "37/40",
    }
    assert adopted_block(None, None, None, 1000) is None  # (e)


def _cli_text(project_root: Path, *, passing: bool, extra: str = "", override: bool) -> str:
    lines = (project_root / "configs" / "base.toml").read_text(encoding="utf-8").splitlines()
    drop = ("n_min ", "override_decision ")
    text = "\n".join(ln for ln in lines if not ln.lstrip("# ").startswith(drop)) + "\n"
    text = text.replace('xi_true = ["0.1", "0.25", "0.4"]', 'xi_true = ["0.1"]')
    text = text.replace("n_u = [50, 75, 100, 150]", "n_u = [50, 100]")
    text = text.replace("replications = 2000", "replications = 10")
    if passing:  # loose criteria: N_u = 50 qualifies
        text = text.replace('max_rmse_xi = "0.15"', 'max_rmse_xi = "10"')
        text = text.replace('max_rel_rmse_var99 = "0.15"', 'max_rel_rmse_var99 = "10"')
    else:  # impossible criterion: rule failure
        text = text.replace('max_rmse_xi = "0.15"', 'max_rmse_xi = "1e-9"')
    if override:
        text = text.replace('var_level = "0.99"', 'var_level = "0.99"\noverride_decision = "D33"')
    return text.replace("[threshold]", f"[threshold]\n{extra}", 1)


@pytest.mark.parametrize(
    ("passing", "extra", "override", "code", "adopted"),
    [
        (True, "n_min = 100", False, 1, None),  # (a) rule 50, config differs
        (True, "", False, 1, None),  # (a) rule 50, config n_min unset
        (False, "n_min = 100", False, 1, None),  # (b) rule failure, no override
        (
            False,
            "n_min = 100",
            True,
            0,
            {"n_min": 100, "source": "D33", "rule_n_min": None, "q_max": "9/10"},
        ),  # (c)
        (
            True,
            "n_min = 50",
            False,
            0,
            {"n_min": 50, "source": "rule", "rule_n_min": 50, "q_max": "19/20"},
        ),  # (d)
        (False, "", False, 0, None),  # (e) both null
    ],
)
def test_cli_nmin_study_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
    passing: bool,
    extra: str,
    override: bool,
    code: int,
    adopted: dict[str, Any] | None,
) -> None:
    cfg_path = tmp_path / "configs" / "base.toml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(
        _cli_text(project_root, passing=passing, extra=extra, override=override), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "reports" / "threshold_selection" / "nmin_study.json"
    assert main(["nmin-study", "configs/base.toml"]) == code
    if code == 1:
        assert not out.exists()  # the guard fails before anything is written
        return
    first = out.read_bytes()
    summary = json.loads(first)
    assert summary["adopted"] == adopted
    assert summary["rule"]["n_min"] == (50 if passing else None)
    assert main(["nmin-study", "configs/base.toml"]) == 0
    assert out.read_bytes() == first  # deterministic
    shutil.rmtree(tmp_path / "reports")
