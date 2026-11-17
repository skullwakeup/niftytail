"""ENV-1 canary (decision D24).

These tests must fail loudly if Windows Smart App Control blocks the compiled wheels of
pandas or statsmodels again. They are deliberately NOT skippable (no skipif/importorskip):
an ImportError here means ENV-1 has re-opened. Synthetic data only; no market data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 20261111


def test_pandas_csv_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2000-01-03", periods=300, name="date")
    df = pd.DataFrame({"x": rng.standard_normal(300)}, index=idx)
    path = tmp_path / "synthetic.csv"
    df.to_csv(path)
    back = pd.read_csv(path, parse_dates=["date"], index_col="date")
    assert back.index.equals(df.index)
    np.testing.assert_allclose(back["x"].to_numpy(), df["x"].to_numpy(), rtol=1e-12)


def test_statsmodels_hac_ols_runs() -> None:
    d = np.random.default_rng(SEED).standard_normal(500)
    res = sm.OLS(d, np.ones(500)).fit(cov_type="HAC", cov_kwds={"maxlags": 9, "kernel": "bartlett"})
    assert abs(res.params[0] - d.mean()) < 1e-12
    assert res.bse[0] > 0
