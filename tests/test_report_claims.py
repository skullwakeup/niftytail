"""docs/report.md and README.md against the tracked JSON they cite (D65 (a)-(f); D49 T6).

Aggregates only: the tracked backtest JSONs, the xi path summary and the threshold evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from conftest import PROJECT_ROOT
from niftytail import dashboard_data as dd

PASSES = {
    "PR W1000": "reports/oos/w1000/backtest.json",
    "PR W2000": "reports/oos/w2000/backtest.json",
    "TRI W1000": "reports/oos/tri_w1000/backtest.json",
}
OTHER_JSON = (
    "reports/oos/xi_path/summary.json",
    "reports/threshold_selection/real/selection.json",
    "reports/threshold_selection/oos_blocks/selection.json",
)
MODELS = ("gaussian", "hs", "pot_gpd", "garch_n", "fhs", "garch_evt")
CONDITIONAL = ("garch_n", "fhs", "garch_evt")
LEVELS = ("0.95", "0.99")
BORDERLINE = (0.01, 0.10)  # D49 T6, closed interval
# Decimals with four or more places that are not JSON values: the C3 bound |d_bar| <= 0.0052
# (results_review.md) and the exact Kupiec power at T = 5,650 (methodology 10.3).
NOT_FROM_JSON = {"0.0052", "0.4508"}
MAIN = "Main finding on the price index (pass-dependent, borderline)"
_NUMBER = re.compile(r"(?<![\w.])-?\d+\.\d{4,}(?:e-\d+)?(?![\w])")


def _load(rel: str) -> Any:
    return json.loads((PROJECT_ROOT / rel).read_text(encoding="utf-8"))


def _text(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


B = {label: _load(rel) for label, rel in PASSES.items()}


def _floats(x: Any) -> set[str]:
    """Every float of a JSON document and its absolute value, at 6 significant digits."""
    if isinstance(x, dict):
        return set().union(*map(_floats, x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*map(_floats, x)) if x else set()
    return {f"{x:.6g}", f"{abs(x):.6g}"} if isinstance(x, float) else set()


def _rows(text: str, first_cell: str) -> list[list[str]]:
    """Cells of the markdown table rows whose first cell matches ``first_cell``."""
    out = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and re.fullmatch(first_cell, cells[0]):
            out.append(cells)
    return out


def _dm(b: dict[str, Any], family: str, a: str, other: str, level: str) -> dict[str, Any]:
    (entry,) = [e for e in b["dm"][family] if (e["a"], e["b"], e["level"]) == (a, other, level)]
    return entry


def _t1_borderline() -> dict[tuple[str, str, str], float]:
    """The deciding p-values of T1 that lie in [0.01, 0.10], keyed (statement, pass, level)."""
    cells: dict[tuple[str, str, str], float] = {}
    for label, b in B.items():
        for lvl in LEVELS:
            cov = b["coverage"][lvl]
            for k in MODELS:  # F1: IND, rejected for the unconditional models only
                cells["IND " + k, label, lvl] = cov[k]["christoffersen"]["p_ind"]
            cells["garch_evt vs pot_gpd", label, lvl] = _dm(
                b, "conditional", "garch_evt", "pot_gpd", lvl
            )["p_holm"]
            cells["hs vs pot_gpd", label, lvl] = _dm(b, "unconditional", "hs", "pot_gpd", lvl)["p"]
        cells["garch_n Kupiec", label, "0.99"] = b["coverage"]["0.99"]["garch_n"]["kupiec"]["p"]
        cells["garch_n ES", label, "0.99"] = b["coverage"]["0.99"]["garch_n"]["es_residual"]["p"]
    return {k: p for k, p in cells.items() if BORDERLINE[0] <= p <= BORDERLINE[1]}


def test_table_1_breaches_match_the_json() -> None:
    rows = _rows(_text("docs/report.md"), r"0\.9[59]")
    assert len(rows) == 6
    for level, label, *counts in rows:
        want = [B[label]["coverage"][level][k]["breaches"] for k in MODELS]
        assert [int(c) for c in counts] == want, (level, label)


def test_table_2_zones_match_the_json() -> None:
    rows = [r for r in _rows(_text("docs/report.md"), r"(PR|TRI) W[12]000") if "/" in r[1]]
    assert [r[0] for r in rows] == list(PASSES)
    for label, *cells in rows:
        for k, cell in zip(MODELS, cells, strict=True):
            z = B[label]["traffic_light"][k]["zones"]
            assert cell == f"{z['green']} / {z['yellow']} / {z['red']}", (label, k)


def test_t1_holds_in_all_three_passes() -> None:
    for label, b in B.items():
        for lvl in LEVELS:
            cov = b["coverage"][lvl]
            for k in MODELS:
                rejected = cov[k]["christoffersen"]["p_ind"] <= 0.05
                assert rejected == (k not in CONDITIONAL), (label, lvl, k)
            beat = _dm(b, "conditional", "garch_evt", "pot_gpd", lvl)
            assert beat["reject"] and beat["d_bar"] < 0, (label, lvl)
            assert not _dm(b, "unconditional", "hs", "pot_gpd", lvl)["reject"], (label, lvl)
        n = b["coverage"]["0.99"]["garch_n"]
        assert n["kupiec"]["reject"] and n["es_residual"]["p"] <= 0.05, label


def test_h_holds_on_the_price_index_only() -> None:
    def never_rejected(b: dict[str, Any]) -> list[str]:
        out = []
        for k in MODELS:
            cells = [b["coverage"][lvl][k] for lvl in LEVELS]
            ps = [
                p
                for c in cells
                for p in (
                    c["christoffersen"]["p_ind"],
                    c["christoffersen"]["p_cc"],
                    c["es_residual"]["p"],
                )
            ]
            if not any(c["kupiec"]["reject"] for c in cells) and min(ps) > 0.05:
                out.append(k)
        return out

    assert never_rejected(B["PR W1000"]) == ["fhs", "garch_evt"]
    assert never_rejected(B["PR W2000"]) == ["fhs", "garch_evt"]
    assert never_rejected(B["TRI W1000"]) == ["fhs"]  # T3: garch_evt's 0.99 ES test rejects
    report = _text("docs/report.md")
    for label, text in (("TRI W1000", "p 0.0438"), ("PR W1000", "PR 0.0582")):
        p = B[label]["coverage"]["0.99"]["garch_evt"]["es_residual"]["p"]
        assert text.endswith(f"{p:.6g}") and text in report
    for q, shown in (("0.86", "0.0367"), ("0.9", "0.0483"), ("0.91", "0.0541")):
        p = B["TRI W1000"]["coverage"]["0.99"][f"garch_evt@{q}"]["es_residual"]["p"]
        assert f"{p:.6g}" == shown and f"@{q} ({shown})" in report


def test_borderline_cells_of_t1_are_labelled_in_readme_and_report() -> None:
    cells = _t1_borderline()
    assert sorted(cells) == [
        ("IND garch_evt", "TRI W1000", "0.95"),
        ("garch_evt vs pot_gpd", "PR W1000", "0.99"),
        ("garch_evt vs pot_gpd", "TRI W1000", "0.99"),
        ("hs vs pot_gpd", "PR W2000", "0.95"),
    ]
    p = {k: f"{v:.6g}" for k, v in cells.items()}
    line = (
        "Borderline in parts (p in [0.01, 0.10]): GARCH-EVT vs POT-GPD at 0.99, Holm p "
        f"{p['garch_evt vs pot_gpd', 'PR W1000', '0.99']} on PR W1000 and "
        f"{p['garch_evt vs pot_gpd', 'TRI W1000', '0.99']} on TRI W1000 "
        "(`dm.conditional[a=garch_evt, b=pot_gpd].p_holm`); IND for GARCH-EVT at 0.95 on TRI "
        f"W1000, p {p['IND garch_evt', 'TRI W1000', '0.95']} "
        "(`coverage.0.95.garch_evt.christoffersen.p_ind`); HS vs POT-GPD at 0.95 on PR W2000, "
        f"p {p['hs vs pot_gpd', 'PR W2000', '0.95']} (`dm.unconditional[a=hs, b=pot_gpd].p`). "
        "Sources: " + ", ".join(f"`{rel}`" for rel in PASSES.values()) + "."
    )
    assert line == dd.T1_BORDERLINE  # the dashboard shows the same line (D65 (g))
    for rel in ("README.md", "docs/report.md"):
        assert line in _text(rel).splitlines(), rel
    assert dd.VERIFIED in _text("README.md")  # D65 (h)


def test_h_is_never_called_a_headline() -> None:
    for rel in ("README.md", "docs/report.md"):
        lines = _text(rel).splitlines()
        assert not any("Headline 2" in ln for ln in lines), rel
        (i,) = [i for i, ln in enumerate(lines) if ln.startswith("> **H.**")]
        label = next(ln for ln in reversed(lines[:i]) if ln.startswith(("#", "**")))
        assert MAIN in label and "eadline" not in label, rel


def test_every_long_decimal_is_a_value_of_the_cited_json() -> None:
    known = set().union(*(_floats(b) for b in B.values()), *(_floats(_load(r)) for r in OTHER_JSON))
    for rel in ("README.md", "docs/report.md"):
        body = _text(rel).split("\n## References")[0]
        found = set(_NUMBER.findall(body))
        assert found, rel
        assert found - known - NOT_FROM_JSON == set(), rel


def test_wording_of_the_selection_order() -> None:
    for rel in ("README.md", "docs/report.md"):
        text = _text(rel)
        assert "before any out-of-sample day was used" not in text, rel
        assert "before any out-of-sample forecast was computed" in text, rel
    report = _text("docs/report.md")
    assert "pre-registered out-of-sample comparison" not in report
    assert "metadata U)" not in report and "841-" not in report
