"""Day-8A-2: filter specification by training BIC (D39 (c), (e)).

The four candidates F1...F4 of ``garch.SPECS`` are fitted by Gaussian QMLE on the same
conditional sample ``t = 2..train_size`` of the training losses (``garch.select_filter``).
The lowest BIC is THE filter of all three conditional models; no diagnostic overrides it.
The diagnostics on the winner, and on F1 for contrast, are reported only.

The tracked output holds aggregates only (D23): no residual, sigma or loss series is written
or logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from niftytail import diagnostics as dg
from niftytail import losses as ls
from niftytail import threshold_real as tr
from niftytail import threshold_residuals as trr
from niftytail.config import Config, DependenceConfig
from niftytail.garch import SPECS, GarchFit, bic, garch_filter, n_params, select_filter

__all__ = [
    "ACF_LAGS",
    "FILTER_DIR",
    "FILTER_NAME",
    "LABEL",
    "LB_REJECT_LEVEL",
    "build_summary",
    "candidate_block",
    "diagnostics_block",
    "run_filter_selection",
]

LABEL = "filter specification by training BIC (aggregates only)"
FILTER_DIR = Path("reports/conditional")
FILTER_NAME = "filter_selection.json"
ACF_LAGS = 5  # D39 (e): ACF of z at lags 1-5
LB_REJECT_LEVEL = 0.01  # D39 (e): a Ljung-Box rejection of the winner here is a limitation


def candidate_block(fit: GarchFit, n: int) -> dict[str, Any]:
    """NLL, k, BIC, parameters and flags of one candidate (``None`` for a term it lacks)."""
    ar, gjr = SPECS[fit.spec]
    k = n_params(fit.spec)
    return {
        "mean": "AR(1)" if ar else "constant",
        "variance": "GJR(1,1)" if gjr else "GARCH(1,1)",
        "k": k,
        "nll": fit.nll,
        "bic": bic(fit.nll, k, n),
        "mu": fit.mu,
        "phi1": fit.phi1 if ar else None,
        "omega": fit.omega,
        "alpha": fit.alpha,
        "gamma": fit.gamma if gjr else None,
        "beta": fit.beta,
        "persistence": fit.alpha + fit.gamma / 2 + fit.beta,
        "start": fit.start,
        "flags": list(fit.flags),
    }


def diagnostics_block(
    eps: ls.FloatArray, z: ls.FloatArray, dep: DependenceConfig
) -> dict[str, Any]:
    """D39 (e): moments, Ljung-Box, ARCH-LM, ACF at lags 1-5 and the sign-bias tests of ``z``."""
    tests = trr.serial_tests(z, dep.ljung_box_lags, dep.arch_lm_lags)
    sb = dg.sign_bias(eps, z)
    single = {
        name: dict(zip(("t", "p"), getattr(sb, name), strict=True))
        for name in ("sign", "negative_size", "positive_size")
    }
    return {
        "moments_z": trr.moments(z),
        "ljung_box": tests["ljung_box"],
        "arch_lm": tests["arch_lm"],
        "acf_z": [float(r) for r in dg.acf(z, ACF_LAGS)],
        "sign_bias": {
            **single,
            "joint": {"lm": sb.joint_lm, "df": sb.joint_df, "p": sb.joint_p},
            "nobs": sb.nobs,
        },
        "ljung_box_rejections_at_1pct": [
            f"{kind}_{lag}"
            for kind, by_lag in tests["ljung_box"].items()
            for lag, res in by_lag.items()
            if res["p"] < LB_REJECT_LEVEL
        ],
    }


def build_summary(
    losses: ls.FloatArray,
    fits: dict[str, GarchFit],
    winner: str,
    cfg: Config,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """The content of ``filter_selection.json`` (aggregates only)."""
    window, lag = losses[1:], float(losses[0])

    def diagnostics(spec: str) -> dict[str, Any]:
        eps, _, z = garch_filter(window, fits[spec], lag)
        return {"spec": spec, **diagnostics_block(eps, z, cfg.dependence)}

    summary: dict[str, Any] = {
        "decision": "D39",
        "label": LABEL,
        "provenance": provenance,
        "design": {
            "n_likelihood_terms": int(window.size),
            "criterion": "BIC = 2 NLL + k ln(n_likelihood_terms); ties: smaller k, lower index",
            "acf_lags": ACF_LAGS,
            "ljung_box_reject_level": LB_REJECT_LEVEL,
        },
        "candidates": {spec: candidate_block(fit, int(window.size)) for spec, fit in fits.items()},
        "winner": winner,
        "diagnostics": {"winner": diagnostics(winner), "F1": diagnostics("F1")},
    }
    summary["body_sha256"] = tr.body_sha256(summary)
    return summary


def run_filter_selection(
    cfg: Config, config_path: str | Path, out_dir: Path = FILTER_DIR
) -> tuple[dict[str, Any], Path]:
    """Load the training slice as Day 7C, apply the D39 (c) rule, write the evidence.

    Raises:
        DataValidationError: from the manifest-verified loader.
        GarchError: if any of the four fits fails (nothing is written or chosen).
        ValueError: if the series has fewer than ``train_size + 1`` prices (nothing is
            written); a ``DiagnosticsError`` from a degenerate residual series is one too.
        OSError: if a pinned artefact or the manifest cannot be read.
    """
    losses, _, provenance = trr.load_training(cfg, config_path)
    fits, winner = select_filter(losses, cfg.garch)
    summary = build_summary(losses, fits, winner, cfg, provenance)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILTER_NAME
    path.write_bytes(tr.summary_json(summary).encode("utf-8"))
    return summary, path
