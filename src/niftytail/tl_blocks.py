"""Traffic-light block table for plan A4 (D51 (j)): per-block zones from the kept run dirs.

The tracked backtest.json holds zone counts only; the A4 strip plot needs the block sequence,
which lives in each run dir's backtest_per_date.json. This command reads ONLY the three
published run dirs (PR W1000 and W2000 from 11A, TRI W1000 from 11C), checks them against the
tracked aggregates (D51 (3)), recomputes the blocks from the per-date hits and writes one
tracked aggregate without dates: reports/oos/traffic_light_blocks.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from niftytail import backtest, oos, tri
from niftytail import experiment as ex
from niftytail.config import Config

__all__ = [
    "LABEL",
    "OUT",
    "PASSES",
    "TlBlocksMismatch",
    "build",
    "changed_paths",
    "run",
]

RUN_11A_COMMIT = "04572a36048b57fe710d87268025369f4b4bf741"
RUN_11C_COMMIT = "4c68343872f483bff2f5f0e88ddc6fdeb084fc95"
OUT = oos.OOS_DIR / "traffic_light_blocks.json"
LABEL = "Basel traffic light by 250-day block (plan A4; descriptive); block index only, no dates"
# (id, label, window, tracked dir, run commit, TRI?)
PASSES: tuple[tuple[str, str, int, Path, str, bool], ...] = (
    ("pr_w1000", "PR W1000 (primary)", 1000, oos.OOS_DIR / "w1000", RUN_11A_COMMIT, False),
    ("pr_w2000", "PR W2000", 2000, oos.OOS_DIR / "w2000", RUN_11A_COMMIT, False),
    ("tri_w1000", "TRI W1000", 1000, oos.OOS_DIR / "tri_w1000", RUN_11C_COMMIT, True),
)
# D51 (3) (ii): the protected hashes (docs/protected_hashes.md) of the tracked backtest.json files.
BACKTEST_SHA256 = {
    "pr_w1000": "bc79fd419d16ae5ac2cf03800ef6b505088bd9ea27fc559078d78cdb35b7f20a",
    "pr_w2000": "f900783ca58644aae3fba27fe51f6b936107959ed01366178ee64e29b44b5f12",
    "tri_w1000": "425aba7b13494c4af47a9b689d624966084870a948c999947daec266ac4006ad",
}
# D51 (3) (iii): status.md Day 11C (none recorded for the PR passes).
PER_DATE_SHA256 = {
    "tri_w1000": "3c88059faf2f980a493c91d9a9d41176b58414f1ea358f1a79c38c81d4ac9656",
}
# D51 (3) (iv): xi_path summary provenance (PR W1000); status.md Day 11C (TRI).
FORECASTS_SHA256 = {
    "pr_w1000": "714523a4fb354cd06cdb1764ed2fc79978ec8e14b1a4f52899dc433423c54cd9",
    "tri_w1000": "f34c95d9d54f4707ecde5f37ed3c1f9d8f17b8daf7967c46b673ca65d663890d",
}


class TlBlocksMismatch(oos.OosError):
    """A D51 (3) check failed. Messages carry pass, key and block index only."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def changed_paths(root: Path) -> set[str]:
    """Repository-relative POSIX paths with uncommitted changes (untracked files included)."""
    res = ex._git(root, "status", "--porcelain", "--untracked-files=all")
    if res is None or res.returncode != 0:
        raise oos.OosError("git status failed")
    return {line[3:].strip().strip('"') for line in res.stdout.splitlines() if line.strip()}


def head_commit(root: Path, allowed: set[str], verify: bool) -> str:
    """D51 (5): a clean tree to write; to verify, only the command's own outputs may differ."""
    if not verify:
        return oos.clean_commit(root)
    commit = ex.git_state(root).commit
    if commit is None or not changed_paths(root) <= allowed:
        raise oos.OosError("verify needs a git commit and a clean tree apart from the outputs")
    return commit


def _run_dir(root: Path, window: int, tracked: dict[str, Any], is_tri: bool) -> Path:
    cfg_sha = tracked["config"]["sha256"]
    commit = tracked["git"]["commit"]
    if is_tri:
        t = tracked["tri"]
        return tri.run_dir(
            root, cfg_sha, t["pr_manifest_sha256"], t["tri_manifest_sha256"], window, commit
        )
    return oos.run_dir(root, cfg_sha, tracked["data_manifest"]["sha256"], window, commit)


def _blocks_from_hits(hits: str, cfg: Config) -> list[dict[str, Any]]:
    h = np.array([c == "1" for c in hits], dtype=np.int64)
    ok = np.array([c != "." for c in hits], dtype=np.bool_)
    return backtest.traffic_light(h, ok, cfg.backtest)


def _pass(
    root: Path, cfg: Config, cfg_sha: str, spec: tuple[str, str, int, Path, str, bool]
) -> dict[str, Any]:
    pid, label, window, tdir, run_commit, is_tri = spec
    tracked_man = _json(root / tdir / oos.MANIFEST)
    if tracked_man["git"]["commit"] != run_commit or tracked_man["config"]["sha256"] != cfg_sha:
        raise TlBlocksMismatch(f"{pid}: tracked manifest commit or config sha256 differs")
    tracked_bt_bytes = (root / tdir / oos.BACKTEST).read_bytes()
    if _sha(tracked_bt_bytes) != BACKTEST_SHA256[pid]:
        raise TlBlocksMismatch(
            f"{pid}: tracked backtest.json sha256 differs from docs/protected_hashes.md"
        )
    rd = _run_dir(root, window, tracked_man, is_tri)
    if not rd.is_dir():
        raise TlBlocksMismatch(f"{pid}: run dir missing")
    man_bytes = (rd / oos.MANIFEST).read_bytes()
    man = json.loads(man_bytes)
    if man["git"] != {"commit": run_commit, "dirty": False} or man["config"]["sha256"] != cfg_sha:
        raise TlBlocksMismatch(f"{pid}: run-dir manifest commit, dirty flag or config differs")
    if _sha((rd / oos.BACKTEST).read_bytes()) != BACKTEST_SHA256[pid]:
        raise TlBlocksMismatch(f"{pid}: run-dir backtest.json differs from the tracked one")
    per_bytes = (rd / oos.PER_DATE).read_bytes()
    if pid in PER_DATE_SHA256 and _sha(per_bytes) != PER_DATE_SHA256[pid]:
        raise TlBlocksMismatch(f"{pid}: backtest_per_date.json sha256 differs from status.md")
    if pid in FORECASTS_SHA256 and ex.file_sha256(rd / oos.FORECASTS) != FORECASTS_SHA256[pid]:
        raise TlBlocksMismatch(f"{pid}: forecasts.csv sha256 differs from the recorded value")
    per = json.loads(per_bytes)
    tl_counts = json.loads(tracked_bt_bytes)["traffic_light"]
    lvl = backtest._dec(cfg.backtest.traffic_light_level)
    keys: dict[str, Any] = {}
    for key in sorted(tl_counts):
        blocks = _blocks_from_hits(per["hits"][lvl][key], cfg)
        stored = [
            {k: v for k, v in b.items() if k not in ("start_date", "end_date")}
            for b in per["traffic_light"][key]
        ]
        if blocks != stored:
            bad = next(
                (i for i, (a, b) in enumerate(zip(blocks, stored, strict=False)) if a != b),
                min(len(blocks), len(stored)),
            )
            raise TlBlocksMismatch(f"{pid} {key}: recomputed block {bad} differs from per-date")
        if backtest._zone_counts(blocks) != tl_counts[key]:
            raise TlBlocksMismatch(f"{pid} {key}: zone counts differ from tracked backtest.json")
        keys[key] = {
            "blocks": [
                {f: b[f] for f in ("block", "n_obs", "breaches", "zone", "inadmissible_rows")}
                for b in blocks
                if b["zone"] is not None
            ],
            "partial_block": next(
                (
                    {"n_obs": b["n_obs"], "breaches": b["breaches"]}
                    for b in blocks
                    if b["zone"] is None
                ),
                None,
            ),
        }
    return {
        "label": label,
        "window": window,
        "run_dir": rd.name,
        "run_manifest_sha256": _sha(man_bytes),
        "run_commit": run_commit,
        "keys": keys,
    }


def build(cfg: Config, config_path: Path, root: Path, commit: str) -> bytes:
    """The traffic_light_blocks.json bytes (D51 (4)), after every D51 (3) check."""
    cfg_sha = _sha((root / config_path).read_bytes())
    doc = {
        "decision": "D51",
        "label": LABEL,
        "level": backtest._dec(cfg.backtest.traffic_light_level),
        "block_size": cfg.backtest.traffic_light_block,
        "provenance": {"commit": commit, "config_sha256": cfg_sha},
        "passes": {spec[0]: _pass(root, cfg, cfg_sha, spec) for spec in PASSES},
    }
    return backtest.to_json(doc).encode()


def run(
    cfg: Config, config_path: str | Path, verify: bool, root: str | Path, log: logging.Logger
) -> int:
    """D51 (j), (5). Exit 0 written/verified, 1 refused or mismatch.

    Raises:
        OosError: dirty tree or no commit; TlBlocksMismatch on a failed check.
    """
    base = Path(root)
    commit = head_commit(base, {OUT.as_posix()}, verify)
    out = base / OUT
    if not verify and out.exists():
        log.error("traffic_light_blocks.json exists; refusing", extra={"path": OUT.as_posix()})
        return 1
    data = build(cfg, Path(config_path), base, commit)
    info = {"sha256": _sha(data), "passes": [p[0] for p in PASSES]}
    if verify:
        on_disk = _sha(out.read_bytes()) if out.is_file() else None
        ok = on_disk == info["sha256"]
        log.log(
            logging.INFO if ok else logging.ERROR,
            "verify: hash matches" if ok else "verify: MISMATCH",
            extra={**info, "on_disk": on_disk},
        )
        return 0 if ok else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    log.info("OK", extra=info)
    return 0
