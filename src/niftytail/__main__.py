"""Command-line entry point: ``python -m niftytail <command> ...``.

Process-start responsibilities (plan A5e):

1. Pin BLAS/OpenMP threads to 1 *before* any numerical library is imported,
   so floating-point reductions are reproducible.
2. Configure structured (JSON-lines) logging.

Commands: ``check-config`` (Day 2), ``validate`` (Day 3, D25), ``eda`` (Day 4, D29),
``nmin-study`` (Day 6, D32), ``threshold-study`` (Day 7B, D35), ``threshold-select``
(Day 7C, D36), ``threshold-select-residuals`` (Day 8A, D38; ``--selected-filter`` D39) and
``filter-select`` (Day 8A-2, D39). ``run`` is added on its scheduled day. Numeric libraries
are imported lazily inside commands, so importing this module never loads numpy before
threads are pinned.
"""

from __future__ import annotations

import os

THREAD_ENV_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_threads(n: int = 1) -> None:
    """Set the BLAS/OpenMP thread-count variables (A5e). Must run before numpy is imported."""
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)


pin_threads()  # at import time: this module is the process entry point

import argparse  # noqa: E402  (imports deliberately follow thread pinning)
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from datetime import UTC, date, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import IO, Any  # noqa: E402

from niftytail.config import ConfigError, config_sha256, load_config  # noqa: E402

_STD_LOG_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonLinesFormatter(logging.Formatter):
    """Format each log record as one JSON object per line.

    Fields: ``time`` (UTC, ISO 8601), ``level``, ``logger`` and ``message``, plus any
    ``extra={...}`` fields passed to the logging call. Logs are diagnostic output, not
    byte-compared artifacts, so timestamps are allowed here (A5e item 10).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in sorted(vars(record).items()):
            if key not in _STD_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=False)


def configure_logging(level: int = logging.INFO, stream: IO[str] | None = None) -> logging.Logger:
    """Attach a single JSON-lines handler to the ``niftytail`` logger and return it."""
    logger = logging.getLogger("niftytail")
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonLinesFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _cmd_check_config(args: argparse.Namespace, log: logging.Logger) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    log.info(
        "configuration valid",
        extra={
            "config_path": str(args.config),
            "config_sha256": config_sha256(args.config),
            "spec_version": cfg.spec_version,
            "n_min_set": cfg.threshold.n_min is not None,
            "q_selected_set": cfg.threshold.q_selected is not None,
        },
    )
    return 0


def _cmd_validate(args: argparse.Namespace, log: logging.Logger) -> int:
    """Validate the raw data (D25); optionally create the manifest first. Exit 0 = no errors."""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)

    raw_dir, manifest = Path(cfg.data.raw_dir), Path(cfg.data.manifest)
    out_dir = Path(cfg.data.validation_dir)
    if not args.init_manifest and (args.retrieved or args.source_url):
        log.error("--retrieved/--source-url are only valid with --init-manifest")
        return 1
    if args.init_manifest:
        if not args.retrieved or not args.source_url:
            log.error("--init-manifest requires --retrieved YYYY-MM-DD and --source-url URL")
            return 1
        try:
            retrieved = date.fromisoformat(args.retrieved)
        except ValueError:
            log.error("--retrieved is not an ISO date (YYYY-MM-DD)")
            return 1
        if not args.source_url.startswith(("https://", "http://")):
            log.error("--source-url must start with https:// or http://")
            return 1
        if manifest.exists():
            log.error(
                "manifest already exists; refusing to overwrite",
                extra={"manifest": manifest.as_posix()},
            )
            return 1
        try:
            infos, errors = dl.inspect_files(raw_dir)
        except dl.DataValidationError as exc:
            infos, errors = (), exc.errors
        if errors:
            for e in errors:
                log.error("data error", extra={"error": e})
            log.error("manifest not written: the raw files have errors")
            return 1
        dl.write_manifest(manifest, infos, source_url=args.source_url, retrieved=retrieved)
        log.info("manifest written", extra={"manifest": manifest.as_posix(), "files": len(infos)})
    report = dl.validate(
        raw_dir,
        manifest,
        start_date=cfg.data.start_date,
        large_return_flag=cfg.validation.large_return_flag,
        gap_flag_days=cfg.validation.gap_flag_days,
        stale_run_min=cfg.validation.stale_run_min,
        spot_check_n=cfg.validation.spot_check_n,
        seed=cfg.runtime.seed,
    )
    outputs = dl.write_outputs(report, out_dir)
    for e in report.errors:
        log.error("data error", extra={"error": e})
    log.log(
        logging.INFO if report.ok else logging.ERROR,
        "validation passed" if report.ok else "validation failed",
        extra={
            "n_files": len(report.files),
            "n_rows": report.n_rows,
            "first_date": report.first_date.isoformat() if report.first_date else None,
            "last_date": report.last_date.isoformat() if report.last_date else None,
            "n_errors": len(report.errors),
            "counts": report.counts,
            "first_close_equals_base_value": report.first_close_equals_base_value,
            "outputs": [p.as_posix() for p in outputs],
        },
    )
    return 0 if report.ok else 1


def _cmd_eda(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-4 EDA (D29): summary JSON and figures. Logs file names only, never values."""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail.eda import run_eda

    try:
        outputs = run_eda(cfg, args.config)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except ValueError as exc:  # losses/diagnostics messages contain no data values
        log.error("eda failed", extra={"error": str(exc)})
        return 1
    log.info("OK", extra={"outputs": [p.as_posix() for p in outputs]})
    return 0


def _cmd_nmin_study(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-6 N_min study (D32, D33): writes nmin_study.json; logs the file name and N_min only.

    Guard (D33): a config n_min must equal the rule result; after a rule failure it is
    accepted only with ``[nmin_study] override_decision``, recorded in the "adopted" block.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import simstudy  # lazy: threads are already pinned (A5e item 5)

    summary = simstudy.run_nmin_study(cfg, config_sha256=config_sha256(args.config))
    try:
        adopted = simstudy.adopted_block(
            summary["rule"]["n_min"],
            cfg.threshold.n_min,
            cfg.nmin_study.override_decision,
            cfg.design.window,
        )
    except ValueError as exc:
        log.error("nmin-study guard failed", extra={"error": str(exc)})
        return 1
    summary["adopted"] = adopted
    out = simstudy.NMIN_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(simstudy.summary_json(summary).encode("utf-8"))
    n_min = None if adopted is None else adopted["n_min"]
    log.info("OK", extra={"outputs": [out.as_posix()], "n_min": n_min})
    return 0


def _cmd_threshold_study(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-7B A3 study (D35): pilot, runtime guard, full study; writes metrics.json and figures.

    Synthetic data only. Logs file names and wall-clock times, never a result. Exit 1,
    writing nothing, if the pilot projects more than ``[simstudy] max_runtime_hours``.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import threshold_study as ts  # lazy: threads are already pinned (A5e item 5)

    workers = ts.default_workers() if args.workers is None else args.workers
    start = time.perf_counter()
    try:
        ts.check_design(cfg)
        pilot_reps = min(ts.PILOT_REPLICATIONS, cfg.simstudy.replications)
        ts.run_records(cfg, pilot_reps, workers)
        pilot = time.perf_counter() - start
        projected = ts.project_seconds(pilot, pilot_reps, cfg.simstudy.replications)
        timing = {"workers": workers, "pilot_seconds": pilot, "projected_seconds": projected}
        if projected > cfg.simstudy.max_runtime_hours * 3600:
            log.error("runtime guard: projection exceeds max_runtime_hours", extra=timing)
            return 1
        log.info("pilot done", extra=timing)
        summary, records = ts.run_study(
            cfg, config_sha256=config_sha256(args.config), workers=workers
        )
    except ValueError as exc:  # messages contain no data values
        log.error("threshold-study failed", extra={"error": str(exc)})
        return 1
    outputs = ts.write_outputs(summary, records, ts.STUDY_DIR)
    log.info(
        "OK",
        extra={
            "outputs": [p.as_posix() for p in outputs],
            "seconds": time.perf_counter() - start,
            **timing,
        },
    )
    return 0


def _cmd_threshold_select(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-7C real selection on the training slice (D36). Logs aggregates only.

    Writes ``reports/threshold_selection/real/`` (selection.json and SVG figures; per-day
    figures under the gitignored ``figures/``). Exit 1 on a rule failure, after the evidence
    is written: nothing is chosen automatically.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import threshold_real as tr

    start = time.perf_counter()
    try:
        summary, outputs = tr.run_selection(cfg, args.config)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (ValueError, OSError) as exc:  # messages contain no data values
        log.error("threshold-select failed", extra={"error": str(exc)})
        return 1
    sel = summary["selection"]
    extra = {
        "outputs": [p.as_posix() for p in outputs],
        "seconds": time.perf_counter() - start,
        "n_train": summary["provenance"]["n_train"],
        "n_candidates": len(summary["candidates"]),
        "q_star": sel["q_star"],
        "q_fs": sel["q_fs"],
        "k_hat": sel["k_hat"],
        "rule_failure": sel["rule_failure"],
        "q_star_at_grid_floor": sel["q_star_at_grid_floor"],
        "split_half_flag": summary["split_half"]["flag"],
    }
    if sel["rule_failure"] is not None:
        log.error("rule failure: evidence written, nothing chosen", extra=extra)
        return 1
    log.info("OK", extra=extra)
    return 0


def _cmd_threshold_select_residuals(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-8A threshold evidence on GARCH residuals of the training slice (D38 (9)).

    Writes ``reports/threshold_selection/real_residuals/`` (selection.json and SVG figures;
    per-day figures under the gitignored ``figures/``). Logs aggregates only. Exit 1 on a
    failed GARCH fit (nothing written) and on a rule failure (evidence written, nothing chosen).

    ``--selected-filter`` (D39 (g)): the DESCRIPTIVE run on the residuals of the BIC-selected
    filter; writes ``selection_filter.json`` and leaves ``selection.json`` alone.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import threshold_residuals as trr
    from niftytail.garch import GarchError

    start = time.perf_counter()
    try:
        summary, outputs = trr.run_selection(cfg, args.config, selected_filter=args.selected_filter)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (ValueError, OSError, GarchError) as exc:  # messages contain no data values
        log.error("threshold-select-residuals failed", extra={"error": str(exc)})
        return 1
    sel = summary["selection"]
    extra = {
        "outputs": [p.as_posix() for p in outputs],
        "seconds": time.perf_counter() - start,
        "n_train": summary["provenance"]["n_train"],
        "n_candidates": len(summary["candidates"]),
        "garch_flags": summary["garch"]["flags"],
        "q_star": sel["q_star"],
        "q_fs": sel["q_fs"],
        "k_hat": sel["k_hat"],
        "rule_failure": sel["rule_failure"],
        "q_star_at_grid_floor": sel["q_star_at_grid_floor"],
        "split_half_flag": summary["split_half"]["flag"],
    }
    if sel["rule_failure"] is not None:
        log.error("rule failure: evidence written, nothing chosen", extra=extra)
        return 1
    log.info("OK", extra=extra)
    return 0


def _cmd_filter_select(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-8A-2 filter specification by training BIC (D39 (c), (e)).

    Writes ``reports/conditional/filter_selection.json`` (aggregates only). Logs the winner
    and the BIC values. Exit 1, writing nothing, if any of the four fits fails.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import filter_selection as fs
    from niftytail.diagnostics import DiagnosticsError
    from niftytail.garch import GarchError

    start = time.perf_counter()
    try:
        summary, path = fs.run_filter_selection(cfg, args.config)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (ValueError, OSError, GarchError, DiagnosticsError) as exc:  # no data values
        log.error("filter-select failed", extra={"error": str(exc)})
        return 1
    log.info(
        "OK",
        extra={
            "outputs": [path.as_posix()],
            "seconds": time.perf_counter() - start,
            "n_train": summary["provenance"]["n_train"],
            "winner": summary["winner"],
            "bic": {spec: c["bic"] for spec, c in summary["candidates"].items()},
        },
    )
    return 0


def _cmd_run(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-11 OOS pass (D44): rolling forecasts + backtests into a commit-keyed run dir.

    Exit 0 done (or verified), 1 refused or mismatch, 2 stop rule (garch_error > 1%).
    Logs aggregates and hashes only, never a date or a per-day value.
    """
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import oos
    from niftytail import threshold_study as ts

    workers = ts.default_workers() if args.workers is None else args.workers
    try:
        return oos.run(cfg, args.config, args.window, workers, args.verify, Path.cwd(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (oos.OosError, ValueError, OSError) as exc:  # messages contain no data values
        log.error("run refused or failed", extra={"error": str(exc)})
        return 1


def _cmd_oos_publish(args: argparse.Namespace, log: logging.Logger) -> int:
    """Publish the OOS aggregates of every window (D44 (g)); writes docs/oos.md."""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import oos  # lazy: threads are already pinned (A5e item 5)

    try:
        outputs = oos.publish(cfg, args.config, Path.cwd())
    except (oos.OosError, OSError, KeyError) as exc:
        log.error("oos-publish refused", extra={"error": str(exc)})
        return 1
    log.info("OK", extra={"outputs": [p.relative_to(Path.cwd()).as_posix() for p in outputs]})
    return 0


def _cmd_oos_blocks(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-11B post-hoc OOS-block threshold check (D45). Logs aggregates and hashes only."""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import oos, oos_blocks
    from niftytail import threshold_study as ts

    workers = ts.default_workers() if args.workers is None else args.workers
    try:
        return oos_blocks.run(cfg, args.config, workers, args.verify, Path.cwd(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (oos.OosError, OSError, KeyError) as exc:  # messages contain no data values
        log.error("oos-blocks refused", extra={"error": str(exc)})
        return 1


def _cmd_oos_blocks_publish(args: argparse.Namespace, log: logging.Logger) -> int:
    """Publish the D45 selection.json and docs/oos_blocks.md."""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import oos, oos_blocks  # lazy: threads are already pinned (A5e item 5)

    try:
        outputs = oos_blocks.publish(cfg, args.config, Path.cwd())
    except (oos.OosError, OSError, KeyError) as exc:
        log.error("oos-blocks-publish refused", extra={"error": str(exc)})
        return 1
    log.info("OK", extra={"outputs": [p.relative_to(Path.cwd()).as_posix() for p in outputs]})
    return 0


def _cmd_xi_path(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-12A rolling xi path with CIs (D47). Logs counts and hashes only."""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import oos, xi_path
    from niftytail import threshold_study as ts

    workers = ts.default_workers() if args.workers is None else args.workers
    try:
        return xi_path.run(cfg, args.config, workers, args.verify, Path.cwd(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (oos.OosError, xi_path.XiPathMismatch, OSError, KeyError) as exc:
        log.error("xi-path refused or STOPPED", extra={"error": str(exc)})
        return 1


def _cmd_xi_path_publish(args: argparse.Namespace, log: logging.Logger) -> int:
    """Publish the D47 summary and docs/xi_path.md."""
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return 1
    from niftytail import oos, xi_path  # lazy: threads are already pinned (A5e item 5)

    try:
        outputs = xi_path.publish(cfg, args.config, Path.cwd())
    except (oos.OosError, OSError, KeyError) as exc:
        log.error("xi-path-publish refused", extra={"error": str(exc)})
        return 1
    log.info("OK", extra={"outputs": [p.relative_to(Path.cwd()).as_posix() for p in outputs]})
    return 0


def _load(args: argparse.Namespace, log: logging.Logger) -> Any:
    try:
        return load_config(args.config)
    except ConfigError as exc:
        log.error(
            "configuration invalid", extra={"config_path": str(args.config), "error": str(exc)}
        )
        return None


def _cmd_validate_tri(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-11C TRI data validation (D48 (a)); writes the TRI manifest once."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import tri

    try:
        return tri.validate_command(cfg, Path.cwd(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1


def _cmd_tri_run(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-11C TRI W = 1000 pass (D48 (b)-(d)). Exit 0, 1 refused/mismatch, 2 stop rule."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import oos, tri
    from niftytail import threshold_study as ts

    workers = ts.default_workers() if args.workers is None else args.workers
    try:
        return tri.run(cfg, args.config, workers, args.verify, Path.cwd(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (oos.OosError, OSError, KeyError) as exc:  # messages contain no data values
        log.error("tri-run refused or STOPPED", extra={"error": str(exc)})
        return 1


def _cmd_tri_publish(args: argparse.Namespace, log: logging.Logger) -> int:
    """Publish the TRI aggregates and docs/oos_tri.md (D48 (d), (e))."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import oos, tri  # lazy: threads are already pinned (A5e item 5)

    try:
        outputs = tri.publish(cfg, args.config, Path.cwd())
    except (oos.OosError, OSError, KeyError) as exc:
        log.error("tri-publish refused", extra={"error": str(exc)})
        return 1
    log.info("OK", extra={"outputs": [p.relative_to(Path.cwd()).as_posix() for p in outputs]})
    return 0


def _cmd_tl_blocks(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-12B traffic-light block table from the published run dirs (D51 (j))."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import oos, tl_blocks  # lazy: threads are already pinned (A5e item 5)

    try:
        return tl_blocks.run(cfg, args.config, args.verify, Path.cwd(), log)
    except (oos.OosError, OSError, KeyError) as exc:  # messages contain no data values
        log.error("tl-blocks refused or STOPPED", extra={"error": str(exc)})
        return 1


def _cmd_figures(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-12B tracked figures from tracked JSON only (D51)."""
    if _load(args, log) is None:
        return 1
    from niftytail import figures, oos  # lazy: threads are already pinned (A5e item 5)

    try:
        return figures.run(args.verify, Path.cwd(), log, args.replace)
    except (oos.OosError, OSError, KeyError) as exc:
        log.error("figures refused or STOPPED", extra={"error": str(exc)})
        return 1


def _cmd_figures_local(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-12C local daily figures, gitignored (D53). Logs no date and no per-day value."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import figures_local, oos  # lazy: threads are already pinned (A5e item 5)

    try:
        return figures_local.run(cfg, args.config, args.verify, Path.cwd(), log)
    except (oos.OosError, OSError, KeyError) as exc:  # messages contain no data values
        log.error("figures-local refused or STOPPED", extra={"error": str(exc)})
        return 1


def _cmd_validate_live(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-13D live-data validation and manifest (D58 (c)). Logs counts only."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import live

    try:
        return live.validate_command(cfg, Path.cwd(), date.today(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1


def _cmd_forecast(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-13D next-day risk forecast (D58 (d)). Logs no date and no per-day value."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import live, oos
    from niftytail import threshold_study as ts

    workers = ts.default_workers() if args.workers is None else args.workers
    try:
        if args.check_run is not None:
            return live.check_run(cfg, args.config, args.check_run, workers, Path.cwd())
        return live.run(cfg, args.config, workers, args.verify, Path.cwd(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except (oos.OosError, ValueError, OSError, KeyError) as exc:  # no data values
        log.error("forecast refused or failed", extra={"error": str(exc)})
        return 1


def _cmd_fetch_api(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-13F end-of-day rows from EODHD (D60). Logs counts only; never the key or the URL."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import data as dl  # lazy: threads are already pinned (A5e item 5)
    from niftytail import eodhd

    try:
        return eodhd.run(cfg, Path.cwd(), log)
    except dl.DataValidationError as exc:
        for e in exc.errors:
            log.error("data error", extra={"error": e})
        return 1
    except eodhd.FetchError as exc:  # its message carries no key, URL or data
        log.error("fetch-api failed: nothing written", extra={"error": str(exc)})
        return 1
    except Exception as exc:  # the class name only, never the message
        log.error("fetch-api failed: nothing written", extra={"error": type(exc).__name__})
        return 1


def _cmd_demo_data(args: argparse.Namespace, log: logging.Logger) -> int:
    """Day-15A synthetic demo for the public dashboard (D62 (b)). Synthetic data only."""
    cfg = _load(args, log)
    if cfg is None:
        return 1
    from niftytail import demo
    from niftytail import threshold_study as ts

    workers = ts.default_workers() if args.workers is None else args.workers
    try:
        return demo.run(cfg, args.config, workers, args.verify, Path.cwd(), log)
    except (ValueError, OSError) as exc:
        log.error("demo-data failed", extra={"error": str(exc)})
        return 1


def _cmd_publish(args: argparse.Namespace, log: logging.Logger) -> int:
    """Publication mirror (D62 (c), D65 (n)); prints counts only and never pushes."""
    import importlib

    try:  # D65 (o): the module is not part of the public copy
        publish = importlib.import_module("niftytail.publish")
    except ModuleNotFoundError:
        log.error("publish is not part of this copy")
        return 1
    try:
        status: int = publish.run(
            Path.cwd(),
            args.mirror,
            review_file=args.review,
            base=args.base,
            message=args.message,
        )
        return status
    except publish.PublishError as exc:  # names the failed step only
        log.error("publish failed", extra={"error": str(exc)})
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m niftytail", description="NIFTYTAIL command-line interface"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check-config", help="load and validate a configuration file")
    p_check.add_argument("config", help="path to a TOML configuration file")
    p_val = sub.add_parser("validate", help="validate the raw data files (D25)")
    p_val.add_argument("config", help="path to a TOML configuration file")
    p_val.add_argument(
        "--init-manifest",
        action="store_true",
        help="create data/raw/MANIFEST.toml first (refuses to overwrite)",
    )
    p_val.add_argument("--retrieved", help="download date YYYY-MM-DD (with --init-manifest)")
    p_val.add_argument("--source-url", help="download page URL (with --init-manifest)")
    p_eda = sub.add_parser("eda", help="Day-4 dependence battery and figures (D29)")
    p_eda.add_argument("config", help="path to a TOML configuration file")
    p_nmin = sub.add_parser("nmin-study", help="Day-6 N_min simulation study (D32)")
    p_nmin.add_argument("config", help="path to a TOML configuration file")
    p_ts = sub.add_parser("threshold-study", help="Day-7B A3 threshold-procedure study (D35)")
    p_ts.add_argument("config", help="path to a TOML configuration file")
    p_ts.add_argument("--workers", type=int, help="worker processes (default: min(cpus - 1, 31))")
    p_sel = sub.add_parser("threshold-select", help="Day-7C real threshold selection (D36)")
    p_sel.add_argument("config", help="path to a TOML configuration file")
    p_res = sub.add_parser(
        "threshold-select-residuals", help="Day-8A threshold evidence on GARCH residuals (D38)"
    )
    p_res.add_argument("config", help="path to a TOML configuration file")
    p_res.add_argument(
        "--selected-filter",
        action="store_true",
        help="descriptive run on the residuals of the BIC-selected filter (D39)",
    )
    p_fs = sub.add_parser("filter-select", help="Day-8A-2 filter specification by BIC (D39)")
    p_fs.add_argument("config", help="path to a TOML configuration file")
    p_run = sub.add_parser("run", help="Day-11 OOS pass: forecasts and backtests (D44)")
    p_run.add_argument("config", help="path to a TOML configuration file")
    p_run.add_argument("--window", type=int, required=True, help="[design] window or a sensitivity")
    p_run.add_argument("--workers", type=int, help="worker processes (default: min(cpus - 1, 31))")
    p_run.add_argument(
        "--verify",
        action="store_true",
        help="recompute in memory and compare hashes; write nothing",
    )
    p_pub = sub.add_parser("oos-publish", help="publish OOS aggregates and docs/oos.md (D44)")
    p_pub.add_argument("config", help="path to a TOML configuration file")
    p_blk = sub.add_parser("oos-blocks", help="Day-11B post-hoc OOS-block threshold check (D45)")
    p_blk.add_argument("config", help="path to a TOML configuration file")
    p_blk.add_argument("--workers", type=int, help="worker processes (default: min(cpus - 1, 31))")
    p_blk.add_argument(
        "--verify",
        action="store_true",
        help="recompute in memory and compare hashes; write nothing",
    )
    p_bpub = sub.add_parser("oos-blocks-publish", help="publish the D45 results and docs")
    p_bpub.add_argument("config", help="path to a TOML configuration file")
    p_xi = sub.add_parser("xi-path", help="Day-12A rolling xi path with profile CIs (D47)")
    p_xi.add_argument("config", help="path to a TOML configuration file")
    p_xi.add_argument("--workers", type=int, help="worker processes (default: min(cpus - 1, 31))")
    p_xi.add_argument(
        "--verify",
        action="store_true",
        help="recompute in memory and compare hashes; write nothing",
    )
    p_xpub = sub.add_parser("xi-path-publish", help="publish the D47 summary and docs")
    p_xpub.add_argument("config", help="path to a TOML configuration file")
    p_vt = sub.add_parser("validate-tri", help="Day-11C TRI data validation (D48)")
    p_vt.add_argument("config", help="path to a TOML configuration file")
    p_tr = sub.add_parser("tri-run", help="Day-11C TRI W = 1000 pass (D48)")
    p_tr.add_argument("config", help="path to a TOML configuration file")
    p_tr.add_argument("--workers", type=int, help="worker processes (default: min(cpus - 1, 31))")
    p_tr.add_argument(
        "--verify",
        action="store_true",
        help="recompute in memory and compare hashes; write nothing",
    )
    p_tp = sub.add_parser("tri-publish", help="publish the TRI aggregates and docs (D48)")
    p_tp.add_argument("config", help="path to a TOML configuration file")
    for name, text in (
        ("tl-blocks", "Day-12B traffic-light block table from the run dirs (D51 (j))"),
        ("figures", "Day-12B tracked figures from tracked JSON (D51)"),
        ("figures-local", "Day-12C local daily figures, gitignored (D53)"),
    ):
        p_new = sub.add_parser(name, help=text)
        p_new.add_argument("config", help="path to a TOML configuration file")
        p_new.add_argument(
            "--verify",
            action="store_true",
            help="recompute in memory and compare hashes; write nothing",
        )
        if name == "figures":  # D54
            p_new.add_argument(
                "--replace",
                action="store_true",
                help="overwrite the committed figures and docs/figures.md (clean tree only)",
            )
    p_vl = sub.add_parser("validate-live", help="Day-13D live-data validation (D58)")
    p_vl.add_argument("config", help="path to a TOML configuration file")
    p_fc = sub.add_parser("forecast", help="Day-13D next-day risk forecast (D58)")
    p_fc.add_argument("config", help="path to a TOML configuration file")
    p_fc.add_argument("--workers", type=int, help="worker processes (default: min(cpus - 1, 31))")
    p_fc.add_argument(
        "--verify",
        action="store_true",
        help="recompute in memory and compare bytes; write nothing",
    )
    p_fc.add_argument(
        "--check-run", help="recompute 20 targets of a run dir and compare (prints k/20 only)"
    )
    p_demo = sub.add_parser("demo-data", help="Day-15A synthetic demo for the public app (D62)")
    p_demo.add_argument("config", help="path to a TOML configuration file")
    p_demo.add_argument("--workers", type=int, help="worker processes (default: min(cpus - 1, 31))")
    p_demo.add_argument(
        "--verify", action="store_true", help="regenerate in memory and compare bytes"
    )
    p_pub2 = sub.add_parser("publish", help="publication mirror (D62, D65); never pushes")
    p_pub2.add_argument("mirror", help="an empty or new directory for the squashed mirror")
    p_pub2.add_argument(
        "--review", help="write the list of rewrites to this file (outside the repo)"
    )
    p_pub2.add_argument("--base", help="the public repository: one commit on top of it")
    p_pub2.add_argument("--message", help="the commit message of the mirror commit")
    p_api = sub.add_parser("fetch-api", help="Day-13F end-of-day rows from EODHD (D60)")
    p_api.add_argument("config", help="path to a TOML configuration file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    args = build_parser().parse_args(argv)
    log = configure_logging()
    if args.command == "check-config":
        return _cmd_check_config(args, log)
    if args.command == "validate":
        return _cmd_validate(args, log)
    if args.command == "eda":
        return _cmd_eda(args, log)
    if args.command == "nmin-study":
        return _cmd_nmin_study(args, log)
    if args.command == "threshold-study":
        return _cmd_threshold_study(args, log)
    if args.command == "threshold-select":
        return _cmd_threshold_select(args, log)
    if args.command == "threshold-select-residuals":
        return _cmd_threshold_select_residuals(args, log)
    if args.command == "filter-select":
        return _cmd_filter_select(args, log)
    if args.command == "run":
        return _cmd_run(args, log)
    if args.command == "oos-publish":
        return _cmd_oos_publish(args, log)
    if args.command == "oos-blocks":
        return _cmd_oos_blocks(args, log)
    if args.command == "oos-blocks-publish":
        return _cmd_oos_blocks_publish(args, log)
    if args.command == "xi-path":
        return _cmd_xi_path(args, log)
    if args.command == "xi-path-publish":
        return _cmd_xi_path_publish(args, log)
    if args.command == "validate-tri":
        return _cmd_validate_tri(args, log)
    if args.command == "tri-run":
        return _cmd_tri_run(args, log)
    if args.command == "tri-publish":
        return _cmd_tri_publish(args, log)
    if args.command == "tl-blocks":
        return _cmd_tl_blocks(args, log)
    if args.command == "figures":
        return _cmd_figures(args, log)
    if args.command == "figures-local":
        return _cmd_figures_local(args, log)
    if args.command == "validate-live":
        return _cmd_validate_live(args, log)
    if args.command == "forecast":
        return _cmd_forecast(args, log)
    if args.command == "publish":
        return _cmd_publish(args, log)
    if args.command == "demo-data":
        return _cmd_demo_data(args, log)
    if args.command == "fetch-api":
        return _cmd_fetch_api(args, log)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
