"""Run a sweep of benchmark cells sequentially, produce a summary.

A sweep is a YAML file listing cell config paths. Cells are run one at a time
as subprocesses, which gives us clean adapter setup/teardown per cell — no
risk of leftover GPU state polluting the next cell's measurement, and a
catastrophic crash in one cell doesn't take the sweep down with it.

Usage:
    python scripts/run_sweep.py configs/sweeps/v1_faster_whisper.yaml \\
        --data-root /workspace/data \\
        --results-dir /workspace/data/results

Resumability: cells whose results JSON already exists are skipped. Use --force
to re-run everything. Run logs land in <results-dir>/logs/<cell_id>.log so
you can see what each cell printed even after the cell process is gone.

Sweep YAML schema:
    name: <str>                 # display name, used in summary header
    description: <str>          # optional human-readable
    cells: [<path>, ...]        # explicit list of cell config paths
    cells_glob: <glob pattern>  # alternative: glob to expand (e.g. configs/cells/v1_fw_*.yaml)

Either `cells` or `cells_glob` must be set; if both are set, `cells` wins.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import click
import yaml


@dataclass
class CellRun:
    cell_config_path: Path
    cell_id: str
    status: str  # "ok" | "degraded" | "error" | "unsupported" | "skipped" | "subprocess_error"
    duration_seconds: float = 0.0
    log_path: Path | None = None
    results_path: Path | None = None
    # Pulled from the per-cell results JSON if available
    rtf_aggregate: float | None = None
    wer: float | None = None
    latency_p95: float | None = None
    gpu_util_mean: float | None = None
    failure_rate: float | None = None
    error_summary: str = ""


@dataclass
class SweepResult:
    sweep_name: str
    sweep_description: str
    started_at_iso: str
    finished_at_iso: str = ""
    total_seconds: float = 0.0
    cells: list[CellRun] = field(default_factory=list)


@click.command()
@click.argument("sweep_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--data-root",
    default="data",
    type=click.Path(file_okay=False, path_type=Path),
    help="Where eval sets are materialized.",
)
@click.option(
    "--results-dir",
    default="results/raw",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for per-cell results JSONs.",
)
@click.option("--force", is_flag=True, help="Re-run cells even if their results JSON exists.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="List the cells that would run, then exit. Useful for sanity-checking a sweep YAML.",
)
def main(sweep_path: Path, data_root: Path, results_dir: Path, force: bool, dry_run: bool) -> None:
    sweep = _load_sweep(sweep_path)
    cell_paths = _resolve_cells(sweep, sweep_path.parent)

    click.echo(f"[sweep] {sweep['name']}: {len(cell_paths)} cells")
    if sweep.get("description"):
        click.echo(f"[sweep] {sweep['description']}")
    for p in cell_paths:
        click.echo(f"  - {p}")

    if dry_run:
        click.echo("[sweep] dry-run mode: not executing.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_started = time.time()
    sweep_started_iso = _iso_now(sweep_started)
    sweep_result = SweepResult(
        sweep_name=sweep["name"],
        sweep_description=sweep.get("description", ""),
        started_at_iso=sweep_started_iso,
    )

    for i, cell_path in enumerate(cell_paths, start=1):
        cell_id = _peek_cell_id(cell_path)
        results_json = results_dir / f"{cell_id}.json"
        click.echo(f"\n[sweep] [{i}/{len(cell_paths)}] {cell_id}")

        if results_json.exists() and not force:
            click.echo(f"  -> skipping (results already at {results_json}; use --force to re-run)")
            run = CellRun(
                cell_config_path=cell_path,
                cell_id=cell_id,
                status="skipped",
                results_path=results_json,
            )
            _hydrate_from_results(run, results_json)
            sweep_result.cells.append(run)
            continue

        log_path = log_dir / f"{cell_id}.log"
        run = _run_one_cell(
            cell_path=cell_path,
            cell_id=cell_id,
            data_root=data_root,
            results_dir=results_dir,
            log_path=log_path,
        )
        if run.results_path and run.results_path.exists():
            _hydrate_from_results(run, run.results_path)
        sweep_result.cells.append(run)

        click.echo(
            f"  -> status={run.status} duration={run.duration_seconds:.1f}s"
            + (f" rtf={run.rtf_aggregate:.4f}" if run.rtf_aggregate is not None else "")
            + (f" wer={run.wer:.4f}" if run.wer is not None else "")
        )

    sweep_finished = time.time()
    sweep_result.finished_at_iso = _iso_now(sweep_finished)
    sweep_result.total_seconds = sweep_finished - sweep_started

    _print_summary(sweep_result)
    summary_path = results_dir / f"sweep_{sweep['name']}_{int(sweep_started)}.json"
    _write_summary_json(sweep_result, summary_path)
    click.echo(f"\n[sweep] summary written to {summary_path}")

    # Exit non-zero if any cell errored, so the sweep can be a CI step.
    if any(c.status in ("error", "subprocess_error") for c in sweep_result.cells):
        sys.exit(1)


def _load_sweep(sweep_path: Path) -> dict:
    with sweep_path.open() as f:
        sweep = yaml.safe_load(f)
    if not isinstance(sweep, dict):
        raise click.ClickException(f"sweep YAML must be a mapping; got {type(sweep).__name__}")
    if "name" not in sweep:
        raise click.ClickException("sweep YAML must have a 'name' field")
    if "cells" not in sweep and "cells_glob" not in sweep:
        raise click.ClickException("sweep YAML must have 'cells' or 'cells_glob'")
    return sweep


def _resolve_cells(sweep: dict, sweep_dir: Path) -> list[Path]:
    if "cells" in sweep:
        # Paths are relative to repo root, not to the sweep YAML's location.
        return [Path(p) for p in sweep["cells"]]
    pattern = sweep["cells_glob"]
    matched = sorted(glob(pattern, recursive=True))
    if not matched:
        raise click.ClickException(f"cells_glob {pattern!r} matched zero files")
    return [Path(p) for p in matched]


def _peek_cell_id(cell_path: Path) -> str:
    """Pull cell_id from the YAML without instantiating the full Pydantic model.

    Avoids importing the bench package (and thus torch) just to plan the sweep.
    """
    if not cell_path.exists():
        raise click.ClickException(f"cell config not found: {cell_path}")
    with cell_path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "cell_id" not in raw:
        raise click.ClickException(f"cell config {cell_path} missing 'cell_id'")
    return raw["cell_id"]


def _run_one_cell(
    cell_path: Path,
    cell_id: str,
    data_root: Path,
    results_dir: Path,
    log_path: Path,
) -> CellRun:
    cmd = [
        sys.executable,
        "scripts/run_cell.py",
        str(cell_path),
        "--data-root",
        str(data_root),
        "--results-dir",
        str(results_dir),
    ]
    start = time.time()
    with log_path.open("w") as logf:
        logf.write(f"# command: {' '.join(cmd)}\n")
        logf.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except Exception as e:  # noqa: BLE001
            duration = time.time() - start
            return CellRun(
                cell_config_path=cell_path,
                cell_id=cell_id,
                status="subprocess_error",
                duration_seconds=duration,
                log_path=log_path,
                error_summary=f"{type(e).__name__}: {e}",
            )
    duration = time.time() - start

    results_json = results_dir / f"{cell_id}.json"
    # run_cell.py exits 0 on ok/degraded, 1 on setup error, 2 on cell error.
    # We don't trust the exit code alone — read the JSON if it exists.
    if results_json.exists():
        return CellRun(
            cell_config_path=cell_path,
            cell_id=cell_id,
            status="ok",  # placeholder, _hydrate_from_results overrides
            duration_seconds=duration,
            log_path=log_path,
            results_path=results_json,
        )
    return CellRun(
        cell_config_path=cell_path,
        cell_id=cell_id,
        status="subprocess_error",
        duration_seconds=duration,
        log_path=log_path,
        error_summary=f"run_cell.py exited {proc.returncode} with no results JSON; see {log_path}",
    )


def _hydrate_from_results(run: CellRun, results_json: Path) -> None:
    """Fill in run.status, .rtf, .wer, .latency_p95, .gpu_util_mean from the results JSON."""
    try:
        with results_json.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        run.error_summary = f"could not read results JSON: {e}"
        return

    outcome = data.get("outcome") or {}
    run.status = outcome.get("status") or run.status
    run.failure_rate = outcome.get("failure_rate")
    if outcome.get("notes"):
        run.error_summary = outcome["notes"]

    rtf = data.get("rtf") or {}
    run.rtf_aggregate = rtf.get("aggregate")

    wer = data.get("wer") or {}
    run.wer = wer.get("wer")

    latency = data.get("latency") or {}
    run.latency_p95 = latency.get("p95_seconds")

    gpu = data.get("gpu") or {}
    run.gpu_util_mean = gpu.get("util_mean_percent")


def _print_summary(sweep: SweepResult) -> None:
    click.echo("\n" + "=" * 80)
    click.echo(f"SWEEP SUMMARY: {sweep.sweep_name}")
    click.echo(f"Total: {sweep.total_seconds:.1f}s across {len(sweep.cells)} cells")
    click.echo("=" * 80)

    header = f"{'cell_id':<40} {'status':<12} {'dur':>7} {'rtf':>8} {'wer':>7} {'p95(s)':>7} {'gpu%':>5}"
    click.echo(header)
    click.echo("-" * len(header))
    for c in sweep.cells:
        click.echo(
            f"{c.cell_id:<40} "
            f"{c.status:<12} "
            f"{c.duration_seconds:>7.1f} "
            f"{(f'{c.rtf_aggregate:.4f}' if c.rtf_aggregate is not None else '-'):>8} "
            f"{(f'{c.wer:.4f}' if c.wer is not None else '-'):>7} "
            f"{(f'{c.latency_p95:.2f}' if c.latency_p95 is not None else '-'):>7} "
            f"{(f'{c.gpu_util_mean:.1f}' if c.gpu_util_mean is not None else '-'):>5}"
        )

    n_ok = sum(1 for c in sweep.cells if c.status == "ok")
    n_degraded = sum(1 for c in sweep.cells if c.status == "degraded")
    n_skipped = sum(1 for c in sweep.cells if c.status == "skipped")
    n_failed = sum(
        1 for c in sweep.cells if c.status in ("error", "subprocess_error", "unsupported")
    )
    click.echo("-" * len(header))
    click.echo(f"ok={n_ok} degraded={n_degraded} skipped={n_skipped} failed={n_failed}")


def _write_summary_json(sweep: SweepResult, path: Path) -> None:
    payload = {
        "sweep_name": sweep.sweep_name,
        "sweep_description": sweep.sweep_description,
        "started_at_iso": sweep.started_at_iso,
        "finished_at_iso": sweep.finished_at_iso,
        "total_seconds": sweep.total_seconds,
        "cells": [
            {
                "cell_id": c.cell_id,
                "cell_config_path": str(c.cell_config_path),
                "status": c.status,
                "duration_seconds": c.duration_seconds,
                "log_path": str(c.log_path) if c.log_path else None,
                "results_path": str(c.results_path) if c.results_path else None,
                "rtf_aggregate": c.rtf_aggregate,
                "wer": c.wer,
                "latency_p95": c.latency_p95,
                "gpu_util_mean": c.gpu_util_mean,
                "failure_rate": c.failure_rate,
                "error_summary": c.error_summary,
            }
            for c in sweep.cells
        ],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _iso_now(epoch_seconds: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


if __name__ == "__main__":
    main()
