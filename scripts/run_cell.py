"""Run one benchmark cell from a YAML config and write results JSON.

Usage:
    python scripts/run_cell.py configs/cells/example_hf_a100_c8.yaml \\
        --results-dir results/raw \\
        --data-root data
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from adapters import build_adapter
from bench.config import CellConfig
from bench.harness import run_cell


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--results-dir",
    default="results/raw",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write the results JSON into.",
)
@click.option(
    "--data-root",
    default="data",
    type=click.Path(file_okay=False, path_type=Path),
    help="Where prepare_data.py wrote the eval sets.",
)
def main(config_path: Path, results_dir: Path, data_root: Path) -> None:
    config = CellConfig.from_yaml(config_path)
    asyncio.run(_run(config, results_dir, data_root))


async def _run(config: CellConfig, results_dir: Path, data_root: Path) -> None:
    adapter = build_adapter(config)
    print(f"[run_cell] config: {config.cell_id}")

    try:
        await adapter.setup()
    except Exception as e:  # noqa: BLE001
        print(f"[run_cell] adapter setup failed: {type(e).__name__}: {e}", file=sys.stderr)
        # Even on setup failure we want to write a results JSON marking the cell
        # as 'error' so the sweep tracker knows we tried it. Build a minimal one.
        from bench.results import CellOutcome, ProvenanceInfo, ResultsJson

        results = ResultsJson(
            provenance=ProvenanceInfo.capture(),
            cell_config=config.model_dump(mode="json"),
            outcome=CellOutcome(
                status="error",
                failure_rate=1.0,
                notes=f"setup failed: {type(e).__name__}: {e}",
            ),
        )
        results.write(results_dir / f"{config.cell_id}.json")
        sys.exit(1)

    try:
        results = await run_cell(config, adapter, data_root=data_root)
        out_path = results_dir / f"{config.cell_id}.json"
        results.write(out_path)
        print(f"[run_cell] wrote {out_path}")
        if results.outcome.status not in ("ok", "degraded"):
            sys.exit(2)
    finally:
        try:
            await adapter.aclose()
        except Exception as e:  # noqa: BLE001
            print(f"[run_cell] aclose failed (non-fatal): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
