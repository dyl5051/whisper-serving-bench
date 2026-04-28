"""Turn results JSONs into the writeup's summary tables and charts.

Reads every results JSON in --results, joins with the cloud pricing in
--pricing, and emits:

    <out>/v1_summary.csv               — one row per cell, all key metrics
    <out>/v1_decision_matrix.md        — framework × concurrency tables (one per GPU)
    <out>/charts/throughput_vs_concurrency.png
    <out>/charts/latency_p95_vs_throughput.png   (Pareto frontier)
    <out>/charts/cost_per_audio_hour.png

Usage:
    python scripts/analyze.py --results results/raw --pricing configs/pricing.yaml --out results/published
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


@click.command()
@click.option(
    "--results",
    "results_dir",
    default="results/raw",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing per-cell results JSONs.",
)
@click.option(
    "--pricing",
    "pricing_path",
    default="configs/pricing.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="GPU pricing YAML.",
)
@click.option(
    "--out",
    "out_dir",
    default="results/published",
    type=click.Path(file_okay=False, path_type=Path),
    help="Where to write tables and charts.",
)
@click.option(
    "--include-degraded",
    is_flag=True,
    help="Include cells with status=degraded in headline summary (default: drop them).",
)
def main(results_dir: Path, pricing_path: Path, out_dir: Path, include_degraded: bool) -> None:
    import pandas as pd
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)

    with pricing_path.open() as f:
        pricing = yaml.safe_load(f)

    cells = _load_cells(results_dir)
    click.echo(f"[analyze] loaded {len(cells)} cells from {results_dir}")
    if not cells:
        click.echo("[analyze] no cells to analyze; exiting", err=True)
        return

    df = pd.DataFrame(cells)
    df = _attach_pricing(df, pricing)

    if not include_degraded:
        before = len(df)
        df = df[df["status"] == "ok"].copy()
        dropped = before - len(df)
        if dropped:
            click.echo(f"[analyze] dropped {dropped} non-ok cells (use --include-degraded to keep)")

    summary_path = out_dir / "v1_summary.csv"
    df.to_csv(summary_path, index=False)
    click.echo(f"[analyze] wrote {summary_path}")

    matrix_path = out_dir / "v1_decision_matrix.md"
    matrix_path.write_text(_render_decision_matrix(df))
    click.echo(f"[analyze] wrote {matrix_path}")

    _plot_throughput_vs_concurrency(df, out_dir / "charts" / "throughput_vs_concurrency.png")
    _plot_latency_pareto(df, out_dir / "charts" / "latency_p95_vs_throughput.png")
    _plot_cost(df, out_dir / "charts" / "cost_per_audio_hour.png")
    click.echo(f"[analyze] wrote charts to {out_dir / 'charts'}")


def _load_cells(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("sweep_"):
            # sweep summaries are not per-cell results
            continue
        try:
            with path.open() as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            click.echo(f"[warn] skipping {path}: malformed JSON ({e})", err=True)
            continue

        cell_config = data.get("cell_config") or {}
        workload = cell_config.get("workload") or {}
        outcome = data.get("outcome") or {}
        latency = data.get("latency") or {}
        rtf = data.get("rtf") or {}
        wer = data.get("wer") or {}
        gpu = data.get("gpu") or {}
        throughput = data.get("throughput") or {}
        provenance = data.get("provenance") or {}

        rows.append(
            {
                "cell_id": cell_config.get("cell_id"),
                "framework": cell_config.get("framework"),
                "model": cell_config.get("model"),
                "gpu_config": cell_config.get("gpu"),  # what the cell asked for
                "gpu_actual": gpu.get("gpu_name"),     # what nvidia-smi reported
                "concurrency": workload.get("concurrency"),
                "iterations": workload.get("iterations"),
                "eval_set": workload.get("eval_set"),
                "status": outcome.get("status"),
                "failure_rate": outcome.get("failure_rate"),
                # metrics
                "latency_p50": latency.get("p50_seconds"),
                "latency_p95": latency.get("p95_seconds"),
                "latency_p99": latency.get("p99_seconds"),
                "rtf_aggregate": rtf.get("aggregate"),
                "rtf_per_request_p50": rtf.get("per_request_p50"),
                "wer": wer.get("wer"),
                "gpu_util_mean": gpu.get("util_mean_percent"),
                "gpu_util_max": gpu.get("util_max_percent"),
                "memory_peak_mib": gpu.get("memory_peak_mib"),
                "audio_seconds_processed": throughput.get("audio_seconds_processed"),
                "wall_seconds": throughput.get("wall_seconds"),
                "requests_per_second": throughput.get("requests_per_second"),
                "git_sha": provenance.get("git_sha"),
                "timestamp_utc": provenance.get("timestamp_utc"),
            }
        )
    return rows


def _attach_pricing(df, pricing: dict) -> Any:
    """Add cost columns: $/audio-hour at on-demand and spot rates."""
    gpus = pricing.get("gpus") or {}

    def lookup(gpu_actual: str | None, key: str) -> float | None:
        if not gpu_actual:
            return None
        # gpu_actual from nvidia-smi looks like "NVIDIA A100-SXM4-80GB" — strip prefix.
        normalized = gpu_actual.replace("NVIDIA ", "").strip()
        rec = gpus.get(normalized) or gpus.get(gpu_actual)
        if rec is None:
            return None
        return rec.get(key)

    df["price_on_demand_usd_per_hour"] = df["gpu_actual"].map(
        lambda g: lookup(g, "on_demand_usd_per_hour")
    )
    df["price_spot_usd_per_hour"] = df["gpu_actual"].map(
        lambda g: lookup(g, "spot_usd_per_hour")
    )

    # Cost per audio-hour transcribed = (GPU $/hr) × (wall_hours / audio_hours)
    #                                 = (GPU $/hr) × RTF_aggregate
    df["cost_usd_per_audio_hour_on_demand"] = (
        df["price_on_demand_usd_per_hour"] * df["rtf_aggregate"]
    )
    df["cost_usd_per_audio_hour_spot"] = (
        df["price_spot_usd_per_hour"] * df["rtf_aggregate"]
    )
    return df


def _render_decision_matrix(df) -> str:
    """Markdown tables: per-GPU framework × concurrency grid."""
    out: list[str] = ["# v1 Decision Matrix\n"]
    for gpu, sub in df.groupby("gpu_actual", dropna=False):
        out.append(f"## {gpu}\n")
        for metric, label, fmt in [
            ("rtf_aggregate", "Aggregate RTF (lower is faster)", "{:.4f}"),
            ("latency_p95", "Latency p95 (seconds)", "{:.3f}"),
            ("wer", "WER", "{:.4f}"),
            ("gpu_util_mean", "GPU util mean (%)", "{:.1f}"),
            ("cost_usd_per_audio_hour_on_demand", "Cost USD/audio-hour (on-demand)", "${:.4f}"),
        ]:
            pivot = sub.pivot_table(
                index="framework",
                columns="concurrency",
                values=metric,
                aggfunc="median",
            )
            if pivot.empty:
                continue
            out.append(f"### {label}\n")
            out.append(_pivot_to_md(pivot, fmt))
            out.append("")
    return "\n".join(out)


def _pivot_to_md(pivot, fmt: str) -> str:
    """Render a pandas pivot table as a markdown table."""
    cols = list(pivot.columns)
    header = "| framework \\ concurrency | " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for fw, row in pivot.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN check
                cells.append("—")
            else:
                cells.append(fmt.format(v))
        lines.append(f"| {fw} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _plot_throughput_vs_concurrency(df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=1,
        ncols=max(1, df["gpu_actual"].nunique()),
        figsize=(6 * max(1, df["gpu_actual"].nunique()), 5),
        squeeze=False,
    )
    for ax, (gpu, sub) in zip(axes.flat, df.groupby("gpu_actual", dropna=False)):
        for fw, fw_sub in sub.groupby("framework"):
            fw_sorted = fw_sub.sort_values("concurrency")
            ax.plot(
                fw_sorted["concurrency"],
                fw_sorted["requests_per_second"],
                marker="o",
                label=fw,
            )
        ax.set_xscale("log", base=2)
        ax.set_title(f"Throughput vs concurrency — {gpu}")
        ax.set_xlabel("concurrency (log2)")
        ax.set_ylabel("requests / second")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_latency_pareto(df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"hf": "o", "faster_whisper": "s", "vllm": "^", "ray_serve": "D", "triton": "P"}
    for (fw, gpu), sub in df.groupby(["framework", "gpu_actual"], dropna=False):
        ax.scatter(
            sub["requests_per_second"],
            sub["latency_p95"],
            label=f"{fw} / {gpu}",
            marker=markers.get(fw, "o"),
            s=80,
            alpha=0.8,
        )
    ax.set_xlabel("Throughput (requests/second)")
    ax.set_ylabel("Latency p95 (seconds)")
    ax.set_title("Latency vs throughput Pareto frontier")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cost(df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("cost_usd_per_audio_hour_on_demand")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df_sorted))))
    ax.barh(
        df_sorted["cell_id"],
        df_sorted["cost_usd_per_audio_hour_on_demand"],
    )
    ax.set_xlabel("USD per audio-hour transcribed (on-demand pricing)")
    ax.set_title("Cost-per-audio-hour, on-demand pricing")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
