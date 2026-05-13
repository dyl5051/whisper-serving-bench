#!/usr/bin/env python3
"""Render the 12 v1 publishable-assets PNG tables from v1.md data.

Spec: writeups/v1_publishable_assets/README.md
Output: writeups/v1_publishable_assets/{01..12}_*.png at 2x resolution.

Run: python scripts/render_v1_table_pngs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT_DIR = Path("writeups/v1_publishable_assets")
DPI = 200

HEADER_BG = "#2C3E50"
HEADER_FG = "white"
WINNER_BG = "#D4EDDA"
BROKEN_BG = "#F8D7DA"
MISSING_BG = "#F2F2F2"
GOTCHA_BG = "#FFF3CD"
ALT_ROW_BG = "#FAFBFC"
EDGE = "#DDDDDD"

mpl.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
mpl.rcParams["font.size"] = 11


def render_table(
    filename: str,
    headers: list[str],
    rows: list[list[str]],
    fig_size: tuple[float, float],
    col_widths: list[float] | None = None,
    cell_highlights: dict[tuple[int, int], str] | None = None,
    cell_bold: set[tuple[int, int]] | None = None,
    text_cols: set[int] | None = None,
    row_scale: float = 1.8,
    font_size: int = 11,
) -> None:
    fig, ax = plt.subplots(figsize=fig_size, dpi=DPI)
    ax.axis("off")

    text_cols = text_cols or set()
    n_cols = len(headers)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_scale)

    # Header
    for col_idx in range(n_cols):
        cell = table[(0, col_idx)]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")
        cell.set_edgecolor("white")
        cell.set_height(cell.get_height() * 1.1)

    # Body — alternating rows + special-case OOM / em-dash
    for row_idx in range(len(rows)):
        for col_idx in range(n_cols):
            cell = table[(row_idx + 1, col_idx)]
            cell.set_edgecolor(EDGE)
            cell.set_facecolor("white" if row_idx % 2 == 0 else ALT_ROW_BG)
            text = rows[row_idx][col_idx]
            if text == "OOM":
                cell.set_facecolor(BROKEN_BG)
                cell.set_text_props(color="#721C24", fontweight="bold")
            elif text == "—":
                cell.set_facecolor(MISSING_BG)
                cell.set_text_props(color="#999999")
            # Text columns get left-aligned via padding hack
            if col_idx in text_cols:
                cell.PAD = 0.04
                cell.set_text_props(ha="left")

    # Explicit highlights (overrides OOM coloring if both set)
    if cell_highlights:
        for (r, c), color in cell_highlights.items():
            table[(r + 1, c)].set_facecolor(color)

    if cell_bold:
        for (r, c) in cell_bold:
            cell = table[(r + 1, c)]
            existing_color = cell.get_text().get_color()
            cell.set_text_props(fontweight="bold", color=existing_color)

    plt.savefig(OUT_DIR / filename, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Table 01: decision table by workload shape (4 rows)
# ----------------------------------------------------------------------------
def render_01_decision_table() -> None:
    # 7 columns — one per decision-relevant metric we graded.
    # Excluded: p50/p99 latency (p95 is the operational summary), GPU util
    # (diagnostic), peak GPU memory (captured via OOM walls in tables 02–09).
    gotcha_row = 2  # "no correct option in v1"
    render_table(
        filename="01_decision_table_by_workload_shape.png",
        headers=[
            "Peak load",
            "p95 budget",
            "Deployment",
            "RTF",
            "p95 actual",
            "WER",
            "Cost / audio-hr",
        ],
        rows=[
            ["≤8 concurrent", "Tight (≤2s)", "HF × L4", "0.044", "1.96s", "1.44%", "$0.027"],
            ["≤8 concurrent", "Loose (10s+)", "HF × L4", "0.044", "~14s", "1.44%", "$0.027"],
            ["32–128 concurrent", "Tight (≤2s)", "No correct option in v1", "—", "—", "—", "—"],
            ["32–128 concurrent", "Loose", "HF × L4", "0.044–0.046", "52–209s", "1.44%", "$0.027"],
        ],
        fig_size=(14, 3.8),
        col_widths=[0.15, 0.13, 0.20, 0.11, 0.13, 0.10, 0.14],
        cell_highlights={
            # green for winner deployments
            (0, 2): WINNER_BG,
            (1, 2): WINNER_BG,
            (3, 2): WINNER_BG,
            # amber for the empty-quadrant row across all 7 cells
            (gotcha_row, 0): GOTCHA_BG,
            (gotcha_row, 1): GOTCHA_BG,
            (gotcha_row, 2): GOTCHA_BG,
            (gotcha_row, 3): GOTCHA_BG,
            (gotcha_row, 4): GOTCHA_BG,
            (gotcha_row, 5): GOTCHA_BG,
            (gotcha_row, 6): GOTCHA_BG,
        },
        cell_bold={
            (0, 2), (1, 2), (3, 2),
            (gotcha_row, 2),  # "No correct option in v1"
        },
        text_cols={0, 1, 2},
        row_scale=2.2,
    )


# ----------------------------------------------------------------------------
# Tables 02–09: A100 and L4 matrix tables
# ----------------------------------------------------------------------------
_CONCURRENCY_HEADERS = ["framework", "c=1", "c=8", "c=32", "c=64", "c=128"]
_CONCURRENCY_WIDTHS = [0.20, 0.13, 0.13, 0.13, 0.13, 0.13]


def render_matrix_table(filename: str, title_metric: str, rows: list[list[str]],
                       hf_winner_cols: list[int] | None = None,
                       broken_vllm: bool = False,
                       cheapest_cell: tuple[int, int] | None = None) -> None:
    cell_highlights: dict[tuple[int, int], str] = {}
    cell_bold: set[tuple[int, int]] = set()
    # HF row (index 1) — always bold if marked winner
    if hf_winner_cols is not None:
        for col in hf_winner_cols:
            cell_highlights[(1, col)] = WINNER_BG
            cell_bold.add((1, col))
    # vLLM hallucination row — light red across data cells
    if broken_vllm:
        for col in range(1, 6):
            if rows[2][col] not in ("—", "OOM"):
                cell_highlights[(2, col)] = BROKEN_BG
    # Cheapest correct cell highlight (bold + green)
    if cheapest_cell is not None:
        r, c = cheapest_cell
        cell_highlights[(r, c)] = WINNER_BG
        cell_bold.add((r, c))

    render_table(
        filename=filename,
        headers=_CONCURRENCY_HEADERS,
        rows=rows,
        fig_size=(11, 3.3),
        col_widths=_CONCURRENCY_WIDTHS,
        cell_highlights=cell_highlights,
        cell_bold=cell_bold,
        text_cols={0},
        row_scale=2.0,
    )


def render_02_a100_rtf() -> None:
    render_matrix_table(
        "02_a100_rtf.png", "A100 SXM4-80GB Aggregate RTF (lower is faster)",
        rows=[
            ["faster-whisper", "0.1024", "0.1023", "0.1109", "OOM", "OOM"],
            ["HF", "0.1037", "0.0717", "0.0719", "0.0728", "0.0721"],
            ["vLLM", "0.0103", "0.0035", "0.0030", "0.0035", "0.0032"],
        ],
    )


def render_03_a100_p95() -> None:
    render_matrix_table(
        "03_a100_p95.png", "A100 SXM4-80GB Latency p95 (seconds)",
        rows=[
            ["faster-whisper", "11.21", "60.89", "162.04", "OOM", "OOM"],
            ["HF", "4.46", "22.48", "84.87", "166.89", "327.32"],
            ["vLLM", "0.90", "4.61", "6.99", "11.39", "15.04"],
        ],
    )


def render_04_a100_wer() -> None:
    render_matrix_table(
        "04_a100_wer.png", "A100 SXM4-80GB WER",
        rows=[
            ["faster-whisper", "4.31%", "5.38%", "3.56%", "—", "—"],
            ["HF", "1.44%", "1.44%", "1.44%", "1.44%", "1.44%"],
            ["vLLM", "107.16%", "113.06%", "97.50%", "56.25%", "57.86%"],
        ],
        hf_winner_cols=[1, 2, 3, 4, 5],
        broken_vllm=True,
    )


def render_05_a100_cost() -> None:
    render_matrix_table(
        "05_a100_cost.png", "A100 SXM4-80GB Cost USD per audio-hour",
        rows=[
            ["faster-whisper", "$0.1424", "$0.1422", "$0.1542", "—", "—"],
            ["HF", "$0.1441", "$0.0997", "$0.1000", "$0.1013", "$0.1003"],
            ["vLLM", "$0.0143", "$0.0048", "$0.0042", "$0.0049", "$0.0044"],
        ],
    )


def render_06_l4_rtf() -> None:
    render_matrix_table(
        "06_l4_rtf.png", "L4 24GB Aggregate RTF",
        rows=[
            ["faster-whisper", "0.1673", "0.1751", "OOM", "OOM", "OOM"],
            ["HF", "0.0446", "0.0442", "0.0442", "0.0442", "0.0456"],
            ["vLLM", "0.0101", "0.0161", "0.0070", "0.0080", "OOM"],
        ],
    )


def render_07_l4_p95() -> None:
    render_matrix_table(
        "07_l4_p95.png", "L4 24GB Latency p95 (seconds)",
        rows=[
            ["faster-whisper", "19.86", "107.12", "—", "—", "—"],
            ["HF", "1.96", "13.75", "52.09", "102.11", "209.12"],
            ["vLLM", "0.40", "9.29", "21.39", "24.81", "—"],
        ],
    )


def render_08_l4_wer() -> None:
    render_matrix_table(
        "08_l4_wer.png", "L4 24GB WER",
        rows=[
            ["faster-whisper", "4.03%", "3.68%", "—", "—", "—"],
            ["HF", "1.44%", "1.44%", "1.44%", "1.44%", "1.44%"],
            ["vLLM", "100.64%", "186.15%", "106.98%", "55.97%", "—"],
        ],
        hf_winner_cols=[1, 2, 3, 4, 5],
        broken_vllm=True,
    )


def render_09_l4_cost() -> None:
    render_matrix_table(
        "09_l4_cost.png", "L4 24GB Cost USD per audio-hour",
        rows=[
            ["faster-whisper", "$0.1004", "$0.1050", "—", "—", "—"],
            ["HF", "$0.0267", "$0.0265", "$0.0265", "$0.0265", "$0.0273"],
            ["vLLM", "$0.0061", "$0.0097", "$0.0042", "$0.0048", "—"],
        ],
        cheapest_cell=(1, 2),  # HF c=8 = $0.0265 (cheapest correct cell)
    )


# ----------------------------------------------------------------------------
# Table 10: how to read (worked examples)
# ----------------------------------------------------------------------------
def render_10_how_to_read() -> None:
    render_table(
        filename="10_how_to_read_tables.png",
        headers=["If your p95 budget is…", "Qualifying cells (L4)", "Cheapest qualifier"],
        rows=[
            ["60s", "HF × c≤32,\nfaster-whisper × c=1", "HF × L4 c=8 at $0.0265/audio-hr"],
            ["5s", "HF × c=1 only", "HF × L4 c=1 at $0.0267/audio-hr"],
        ],
        fig_size=(11, 2.5),
        col_widths=[0.18, 0.32, 0.38],
        cell_highlights={(0, 2): WINNER_BG, (1, 2): WINNER_BG},
        cell_bold={(0, 2), (1, 2)},
        text_cols={1, 2},
        row_scale=2.4,
    )


# ----------------------------------------------------------------------------
# Table 11: vLLM hallucination mitigations
# ----------------------------------------------------------------------------
def render_11_vllm_mitigations() -> None:
    render_table(
        filename="11_vllm_hallucination_mitigations.png",
        headers=["Mitigation", "WER", "What it does"],
        rows=[
            [
                "Prompt injection",
                "153%",
                "Model completes the prompt rather than\ntranscribing — WORSE than baseline.",
            ],
            [
                "Post-hoc gzip CR reject (>2.4)",
                "97%",
                "Rejects loop hallucinations, but most outputs\nare well-formed novel sentences.",
            ],
            [
                "silero-vad audio pre-filter",
                "102%",
                "Hallucinations persist even on speech-only audio.\nSilence isn't the root cause.",
            ],
        ],
        fig_size=(11, 3.2),
        col_widths=[0.26, 0.10, 0.50],
        cell_highlights={
            (0, 1): BROKEN_BG,
            (1, 1): BROKEN_BG,
            (2, 1): BROKEN_BG,
        },
        cell_bold={(0, 1), (1, 1), (2, 1)},
        text_cols={0, 2},
        row_scale=2.6,
    )


# ----------------------------------------------------------------------------
# Table 12: managed STT pricing
# ----------------------------------------------------------------------------
def render_12_managed_pricing() -> None:
    render_table(
        filename="12_managed_stt_pricing.png",
        headers=["Service", "Model", "~$/audio-hour", "Notes"],
        rows=[
            ["AWS Transcribe", "proprietary", "~$1.44", "$0.024/min standard;\nbatch discounts at scale"],
            ["Google Cloud Speech-to-Text", "proprietary", "~$0.96", "$0.016/min standard tier"],
            ["Azure Speech", "proprietary", "~$1.00", "~$0.0167/min"],
            ["OpenAI Whisper API", "whisper-1 / large-v2", "~$0.36", "$0.006/min"],
            ["Deepgram", "Nova-3", "~$0.43", "$0.0043/min on prerecorded"],
            ["AssemblyAI", "Universal-2", "~$0.37", "$0.0062/min"],
        ],
        fig_size=(11, 4.5),
        col_widths=[0.28, 0.18, 0.16, 0.30],
        text_cols={0, 1, 3},
        row_scale=2.2,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    renderers = [
        render_01_decision_table,
        render_02_a100_rtf,
        render_03_a100_p95,
        render_04_a100_wer,
        render_05_a100_cost,
        render_06_l4_rtf,
        render_07_l4_p95,
        render_08_l4_wer,
        render_09_l4_cost,
        render_10_how_to_read,
        render_11_vllm_mitigations,
        render_12_managed_pricing,
    ]
    for r in renderers:
        r()
        print(f"  rendered {r.__name__}")
    print(f"[done] {len(renderers)} PNGs at {OUT_DIR}/")


if __name__ == "__main__":
    main()
