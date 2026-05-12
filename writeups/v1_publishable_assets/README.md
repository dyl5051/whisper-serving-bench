# v1 publishable assets

What you need to render before posting `writeups/v1.md` to Substack (or any other host that doesn't render markdown tables natively).

This directory holds the rendered PNGs once they exist. Generate or update them before publishing v1; replicate this dir for v1.1/v2 etc.

## Source of truth

**`writeups/v1.md` on the latest commit is canonical.** Specifically, the file's tables are the ground truth — `results/published/v1_partial/v1_decision_matrix.md` is older and missing three of the tables the post needs.

## The 12 tables to render

Numbered for filename consistency. All must come from the **current `writeups/v1.md`**, not from `v1_decision_matrix.md`.

| # | filename | section in v1.md | source |
|---|---|---|---|
| 01 | `decision_table_by_workload_shape.png` | "The 30-second answer" | 4-row workload-shape table |
| 02 | `a100_rtf.png` | "The decision matrix" → "A100 SXM4-80GB" | RTF table |
| 03 | `a100_p95.png` | same | Latency p95 table |
| 04 | `a100_wer.png` | same | WER table |
| 05 | `a100_cost.png` | same | Cost USD/audio-hour table |
| 06 | `l4_rtf.png` | "The decision matrix" → "L4 24GB" | RTF table |
| 07 | `l4_p95.png` | same | Latency p95 table |
| 08 | `l4_wer.png` | same | WER table |
| 09 | `l4_cost.png` | same | Cost USD/audio-hour table |
| 10 | `how_to_read_tables.png` | "The decision matrix" tail | 2-row worked-example table (budget/qualifying/cheapest) |
| 11 | `vllm_hallucination_mitigations.png` | "Finding 1" | 3-row mitigation table (prompt / crfilter / vad) |
| 12 | `managed_stt_pricing.png` | "Should you self-host at all?" | 6-row managed-services pricing table |

**Tables 01, 10, 11 are new since `v1_decision_matrix.md` was generated.** If you're rendering from `whisper_bench_tables.html`, verify those three are in the HTML before relying on it; if not, hand-render them from `v1.md`.

## Styling spec

Consistent across all 12 PNGs:

- **Font:** sans-serif. Inter, Helvetica Neue, or system-ui. No Times New Roman.
- **Sizing:** render at 2× resolution (1440px wide target for ~720px Substack inline display). Retina-friendly.
- **Header row:** distinct from data rows — bolded text, slightly darker background, or a divider line.
- **Bolded cells:** preserve markdown bolding. Specifically:
  - `1.44%` WER cells (all 10 of them in the HF row, both GPUs) bolded
  - `$0.0265` cell in the L4 cost table bolded (the cheapest correct cell)
  - "No correct option in v1" cell in the decision table bolded
- **Numeric alignment:** right-aligned in numeric columns, left-aligned in label/text columns.
- **No gridlines or chartjunk.** Clean table styling; let the numbers speak.
- **Color:** monochrome OK. If using color, single-color highlighting on the bolded cells (light green for "winner," light red for "broken/OOM"). Avoid rainbow-colored full tables.
- **Whitespace:** ~12-16px padding inside cells. Don't crowd.

## Aspect ratio guidance

The 5-column-by-N-row tables (concurrency tables) work best as wide rectangles. The 4-row workload-shape table is closer to square. The 6-row pricing table can be taller than wide. Don't force a uniform aspect ratio — let each table find its natural shape.

## Rendering paths (choose one)

In rough order of fidelity:

1. **Screenshot from `whisper_bench_tables.html` in a browser** at 2× zoom. Highest fidelity if the HTML was rendered cleanly. Caveat: verify all 12 tables are in the HTML; the three new ones (01, 10, 11) may not be.

2. **Headless Chromium with Puppeteer / Playwright** — same input, fully programmatic. Best if rendering will be repeated for v1.1+.

3. **Matplotlib `pyplot.table()` from data in `whisper_bench_tables.xlsx`** — most portable but typically uglier than HTML+browser. Last resort.

4. **Datawrapper** (datawrapper.de) for interactive tables that don't need to be PNGs — but Substack only embeds images so this defeats the purpose.

## Distribution checklist for v1 release

Tracking what needs to be done before posting (carry over from `docs/WRITEUP_OUTLINE.md`'s editorial + distribution checklists):

### Pre-publish

- [ ] All 12 PNGs rendered and dropped into `writeups/v1_publishable_assets/`
- [ ] Every PNG's numbers match `writeups/v1.md` at the publish-commit SHA
- [ ] Reproduction recipe in README has been verified by a stranger (DM one infra friend before launch)
- [ ] No claim of the form "X is faster than Y" without citing specific cells + concurrency level
- [ ] Title and opening paragraph fit the screenshot a CTO would tweet
- [ ] Author byline + publish date added to the Substack version (not part of `v1.md`)

### Publish

- [ ] Primary post on personal blog or Substack with all 12 PNGs inline
- [ ] LinkedIn post: use `writeups/v1_post.md` content, link to primary
- [ ] HackerNews submission: link to primary (Tuesday morning Pacific is the standard slot)
- [ ] Tweet thread queued: headline + 3-4 findings + repo link + 2-3 attached PNGs (decision table + WER table + vLLM mitigation table)
- [ ] /r/MachineLearning crosspost (read their rules first)
- [ ] /r/LocalLLaMA crosspost
- [ ] DM vLLM Whisper integration maintainer a few hours before public post
- [ ] DM faster-whisper maintainers if any bug is implied
- [ ] LinkedIn second post a few days after, with a chart screenshot

### Post-publish

- [ ] Monitor GitHub issues for reproduction reports
- [ ] If any number is wrong in a way that changes a finding, edit + add a correction note (don't silently amend)
- [ ] Update this checklist with v1.1 carryover lessons before starting v1.1

## Reuse for v1.1+

When generating assets for v1.1 (Triton + TensorRT-LLM) or later releases, duplicate this directory:

```
writeups/v1_1_publishable_assets/
  README.md  (copy this file, update the table list)
  *.png
```

Numbering convention (01, 02, ...) makes alphabetical ordering match table order in the post — easier to spot a missing one.
