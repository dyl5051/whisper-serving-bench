"""WER (Word Error Rate) computation.

Wraps jiwer.wer with our normalization applied to both sides. We intentionally
do NOT expose a "raw WER" function — every WER value in the benchmark must be
post-normalization to be comparable across frameworks. The pre-normalization
text is preserved in the per-request log if you need to inspect it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from bench.normalize import normalize


@dataclass(frozen=True)
class WerBreakdown:
    """Per-corpus WER + the underlying edit-distance counts.

    Reporting substitutions/insertions/deletions separately is useful for
    diagnosing framework-specific failure modes (e.g. one framework hallucinating
    repeated words shows up as high insertions; another truncating long outputs
    shows up as high deletions).
    """

    wer: float
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    reference_word_count: int


def compute_wer(references: list[str], hypotheses: list[str]) -> WerBreakdown:
    """Compute corpus-level WER after normalization.

    Args:
        references: ground-truth transcripts, one per clip.
        hypotheses: model outputs, one per clip, in the same order as references.

    Returns:
        WerBreakdown with the rate plus the underlying edit operation counts.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must be same length: "
            f"{len(references)} vs {len(hypotheses)}"
        )
    if not references:
        raise ValueError("Cannot compute WER on an empty corpus.")

    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]

    # Filter out clips where the normalized reference is empty — these would
    # divide-by-zero and aren't meaningful WER samples anyway.
    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps, strict=True) if r]
    if not pairs:
        raise ValueError("All references normalized to empty strings — eval set is broken.")

    refs, hyps = zip(*pairs, strict=True)

    output = jiwer.process_words(list(refs), list(hyps))

    return WerBreakdown(
        wer=output.wer,
        substitutions=output.substitutions,
        insertions=output.insertions,
        deletions=output.deletions,
        hits=output.hits,
        reference_word_count=sum(len(r.split()) for r in refs),
    )
