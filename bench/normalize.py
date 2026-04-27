"""Text normalization for WER computation.

We use OpenAI's EnglishTextNormalizer (shipped with the openai-whisper package)
because it's the de facto standard for Whisper WER reporting. Normalizing both
the hypothesis and the reference with the same function is critical: different
serving frameworks emit transcripts with different default punctuation, casing,
and number formatting, and unnormalized comparisons will show fake WER deltas
between frameworks that are actually identical at the model level.

If you need to compare these numbers to other published Whisper benchmarks,
make sure they also used this normalizer. Many don't, which is why their
numbers are often slightly worse than ours.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_normalizer():
    """Return a cached singleton EnglishTextNormalizer.

    Imported lazily because openai-whisper is a heavy import (pulls torch,
    triggers CUDA detection) and not every code path needs it.
    """
    from whisper.normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    """Apply Whisper's English text normalizer.

    Lowercases, strips punctuation, expands contractions, normalizes numbers,
    collapses whitespace. Safe to call on already-normalized text (idempotent).
    """
    if not text:
        return ""
    return get_normalizer()(text).strip()
