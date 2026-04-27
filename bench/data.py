"""Eval set loading.

An EvalSet is a list of AudioClip records. Each clip has an audio file path, a
reference transcript, and the audio duration in seconds (needed for RTF).

EvalSets are described in configs/eval_sets.yaml and materialized to disk by
scripts/prepare_data.py. At benchmark runtime we just load the manifest and
hand AudioClips to the adapter — we never re-download or re-slice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AudioClip:
    """A single audio file with a reference transcript.

    Attributes:
        clip_id: Stable identifier for results joining (e.g. "tedlium_AaronHuey_2010_chunk03").
        audio_path: Absolute path to the audio file on disk (WAV/FLAC/MP3 — adapter handles).
        duration_seconds: True audio length, used for RTF computation.
        reference_text: Ground-truth transcript, pre-normalization.
    """

    clip_id: str
    audio_path: Path
    duration_seconds: float
    reference_text: str


@dataclass(frozen=True)
class EvalSet:
    name: str
    description: str
    clips: list[AudioClip]

    @property
    def total_audio_seconds(self) -> float:
        return sum(c.duration_seconds for c in self.clips)

    def __len__(self) -> int:
        return len(self.clips)


def load_eval_set(name: str, data_root: Path = Path("data")) -> EvalSet:
    """Load an eval set by name.

    Looks for data_root/{name}/manifest.jsonl, where each line is a JSON record:
        {"clip_id": ..., "audio_path": ..., "duration_seconds": ..., "reference_text": ...}

    Audio paths in the manifest are relative to the manifest's directory.
    """
    manifest_path = data_root / name / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Eval set manifest not found: {manifest_path}. "
            f"Run `python scripts/prepare_data.py --eval-set {name}` to generate it."
        )

    description_path = data_root / name / "DESCRIPTION.txt"
    description = description_path.read_text().strip() if description_path.exists() else ""

    clips: list[AudioClip] = []
    with manifest_path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                clips.append(
                    AudioClip(
                        clip_id=record["clip_id"],
                        audio_path=manifest_path.parent / record["audio_path"],
                        duration_seconds=float(record["duration_seconds"]),
                        reference_text=record["reference_text"],
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise ValueError(
                    f"Malformed manifest line {line_num} in {manifest_path}: {e}"
                ) from e

    if not clips:
        raise ValueError(f"Eval set {name!r} loaded from {manifest_path} is empty.")

    return EvalSet(name=name, description=description, clips=clips)


def load_eval_set_definitions(path: Path = Path("configs/eval_sets.yaml")) -> dict[str, dict]:
    """Load the catalog of available eval sets and their preparation parameters.

    The catalog is consumed by scripts/prepare_data.py to know what to download
    and how to slice it. The benchmark harness itself doesn't read this — it
    only consumes prepared manifests via load_eval_set().
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval set catalog not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}
