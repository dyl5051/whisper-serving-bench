"""Download and materialize eval sets to local disk.

Reads configs/eval_sets.yaml and produces, for each eval set:
    data/<name>/manifest.jsonl        # one record per clip
    data/<name>/clips/*.wav           # the audio files
    data/<name>/DESCRIPTION.txt       # human-readable description

Idempotent: skips eval sets whose manifest already exists. Use --force to
re-materialize.

Run inside the Docker image (which has datasets, soundfile, librosa installed):
    docker run --rm -v $(pwd)/data:/app/data whisper-bench \\
        python scripts/prepare_data.py --eval-set librispeech_test_clean_subset
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from bench.data import load_eval_set_definitions


@click.command()
@click.option(
    "--eval-set",
    "eval_set_names",
    multiple=True,
    help="Name of eval set(s) to prepare. Repeat for multiple. Default: all.",
)
@click.option(
    "--data-root",
    default="data",
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory for materialized eval sets.",
)
@click.option(
    "--catalog",
    default="configs/eval_sets.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the eval set catalog YAML.",
)
@click.option("--force", is_flag=True, help="Re-materialize even if manifest already exists.")
def main(
    eval_set_names: tuple[str, ...],
    data_root: Path,
    catalog: Path,
    force: bool,
) -> None:
    catalog_data = load_eval_set_definitions(catalog)
    targets = list(eval_set_names) if eval_set_names else list(catalog_data.keys())

    for name in targets:
        if name not in catalog_data:
            click.echo(f"[skip] {name}: not in catalog", err=True)
            continue
        spec = catalog_data[name]
        out_dir = data_root / name
        manifest_path = out_dir / "manifest.jsonl"

        if manifest_path.exists() and not force:
            click.echo(f"[skip] {name}: manifest exists at {manifest_path} (use --force to rebuild)")
            continue

        click.echo(f"[prepare] {name} (source={spec['source']})")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "DESCRIPTION.txt").write_text(spec.get("description", "").strip() + "\n")

        if spec["source"] == "huggingface_datasets":
            _prepare_hf_dataset(spec, out_dir)
        elif spec["source"] == "huggingface_datasets_sliced":
            _prepare_hf_dataset_sliced(spec, out_dir)
        else:
            click.echo(f"[error] {name}: unknown source {spec['source']!r}", err=True)
            continue

        click.echo(f"[done] {name}: manifest written to {manifest_path}")


def _prepare_hf_dataset(spec: dict, out_dir: Path) -> None:
    """Materialize a HuggingFace dataset directly, one clip per row."""
    import soundfile as sf
    from datasets import load_dataset

    ds = load_dataset(
        spec["dataset"],
        spec.get("config"),
        split=spec["split"],
        trust_remote_code=True,
    )

    limit = spec.get("limit")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    audio_col = spec["audio_column"]
    text_col = spec["text_column"]
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as mf:
        for i, row in enumerate(ds):
            audio_data = row[audio_col]
            # HF datasets returns audio as {"array": np.ndarray, "sampling_rate": int, "path": str}
            arr = audio_data["array"]
            sr = audio_data["sampling_rate"]
            duration = len(arr) / sr
            clip_id = f"{i:06d}"
            wav_path = clips_dir / f"{clip_id}.wav"
            sf.write(wav_path, arr, sr, subtype="PCM_16")
            mf.write(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "audio_path": f"clips/{clip_id}.wav",
                        "duration_seconds": duration,
                        "reference_text": row[text_col],
                    }
                )
                + "\n"
            )


def _prepare_hf_dataset_sliced(spec: dict, out_dir: Path) -> None:
    """Materialize a dataset by slicing long-form audio into fixed-length chunks.

    Used for the streaming workload where we want uniform-length inputs. Picks
    talks from the dataset that exceed source_audio_seconds_min, slices each
    into chunk_seconds chunks, and stops when num_chunks chunks are produced.

    Reference transcripts: HF's TED-LIUM provides per-utterance transcripts but
    the audio is per-talk. To get the reference for a 30-second slice, we'd
    need to align word timestamps. For v1, we punt on this: the slice's
    "reference_text" is the empty string, which means WER for the streaming
    workload will be unmeasurable — but throughput/latency are the headline
    metrics anyway. WER in v1 is reported from the librispeech_test_clean cells.

    A future improvement is to use the per-utterance segments directly (which
    do have transcripts) and concatenate adjacent utterances until ~30s. That's
    a v2 task.
    """
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    ds = load_dataset(
        spec["dataset"],
        spec.get("config"),
        split=spec["split"],
        trust_remote_code=True,
    )

    chunk_seconds = int(spec.get("chunk_seconds", 30))
    num_chunks = int(spec.get("num_chunks", 50))
    min_seconds = float(spec.get("source_audio_seconds_min", 60))
    audio_col = spec["audio_column"]

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    chunks_written = 0
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as mf:
        for row in ds:
            if chunks_written >= num_chunks:
                break
            audio_data = row[audio_col]
            arr = np.asarray(audio_data["array"])
            sr = audio_data["sampling_rate"]
            total_seconds = len(arr) / sr
            if total_seconds < min_seconds:
                continue

            samples_per_chunk = int(chunk_seconds * sr)
            n_chunks_in_talk = len(arr) // samples_per_chunk
            for chunk_idx in range(n_chunks_in_talk):
                if chunks_written >= num_chunks:
                    break
                start = chunk_idx * samples_per_chunk
                end = start + samples_per_chunk
                chunk = arr[start:end]
                clip_id = f"chunk_{chunks_written:04d}"
                wav_path = clips_dir / f"{clip_id}.wav"
                sf.write(wav_path, chunk, sr, subtype="PCM_16")
                mf.write(
                    json.dumps(
                        {
                            "clip_id": clip_id,
                            "audio_path": f"clips/{clip_id}.wav",
                            "duration_seconds": chunk_seconds,
                            "reference_text": "",  # see docstring re: WER
                        }
                    )
                    + "\n"
                )
                chunks_written += 1

    if chunks_written < num_chunks:
        click.echo(
            f"[warn] only produced {chunks_written}/{num_chunks} chunks — "
            f"dataset may not have enough long-form audio matching min_seconds={min_seconds}",
            err=True,
        )


if __name__ == "__main__":
    main()
