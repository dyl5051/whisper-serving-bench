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
        elif spec["source"] == "huggingface_datasets_concatenated":
            _prepare_hf_dataset_concatenated(spec, out_dir)
        else:
            click.echo(f"[error] {name}: unknown source {spec['source']!r}", err=True)
            continue

        click.echo(f"[done] {name}: manifest written to {manifest_path}")


def _prepare_hf_dataset(spec: dict, out_dir: Path) -> None:
    """Materialize a HuggingFace dataset directly, one clip per row.

    Uses streaming=True to avoid bulk-downloading the entire dataset config.
    Some HF datasets (notably librispeech_asr post-parquet-conversion) don't
    cleanly partition splits at the file level, so a non-streaming
    `load_dataset(..., split="test")` ends up downloading train splits too.
    Streaming mode is lazy: it only fetches rows we actually iterate.

    We pass `decode=False` on the audio column so HF datasets hands us raw
    bytes instead of trying to decode via torchcodec — torchcodec needs
    FFmpeg shared libs that aren't present on every system. soundfile
    handles WAV and FLAC (which is what LibriSpeech ships) natively without
    that dep.
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(
        spec["dataset"],
        spec.get("config"),
        split=spec["split"],
        streaming=True,
    )

    audio_col = spec["audio_column"]
    text_col = spec["text_column"]

    ds = ds.cast_column(audio_col, Audio(decode=False))

    limit = spec.get("limit")
    if limit:
        ds = ds.take(limit)

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as mf:
        for i, row in enumerate(ds):
            arr, sr = _decode_audio_row(row[audio_col])
            if arr is None:
                continue
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


def _decode_audio_row(audio_data: dict):
    """Decode a row's audio dict (from Audio(decode=False)) to (ndarray, sample_rate).

    Returns (None, None) if the row has no usable audio bytes — caller should skip.
    """
    import io

    import soundfile as sf

    if not audio_data:
        return None, None
    raw = audio_data.get("bytes")
    path = audio_data.get("path")
    if raw:
        arr, sr = sf.read(io.BytesIO(raw))
        return arr, sr
    if path:
        # Streaming sometimes hands us a remote path with no bytes attached;
        # soundfile can read local files, but this branch is best-effort.
        try:
            arr, sr = sf.read(path)
            return arr, sr
        except Exception:  # noqa: BLE001
            return None, None
    return None, None


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
    from datasets import Audio, load_dataset

    ds = load_dataset(
        spec["dataset"],
        spec.get("config"),
        split=spec["split"],
        streaming=True,
    )
    ds = ds.cast_column(spec["audio_column"], Audio(decode=False))

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
            arr, sr = _decode_audio_row(row[audio_col])
            if arr is None:
                continue
            arr = np.asarray(arr)
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


def _prepare_hf_dataset_concatenated(spec: dict, out_dir: Path) -> None:
    """Materialize a dataset by concatenating adjacent utterances into ~target_seconds chunks.

    Walks the dataset (optionally grouped by `group_by_column`, e.g. speaker_id),
    appending utterances to a buffer until the buffer's audio duration meets or
    exceeds `target_seconds`, then emits the buffer as one chunk and starts a
    new buffer.

    Reference text is the space-joined concatenation of the constituent
    utterances' transcripts. Audio is sample-concatenated assuming all rows
    share a sample rate (true for LibriSpeech at 16kHz mono).

    Used for the v1 streaming workload (LibriSpeech) where we need reference
    transcripts for WER but want chunks long enough that the encoder doesn't
    finish faster than the framework's per-request overhead can be measured.
    """
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(
        spec["dataset"],
        spec.get("config"),
        split=spec["split"],
        streaming=True,
    )
    ds = ds.cast_column(spec["audio_column"], Audio(decode=False))

    target_seconds = float(spec.get("target_seconds", 30.0))
    num_chunks = int(spec.get("num_chunks", 50))
    audio_col = spec["audio_column"]
    text_col = spec["text_column"]
    group_by = spec.get("group_by_column")

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    # Streaming-friendly grouping: we can't materialize all rows into per-speaker
    # buckets up front (would defeat streaming). Instead we walk the stream and
    # maintain a per-speaker buffer dict; each row gets appended to its speaker's
    # buffer; whenever any speaker's buffer reaches target_seconds, emit it as a
    # chunk and reset that buffer. Stop once num_chunks have been written.
    buffers: dict = {}
    chunks_written = 0
    manifest_path = out_dir / "manifest.jsonl"

    with manifest_path.open("w") as mf:
        for row in ds:
            if chunks_written >= num_chunks:
                break

            arr, row_sr = _decode_audio_row(row[audio_col])
            if arr is None:
                continue
            arr = np.asarray(arr)
            group_key = row[group_by] if group_by else "_all_"

            if group_key not in buffers:
                buffers[group_key] = {
                    "audio": [],
                    "text": [],
                    "samples": 0,
                    "sr": row_sr,
                }
            buf = buffers[group_key]

            # Skip rows with mismatched sample rate within a group;
            # we don't resample on the fly.
            if buf["sr"] != row_sr:
                continue

            buf["audio"].append(arr)
            buf["text"].append(row[text_col])
            buf["samples"] += len(arr)

            if buf["samples"] / buf["sr"] >= target_seconds:
                concatenated = np.concatenate(buf["audio"])
                clip_id = f"libri_concat_{chunks_written:04d}"
                wav_path = clips_dir / f"{clip_id}.wav"
                sf.write(wav_path, concatenated, buf["sr"], subtype="PCM_16")
                mf.write(
                    json.dumps(
                        {
                            "clip_id": clip_id,
                            "audio_path": f"clips/{clip_id}.wav",
                            "duration_seconds": len(concatenated) / buf["sr"],
                            "reference_text": " ".join(buf["text"]),
                        }
                    )
                    + "\n"
                )
                chunks_written += 1
                # Reset this speaker's buffer; other speakers' buffers persist.
                buffers[group_key] = {
                    "audio": [],
                    "text": [],
                    "samples": 0,
                    "sr": row_sr,
                }

    if chunks_written < num_chunks:
        click.echo(
            f"[warn] only produced {chunks_written}/{num_chunks} chunks — "
            f"stream exhausted before target reached",
            err=True,
        )


if __name__ == "__main__":
    main()
