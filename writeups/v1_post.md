# I benchmarked 3 Whisper serving frameworks. The fastest one is unusable.

I spent the last two weeks running 30 benchmark cells comparing **HF Transformers**, **faster-whisper**, and **vLLM** serving OpenAI's Whisper-large-v3 on NVIDIA L4 and A100 GPUs, swept across concurrency levels of 1, 8, 32, 64, and 128. Here's what I found.

A quick note on how I'm using "concurrency": it refers to **your workload's offered load** — how many transcribe requests are hitting the service at the same time — not a config knob you turn. You can't pick c=8; your users do. The point of sweeping across concurrencies in the benchmark is so you can match your own workload to the right row in the matrix.

## Why I did this

I run ML / ML infra at a healthtech startup. We use Whisper as the speech-to-text layer behind a system of record — clinicians record sessions, we transcribe, the transcripts feed structured records that influence downstream decisions. That puts real pressure on three axes simultaneously:

- **Quality** — a transcription error is a clinical-data error. WER matters more than throughput.
- **Cost** — we're transcribing hours of audio per provider per day. $0.05 vs $0.50 per audio-hour compounds fast.
- **Operational simplicity** — when something goes wrong at 3am, I need to be able to debug the inference path quickly.

When I went to pick a serving framework for production, I couldn't find a benchmark that took all three seriously. Most "Whisper benchmark" posts focus on raw throughput on synthetic audio. The ones that mention quality usually don't normalize text the way OpenAI's reference normalizer does, so the WER numbers aren't comparable across frameworks. And nobody publishes per-request JSONs, so you can't dig into what actually happened on a given clip when a number looks weird.

So I built one. The methodology choices (Whisper's `EnglishTextNormalizer`, full per-request provenance, closed-loop concurrency, failures-as-findings) come from production needs, not academic preference.

## The headline

**vLLM × A100 at concurrency=8** runs at **285× real-time** for **$0.0048 per audio-hour**. That's **30× cheaper** than faster-whisper, **180× cheaper** than HF baseline.

Its Word Error Rate at that cell: **113%**.

Not a typo. WER over 100% means vLLM hallucinates *more words* than the reference transcript contained. On clean LibriSpeech audio that Whisper-large-v3 transcribes at ~1.4% WER everywhere else, vLLM produces 50-186% WER across the entire sweep.

Root cause: vLLM's Whisper integration ships without the standard ASR safeguards (compression-ratio threshold, no-speech threshold, temperature fallback) that openai-whisper and faster-whisper apply by default. The continuous-batching engine works as advertised; the Whisper wrapper around it is missing the silence-detection guards that prevent the model from freelancing on every quiet moment. **Until upstream fixes this, vLLM × Whisper is a benchmark trap, not a deployment.**

If you're routing patient audio through vLLM today thinking you're getting Whisper-quality transcripts at 1/30th the cost: you're not.

## The framework triangle

Three frameworks, each making a structurally different bet:

| | the bet | what it optimizes |
|---|---|---|
| **HF Transformers** | "no bet — use the reference model code" | nothing (unoptimized baseline) |
| **faster-whisper** | "recompile this specific model in optimized C++" | model-specific (Whisper-only) |
| **vLLM** | "build a general serving engine, adapt to Whisper" | framework-first |

Anything beyond these three (Ray Serve, Triton, ONNX Runtime, etc.) is either a serving infrastructure layer that sits *on top of* one of these engines, or a duplicate of one of these three bets. The triangle is the smallest comparison space that's actually informative.

## What I found

### 1. The fastest framework is unusable.
vLLM: WER 50-186% across all cells. The cheapest, fastest cell in the entire matrix is unshippable.

### 2. The quality champion can't scale.
HF baseline produces **1.44% WER, identical across every cell on every GPU.** Best transcription quality of the three. But PyTorch's `model.generate()` isn't thread-safe — concurrent calls corrupt CUDA state and crash the GPU. Realistic naive serving is therefore lock-serialized: N concurrent requests queue through one model. RTF stays flat at ~0.044 from c=1 to c=128. Submit 128 concurrent requests, the unlucky tail waits 3.5 minutes.

### 3. L4 wins cost-per-audio-hour. A100 wins latency.
HF × L4 at c=8 costs **$0.0265 per audio-hour** with 1.44% WER. The same configuration on A100 costs $0.0997 — **3.7× more expensive for identical quality.** Match the GPU to the workload: batch + cost-sensitive → L4. Real-time + latency-sensitive → A100.

### 4. OOM walls are findings.
faster-whisper × L4 caps at c=8 (24 GB memory ceiling). vLLM × L4 dies at c=128. These structural ceilings tell you which framework + GPU pair can scale to what concurrency — independent of accuracy.

## What I'd actually deploy

A note before the list: these recommendations are organized by **the workload you're serving**, not by knobs you choose. The framework + GPU is the decision; the offered concurrent load is the input that picks which row of the matrix applies.

- **Low offered load (a few concurrent requests) + cost-sensitive** (overnight backlogs, internal tooling) → **HF Transformers on L4** with long-form chunking enabled. **$0.027/audio-hour, 1.44% WER, p95 ~2-14s** depending on how concurrent your peak gets. Cheapest correct cell in our matrix. *Caveat: HF needs careful config (`return_timestamps=True`, `truncation=False`, `return_attention_mask=True`, and a `threading.Lock` around `generate()`) — without those you silently break.*
- **Low offered load + you want battle-tested defaults** → **faster-whisper on L4**. Higher WER (~4%) and ~3× the cost of HF, but ships with the ASR-specific safeguards (no-speech / compression-ratio / temperature fallback) that make it more robust on messier real-world audio than this clean-LibriSpeech eval shows. The conservative pick.
- **Latency-sensitive at low load** (live captions, voice agents, ≤8 concurrent streams) → **HF or faster-whisper on A100, single-stream**. A100 wins per-request latency when the host isn't contended. Pay 2-4× more for ~2× latency improvement.
- **High offered load (32+ concurrent), latency budget under 60s** → **no correct option in v1.** HF queues hit minutes; faster-whisper OOMs; vLLM hallucinates. Either bring multi-replica serving (Ray Serve, planned for v1.2) or wait for v1.1's Triton + TensorRT-LLM.
- **vLLM × Whisper at any load** → don't, until upstream fixes the hallucination issue.

## Three things I didn't expect

**The first HF run produced 32% WER.** Headline number was bad enough that I almost shipped the cell as "HF baseline has poor quality." Then I read the per-request hypotheses against references and noticed the hypothesis was always ~50 words even when the reference was ~150. Diagnosis: the eval clips averaged 35.5s but the HF adapter was silently truncating audio to Whisper's 30s encoder window. Fixed by switching to long-form generation (`return_timestamps=True` + `truncation=False` + `return_attention_mask=True`). WER dropped to 1.44%. **Don't trust headline numbers; read the actual transcripts.**

**The first HF concurrency run crashed the GPU.** At c=8 every request failed with a CUDA device-side assert. Root cause: PyTorch's `model.generate()` isn't thread-safe, and the harness uses async workers in a thread pool. Concurrent calls corrupted internal state and wedged the GPU. Adding a `threading.Lock` around `generate()` exposed the right baseline behavior — concurrent submissions queue through one GPU. **Naive concurrent serving on PyTorch needs a lock; if your serving framework promises thread-safety, double-check.**

**Three of four A100 deployments were poisoned by noisy neighbors.** Wall-clock 5× higher than expected, GPU utilization 20% instead of 90%, `nvidia-smi` queries timing out >2 seconds. Shared-host CPU contention from other RunPod tenants hammering the same physical machine. Took four pod redeploys to land on a quiet enough host to finish the sweep. **Cloud GPU benchmarks have inherent host-quality variance; absolute wall-clock numbers carry bias even with multiple iterations. Sanity-check with a smoke test before launching a full sweep.**

## On honest caveats

The v1 ships with documented limitations:

- HF × A100 wall numbers carry ~1.5-2.3× host-contention bias (within-sweep relationships are still correct; absolute numbers should be read as upper bounds)
- WER on LibriSpeech-clean is unrealistically low for production speech (sanity check, not an accuracy claim)
- Concurrency is closed-loop, not Poisson — models active users, not bursty traffic
- Ray Serve and Triton+TensorRT-LLM are deferred to v1.1+

Every measurement choice and every limitation is in `docs/METHODOLOGY.md`. The point isn't a perfect benchmark — it's an honest one.

## Reproducing this yourself

Repo, harness, raw per-cell JSONs, decision matrix: **[github.com/dyl5051/whisper-serving-bench](https://github.com/dyl5051/whisper-serving-bench)**

Every cell is reproducible end-to-end. The harness produces versioned per-cell JSONs with full provenance: git SHA, GPU model, driver version, hostname, every per-request timestamp. A stranger can reproduce one cell in under an hour. If your reproduction differs from mine by more than the stddev I report, open an issue.

## What's next

- **v1.1**: Triton + TensorRT-LLM. The "vLLM-but-without-the-hallucinations" bet.
- **v1.2**: Ray Serve. Different axis — serving infrastructure overhead vs. inference engine choice.
- **v2**: Long-form audio + variable input lengths + messier datasets (TED-LIUM, Common Voice).

Each release ships its own writeup. Public iteration over a single big drop. Star the repo if you want to follow along.
