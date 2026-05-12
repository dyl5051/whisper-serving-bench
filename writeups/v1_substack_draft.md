# Choosing a Whisper serving framework: a 30-cell decision matrix

**The cheapest correct Whisper-large-v3 deployment in this benchmark: Hugging Face Transformers on a single NVIDIA L4 — $0.027/audio-hour, 1.44% WER, at any concurrency.** Latency degrades from p95 1.96s to 209s as offered load grows from 1 to 128 in-flight requests, but cost and quality stay flat. vLLM is faster on raw throughput; on Whisper specifically, its transcripts are unusable today (50-186% WER — more on that below).

In the matrix below, "concurrency" is your workload's **offered load** — how many in-flight requests at peak — not a knob you tune. Identify your expected peak, find the matching row, read the latency and cost.

`[IMAGE: 01_decision_table_by_workload_shape.png — "Decision table by workload shape (4 rows)"]`

Two things worth pointing out:

- **HF × L4 cost is constant across concurrency at $0.027.** The lock-serialized baseline has flat throughput, so your bill doesn't change with load; only tail latency does. The framework + GPU decision is settled before you look at the latency column.
- **High concurrency + tight latency is the empty quadrant.** No framework in v1 serves correct Whisper transcripts at c=32+ under a 2-second p95 budget. That's not a failure of the benchmark — it's the actual state of open-source Whisper serving in mid-2026.

## Why this benchmark exists

Most published Whisper benchmarks measure one of two things: raw model throughput on a single framework with no concurrent load, or a vendor's own serving stack on their own preferred hardware. Neither answers the question production teams actually have: given my workload shape — N concurrent users, M seconds of latency budget, and a hard cost ceiling — which framework gives me the best dollars-per-audio-hour at acceptable tail latency?

This benchmark is built to answer that question. Three serving frameworks, two GPU classes, five concurrency levels under a closed-loop streaming workload, with versioned per-cell results JSONs and a reproduction recipe a stranger can follow.

## What we measured

- **Model**: OpenAI Whisper-large-v3, FP16, English-only, beam size 5. The latest fully open-weights Whisper checkpoint.
- **Eval audio**: LibriSpeech `test-clean` utterances greedily concatenated within speaker boundaries to ≥30s pseudo-clips. 50 clips averaging 35.5s (max 46s), totaling ~30 minutes of audio. 3 iterations per cell = 150 timed requests.
- **Frameworks**: Hugging Face Transformers (the unoptimized PyTorch baseline), faster-whisper (CTranslate2-compiled, same weights, optimized C++ execution), and vLLM (general-purpose LLM serving engine with continuous batching and PagedAttention; Whisper integration is recent, v0.6+).
- **GPUs**: NVIDIA A100 SXM4-80GB (premium, ~$1.49/hr on RunPod on-demand), NVIDIA L4 24GB (cost-effective, ~$0.60/hr).
- **Concurrency**: 1, 8, 32, 64, 128 closed-loop async workers each pulling from a shared queue.
- **Metrics per cell**: latency p50/p95/p99, aggregate RTF (real-time factor = wall-clock seconds per second of audio), WER post-normalization with Whisper's `EnglishTextNormalizer`, GPU utilization mean + max sampled at 1 Hz, peak GPU memory, cost-per-audio-hour at RunPod's on-demand pricing snapshot.

30 cells total (3 frameworks × 2 GPUs × 5 concurrencies). Every cell ran 10 sequential warmup requests excluded from timing, then 3 iterations through the eval set. Median across iterations is the headline number.

## What we deliberately didn't measure

Each item is a hook for a follow-up:

- **Ray Serve and NVIDIA Triton.** The original v1 spec was five frameworks. Vanilla Triton (Python backend) duplicates the HF baseline's story without batching primitives, and Ray Serve tests a different axis (serving infrastructure overhead, not inference engine performance). Both are deferred to v1.1+.
- **Triton with TensorRT-LLM compilation.** TensorRT-LLM's Whisper kernels are reportedly multi-x faster than PyTorch — this is the "easy path vs fast path" comparison and ships in v1.1.
- **T4 and CPU baselines.** "Is a GPU even worth it for ASR?" gets its own follow-up in v1.2.
- **Long-form audio and variable-length workloads.** v1's streaming cells use 30s+ pseudo-clips averaging 35.5s. Whisper's chunked-decoding behavior on hour-long audio is where most benchmarks lie; v2 addresses it head-on.
- **Model variants.** distil-whisper, large-v3-turbo. v2.1.
- **Open-loop (Poisson) traffic.** v1 is closed-loop (N active users). Bursty traffic produces a different tail latency story; possibly v3.

## The decision matrix

### A100 SXM4-80GB

**Aggregate RTF (lower is faster)**

`[IMAGE: 02_a100_rtf.png — "A100 SXM4-80GB Aggregate RTF, 3 frameworks × 5 concurrencies"]`

**Latency p95 (seconds)**

`[IMAGE: 03_a100_p95.png — "A100 SXM4-80GB Latency p95"]`

**WER (lower is better; >100% = framework hallucinates more words than reference contains)**

`[IMAGE: 04_a100_wer.png — "A100 SXM4-80GB WER"]`

**Cost USD per audio-hour (on-demand RunPod)**

`[IMAGE: 05_a100_cost.png — "A100 SXM4-80GB Cost USD per audio-hour"]`

### L4 24GB

**Aggregate RTF**

`[IMAGE: 06_l4_rtf.png — "L4 24GB Aggregate RTF"]`

**Latency p95 (seconds)**

`[IMAGE: 07_l4_p95.png — "L4 24GB Latency p95"]`

**WER**

`[IMAGE: 08_l4_wer.png — "L4 24GB WER"]`

**Cost USD per audio-hour**

`[IMAGE: 09_l4_cost.png — "L4 24GB Cost USD per audio-hour"]`

**How to read these tables.** Cross-reference latency-p95 against cost. Two worked examples:

`[IMAGE: 10_how_to_read_tables.png — "Two worked examples mapping latency budget → qualifying cells → cheapest winner"]`

`OOM` cells are findings, not omissions: faster-whisper × L4 at c=32+ runs out of memory on the 24GB card because each request holds full 30s context; vLLM × L4 OOMs at c=128 from KV cache pressure.

## Three findings worth screenshotting

### Finding 1: The fastest framework is unusable on Whisper today

**vLLM is 16-26× faster than faster-whisper on identical hardware.** At c=8 on A100, vLLM hits RTF 0.0035 (285× real-time) at $0.0048 per audio-hour. The same workload on faster-whisper costs $0.142/audio-hour — vLLM is **30× cheaper**.

But the WER tells a different story. **vLLM produces 50-186% WER across every cell on every GPU.** WER over 100% means the framework hallucinates *more words* than the reference transcript contains. On the same LibriSpeech-clean audio that Whisper-large-v3 transcribes at ~1.4% WER through HF or 4% through faster-whisper, vLLM emits entire sentences of fabricated text.

Root cause: vLLM's Whisper integration ships without the standard ASR safeguards — compression-ratio threshold, no-speech threshold, temperature fallback — that openai-whisper and faster-whisper apply by default.

**We tested three obvious surface-level mitigations at A100 c=1 before publishing.** None rescued the WER:

`[IMAGE: 11_vllm_hallucination_mitigations.png — "vLLM hallucination mitigation experiment: prompt / crfilter / vad, all WER >95%"]`

Spot-checking sample hypotheses reveals the deeper issue: **vLLM × Whisper isn't producing noise or repetition loops — it's emitting coherent English narrative that has nothing to do with the audio.**

> Reference (LibriSpeech): *"Concord returned to its place amidst the tents..."*
> vLLM hypothesis: *"Concerns and fears. By the time I was there I was already in the middle of the night and I was already feeling the heat of the night..."*

The model isn't getting confused by silence; it's largely *ignoring the audio entirely* and generating prior-distribution text. The prompt variant literally completes the prompt ("The following is a clear English narration..." → "...short version of the original text"). The compression-ratio filter doesn't help because the hallucinations don't loop — they're well-formed sentences. VAD doesn't help because the hallucination isn't caused by silence.

**Implication:** the fix isn't a config tweak. It's an integration rewrite — vLLM needs to either correctly route Whisper's audio embeddings into the decoder, or implement the full openai-whisper-style ASR safeguard loop (mid-batch retry with temperature fallback) inside its scheduler. Until that lands upstream, vLLM × Whisper is a benchmark trap, not a deployment. The throughput numbers are real; they just don't describe a system that can transcribe audio.

### Finding 2: HF baseline doesn't scale with concurrency

**HF baseline RTF is flat from c=1 to c=128 at ~0.044 on L4 — concurrency buys you zero throughput.** That's not a bug; it's the structural fingerprint of naive single-model serving. PyTorch's `model.generate()` is not thread-safe; concurrent calls corrupt CUDA state and crash the GPU with a device-side assert. Realistic naive deployment therefore lock-serializes concurrent requests: N submissions queue through one model on one GPU.

The p95 latency table shows the cost of that serialization clearly: **at c=128, HF p95 is 209 seconds on L4 and 327 seconds on A100.** Submit a request when 127 others are ahead of you and you wait minutes. Throughput is constant; tail latency grows linearly with queue depth.

**Implication:** this is *the* gap that batched (vLLM continuous batching) and replicated (Ray Serve multi-replica) frameworks fill in production. If your workload involves high concurrency, the HF baseline's flat-throughput / linear-latency profile is your baseline cost of *not* using a serving framework.

**Faster-whisper inherits a softer version of the same limitation.** CTranslate2 doesn't tensor-batch across requests; faster-whisper × A100 RTF is also nearly flat across concurrency at ~0.10. The model-specific-optimization bet wins on per-request speed but doesn't fix the scaling story.

### Finding 3: L4 is the right cost decision for most Whisper workloads

The cheapest correct cell in the entire 30-cell matrix is **HF × L4 at c=8 — $0.0265 per audio-hour with 1.44% WER**. The same configuration on A100 SXM4-80GB costs $0.0997/audio-hour — **3.7× more expensive for identical transcription quality**. The faster-whisper-on-L4 row is similar: ~$0.10/audio-hour vs ~$0.14/audio-hour on A100 for the same cells, and both deliver 4% WER.

**Why L4 wins on Whisper specifically:** Whisper-large-v3 is small enough (1.5B parameters in FP16 = ~3GB) that the L4's 24GB of VRAM and ~485 TFLOPS of FP16 are not the bottleneck for single-stream inference. The A100's headroom buys you better latency under concurrency and higher batching capacity — both irrelevant when the framework underneath can't batch anyway. Production teams reaching for A100 because "more is better" are paying 3-4× the marginal cost for capacity their inference framework can't capitalize on.

**Implication:** the decision rule for picking a GPU for Whisper inference is workload-driven, not "default to the biggest GPU." For batch and cost-sensitive workloads (overnight backlog transcription, internal tooling), L4 is the correct default. A100 earns its premium only when you need per-request latency at low concurrency, or when you're running a framework that actually uses the headroom — which today means waiting for vLLM's Whisper integration to be fixed or v1.1's TensorRT-LLM.

## Should you self-host at all?

The 30-cell matrix answers "which self-hosted framework wins." It doesn't answer the prior question: should you self-host Whisper in the first place, or just use a managed service? For most teams, that's the more important decision.

Approximate managed-STT pricing as of mid-2026 (verify before quoting; these change quarterly):

`[IMAGE: 12_managed_stt_pricing.png — "Managed STT pricing: AWS, Google, Azure, OpenAI, Deepgram, AssemblyAI"]`

Compared to the cheapest self-hosted cell (**HF × L4 at $0.027/audio-hour**), Deepgram and AssemblyAI are ~13-16× more expensive, and AWS Transcribe is ~53× more expensive. *On paper*, self-hosting wins by an order of magnitude.

What the $/audio-hour comparison hides:

- **Self-hosted price is best-case.** It assumes 100% GPU utilization, no cold starts, no autoscaling friction, no replica overhead, no failed-request retries, no idle time between requests. Real-world utilization on a single L4 with bursty traffic might be 30-50%, which roughly doubles or triples your effective $/audio-hour.
- **Self-hosted price excludes operational cost.** Engineering time to maintain the deployment (CUDA upgrades, framework version bumps, capacity planning, on-call), monitoring infra, multi-region failover, audit logs — all of that is "free" with managed and meaningful with self-hosted. At small scale that's the larger line item.
- **Managed services bundle reliability and SLAs.** Deepgram quotes 99.9% uptime; self-hosted on a single Pod is whatever your provider's pod-level reliability is (in our experience, "varies").
- **Compliance and data residency.** Some industries (healthtech, gov, finance) need on-prem or VPC-controlled inference where managed services aren't an option. That decision is made before cost enters the picture.

**The decision rule:** self-host if you have (a) high sustained volume that amortizes the operational cost across many audio-hours, OR (b) compliance / data-residency requirements that managed services can't meet. Otherwise, managed services almost always win on total cost of ownership at small-to-medium scale.

*Author's note: this benchmark was motivated by case (b). In healthtech infrastructure, clinician audio can't leave the VPC, so managed services aren't on the table even when they'd be cheaper on paper. Your situation may differ — and if (a) and (b) don't apply to you, the right answer is probably a managed provider.*

## Methodology in 90 seconds

The four most-defensible measurement choices:

1. **Closed-loop concurrency** — N active workers each waiting for their previous response before sending the next. Models "N concurrent users actively using the system," which matches real ASR usage patterns better than open-loop Poisson arrivals would for this use case.
2. **Identical text normalization** via Whisper's own `EnglishTextNormalizer` — different frameworks emit different default punctuation, casing, and number formatting; unnormalized WER comparisons are not meaningful. WER is computed by `jiwer` against the normalized hypothesis and reference.
3. **Warmup excluded, iterations median** — 10 sequential warmup requests precede each cell to load the model, allocate GPU memory, and populate any framework-internal caches; they don't count toward the headline. Each cell then runs 3 iterations through the full eval set; median across iterations is the headline, standard deviation is in the per-cell JSON.
4. **Failures are findings** — OOM at high concurrency, framework refusing a config, request timeouts — are recorded as cell outcomes (`degraded`, `error`, `unsupported`) and emitted to the results JSON. Never silently dropped.

Two caveats shipping with v1:

- **HF × A100 wall-clock numbers carry ~1.5-2.3× host-contention bias.** RunPod's US-MO-1 A100 inventory was running with noisy CPU neighbors during data collection; GPU utilization on HF A100 cells came back at 20-32% (vs L4's ~92%) — clear signal that the bottleneck was CPU, not GPU. Within-sweep relationships (WER stability, RTF flatness across concurrencies, linear p95 scaling) are correct and reproducible; absolute wall-clock numbers should be read as upper bounds.
- **WER on LibriSpeech-clean derived audio is unrealistically low** for production speech (which is messier — accents, noise, overlapping speakers). WER here is a sanity check that the framework didn't break the model, not a model-accuracy claim.

## Reproduce in under an hour

Full reproduction recipe, raw JSONs, methodology doc, per-cell configs: **[github.com/dyl5051/whisper-serving-bench](https://github.com/dyl5051/whisper-serving-bench)**

Every cell runs end-to-end from a single Docker container. The harness produces a versioned per-cell JSON with full provenance: git SHA, Docker tag, GPU model + driver version, hostname, Python version, every per-request timestamp + transcription + reference. If your reproduction differs from ours by more than the standard deviation we report, open an issue. If it differs by a lot, we want to know.

## What's next, and the cadence

- **v1.1** (target: 1-2 weeks from v1): Triton + TensorRT-LLM. The "easy path vs fast path" comparison.
- **v1.2** (target: 2-3 weeks from v1): Ray Serve + T4 + CPU baselines. Tests serving-infra overhead and the "is a GPU worth it for ASR" question.
- **v2** (target: 4-6 weeks from v1): Long-form audio + variable input lengths + messier datasets (TED-LIUM, Common Voice).
- **v2.1**: distil-whisper + whisper-large-v3-turbo — the accuracy-vs-cost frontier across model sizes.

Each release ships its own writeup. Public iteration over a single big drop — the deadlines force shipping discipline.

## Acknowledgments

This benchmark stands on the work of: the OpenAI Whisper paper authors and the original openai-whisper reference implementation; Hugging Face Transformers for hosting the canonical PyTorch port; Guillaume Klein and the faster-whisper team for the CTranslate2 re-implementation; the vLLM team for the continuous-batching engine; and NVIDIA for the open Triton and TensorRT-LLM tooling we'll be using in v1.1.

Repo, raw JSONs, methodology, and per-cell configs: **[github.com/dyl5051/whisper-serving-bench](https://github.com/dyl5051/whisper-serving-bench)**

Corrections and reproductions welcome via GitHub issues — the artifact is the repo, not just this post.

---

## How to use this draft in Substack

Paste the body above into Substack's editor, then find each `[IMAGE: NN_*.png — "..."]` line and replace it with the corresponding image from `writeups/v1_publishable_assets/`. The 12 numbered prefixes match the order they appear in the post — drag-and-drop them in sequence as you scroll.

After paste, Substack typically strips some markdown formatting. Two cleanup passes:

1. Re-bold the key numbers (search for the inline numbers that were `**bold**` and re-apply Substack's bold formatting).
2. Re-italicize the blockquoted vLLM hypothesis example and the *Author's note* paragraph.

Subtitle suggestion: *"$0.027/audio-hour vs $0.43/audio-hour with managed services — but only if you can self-host."*

Hero image suggestion: `results/published/v1_partial/charts/cost_per_audio_hour.png` (already in the repo) as the post thumbnail.
