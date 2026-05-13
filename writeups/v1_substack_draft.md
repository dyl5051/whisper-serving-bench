# Choosing a Whisper serving framework: a 30-cell decision matrix

Whisper-large-v3 has quietly become the default open-source speech-to-text model — if you've used a meeting tool, voice agent, or transcription product in the last two years, there's a good chance Whisper is somewhere in the stack. OpenAI doesn't host large-v3 directly (their hosted Whisper API runs an older variant), so anyone who wants the latest model self-hosts. That means picking a serving framework.

I needed to pick one. I run ML infrastructure at a healthtech startup; we transcribe clinician sessions into a structured system of record. Three things matter simultaneously: WER (a transcription error is a clinical-data error), cost (hours of audio per provider per day adds up fast), and operational simplicity. Every Whisper benchmark I could find took at most one of those seriously. So I built one that takes all three.

**What I found: the fastest framework hits 285× real-time at half a cent per audio-hour. Its Word Error Rate at that cell: 113%.**

WER over 100% means the framework outputs *more hallucinated words than the reference contained.* The fastest, cheapest cell in the entire matrix is unshippable. The cheapest *correct* deployment turned out to be Hugging Face Transformers on a single NVIDIA L4 — a $0.60/hr GPU you probably weren't planning to use — at **$0.027/audio-hour and 1.44% WER**, regardless of concurrency. Latency degrades from p95 1.96s to 209s as offered load grows from 1 to 128 in-flight requests, but cost and quality stay flat.

In the matrix below, "concurrency" is your workload's **offered load** — how many in-flight requests at peak — not a knob you tune. Identify your expected peak, find the matching row, read the metrics.

`[IMAGE: 01_decision_table_by_workload_shape.png — "Decision table: HF × L4 at each measured concurrency (1, 8, 32, 64, 128). Columns: deployment / RTF / p95 / WER / cost / audio-hour"]`

Three things worth pointing out:

- **The framework choice is settled before you look at the rows.** HF × L4 wins at every concurrency we measured. Cost, RTF, and WER are essentially constant down the table — the only thing that changes is p95 latency.
- **p95 latency scales linearly with concurrency.** ~1.6s at c=1, then +1.6s for every additional in-flight request behind the lock. That's the structural cost of running a single-model PyTorch deployment under concurrent load (see Finding 2 for why).
- **Tight latency + high concurrency is the empty zone v1 can't fill.** If you need p95 <2s, stay at c=1. If you accept tens of seconds of p95, c≤32 works. v1 has no deployment that delivers both single-digit p95 AND high concurrency simultaneously with correct transcripts — HF queues, faster-whisper OOMs, vLLM hallucinates. That gap is what v1.1 (TensorRT-LLM) is meant to close.

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

**Scope of these findings.** Everything below describes v1's measurement regime: single-instance serving on a single GPU, closed-loop concurrent requests at 1-128 in-flight, 30s+ pseudo-clips of clean read audio, on-demand RunPod pricing as of mid-2026. Outside this regime — multi-node fleets, bursty Poisson-distributed traffic, true streaming / incremental decoding, sub-second latency budgets, or hour-long audio — the answer changes, sometimes substantially. Each finding states its scope explicitly and calls out where we expect it to flip; v1.x releases will measure the regimes v1 didn't.

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

### Finding 3: For unbatched frameworks, L4 wins cost. For batched frameworks, A100 likely does.

The cheapest *correct* cell in the entire 30-cell matrix is **HF × L4 at c=8 — $0.0265 per audio-hour with 1.44% WER**. The same configuration on A100 SXM4-80GB costs $0.0997/audio-hour — **3.7× more expensive for identical transcription quality**. Faster-whisper has the same shape: ~$0.10/audio-hour on L4 vs ~$0.14 on A100 for comparable cells, both at ~4% WER.

The math under that: cost-per-audio-hour = `GPU_hourly_cost × RTF`. L4 is ~$0.60/hr; A100 is ~$1.49/hr. For A100 to win on cost, its RTF needs to be more than 2.5× lower than L4's. On HF and faster-whisper, A100 is ~2× faster at best — close, but not enough to overcome the hourly cost ratio.

**That math inverts the moment the framework can batch.** vLLM × A100 c=8 is the **single cheapest cell in our entire matrix at $0.0048/audio-hour** — half the cost of vLLM × L4 c=8 ($0.0097), and roughly **5× cheaper than HF × L4 c=8**. The 80GB A100 packs more KV-cache pages than the 24GB L4, and continuous batching scales nonlinearly in cache capacity, so A100's hourly premium gets amortized across many concurrent requests. The catch: vLLM produces 113% WER at that cell (Finding 1), so this cheapest cell is unusable. But the cost *pattern* is real, and it's what we expect to see again with v1.1's Triton + TensorRT-LLM — which keeps Whisper's ASR safeguards intact.

**Scope of this finding.** "L4 wins cost" holds within v1's measurement regime: **unbatched, single-instance serving on a single GPU**. The moment you have a working batched framework — TensorRT-LLM-compiled Triton, a future fixed vLLM, or any other engine that uses the bigger GPU's memory for continuous batching — the cost story likely flips, and A100 wins. v1's data shows this pattern already; vLLM's quality bug obscures it. v1.1 will measure it cleanly.

**Implication for production teams:** if you're deploying HF or faster-whisper today, L4 is the right call. If you're investing in a batched serving stack, don't default to L4 on the strength of this benchmark alone — re-evaluate A100 once the batched framework is in place. Watch for v1.1.

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
