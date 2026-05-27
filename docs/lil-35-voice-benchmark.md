# LIL-35 Live Voice Benchmark Plan

This benchmark exists to decide whether the stable voice path needs ASR or
turn-taking changes before we build more memory and UX features.

The benchmark must run against the real desktop microphone path:

```text
Microphone -> AudioService -> Silero VAD -> delayed speech commit timing -> Faster-Whisper
```

Prepared local clips are intentionally not the primary path for this ticket.
They are too clean and do not reproduce Windows microphone behavior, natural
pauses, VAD segmentation, or user-perceived cutoffs.

## Current Hypothesis

Keep the default architecture as:

```text
Silero VAD -> Faster-Whisper -> LLM text -> Kokoro -> RVC
```

Do not switch the default provider until live measurements show a clear win.

## What To Measure Live

- ASR quality for short French and English conversational phrases.
- Time to transcript for each ASR model/profile.
- Warmup/cold-start cost, excluded from steady-state model comparison.
- Realtime factor on the user's Windows machine.
- Language detection behavior on FR/EN input.
- Low-confidence or hallucinated transcripts.
- Whether VAD/commit timing cuts the user off or waits too long.
- Approximate `VAD end -> transcript ready` latency, calculated as
  `speech_commit_delay_ms + ASR transcription time`.
- Whether a new segment starts before the previous delayed commit would have
  fired. This helps distinguish a real turn split from a segment that the app
  would likely merge during the commit grace period.

## Live Benchmark Command

Run:

```bash
python scripts/benchmark_asr.py \
  --segments 5 \
  --models small,large-v3-turbo,large-v3 \
  --device cuda \
  --compute-type float16 \
  --warmup-runs 1
```

Flow:

1. The script starts the same `AudioService` used by the desktop app.
2. Speak naturally into the real microphone.
3. Each VAD segment is saved under `data/benchmarks/live_voice/segments/`.
4. After capture, enter what you actually said for each segment.
5. Mark whether the app/VAD cut you off.
6. The script runs discarded warmup transcriptions for each selected model.
7. The script transcribes the exact same live segments with each selected model.
8. Reports are written to:
   - `data/benchmarks/live_voice.json`
   - `data/benchmarks/live_voice.md`

Useful options:

```bash
python scripts/benchmark_asr.py \
  --segments 8 \
  --language fr \
  --models small,large-v3-turbo \
  --vad-required-misses 20 \
  --speech-commit-delay-ms 700
```

## Decision Rules

- If `small` has bad live WER on short FR/EN phrases, test
  `large-v3-turbo`.
- If `large-v3-turbo` is much more accurate and still responsive, expose it as
  a documented recommended profile before changing defaults.
- If ASR is accurate on the saved live segment but the interaction feels bad,
  focus on VAD thresholds, delayed commit, microphone capture, or echo
  suppression rather than model size.
- If transcripts are correct but responses feel slow, keep ASR and optimize
  downstream latency separately.
- If the user reports cutoffs while ASR WER is high, fix turn-taking before
  interpreting the ASR model comparison.
- Use the `no cutoffs` WER/CER columns for model-quality comparison when one
  spoken phrase was split across multiple VAD segments.
- Ignore warmup runs for steady-state model choice, but keep them visible so
  startup cost is not hidden.

## First Local Candidates

Use the live script for the initial reproducible table:

- `small`, current default.
- `large-v3-turbo`, likely accuracy upgrade with lower cost than full
  `large-v3`.
- `large-v3`, quality ceiling if the GPU can tolerate it.
- French distilled Whisper variants already listed in `WhisperProvider`, only
  if we decide French quality is the main blocker.

## Model Review - May 2026

The local benchmark and external model review point to the same direction, but
this is a public open-source project. Do not optimize the default only for one
RTX 4070 development machine.

- Keep `small` as the conservative low-latency baseline for broad hardware
  compatibility, especially older GPUs, CPU fallback, and Windows laptops.
- Treat `large-v3-turbo` as the first serious local quality profile for users
  with enough VRAM, because it is multilingual and already works through the
  current Faster-Whisper integration.
- Do not switch the runtime default inside this benchmark PR. The benchmark
  proves `small` is risky on English / mixed turns, but default changes should
  be a separate product PR with documented profiles and fallback guidance.

Suggested public profiles:

- `balanced` / default: `small`, `cuda` when available, `float16` on GPU,
  CPU fallback via the existing provider fallback behavior.
- `quality-local`: `large-v3-turbo` for users with enough VRAM who prioritize
  multilingual accuracy over the lowest ASR latency.
- `research-streaming`: Parakeet TDT 0.6B V3, Canary, or Voxtral Realtime only
  behind separate adapters/prototypes, because they change runtime assumptions.

Candidates reviewed:

- `large-v3-turbo`: best near-term quality-profile fit for this repository. It
  is compatible with the current Faster-Whisper path and performed much better
  than `small` on the live mixed-language failure cases, at a roughly 120 ms
  median ASR latency cost on the development RTX 4070. That latency delta must
  not be generalized to all users.
- `large-v3`: useful as a quality ceiling, but heavier than needed for the
  current realtime desktop target.
- `distil-large-v3.5`: interesting for English, but its own model card says
  Whisper Turbo is recommended for multilingual speech recognition. Not a
  better fit than `large-v3-turbo` for French + English today.
- `Parakeet TDT 0.6B V3`: very strong research candidate. It supports 25
  European languages, automatic language detection, punctuation,
  capitalization, timestamps, long-form and streaming options. It is not a
  drop-in Faster-Whisper replacement and should be evaluated in a separate
  adapter/prototype ticket.
- `Canary-Qwen 2.5B`: strong English accuracy candidate, but less aligned with
  the current lightweight local realtime path.
- `Voxtral Mini 4B Realtime`: strategically interesting for future streaming
  ASR / turn-taking because it is built for realtime transcription, but the
  local model card recommends a single GPU with at least 16 GB memory for BF16
  vLLM serving. Many open-source users will have less VRAM, so this is not a
  safe default path.

Latest local benchmark interpretation:

- `small` is fast, but it can silently fail on English or mixed-language turns.
  In the latest 10-segment run it hallucinated a long French transcript for an
  English ASR question.
- `large-v3-turbo` is slower but much safer: it still missed the language on
  one English technical question, but preserved the user intent far better.
- The phrase split around "can you tell me more / about the manga One Piece"
  is a turn-taking problem, not an ASR model-quality result. Use the `no
  cutoffs` columns for model comparison when this happens.

Recommended follow-up tickets after this PR:

- Add documented ASR profiles: broad-compatible default (`small`) and
  quality-local (`large-v3-turbo`) without forcing heavier hardware on all
  users.
- Add a turn-taking / segment-merge ticket to handle natural pauses before the
  delayed commit fires.
- Create a research/prototype ticket for Parakeet TDT 0.6B V3 as a possible
  future ASR backend.

## External Signals

- Artificial Analysis' Q3 2025 State of AI report says its AA-WER index uses
  real-world speech with varied accents, domain language, and challenging
  acoustic conditions, and shows NVIDIA open-weight ASR models closing part of
  the accuracy gap with proprietary STT.
- Artificial Analysis' Speech-to-Text leaderboard compares WER, speed factor,
  and price. Its current open-weights notes list Voxtral Small, Parakeet TDT
  0.6B V3, and Whisper Large v2 among the top open-weight STT models by
  accuracy, while Parakeet TDT 0.6B V3 is listed as the fastest model by speed
  factor among providers shown there.
- The Open ASR Leaderboard paper standardizes WER and RTFx across English
  short-form, English long-form, and multilingual short-form tracks, matching
  the metrics this local benchmark now reports.
- Hugging Face's Open ASR Leaderboard write-up notes that Whisper Large v3 is a
  strong open model, while Parakeet/CTC-style models can be dramatically faster
  with some tradeoff in WER and language coverage.
- NVIDIA's Parakeet TDT 0.6B V2 model card reports a 600M English ASR model
  with punctuation, capitalization, timestamps, and high throughput.
- NVIDIA's model selection guide recommends Parakeet TDT 0.6B V3 for European
  multilingual ASR with automatic language detection, timestamps, and streaming
  options.
- Mistral's Voxtral Realtime model card reports a multilingual realtime ASR
  model with configurable latency, but also says BF16 vLLM serving expects a
  single GPU with at least 16 GB memory.
- Moonshine v2 targets latency-critical streaming ASR by avoiding the
  encode-the-whole-utterance latency profile of full-attention encoders.

Sources:

- https://artificialanalysis.ai/speech-to-text
- https://artificialanalysis.ai/downloads/state-of-ai/2025/Q3-2025-Artificial-Analysis-State-of-AI-Highlights-Report.pdf
- https://huggingface.co/blog/open-asr-leaderboard
- https://arxiv.org/abs/2510.06961
- https://docs.nvidia.com/nemo/speech/nightly/starthere/choosing_a_model.html
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/distil-whisper/distil-large-v3.5
- https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- https://arxiv.org/abs/2602.12241
