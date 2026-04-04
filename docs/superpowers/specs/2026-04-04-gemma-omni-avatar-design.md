# Design: Gemma-Omni Real-Time Avatar with Screen Sharing

**Date:** 2026-04-04
**Author:** Khalil + Claude
**Status:** Approved

## Overview

Upgrade the Local-AI-Companion to a real-time conversational avatar that can listen, see the user's screen, think, and speak with a cloned voice and emotions — running locally on an RTX 4070 12GB + 32GB RAM.

### Goals

- Real-time voice conversation with < 2s perceived latency
- Live screen sharing (avatar sees the user's screen continuously)
- Voice cloning (March 7th character voice)
- Emotional expressiveness (laughs, sighs, excitement in speech)
- French + English support
- 100% local, no cloud dependency
- Backward-compatible with existing pipeline/omni modes

### Non-Goals

- Custom Live2D model creation (use existing March 7th model)
- Persistent memory across sessions (future work)
- PC control / tool use (future work)
- Camera/webcam input (future work)

---

## Tech Stack

| Component | Model / Tool | Quantization | VRAM | Role |
|---|---|---|---|---|
| **Brain** | Gemma 4 E4B-it | TorchAO int4 | ~4.1 GB | ASR + LLM + Vision (native) |
| **Voice** | Chatterbox Multilingual | ONNX Q4 | ~2.0 GB | TTS + Voice Clone + Emotions |
| **VAD** | Silero VAD v5 | ONNX | CPU | Speech detection |
| **Screen Capture** | mss | — | CPU | Screenshot capture |
| **Avatar** | Live2D Cubism SDK | — | GPU (minimal) | Avatar rendering |
| **Transport** | FastAPI + WebSocket | — | — | Frontend/backend communication |
| **Total VRAM** | | | **~7.3 GB / 12 GB** | ~4.7 GB headroom |

### Key Technical Decisions

1. **Gemma E4B over separate ASR+LLM**: Single model handles audio + vision + reasoning. Native `<|audio|>` + `<|image|>` tokens in same prompt confirmed by Google docs.
2. **TorchAO int4 over bitsandbytes**: bitsandbytes has a known silent failure bug on Transformers v5.x. TorchAO `int4_weight_only` is the stable alternative.
3. **Chatterbox Multilingual ONNX Q4 over Qwen3-TTS/Kokoro**: Only option that combines voice cloning + emotion tags (`[laugh]`, `[chuckle]`) + 23 languages (including French) in a lightweight package.
4. **Pipeline architecture over omni model**: Streaming LLM-to-TTS sentence-by-sentence gives much better perceived latency than waiting for full response.
5. **ONNX Runtime for TTS**: Separates TTS from PyTorch runtime, reducing CUDA memory fragmentation.

---

## Architecture

### High-Level Flow

```
Mic → Silero VAD → speech detected
                        ↓ audio bytes (int16 PCM, 16kHz)
                        
mss → ScreenBuffer → latest frame
                        ↓ PIL Image
                        
    ┌───────────────────────────────────────────┐
    │          GemmaOmniPipeline                 │
    │                                           │
    │  audio + image + history → Gemma E4B      │
    │  → streaming tokens                       │
    │  → sentence splitting                     │
    │  → emotion parsing (dual output)          │
    │  → Chatterbox TTS (per sentence)          │
    │  → audio chunks + volume arrays           │
    └───────────┬──────────────┬────────────────┘
                │              │
                ▼              ▼
        Audio playback    Live2D avatar
        (voice clone)     (expression + lip-sync)
```

### Streaming Pipeline (Latency Optimization)

The key to low perceived latency is **pipelining** LLM generation with TTS synthesis:

```
Gemma generates: token → token → token → "." (sentence boundary)
                                              │
                                    sentence sent to Chatterbox
                                    TTS starts generating audio
                                              │
                              WHILE Gemma continues next sentence
```

**Estimated latency (pipelined):**

| Stage | Time |
|---|---|
| VAD end-of-speech (~15 misses) | ~480ms |
| Gemma audio+image encoding | ~200ms |
| Gemma prefill + first sentence (~15 tokens) | ~500ms |
| Chatterbox first audio chunk | ~200ms |
| **Avatar starts speaking** | **~1.4s** |

With filler words ("Hmm..."): **~1.0s**

### Screen Sharing: Smart Hybrid Approach

Three vision modes, selected automatically or by user request:

| Mode | When | Frames | Token Budget | Latency Impact |
|---|---|---|---|---|
| **Passive** | Every conversation turn | 1 (latest) | 70-140 | Minimal |
| **Replay** | "What just happened?" | 10-30 (last N seconds) | 70 per frame | Moderate |
| **Detail** | "Read this text" / "Look closely" | 1 (high-res) | 560-1120 | Notable |

The `ScreenBuffer` runs in a background thread:
- Captures via `mss` every 2 seconds
- Pixel diff to skip identical frames (CPU, near-zero cost)
- Circular buffer of 30 frames
- No GPU usage until frames are sent to Gemma

### Callback Interface (Backward Compatible)

`GemmaOmniPipeline` exposes the same callback interface as existing pipelines:

```python
callbacks = {
    "on_transcription": fn,       # transcribed text from audio
    "on_response_start": fn,      # response generation started
    "on_response_chunk": fn,      # token-by-token streaming
    "on_response_end": fn,        # full response complete
    "on_audio_ready": fn,         # AudioPayload (WAV + volumes + expression)
    "on_expression_change": fn,   # Live2D expression update
}
```

This means **all existing frontends (Live2D overlay, Web UI, Desktop companion) work without modification**.

---

## New Modules

### 1. `GemmaProvider` — `src/omni/gemma_provider.py`

Wrapper around Gemma 4 E4B with TorchAO int4 quantization.

**Responsibilities:**
- Lazy model loading with `AutoModelForMultimodalLM` + `AutoProcessor`
- TorchAO `int4_weight_only` quantization (group_size=128)
- `chat(messages, audio, images)` — single-turn inference
- `chat_stream(messages, audio, images)` — streaming token generation
- `transcribe(audio_bytes)` — ASR-only mode
- Thread-safe loading with lock (same pattern as existing `MiniCPMoProvider`)

**Key implementation details:**
- Prompt format uses `<|audio|>` and `<|image|>` placeholder tokens
- Audio placed before image, both before text (per Google docs)
- SDPA attention implementation (FlashAttention if available)
- `torch.no_grad()` context for all inference
- History trimming to `context_max_turns` (default 10)

### 2. `ChatterboxTTSProvider` — `src/tts/chatterbox_provider.py`

Extends `BaseTTS` (defined in `src/tts/base.py`) using Chatterbox Multilingual ONNX Q4.

**Implements all `BaseTTS` abstract methods:**
- `synthesize(text, output_path)` — text with emotion tags → WAV → `TTSResult`
- `synthesize_stream(text)` — async generator yielding audio chunks
- `list_voices(language)` — returns available Chatterbox voices for language
- `set_voice(voice_id)` — sets reference audio path for voice cloning
- `set_rate(rate)` — adjusts speech rate (maps to Chatterbox speed param)
- `set_pitch(pitch)` — no-op (Chatterbox handles pitch internally)

**Additional methods:**
- `load()` — load ONNX Q4 model from HuggingFace cache
- `set_reference_audio(ref_audio_path)` — pre-load voice reference for cloning
- Configurable `exaggeration` (0.0-1.0) per character

**Emotion tags supported in text:**
- `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]` — interpreted natively by Chatterbox

### 3. `ScreenBuffer` — `src/vision/screen_buffer.py`

Continuous screen capture with change detection.

**Responsibilities:**
- Background thread capturing via `mss` at configurable interval
- Pixel diff (numpy array comparison) to skip unchanged frames
- Circular buffer (deque) of last N frames as PIL Images
- `get_latest()` — single frame for passive mode
- `get_recent(n)` — N frames for replay/video mode
- `start()` / `stop()` lifecycle management

### 4. `GemmaOmniPipeline` — `src/omni/gemma_omni_pipeline.py`

Main orchestrator for the gemma-omni mode.

**Responsibilities:**
- Wires together GemmaProvider + ChatterboxTTSProvider + ScreenBuffer
- `process_speech(audio_bytes)` — full pipeline: audio+screen → Gemma → emotion parse → TTS → callbacks
- Sentence splitting from streaming tokens
- Dual emotion routing: Chatterbox tags stay in TTS text, style markers route to Live2D
- Audio volume analysis for lip-sync (reuses existing `analyze_audio_volumes`)
- Vision mode selection (passive/replay/detail)
- Conversation history management

### 5. `VRAMMonitor` — `src/utils/vram_monitor.py`

Debug utility for VRAM tracking.

**Responsibilities:**
- Log allocated/reserved VRAM at each pipeline stage
- Warn when usage exceeds configurable threshold (default 80%)
- Enabled via config `debug.vram_monitor: true`

---

## Modified Existing Files

| File | Change |
|---|---|
| `config/config.yaml` | Add `gemma` section (model, quantization, vision, screen) and `tts.chatterbox` section |
| `config/characters/march7th.yaml` | Add `voice.chatterbox_ref_audio`, `voice.chatterbox_exaggeration`, update `system_prompt` with emotion tag instructions |
| `src/assistant/app.py` | Add `elif mode == "gemma-omni"` branch to instantiate `GemmaOmniPipeline` |
| `src/server/websocket.py` | Add routing for gemma-omni mode in WebSocket handler |
| `src/utils/emotion_detector.py` | Add dual-output: clean text for Chatterbox (keep `[laugh]` tags) + expression for Live2D |
| `requirements.txt` | Add `torchao`, `onnxruntime-gpu`, `chatterbox-onnx` |

## Unchanged Files

- All frontend files (Live2D, Web UI, Desktop)
- `AudioService` and Silero VAD
- Existing `ConversationPipeline` and `OmniPipeline` (kept as fallback modes)
- Character loader utility
- Language detection utility

---

## Configuration

### `config.yaml` — New Sections

```yaml
mode: "gemma-omni"

gemma:
  model_id: "google/gemma-4-E4B-it"
  quantization: "int4"
  device: "cuda"
  dtype: "bfloat16"
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.95
  context_max_turns: 10
  vision:
    enabled: true
    token_budget: 140
    detail_token_budget: 1120
  screen:
    enabled: true
    interval: 2.0
    max_buffer: 30
    change_threshold: 0.05
    include_in_conversation: true

tts:
  provider: "chatterbox"
  chatterbox:
    model_id: "onnx-community/chatterbox-multilingual-ONNX"
    quantized: true
    exaggeration: 0.5
    cfg_weight: 0.5
  stream_tts: true
  auto_detect_language: true

asr:
  provider: "gemma-native"
```

### Character YAML — Chatterbox Voice Config

```yaml
voice:
  chatterbox_ref_audio: "resources/voices/march7th/reference.wav"
  chatterbox_exaggeration: 0.6
  chatterbox_language: "fr"
```

---

## VRAM Management

### Budget

| Component | VRAM |
|---|---|
| Gemma E4B int4 (weights + encoders) | ~4.1 GB |
| Gemma KV cache (4K context) | ~0.2 GB |
| Chatterbox ONNX Q4 (all components) | ~2.0 GB |
| CUDA runtime + PyTorch | ~1.0 GB |
| **Total** | **~7.3 GB** |
| **Headroom on 12 GB** | **~4.7 GB** |

### Optimization Rules

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8` set at top of `src/assistant/app.py` and `src/server/app.py` (before any torch import, same pattern as existing `HF_HUB_OFFLINE` env var)
2. Context limited to 10 turns (~4K tokens max) — prevents KV cache explosion
3. `torch.no_grad()` wraps all inference
4. `torch.cuda.empty_cache()` called between Gemma and Chatterbox inference passes
5. SDPA/FlashAttention for attention computation
6. Chatterbox runs on ONNX Runtime (separate from PyTorch memory pool)
7. VAD + screen capture on CPU (zero GPU competition)
8. Vision token budget defaults to 140 (not 1120)

### OOM Fallback Chain

1. Reduce `max_new_tokens` (256 → 128)
2. Reduce vision `token_budget` (140 → 70)
3. Disable screen capture temporarily
4. Last resort: model swap (unload Gemma → TTS → reload)

---

## Emotion Pipeline

### Flow

```
LLM output: "C'est genial *excited* j'adore [laugh] !"
                    │
            EmotionDetector (extended)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  TTS text:               Live2D expression:
  "C'est genial            "excited"
   j'adore [laugh] !"      → mapped to "开心"
```

**Rules:**
- **Chatterbox tags allowlist:** `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]` → **kept** in TTS text (these are the ONLY bracket tags NOT stripped)
- **Live2D markers:** `*excited*`, `(happy)`, `<blush>` → **removed** from TTS text, routed to Live2D expression
- **Bracket disambiguation:** The `EmotionDetector.strip_markers()` method is updated with an explicit `CHATTERBOX_TAGS` allowlist. Brackets matching the allowlist are preserved; all others are stripped.
- System prompt instructs Gemma to use `[laugh]` etc. naturally in conversation

```python
# In EmotionDetector (updated)
CHATTERBOX_TAGS = {"laugh", "chuckle", "cough", "sigh"}

def strip_markers_for_tts(self, text: str) -> str:
    """Strip emotion markers but KEEP Chatterbox tags."""
    result = text
    # Remove (happy), *excited*, <blush>
    result = re.sub(r'\((\w+)\)', '', result)
    result = re.sub(r'\*(\w+)\*', '', result)
    result = re.sub(r'<(\w+)>', '', result)
    # Remove [brackets] EXCEPT Chatterbox tags
    result = re.sub(
        r'\[(\w+)\]',
        lambda m: m.group(0) if m.group(1).lower() in self.CHATTERBOX_TAGS else '',
        result
    )
    return re.sub(r'\s+', ' ', result).strip()
```

**Note:** The existing `EmotionDetector` at `src/utils/emotion_detector.py` is the canonical implementation. The inline duplicate in `conversation_pipeline.py` should be refactored to import from `emotion_detector.py` as part of Phase 2.

---

## Implementation Phases

### Phase 1 — Foundation ("it speaks")

- GemmaProvider: load + chat + audio input
- ChatterboxTTSProvider: load + synthesize + voice cloning
- GemmaOmniPipeline: wire audio → Gemma → Chatterbox → audio output
- Config: gemma + chatterbox sections
- App: add gemma-omni mode
- **Milestone:** Speak into mic → March 7th voice responds

### Phase 2 — Streaming & Emotions ("it's fast and expressive")

- GemmaProvider: chat_stream (token-by-token)
- Sentence splitter + streaming LLM → TTS
- EmotionDetector: dual output (Live2D + Chatterbox tags)
- March 7th system prompt with emotion tag instructions
- VRAMMonitor
- **Milestone:** < 2s latency, avatar expressions change, natural laughs

### Phase 3 — Vision / Screen Sharing ("it sees your screen")

- ScreenBuffer: continuous capture + pixel diff
- GemmaProvider: image input + audio+image combined prompt
- Pipeline: inject screenshot in every conversation turn
- 3 vision modes (passive / replay / detail)
- **Milestone:** "What's on my screen?" → correct description

### Phase 4 — Frontend Integration ("everything connected")

- Server: WebSocket routing to GemmaOmniPipeline
- Server: progressive model loading messages
- Desktop: app.py wired to gemma-omni
- Web UI: settings panel for chatterbox params
- **Milestone:** Web UI + Desktop overlay fully functional

### Phase 5 — Polish ("it's smooth")

- Filler words while Gemma thinks
- VAD tuning (reduce silence detection delay)
- TTS prompt caching (pre-compute voice reference)
- KV cache reuse between turns
- OOM graceful fallback
- F3 interrupt hotkey (existing TODO)
- **Milestone:** Fluid, reliable conversation experience

---

## Dependencies to Add

```
# requirements.txt additions
torchao                    # TorchAO int4 quantization for Gemma
onnxruntime-gpu            # ONNX Runtime with CUDA for Chatterbox
chatterbox-onnx            # Chatterbox ONNX inference package
```

---

## Error Handling & Resilience

### Streaming Pipeline Failures

The streaming LLM→TTS pipeline can fail mid-sentence. Each failure mode has a defined recovery:

| Failure | Detection | Recovery |
|---|---|---|
| Gemma inference timeout (>30s) | `asyncio.wait_for` timeout | Cancel generation, send partial response to TTS, log warning |
| Gemma OOM mid-generation | `torch.cuda.OutOfMemoryError` | Clear KV cache, reduce `max_new_tokens`, retry once |
| Chatterbox TTS hangs | Watchdog timer (10s per sentence) | Skip sentence, continue with next, log error |
| Chatterbox produces silence | Check audio RMS < threshold | Re-synthesize with lower exaggeration, or fallback to Kokoro |
| ScreenBuffer capture fails | Exception in mss call | Disable vision for this turn, continue audio-only |

### Lifecycle Management

```python
class GemmaOmniPipeline:
    async def startup(self):
        # Load models with progress callbacks
        # Verify VRAM budget before loading second model
        
    async def shutdown(self):
        # Stop ScreenBuffer thread
        # Unload Chatterbox (del + gc + empty_cache)
        # Unload Gemma (del + gc + empty_cache)
        # Final VRAM cleanup
        
    async def health_check(self) -> dict:
        # Return VRAM usage, model status, buffer status
```

### Init Fallback

If `gemma-omni` mode fails to initialize (model download error, ONNX not installed, VRAM insufficient):
1. Log error with clear message
2. Automatically fall back to `pipeline` mode
3. Notify user via WebSocket: `{"type": "mode_fallback", "from": "gemma-omni", "to": "pipeline", "reason": "..."}`

---

## Testing Strategy

### Phase 1 Smoke Tests (before moving to Phase 2)

| Test | Command / Check | Pass Criteria |
|---|---|---|
| Gemma loads in int4 | `GemmaProvider.load()` | VRAM < 6GB, no errors |
| Gemma text chat | `chat("Bonjour")` | Coherent French response |
| Gemma audio input | `chat(audio=5s_wav)` | Correct transcription + response |
| Gemma image input | `chat(images=[screenshot])` | Describes screen content |
| Gemma audio+image | `chat(audio=wav, images=[img])` | Responds to both modalities |
| Chatterbox loads ONNX Q4 | `ChatterboxTTSProvider.load()` | Combined VRAM < 9GB |
| Chatterbox voice clone | `synthesize("Bonjour", ref=march7th.wav)` | Audio in March 7th voice |
| Chatterbox emotion tags | `synthesize("Ha [laugh] fun!")` | Audible laugh in output |
| Full pipeline | Mic → Gemma → Chatterbox → speaker | End-to-end works |
| VRAM stress | 20 consecutive conversation turns | No OOM, VRAM stable |

### Critical Validation (Phase 1, Task 1.2)

**If Gemma E4B audio input does not work** (model is 2 days old, audio support confirmed in docs but untested in practice):
1. Immediately switch to: Faster-Whisper small (CPU) for ASR + Gemma text+image only
2. VRAM impact: Whisper on CPU = 0 GPU, total VRAM unchanged
3. Latency impact: +300-500ms for Whisper ASR step
4. Update config: `asr.provider: "whisper"` instead of `"gemma-native"`

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Gemma E4B audio input broken in practice | Medium (docs confirm, but model is 2 days old) | **High** | Phase 1 validation gate. Fallback: Whisper ASR (CPU) + Gemma text+image. Clear trigger: if audio fails in task 1.2, switch immediately. |
| Gemma E4B audio+image combo fails | Low (prompt format `<\|audio\|>` + `<\|image\|>` confirmed by Google) | High | Fallback: process audio and image in separate calls (2x latency on vision turns) |
| Chatterbox Multilingual ONNX Q4 quality insufficient | Medium | Medium | Fallback: Kokoro (already integrated) or Chatterbox full precision |
| **ONNX Runtime GPU + PyTorch CUDA coexistence** | Medium | Medium | ORT creates separate CUDA allocator. May add 0.5-1GB overhead. Monitor via VRAMMonitor. If conflict: run Chatterbox on CPU (32GB RAM available). |
| **Chatterbox ONNX Q4 package not available** | Low-Medium | High | Verify `onnx-community/chatterbox-multilingual-ONNX` exists on HuggingFace before starting Phase 1. Fallback: use PyTorch Chatterbox with manual quantization. |
| VRAM exceeds 12GB under load | Low (~7GB estimated, but KV cache + ORT overhead could add 1-2GB) | High | OOM fallback chain + VRAMMonitor alerts. Real-world VRAM likely ~8-9GB with overhead. |
| Gemma E4B tooling immature (2 days old) | **High** | **High** | Pin transformers + torchao versions, test thoroughly, keep MiniCPM-o AND pipeline as fallback modes |
| French voice cloning quality poor | Medium | Low | Test with March 7th reference, adjust exaggeration, try multiple refs |
| FlashAttention not available on Windows | Medium | Low | SDPA fallback (default in PyTorch, slightly less efficient) |
| **Windows-specific issues (TorchAO, ONNX RT)** | Medium | Medium | TorchAO int4 and ONNX Runtime GPU less tested on Windows. Pin known-working versions. Test early in Phase 1. |

---

## References

- [Gemma 4 E4B Model Card](https://huggingface.co/google/gemma-4-E4B-it)
- [Gemma 4 Prompt Formatting (audio+image)](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- [Chatterbox Multilingual ONNX](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [TorchAO Quantization](https://github.com/pytorch/ao)
- [Unsloth Gemma 4 Guide](https://unsloth.ai/docs/models/gemma-4)
- [Open-LLM-VTuber (reference architecture)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
