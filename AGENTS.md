# AGENTS.md — Shared AI Agent Instructions

This file is read automatically by Codex, Claude Code, and other AI coding agents.

## Project Overview

**Local-AI-Companion** is a Windows-first, offline-first AI companion with:
- Live2D desktop shell
- realtime voice conversation
- local ASR + local TTS by default
- optional cloud LLM support when explicitly enabled
- optional multimodal / premium paths that are not the main production architecture

The project goal is not “support every model equally”. The goal is a **stable, debuggable, realtime desktop voice companion** on a single consumer GPU.

## Source Of Truth

When in doubt, use these as the current truth of the project:

1. `config/config.yaml`
2. `src/assistant/pipeline_runtime.py`
3. `src/assistant/conversation_pipeline.py`
4. `src/assistant/app.py`
5. `src/server/websocket.py`
6. `README.md`

Do **not** assume older experimental branches, stale comments, or old provider code represent the intended architecture.

## Current Architecture (Mainline / Stable)

The current stable path on `main` is:

```text
Microphone -> Silero VAD -> Faster-Whisper -> LLM text -> Kokoro TTS -> RVC -> Audio playback
```

### What “stable” means here

- `mode: "pipeline"`
- `ASR: whisper`
- `LLM: ollama` by default
- `TTS: kokoro`
- `RVC: enabled`
- desktop + websocket paths share the same runtime construction logic
- preload / warmup / cleanup are centrally managed

### Current defaults from `config/config.yaml`

- `mode: "pipeline"`
- `pipeline.reply_language: "en"`
- `llm.provider: "ollama"`
- `llm.ollama.model: "qwen3.5:4b"`
- `tts.provider: "kokoro"`
- `tts.rvc.enabled: true`
- `asr.provider: "whisper"`
- `asr.model_size: "small"`
- `asr.device: "cuda"`

## Primary vs Secondary Paths

### Primary product path

This is the path agents should optimize for first:

- `pipeline`
- `faster-whisper`
- `Ollama` local text model
- `Kokoro -> RVC`
- desktop overlay + websocket backend

### Secondary / optional paths

These exist in the repo, but they are **not** the main architecture:

- `OpenRouter` as optional LLM provider
- `Qwen3-TTS` as premium local TTS path
- `Qwen3-ASR`
- `Chatterbox`
- `Edge TTS`
- `mode: "omni"` (`MiniCPM-o`)
- `mode: "gemma-omni"`

Agents may improve these paths, but must not accidentally make them the implicit default.

## Important Architectural Decisions (April 2026)

These decisions were made deliberately and should be preserved unless there is strong evidence to change them.

### 1. One stable voice path beats many half-stable ones

The repository used to drift toward multiple competing “default” pipelines. That is no longer acceptable.

The main path is:
- `ASR -> LLM text -> Kokoro -> RVC`

If you add or improve another path, keep it clearly secondary.

### 2. `Kokoro -> RVC` is the realtime default

Why:
- low latency
- simpler operational profile than Qwen3-TTS for realtime desktop use
- good enough quality with character voice conversion
- more stable on a single GPU desktop setup

### 3. `Qwen3-TTS` is not the default production path

It remains useful for premium local voice experiments, but it is not the baseline the project is currently simplified around.

### 4. `Whisper` is the default ASR path

Current default is still `whisper small` for stability and cost.
Future upgrades such as `whisper-large-v3` or `large-v3-turbo` are expected evaluation paths, but not yet the default.

### 5. `PipelineRuntime` owns backend construction and lifecycle

Creation, preload, warmup, cleanup, readiness, and degraded/error state handling were intentionally centralized.

Agents should avoid reintroducing duplicated provider wiring in multiple entry points.

## Runtime Ownership

The shared runtime owner is:

- `src/assistant/pipeline_runtime.py`

This file is the canonical place for:
- creating LLM / TTS / ASR / RVC providers
- building `ConversationConfig`
- preload / warmup policy
- cleanup of runtime services
- backend readiness / degraded state resolution

### Rule

If you need to change how pipeline services are created or managed, update `pipeline_runtime.py` first, then adapt callers.

Do **not** fork the construction logic separately in:
- `src/assistant/app.py`
- `src/server/websocket.py`
- new scripts / side entry points

## Entry Points

### Canonical runtime entry points

- `run_assistant.py` / `src/assistant/app.py`
  - desktop overlay / desktop bridge
- `src/server/app.py`
  - FastAPI + WebSocket browser backend
- `src/server/websocket.py`
  - browser realtime orchestration and payload delivery

### CLI path

- `main.py`

`main.py` is still useful for smoke tests and manual CLI usage, but it is a simpler path and should **not** be treated as the canonical source of backend architecture truth.

For architecture-level changes, prioritize:
- `src/assistant/*`
- `src/server/*`
- `config/config.yaml`

## Conversation Pipeline

The core pipeline lives in:
- `src/assistant/conversation_pipeline.py`

Important behaviors already in place:
- sentence-level TTS streaming
- `TTSTaskManager` decoupling LLM streaming from TTS synthesis
- optional RVC post-processing with safe fallback to original audio
- language strategy driven by config
- support for reply-language enforcement before TTS output

### Reply language

Pipeline mode currently uses:
- `pipeline.reply_language: "en"`

Meaning:
- user can speak French or another supported language
- the system understands the input
- the configured reply language can still be English

Agents must not assume “reply in the same language as input” is the runtime truth just because a character prompt says so.

If there is a conflict between character text and pipeline config, the **runtime config** wins.

## TTS Stack

### Stable TTS path

Primary files:
- `src/tts/kokoro_provider.py`
- `src/tts/rvc_provider.py`
- `src/tts/tts_task_manager.py`

Important behavior:
- TTS synthesis must stay sequential on a single GPU
- RVC is optional at runtime, but enabled by default in config
- if RVC fails, audio should degrade gracefully to the base TTS output

### Routed / fallback TTS

Relevant file:
- `src/tts/routed_provider.py`

This exists for optional or fallback provider routing, but agents should not treat routing complexity as the product center.
The product center is still `Kokoro -> RVC`.

### Premium / experimental TTS

Relevant files:
- `src/tts/qwen3_tts_provider.py`
- `src/tts/chatterbox_provider.py`
- `src/tts/edge_tts_provider.py`

These should remain clearly secondary unless the project explicitly changes direction.

## ASR Stack

### Stable ASR path

Primary files:
- `src/asr/whisper_provider.py`
- `src/asr/base.py`

Current default:
- `faster-whisper`
- `model_size: small`
- `device: cuda`
- `compute_type: float16`
- `beam_size: 3`

### Advanced / secondary ASR

Relevant file:
- `src/asr/qwen3_asr_provider.py`

This is not the current production baseline.

## Config Loading Rules

Use:
- `src/utils/config_loader.py`
- `load_yaml_config(...)`

All major entry points were aligned to use the shared loader so that:
- `config/config.yaml` provides tracked defaults
- `config/config.local.yaml` provides local, ignored overrides

### Rule

Do **not** read YAML directly with `yaml.safe_load(...)` in a new entry point if the intention is to load app config.
Use `load_yaml_config(...)` so local overrides are respected.

## Character System

Character presets live in:
- `config/characters/*.yaml`

Current default preset:
- `march7th`

Typical character concerns:
- system prompt / personality
- Live2D mappings
- voice hints such as Kokoro voice or reference audio paths

### Important nuance

Character presets can influence style and voice hints, but they should not silently redefine core pipeline behavior already controlled in `config/config.yaml`.

Example:
- if `pipeline.reply_language` is set, agents must reason from that first

## Desktop / Frontend Notes

Desktop backend lives in:
- `src/assistant/app.py`

Frontend/browser backend lives in:
- `src/server/app.py`
- `src/server/websocket.py`
- `frontend/`

Notable recent stabilization work already landed:
- shared runtime ownership
- backend ready / warming / degraded / error state reporting
- bridge-server mode for desktop shell integration
- better consistency between desktop and websocket runtime behavior
- Live2D frontend status handling improvements

## GPU And Performance Constraints

This project targets a single consumer GPU desktop.

### Hard rules

- Do not parallelize multiple heavy CUDA TTS operations at once.
- Be very cautious when adding concurrent GPU workloads.
- Prefer sequential TTS and explicit queues.
- Avoid duplicate model loads across processes unless the tradeoff is deliberate.
- Warmup is acceptable when it improves steady-state responsiveness, but be honest about startup cost.

### Operational reality

A “good PC” can still lag badly if:
- multiple Python workers duplicate model memory
- Torch / ONNX / CUDA initialize at the same time
- desktop app, browser, NVIDIA Broadcast, and test workers all run together

Agents should optimize for the actual desktop user experience, not only benchmark throughput.

## Error Handling Expectations

- Log clearly.
- Prefer degraded mode over crashes.
- Fail open for optional subsystems when possible.
- TTS / RVC failures should fall back gracefully.
- Runtime state should become `degraded` or `error` explicitly rather than hanging silently.

## Known Cleanup Direction

The codebase is intentionally being simplified around one stable path.

Agents should support that cleanup direction.

### Keep

- `pipeline` mode
- `PipelineRuntime`
- `TTSTaskManager`
- `Kokoro -> RVC`
- `faster-whisper`
- config-driven character setup
- desktop + websocket shared runtime behavior

### Treat as secondary

- `Qwen3-TTS`
- `Qwen3-ASR`
- `Chatterbox`
- `Edge TTS`
- `omni`
- `gemma-omni`

### Avoid

- reintroducing multiple competing “default” architectures
- spreading provider construction logic across files
- merging broad mixed-purpose branches into `main`
- treating old experiments as if they were current product decisions

## Testing

Run the full suite when the touched area is broad:

```bash
pytest tests -q
```

Useful focused tests:

```bash
pytest tests/test_pipeline_runtime.py -q
pytest tests/test_conversation_pipeline_rvc.py -q
pytest tests/test_conversation_pipeline_tts_strategy.py -q
pytest tests/test_websocket_openrouter.py -q
pytest tests/test_whisper_provider.py -q
pytest tests/test_tts_task_manager.py -q
```

If you changed docs only, say so clearly instead of pretending runtime was validated.

## Branch & PR Discipline

These rules are mandatory for all AI agents working on this repository.

- **Do not use one branch for multiple unrelated topics.** Keep each branch focused on one technical goal only.
- **Do not merge a giant mixed branch into `main`.** If a branch has accumulated unrelated work, create a new clean branch from `main` and cherry-pick only the intended commits.
- **Backend, frontend, desktop, docs, and assets should not be mixed in the same PR unless the change genuinely requires all of them.**
- **Before proposing a PR to `main`, the agent must check**:
  - diff scope against `main`
  - current worktree status
  - relevant automated tests
  - known stability risks not covered by tests
- **If the branch scope is too broad, the agent must say so clearly** and recommend splitting or creating an integration branch instead of forcing a risky PR.
- **Prefer small, reviewable PRs** that can be tested and reverted easily.
- **Never hide uncertainty.** If the system seems stable only on a subset of flows, say exactly which flows were validated and which were not.

## Important Constraints

- **Offline-first, not cloud-forbidden.** Local ASR and TTS are the baseline. Cloud LLM use via OpenRouter is optional and supported.
- **Voice cloning is required**, but the current stable implementation is `Kokoro -> RVC`, not “Qwen3 everywhere by default”.
- **Single GPU** — ASR, LLM, and TTS may share one GPU. Avoid concurrent CUDA-heavy operations.
- **Windows primary** — Development is on Windows 11. Use forward slashes in code paths where appropriate, but account for Windows-specific subprocess, DLL, and venv behavior.
- **Config over hardcoding** — if a behavior should vary by machine or deployment, put it in config or a local override.
