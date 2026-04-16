# AGENTS.md — Shared AI Agent Instructions

This file is read automatically by Codex, Claude Code, and other AI coding agents.

## Project Overview

**Local-AI-Companion** is an offline-first AI assistant with Live2D avatar, voice conversation (ASR + LLM + TTS), and optional vision. The goal is a fluid, real-time conversational experience running entirely on local hardware (RTX 4070-class GPU).

## Architecture

```
Microphone → VAD (Silero) → ASR (Qwen 3 ASRt) → LLM (Gemma/Ollama) → TTS (Qwen3-TTS) → Audio Playback
                                                                              ↑
                                                                     Optional RVC post-processing
```

### Three modes
- **pipeline** (`mode: "pipeline"`): Separate ASR → LLM → TTS chain. Default.
- **omni** (`mode: "omni"`): MiniCPM-o handles everything in one model.
- **gemma-omni** (`mode: "gemma-omni"`): Gemma E2B (ASR+LLM+Vision) + Chatterbox TTS.

### Three entry points
- `main.py` — CLI chatbot
- `run_assistant.py` / `src/assistant/app.py` — Live2D desktop overlay (pywebview)
- `src/server/app.py` — FastAPI + WebSocket server for browser frontend

## Key Files

| Area | Files |
|------|-------|
| **ASR** | `src/asr/base.py`, `src/asr/whisper_provider.py` |
| **LLM** | `src/llm/base.py`, `src/llm/ollama_llm.py`, `src/llm/gemma_text_vision_llm.py` |
| **TTS** | `src/tts/base.py`, `src/tts/qwen3_tts_provider.py`, `src/tts/kokoro_provider.py`, `src/tts/edge_tts_provider.py`, `src/tts/chatterbox_provider.py` |
| **TTS Task Manager** | `src/tts/tts_task_manager.py` — async queue decoupling LLM stream from TTS |
| **VAD** | `src/vad/silero_vad.py` |
| **Pipeline** | `src/assistant/conversation_pipeline.py` |
| **WebSocket** | `src/server/websocket.py` |
| **Sentence Splitting** | `src/utils/sentence_splitter.py` |
| **Config** | `config/config.yaml`, `config/characters/*.yaml` |
| **Tests** | `tests/` |

## Coding Conventions

- **Language**: Python 3.11+. Type hints everywhere.
- **Async**: All pipelines are async. Blocking ops (model inference, TTS, ASR) go through `loop.run_in_executor()`.
- **GPU safety**: Only one CUDA operation at a time on a single GPU. TTS synthesis is sequential per response — never parallelize TTS calls on the same GPU.
- **TTS providers**: All implement `BaseTTS` (see `src/tts/base.py`). The `synthesize()` method is the main entry point.
- **Config-driven**: Behavior is controlled via `config/config.yaml`. Avoid hardcoded values.
- **No over-engineering**: Keep solutions simple, pragmatic, and maintainable.
- **Error handling**: Log errors, fall back gracefully. Don't crash the pipeline on TTS/ASR failures.

## Current Optimizations (April 2025)

The pipeline uses a `TTSTaskManager` pattern to decouple LLM streaming from TTS:
- LLM tokens stream into a `SentenceSplitter` (with `faster_first_response` for quick first audio)
- Complete sentences are queued to a TTS worker task
- TTS worker synthesizes sequentially (GPU-safe) and delivers via callbacks
- The LLM stream **never blocks** while TTS is running

VAD end-of-speech detection uses ~480ms silence threshold in pipeline mode (configurable via `pipeline.vad_required_misses` in config.yaml).

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_sentence_splitter.py -v
```

## Branch & PR Discipline

These rules are mandatory for all AI agents working on this repository.

- **Do not use one branch for multiple unrelated topics.** Keep each branch focused on one technical goal only (example: TTS stability, ASR accuracy, websocket cleanup, frontend redesign).
- **Do not merge a giant mixed branch into `main`.** If a branch has accumulated unrelated work, create a new clean branch from `main` and cherry-pick only the intended commits.
- **Backend, frontend, desktop, docs, and assets should not be mixed in the same PR unless the change genuinely requires all of them.**
- **Before proposing a PR to `main`, the agent must check**:
  - the diff scope against `main`
  - the current worktree status
  - the relevant automated tests
  - any known stability risks not covered by tests
- **If the branch scope is too broad, the agent must say so clearly** and recommend splitting or creating an integration branch instead of forcing a risky PR.
- **Prefer small, reviewable PRs** that can be tested and reverted easily.
- **Never hide uncertainty.** If the system seems stable only on a subset of flows, say exactly which flows were validated and which were not.

## Sub-Agent Workflow

These rules are mandatory when using Codex sub-agents on this repository.

- **Use specialized sub-agents for important or risky work.** Especially when a task touches runtime orchestration, `src/server/websocket.py`, `src/assistant/conversation_pipeline.py`, desktop turn control, provider preload/fallback, GPU sequencing, or cross-mode behavior.
- **Do not create sub-agents by job title alone.** Prefer technical ownership such as runtime orchestration, realtime transport, voice backends, desktop shell, or quality review.
- **Default to direct work for small tasks.** If a change is limited to one simple flow or a couple of files, the main agent should usually handle it directly.
- **Use sub-agents for parallel exploration more than parallel editing.** Avoid having multiple agents edit the same hotspot files at once.
- **Cap active builder sub-agents at two per initiative.** One main agent must remain responsible for decomposition, integration, validation, and the final PR.
- **One ticket, one branch, one PR.** Sub-agents must not turn a focused branch into a mixed branch with unrelated changes.
- **Always report validation boundaries clearly.** Any sub-agent handoff must state files touched, tests run, manual checks performed, and known risks or unvalidated flows.

## Important Constraints

- **Everything local** (except optionally the LLM later). No cloud APIs for ASR or TTS.
- **Voice cloning required** — Qwen3-TTS is the default TTS because it supports custom voices (e.g., March 7th).
- **Single GPU** — ASR, LLM, and TTS may share one GPU. Avoid concurrent CUDA operations.
- **Windows primary** — Development is on Windows 11. Use forward slashes in code paths, but be aware of Windows-specific issues (DLL conflicts, subprocess handling).
