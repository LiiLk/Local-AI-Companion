# Tauri 2 Desktop Shell

This directory is the Tauri 2 migration scaffold for the desktop avatar shell.

## Goal

Replace the current `pywebview` overlay with a Windows-native desktop shell that can support:

- frameless transparent window
- always-on-top avatar
- better control over click-through / drag regions
- cleaner split between avatar UI and control UI

The Python ASR / LLM / TTS backend is intentionally kept separate. This scaffold only prepares the desktop shell.

## Current Status

- `src-tauri/` exists and mirrors the desktop bridge commands used by the current frontend.
- `frontend/live2d/desktop-bridge.js` now abstracts `pywebview` vs `Tauri`.
- In Tauri mode, the frontend talks to the desktop assistant bridge on `ws://127.0.0.1:8765`.
- The microphone/VAD stay on the Python desktop backend. Tauri is only the transparent shell.
- The Rust commands are still placeholders. They are not on the critical path for the current shell iteration.

## Prerequisites

- Node.js 20+
- a package manager (`npm` is enough)
- Rust toolchain (`rustup`, `cargo`)
- WebView2 runtime on Windows

## First Run

Start the desktop backend first:

```bash
cd /mnt/c/Users/Khalil/Documents/Local-AI-Companion
venv/Scripts/python.exe run_assistant.py --bridge-server --bridge-port 8765
```

Then, from this folder:

```bash
npm install
npm run dev
```

## Important Limitation

This scaffold now points `frontendDist` directly to `frontend/live2d`.
For reliability, the Live2D shell reads its SDK/model files from `frontend/live2d/runtime-assets/` instead of the shared top-level `/assets` tree.
The current March 7th runtime pack is also normalized for Tauri (`march7th_tauri`) to avoid Windows/dev-server issues with space-heavy asset paths.
That keeps the Tauri shell self-contained and avoids the current Windows asset-resolution issues.

The initial Tauri shell now reuses the direct desktop backend path for mic/VAD/ASR/LLM/TTS.
That keeps the transparent shell and the small menu without reintroducing the heavier web backend.
