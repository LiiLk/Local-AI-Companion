# Desktop Migration: Tauri 2

## Why we are migrating

The current `pywebview` shell is good enough to validate the voice pipeline, but it is the wrong foundation for a real Windows desktop avatar.

Observed limitations:

- white host background / weak transparency behavior on Windows
- overlay still feels like a floating web page
- difficult path to click-through and native desktop-pet behavior

Official references:

- pywebview API: `transparent=True` is not supported on Windows  
  <https://pywebview.flowrl.com/api/>
- pywebview changelog: transparent window support is explicitly not available on Windows, with historical EdgeChromium caveats  
  <https://pywebview.flowrl.com/changelog>
- Tauri 2 window customization guide  
  <https://v2.tauri.app/learn/window-customization/>
- Community reference for a real desktop pet built on Tauri 2  
  <https://github.com/liwenka1/bongo-cat-next>

## Target architecture

### Keep

- Python backend for ASR / LLM / TTS
- existing conversation pipeline
- existing Live2D frontend logic

### Replace

- `pywebview` desktop shell

### Add

- Tauri 2 window host
- desktop bridge abstraction for commands/events
- later: a dedicated native asset loading strategy for Live2D resources

## Migration milestones

### Milestone 1: bridge abstraction

- introduce a frontend bridge that can talk to either `pywebview` or `Tauri`
- keep current desktop frontend usable during migration

### Milestone 2: Tauri shell scaffold

- add Tauri config, Rust entry point, package.json
- mirror the desktop command surface:
  - `send_text`
  - `interrupt`
  - `toggle_mute`
  - `get_runtime_state`
- `toggle_debug`

Status:

- done for the scaffold
- current implementation loads the existing `frontend/live2d` shell in Tauri

### Milestone 3: backend wiring

- replace placeholder Rust commands with a real bridge to Python
- recommended first transport: local websocket or local IPC

Status:

- first pragmatic transport is now local WebSocket from the frontend directly to the Python backend
- this avoids blocking the migration on a Rust ↔ Python sidecar too early
- Rust-native command transport remains an optional later refinement

### Milestone 4: desktop-native overlay

- split into two windows:
  - avatar-only transparent window
  - compact controls / subtitles window
- add click-through for non-interactive regions
- add system tray and desktop-pet behaviors

## Notes

This migration should not change the validated ASR / LLM / TTS backend first. The shell is the concern, not the conversation stack.
