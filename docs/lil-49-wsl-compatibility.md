# LIL-49 — PLAT-01 WSL2 compatibility (Windows + WSL)

**Linear:** [LIL-49](https://linear.app/lilkorp/issue/LIL-49/plat-01-compatibilite-wsl2-chemins-workers-audio-docs-en-plus-de)
**Branch:** `feature/lil-49-wsl-run-path`
**Status:** In progress (option A hybrid pet wired + documented)
**Related:** [LIL-48](https://linear.app/lilkorp/issue/LIL-48) Parakeet ASR — **standby** until platform path is stable
**Last updated:** 2026-07-11

This note follows the same ticket-doc pattern as `docs/lil-35-voice-benchmark.md` and `docs/python-dependency-audit.md` (LIL-40): problem, scope, how to run, acceptance, non-goals.

---

## Why

Local AI Companion is **Windows-first** for the desktop avatar (Qt / Tauri / Live2D).
Agent development (Grok Build) and some contributors run the repo under **WSL2**.

Breakage observed on WSL:

| Area | Symptom | Root cause |
|---|---|---|
| Venv | `venv/Scripts/activate` missing | Linux layout is `venv/bin/activate` |
| Workers (RVC / Qwen) | Python executable not found | Config used `.\venv\Scripts\python.exe` |
| Process cleanup | Windows-only `taskkill` | No POSIX process-group kill |
| Microphone | `sounddevice` reports **0** devices | ALSA→Pulse plugin, WSLg, or Windows mic permission missing |
| Live2D | QtWebEngine blocks WebGL after EGL/Zink errors | WSLg has no DRM render node usable by Chromium |

CUDA is supported by WSL2 when the compatible NVIDIA driver is installed on
Windows. The project diagnostic reports the effective PyTorch CUDA state; it
does not install or modify GPU drivers.

---

## What we keep / change

### Keep

- Windows as primary **desktop product** target
- Stable path: `pipeline` + Whisper + Kokoro + RVC
- Ticket-linked docs under `docs/lil-*.md`
- Optional advanced providers unchanged in behavior

### Change (this ticket)

- Cross-platform path helper: `src/utils/platform_compat.py`
- Workers resolve Python via Scripts↔bin mapping
- Explicit worker environments in `config/config.yaml`, mapped `Scripts↔bin` without unsafe fallback
- Entry points call `ensure_wsl_desktop_env()` before audio/Qt imports
- WSL defaults to deterministic Qt/Mesa/Chromium CPU WebGL for the bundled local Live2D page
- Contributor docs for WSL + Windows side by side
- Diagnostic: `scripts/check_platform.py`
- Idempotent WSL bootstrap: `scripts/setup_wsl.sh`
- Isolated WSL RVC installer: `scripts/install_rvc_wsl.sh`
- **Option A hybrid pet:** WSL backend + Windows-native transparent Qt shell (see below)

### Non-goals

- Full native Linux desktop-pet parity (transparency, global hotkeys) under WSLg alone
- Finishing LIL-48 Parakeet bakeoff (see standby note on LIL-48)
- Shipping WSL installers for every experimental provider; Qwen/Gemma/MiniCPM remain opt-in

---

## Changelog — what shipped on this branch

### Phase 1 — WSL run path (paths / workers / scripts)

| Artifact | Change |
|----------|--------|
| `src/utils/platform_compat.py` | **New.** Detect WSL, map `Scripts↔bin`, `kill_process_tree`, `ensure_wsl_desktop_env`, project root helpers |
| `src/tts/rvc_provider.py` | Resolve worker Python via platform helper; POSIX process-group kill |
| `src/tts/qwen3_tts_provider.py` | Same as RVC |
| `src/asr/qwen3_asr_provider.py` | Same as RVC |
| `src/assistant/audio_service.py` | WSL-friendly audio defaults / Pulse hints |
| `config/config.yaml` | Portable relative worker paths (`venv/Scripts/...` maps on WSL) |
| `run_assistant.py` | Call platform env setup early on WSL |
| `scripts/check_platform.py` | **New.** Diagnostic (platform, CUDA, audio devices) |
| `scripts/setup_wsl.sh` | **New.** Idempotent WSL bootstrap |
| `scripts/install_rvc_wsl.sh` | **New.** Optional RVC overlay on WSL |
| `scripts/run_wsl.sh` | **New.** Secondary modes: `web` (default), `cli`, `cli-voice`, `bridge`, `desktop`/`wslg`, `check` |
| `tests/test_platform_compat.py` | **New.** Path + process helper unit tests |
| `tests/test_audio_service.py` | WSL-related audio coverage |
| `README.md` | Dual path (Windows / WSL hybrid / WSL web) |
| `docs/lil-49-wsl-compatibility.md` | **This file** |

### Phase 2 — Option A: hybrid desktop pet (WSL brain + Windows shell)

Decision: real desktop-incrusted pets need a **native Windows** transparent window (Qt/Tauri), not a browser tab and not WSLg software GL. Research references: Open-LLM-VTuber-style pets, Microsoft WSL localhost forwarding + interop.

| Artifact | Change |
|----------|--------|
| `src/utils/wsl_hybrid_ui.py` | **New.** Find Windows Python, stage the selected shell/model in ignored `.runtime/`, launch it through WSL interop, and own its cleanup |
| `scripts/windows_pet_shell.py` | **New.** Windows-only entry: Qt transparent pet + `BridgeProxyAssistant` |
| `src/desktop/bridge_proxy.py` | **New.** WebSocket client duck-typing `Live2DAssistant` for the Qt shell (commands + frontend events) |
| `desktop/qt_avatar_shell.py` | Optional `page_url` for remote/static page load |
| `src/assistant/app.py` | On WSL: default `hybrid_windows_ui=True` (bridge + launch Windows pet). Flags: `--hybrid-windows-ui`, `--force-wsl-desktop`, `LOCAL_AI_FORCE_WSL_DESKTOP=1` |
| `run_assistant.py` | Single Windows/WSL desktop entry point; WSL automatically selects the hybrid shell |
| `tests/test_wsl_hybrid_ui.py` | **New.** Path conversion, env overrides, file sync unit tests |
| Bridge origins | Clients without an `Origin` header are accepted; browser `null`/`file:` origins are rejected, while localhost HTTP and the native app schemes remain allowed |

#### Launch bugfix (2026-07-10)

`subprocess.Popen(..., cwd="C:\\Users\\...")` from WSL failed with `FileNotFoundError` (Linux has no `C:\` cwd). **Fix:** `cwd` stays a Linux path (`/mnt/c/...`); only `argv` script path and `PYTHONPATH` use Windows-style paths for the Windows Python process.

#### Hybrid TTS playback bugfix (2026-07-11)

The WSL backend generated and broadcast valid Kokoro WAV payloads, but the
Windows Qt proxy only applied bridged frontend events to the native HUD. It did
not invoke the embedded page's `onAudioReady` callback, so WebAudio never
received the voice data. `QtAvatarShell.dispatch_frontend_event()` now forwards
every structured event to the page as well as updating the HUD. The in-process
desktop path uses the same structured dispatch without issuing a duplicate
JavaScript callback.

#### Env vars (hybrid)

| Variable | Role |
|----------|------|
| `LOCAL_AI_WINDOWS_PYTHON` | Force path to Windows `python.exe` (must have PyQt6 + WebEngine) |
| `LOCAL_AI_WINDOWS_CHECKOUT` | Force Windows-drive clone used for script cwd / Live2D assets |
| `LOCAL_AI_WINDOWS_USER` | Hint for `C:\Users\<name>\Documents\Local-AI-Companion` discovery |
| `LOCAL_AI_FORCE_WSL_DESKTOP=1` | Disable hybrid; run Qt inside WSLg instead |

Default discovery also looks at
`/mnt/c/Users/<user>/Documents/Local-AI-Companion/venv/Scripts/python.exe`.

### Phase 3 — Latency / background resource notes (ops, not product code)

Observed on a slow hybrid turn (`logs/assistant.log` ~22:18):

| Stage | Slow run | Healthy earlier run (same day) |
|-------|----------|--------------------------------|
| Startup total | ~**87 s** (TTS preload ~73 s) | ~13–30 s |
| ASR (2 s speech) | ~**22 s** | ~1 s |
| First LLM token | ~**66 s** | ~0.5 s |
| Full turn | cancelled ~2 min | ~**3.5 s** |

Root causes (not “local LLM is always slow”):

1. **VRAM contention** on RTX 4070 12 GB — Ollama `qwen3.5:4b` kept **~6–8 GB** resident; Whisper + Kokoro on top thrash the GPU.
2. Under pressure, ASR language detect + decode and Ollama first content token balloon.
3. With free VRAM + `think: false`, the same Ollama model answers in **&lt;1 s** (curl check).

#### Validated VRAM budget fix (2026-07-11)

The Windows Ollama daemon inherited a `65536`-token context when the client did
not send `num_ctx`. A live cold/warm comparison from WSL showed:

| Ollama context | Model VRAM | Total GPU usage after load | Warm request |
|---|---:|---:|---:|
| inherited `65536` | 7.81 GB | 9.21 / 12 GB | 0.41 s |
| explicit `4096` | 5.89 GB | 7.39 / 12 GB | 0.32 s |

`llm.ollama.options.num_ctx: 4096` is now the tracked default. This recovers
about 1.9 GB for Whisper and Kokoro. Companion history is already bounded, so a
65k context provides no useful benefit on the stable realtime path.

The WSL-to-Windows Ollama TCP connection itself completed in under 1 ms during
the same check. The multi-second spikes were GPU contention, not bridge or WSL
network transport latency.

The Windows checkout also used a local OpenRouter override while WSL used the
tracked Ollama default. Performance comparisons must use the same commit, LLM
provider, ASR profile, and prompt before attributing a difference to WSL.

Background processes cleaned on contributor machine (2026-07-10):

| Process | Action |
|---------|--------|
| Leftover `python -m src.server` on `:8000` (~3 h old) | Killed (was not meant to stay up) |
| Hermes user systemd services (`hermes-*`, gateways, camofox) | `stop` + `disable` (user unused) |
| Ollama | Left running (needed for companion); model may still hold VRAM until unloaded |

Unload Ollama VRAM if needed:

```bash
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3.5:4b","keep_alive":0}'
```

---

## Architecture

```text
config.yaml (portable paths)
        │
        ▼
platform_compat.resolve_python_executable()
  - null → sys.executable
  - .../Scripts/python.exe → .../bin/python on WSL/Linux
  - .../bin/python → .../Scripts/python.exe on Windows
  - explicit missing path → explicit error (never another venv)
        │
        ▼
RVC / Qwen3-TTS / Qwen3-ASR worker subprocess
  - start_new_session=True on POSIX only
  - kill_process_tree() → taskkill (Windows) / killpg (WSL/Linux)
```

---

## One desktop command

| Environment | What you get | Command |
|-------------|--------------|---------|
| **Windows** | Local backend + native Live2D shell | `python run_assistant.py` |
| **WSL hybrid** | Backend in WSL + Windows-native Live2D shell | `python run_assistant.py` |

`scripts/run_wsl.sh` exposes optional browser, CLI, diagnostic, and bridge-only
tools. It is not a second desktop launcher.

### Hybrid option A: backend WSL + **Windows desktop pet** (recommended under WSL)

True “incrusted on the desktop” pets (Open-LLM-VTuber, nizima, Tauri pets) always
use a **native Windows window** (transparent + topmost + click-through), never a
normal browser tab. We do the same:

```text
WSL:     ASR / LLM / TTS / mic   →  bridge ws://127.0.0.1:8765
Windows: scripts/windows_pet_shell.py (PyQt6)
         same transparent Qt overlay as native Windows path
         talks to the bridge (src/desktop/bridge_proxy.py)
```

```bash
# Inside WSL (default)
source venv/bin/activate
python run_assistant.py
```

Requirements on **Windows**:

- Python 3.12 venv with `PyQt6`, `PyQt6-WebEngine`, `websockets`
  (your `C:\Users\…\Documents\Local-AI-Companion\venv` is used when present)
- Optional override: `LOCAL_AI_WINDOWS_PYTHON=C:\path\to\python.exe`

If the Qt pet cannot start, the launcher logs a local browser URL and keeps its
small fallback HTTP server under the assistant lifecycle. The server exposes
only the staged `.runtime/windows-pet-shell` files, not the repository root, and
does not spawn an unowned browser process.

Force Qt inside WSLg (not recommended):

```bash
LOCAL_AI_FORCE_WSL_DESKTOP=1 python run_assistant.py
```

Bridge only (no UI):

```bash
python run_assistant.py --bridge-server
```

Manual two-terminal setup (debug):

```bash
# WSL terminal 1
python run_assistant.py --bridge-server --bridge-port 8765

# Windows PowerShell
cd C:\Users\<WindowsUser>\Documents\Local-AI-Companion
venv\Scripts\python.exe scripts\windows_pet_shell.py --bridge-url ws://127.0.0.1:8765
```

### Why this split is intentional (authoritative sources)

- **Microsoft WSL networking:** a server listening in WSL2 is reachable from the
  Windows host browser via `localhost` (localhost forwarding is the default).
  See [Accessing network applications with WSL](https://learn.microsoft.com/en-us/windows/wsl/networking)
  and [`localhostForwarding` in `.wslconfig`](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).
- **NVIDIA CUDA on WSL:** install the **Windows** NVIDIA driver only; do **not**
  install a Linux GPU display driver inside WSL (it overwrites the stubbed
  `libcuda` mapping).
  See [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
  and [Microsoft: Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).
- **WSL environment setup:** official install/update flow is `wsl --install` /
  `wsl --update` — [Microsoft WSL docs](https://learn.microsoft.com/en-us/windows/wsl/).

That is why the **WSL path** is optimized for **web + CLI + CUDA backend**, while
the **Windows path** keeps the desktop avatar stack (Qt/WebView2/Live2D) that was
built for native Windows.

---

## How to run on WSL2 (Option B)

### 1. Prepare the Windows host

From an Administrator PowerShell:

```powershell
wsl --install
wsl --update
```

Install the normal NVIDIA Windows driver if CUDA is needed. NVIDIA explicitly
warns not to install a Linux display driver inside WSL; the Windows driver is
projected into WSL2.

If `nvidia-smi` works in WSL but the diagnostic reports `cuda_available:
false`, the project venv likely contains a CPU-only PyTorch build. Install the
appropriate CUDA wheel using the [official PyTorch selector](https://pytorch.org/get-started/locally/).

### 2. One-command project setup

```bash
cd ~/Local-AI-Companion   # or your clone path
bash scripts/setup_wsl.sh --with-rvc
source venv/bin/activate
python run_assistant.py
```

The launcher uses `./venv` directly, so activation is not required. The RVC
installer reuses the main environment's Torch/CUDA runtime instead of
downloading a duplicate GPU stack.

### 3. Optional browser run

```bash
bash scripts/run_wsl.sh web
```

This starts the FastAPI/WebSocket stack on **http://127.0.0.1:8000**.

Then on **Windows**, open Edge/Chrome:

```text
http://localhost:8000/web/
```

This optional path keeps the interface in a Windows browser:

- Live2D renders with **native Windows GPU WebGL** (not Qt/WSLg software GL)
- Microphone uses the **browser** (`getUserMedia`) on Windows
- ASR / LLM / TTS stay in the **WSL backend** (CUDA OK)

Put each character model under a project-relative directory such as
`assets/models/my-character/`, then set `live2d.model_path` and
`live2d.settings_file` in its character preset. The same canonical configuration
is resolved for the web and desktop shells.

Other modes:

```bash
python run_assistant.py             # desktop assistant (auto-hybrid under WSL)
bash scripts/run_wsl.sh check       # CUDA / audio / path diagnostic
bash scripts/run_wsl.sh cli         # text chat
bash scripts/run_wsl.sh cli-voice   # microphone CLI
bash scripts/run_wsl.sh bridge      # ws://127.0.0.1:8765 for external shells
bash scripts/run_wsl.sh desktop     # experimental Qt inside WSLg only (forces LOCAL_AI_FORCE_WSL_DESKTOP)
```

### 4. Platform diagnostics

```bash
bash scripts/run_wsl.sh check
# or
python scripts/check_platform.py
```

Expect:

- `platform: wsl`
- `cuda_available: true` when NVIDIA driver + WSL CUDA are set up
- Audio input may remain `0` until Windows mic privacy and WSLg are available

### 5. Desktop avatar under WSL (optional / experimental)

```bash
bash scripts/run_wsl.sh desktop
```

**Visual policy:** Live2D framing JS stays **identical to `main` / Windows**.
Host-only WSL fixes: software WebGL, `QT_SCALE_FACTOR=1`, zoom 1.0.
EGL/DRM console noise under WSLg is expected. For the polished mascot, use Windows.

Live2D character assets are local project data and may be excluded from git.

### Microphone under WSL

1. Windows Settings → Privacy → Microphone → allow desktop apps / your terminal.
2. Update WSL (`wsl --update`) and ensure WSLg is running (`echo $PULSE_SERVER`, `/mnt/wslg/PulseServer`).
3. WSLg normally configures `PULSE_SERVER`; the app only fills it when missing and the WSLg socket exists.
4. Re-run `python scripts/check_platform.py` until `input_devices > 0`.

If mic stays unavailable, run voice capture on **Windows host** or use the browser WebSocket frontend with host audio.

---

## How to run on Windows (unchanged intent)

```powershell
cd Local-AI-Companion
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python run_assistant.py
```

Portable relative config values like `venv/Scripts/python.exe` remain native on
Windows and map to `venv/bin/python` under WSL. Dedicated providers never fall
back silently to the main environment when their configured venv is missing.

---

## Traceability

| Artifact | Location |
|---|---|
| Linear ticket | LIL-49 |
| Standby related work | LIL-48 (Parakeet) + stash `WIP LIL-48 parakeet standby` |
| Code helper | `src/utils/platform_compat.py` |
| Hybrid launcher | `src/utils/wsl_hybrid_ui.py` |
| Windows pet shell | `scripts/windows_pet_shell.py` |
| Bridge client (Windows) | `src/desktop/bridge_proxy.py` |
| Workers | `src/tts/rvc_provider.py`, `src/tts/qwen3_tts_provider.py`, `src/asr/qwen3_asr_provider.py` |
| Config | `config/config.yaml` portable worker paths |
| Tests | `tests/test_platform_compat.py`, `tests/test_wsl_hybrid_ui.py` |
| Diagnostics/setup | `scripts/check_platform.py`, `scripts/setup_wsl.sh` |
| Desktop run | `python run_assistant.py` on Windows and WSL |
| WSL secondary tools | `scripts/run_wsl.sh` (`web`, `cli`, `cli-voice`, `bridge`, `desktop`, `check`) |
| WSL RVC installer | `scripts/install_rvc_wsl.sh` |
| Git branch | `feature/lil-49-wsl-run-path` |
| External refs | [MS WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking), [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) |

---

## Acceptance checklist

- [x] Ticket LIL-49 created and linked from this doc
- [x] LIL-48 marked standby with pointer to this work
- [x] Deterministic Windows↔WSL path and process-cleanup unit tests
- [x] `check_platform.py` text and JSON output execute on contributor WSL
- [x] WSL bootstrap and RVC scripts pass shell syntax validation
- [x] Targeted WSL, platform, desktop bridge, and frontend tests green on branch
- [x] FastAPI/WebSocket backend starts on WSL loopback
- [x] Hybrid option A: `windows_pet_shell` + `bridge_proxy` + auto-launch from WSL
- [x] Hybrid launch smoke: BridgeProxy connected + pet shell loaded (fake bridge + Windows Python)
- [x] Launch fix: Popen `cwd` uses Linux `/mnt/c/...` when spawning Windows Python from WSL
- [x] Changelog section in this doc (phases 1–3)
- [ ] CUDA-enabled PyTorch validated in the supported WSL venv
- [ ] Mic path validated when Pulse/WSLg available
- [ ] RVC overlay install + voice conversion smoke validated on WSL
- [ ] End-user smoke: full voice turn under hybrid with free VRAM (target ~few seconds like 21:39 log)
- [ ] Windows-native smoke still starts (`run_assistant.py` / workers)

---

## Decision rules

| Goal | Preferred environment |
|---|---|
| Ship / polish desktop avatar | Windows 11 native |
| Agent coding + CUDA ASR/TTS + hybrid pet | WSL2 backend + Windows Qt shell (option A) |
| Live mic bakeoff (LIL-35/48) | Prefer Windows until WSL `input_devices > 0` |
| Avoid multi-minute voice latency | Free GPU VRAM first (unload extra Ollama models; don’t leave stray `src.server` running) |

---

## Logs to inspect when something feels slow

| Log | Path |
|-----|------|
| Assistant / pipeline / startup profile | `logs/assistant.log` |
| Windows pet shell (hybrid) | `logs/windows_pet_shell.log` |
| Conversation turns | `logs/conversation.jsonl` |

Useful grep patterns in `assistant.log`:

- `startup profile status=`
- `First LLM chunk after`
- `First TTS audio latency`
- `Transcription:`
- `speech turn`

---

## References

- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Install WSL (Microsoft)](https://learn.microsoft.com/en-us/windows/wsl/install)
- [WSLg architecture and audio](https://github.com/microsoft/wslg)
- [WSL networking / localhost](https://learn.microsoft.com/en-us/windows/wsl/networking)
- [Run Windows tools from Linux (interop)](https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-windows-tools-from-linux)
