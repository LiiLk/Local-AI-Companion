# LIL-19 Security Scan Findings

This report records the scoped LIL-19 review against
`docs/security-threat-model.md`.

## Scope

Reviewed surfaces:

- FastAPI REST and WebSocket runtime
- Desktop WebSocket bridge
- local config, secrets, logs, and runtime memory
- RVC/Qwen worker subprocess boundaries
- optional cloud and model supply-chain boundaries
- prior local static reports under ignored `reports/`

Two parallel read-only security explorers reviewed the WebSocket/frontend and
desktop/config/worker slices. Their candidate findings were validated against
the current code before fixes were applied.

## Fixed Findings

### 1. Browser WebSocket Origin not checked

Severity: Medium by default, High if the backend is intentionally exposed or if
the attacker can trigger memory/cloud-bearing turns.

Evidence:

- `src/server/websocket.py` accepted `/ws/{client_id}` before checking
  `Origin`.
- Accepted commands included text, audio, interrupt, clear, and preload.
- Responses and generated audio are sent back to the same WebSocket client.

Fix:

- Added `resolve_websocket_allowed_origins(...)` and
  `is_websocket_origin_allowed(...)` in `src/server/settings.py`.
- The WebSocket endpoint now rejects disallowed browser origins before
  `accept()`.
- The default WebSocket allowlist follows the same local-first origins as CORS.
- Non-browser local clients without an `Origin` header remain compatible.

### 2. Desktop bridge accepted arbitrary browser origins

Severity: High for the desktop bridge because accepted clients can send text and
receive frontend events containing transcriptions/responses.

Evidence:

- `src/assistant/app.py` accepted bridge clients without checking origin.
- The bridge command set includes `send_text`, `interrupt`, `toggle_mute`,
  `get_runtime_state`, and `toggle_debug`.
- Connected bridge clients receive frontend events.

Fix:

- Added a desktop bridge origin gate before registering clients.
- Allowed origins are restricted to local desktop/webview origins such as
  `127.0.0.1`, `localhost`, `tauri.localhost`, and `ipc.localhost`.
- This blocks normal malicious remote webpages such as `https://evil.example`
  while preserving local desktop clients.

Remaining hardening:

- A per-run bridge token would be stronger before adding any privileged future
  bridge command. It was not added in this PR to avoid breaking current Tauri
  launch mechanics.

### 3. WebSocket payloads had no app-level size ceilings

Severity: Medium.

Evidence:

- The WebSocket endpoint decoded JSON text, base64 audio, PCM frames, and audio
  sample arrays without explicit local policy limits.
- Large payloads could force memory, CPU, ffmpeg, ASR, VAD, or GPU work.

Fix:

- Added conservative app-level ceilings:
  - text frame: 1,000,000 characters
  - user text: 8,000 characters
  - audio payload: 8 MiB
  - JSON audio samples: 30 seconds at 48 kHz
- Oversized payloads now return an error and are not processed.

Remaining hardening:

- A global connection/preload throttle would further reduce local DoS risk, but
  that is broader lifecycle work and should be split if needed.

### 4. `/api/config` exposed local reference-audio paths

Severity: Low.

Evidence:

- For Chatterbox/Qwen3 TTS, `tts_voice` could return a configured local
  reference-audio path.
- This did not expose API keys, but it could leak local username/path metadata.

Fix:

- Added `public_voice_label(...)`.
- `/api/config` now returns plain voice IDs as-is and strips path values to a
  basename only.

### 5. RVC `.pth` voice model trust boundary lacked pinning

Severity: High when users download untrusted `.pth` voice models.

Evidence:

- RVC `.pth` files are PyTorch checkpoints and can be pickle-compatible.
- The current RVC stack needs legacy checkpoint loading behavior for InferRVC.

Fix:

- Added optional `model_sha256` and `index_sha256` config fields.
- `RVCConverter` verifies configured SHA-256 hashes before loading or spawning
  the worker.
- Mismatches fail closed before the model file is loaded.
- `config/config.yaml` now documents the `.pth` trust risk and hash pin fields.

Remaining hardening:

- Publicly distributed voice packs should publish hashes.
- If the RVC ecosystem supports a safer non-pickle model format later, prefer
  that path.

## Deferred Findings

### Qwen3 optional models are not revision-pinned

Severity: Medium.

Evidence:

- Qwen3-TTS and Qwen3-ASR optional providers call `from_pretrained(...)`
  without a revision.

Disposition:

- Deferred. These modes are secondary/advanced, not the stable default.
- This should become a separate supply-chain follow-up if Qwen3 modes are
  promoted or documented as public supported paths.

## Suppressed Findings

- Static mounts: suppressed because mounted paths are fixed project directories,
  not user-selected filesystem paths.
- WebM `ffmpeg` conversion command injection: suppressed because subprocess
  calls use argument lists and server-created temp files.
- Raw config secret exposure: suppressed because `/api/config` uses a constrained
  response model and does not return API keys.
- Subprocess command injection in worker providers: suppressed for reviewed
  provider launches because commands use argument lists with `shell=False`.
- OpenRouter key logging: suppressed; the key is used in headers and is not
  intentionally logged in reviewed code.

## Validation

Focused tests:

```powershell
.venv-prcheck\Scripts\python.exe -m pytest tests/test_server_settings.py tests/test_server_routes.py tests/test_websocket_security_limits.py tests/test_rvc_provider.py tests/test_pipeline_runtime.py tests/test_websocket_turn_scheduling.py tests/test_live2d_assistant.py -q --basetemp .codex_test_artifacts\pytest-lil19d -o cache_dir=.codex_test_artifacts\pytest-cache-lil19d
```

Result:

```text
57 passed in 4.37s
```
