# Local Security Threat Model

This document is the baseline repository threat model for Local-AI-Companion.
It is not a vulnerability report for one pull request. Its goal is to make the
project's trust boundaries explicit so future security scans and code reviews
can calibrate severity consistently.

## Overview

Local-AI-Companion is a Windows-first, offline-first desktop voice companion.
The stable product path is:

```text
Microphone -> Silero VAD -> Faster-Whisper -> LLM text -> Kokoro TTS -> RVC -> Audio playback
```

The primary runtime surfaces are:

- the FastAPI backend in `src/server/app.py`
- the browser/WebSocket orchestration in `src/server/websocket.py`
- the desktop bridge and Live2D desktop app in `src/assistant/app.py`
- the Tauri desktop shell scaffold in `desktop/tauri/src-tauri/`
- provider construction and lifecycle management in `src/assistant/pipeline_runtime.py`
- local configuration loading in `src/utils/config_loader.py`
- local logs and runtime memory in `src/utils/logging_setup.py` and
  `src/assistant/conversation_memory.py`
- local worker subprocesses for optional or isolated providers such as RVC,
  Qwen3-TTS, and Qwen3-ASR

The project is local-first, not cloud-forbidden. Local ASR and TTS are the
baseline. Ollama is the default LLM path. OpenRouter is optional and must be
enabled deliberately with an API key from the environment or local config.

The main assets to protect are:

- microphone audio, transcripts, assistant responses, and conversation memory
- local logs under `logs/`
- runtime memory under `data/memory/`
- API keys and tokens in environment variables or `config/config.local.yaml`
- local model files, voice models, RVC indexes, and downloaded provider assets
- the user's local machine, GPU resources, and desktop session
- optional cloud-bound prompts and responses when OpenRouter or other remote
  providers are enabled

## Threat Model, Trust Boundaries, and Assumptions

### Actors

- **Local operator**: the person running the companion on their own machine.
  They control local configuration, model choices, and whether optional cloud
  providers are enabled.
- **Local frontend**: the bundled Live2D frontend, Tauri WebView, or local
  browser UI talking to the backend.
- **Local backend**: the FastAPI server, desktop bridge, provider runtime, and
  local workers.
- **Optional cloud provider**: OpenRouter or another explicitly configured
  remote service.
- **Untrusted local webpage or process**: a website or local process running on
  the same machine that may try to connect to localhost services.
- **LAN attacker**: another device on the local network. This actor is out of
  reach by default, but becomes relevant if the user binds the backend to
  `0.0.0.0` or another non-loopback interface.
- **Repository contributor**: a person changing source code, dependencies,
  scripts, or docs before publication.

### Trust boundaries

1. **Frontend to backend WebSocket boundary**
   - `src/server/websocket.py` accepts real-time text and audio messages.
   - The endpoint is unauthenticated and relies on loopback deployment as the
     default security boundary.
   - If the backend is exposed beyond loopback, this boundary must be treated as
     remote-attacker reachable.

2. **Desktop bridge boundary**
   - `src/assistant/app.py` exposes a small local WebSocket bridge on
     `127.0.0.1:8765` by default.
   - Its command set is intentionally narrow: send text, interrupt, toggle mute,
     get runtime state, and toggle debug.
   - It must not grow into a general filesystem, shell, or arbitrary Python
     execution bridge without a new security design.

3. **Tauri/WebView boundary**
   - `desktop/tauri/src-tauri/tauri.conf.json` enables a local WebView shell and
     only connects to the local bridge by default.
   - Tauri capabilities are minimal in `desktop/tauri/src-tauri/capabilities/`.
   - The shell should remain a thin local UI layer unless the project explicitly
     introduces privileged desktop APIs.

4. **Runtime to subprocess worker boundary**
   - RVC, Qwen3-TTS, and Qwen3-ASR can run in local worker subprocesses.
   - Worker scripts and Python paths are operator-controlled through config.
   - Current subprocess calls use argument lists rather than shell strings in
     the reviewed paths, which reduces command-injection risk.
   - Compromise of local config, worker scripts, or worker virtual environments
     is equivalent to local code execution by the user account.

5. **Runtime to local filesystem boundary**
   - The app writes logs, conversation JSONL, memory summaries, temporary audio
     files, and model artifacts.
   - `logs/`, `data/memory/`, `reports/`, virtual environments, model files,
     and local config files are ignored by git.
   - These files can still contain private local data and should be treated as
     sensitive on the user's machine.

6. **Runtime to optional cloud provider boundary**
   - OpenRouter sends prompt content and receives model responses over HTTPS
     when explicitly enabled.
   - The API key is read from config or `OPENROUTER_API_KEY`.
   - Enabling a cloud LLM changes the privacy model: spoken content, memory
     context, and prompts may leave the machine.

7. **Model and dependency supply-chain boundary**
   - Downloaded models, model revisions, Python dependencies, npm packages, and
     Rust crates are external inputs.
   - Optional model paths are secondary, but they can still affect publication
     risk if documented as supported.

### Assumptions

- Default runtime deployment is single-user and local to the operator's Windows
  desktop.
- `config/config.yaml` defaults the backend to `127.0.0.1`, with LAN exposure as
  explicit opt-in only.
- `config/config.local.yaml`, environment variables, and local virtual
  environments are trusted operator-controlled inputs, not remote user input.
- A malicious repository contributor, compromised dependency, or malicious local
  model file is in scope for supply-chain review.
- A local OS user with filesystem access to the project directory can read local
  logs and memory. Encrypting local memory is not currently a project guarantee.
- The current threat model does not assume multi-user hosting, public internet
  deployment, or tenant isolation.

## Attack Surface, Mitigations, and Attacker Stories

### FastAPI and browser WebSocket

The browser backend exposes REST endpoints for health/config/voices and a
WebSocket endpoint at `/ws/{client_id}`. The WebSocket accepts text, JSON audio
payloads, raw audio bytes, interrupt/clear commands, and preload requests.

Relevant mitigations already present:

- `config/config.yaml` defaults `server.host` to `127.0.0.1`.
- `src/server/settings.py` derives local CORS origins from the configured port.
- `allow_credentials` defaults to `false` and is forced off when wildcard CORS
  is configured.
- Static mounts serve known project frontend/assets directories, not arbitrary
  filesystem paths.
- Temporary audio files use `tempfile.NamedTemporaryFile(...)`.
- WebM conversion uses `subprocess.run([...])` rather than `shell=True`.

Attacker stories:

- A LAN attacker should not reach the backend with defaults. If the user binds
  to `0.0.0.0`, unauthenticated WebSocket control becomes a high-risk surface.
- A malicious local webpage may attempt to connect to `ws://127.0.0.1:<port>`.
  CORS does not protect WebSocket upgrades by itself in the same way it protects
  regular fetches, so explicit origin validation would be a useful hardening
  layer.
- A local attacker could send large or frequent audio messages to force model
  preload, GPU work, disk writes, or CPU-heavy transcription. This is primarily
  denial-of-service on the local machine unless the backend is exposed remotely.

Recommended follow-ups:

- Add explicit WebSocket `Origin` allowlist checks matching the local frontend
  origins.
- Keep message size, audio duration, and preload throttling bounded.
- Treat non-loopback backend binding as a documented insecure/advanced mode
  unless authentication or a session token is added.

### Desktop bridge and Tauri shell

The desktop bridge is local-only by default and deliberately exposes a narrow
command vocabulary. The Tauri scaffold currently returns local state and does
not expose shell, dialog, filesystem, or arbitrary command capabilities.

Relevant mitigations already present:

- `DesktopBridgeServer` binds to `127.0.0.1` by default.
- The bridge uses `max_size=8 * 1024 * 1024`.
- Unknown or malformed bridge messages are rejected.
- Tauri capabilities are limited to `core:default` and window dragging.
- The Tauri CSP restricts connections to the local bridge and Tauri IPC.

Attacker stories:

- A malicious local webpage or process could attempt to connect to the bridge
  port while it is running. Current commands are limited, but they can still
  trigger assistant text input, mute toggling, interrupts, and debug UI changes.
- If future bridge commands add filesystem, shell, screen capture, or settings
  mutation, the bridge must be reclassified as a privileged local API and should
  require a nonce, token, or stronger origin/session control.

Recommended follow-ups:

- Keep the desktop bridge command set narrow.
- Add a local per-run bridge token before exposing any privileged operation.
- Do not enable broad Tauri plugins without updating this threat model.

### Configuration and secrets

Tracked defaults live in `config/config.yaml`; local overrides belong in
`config/config.local.yaml`. OpenRouter keys are intentionally null in tracked
config and can be supplied through `OPENROUTER_API_KEY`.

Relevant mitigations already present:

- `config/config.local.yaml`, `.env`, reports, logs, runtime memory, models, and
  virtual environments are git-ignored.
- `OpenRouterLLM` reads the key from config or environment and sends it in an
  authorization header; the reviewed code does not intentionally log the key.
- Runtime memory curation skips content that looks like API keys, tokens,
  passwords, private keys, or common secret patterns.

Attacker stories:

- Accidentally committing `config/config.local.yaml`, `.env`, reports, or copied
  logs could leak API keys or private transcripts.
- Prompt injection can still cause the assistant to include private local memory
  context in a cloud LLM prompt if OpenRouter is enabled.
- Local config can point workers to arbitrary local scripts. This is acceptable
  as operator-controlled power, but it must never become remotely configurable
  through a WebSocket or REST endpoint.

Recommended follow-ups:

- Keep local config and secret scanning in the release checklist.
- Never add an endpoint that returns raw config.
- If settings mutation is added later, keep secrets write-only/redacted and
  require local UI trust controls.

### Logs, transcripts, and runtime memory

The project stores operational logs and conversation logs under `logs/`. Runtime
conversation memory is stored under `data/memory/`.

Relevant mitigations already present:

- Runtime logs use rotating files.
- `logs/` and `data/memory/` are git-ignored.
- Memory curation has explicit secret-looking content filters.
- Prompt context is bounded by configuration.

Attacker stories:

- A local user or malware with access to the project directory can read private
  transcripts, memory summaries, and runtime logs.
- Conversation logs may contain raw text that the curation layer intentionally
  avoided storing in summary memory.
- If logs or reports are pasted into issues, PRs, or public chat, they can leak
  private data even when git ignore rules are correct.

Recommended follow-ups:

- Document that logs and memory are private local data.
- Keep logs and memory out of support bundles by default unless the user
  explicitly reviews them.
- Consider a local "clear private data" command before wider publication.

### Local workers, models, and supply chain

RVC, Qwen3-TTS, and Qwen3-ASR can use worker subprocesses to isolate dependency
stacks. Model downloads and optional provider paths introduce supply-chain risk.

Relevant mitigations already present:

- Worker subprocesses are launched with explicit argument lists in reviewed
  paths.
- Qwen3 worker inputs validate local Python executable and `.py` worker script
  paths before startup.
- Static-analysis follow-up work already documented current subprocess handling
  in `docs/security-static-findings.md`.
- Some Hugging Face download paths now accept optional revisions and warn when
  no revision is configured.

Attacker stories:

- A malicious dependency, unpinned model revision, or compromised worker
  environment can execute code locally during import or model load.
- A malicious local RVC model or model-adjacent file should be treated as
  untrusted input until the relevant loader behavior is understood.
- Worker stderr may contain local paths or provider errors; avoid copying it
  verbatim into public issues without review.

Recommended follow-ups:

- Continue pinning model revisions for supported public paths.
- Keep optional experimental modes clearly secondary until their supply-chain
  and runtime story is documented.
- Do not accept worker script paths, Python paths, or model paths from remote
  WebSocket clients.

### Optional cloud, vision, and screen-sensitive features

The stable product is local voice-first. Optional cloud LLM, multimodal, vision,
and screen features change the privacy boundary.

Attacker stories:

- Cloud LLM prompts may include user speech, memory summary, or tool results.
- Screen capture or vision features can expose content from other applications.
- Prompt injection can attempt to move private data from local context into a
  remote provider response or request.

Recommended follow-ups:

- Treat cloud LLM and screen/vision as explicit opt-in privacy modes.
- Make privacy implications visible in docs and local config comments.
- Before adding computer-control features, define a separate capability and
  authorization model.

## Severity Calibration (Critical, High, Medium, Low)

### Critical

Use Critical only when an issue can plausibly cause remote code execution,
secret exfiltration, or broad local machine compromise under realistic product
usage.

Examples:

- A remotely reachable backend path allows arbitrary command execution or
  attacker-controlled worker script execution.
- A public PR commits a real API key, token, private key, or credential file.
- A default configuration exposes unauthenticated desktop control beyond
  loopback and allows privileged filesystem or shell access.
- A dependency or model loading path executes untrusted remote code by default
  in the stable pipeline without user opt-in.

### High

Use High when an issue exposes sensitive local data, enables meaningful remote
or LAN control when a plausible deployment footgun is used, or creates a strong
path to local compromise.

Examples:

- Binding the backend to `0.0.0.0` exposes unauthenticated WebSocket control to
  the LAN.
- A malicious local webpage can reliably control the assistant through localhost
  WebSocket commands in a way that captures private audio, leaks memory, or
  causes cloud requests.
- Logs, memory, or reports containing private transcripts are accidentally made
  public.
- A settings endpoint lets remote clients modify worker paths, model paths, or
  API keys.

### Medium

Use Medium when an issue affects privacy, availability, or integrity but depends
on local access, explicit optional modes, or limited attacker control.

Examples:

- Missing WebSocket origin checks allow local webpage nuisance control of
  non-privileged commands.
- Large audio messages or repeated preload requests can cause local CPU/GPU
  denial-of-service.
- Optional cloud mode can leak prompt context because privacy boundaries are not
  clear enough to the user.
- Unpinned optional model revisions reduce reproducibility and increase supply
  chain uncertainty for secondary modes.

### Low

Use Low for local metadata leaks, minor hardening gaps, or issues that require
trusted local operator control without crossing a meaningful trust boundary.

Examples:

- `/api/config` exposes provider names, model IDs, or voice names but no
  secrets.
- A local-only bridge command toggles debug UI or mute state without exposing
  private data.
- Temporary audio files may briefly exist under the OS temp directory but are
  not predictable public paths.
- Worker stderr includes local paths in developer logs.
