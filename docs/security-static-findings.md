# Static Security Findings

This note records the pragmatic handling of `LIL-42` static-analysis findings.

## Fixed in code

- `src/tts/edge_tts_provider.py`
  - Replaced the hardcoded `/tmp/tts_output_<hash>.mp3` path with
    `tempfile.NamedTemporaryFile(delete=False)`.
  - This removes a Windows-incompatible path and avoids predictable file names.
- `src/asr/qwen3_asr_provider.py`
  - Validates worker Python executable and worker script paths before import
    checks and worker startup.
  - Keeps subprocess execution as argument lists with `shell=False`.
  - Validates the Windows `taskkill` PID before process-tree cleanup.
- `src/tts/qwen3_tts_provider.py`
  - Applies the same worker path validation and explicit `shell=False`.
  - Validates the Windows `taskkill` PID before process-tree cleanup.
- `src/desktop/__main__.py`
  - Uses `127.0.0.1` as the default backend host.
  - Normalizes `localhost` to loopback.
  - Keeps `0.0.0.0` available only as an explicit opt-in and logs a warning.

## Hugging Face model revisions

Static tools flag `snapshot_download`, `hf_hub_download`, and `from_pretrained`
when model revisions are not pinned. That is a real reproducibility concern, but
pinning every optional model to a commit SHA in one sweep would be a product
decision, not a safe mechanical fix.

Current handling:

- `src/asr/whisper_provider.py`
  - French Whisper downloads now accept an optional `revision` in the model
    config and pass it to `snapshot_download`.
  - If no revision is configured, the provider logs a warning before download.
- `src/tts/chatterbox_provider.py`
  - Chatterbox accepts optional `model_revision` and passes it to
    `hf_hub_download`.
  - If no revision is configured, the provider logs a warning before download.

Remaining optional providers such as Gemma/MiniCPM still need a dedicated model
pinning policy before publication if those modes become supported public paths.
For now they remain secondary/experimental and should not block the stable
pipeline hardening.

## False-positive notes

- `src/omni/gemma_provider.py` credential logging warning appears to be a false
  positive in current code inspection: the provider logs model load/generation
  metadata, not API keys or bearer tokens.
- `src/assistant/app.py` subprocess findings should be validated against the
  exact static-analysis output before changing runtime launch behavior. Current
  LIL-42 fixes avoid broad desktop lifecycle changes.
