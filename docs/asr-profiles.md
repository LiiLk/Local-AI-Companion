# ASR Profiles (LIL-37)

The Whisper ASR provider exposes named **profiles** so you can trade off quality
vs hardware without hand-tuning model settings. Select one with `asr.profile`.

## Profiles

| Profile | Model | Beam | ~VRAM | When to use |
|---|---|---|---|---|
| `balanced` *(public default)* | `whisper small` | 3 | ~2 GB | Broad compatibility, older / smaller GPUs |
| `quality-local` | `whisper large-v3-turbo` | 5 | ~6 GB | Much better French/English accuracy; needs a capable GPU |

`balanced` stays the **public default** so the project runs on a wide range of
hardware. `quality-local` is a deliberate **opt-in**, not silently enabled.

## How to switch

In `config/config.local.yaml` (local, git-ignored):

```yaml
asr:
  profile: "quality-local"
```

### Tight VRAM

`large-v3-turbo` at `float16` needs ~6 GB. On a smaller GPU, roughly halve the
memory with int8 weights at minimal quality cost:

```yaml
asr:
  profile: "quality-local"
  compute_type: "int8_float16"
```

### Power-user override

An explicit `asr.model_size` / `asr.beam_size` overrides the selected profile.

## Why not Parakeet / Voxtral here?

They are strong but require a **separate runtime** (NVIDIA NeMo / vLLM), not the
`faster-whisper` path these profiles use. They are tracked in their own tickets
(cloud-STT provider and a non-Whisper local bakeoff), not in this profile system.

## Debugging mis-transcriptions

Set `ASR_DEBUG_DIR` to capture the exact audio + transcription on each turn
(no-op when the env var is unset), then replay the captured WAVs through several
profiles deterministically:

```powershell
$env:ASR_DEBUG_DIR = "logs/asr_debug"   # then run the avatar and speak
venv\Scripts\python.exe scripts\asr_replay_debug.py
```
