# ASR Providers & Profiles

## Public default: `whisper` (LIL-37)

Stable path ASR remains **faster-whisper** with named profiles:

| Profile | Model | Beam | ~VRAM | When to use |
|---|---|---|---|---|
| `balanced` *(public default)* | `whisper small` | 3 | ~2 GB | Broad compatibility, older / smaller GPUs |
| `quality-local` | `whisper large-v3-turbo` | 5 | ~6 GB | Much better French/English accuracy; needs a capable GPU |

```yaml
asr:
  provider: "whisper"
  profile: "balanced"   # or "quality-local"
```

### Tight VRAM on quality-local

```yaml
asr:
  provider: "whisper"
  profile: "quality-local"
  compute_type: "int8_float16"
```

An explicit `asr.model_size` / `asr.beam_size` overrides the selected profile.

---

## Opt-in: `parakeet` (LIL-48)

**NVIDIA Parakeet-TDT-0.6b-v3** via [`onnx-asr`](https://pypi.org/project/onnx-asr/)
(ONNX Runtime — **no NeMo install**). This is the integration path chosen to stay
Windows-friendly: full NeMo is the risk called out in LIL-45/LIL-48; ONNX avoids it.

### Status

| Item | Value |
|---|---|
| Public default? | **No** — opt-in only until Go decision is documented |
| Install impact if unused | None (`requirements-optional-parakeet.txt`) |
| Languages | 25 European (fr, en, es, de, …) — **not** zh/ja/ar/ko/hi |
| Runtime | CPU by default (`CPUExecutionProvider`) |

### Enable

```bash
python -m pip install -r requirements-optional-parakeet.txt
```

`config/config.local.yaml`:

```yaml
asr:
  provider: "parakeet"
  parakeet:
    model_name: "nemo-parakeet-tdt-0.6b-v3"
    quantization: "int8"          # CPU path (recommended)
    # providers: ["CUDAExecutionProvider"]   # optional GPU
```

First load downloads the ONNX weights from Hugging Face.

### When to prefer Parakeet vs Whisper

| Need | Choice |
|---|---|
| FR/EN conversation, want CPU-friendly speed | Try **parakeet** |
| Non-European language | **whisper** |
| Zero optional deps / stock install | **whisper** |
| GPU quality-local Whisper already good enough | Keep **whisper** `quality-local` |

Provisional agent notes (not a formal bakeoff): Parakeet may approach
`large-v3-turbo` quality at high realtime factor on CPU. Re-measure on *your*
machine with captured WAVs before promoting to default. See
`docs/lil-48-parakeet-asr.md`.

---

## Why not NeMo / Voxtral here?

- **NeMo**: heavy, Windows-painful — exactly the risk LIL-48 flagged for a public default.
- **Voxtral / vLLM**: separate runtime; tracked elsewhere if needed.

---

## Debugging mis-transcriptions

Capture live turns:

```powershell
$env:ASR_DEBUG_DIR = "logs/asr_debug"   # then run the avatar and speak
```

Replay through Whisper profiles **and** Parakeet (if installed):

```powershell
venv\Scripts\python.exe scripts\asr_replay_debug.py
venv\Scripts\python.exe scripts\asr_replay_debug.py --include-parakeet
```
