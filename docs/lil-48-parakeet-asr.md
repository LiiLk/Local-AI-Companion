# LIL-48 — Parakeet TDT 0.6B v3 as public-default candidate

## Goal

Evaluate whether **Parakeet-TDT-0.6b-v3** can become a better **public ASR default**
than Whisper `small` / `quality-local`, without forcing a heavy NeMo stack on every user.

Related: [LIL-45](https://linear.app/lilkorp/issue/LIL-45), [LIL-37](https://linear.app/lilkorp/issue/LIL-37), [LIL-35](https://linear.app/lilkorp/issue/LIL-35).

## Decision (current)

| Question | Answer |
|---|---|
| Provider implemented? | **Yes** — `asr.provider: parakeet` → `ParakeetASRProvider` |
| Runtime | **onnx-asr** (ONNX Runtime), **not** full NVIDIA NeMo |
| Public default? | **No** — remains **opt-in** |
| Heavy dep if unused? | **No** — optional file only |
| Go / No-Go for default | **Pending** your Windows live bakeoff |

Why not flip default yet (ticket AC):

1. No checked-in FR/EN WER + latency + VRAM table on *this* machine.
2. Language coverage is EU-only (breaking change for non-EU users if defaulted).
3. First-run model download + optional install must stay non-blocking for stock users.

## What shipped in code

| Piece | Path |
|---|---|
| Provider | `src/asr/parakeet_provider.py` |
| Factory | `create_pipeline_asr` in `src/assistant/pipeline_runtime.py` |
| Config | `config/config.yaml` → `asr.parakeet.*` (provider stays `whisper`) |
| Optional deps | `requirements-optional-parakeet.txt` |
| Unit tests | `tests/test_parakeet_provider.py` |
| Replay tool | `scripts/asr_replay_debug.py --include-parakeet` |

## Install + enable

```bash
python -m pip install -r requirements-optional-parakeet.txt
```

`config/config.local.yaml`:

```yaml
asr:
  provider: "parakeet"
```

Cold start will download ONNX weights once.

## Bakeoff you should run (owner tests)

Automated unit tests **do not** load the ONNX model (no download in CI).

### 1. Install smoke

```powershell
python -c "from src.asr.parakeet_provider import ParakeetASRProvider; p=ParakeetASRProvider(); p.preload(); print(p.get_model_info())"
```

### 2. Replay captured WAVs

Capture live turns with Whisper first:

```powershell
$env:ASR_DEBUG_DIR = "logs\asr_debug"
python run_assistant.py
# speak FR + EN short phrases, mixed, fast speech
```

Then:

```powershell
python scripts\asr_replay_debug.py --include-parakeet
python scripts\asr_replay_debug.py --include-parakeet --language fr
```

Compare columns:

- runtime capture text
- whisper `small` beam 3
- whisper `large-v3-turbo` beam 5
- parakeet int8

### 3. Live desktop path (required for Go)

Same mic path as LIL-35:

```text
Mic → AudioService → Silero VAD → delayed commit → Parakeet
```

Record:

| Metric | How |
|---|---|
| Quality FR/EN | subjective + hard errors on short phrases |
| Latency | VAD end → transcript ready |
| Warmup | first load only |
| RAM / VRAM | Task Manager / `nvidia-smi` during short clips |
| Windows install friction | time + failures from clean venv |

Optional: include `qwen3` ASR if that worker is already set up.

## Go criteria (suggested)

Promote to public default only if **all** hold:

1. FR conversational quality ≥ `quality-local` on your real clips (or clearly better than `balanced` with acceptable quality gap vs turbo).
2. Steady-state latency competitive with `small` on your machine (CPU path is expected to be strong).
3. Install from `requirements-optional-parakeet.txt` is reliable on Windows 11 + your GPU driver stack.
4. Docs/README language limitation is acceptable as product default (or Whisper remains auto-fallback — not implemented yet).

Otherwise: **keep opt-in**, document No-Go reasons here, close ticket as research complete.

## Provisional notes from earlier agent work

A previous branch commit claimed live FR/EN parity with `large-v3-turbo` at ~17× RTFx on CPU via onnx-asr. Treat as **hypothesis**, not acceptance evidence, until steps above are filled.

## Non-goals

- Do not force NeMo into the main venv.
- Do not change Whisper profiles.
- Do not add cloud STT here (LIL-46).
