"""Replay captured ASR debug WAVs through different Whisper model/beam configs.

Deterministic A/B test for LIL-37: run the exact same captured audio through
`small` (reproduce the runtime bug) and `large-v3-turbo` (candidate fix),
isolating the beam-size effect too.

Usage (from repo root, runtime venv):
    venv\\Scripts\\python.exe scripts\\asr_replay_debug.py [--language fr] [wav ...]

Defaults to logs/asr_debug/*.wav. Language defaults to auto-detect; pass
--language to replay a forced-language capture under the same conditions.
Loads one model at a time to keep VRAM bounded.
"""
import argparse
import gc
import glob
import os
import sys
from pathlib import Path

# Don't recursively re-capture during replay.
os.environ.pop("ASR_DEBUG_DIR", None)

# Allow running as a plain script from anywhere (add repo root to path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.whisper_provider import WhisperProvider

# (model_size, [beam sizes to test])
PLAN = [
    ("small", [3, 5]),            # 3 = balanced profile (reproduce); 5 = isolate beam
    ("large-v3-turbo", [5]),      # quality-local profile
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay captured ASR WAVs through Whisper profiles."
    )
    parser.add_argument("wavs", nargs="*", help="WAV files (default: logs/asr_debug/*.wav)")
    parser.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g. fr) to match a forced-language capture. "
        "Default: auto-detect (matches asr.language: auto).",
    )
    args = parser.parse_args()

    wavs = [str(w) for w in (args.wavs or sorted(glob.glob("logs/asr_debug/*.wav")))]
    if not wavs:
        print("No WAVs found in logs/asr_debug/.")
        return

    captured = {}
    for wav in wavs:
        txt = Path(wav).with_suffix(".txt")
        captured[wav] = txt.read_text(encoding="utf-8").strip() if txt.exists() else "(none)"

    results = {}  # (wav, model, beam) -> text
    for model, beams in PLAN:
        print(f"\n>>> Loading {model} ...", flush=True)
        provider = WhisperProvider(model_size=model, device="auto", beam_size=beams[0])
        for wav in wavs:
            for beam in beams:
                provider.beam_size = beam
                try:
                    res = provider.transcribe(wav, language=args.language)
                    results[(wav, model, beam)] = res.text
                except Exception as exc:  # noqa: BLE001
                    results[(wav, model, beam)] = f"ERROR: {exc}"
        del provider
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    lang_note = args.language or "auto"
    for wav in wavs:
        print("\n" + "=" * 84)
        print(f"WAV: {Path(wav).name}  (replay language={lang_note})")
        print(f"  runtime capture (live config): {captured[wav]!r}")
        print("  --- replay ---")
        for model, beams in PLAN:
            for beam in beams:
                text = results.get((wav, model, beam), "(missing)")
                print(f"  [{model:>15} | beam={beam}] -> {text!r}")


if __name__ == "__main__":
    main()
