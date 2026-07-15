"""Replay captured ASR debug WAVs through Whisper profiles and optional Parakeet.

Deterministic A/B for LIL-37 / LIL-48: run the exact same captured audio through
`small` (balanced), `large-v3-turbo` (quality-local), and optionally Parakeet.

Usage (from repo root, runtime venv):
    venv\\Scripts\\python.exe scripts\\asr_replay_debug.py [--language fr] [wav ...]
    venv\\Scripts\\python.exe scripts\\asr_replay_debug.py --include-parakeet

Defaults to logs/asr_debug/*.wav. Language defaults to auto-detect; pass
--language to replay a forced-language capture under the same conditions.
Loads one model at a time to keep VRAM / RAM bounded.
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

# Don't recursively re-capture during replay.
os.environ.pop("ASR_DEBUG_DIR", None)

# Allow running as a plain script from anywhere (add repo root to path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.whisper_provider import WhisperProvider

# (model_size, [beam sizes to test])
WHISPER_PLAN = [
    ("small", [3, 5]),            # 3 = balanced profile (reproduce); 5 = isolate beam
    ("large-v3-turbo", [5]),      # quality-local profile
]


def _cleanup() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay captured ASR WAVs through Whisper profiles / Parakeet."
    )
    parser.add_argument("wavs", nargs="*", help="WAV files (default: logs/asr_debug/*.wav)")
    parser.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g. fr) to match a forced-language capture. "
        "Default: auto-detect (matches asr.language: auto).",
    )
    parser.add_argument(
        "--include-parakeet",
        action="store_true",
        help="Also run Parakeet (requires: pip install -r requirements-optional-parakeet.txt).",
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

    results: dict[tuple, str] = {}
    latencies: dict[tuple, float] = {}

    for model, beams in WHISPER_PLAN:
        print(f"\n>>> Loading whisper {model} ...", flush=True)
        provider = WhisperProvider(model_size=model, device="auto", beam_size=beams[0])
        for wav in wavs:
            for beam in beams:
                provider.beam_size = beam
                key = (wav, f"whisper:{model}", beam)
                try:
                    t0 = time.perf_counter()
                    res = provider.transcribe(wav, language=args.language)
                    latencies[key] = time.perf_counter() - t0
                    results[key] = res.text
                except Exception as exc:  # noqa: BLE001
                    results[key] = f"ERROR: {exc}"
                    latencies[key] = float("nan")
        del provider
        _cleanup()

    if args.include_parakeet:
        print("\n>>> Loading parakeet ...", flush=True)
        try:
            from src.asr.parakeet_provider import ParakeetASRProvider

            if not ParakeetASRProvider.is_available():
                raise ImportError(
                    "onnx-asr missing — pip install -r requirements-optional-parakeet.txt"
                )
            parakeet = ParakeetASRProvider()
            parakeet.preload()
            for wav in wavs:
                key = (wav, "parakeet", "int8")
                try:
                    t0 = time.perf_counter()
                    res = parakeet.transcribe(wav, language=args.language)
                    latencies[key] = time.perf_counter() - t0
                    results[key] = res.text
                except Exception as exc:  # noqa: BLE001
                    results[key] = f"ERROR: {exc}"
                    latencies[key] = float("nan")
            del parakeet
            _cleanup()
        except Exception as exc:  # noqa: BLE001
            print(f"Parakeet unavailable: {exc}")

    lang_note = args.language or "auto"
    for wav in wavs:
        print("\n" + "=" * 84)
        print(f"WAV: {Path(wav).name}  (replay language={lang_note})")
        print(f"  runtime capture (live config): {captured[wav]!r}")
        print("  --- replay ---")
        for model, beams in WHISPER_PLAN:
            for beam in beams:
                key = (wav, f"whisper:{model}", beam)
                text = results.get(key, "(missing)")
                ms = latencies.get(key)
                timing = f"  [{ms*1000:.0f} ms]" if isinstance(ms, float) and ms == ms else ""
                print(f"  [whisper:{model:>15} | beam={beam}]{timing} -> {text!r}")
        if args.include_parakeet:
            key = (wav, "parakeet", "int8")
            text = results.get(key, "(missing)")
            ms = latencies.get(key)
            timing = f"  [{ms*1000:.0f} ms]" if isinstance(ms, float) and ms == ms else ""
            print(f"  [parakeet int8              ]{timing} -> {text!r}")


if __name__ == "__main__":
    main()
