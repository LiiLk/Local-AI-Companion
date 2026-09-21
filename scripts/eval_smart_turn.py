"""Manual Smart Turn v3.2 evaluation over a folder of WAV files.

Loads the real ONNX model (downloads it on first run) and prints the verdict,
probability and inference time for every ``*.wav`` in the given directory.

Usage (manual, not part of the test suite)::

    python scripts/eval_smart_turn.py path/to/wav_folder

Only 16 kHz mono WAV files (or files that soundfile can read without
resampling) are supported by the detector; other sample rates are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.vad.smart_turn import SmartTurnConfig, SmartTurnDetector


def _read_wav(path: Path):
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Smart Turn v3.2 on WAV files")
    parser.add_argument("folder", type=Path, help="Folder containing .wav files")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SmartTurnConfig.threshold,
        help="Decision threshold (default: 0.5)",
    )
    args = parser.parse_args(argv)

    if not args.folder.is_dir():
        print(f"Not a directory: {args.folder}")
        return 1

    detector = SmartTurnDetector(SmartTurnConfig(threshold=args.threshold))
    if not detector.warmup():
        print("Smart Turn model could not be loaded.")
        return 1

    wav_files = sorted(args.folder.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files in {args.folder}")
        return 1

    for path in wav_files:
        try:
            audio, sample_rate = _read_wav(path)
        except Exception as exc:
            print(f"{path.name}: failed to read ({exc})")
            continue

        if sample_rate != detector.config.sample_rate:
            print(f"{path.name}: skipped ({sample_rate}Hz != 16000Hz)")
            continue

        started = time.perf_counter()
        verdict = detector.predict(audio, sample_rate)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if verdict is None:
            print(f"{path.name}: unavailable")
            continue
        is_complete, probability = verdict
        label = "Complete" if is_complete else "Incomplete"
        print(
            f"{path.name}: {label} p={probability:.4f} "
            f"duration={len(audio) / sample_rate:.2f}s infer={elapsed_ms:.1f}ms"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())