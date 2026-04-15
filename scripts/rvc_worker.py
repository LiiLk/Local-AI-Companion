from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent RVC worker")
    parser.add_argument("--model-path", required=False)
    parser.add_argument("--index-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--f0-up-key", type=float, default=0.0)
    parser.add_argument("--output-freq", type=int)
    parser.add_argument("--site-packages-dir")
    parser.add_argument("--check-imports", action="store_true")
    return parser.parse_args()


def add_isolated_site_packages(site_packages_dir: str | None) -> None:
    if not site_packages_dir:
        return

    site_dir = Path(site_packages_dir).resolve()
    if not site_dir.exists():
        raise FileNotFoundError(f"RVC site-packages directory not found: {site_dir}")

    sys.path.insert(0, str(site_dir))
    if hasattr(os, "add_dll_directory"):
        for dll_dir in (site_dir, site_dir / "Library" / "bin"):
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))


def configure_environment(args: argparse.Namespace) -> None:
    if args.output_freq:
        os.environ["RVC_OUTPUTFREQ"] = str(args.output_freq)


def patch_torch_load() -> None:
    import torch

    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    if hasattr(torch, "serialization"):
        torch.serialization.load = compat_torch_load


def sanitize_argv() -> None:
    sys.argv = [sys.argv[0]]


def create_rvc(args: argparse.Namespace):
    from inferrvc import RVC
    from inferrvc.configs.config import Config

    config = Config()
    if args.device == "cpu":
        config.device = "cpu"
        config.is_half = False
        config.x_pad = 1
        config.x_query = 6
        config.x_center = 38
        config.x_max = 41
    else:
        config.device = args.device

    return RVC(
        model=str(Path(args.model_path).resolve()),
        index=str(Path(args.index_path).resolve()) if args.index_path else None,
        config=config,
    )


def disable_internal_output_resample(rvc) -> int:
    """
    Force InferRVC to keep the model's native sample rate.

    InferRVC's built-in output resampler can fail on recent Torch/Torchaudio
    combinations when the model runs in half precision on CUDA. We keep the raw
    model output here and perform any optional resampling ourselves on CPU.
    """
    target_sr = int(getattr(rvc, "tgt_sr", 0) or 0)
    if target_sr > 0:
        setattr(rvc, "outputfreq", target_sr)
    return target_sr


def resample_audio_if_needed(audio, sample_rate: int, target_rate: int | None):
    if not target_rate or target_rate == sample_rate:
        return audio, sample_rate

    from scipy.signal import resample_poly

    up = target_rate
    down = sample_rate
    factor = math.gcd(up, down)
    up //= factor
    down //= factor

    resampled = resample_poly(audio, up, down).astype(audio.dtype, copy=False)
    return resampled, target_rate


def print_json(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def run_conversion(rvc, request: dict) -> None:
    import numpy as np
    import soundfile as sf

    input_path = Path(request["input_path"]).resolve()
    output_path = Path(request["output_path"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = sf.read(input_path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    minimum_duration_sec = (
        float(getattr(getattr(rvc, "config", None), "x_pad", 3)) + 0.05
    )
    current_duration_sec = audio.shape[0] / float(sample_rate)
    padded_input_path = None
    if current_duration_sec <= minimum_duration_sec:
        target_samples = int(np.ceil(sample_rate * minimum_duration_sec))
        pad_samples = max(0, target_samples - audio.shape[0])
        if pad_samples > 0:
            pad_left = pad_samples // 2
            pad_right = pad_samples - pad_left
            pad_mode = "edge" if audio.shape[0] > 1 else "constant"
            audio = np.pad(audio, (pad_left, pad_right), mode=pad_mode)
            padded_input_path = output_path.with_suffix(".input.wav")
            sf.write(padded_input_path, audio, sample_rate)
            input_path = padded_input_path

    try:
        result = rvc(
            str(input_path),
            f0_up_key=request.get("f0_up_key", 0.0),
            f0_method=request.get("f0_method", "rmvpe"),
            index_rate=request.get("index_rate", 0.75),
            protect=request.get("protect", 0.33),
            output_volume=getattr(rvc, "NO_CHANGE", 2),
        )

        audio = (
            result.detach().float().cpu().numpy()
            if hasattr(result, "detach")
            else result
        )
        sample_rate = int(
            getattr(rvc, "tgt_sr", None) or getattr(rvc, "outputfreq", None) or 44100
        )
        audio, sample_rate = resample_audio_if_needed(
            np.asarray(audio).squeeze(),
            sample_rate,
            request.get("output_freq"),
        )
        sf.write(output_path, audio.squeeze(), sample_rate)
        print_json({"status": "ok", "output_path": str(output_path)})
    finally:
        if padded_input_path:
            padded_input_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    try:
        add_isolated_site_packages(args.site_packages_dir)
        configure_environment(args)
        sanitize_argv()
        patch_torch_load()

        import inferrvc  # noqa: F401

        if args.check_imports:
            print_json({"status": "ok", "backend": "inferrvc"})
            return 0

        if not args.model_path:
            raise ValueError("--model-path is required unless --check-imports is used")

        rvc = create_rvc(args)
        disable_internal_output_resample(rvc)
        print_json({"status": "ready", "backend": "inferrvc"})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            command = request.get("command")

            if command == "shutdown":
                print_json({"status": "bye"})
                return 0

            if command != "convert":
                print_json(
                    {"status": "error", "message": f"Unsupported command: {command}"}
                )
                continue

            request.setdefault("f0_up_key", args.f0_up_key)
            request.setdefault("f0_method", args.f0_method)
            request.setdefault("index_rate", args.index_rate)
            request.setdefault("protect", args.protect)
            request.setdefault("output_freq", args.output_freq)
            run_conversion(rvc, request)

        return 0
    except Exception as exc:
        print_json(
            {
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
