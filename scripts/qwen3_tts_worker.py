from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def configure_torch_cache_dirs() -> None:
    cache_root = Path(
        os.environ.get("LOCALAPPDATA")
        or (Path.home() / "AppData" / "Local")
    ) / "Local-AI-Companion" / "torch-cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Qwen3-TTS worker")
    parser.add_argument("--model-id", required=False)
    parser.add_argument("--mode", default="voice_clone")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--speaker")
    parser.add_argument("--instruct")
    parser.add_argument("--ref-audio-path")
    parser.add_argument("--ref-text")
    parser.add_argument("--x-vector-only-mode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--site-packages-dir")
    parser.add_argument("--check-imports", action="store_true")
    return parser.parse_args()


def add_isolated_site_packages(site_packages_dir: str | None) -> None:
    if not site_packages_dir:
        return

    site_dir = Path(site_packages_dir).resolve()
    if not site_dir.exists():
        raise FileNotFoundError(f"Qwen3-TTS site-packages directory not found: {site_dir}")

    sys.path.insert(0, str(site_dir))
    if hasattr(os, "add_dll_directory"):
        for dll_dir in (site_dir, site_dir / "Library" / "bin"):
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))


def print_json(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(str(name).lower(), torch.bfloat16)


def resolve_attn_implementation(name: str) -> str:
    if name != "flash_attention_2":
        return name

    try:
        import flash_attn  # noqa: F401
    except Exception:
        return "sdpa"

    return "flash_attention_2"


def normalize_language(value: str) -> str:
    mapping = {
        "auto": "Auto",
        "": "Auto",
        "fr": "French",
        "fr-fr": "French",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
        "de": "German",
        "it": "Italian",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
    }
    key = (value or "auto").strip().lower()
    return mapping.get(key, key.title() if key else "Auto")


def load_model(args: argparse.Namespace):
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model_id,
        device_map=args.device,
        dtype=resolve_dtype(args.dtype),
        attn_implementation=resolve_attn_implementation(args.attn_implementation),
    )

    voice_clone_prompt = None
    if args.mode == "voice_clone" and args.ref_audio_path:
        kwargs = {
            "ref_audio": str(Path(args.ref_audio_path).resolve()),
            "x_vector_only_mode": parse_bool(args.x_vector_only_mode, default=args.ref_text is None),
        }
        if args.ref_text:
            kwargs["ref_text"] = args.ref_text
        voice_clone_prompt = model.create_voice_clone_prompt(**kwargs)

    return model, voice_clone_prompt


def synthesize_to_file(model, voice_clone_prompt, args: argparse.Namespace, request: dict) -> tuple[Path, float | None]:
    import numpy as np
    import soundfile as sf

    text = request["text"]
    output_path = Path(request["output_path"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    language = normalize_language(request.get("language") or args.language)
    mode = request.get("mode") or args.mode
    speaker = request.get("speaker") or args.speaker
    instruct = request.get("instruct") or args.instruct

    if mode == "custom_voice":
        kwargs = {
            "text": text,
            "language": language,
            "speaker": speaker,
        }
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sample_rate = model.generate_custom_voice(**kwargs)
    elif mode == "voice_design":
        if not instruct:
            raise ValueError("voice_design mode requires a non-empty instruct prompt")
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
    else:
        if voice_clone_prompt is None:
            raise ValueError("voice_clone mode requires a valid reference audio prompt")
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )

    audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    sf.write(str(output_path), audio_int16, sample_rate, subtype="PCM_16")
    duration = len(audio_int16) / sample_rate if sample_rate else None
    return output_path, duration


def main() -> int:
    args = parse_args()
    try:
        configure_torch_cache_dirs()
        add_isolated_site_packages(args.site_packages_dir)

        import qwen_tts  # noqa: F401

        if args.check_imports:
            print_json({"status": "ok", "backend": "worker"})
            return 0

        if not args.model_id:
            raise ValueError("--model-id is required unless --check-imports is used")

        model, voice_clone_prompt = load_model(args)
        print_json({"status": "ready", "backend": "worker"})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            command = request.get("command")

            if command == "shutdown":
                print_json({"status": "bye"})
                return 0

            if command != "synthesize":
                print_json({"status": "error", "message": f"Unsupported command: {command}"})
                continue

            output_path, duration = synthesize_to_file(model, voice_clone_prompt, args, request)
            print_json(
                {
                    "status": "ok",
                    "output_path": str(output_path),
                    "duration": duration,
                }
            )

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
