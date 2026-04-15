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


LANGUAGE_NAMES = {
    "fr": "French",
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "el": "Greek",
    "hi": "Hindi",
    "ms": "Malay",
    "fa": "Persian",
}
LANGUAGE_CODES = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Qwen3-ASR worker")
    parser.add_argument("--model-id", required=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", default="256")
    parser.add_argument("--site-packages-dir")
    parser.add_argument("--check-imports", action="store_true")
    return parser.parse_args()


def add_isolated_site_packages(site_packages_dir: str | None) -> None:
    if not site_packages_dir:
        return

    site_dir = Path(site_packages_dir).resolve()
    if not site_dir.exists():
        raise FileNotFoundError(f"Qwen3-ASR site-packages directory not found: {site_dir}")

    sys.path.insert(0, str(site_dir))
    if hasattr(os, "add_dll_directory"):
        for dll_dir in (site_dir, site_dir / "Library" / "bin"):
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))


def print_json(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def resolve_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(str(name).lower(), torch.bfloat16)


def normalize_language_code(language: str | None) -> str | None:
    if not language:
        return None

    value = str(language).strip().lower()
    if not value or value == "auto":
        return None
    if value in LANGUAGE_NAMES:
        return value
    if value in LANGUAGE_CODES:
        return LANGUAGE_CODES[value]
    if "-" in value:
        prefix = value.split("-", 1)[0]
        if prefix in LANGUAGE_NAMES:
            return prefix
    return value


def to_language_name(language: str | None) -> str | None:
    code = normalize_language_code(language)
    if not code:
        return None
    return LANGUAGE_NAMES.get(code, code)


def load_model(args: argparse.Namespace):
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        args.model_id,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device,
        max_new_tokens=int(args.max_new_tokens),
    )
    return model


def transcribe_request(model, request: dict) -> dict:
    audio_path = request["audio_path"]
    language = to_language_name(request.get("language"))
    results = model.transcribe(audio=audio_path, language=language)

    if not results:
        return {"text": "", "language": normalize_language_code(request.get("language"))}

    result = results[0]
    detected_language = getattr(result, "language", None)
    return {
        "text": (result.text or "").strip(),
        "language": normalize_language_code(detected_language) or normalize_language_code(request.get("language")),
    }


def main() -> int:
    args = parse_args()
    try:
        configure_torch_cache_dirs()
        add_isolated_site_packages(args.site_packages_dir)

        import qwen_asr  # noqa: F401

        if args.check_imports:
            print_json({"status": "ok", "backend": "worker"})
            return 0

        if not args.model_id:
            raise ValueError("--model-id is required unless --check-imports is used")

        model = load_model(args)
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

            if command != "transcribe":
                print_json({"status": "error", "message": f"Unsupported command: {command}"})
                continue

            payload = transcribe_request(model, request)
            payload["status"] = "ok"
            print_json(payload)

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
