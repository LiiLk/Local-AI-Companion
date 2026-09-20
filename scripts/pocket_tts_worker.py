import argparse
import base64
import json
import os
from pathlib import Path
import sys


MAX_REQUEST_BYTES = 1048576


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def send(payload):
        output.write(json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n")
        output.flush()

    provider = None
    try:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from src.tts.pocket_tts_provider import PocketTTSProvider

            provider = PocketTTSProvider(voice=args.voice, language=args.language)
            provider.preload()
        except Exception:
            send({"status": "error", "error": "startup_failed"})
            return 1
        send({"status": "ready", "protocol": 1})
        while True:
            line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                return 0
            request_id = None
            try:
                if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                    raise ValueError("Invalid request size")
                request = json.loads(line)
                request_id = request["id"]
                if not isinstance(request_id, int):
                    raise ValueError("Invalid request identifier")
                provider.set_voice(request["voice"])
                operation = request["operation"]
                if operation == "preload":
                    provider.preload()
                    response = {}
                elif operation == "warmup":
                    provider.warmup()
                    response = {}
                elif operation == "synthesize":
                    result = provider._synthesize_sync(request["text"])
                    response = {
                        "audio": base64.b64encode(result.audio_data).decode("ascii"),
                        "metadata": result.metadata,
                    }
                else:
                    raise ValueError("Unknown operation")
                send({"id": request_id, "status": "ok", **response})
            except Exception:
                send({"id": request_id, "status": "error", "error": "request_failed"})
                return 1
    finally:
        if provider is not None:
            provider.cleanup()
        output.close()


if __name__ == "__main__":
    raise SystemExit(main())
