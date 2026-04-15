"""
Qwen3-ASR provider with optional worker isolation.

The worker backend is the recommended deployment for this project because
Qwen-ASR pins ``transformers==4.57.6`` while Gemma 4 requires the 5.x line.
Running Qwen3-ASR in its own Python environment keeps the main app stable.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import wave
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import numpy as np

from .base import ASRResult, ASRSegment, BaseASR

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QWEN3_ASR_WORKER = PROJECT_ROOT / "scripts" / "qwen3_asr_worker.py"

# Full language name mapping for Qwen3-ASR
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


class Qwen3ASRProvider(BaseASR):
    """
    ASR provider using Qwen3-ASR (0.6B or 1.7B).

    The default ``worker`` backend keeps Qwen-ASR in a dedicated environment,
    so Gemma 4 can continue using the main app's ``transformers`` 5.x runtime.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-0.6B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        backend: str = "worker",
        python_path: str | Path | None = None,
        site_packages_dir: str | Path | None = None,
        worker_script: str | Path | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.backend = (backend or "worker").lower()
        self.python_path = (
            Path(python_path).resolve() if python_path else Path(sys.executable).resolve()
        )
        self.site_packages_dir = (
            Path(site_packages_dir).resolve() if site_packages_dir else None
        )
        self.worker_script = (
            Path(worker_script).resolve()
            if worker_script
            else DEFAULT_QWEN3_ASR_WORKER.resolve()
        )
        self._model = None
        self._load_lock = threading.Lock()
        self._worker_process: subprocess.Popen[str] | None = None
        self._worker_lock = threading.Lock()
        self._worker_stderr: deque[str] = deque(maxlen=50)
        self._worker_stderr_thread: threading.Thread | None = None

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
            taskkill = system_root / "System32" / "taskkill.exe"
            command = [str(taskkill if taskkill.exists() else "taskkill"), "/PID", str(process.pid), "/T", "/F"]
            try:
                subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
                return
            except Exception:
                pass

        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass

    @staticmethod
    def _normalize_language_code(language: Optional[str]) -> Optional[str]:
        if not language:
            return None

        value = language.strip().lower()
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

    @classmethod
    def _worker_import_check(
        cls,
        python_path: Path,
        worker_script: Path,
        site_packages_dir: Path | None = None,
    ) -> bool:
        command = [str(python_path), str(worker_script), "--check-imports"]
        if site_packages_dir and site_packages_dir.exists():
            command.extend(["--site-packages-dir", str(site_packages_dir)])

        try:
            result = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
        except Exception:
            return False

        if result.returncode != 0:
            return False

        stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            return False

        try:
            payload = json.loads(stdout_lines[-1])
        except json.JSONDecodeError:
            return False

        return payload.get("status") == "ok"

    @classmethod
    def is_available(
        cls,
        backend: str = "worker",
        python_path: str | Path | None = None,
        site_packages_dir: str | Path | None = None,
        worker_script: str | Path | None = None,
    ) -> bool:
        backend = (backend or "worker").lower()
        if backend == "inprocess":
            return True

        python_path_resolved = (
            Path(python_path).resolve() if python_path else Path(sys.executable).resolve()
        )
        site_packages_resolved = (
            Path(site_packages_dir).resolve() if site_packages_dir else None
        )
        worker_script_resolved = (
            Path(worker_script).resolve()
            if worker_script
            else DEFAULT_QWEN3_ASR_WORKER.resolve()
        )
        return cls._worker_import_check(
            python_path=python_path_resolved,
            worker_script=worker_script_resolved,
            site_packages_dir=site_packages_resolved,
        )

    def _read_worker_stderr(self) -> None:
        process = self._worker_process
        if not process or not process.stderr:
            return

        try:
            for line in process.stderr:
                if not line:
                    break
                self._worker_stderr.append(line.rstrip())
        except Exception:
            return

    def _worker_command(self) -> list[str]:
        command = [
            str(self.python_path),
            str(self.worker_script),
            "--model-id",
            self.model_id,
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        if self.site_packages_dir and self.site_packages_dir.exists():
            command.extend(["--site-packages-dir", str(self.site_packages_dir)])
        return command

    def _spawn_worker(self) -> None:
        if self._worker_process is not None:
            return

        process = subprocess.Popen(
            self._worker_command(),
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        self._worker_process = process
        self._worker_stderr.clear()
        self._worker_stderr_thread = threading.Thread(
            target=self._read_worker_stderr,
            daemon=True,
            name="Qwen3ASRWorkerStderr",
        )
        self._worker_stderr_thread.start()

        if not process.stdout:
            raise RuntimeError("Qwen3-ASR worker stdout is unavailable")

        startup_lines: list[str] = []
        payload: dict | None = None

        while True:
            ready_line = process.stdout.readline()
            if not ready_line:
                stderr = "\n".join(self._worker_stderr)
                details = "\n".join(startup_lines).strip()
                raise RuntimeError(details or stderr or "Qwen3-ASR worker failed to start")

            ready_line = ready_line.strip()
            if not ready_line:
                continue

            startup_lines.append(ready_line)
            try:
                payload = json.loads(ready_line)
                break
            except json.JSONDecodeError:
                continue

        if payload.get("status") != "ready":
            stderr = "\n".join(self._worker_stderr)
            details = payload.get("message") or "\n".join(startup_lines).strip()
            raise RuntimeError(details or stderr or "Qwen3-ASR worker failed to initialize")

    def _shutdown_worker(self) -> None:
        process = self._worker_process
        if process is None:
            return

        try:
            if process.stdin and process.stdout:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
                process.stdout.readline()
        except Exception:
            pass

        try:
            process.wait(timeout=5)
        except Exception:
            self._kill_process_tree(process)

        self._worker_process = None
        self._worker_stderr_thread = None

    def _send_worker_command(self, payload: dict) -> dict:
        with self._worker_lock:
            self._spawn_worker()
            process = self._worker_process
            if not process or not process.stdin or not process.stdout:
                raise RuntimeError("Qwen3-ASR worker is not available")

            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()

            line = process.stdout.readline().strip()
            if not line:
                stderr = "\n".join(self._worker_stderr)
                raise RuntimeError(stderr or "Qwen3-ASR worker stopped unexpectedly")

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid response from Qwen3-ASR worker: {line}") from exc

            if response.get("status") != "ok":
                raise RuntimeError(response.get("message", "Qwen3-ASR worker failed"))

            return response

    def _get_model(self):
        if self.backend == "worker":
            self._spawn_worker()
            return None

        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            import torch
            from qwen_asr import Qwen3ASRModel

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

            logger.info(
                "Loading Qwen3-ASR model %s on %s (%s)...",
                self.model_id,
                self.device,
                self.dtype,
            )
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
                device_map=self.device,
                max_new_tokens=self.max_new_tokens,
            )
            logger.info("Qwen3-ASR loaded.")
            return self._model

    def preload(self) -> None:
        self._get_model()

    def _write_temp_wav(self, audio_input) -> tuple[Path, bool]:
        if isinstance(audio_input, (str, Path)):
            return Path(audio_input).resolve(), False

        if not isinstance(audio_input, np.ndarray):
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        audio = np.asarray(audio_input, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_path = Path(tmp_name)

        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        return tmp_path.resolve(), True

    def _transcribe_inprocess(
        self,
        audio_input,
        language: Optional[str] = None,
    ) -> ASRResult:
        model = self._get_model()

        if isinstance(audio_input, np.ndarray):
            audio_arg = (audio_input, 16000)
        elif isinstance(audio_input, (str, Path)):
            audio_arg = str(audio_input)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        language_code = self._normalize_language_code(language)
        lang_name = LANGUAGE_NAMES.get(language_code, None) if language_code else None

        logger.info("Qwen3-ASR transcribing (lang=%s)...", lang_name or "auto")

        results = model.transcribe(audio=audio_arg, language=lang_name)

        if not results:
            return ASRResult(text="", language=language_code)

        result = results[0]
        detected_lang = getattr(result, "language", None)
        detected_code = language_code
        if detected_lang:
            detected_code = self._normalize_language_code(detected_lang) or language_code

        text = result.text.strip() if result.text else ""
        logger.info("Qwen3-ASR result: %r (lang=%s)", text[:80], detected_code)
        return ASRResult(text=text, language=detected_code)

    def transcribe(
        self,
        audio_input,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> ASRResult:
        """
        Transcribe audio to text.

        Args:
            audio_input: numpy float32 array (16 kHz) or path to audio file.
            language: ISO 639-1 code ("fr", "en", …) or None for auto-detect.
            initial_prompt: Unused, kept for interface compatibility.
        """
        _ = initial_prompt

        if self.backend != "worker":
            return self._transcribe_inprocess(audio_input, language=language)

        audio_path, should_delete = self._write_temp_wav(audio_input)
        try:
            response = self._send_worker_command(
                {
                    "command": "transcribe",
                    "audio_path": str(audio_path),
                    "language": self._normalize_language_code(language),
                }
            )
            text = (response.get("text") or "").strip()
            detected_code = self._normalize_language_code(response.get("language")) or self._normalize_language_code(language)
            return ASRResult(text=text, language=detected_code)
        finally:
            if should_delete:
                audio_path.unlink(missing_ok=True)

    def transcribe_stream(
        self,
        audio_path,
        language: Optional[str] = None,
    ) -> AsyncGenerator[ASRSegment, None]:
        """Not implemented — Qwen3-ASR streaming requires a dedicated server backend."""
        raise NotImplementedError("Streaming requires a dedicated Qwen3-ASR server backend")

    def get_supported_languages(self) -> List[str]:
        return list(LANGUAGE_NAMES.keys())

    def get_model_info(self) -> dict:
        return {
            "name": self.model_id,
            "type": "Qwen3-ASR",
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
        }

    def cleanup(self) -> None:
        self._shutdown_worker()
        if self._model is not None:
            self._model = None
