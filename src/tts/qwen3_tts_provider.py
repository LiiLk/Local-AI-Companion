"""
Qwen3-TTS provider for low-latency local voice cloning.

The recommended deployment for this project is the ``worker`` backend:
run Qwen3-TTS inside a dedicated Python environment so its Transformers 4.x
stack stays isolated from Gemma in the main application environment.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import logging
import os
# Worker subprocesses are local, argument-list based, and validated.
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soundfile as sf

from src.utils.language_detection import detect_language, normalize_language_code

from .base import BaseTTS, TTSResult, Voice

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QWEN3_SITE_PACKAGES = PROJECT_ROOT / ".qwen3-tts-overlay"
DEFAULT_QWEN3_WORKER = PROJECT_ROOT / "scripts" / "qwen3_tts_worker.py"

LANGUAGE_NAMES = {
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

WARMUP_TEXTS = {
    "de": "Hallo.",
    "en": "Hello.",
    "es": "Hola.",
    "fr": "Bonjour.",
    "it": "Ciao.",
    "ja": "こんにちは。",
    "ko": "안녕하세요.",
    "pt": "Olá.",
    "ru": "Привет.",
    "zh": "你好。",
}

DEFAULT_CUSTOM_SPEAKERS = [
    Voice(id="Vivian", name="Vivian", language="zh", gender="Female"),
    Voice(id="Serena", name="Serena", language="zh", gender="Female"),
    Voice(id="Uncle_Fu", name="Uncle Fu", language="zh", gender="Male"),
    Voice(id="Dylan", name="Dylan", language="zh", gender="Male"),
    Voice(id="Eric", name="Eric", language="zh", gender="Male"),
    Voice(id="Ryan", name="Ryan", language="en", gender="Male"),
    Voice(id="Aiden", name="Aiden", language="en", gender="Male"),
    Voice(id="Ono_Anna", name="Ono Anna", language="ja", gender="Female"),
    Voice(id="Sohee", name="Sohee", language="ko", gender="Female"),
]

CUSTOM_SPEAKER_LANGUAGE_CODES = {
    voice.id: normalize_language_code(voice.language) or "en"
    for voice in DEFAULT_CUSTOM_SPEAKERS
}


class Qwen3TTSProvider(BaseTTS):
    """TTS provider backed by Qwen3-TTS."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mode: str = "voice_clone",
        language: str = "auto",
        speaker: str | None = None,
        instruct: str | None = None,
        ref_audio_path: str | Path | None = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        backend: str = "worker",
        python_path: str | Path | None = None,
        site_packages_dir: str | Path | None = None,
        worker_script: str | Path | None = None,
        request_timeout_sec: float = 20.0,
    ):
        self.model_id = model_id
        self.mode = mode
        self.language = language
        self.speaker = speaker or "Ryan"
        self.instruct = instruct or ""
        self.ref_audio_path = Path(ref_audio_path).resolve() if ref_audio_path else None
        self.ref_text = ref_text.strip() if isinstance(ref_text, str) and ref_text.strip() else None
        self.x_vector_only_mode = x_vector_only_mode
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.backend = backend
        self.python_path = (
            Path(python_path).resolve() if python_path else Path(sys.executable).resolve()
        )
        self.site_packages_dir = (
            Path(site_packages_dir).resolve()
            if site_packages_dir
            else DEFAULT_QWEN3_SITE_PACKAGES.resolve()
        )
        self.worker_script = (
            Path(worker_script).resolve()
            if worker_script
            else DEFAULT_QWEN3_WORKER.resolve()
        )
        self.request_timeout_sec = float(request_timeout_sec)

        self._model = None
        self._voice_clone_prompt = None
        self._load_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen3-tts")
        self._worker_process: subprocess.Popen[str] | None = None
        self._worker_lock = threading.Lock()
        self._worker_stderr: deque[str] = deque(maxlen=50)
        self._worker_stderr_thread: threading.Thread | None = None
        self._rate = "+0%"
        self._pitch = "+0Hz"
        self._language_hint: str | None = None
        self._warmed_up = False
        self._attn_implementation_actual: str | None = None
        self._worker_ready_payload: dict = {}
        self.disabled_reason: str | None = None

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            pid = int(process.pid)
            if pid <= 0:
                return
            system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
            taskkill = system_root / "System32" / "taskkill.exe"
            taskkill_exe = str(taskkill.resolve()) if taskkill.is_file() else "taskkill"
            command = [taskkill_exe, "/PID", str(pid), "/T", "/F"]
            try:
                # Fixed taskkill command with validated PID and shell=False.
                subprocess.run(  # nosec B603
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                    shell=False,
                )
                return
            except Exception:
                logger.debug("Failed to terminate Qwen3-TTS worker process tree", exc_info=True)

        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            logger.debug("Failed to kill Qwen3-TTS worker process", exc_info=True)

    @staticmethod
    def resolve_mode_for_model(
        model_id: str | None,
        requested_mode: str | None,
    ) -> tuple[str, str | None]:
        mode = (requested_mode or "voice_clone").strip().lower().replace("-", "_")
        model_name = (model_id or "").strip().lower()

        if "customvoice" in model_name and mode == "voice_clone":
            return (
                "custom_voice",
                "Qwen3-TTS CustomVoice model does not support voice_clone; using custom_voice.",
            )

        return mode, None

    def _resolve_dtype(self):
        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping.get(str(self.dtype).lower(), torch.bfloat16)

    def _resolve_attn_implementation(self) -> str:
        if self.attn_implementation != "flash_attention_2":
            return self.attn_implementation

        if importlib.util.find_spec("flash_attn") is None:
            logger.warning("flash-attn is not installed; falling back to sdpa for Qwen3-TTS")
            return "sdpa"

        return "flash_attention_2"

    def _normalize_language(self, language: str | None = None) -> str:
        value = (language or self.language or "auto").strip().lower()
        return LANGUAGE_NAMES.get(value, value.title() if value else "Auto")

    def _normalize_language_code(self, language: str | None) -> str | None:
        return normalize_language_code(language)

    @staticmethod
    def _describe_exception(exc: BaseException) -> str:
        message = str(exc).strip()
        if message:
            return f"{type(exc).__name__}: {message}"
        return type(exc).__name__

    def _ensure_enabled(self) -> None:
        if self.disabled_reason:
            raise RuntimeError(f"Qwen3-TTS is disabled for this session ({self.disabled_reason})")

    def _disable_for_session(self, phase: str, exc: BaseException) -> None:
        reason = f"{phase}: {self._describe_exception(exc)}"
        if self.disabled_reason == reason:
            return

        self.disabled_reason = reason
        logger.warning("Qwen3-TTS disabled for current session: %s", reason)

        if self.backend == "worker":
            self._force_kill_worker()
            return

        if self._model is not None:
            del self._model
            self._model = None
        self._voice_clone_prompt = None

    def _reset_session_failure(self) -> None:
        self.disabled_reason = None

    def _resolve_warmup_language_code(self) -> str:
        if self.mode == "custom_voice":
            speaker_code = CUSTOM_SPEAKER_LANGUAGE_CODES.get(self.speaker or "")
            if speaker_code:
                return speaker_code

        hint_code = self._normalize_language_code(self._language_hint)
        if hint_code:
            return hint_code

        configured_code = self._normalize_language_code(self.language)
        if configured_code:
            return configured_code

        return "en"

    def _warmup_text(self) -> str:
        language_code = self._resolve_warmup_language_code()
        return WARMUP_TEXTS.get(language_code, WARMUP_TEXTS["en"])

    def _resolve_request_language(self, text: str) -> str:
        explicit_code = self._normalize_language_code(self.language)
        if explicit_code:
            return self._normalize_language(explicit_code)

        hint_code = self._normalize_language_code(self._language_hint)
        if hint_code:
            return self._normalize_language(hint_code)

        if text and text.strip():
            detected = detect_language(text, default="en")
            return self._normalize_language(str(detected))

        return self._normalize_language(hint_code)

    def _resolve_x_vector_only_mode(self) -> bool:
        if self.x_vector_only_mode is not None:
            return bool(self.x_vector_only_mode)
        return self.ref_text is None

    @classmethod
    def _worker_import_check(
        cls,
        python_path: Path,
        worker_script: Path,
        site_packages_dir: Path | None = None,
    ) -> bool:
        try:
            cls._validate_worker_process_inputs(python_path, worker_script)
        except RuntimeError as exc:
            logger.debug("Qwen3-TTS worker import check skipped: %s", exc)
            return False

        command = [str(python_path), str(worker_script), "--check-imports"]
        if site_packages_dir and site_packages_dir.exists():
            command.extend(["--site-packages-dir", str(site_packages_dir)])

        try:
            # Validated local worker script, shell=False.
            result = subprocess.run(  # nosec B603
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
                shell=False,
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
            return importlib.util.find_spec("qwen_tts") is not None

        python_path = (
            Path(python_path).resolve() if python_path else Path(sys.executable).resolve()
        )
        site_packages_dir = (
            Path(site_packages_dir).resolve()
            if site_packages_dir
            else DEFAULT_QWEN3_SITE_PACKAGES.resolve()
        )
        worker_script = (
            Path(worker_script).resolve()
            if worker_script
            else DEFAULT_QWEN3_WORKER.resolve()
        )
        return cls._worker_import_check(
            python_path=python_path,
            worker_script=worker_script,
            site_packages_dir=site_packages_dir,
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
            "--mode",
            self.mode,
            "--language",
            self.language or "auto",
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--attn-implementation",
            self.attn_implementation,
        ]
        if self.speaker:
            command.extend(["--speaker", self.speaker])
        if self.instruct:
            command.extend(["--instruct", self.instruct])
        if self.ref_audio_path:
            command.extend(["--ref-audio-path", str(self.ref_audio_path)])
        if self.ref_text:
            command.extend(["--ref-text", self.ref_text])
        if self.x_vector_only_mode is not None:
            command.extend(["--x-vector-only-mode", "true" if self.x_vector_only_mode else "false"])
        if self.site_packages_dir and self.site_packages_dir.exists():
            command.extend(["--site-packages-dir", str(self.site_packages_dir)])
        return command

    @staticmethod
    def _validate_worker_process_inputs(python_path: Path, worker_script: Path) -> None:
        if not python_path.is_file():
            raise RuntimeError(f"Qwen3-TTS worker Python executable not found: {python_path}")
        if not worker_script.is_file():
            raise RuntimeError(f"Qwen3-TTS worker script not found: {worker_script}")
        if worker_script.suffix.lower() != ".py":
            raise RuntimeError(f"Qwen3-TTS worker script must be a Python file: {worker_script}")

    def _spawn_worker(self) -> None:
        if self._worker_process is not None:
            return

        self._validate_worker_process_inputs(self.python_path, self.worker_script)
        # Validated local worker script, shell=False.
        process = subprocess.Popen(  # nosec B603
            self._worker_command(),
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )

        self._worker_process = process
        self._worker_stderr.clear()
        self._worker_stderr_thread = threading.Thread(
            target=self._read_worker_stderr,
            daemon=True,
            name="Qwen3TTSWorkerStderr",
        )
        self._worker_stderr_thread.start()

        if not process.stdout:
            raise RuntimeError("Qwen3-TTS worker stdout is unavailable")

        startup_lines: list[str] = []
        payload: dict | None = None

        while True:
            ready_line = process.stdout.readline()
            if not ready_line:
                stderr = "\n".join(self._worker_stderr)
                details = "\n".join(startup_lines).strip()
                raise RuntimeError(
                    details or stderr or "Qwen3-TTS worker failed to start"
                )

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
            raise RuntimeError(details or stderr or "Qwen3-TTS worker failed to initialize")

        self._worker_ready_payload = payload
        self._attn_implementation_actual = payload.get("attn_implementation") or self.attn_implementation
        logger.info("Qwen3-TTS worker ready (attn=%s)", self._attn_implementation_actual)

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
            logger.debug("Failed to request Qwen3-TTS worker shutdown", exc_info=True)

        try:
            process.wait(timeout=5)
        except Exception:
            self._kill_process_tree(process)

        self._worker_process = None
        self._worker_stderr_thread = None

    def _force_kill_worker(self) -> None:
        process = self._worker_process
        if process is None:
            return
        self._kill_process_tree(process)
        self._worker_process = None
        self._worker_stderr_thread = None

    def _restart_worker(self) -> None:
        if self.backend == "worker":
            self._shutdown_worker()

    def cancel_inflight(self) -> None:
        """
        Best-effort cancellation for long-running synth requests.
        This is used when a speech turn is interrupted (barge-in).
        """
        if self.backend != "worker":
            return
        self._force_kill_worker()

    def _send_worker_command(self, payload: dict) -> dict:
        with self._worker_lock:
            self._spawn_worker()
            process = self._worker_process
            if not process or not process.stdin or not process.stdout:
                raise RuntimeError("Qwen3-TTS worker is not available")

            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()

            line = process.stdout.readline().strip()
            if not line:
                stderr = "\n".join(self._worker_stderr)
                raise RuntimeError(stderr or "Qwen3-TTS worker stopped unexpectedly")

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid response from Qwen3-TTS worker: {line}") from exc

            if response.get("status") != "ok":
                raise RuntimeError(response.get("message", "Qwen3-TTS worker synthesis failed"))

            return response

    def _load_model_inprocess(self):
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            try:
                from qwen_tts import Qwen3TTSModel
            except ImportError as exc:
                raise ImportError(
                    "qwen-tts is not installed in the main environment."
                ) from exc

            logger.info("Loading Qwen3-TTS in-process from %s...", self.model_id)
            actual_attn = self._resolve_attn_implementation()
            self._attn_implementation_actual = actual_attn
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=self._resolve_dtype(),
                attn_implementation=actual_attn,
            )
            logger.info("Qwen3-TTS in-process ready (attn=%s)", actual_attn)

            if self.mode == "voice_clone" and self.ref_audio_path and self.ref_audio_path.exists():
                self._voice_clone_prompt = self._create_voice_clone_prompt()

    def _load_model(self):
        self._ensure_enabled()
        if self.backend == "worker":
            self._spawn_worker()
            return
        self._load_model_inprocess()

    def preload(self) -> None:
        self._ensure_enabled()
        try:
            self._load_model()
        except Exception as exc:
            self._disable_for_session("preload failed", exc)
            raise

    @property
    def prefer_full_response_tts(self) -> bool:
        """
        `custom_voice` is the low-latency path and works better with sentence-level
        queueing. Voice cloning/design stay full-response for stability.
        """
        return self.mode != "custom_voice"

    def warmup(self) -> None:
        """Pay the first synthesis cost before the first real user reply."""
        self._ensure_enabled()
        if self._warmed_up:
            return

        warmup_text = self._warmup_text()
        try:
            if self.backend == "worker":
                self._synthesize_worker_sync(warmup_text)
            else:
                self._generate_sync_inprocess(warmup_text)
            self._warmed_up = True
        except Exception as exc:
            self._disable_for_session("warmup failed", exc)
            raise

    def _create_voice_clone_prompt(self):
        if not self.ref_audio_path or not self.ref_audio_path.exists():
            raise ValueError(
                "Qwen3-TTS voice cloning requires a valid reference audio file."
            )

        kwargs = {
            "ref_audio": str(self.ref_audio_path),
            "x_vector_only_mode": self._resolve_x_vector_only_mode(),
        }
        if self.ref_text:
            kwargs["ref_text"] = self.ref_text

        return self._model.create_voice_clone_prompt(**kwargs)

    def _generate_sync_inprocess(self, text: str) -> tuple[np.ndarray, int]:
        self._load_model_inprocess()
        language = self._resolve_request_language(text)

        if self.mode == "custom_voice":
            kwargs = {
                "text": text,
                "language": language,
                "speaker": self.speaker,
            }
            if self.instruct:
                kwargs["instruct"] = self.instruct
            wavs, sample_rate = self._model.generate_custom_voice(**kwargs)
        elif self.mode == "voice_design":
            if not self.instruct:
                raise ValueError(
                    "Qwen3-TTS voice_design mode requires a non-empty instruct prompt."
                )
            wavs, sample_rate = self._model.generate_voice_design(
                text=text,
                language=language,
                instruct=self.instruct,
            )
        else:
            prompt = self._voice_clone_prompt
            if prompt is None:
                prompt = self._create_voice_clone_prompt()
                self._voice_clone_prompt = prompt

            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
            )

        audio = np.asarray(wavs[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        return audio, int(sample_rate)

    def _synthesize_worker_sync(
        self,
        text: str,
        output_path: Path | None = None,
    ) -> TTSResult:
        request_started = time.perf_counter()
        request = {
            "command": "synthesize",
            "text": text,
            "language": self._resolve_request_language(text),
        }
        if output_path is not None:
            request["output_path"] = str(output_path)

        response = self._send_worker_command(request)
        duration = response.get("duration")
        metadata = {
            "backend": "worker",
            "attn_implementation": response.get("attn_implementation") or self._attn_implementation_actual,
            "synth_ms": float(response.get("synth_ms", 0.0) or 0.0),
            "file_write_ms": float(response.get("file_write_ms", 0.0) or 0.0),
            "file_read_ms": 0.0,
            "provider_roundtrip_ms": (time.perf_counter() - request_started) * 1000,
        }

        generated_path = None
        if response.get("output_path"):
            generated_path = Path(response["output_path"]).resolve()

        if output_path is not None:
            if generated_path is not None:
                return TTSResult(audio_path=generated_path, duration=duration, metadata=metadata)

            audio_data = None
            if response.get("audio_base64"):
                audio_data = base64.b64decode(response["audio_base64"])
            if audio_data is not None:
                write_started = time.perf_counter()
                output_path.write_bytes(audio_data)
                metadata["file_write_ms"] += (time.perf_counter() - write_started) * 1000
            return TTSResult(audio_path=output_path, duration=duration, metadata=metadata)

        audio_data: bytes | None = None
        if response.get("audio_base64"):
            audio_data = base64.b64decode(response["audio_base64"])

        if audio_data is not None:
            return TTSResult(audio_data=audio_data, duration=duration, metadata=metadata)

        if generated_path is not None:
            read_started = time.perf_counter()
            audio_data = generated_path.read_bytes()
            metadata["file_read_ms"] += (time.perf_counter() - read_started) * 1000
            return TTSResult(audio_data=audio_data, duration=duration, metadata=metadata)

        raise RuntimeError("Qwen3-TTS worker returned neither audio bytes nor output path")

    async def synthesize(

        self,
        text: str,
        output_path: Path | None = None,
    ) -> TTSResult:
        self._ensure_enabled()
        loop = asyncio.get_running_loop()
        try:
            if self.backend == "worker":
                return await loop.run_in_executor(
                    self._executor,
                    self._synthesize_worker_sync,
                    text,
                    output_path,
                )

            audio, sample_rate = await loop.run_in_executor(
                self._executor,
                self._generate_sync_inprocess,
                text,
            )

            audio_int16 = (audio * 32767).astype(np.int16)
            duration = len(audio_int16) / sample_rate if sample_rate else None
            metadata = {
                "backend": "inprocess",
                "attn_implementation": self._attn_implementation_actual,
                "file_write_ms": 0.0,
                "file_read_ms": 0.0,
            }

            if output_path is not None:
                write_started = time.perf_counter()
                sf.write(str(output_path), audio_int16, sample_rate, subtype="PCM_16")
                metadata["file_write_ms"] = (time.perf_counter() - write_started) * 1000
                return TTSResult(audio_path=output_path, duration=duration, metadata=metadata)

            buffer = io.BytesIO()
            sf.write(buffer, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
            return TTSResult(audio_data=buffer.getvalue(), duration=duration, metadata=metadata)
        except Exception as exc:
            self._disable_for_session("synthesis failed", exc)
            raise

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        result = await self.synthesize(text)
        data = result.audio_data
        if data is None and result.audio_path:
            data = result.audio_path.read_bytes()
        if not data:
            return

        chunk_size = 64 * 1024
        for index in range(0, len(data), chunk_size):
            yield data[index:index + chunk_size]

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        if self.mode != "custom_voice":
            return [
                Voice(
                    id=str(self.ref_audio_path) if self.ref_audio_path else "reference",
                    name="Reference Voice",
                    language=language or self.language or "auto",
                    gender="Unknown",
                )
            ]

        voices = DEFAULT_CUSTOM_SPEAKERS
        if not language:
            return voices

        language_prefix = language.lower()
        return [voice for voice in voices if voice.language.lower().startswith(language_prefix)]

    def set_voice(self, voice_id: str) -> None:
        path = Path(voice_id)
        if path.exists():
            self.ref_audio_path = path.resolve()
            self._voice_clone_prompt = None
            self._reset_session_failure()
            self._restart_worker()
            return

        self.speaker = voice_id
        self._reset_session_failure()
        self._restart_worker()

    def set_language(self, language: str | None) -> None:
        self._language_hint = self._normalize_language_code(language)

    def set_rate(self, rate: str) -> None:
        self._rate = rate

    def set_pitch(self, pitch: str) -> None:
        self._pitch = pitch

    def set_reference_audio(self, ref_audio_path: str | Path, ref_text: str | None = None) -> None:
        self.ref_audio_path = Path(ref_audio_path).resolve()
        self.ref_text = ref_text.strip() if isinstance(ref_text, str) and ref_text.strip() else None
        self._voice_clone_prompt = None
        self._reset_session_failure()
        self._restart_worker()

    def cleanup(self) -> None:
        self._shutdown_worker()
        if self._model is not None:
            del self._model
            self._model = None
        self._voice_clone_prompt = None
        self._executor.shutdown(wait=False)
