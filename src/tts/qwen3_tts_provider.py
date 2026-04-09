"""
Qwen3-TTS provider for low-latency local voice cloning.

The recommended deployment for this project is the ``worker`` backend:
run Qwen3-TTS inside a dedicated Python environment so its Transformers 4.x
stack stays isolated from Gemma in the main application environment.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soundfile as sf

from src.utils.language_detection import LanguageCode, detect_language

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
        if language is None:
            return None

        value = language.strip().lower()
        if not value or value == "auto":
            return None

        if value.startswith("fr"):
            return "fr"
        if value.startswith("en"):
            return "en"
        if value.startswith("de"):
            return "de"
        if value.startswith("it"):
            return "it"
        if value.startswith("es"):
            return "es"
        if value.startswith("pt"):
            return "pt"
        if value.startswith("ru"):
            return "ru"
        if value.startswith("ja"):
            return "ja"
        if value.startswith("ko"):
            return "ko"
        if value.startswith("zh"):
            return "zh"

        return value

    def _resolve_request_language(self, text: str) -> str:
        explicit_code = self._normalize_language_code(self.language)
        if explicit_code:
            return self._normalize_language(explicit_code)

        hint_code = self._normalize_language_code(self._language_hint)
        if hint_code and hint_code not in {"fr", "en"}:
            return self._normalize_language(hint_code)

        if text and text.strip():
            default_lang = LanguageCode.FRENCH if hint_code == "fr" else LanguageCode.ENGLISH
            detected = detect_language(text, default=default_lang)
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
            process.kill()

        self._worker_process = None
        self._worker_stderr_thread = None

    def _restart_worker(self) -> None:
        if self.backend == "worker":
            self._shutdown_worker()

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
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=self._resolve_dtype(),
                attn_implementation=self._resolve_attn_implementation(),
            )

            if self.mode == "voice_clone" and self.ref_audio_path and self.ref_audio_path.exists():
                self._voice_clone_prompt = self._create_voice_clone_prompt()

    def _load_model(self):
        if self.backend == "worker":
            self._spawn_worker()
            return
        self._load_model_inprocess()

    def preload(self) -> None:
        self._load_model()

    def warmup(self) -> None:
        """Pay the first synthesis cost before the first real user reply."""
        if self._warmed_up:
            return

        warmup_text = "Bonjour."
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            self._synthesize_worker_sync(warmup_text, tmp_path)
            self._warmed_up = True
        finally:
            tmp_path.unlink(missing_ok=True)

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
        tmp_output = output_path
        created_temp = False
        if tmp_output is None:
            fd, tmp_name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_output = Path(tmp_name)
            created_temp = True

        try:
            response = self._send_worker_command(
                {
                    "command": "synthesize",
                    "text": text,
                    "language": self._resolve_request_language(text),
                    "output_path": str(tmp_output),
                }
            )
            generated_path = Path(response["output_path"]).resolve()
            duration = response.get("duration")
            if output_path is not None:
                return TTSResult(audio_path=generated_path, duration=duration)

            audio_data = generated_path.read_bytes()
            return TTSResult(audio_data=audio_data, duration=duration)
        finally:
            if created_temp and tmp_output:
                tmp_output.unlink(missing_ok=True)

    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None,
    ) -> TTSResult:
        loop = asyncio.get_running_loop()

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

        if output_path is not None:
            sf.write(str(output_path), audio_int16, sample_rate, subtype="PCM_16")
            return TTSResult(audio_path=output_path, duration=duration)

        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
        return TTSResult(audio_data=buffer.getvalue(), duration=duration)

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
            self._restart_worker()
            return

        self.speaker = voice_id
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
        self._restart_worker()

    def cleanup(self) -> None:
        self._shutdown_worker()
        if self._model is not None:
            del self._model
            self._model = None
        self._voice_clone_prompt = None
        self._executor.shutdown(wait=False)
